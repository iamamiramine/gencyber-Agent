from __future__ import annotations

import logging
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Annotated, Any, Callable, Dict, Optional, TypedDict

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from core.agents.generative_agent import GenerativeAgentState
from infrastructure.observability import langfuse_tracer
from infrastructure.repository.mongodb_repository import get_mongodb_client

logger = logging.getLogger(__name__)

_DEFAULT_GENERATION_CYCLE_MAX = 60
_DEFAULT_RECURSION_LIMIT = int(os.getenv("GENCYBER_RECURSION_LIMIT", "300"))


def _routing_state_view(state: Any) -> Dict[str, Any]:
    if isinstance(state, BaseModel):
        return state.model_dump()
    if isinstance(state, SimpleNamespace):
        return vars(state)
    return dict(state)


def serialize_graph_topology(compiled: Any) -> Dict[str, Any]:
    """Project a compiled LangGraph into a JSON-able ``{nodes, edges, ...}`` shape.

    Reads ``compiled.get_graph()`` directly so callers never hand-maintain a
    duplicate of the wiring. Conditional-branch keys (``edge.data``) become edge
    labels; the synthetic ``__start__`` / ``__end__`` ids are preserved as-is so
    the frontend can normalize them.
    """
    drawable = compiled.get_graph()

    nodes = [{"id": str(node_id)} for node_id in drawable.nodes]

    edges: list[Dict[str, Any]] = []
    entry: Optional[str] = None
    for edge in drawable.edges:
        source = str(edge.source)
        target = str(edge.target)
        data = getattr(edge, "data", None)
        label = str(data) if isinstance(data, str) and data.strip() else None
        conditional = bool(getattr(edge, "conditional", False))
        edges.append(
            {
                "source": source,
                "target": target,
                "label": label,
                "conditional": conditional,
            }
        )
        if source == "__start__" and entry is None:
            entry = target

    return {
        "nodes": nodes,
        "edges": edges,
        "entry": entry,
        "start": "__start__",
        "end": "__end__",
    }


class WorkflowGraphMetadata(TypedDict, total=False):
    """Run-level keys (optional, ``TypedDict``)."""

    timestamp: Annotated[datetime | None, "system"]
    created_at: Annotated[datetime | None, "system"]
    error: Annotated[str | None, "system"]
    generated_response: Annotated[str | None, "system"]
    submission_verified: Annotated[bool | None, "system"]
    submission_rejection_reason: Annotated[str | None, "system"]


class WorkflowGraphState(
    GenerativeAgentState,
    WorkflowGraphMetadata,
    total=False,
):
    """Merged channel schema for the single-agent ``StateGraph``."""

class WorkflowGraph:
    """Single generative agent + execute_script / write_script / submit_goal tools."""

    # Agents (by registry name) this builder needs from the pipeline registry. The
    # service validates these are present and passes their runtimes to ``build``.
    REQUIRED_AGENTS: frozenset = frozenset({"generative"})

    def __init__(
        self,
        session_id: str,
        *,
        generative_agent: Callable[..., Any],
        execute_script_tool: Callable[..., Any],
        write_script_tool: Callable[..., Any],
        submit_goal_tool: Callable[..., Any],
        checkpointing: bool = True,
    ) -> None:
        self.session_id = session_id
        self.generation_cycle_max = _DEFAULT_GENERATION_CYCLE_MAX
        self.recursion_limit = _DEFAULT_RECURSION_LIMIT
        # When False the graph compiles without a MongoDB checkpointer — used for
        # cheap, side-effect-free topology serialization (see ``topology``).
        self._checkpointing = checkpointing

        self._generative = generative_agent
        self._execute_script = execute_script_tool
        self._write_script = write_script_tool
        self._submit_goal = submit_goal_tool

        self.graph = self.create_graph()

    def _decide_after_generation(self, s: Dict[str, Any]) -> str:
        if bool(s.get("submission_verified")):
            return "end"
        sg = s.get("submitted_goal")
        if sg is not None and str(sg).strip():
            return "submit_goal"
        ws = s.get("write_script")
        if ws is not None and str(ws).strip():
            return "write_script"
        cmd = s.get("command")
        if cmd is not None and str(cmd).strip():
            return "script"
        # No actionable output — but the run does NOT end here. Loop back so the agent
        # tries again; only a verified submission ends the run.
        return "continue"

    def _route_after_generation(self, state: Dict[str, Any]) -> str:
        s = _routing_state_view(state)
        decision = self._decide_after_generation(s)
        # Conditional-edge functions are not runnables, so the callback bus never sees
        # them — emit the routing decision as a trace event explicitly.
        langfuse_tracer.record_event(
            "route-after-generation",
            decision=decision,
            has_submitted_goal=bool(str(s.get("submitted_goal") or "").strip()),
            has_command=bool(str(s.get("command") or "").strip()),
            has_write_script=bool(str(s.get("write_script") or "").strip()),
            submission_verified=bool(s.get("submission_verified")),
        )
        return decision

    def _route_after_submit(self, state: Dict[str, Any]) -> str:
        s = _routing_state_view(state)
        decision = "end" if bool(s.get("submission_verified")) else "continue"
        langfuse_tracer.record_event(
            "route-after-submit",
            decision=decision,
            submission_verified=bool(s.get("submission_verified")),
        )
        return decision

    def _wire_state_graph(self) -> StateGraph:
        """Wire nodes/edges of the (uncompiled) ``StateGraph``.

        Topology lives here so both the live graph and ``topology`` serialization
        share one source of truth — there is no hand-maintained duplicate.
        """
        workflow = StateGraph(WorkflowGraphState)

        workflow.add_node("generative", self._generative)
        workflow.add_node("execute_script_tool", self._execute_script)
        workflow.add_node("write_script_tool", self._write_script)
        workflow.add_node("submit_goal_tool", self._submit_goal)

        workflow.add_edge(START, "generative")
        workflow.add_conditional_edges(
            "generative",
            self._route_after_generation,
            {
                "write_script": "write_script_tool",
                "script": "execute_script_tool",
                "submit_goal": "submit_goal_tool",
                "continue": "generative",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "submit_goal_tool",
            self._route_after_submit,
            {"continue": "generative", "end": END},
        )
        workflow.add_edge("execute_script_tool", "generative")
        workflow.add_edge("write_script_tool", "generative")
        return workflow

    def create_graph(self) -> Any:
        workflow = self._wire_state_graph()

        # Topology-only callers compile without a checkpointer so no MongoDB
        # connection (and no run-time side effects) are needed.
        if not self._checkpointing:
            return workflow.compile()

        mongo_client = get_mongodb_client()
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")
        checkpointer = MongoDBSaver(mongo_client, db_name=db_name)

        return workflow.compile(checkpointer=checkpointer)

    @classmethod
    def topology(cls) -> Dict[str, Any]:
        """Serialize the real compiled graph shape (no MongoDB, no side effects).

        Builds an instance with no-op node callables and ``checkpointing=False``
        so we can read the actual ``compiled.get_graph()`` — the viz therefore
        reflects whatever ``_wire_state_graph`` wires, with no duplicate map.
        """

        def _noop(state: Any) -> Any:
            return state

        inst = cls(
            session_id="__topology__",
            generative_agent=_noop,
            execute_script_tool=_noop,
            write_script_tool=_noop,
            submit_goal_tool=_noop,
            checkpointing=False,
        )
        return serialize_graph_topology(inst.graph)

    @classmethod
    def build(
        cls,
        *,
        session_id: str,
        agents: Dict[str, Any],
        execute_script_tool: Callable[..., Any],
        write_script_tool: Callable[..., Any],
        submit_goal_tool: Callable[..., Any],
    ) -> "WorkflowGraph":
        """Construct the live graph from resolved agent runtimes + tool callables.

        Lets the service stay graph-agnostic: it validates ``REQUIRED_AGENTS`` and
        hands every builder the same ``agents`` map and tools; each builder picks the
        agents it needs.
        """
        return cls(
            session_id=session_id,
            generative_agent=agents["generative"],
            execute_script_tool=execute_script_tool,
            write_script_tool=write_script_tool,
            submit_goal_tool=submit_goal_tool,
        )

    def _build_initial_state(self, query: str) -> Dict[str, Any]:
        """Per-invoke input merged with the checkpointer for ``thread_id``."""
        now = datetime.now()
        return {
            "query": query,
            "session_id": self.session_id,
            "timestamp": now,
            "created_at": now,
            "submitted_goal": None,
            "command": None,
            "write_script": None,
            "write_script_language": None,
            "script_output": None,
            "generative_agent_response": None,
            "query_to_process": None,
            "submission_verified": None,
            "submission_rejection_reason": None,
        }

    def invoke(
        self, query: str
    ) -> Dict[str, Any]:
        logger.info("Invoking agent graph query=%s", query)
        state_dict = self._build_initial_state(query=query)
        try:
            with langfuse_tracer.traced_run(
                self.session_id,
                name=type(self).__name__,
                tags=[type(self).__name__, "invoke"],
            ) as handler:
                config: Dict[str, Any] = {
                    "configurable": {"thread_id": self.session_id},
                    "recursion_limit": self.recursion_limit,
                }
                if handler is not None:
                    config["callbacks"] = [handler]
                final_state = self.graph.invoke(state_dict, config=config)
                langfuse_tracer.record_output(final_state)
            print("FINAL_STATE", final_state, flush=True)
            logger.info("Workflow execution completed")
            return final_state
        except Exception as e:
            logger.error("Error invoking workflow: %s", e)
            return {
                "error": str(e),
                "generated_response": f"I encountered an error processing your request: {str(e)}",
            }

GRAPH_REGISTRY: Dict[str, Any] = {
    "WorkflowGraph": WorkflowGraph,
}

# Registered after WorkflowGraph + serialize_graph_topology are defined so the deep
# workflow modules can import serialize_graph_topology without an import cycle.
from application.langgraph.models.deep_generative_workflow import (  # noqa: E402
    DeepGenerativeWorkflow,
)
from application.langgraph.models.deep_planner_workflow import (  # noqa: E402
    DeepPlannerWorkflow,
)

GRAPH_REGISTRY["DeepGenerativeWorkflow"] = DeepGenerativeWorkflow
GRAPH_REGISTRY["DeepPlannerWorkflow"] = DeepPlannerWorkflow