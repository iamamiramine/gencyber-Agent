from __future__ import annotations

import logging
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Annotated, Any, Callable, Dict, TypedDict

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from core.agents.generative_agent import GenerativeAgentState
from core.agents.pm_agent import PMAgentState
from core.agents.reasoning_agent import ReasoningAgentState
from core.agents.recon_agent import ReconAgentState
from infrastructure.repository.mongodb_repository import get_mongodb_client

logger = logging.getLogger(__name__)

_DEFAULT_REASONING_CYCLE_MAX = 50
_DEFAULT_RECURSION_LIMIT = 100


def _routing_state_view(state: Any) -> Dict[str, Any]:
    if isinstance(state, BaseModel):
        return state.model_dump()
    if isinstance(state, SimpleNamespace):
        return vars(state)
    return dict(state)


class WorkflowGraphMetadata(TypedDict, total=False):
    """Run-level keys (optional, ``TypedDict``)."""

    timestamp: Annotated[datetime | None, "system"]
    created_at: Annotated[datetime | None, "system"]
    error: Annotated[str | None, "system"]
    generated_response: Annotated[str | None, "system"]


class WorkflowGraphState(
    PMAgentState,
    ReconAgentState,
    ReasoningAgentState,
    GenerativeAgentState,
    WorkflowGraphMetadata,
    total=False,
):
    """Merged channel schema for ``StateGraph`` (same keys as agent slices + metadata)."""

class WorkflowGraph:
    """Workflow graph with explicit edges (no YAML topology parsing)."""

    def __init__(
        self,
        session_id: str,
        *,
        pm_agent: Callable[..., Any],
        recon_agent: Callable[..., Any],
        reasoning_agent: Callable[..., Any],
        generative_agent: Callable[..., Any],
        execute_script_tool: Callable[..., Any],
    ) -> None:
        self.session_id = session_id
        self.reasoning_cycle_max = _DEFAULT_REASONING_CYCLE_MAX
        self.recursion_limit = _DEFAULT_RECURSION_LIMIT

        self._pm = pm_agent
        self._recon = recon_agent
        self._reasoning = reasoning_agent
        self._generative = generative_agent
        self._execute_script = execute_script_tool

        self.graph = self.create_graph()

    def _route_after_reasoning(self, state: Dict[str, Any]) -> str:
        s = _routing_state_view(state)
        sg = s.get("submitted_goal")
        if sg is not None and str(sg).strip():
            return "end"
        if bool(s.get("should_stop")):
            return "end"
        cycle_count = int(s.get("reasoning_cycle_count") or 0)
        if cycle_count > self.reasoning_cycle_max:
            logger.warning(
                "Reasoning cycle limit reached (max=%s); routing to end",
                self.reasoning_cycle_max,
            )
            return "end"
        return "continue"

    def _route_after_generation(self, state: Dict[str, Any]) -> str:
        s = _routing_state_view(state)
        sg = s.get("submitted_goal")
        if sg is not None and str(sg).strip():
            return "end"
        if bool(s.get("should_stop")):
            return "end"
        cmd = s.get("command")
        if cmd is not None and str(cmd).strip():
            return "script"
        return "follow_up"

    def _route_after_script(self, state: Dict[str, Any]) -> str:
        s = _routing_state_view(state)
        sg = s.get("submitted_goal")
        if sg is not None and str(sg).strip():
            return "end"
        if bool(s.get("should_stop")):
            return "end"
        return "continue"

    def create_graph(self) -> StateGraph:
        workflow = StateGraph(WorkflowGraphState)

        workflow.add_node("pm", self._pm)
        workflow.add_node("recon", self._recon)
        workflow.add_node("reasoning", self._reasoning)
        workflow.add_node("generative", self._generative)
        workflow.add_node("execute_script_tool", self._execute_script)

        workflow.add_edge(START, "pm")
        workflow.add_edge("pm", "recon")
        workflow.add_edge("recon", "reasoning")

        workflow.add_conditional_edges(
            "reasoning",
            self._route_after_reasoning,
            {"continue": "generative", "end": END},
        )
        workflow.add_conditional_edges(
            "generative",
            self._route_after_generation,
            {
                "script": "execute_script_tool",
                "follow_up": "reasoning",
                "end": END,
            },
        )
        # Re-run recon after every command so environmental context reflects the latest output
        # before reasoning / generative plan the next step.
        workflow.add_conditional_edges(
            "execute_script_tool",
            self._route_after_script,
            {"continue": "recon", "end": END},
        )
        mongo_client = get_mongodb_client()
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")
        checkpointer = MongoDBSaver(mongo_client, db_name=db_name)

        return workflow.compile(checkpointer=checkpointer)

    def _build_initial_state(self, query: str) -> Dict[str, Any]:
        """
        Per-invoke input merged with the checkpointer for ``thread_id``.
        """
        now = datetime.now()
        state_dict = {
            "query": query,
            "session_id": self.session_id,
            "timestamp": now,
            "created_at": now,
            "should_stop": False,
            "submitted_goal": None,
            "command": None,
            "script_output": None,
            "generative_agent_response": None,
            "query_to_process": None,
            # PM + recon (cleared so nothing from the previous checkpoint leaks in)
            "objectives": [],
            "constraints": [],
            "goal_format": None,
            "planning_context": None,
            "context": None,
            # Reasoning (must clear so a new task tree is built for this query)
            "reasoning_task_tree": None,
            "reasoning_candidate_tasks": [],
            "reasoning_recommended_task": None,
            "reasoning_initialized": False,
            "reasoning_last_input": None,
            "reasoning_cycle_count": 0,
            "reasoning_agent_response": None,
        }
        return state_dict

    def invoke(self, query: str) -> Dict[str, Any]:
        logger.info("Invoking agent graph query=%s", query)
        state_dict = self._build_initial_state(query=query)
        try:
            config = {
                "configurable": {"thread_id": self.session_id},
                "recursion_limit": self.recursion_limit,
            }
            final_state = self.graph.invoke(state_dict, config=config)
            print("FINAL_STATE", final_state, flush=True)
            logger.info("Workflow execution completed")
            return final_state
        except Exception as e:
            logger.error("Error invoking workflow: %s", e)
            return {
                "error": str(e),
                "generated_response": f"I encountered an error processing your request: {str(e)}",
            }
