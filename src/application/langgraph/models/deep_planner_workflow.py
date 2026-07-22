"""DeepAgents-backed hierarchical planner workflow.

A second DeepAgents graph variant that puts a **planner** at the top. The planner
is a native tool-calling deep agent that decomposes the briefing into a plan
(``write_todos``), works the plan step by step, and owns the final submission. It
reuses the EXACT same machinery as :class:`DeepGenerativeWorkflow` — the three
workbench-backed tools, the shared :class:`DeepGenerativeState`, and
:class:`FlagGateMiddleware` (so routing parity holds: the run only ends on an
accepted submission).

Phase 3 makes the planner a pure orchestrator: it owns only the final
``submit_goal`` (plus the auto-added ``write_todos`` and ``task`` tools) and
delegates all hands-on execution to a **generative** subagent via the DeepAgents
``task`` tool. The subagent shares the same workbench terminal and the same
:class:`DeepGenerativeState` channels, so evidence it gathers
(``execution_evidence``, ``script_output``) and a flag it recovers/submits
(``submitted_goal`` / ``submission_verified``) flow back to the planner — which
lets the top-level :class:`FlagGateMiddleware` end the run on acceptance.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.mongodb import MongoDBSaver

from infrastructure.repository.mongodb_repository import get_mongodb_client

from application.langgraph.helpers.skills_helper import (
    compose_subagent_prompt,
    discover_ctf_skills,
    make_read_skill_note_tool,
    make_search_skill_tool,
    section_retrieval_enabled,
)
from application.langgraph.models.deep_generative_workflow import (
    ConsolidationMiddleware,
    DeepGenerativeWorkflow,
    EscalationMiddleware,
    HistoryBoundMiddleware,
    StepBudgetMiddleware,
    SubagentContextMiddleware,
    build_escalation_model,
    _make_recall_tool,
    _make_tools,
)

logger = logging.getLogger(__name__)


class DeepPlannerWorkflow(DeepGenerativeWorkflow):
    """Planner deep agent. Same runtime adapter surface as the generative workflow."""

    REQUIRED_AGENTS: frozenset = frozenset({"planner", "generative"})

    @classmethod
    def _build_subagents(
        cls,
        *,
        agents: Dict[str, Any],
        execute_script_tool: Callable[..., Any],
        write_script_tool: Callable[..., Any],
        submit_goal_tool: Callable[..., Any],
        session_id: str,
    ) -> Optional[List[Any]]:
        """The specialist worker subagents the planner delegates to via ``task``.

        One subagent per attack category (the 9 ``ctf-*`` skill packs). Every
        specialist shares the SAME worker model + base prompt (from the
        ``generative`` agent definition) and the SAME execution toolset
        (execute_script + write_script + submit_goal) built from the workbench
        state-callables; each additionally gets a ``read_skill_note`` tool sandboxed
        to its own skill pack for lazy deep-dive lookups. Routing is data-driven:
        each subagent's ``description`` is the skill's frontmatter (when to use /
        not use it), which the ``task`` tool surfaces to the planner for selection.

        The parent ``state_schema`` (:class:`DeepGenerativeState`) is forwarded by
        ``create_deep_agent`` so the shared channels (``execution_evidence``,
        ``submitted_goal``, ``submission_verified``) flow to and from each specialist.
        """
        generative = agents.get("generative")
        if generative is None:
            return None
        sub_model = getattr(generative, "model", None)
        if sub_model is None:
            return None
        base_prompt = getattr(generative, "system_prompt", "") or ""
        # Strong model for the escalation ladder, built once and shared by every
        # specialist's EscalationMiddleware. Honors the per-run escalation target set
        # for this session (set_escalation_config); None when disabled/unbuildable.
        strong_model = build_escalation_model(generative, session_id=session_id)

        # Skills are consulted the way Claude Code uses them: progressive disclosure
        # driven by the model, NOT a coercive gate. The specialist prompt frames the
        # SKILL.md index + on-demand notes (compose_subagent_prompt), and the model
        # reads the relevant note/section when it judges it useful — it is never
        # blocked from writing a solution or submitting. The old require_skill_note
        # gate blocked 25% of submissions before validation and produced a
        # submit->"read a note first"->submit bounce on weak models without improving
        # solutions (notes were read to unlock, not applied). Capability to actually
        # apply a note's technique comes from the escalation ladder (a stronger model),
        # not from forcing a read. ``GENCYBER_REQUIRE_SKILL_NOTE=1`` re-enables the gate.
        require_note = os.getenv("GENCYBER_REQUIRE_SKILL_NOTE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        execute_script, write_script, submit_goal = _make_tools(
            execute_script_tool,
            write_script_tool,
            submit_goal_tool,
            session_id,
            require_skill_note=require_note,
            model=sub_model,
        )

        skills = discover_ctf_skills()
        if not skills:
            logger.warning("no ctf-* skill packs discovered; planner has no specialists")
            return None

        # One recall tool for the run; specialists share it (it is session-scoped and
        # stateless) so each can pull prior evidence on demand instead of receiving
        # the full transcript (Piece 3).
        recall_evidence = _make_recall_tool(session_id)

        # Phase 4b-B: when section retrieval is on, give specialists a search_skill tool
        # so they can locate the right note/section instead of loading whole notes.
        # Off by default → toolset is byte-for-byte the legacy four tools + recall.
        use_search = section_retrieval_enabled()

        subagents: List[Dict[str, Any]] = []
        for skill in skills:
            read_skill_note = make_read_skill_note_tool(skill.skill_dir, skill.notes)
            tools = [
                execute_script,
                write_script,
                submit_goal,
                read_skill_note,
                recall_evidence,
            ]
            if use_search:
                tools.append(make_search_skill_tool(skill.skill_dir, skill.notes))
            subagents.append(
                {
                    "name": skill.name,
                    "description": skill.description,
                    "system_prompt": compose_subagent_prompt(base_prompt, skill),
                    "tools": tools,
                    "model": sub_model,
                    # Inject prior findings (fold cards, or the legacy transcript when
                    # memory is disabled) so a re-delegated specialist builds on earlier
                    # work; fold this run back into a strategy card on completion.
                    # StepBudgetMiddleware ends this specialist the moment the shared
                    # per-challenge action budget is spent (a specialist otherwise has no
                    # enforced recursion limit and would spin after the budget trips).
                    "middleware": [
                        HistoryBoundMiddleware(),
                        StepBudgetMiddleware(session_id),
                        SubagentContextMiddleware(model=sub_model),
                        # Escalate this specialist to the strong model once it stalls or
                        # submits a bad flag (no-op when strong_model is None).
                        EscalationMiddleware(session_id, strong_model),
                    ],
                }
            )
        logger.info(
            "planner specialists: %s", [s["name"] for s in subagents]
        )
        return subagents

    @classmethod
    def build(
        cls,
        *,
        session_id: str,
        agents: Dict[str, Any],
        execute_script_tool: Callable[..., Any],
        write_script_tool: Callable[..., Any],
        submit_goal_tool: Callable[..., Any],
    ) -> "DeepPlannerWorkflow":
        planner = agents["planner"]
        model = getattr(planner, "model", None)
        if model is None:
            raise ValueError(
                "DeepPlannerWorkflow requires the 'planner' agent to expose a "
                "tool-bindable .model (use agent_class_key: DeepGenerativeAgent)"
            )
        system_prompt = getattr(planner, "system_prompt", "") or ""

        # The planner orchestrates; it never runs shell commands itself. It keeps
        # only the final submission tool — execution lives on the generative
        # subagent it delegates to via the auto-added ``task`` tool.
        _, _, submit_goal = _make_tools(
            execute_script_tool, write_script_tool, submit_goal_tool, session_id,
            model=model,
        )
        tools = [submit_goal]
        subagents = cls._build_subagents(
            agents=agents,
            execute_script_tool=execute_script_tool,
            write_script_tool=write_script_tool,
            submit_goal_tool=submit_goal_tool,
            session_id=session_id,
        )

        mongo_client = get_mongodb_client()
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")
        checkpointer = MongoDBSaver(mongo_client, db_name=db_name)

        compiled = cls._compile(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            subagents=subagents,
            checkpointer=checkpointer,
            session_id=session_id,
            # Planner-only: consolidate recovered evidence into a submission instead of
            # delegating endless recon (advisory nudge; no hard cap, no model change).
            extra_middleware=[ConsolidationMiddleware()],
        )
        return cls(session_id, graph=compiled)

    @classmethod
    def topology(cls) -> Dict[str, Any]:
        """Serialize the compiled planner shape (no MongoDB, no side effects)."""
        from application.langgraph.models.langraph_model import serialize_graph_topology
        from langchain_openai import ChatOpenAI

        dummy_model = ChatOpenAI(
            model="openai/gpt-4o-mini",
            api_key="dummy",
            base_url="https://openrouter.ai/api/v1",
        )
        noop = RunnableLambda(lambda state: {})
        _, _, submit_goal = _make_tools(noop, noop, noop, "__topology__")
        tools = [submit_goal]
        stub_generative = SimpleNamespace(model=dummy_model, system_prompt="sub")
        subagents = cls._build_subagents(
            agents={"generative": stub_generative},
            execute_script_tool=noop,
            write_script_tool=noop,
            submit_goal_tool=noop,
            session_id="__topology__",
        )
        compiled = cls._compile(
            model=dummy_model,
            system_prompt="topology",
            tools=tools,
            subagents=subagents,
            checkpointer=False,
        )
        return serialize_graph_topology(compiled)
