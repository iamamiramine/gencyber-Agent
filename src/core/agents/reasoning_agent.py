import json
import logging
import re
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.prompts.chat import (
    HumanMessage,
    AIMessage,
)
from core.agents.base_agent_state_spec import BaseStatefulAgent
from core.helpers.llm_context_limits import (
    max_generative_response_snippet_chars,
    max_reasoning_objective_chars,
    max_recon_context_chars,
    max_script_output_chars,
    truncate_middle,
)
from domain.models.structured_outputs.reasoning_agent_structured_output import ReasoningAgentResponse
from infrastructure.services.memory_logger_service import get_memory_logger

# Configure logging
logger = logging.getLogger(__name__)


# Matches "1. Title (to-do)" and "1.1 Subtitle (failed)" — note the dot after the task id.
TASK_LINE_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.?\s+.*\((to-do|completed|not applicable|failed|blocked)\)\s*$",
    re.IGNORECASE,
)

_STATUS_PAREN = re.compile(
    r"\s*\((?:to-do|completed|not\s+applicable|failed|blocked)\)\s*$",
    re.IGNORECASE,
)

class ReasoningAgentState(TypedDict, total=False):
    """Keys the reasoning agent reads or writes (optional keys, ``TypedDict``)."""

    query: Annotated[str | None, "graph"]
    reasoning_task_tree: Annotated[str | None, "reasoning_agent"]
    script_output: Annotated[str | None, "graph"]
    generative_agent_response: Annotated[str | None, "graph"]
    reasoning_initialized: Annotated[bool | None, "reasoning_agent"]
    reasoning_last_input: Annotated[str | None, "reasoning_agent"]
    reasoning_candidate_tasks: Annotated[List[str] | None, "reasoning_agent"]
    reasoning_recommended_task: Annotated[str | None, "reasoning_agent"]
    reasoning_cycle_count: Annotated[int | None, "reasoning_agent"]
    reasoning_agent_response: Annotated[str | None, "reasoning_agent"]
    session_id: Annotated[str | None, "system"]
    objectives: Annotated[List[Any] | None, "pm_agent"]
    constraints: Annotated[List[Any] | None, "pm_agent"]
    goal_format: Annotated[str | None, "pm_agent"]
    planning_context: Annotated[str | None, "pm_agent"]
    context: Annotated[str | None, "recon_agent"]


class ReasoningAgent(BaseStatefulAgent):
    """Maintains a hierarchical task tree and recommends the next leaf task for execution."""

    agent_name = "reasoning"
    description = "Maintains the task tree and sets the recommended next leaf task for execution."
    state_schema = ReasoningAgentState

    def __init__(
        self,
        llm,
        generation_config,
        chat_history,
        system_prompt,
        formatter,
        model_params=None,
        pipeline_params=None,
        model_config_raw=None,
        node_id: Optional[str] = None,
    ):
        self.llm = llm
        self.generation_config = generation_config or {}
        self.chat_history = chat_history
        self.system_prompt = system_prompt
        self.formatter = formatter
        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.model_config_raw = model_config_raw or {}
        self.memory_logger = get_memory_logger()
        self._node_id = node_id or "reasoning"

    def format_chat_history(self, chat_history):
        return self.formatter.format_chat_history(chat_history, self.generation_config["model_name"])

    def _extract_tree_lines(self, task_tree: str) -> dict[str, str]:
        lines: dict[str, str] = {}
        for raw_line in (task_tree or "").splitlines():
            line = raw_line.strip()
            match = TASK_LINE_PATTERN.match(line)
            if not match:
                continue
            task_id = match.group(1)
            lines[task_id] = line
        return lines

    def _leaf_ids(self, lines: dict[str, str]) -> set[str]:
        ids = list(lines.keys())
        leafs = set(ids)
        for task_id in ids:
            prefix = f"{task_id}."
            if any(other.startswith(prefix) for other in ids if other != task_id):
                leafs.discard(task_id)
        return leafs

    @staticmethod
    def _strip_task_status(line: str) -> str:
        """Task line without trailing (to-do)|(completed)|… — compare titles for non-leaf stability."""
        return _STATUS_PAREN.sub("", (line or "").strip()).strip()

    def _is_leaf_only_update(self, previous_tree: str, new_tree: str) -> bool:
        """
        Accept updates that add/remove leaf task ids or change leaf lines/status,
        as long as every line that was a **non-leaf** in the previous tree still
        exists with the **same title text** (status suffix on parents may change,
        e.g. (to-do) → (completed) when a phase closes).

        Dropping a non-leaf id while keeping children is still rejected here; those
        updates are applied anyway after a failed retry (see ``__call__``).
        """
        if not previous_tree:
            return True

        previous_lines = self._extract_tree_lines(previous_tree)
        new_lines = self._extract_tree_lines(new_tree)

        non_leaf_ids = set(previous_lines.keys()) - self._leaf_ids(previous_lines)
        for task_id in non_leaf_ids:
            if task_id not in new_lines:
                return False
            if self._strip_task_status(previous_lines[task_id]) != self._strip_task_status(
                new_lines[task_id]
            ):
                return False

        return True

    @staticmethod
    def _format_planning_block(view: SimpleNamespace) -> str:
        chunks: list[str] = []
        objs = getattr(view, "objectives", None)
        if objs:
            chunks.append("Objectives:\n" + "\n".join(f"- {o}" for o in objs))
        cons = getattr(view, "constraints", None)
        if cons:
            chunks.append("Constraints:\n" + "\n".join(f"- {c}" for c in cons))
        gf = getattr(view, "goal_format", None)
        if gf and str(gf).strip():
            chunks.append(f"Goal / answer format:\n{gf}")
        pc = getattr(view, "planning_context", None)
        if pc and str(pc).strip():
            chunks.append(f"Planning brief:\n{pc}")
        ctx = getattr(view, "context", None)
        if ctx and str(ctx).strip():
            ctx_t = truncate_middle(
                str(ctx).strip(),
                max_recon_context_chars(),
                label="recon context",
            )
            chunks.append(f"Environmental / recon context:\n{ctx_t}")
        return "\n\n".join(chunks).strip()

    def _derive_todo_leaf_tasks(self, task_tree: str) -> list[str]:
        lines = self._extract_tree_lines(task_tree)
        if not lines:
            return []
        leaf_ids = self._leaf_ids(lines)
        todos: list[str] = []
        failed_blocked: list[str] = []
        for task_id in sorted(leaf_ids, key=lambda x: [int(p) for p in x.split(".")]):
            line = lines[task_id]
            low = line.lower()
            if "(failed)" in low or "(blocked)" in low:
                failed_blocked.append(line)
            elif "(to-do)" in low:
                todos.append(line)
        return failed_blocked + todos

    @staticmethod
    def _resolve_recommended_task(
        recommended: Optional[str], candidate_tasks: List[str]
    ) -> Optional[str]:
        if not candidate_tasks:
            return None
        if not recommended:
            return None
        if recommended in candidate_tasks:
            return recommended
        r = recommended.strip()
        for c in candidate_tasks:
            if r == c or r in c or c in r:
                return c
            stripped = re.sub(r"\s*\([^)]*\)\s*$", "", c.strip())
            desc = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", stripped).strip()
            if desc and (
                r.lower() == desc.lower()
                or r.lower() in desc.lower()
                or desc.lower() in r.lower()
            ):
                return c
        return None

    def _append_recon_context(self, view: SimpleNamespace, evidence: str) -> str:
        """Re-run recon after each command refreshes ``context``; always surface it on updates."""
        ctx = getattr(view, "context", None)
        if not ctx or not str(ctx).strip():
            return evidence
        ctx_t = truncate_middle(
            str(ctx).strip(),
            max_recon_context_chars(),
            label="recon context",
        )
        return f"{evidence}\n\nEnvironmental / recon context (latest):\n{ctx_t}"

    def _build_input(self, view: SimpleNamespace, initialized: bool) -> tuple[str, str]:
        query = view.query or ""
        task_tree = view.reasoning_task_tree or ""
        script_output = view.script_output
        generative_response = view.generative_agent_response

        if not initialized or not task_tree:
            mode = "initialize"
            extra = self._format_planning_block(view)
            q = truncate_middle(
                query.strip(),
                max_reasoning_objective_chars(),
                label="user objective",
            )
            evidence = f"User objective:\n{q}\n\n"
            if extra:
                evidence += extra + "\n\n"
            return mode, evidence

        mode = "update"
        if script_output:
            so = truncate_middle(
                str(script_output),
                max_script_output_chars(),
                label="execution output",
            )
            evidence = f"Latest execution output:\n{so}"
        elif generative_response:
            gr = truncate_middle(
                str(generative_response),
                max_generative_response_snippet_chars(),
                label="generative output",
            )
            evidence = f"Latest generative output:\n{gr}"
        else:
            evidence = f"Latest user input:\n{truncate_middle(query.strip(), max_reasoning_objective_chars(), label='user input')}"
        evidence = self._append_recon_context(view, evidence)
        return mode, evidence

    def _invoke_reasoner(self, evidence: str, current_tree: str, mode: str) -> ReasoningAgentResponse:
        payload = (
            f"Mode: {mode}\n\n"
            f"Current task tree:\n{current_tree or 'None'}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Rules:\n"
            "1) Keep numbered tree format with statuses (to-do/completed/not applicable).\n"
            "2) During update mode, keep each existing **non-leaf task id**; you may change its "
            "**status** suffix (e.g. parent (to-do)→(completed)). Prefer stable **titles** for non-leaves; "
            "if **Environmental / recon context** requires reprioritization, add **new** branches (new ids) "
            "or mark obsolete work (not applicable) rather than deleting parent ids that still have children.\n"
            "3) Infer success vs failure only from the execution evidence itself (output text and any "
            "exit/success metadata the tool included). For failed runs, use (failed) or (blocked) on the "
            "leaf—never (completed). Add remediation subtasks when appropriate.\n"
            "4) Align **recommended_task** with recon priorities when they conflict with an older branch.\n"
            "5) Return candidate leaf tasks and one recommended next task.\n"
        )
        self.chat_history.add_user_message(HumanMessage(content=payload))
        history_with_memory = self.format_chat_history(self.chat_history)
        response = self.llm.invoke({"system": str(self.system_prompt), "history": history_with_memory})
        response_json = response.model_dump_json() if hasattr(response, "model_dump_json") else str(response)
        self.chat_history.add_ai_message(AIMessage(content=response_json))
        if isinstance(response, dict):
            return ReasoningAgentResponse(**response)
        return response

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        view = self.read_state(state)
        initialized = bool(view.reasoning_initialized)
        current_tree = view.reasoning_task_tree or ""
        mode, evidence = self._build_input(view, initialized)
        last_input = view.reasoning_last_input

        if initialized and evidence == last_input and current_tree:
            candidate_tasks = self._derive_todo_leaf_tasks(current_tree)
            recommended = candidate_tasks[0] if candidate_tasks else None
            out = {
                "reasoning_candidate_tasks": candidate_tasks,
                "reasoning_recommended_task": recommended,
                "reasoning_agent_response": json.dumps(
                    {
                        "task_tree": current_tree,
                        "candidate_tasks": candidate_tasks,
                        "recommended_task": recommended,
                        "reasoning": ["Skipped update because evidence is unchanged."],
                        "update_applied": False,
                    }
                ),
            }
            print(f"REASONING_AGENT_RESPONSE skipped=True tree_present=True", flush=True)
            return out

        try:
            structured_response = self._invoke_reasoner(evidence=evidence, current_tree=current_tree, mode=mode)
            next_tree = structured_response.task_tree

            print("REASONING_AGENT_RESPONSE", structured_response, flush=True)

            if mode == "update" and current_tree and not self._is_leaf_only_update(current_tree, next_tree):
                correction_evidence = (
                    f"{evidence}\n\n"
                    "Validation error: you removed a non-leaf id, changed a non-leaf **title** (not just "
                    "status), or broke parent/child ids. Regenerate: keep every previous non-leaf **id** "
                    "with the same title text (status may change); add new ids for new work; edit leaves "
                    "freely. If recon demands a new line of attack, add sibling/subtasks with new numbers."
                )
                retry_response = self._invoke_reasoner(
                    evidence=correction_evidence,
                    current_tree=current_tree,
                    mode=mode,
                )
                if self._is_leaf_only_update(current_tree, retry_response.task_tree):
                    structured_response = retry_response
                    next_tree = retry_response.task_tree
                else:
                    logger.warning(
                        "ReasoningAgent: strict tree validation failed after retry; applying model tree "
                        "so recon/evidence can reshape the plan (candidate_tasks from new tree)"
                    )
                    structured_response = retry_response
                    next_tree = retry_response.task_tree

            candidate_tasks = self._derive_todo_leaf_tasks(next_tree)
            recommended_task = self._resolve_recommended_task(
                structured_response.recommended_task, candidate_tasks
            )
            if not recommended_task and candidate_tasks:
                llm_cands = structured_response.candidate_tasks or []
                for c in llm_cands:
                    mc = self._resolve_recommended_task(c, candidate_tasks)
                    if mc:
                        recommended_task = mc
                        break
                if not recommended_task:
                    recommended_task = candidate_tasks[0]

            response_snapshot = structured_response.model_copy(
                update={
                    "task_tree": next_tree,
                    "candidate_tasks": candidate_tasks,
                    "recommended_task": recommended_task,
                }
            )

            cycle_next = int(view.reasoning_cycle_count or 0) + 1
            updates: Dict[str, Any] = {
                "reasoning_task_tree": next_tree,
                "reasoning_candidate_tasks": candidate_tasks,
                "reasoning_recommended_task": recommended_task,
                "reasoning_initialized": True,
                "reasoning_last_input": evidence,
                "reasoning_cycle_count": cycle_next,
                "reasoning_agent_response": response_snapshot.model_dump_json(),
            }

            merged = self.snapshot_after(state, updates)
            try:
                self.memory_logger.log_comprehensive_interaction(
                    session_id=view.session_id or "unknown",
                    agent_type="reasoning_agent",
                    original_query=view.query or "",
                    query_to_process=evidence,
                    state=merged,
                    structured_response=response_snapshot.model_dump(),
                )
            except Exception as log_error:
                logger.warning(f"Failed to log comprehensive memory for reasoning agent: {log_error}")

            return updates
        except Exception as exc:
            logger.error("ReasoningAgent failed: %s", exc)
            fallback_tree = current_tree or "1 Reconnaissance - (to-do)"
            candidate_tasks = self._derive_todo_leaf_tasks(fallback_tree)
            cycle_next = int(view.reasoning_cycle_count or 0) + 1
            return {
                "reasoning_task_tree": fallback_tree,
                "reasoning_candidate_tasks": candidate_tasks,
                "reasoning_recommended_task": candidate_tasks[0] if candidate_tasks else None,
                "reasoning_initialized": bool(current_tree),
                "reasoning_last_input": evidence,
                "reasoning_cycle_count": cycle_next,
                "reasoning_agent_response": json.dumps(
                    {"error": str(exc), "task_tree": fallback_tree, "candidate_tasks": candidate_tasks}
                ),
            }