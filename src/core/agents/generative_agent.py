# generative_agent.py
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.prompts.chat import HumanMessage, AIMessage
from core.agents.base_agent_state_spec import BaseStatefulAgent
from core.helpers.llm_context_limits import (
    max_recon_context_chars,
    max_reasoning_objective_chars,
    max_script_output_chars,
    truncate_middle,
)
from domain.models.structured_outputs.generative_agent_structured_output import GenerativeAgentResponse
from infrastructure.services.memory_logger_service import get_memory_logger

logger = logging.getLogger(__name__)


class GenerativeAgentState(TypedDict, total=False):
    """Keys the generative agent reads or updates on graph state (optional keys, ``TypedDict``)."""

    query: Annotated[str | None, "graph"]
    script_output: Annotated[str | None, "graph"]
    reasoning_recommended_task: Annotated[str | None, "reasoning_agent"]
    query_to_process: Annotated[str | None, "generative_agent"]
    generative_agent_response: Annotated[str | None, "graph"]
    command: Annotated[str | None, "generative_agent"]
    write_script: Annotated[str | None, "generative_agent"]
    write_script_language: Annotated[str | None, "generative_agent"]
    submitted_goal: Annotated[str | None, "generative_agent"]
    should_stop: Annotated[bool | None, "generative_agent"]
    session_id: Annotated[str | None, "system"]
    objectives: Annotated[List[Any] | None, "pm_agent"]
    constraints: Annotated[List[Any] | None, "pm_agent"]
    goal_format: Annotated[str | None, "pm_agent"]
    planning_context: Annotated[str | None, "pm_agent"]
    context: Annotated[str | None, "recon_agent"]


class GenerativeAgent(BaseStatefulAgent):
    """Agent responsible for generating responses using the LLM."""

    agent_name = "generative"
    description = "Produces the next shell command or submitted goal from LLM structured output."
    state_schema = GenerativeAgentState

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
        self._node_id = node_id or "generative"

    def format_chat_history(self, chat_history):
        return self.formatter.format_chat_history(chat_history, self.generation_config["model_name"])

    @staticmethod
    def should_stop_execution(response: GenerativeAgentResponse) -> bool:
        """Termination uses ``submitted_goal`` only (single source of truth)."""
        sg = response.submitted_goal
        return sg is not None and str(sg).strip() != ""

    def clear_chat_history_except_persistent(self) -> None:
        """Preserve cross-turn cues (final submissions) like the legacy password snapshot."""
        preserved: list[AIMessage] = []
        for message in self.chat_history.messages:
            if not isinstance(message, AIMessage):
                continue
            content = message.content
            text = content if isinstance(content, str) else str(content)
            if text.startswith("Submitted goal:"):
                preserved.append(message)
                continue
            stripped = text.strip()
            if stripped.startswith("{") and '"submitted_goal"' in stripped:
                try:
                    data = json.loads(stripped)
                    if isinstance(data, dict) and data.get("submitted_goal"):
                        preserved.append(message)
                except json.JSONDecodeError:
                    pass

        self.chat_history.clear()
        for msg in preserved:
            self.chat_history.add_ai_message(msg)

    def _session_context_overlay(self, view: SimpleNamespace) -> str:
        parts: list[str] = []
        objs = getattr(view, "objectives", None)
        if objs:
            parts.append("Objectives:\n" + "\n".join(f"- {o}" for o in objs))
        cons = getattr(view, "constraints", None)
        if cons:
            parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in cons))
        gf = getattr(view, "goal_format", None)
        if gf and str(gf).strip():
            parts.append(f"Goal / answer format:\n{gf}")
        pc = getattr(view, "planning_context", None)
        if pc and str(pc).strip():
            pc_t = truncate_middle(
                str(pc).strip(),
                max_recon_context_chars(),
                label="planning brief",
            )
            parts.append(f"Planning brief:\n{pc_t}")
        ctx = getattr(view, "context", None)
        if ctx and str(ctx).strip():
            ctx_t = truncate_middle(
                str(ctx).strip(),
                max_recon_context_chars(),
                label="recon context",
            )
            parts.append(f"Recon / environmental context:\n{ctx_t}")
        return "\n\n".join(parts).strip()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        view = self.read_state(state)
        query = view.query or ""
        if not query.strip():
            return {}

        updates: Dict[str, Any] = {}

        try:
            script_output = view.script_output
            recommended_task = view.reasoning_recommended_task

            if script_output is not None:
                query_to_process = truncate_middle(
                    str(script_output),
                    max_script_output_chars(),
                    label="terminal output",
                )
                updates["command"] = None
                updates["write_script"] = None
                updates["write_script_language"] = None
            else:
                self.clear_chat_history_except_persistent()
                query_to_process = truncate_middle(
                    query.strip(),
                    max_reasoning_objective_chars(),
                    label="task briefing",
                )

            if recommended_task:
                rt = truncate_middle(
                    str(recommended_task),
                    8000,
                    label="recommended task",
                )
                query_to_process = (
                    f"{query_to_process}\n\n"
                    f"Suggested next step:\n{rt}"
                )

            updates["query_to_process"] = query_to_process

            system_base = str(self.system_prompt)
            overlay = self._session_context_overlay(view)
            if overlay:
                system_base = (
                    f"{system_base}\n\n<SessionPlanningAndContext>\n{overlay}\n</SessionPlanningAndContext>"
                )

            self.chat_history.add_user_message(HumanMessage(content=query_to_process))
            history_with_memory = self.format_chat_history(self.chat_history)

            structured_response = self.llm.invoke({"system": system_base, "history": history_with_memory})

            logger.info("GENERATIVE_AGENT_RESPONSE %s", structured_response)

            try:
                updates["generative_agent_response"] = (
                    structured_response.model_dump_json()
                    if hasattr(structured_response, "model_dump_json")
                    else str(structured_response)
                )
            except Exception:
                updates["generative_agent_response"] = str(structured_response)

            sg = structured_response.submitted_goal
            ws = getattr(structured_response, "write_script", None)
            if sg is not None and str(sg).strip():
                submitted = str(sg).strip()
                updates["command"] = None
                updates["write_script"] = None
                updates["write_script_language"] = None
                updates["submitted_goal"] = submitted
                updates["should_stop"] = True
                self.chat_history.add_ai_message(
                    AIMessage(content=f"Submitted goal: {submitted}")
                )
            elif ws is not None and str(ws).strip():
                lang = getattr(structured_response, "write_script_language", None)
                updates["write_script"] = str(ws).strip()
                updates["write_script_language"] = (str(lang).strip() if lang else None) or "py"
                updates["command"] = None
                updates["submitted_goal"] = None
                updates["should_stop"] = False
                preview = updates["write_script"][:200] + ("…" if len(updates["write_script"]) > 200 else "")
                self.chat_history.add_ai_message(
                    AIMessage(
                        content=f"Write script ({updates['write_script_language']}): {preview}"
                    )
                )
                if structured_response.reasoning:
                    reasoning_content = " | ".join(structured_response.reasoning)
                    self.chat_history.add_ai_message(AIMessage(content=f"Reasoning: {reasoning_content}"))
            else:
                cmd = structured_response.command
                if cmd:
                    self.chat_history.add_ai_message(AIMessage(content=f"Command: {cmd}"))
                if structured_response.reasoning:
                    reasoning_content = " | ".join(structured_response.reasoning)
                    self.chat_history.add_ai_message(AIMessage(content=f"Reasoning: {reasoning_content}"))
                updates["command"] = cmd
                updates["write_script"] = None
                updates["write_script_language"] = None
                updates["should_stop"] = False
                updates["submitted_goal"] = None

            print(
                f"GENERATIVE_AGENT_RESPONSE should_stop={updates.get('should_stop')} "
                f"submitted_goal={updates.get('submitted_goal')!r} command={updates.get('command')!r} "
                f"write_script_len={len(updates.get('write_script') or '')}",
                flush=True,
            )

            merged = self.snapshot_after(state, updates)
            try:
                if updates.get("submitted_goal"):
                    structured_output = {
                        "type": "submitted_goal",
                        "submitted_goal": updates.get("submitted_goal"),
                        "reasoning": structured_response.reasoning,
                        "ethical": structured_response.ethical,
                    }
                elif updates.get("write_script"):
                    structured_output = {
                        "type": "write_script",
                        "write_script_language": updates.get("write_script_language"),
                        "reasoning": structured_response.reasoning,
                        "ethical": structured_response.ethical,
                    }
                else:
                    structured_output = {
                        "type": "command",
                        "reasoning": structured_response.reasoning,
                        "command": structured_response.command,
                        "ethical": structured_response.ethical,
                    }
                self.memory_logger.log_comprehensive_interaction(
                    session_id=view.session_id or "unknown",
                    agent_type="generation_agent",
                    original_query=query,
                    query_to_process=query_to_process,
                    state=merged,
                    structured_response=structured_output,
                )
            except Exception as e:
                logger.warning("Failed to log comprehensive memory for generation agent: %s", e)

            return updates

        except Exception as e:
            logger.exception("Generative agent failed: %s", e)
            print(f"GENERATIVE_AGENT_RESPONSE error=True {e!r}", flush=True)
            return {
                "should_stop": True,
                "generative_agent_response": f"I encountered an error generating a response: {e!s}",
            }
