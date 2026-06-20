# generative_agent.py
from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Dict, Optional, TypedDict

from langchain_core.prompts.chat import AIMessage, HumanMessage

from core.agents.base_agent_state_spec import BaseStatefulAgent
from infrastructure.services.memory_logger_service import get_memory_logger

logger = logging.getLogger(__name__)


class GenerativeAgentState(TypedDict, total=False):
    """Keys the generative agent reads or updates on graph state (optional keys, ``TypedDict``)."""

    query: Annotated[str | None, "graph"]
    script_output: Annotated[str | None, "graph"]
    query_to_process: Annotated[str | None, "generative_agent"]
    generative_agent_response: Annotated[str | None, "graph"]
    command: Annotated[str | None, "generative_agent"]
    write_script: Annotated[str | None, "generative_agent"]
    write_script_language: Annotated[str | None, "generative_agent"]
    submitted_goal: Annotated[str | None, "generative_agent"]
    session_id: Annotated[str | None, "system"]
    submission_verified: Annotated[bool | None, "generative_agent"]
    submission_rejection_reason: Annotated[str | None, "generative_agent"]


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
        self.persistent_submitted_goals: list[str] = []

    def format_chat_history(self, chat_history):
        return self.formatter.format_chat_history(
            chat_history, self.generation_config["model_name"]
        )

    def clear_chat_history_except_persistent(self) -> None:
        """Wipe the per-turn working memory but keep any recovered submission."""
        for message in self.chat_history.messages:
            if isinstance(message, AIMessage):
                content = message.content
                text = content if isinstance(content, str) else str(content)
                if text.startswith('{"submitted_goal":'):
                    self.persistent_submitted_goals.append(text)
                    break

        self.chat_history.clear()
        for goal_content in self.persistent_submitted_goals:
            try:
                self.chat_history.add_ai_message(
                    AIMessage(content=json.loads(goal_content))
                )
            except json.JSONDecodeError:
                self.chat_history.add_ai_message(AIMessage(content=goal_content))

    def _build_system_prompt(self, state: Dict[str, Any]) -> str:
        """Resolve the system prompt for this turn."""
        return str(self.system_prompt)

    def _resolve_turn_input(
        self, view: Any, state: Dict[str, Any]
    ) -> tuple[str, bool]:
        """Resolve this turn's user-message input."""
        script_output = view.script_output
        observation_turn = script_output is not None
        if observation_turn:
            return str(script_output), True
        self.clear_chat_history_except_persistent()
        return (view.query or "").strip(), False

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        view = self.read_state(state)
        query = view.query or ""
        if not query.strip():
            return {}

        updates: Dict[str, Any] = {}

        try:
            query_to_process, observation_turn = self._resolve_turn_input(view, state)

            updates["query_to_process"] = query_to_process

            system_base = self._build_system_prompt(state)

            self.chat_history.add_user_message(HumanMessage(content=query_to_process))
            history_with_memory = self.format_chat_history(self.chat_history)

            structured_response = self.llm.invoke(
                {"system": system_base, "history": history_with_memory}
            )

            logger.info("GENERATIVE_AGENT_RESPONSE %s", structured_response)

            try:
                updates["generative_agent_response"] = (
                    structured_response.model_dump_json()
                    if hasattr(structured_response, "model_dump_json")
                    else str(structured_response)
                )
            except Exception:
                updates["generative_agent_response"] = str(structured_response)

            reasoning_content = (
                " | ".join(structured_response.reasoning)
                if structured_response.reasoning
                else ""
            )

            sg = structured_response.submitted_goal
            ws = getattr(structured_response, "write_script", None)
            cmd = structured_response.command

            if sg is not None and str(sg).strip():
                submitted = str(sg).strip()
                updates["command"] = None
                updates["write_script"] = None
                updates["write_script_language"] = None
                updates["submitted_goal"] = submitted
                updates["submission_verified"] = None
                updates["submission_rejection_reason"] = None
                self.chat_history.add_ai_message(
                    AIMessage(content=json.dumps({"submitted_goal": submitted}))
                )
                action_type = "submitted_goal"
            elif ws is not None and str(ws).strip():
                lang = getattr(structured_response, "write_script_language", None)
                updates["write_script"] = str(ws).strip()
                updates["write_script_language"] = (
                    (str(lang).strip() if lang else None) or "py"
                )
                updates["command"] = None
                updates["submitted_goal"] = None
                preview = updates["write_script"][:200] + (
                    "…" if len(updates["write_script"]) > 200 else ""
                )
                self.chat_history.add_ai_message(
                    AIMessage(
                        content=f"Write script ({updates['write_script_language']}): {preview}"
                    )
                )
                action_type = "write_script"
            elif cmd is not None and str(cmd).strip():
                updates["command"] = str(cmd)
                updates["write_script"] = None
                updates["write_script_language"] = None
                updates["submitted_goal"] = None
                self.chat_history.add_ai_message(AIMessage(content=f"Command: {cmd}"))
                action_type = "command"
            else:
                updates["command"] = None
                updates["write_script"] = None
                updates["write_script_language"] = None
                updates["submitted_goal"] = None
                action_type = "follow_up"

            if reasoning_content:
                self.chat_history.add_ai_message(
                    AIMessage(content=f"Reasoning: {reasoning_content}")
                )

            print(
                f"GENERATIVE_AGENT_RESPONSE observation_turn={observation_turn} "
                f"action={action_type} submitted_goal={updates.get('submitted_goal')!r} "
                f"command={updates.get('command')!r} "
                f"write_script_len={len(updates.get('write_script') or '')}",
                flush=True,
            )

            merged = self.snapshot_after(state, updates)
            try:
                structured_output: Dict[str, Any] = {
                    "type": action_type,
                    "reasoning": structured_response.reasoning,
                    "ethical": structured_response.ethical,
                }
                if action_type == "submitted_goal":
                    structured_output["submitted_goal"] = updates.get("submitted_goal")
                elif action_type == "write_script":
                    structured_output["write_script_language"] = updates.get(
                        "write_script_language"
                    )
                elif action_type == "command":
                    structured_output["command"] = structured_response.command
                self.memory_logger.log_comprehensive_interaction(
                    session_id=view.session_id or "unknown",
                    agent_type="generation_agent",
                    original_query=query,
                    query_to_process=query_to_process,
                    state=merged,
                    structured_response=structured_output,
                )
            except Exception as e:
                logger.warning(
                    "Failed to log comprehensive memory for generation agent: %s", e
                )

            return updates

        except Exception as e:
            logger.exception("Generative agent failed: %s", e)
            print(f"GENERATIVE_AGENT_RESPONSE error=True {e!r}", flush=True)
            return {
                "generative_agent_response": (
                    f"I encountered an error generating a response: {e!s}"
                ),
            }
