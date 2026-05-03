# recon_agent.py
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import logging

from langchain_core.prompts.chat import HumanMessage, AIMessage
from core.agents.base_agent_state_spec import BaseStatefulAgent
from core.helpers.llm_context_limits import max_recon_context_chars, max_script_output_chars, truncate_middle

logger = logging.getLogger(__name__)


class ReconAgentState(TypedDict, total=False):
    """Keys the recon agent reads from graph state (optional keys, ``TypedDict``)."""

    script_output: Annotated[str | None, "graph"]
    objectives: Annotated[List[Any] | None, "pm_agent"]
    constraints: Annotated[List[Any] | None, "pm_agent"]
    goal_format: Annotated[str | None, "pm_agent"]
    planning_context: Annotated[str | None, "pm_agent"]
    context: Annotated[str | None, "recon_agent"]


class ReconAgent(BaseStatefulAgent):
    """Produce environmental context."""

    agent_name = "recon"
    description = "Enriches chat context from planning fields and recent execution output; writes recon ``context`` on graph state."
    state_schema = ReconAgentState

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
        self._node_id = node_id or "recon"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        view = self.read_state(state)
        script_output = view.script_output
        env_info = ""
        if script_output:
            snippet = truncate_middle(
                str(script_output),
                min(4000, max_script_output_chars()),
                label="execution output",
            )
            env_info = f"\n\nCurrent execution output (if any):\n{snippet}"

        plan_parts: list[str] = []
        for key in ("objectives", "constraints", "goal_format", "planning_context"):
            val = getattr(view, key, None)
            if val is not None and str(val).strip():
                plan_parts.append(f"{key}: {val}")
        plan_text = "\n".join(plan_parts).strip() or "(no structured plan fields yet)"
        plan_text = truncate_middle(
            plan_text,
            max_recon_context_chars(),
            label="planning context",
        )

        try:
            self.chat_history.add_user_message(
                HumanMessage(
                    content=f"Prior planning context:\n{plan_text}{env_info}"
                )
            )
            history = self.formatter.format_chat_history(
                self.chat_history, self.generation_config.get("model_name", "")
            )
            response = self.llm.invoke({
                "system": self.system_prompt,
                "history": history,
            })
            if hasattr(response, "to_injectable_context"):
                text = response.to_injectable_context()
            else:
                text = response.model_dump_json()
            self.chat_history.add_ai_message(AIMessage(content=text))
            ctx_block = text.strip() if isinstance(text, str) else str(text)
            cap = max(max_recon_context_chars(), 16_000)
            ctx_block = truncate_middle(ctx_block, cap, label="recon summary") or ctx_block
            print(f"RECON_AGENT_RESPONSE context_len={len(ctx_block)}", flush=True)
            return {"context": ctx_block}
        except Exception as e:
            logger.exception("Recon step failed: %s", e)
            print(f"RECON_AGENT_RESPONSE error=True {e!r}", flush=True)
            return {}
