# pm_agent.py
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import logging

from langchain_core.prompts.chat import HumanMessage, AIMessage
from core.agents.base_agent_state_spec import BaseStatefulAgent

logger = logging.getLogger(__name__)


class PMAgentState(TypedDict, total=False):
    """Keys the PM (briefing) agent reads or writes (optional keys, ``TypedDict``)."""

    query: Annotated[str | None, "graph"]
    objectives: Annotated[List[Any] | None, "pm_agent"]
    constraints: Annotated[List[Any] | None, "pm_agent"]
    goal_format: Annotated[str | None, "pm_agent"]
    planning_context: Annotated[str | None, "pm_agent"]


class PMAgent(BaseStatefulAgent):
    """Parses the initial task briefing into structured session planning fields in state."""

    agent_name = "pm"
    description = "Parses the task briefing into objectives, constraints, and goal format on graph state."
    state_schema = PMAgentState

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
        self._node_id = node_id or "pm"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        view = self.read_state(state)
        query = (view.query or "").strip()
        if not query:
            return {}

        try:
            self.chat_history.add_user_message(
                HumanMessage(content=f"Task briefing:\n\n{query}")
            )
            history = self.formatter.format_chat_history(
                self.chat_history, self.generation_config.get("model_name", "")
            )
            response = self.llm.invoke({
                "system": self.system_prompt,
                "history": history,
            })

            updates = self._structured_updates_from_response(response)

            if hasattr(response, "to_injectable_context"):
                text = response.to_injectable_context()
            else:
                text = response.model_dump_json()
            self.chat_history.add_ai_message(AIMessage(content=text))
            print(f"PM_AGENT_RESPONSE {updates}", flush=True)
            return updates
        except Exception as e:
            logger.exception("Brief intake failed: %s", e)
            print(f"PM_AGENT_RESPONSE error=True {e!r}", flush=True)
            return {}

    def _structured_updates_from_response(self, response: Any) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        try:
            objectives = getattr(response, "objectives", None)
            if isinstance(objectives, list):
                updates["objectives"] = list(objectives)
            constraints = getattr(response, "constraints", None)
            if isinstance(constraints, list):
                updates["constraints"] = list(constraints)
            goal_format = getattr(response, "flag_or_goal_format", None)
            if isinstance(goal_format, str) and goal_format.strip():
                updates["goal_format"] = goal_format.strip()
            if hasattr(response, "to_injectable_context"):
                pc = response.to_injectable_context()
                if isinstance(pc, str) and pc.strip():
                    updates["planning_context"] = pc.strip()
        except Exception as e:
            logger.warning("Could not promote structured fields to state: %s", e)
        return updates
