# deep_generative_agent.py
"""DeepAgents-backed generative agent.

Same role as :class:`core.agents.generative_agent.GenerativeAgent` — drive a
ReAct loop over the workbench terminal — but built on the LangChain *DeepAgents*
native tool-calling loop instead of structured-output routing. This class is a
thin *config carrier*: the pipeline builds it with the same constructor signature
as every other agent, and the :class:`DeepGenerativeWorkflow` builder reads its
``model`` + ``system_prompt`` to assemble the deep agent graph.

The structured-output ``chain`` the pipeline passes in is intentionally ignored:
a DeepAgent binds tools to a RAW chat model, so we rebuild one from the same
model params via :func:`create_llm`.
"""

from __future__ import annotations

import logging
from typing import Optional

from application.langgraph.helpers.langraph_helpers import create_llm

logger = logging.getLogger(__name__)


class DeepGenerativeAgent:
    """Carries the raw model + resolved system prompt for the deep agent graph."""

    agent_name = "generative"
    description = "DeepAgents native tool-calling generative agent."

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
        # ``llm`` here is the structured-output chain the pipeline builds for every
        # agent. A DeepAgent needs a raw tool-bindable chat model, so we ignore the
        # chain and rebuild the model from the same params.
        self.chain = llm
        self.generation_config = generation_config or {}
        self.chat_history = chat_history
        self.system_prompt = system_prompt
        self.formatter = formatter
        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.model_config_raw = model_config_raw or {}
        self._node_id = node_id or "generative"

        if model_params is None or pipeline_params is None:
            raise ValueError(
                "DeepGenerativeAgent requires model_params and pipeline_params to "
                "build a tool-bindable model"
            )

        self.model = create_llm(
            model_params=model_params,
            pipeline_params=pipeline_params,
            model_config_raw=self.model_config_raw,
        )
