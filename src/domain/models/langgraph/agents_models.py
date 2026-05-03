from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type
from langchain_community.chat_message_histories import ChatMessageHistory


@dataclass(slots=True)
class AgentConfig:
    name: str
    prompt_key: str
    structured_output: Type
    agent_cls: Type
    share_history_key: Optional[str] = None
    bind_max_tokens: bool = True
    base_path: str = "config"
    extra_prompt_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRuntime:
    config: AgentConfig
    model: Any = None
    chain: Any = None
    agent: Any = None
    system_prompt: Optional[str] = None
    chat_history: Optional[ChatMessageHistory] = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.chain is not None and self.agent is not None