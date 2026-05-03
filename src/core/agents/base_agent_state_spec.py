"""Base declaration for agent graph-state contracts (parallel to LangChain ``BaseTool`` metadata)."""

from __future__ import annotations

from abc import ABC
from types import SimpleNamespace
from typing import Any, ClassVar, Dict, Mapping, get_type_hints


class BaseStatefulAgent(ABC):
    """Declare ``agent_name``, ``description``, and ``state_schema`` like tools declare name/description/args_schema."""

    agent_name: ClassVar[str]
    description: ClassVar[str]
    # Each agent: a ``TypedDict`` (total=False) listing this node's state keys.
    state_schema: ClassVar[type]

    @classmethod
    def get_state_json_schema(cls) -> dict[str, Any]:
        """Loose JSON-schema hint for this slice (``TypedDict`` has no Pydantic ``model_json_schema``)."""
        name = getattr(cls.state_schema, "__name__", "StateSlice")
        try:
            keys = list(get_type_hints(cls.state_schema).keys())
        except Exception:
            keys = []
        return {
            "title": name,
            "type": "object",
            "additionalProperties": True,
            "x-typeddict-keys": keys,
        }

    @classmethod
    def _state_slice_dict(cls, state: Mapping[str, Any]) -> Dict[str, Any]:
        keys = get_type_hints(cls.state_schema).keys()
        d = dict(state)
        return {k: d.get(k) for k in keys}

    @classmethod
    def validate_state_projection(cls, state: Mapping[str, Any]) -> SimpleNamespace:
        """Project graph state to this agent's keys; attribute access via ``SimpleNamespace``."""
        return SimpleNamespace(**cls._state_slice_dict(state))

    def read_state(self, state: Mapping[str, Any]) -> SimpleNamespace:
        """Typed projection of the graph dict using this agent's ``state_schema``."""
        return self.validate_state_projection(state)

    @staticmethod
    def snapshot_after(state: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
        """Graph state as if ``updates`` were merged (e.g. for logging); does not mutate ``state``."""
        merged: Dict[str, Any] = dict(state)
        merged.update(updates)
        return merged
