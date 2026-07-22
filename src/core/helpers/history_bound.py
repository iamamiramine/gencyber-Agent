"""Pairing-safe message-history trimming for long agent runs.

Extracted from the deep workflow so it is dependency-free (no langgraph/deepagents
imports) and unit-testable on any Python: message *kind* is detected by duck-typing
(a ToolMessage carries ``tool_call_id``), never by importing the concrete classes.

Why: on a long spiral (consolidated-experiment regexpire, 867 steps / 113 submit
calls) an agent's message history grew unbounded and blew past the model's 128k
context, returning an HTTP 400 that killed the run. :func:`select_messages_to_drop`
picks the oldest whole tool-rounds to remove so the history fits a char budget,
always keeping the initial briefing and a recent suffix, and only ever cutting at a
round boundary so no tool_call is left without its response (which OpenAI rejects).
"""

from __future__ import annotations

from typing import Any, List


def message_chars(m: Any) -> int:
    """Rough size of one message: its text content plus any tool-call args."""
    content = getattr(m, "content", "")
    n = len(content if isinstance(content, str) else str(content))
    for tc in getattr(m, "tool_calls", None) or []:
        try:
            n += len(str(tc.get("args") or ""))
        except AttributeError:
            pass
    return n


def is_tool_message(m: Any) -> bool:
    """True for a ToolMessage (tool result), detected without importing the class."""
    return getattr(m, "tool_call_id", None) is not None or type(m).__name__ == "ToolMessage"


def is_round_start(m: Any) -> bool:
    """A safe cut boundary: any message that is NOT a tool result (i.e. a Human or
    assistant turn). Cutting the kept suffix so it BEGINS at a round start guarantees
    we never keep a ToolMessage whose assistant turn was dropped, nor drop an
    assistant turn whose tool results are kept — the two ways trimming orphans a call.
    """
    return not is_tool_message(m)


def select_messages_to_drop(
    messages: List[Any],
    *,
    limit_chars: int,
    keep_recent: int,
) -> List[Any]:
    """Pick the oldest messages to remove so history fits ``limit_chars``.

    Returns the message objects to drop (the caller emits a ``RemoveMessage`` per id).
    Always keeps ``messages[0]`` (the briefing) and at least ``keep_recent`` trailing
    messages, and only cuts at a round boundary. Returns ``[]`` when disabled, under
    budget, when there is too little history to trim, or when no safe cut exists.
    """
    if limit_chars <= 0 or len(messages) <= keep_recent + 1:
        return []
    total = sum(message_chars(m) for m in messages)
    if total <= limit_chars:
        return []
    # Largest boundary at or before (len - keep_recent), never index <= 1 (keep the
    # briefing at 0). A larger cut removes more, shrinking the history further.
    target = min(len(messages) - keep_recent, len(messages) - 1)
    cut = None
    for i in range(target, 1, -1):
        if is_round_start(messages[i]):
            cut = i
            break
    if cut is None or cut <= 1:
        return []
    return [m for m in messages[1:cut] if getattr(m, "id", None) is not None]
