"""Central caps for text injected into LLM calls to avoid context_length_exceeded."""

from __future__ import annotations

import os
from typing import Optional


def _ienv(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def max_chat_history_chars() -> int:
    """Tail of formatted chat history string (per agent) sent with system prompt."""
    return _ienv("LLM_CHAT_HISTORY_MAX_CHARS", 100_000)


def max_script_output_chars() -> int:
    """Terminal / tool output embedded in one user turn."""
    return _ienv("LLM_SCRIPT_OUTPUT_MAX_CHARS", 45_000)


def max_recon_context_chars() -> int:
    """Recon / environmental context block in overlays and reasoning."""
    return _ienv("LLM_RECON_CONTEXT_MAX_CHARS", 12_000)


def max_reasoning_objective_chars() -> int:
    """Initial user objective string in reasoning initialize mode."""
    return _ienv("LLM_REASONING_OBJECTIVE_MAX_CHARS", 24_000)


def max_generative_response_snippet_chars() -> int:
    """Prior generative JSON/text passed as evidence to reasoning."""
    return _ienv("LLM_GENERATIVE_EVIDENCE_MAX_CHARS", 24_000)


def truncate_middle(text: Optional[str], max_chars: int, *, label: str = "content") -> Optional[str]:
    """Keep start + end so headers and recent lines remain visible."""
    if text is None or max_chars <= 0:
        return text
    s = str(text)
    if len(s) <= max_chars:
        return s
    sep = 96
    piece = max(400, (max_chars - sep) // 2)
    head = piece
    tail = piece
    omitted = len(s) - head - tail
    if omitted <= 0:
        return s[-max_chars:]
    return (
        s[:head]
        + f"\n\n[… {label}: {omitted} characters omitted …]\n\n"
        + s[-tail:]
    )
