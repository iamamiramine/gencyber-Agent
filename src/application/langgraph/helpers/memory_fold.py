"""Cheap-LLM fold helpers for the agent memory subsystem.

Two consolidation operations (DYNAMICS axis), both running on the existing
specialist model handle (gpt-4o-mini over OpenRouter):

- ``abstract_observation`` — Piece 2a: compress one verbose tool output to a salient
  observation before it enters the message history.
- ``fold_subtrajectory`` — Piece 2b: distill a finished specialist run into a
  compact ``{findings, artifacts, next_step}`` strategy card.

Both are **fail-open**: any model/parse error falls back to a safe default so a
fold failure never breaks a run. See ``docs/agent-memory-design.md``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Abstract only outputs larger than this; smaller outputs pass through verbatim.
ABSTRACT_MIN_CHARS = int(os.getenv("MEMORY_ABSTRACT_MIN_CHARS", "2000"))
# Hard cap on raw text handed to the fold model (protects the fold call itself).
_FOLD_INPUT_CAP = int(os.getenv("MEMORY_FOLD_INPUT_CAP", "16000"))

_ABSTRACT_SYS = (
    "You compress raw command-line / tool output into a short, factual observation "
    "for an autonomous CTF agent. Preserve EXACTLY: commands run, file paths, ports, "
    "hostnames, credentials, hashes, and any flag-format strings (e.g. flag{...}, "
    "csawctf{...}). Drop banners, repetition, and decorative noise. Output only the "
    "compressed observation, no preamble. Target under 120 words."
)

_FOLD_SYS = (
    "You fold a completed specialist sub-trajectory into a compact result card for an "
    "autonomous CTF planner. Read the task and the transcript, then return STRICT JSON "
    'with exactly these keys: {"findings": str, "artifacts": str, "next_step": str}. '
    "'findings' = concrete facts discovered (paths, ports, creds, vuln, partial flag). "
    "'artifacts' = files written or tools produced (paths). "
    "'next_step' = the single most useful next action toward the goal. "
    "Preserve flag-format strings verbatim. No prose outside the JSON."
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text", "")))
            else:
                parts.append(str(p))
        return " ".join(parts)
    return str(content or "")


def abstract_observation(model: Any, command: Optional[str], raw_output: Optional[str]) -> str:
    """Return a compressed observation for ``raw_output``.

    Small outputs (< ``ABSTRACT_MIN_CHARS``) are returned verbatim — abstraction is
    only worth an LLM call on genuinely verbose output. Fail-open: on any error the
    raw output is returned unchanged.
    """
    raw = raw_output or ""
    if len(raw) < ABSTRACT_MIN_CHARS or model is None:
        return raw
    try:
        msg = (
            f"Command:\n{command or '(n/a)'}\n\n"
            f"Raw output (may be truncated):\n{raw[:_FOLD_INPUT_CAP]}"
        )
        resp = model.invoke([SystemMessage(content=_ABSTRACT_SYS), HumanMessage(content=msg)])
        text = _content_text(getattr(resp, "content", "")).strip()
        return text or raw
    except Exception as exc:  # fail-open
        logger.warning("memory_fold.abstract_observation failed: %s", exc)
        return raw


def _render_transcript(messages: List[Any], cap: int) -> str:
    """Flatten a message list to a compact text transcript for folding."""
    lines: List[str] = []
    for m in messages or []:
        role = getattr(m, "type", None) or m.__class__.__name__
        text = _content_text(getattr(m, "content", ""))
        if not text.strip():
            continue
        lines.append(f"[{role}] {text}")
    joined = "\n".join(lines)
    if len(joined) > cap:
        # Keep the tail — most recent findings matter most for the next step.
        joined = "…(earlier omitted)\n" + joined[-cap:]
    return joined


def _coerce_card(text: str) -> Optional[Dict[str, str]]:
    """Parse the fold model's JSON (tolerant of code fences / surrounding prose)."""
    if not text:
        return None
    candidate = text.strip()
    fence = re.search(r"\{.*\}", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(0)
    try:
        data = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return {
        "findings": str(data.get("findings", "")).strip(),
        "artifacts": str(data.get("artifacts", "")).strip(),
        "next_step": str(data.get("next_step", "")).strip(),
    }


def fold_subtrajectory(
    model: Any, task: Optional[str], messages: List[Any]
) -> Optional[Dict[str, str]]:
    """Fold a finished specialist run into a ``{findings, artifacts, next_step}`` card.

    Returns ``None`` (caller skips persistence) if there is nothing to fold or the
    model/parse fails — fail-open.
    """
    if model is None:
        return None
    transcript = _render_transcript(messages, _FOLD_INPUT_CAP)
    if not transcript.strip():
        return None
    try:
        msg = f"Task:\n{task or '(unspecified)'}\n\nTranscript:\n{transcript}"
        resp = model.invoke([SystemMessage(content=_FOLD_SYS), HumanMessage(content=msg)])
        text = _content_text(getattr(resp, "content", ""))
        card = _coerce_card(text)
        if card is None:
            # Fall back to a minimal card so something useful is still carried forward.
            return {"findings": text.strip()[:1500], "artifacts": "", "next_step": ""}
        return card
    except Exception as exc:  # fail-open
        logger.warning("memory_fold.fold_subtrajectory failed: %s", exc)
        return None
