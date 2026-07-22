"""Function-split agent memory stores (token-level, MongoDB-backed).

Replaces the flat ``execution_evidence`` string in the deep-planner workflow with
typed, session-scoped stores so memory can be split by *function* (case / feedback
/ strategy), folded, and retrieved on demand instead of re-injected wholesale.

See ``docs/agent-memory-design.md`` (what/where) and ``docs/agent-memory-theory.md``
(why). Everything here is **fail-open**: any error is logged and swallowed so a
memory outage degrades run quality but never breaks a run.

Collections (all in ``MONGODB_DATABASE``, partitioned by ``session_id``):

- ``agent_memory_cases``    — ``{session_id, ts, subagent, command, raw_output, abstracted}``
- ``agent_memory_feedback`` — ``{session_id, ts, submitted_goal, rejection_reason}``
- ``agent_memory_strategy`` — fold cards ``{session_id, ts, subagent, task, findings, artifacts, next_step}``
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from infrastructure.repository.mongodb_repository import get_mongodb_client

logger = logging.getLogger(__name__)

_CASES = "agent_memory_cases"
_FEEDBACK = "agent_memory_feedback"
_STRATEGY = "agent_memory_strategy"

# Per-recall snippet budget applied AFTER relevance selection (not a blind tail-cut).
_RECALL_SNIPPET_CHARS = int(os.getenv("MEMORY_RECALL_SNIPPET_CHARS", "1500"))

_WORD_RE = re.compile(r"[a-zA-Z0-9_./-]{2,}")

# Module-level cache so we build indexes once per process.
_db = None
_indexed = False


def _get_db():
    global _db
    if _db is None:
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")
        _db = get_mongodb_client()[db_name]
        _ensure_indexes(_db)
    return _db


def _ensure_indexes(db) -> None:
    global _indexed
    if _indexed:
        return
    try:
        for coll in (_CASES, _FEEDBACK, _STRATEGY):
            db[coll].create_index([("session_id", 1), ("ts", -1)])
        _indexed = True
    except Exception as exc:  # fail-open
        logger.warning("agent_memory: index setup failed: %s", exc)


def _tokens(text: str) -> set:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


# --------------------------------------------------------------------------- #
# Formation (writes)
# --------------------------------------------------------------------------- #
def record_case(
    session_id: Optional[str],
    subagent: Optional[str],
    command: Optional[str],
    raw_output: Optional[str],
    abstracted: Optional[str] = None,
) -> None:
    """Persist one tool observation (case-based experiential memory)."""
    if not session_id:
        return
    try:
        _get_db()[_CASES].insert_one(
            {
                "session_id": session_id,
                "ts": datetime.utcnow(),
                "subagent": subagent or "",
                "command": command or "",
                "raw_output": raw_output or "",
                "abstracted": abstracted or "",
            }
        )
    except Exception as exc:  # fail-open
        logger.warning("agent_memory.record_case failed: %s", exc)


def record_feedback(
    session_id: Optional[str],
    submitted_goal: Optional[str],
    rejection_reason: Optional[str],
) -> None:
    """Persist a rejected submission and its reason (feedback memory)."""
    if not session_id:
        return
    try:
        _get_db()[_FEEDBACK].insert_one(
            {
                "session_id": session_id,
                "ts": datetime.utcnow(),
                "submitted_goal": submitted_goal or "",
                "rejection_reason": rejection_reason or "",
            }
        )
    except Exception as exc:  # fail-open
        logger.warning("agent_memory.record_feedback failed: %s", exc)


def record_fold(session_id: Optional[str], card: Dict[str, Any]) -> None:
    """Persist a folded sub-trajectory result card (strategy memory)."""
    if not session_id or not isinstance(card, dict):
        return
    try:
        doc = {
            "session_id": session_id,
            "ts": datetime.utcnow(),
            "subagent": str(card.get("subagent") or ""),
            "task": str(card.get("task") or ""),
            "findings": str(card.get("findings") or ""),
            "artifacts": str(card.get("artifacts") or ""),
            "next_step": str(card.get("next_step") or ""),
        }
        _get_db()[_STRATEGY].insert_one(doc)
    except Exception as exc:  # fail-open
        logger.warning("agent_memory.record_fold failed: %s", exc)


# --------------------------------------------------------------------------- #
# Retrieval (reads)
# --------------------------------------------------------------------------- #
def recent_folds(session_id: Optional[str], n: int = 6) -> List[Dict[str, Any]]:
    """Newest-first fold cards for this session (for cross-delegation injection)."""
    if not session_id:
        return []
    try:
        cur = (
            _get_db()[_STRATEGY]
            .find({"session_id": session_id}, {"_id": 0})
            .sort("ts", -1)
            .limit(max(1, int(n)))
        )
        return list(cur)
    except Exception as exc:  # fail-open
        logger.warning("agent_memory.recent_folds failed: %s", exc)
        return []


def recall(session_id: Optional[str], query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Top-k prior cases/strategy ranked by recency + keyword overlap with ``query``.

    v1 ranking (no embeddings): score = term_overlap + recency_bonus. A single CTF
    session yields tens of observations, where lexical ranking is adequate; the
    write/read split here leaves a clean seam to swap in vector retrieval later.
    """
    if not session_id:
        return []
    q_tokens = _tokens(query)
    try:
        db = _get_db()
        # Pull a recency-bounded candidate window, then rank in Python.
        cases = list(
            db[_CASES]
            .find({"session_id": session_id}, {"_id": 0})
            .sort("ts", -1)
            .limit(200)
        )
        strategy = list(
            db[_STRATEGY]
            .find({"session_id": session_id}, {"_id": 0})
            .sort("ts", -1)
            .limit(100)
        )
    except Exception as exc:  # fail-open
        logger.warning("agent_memory.recall failed: %s", exc)
        return []

    scored: List[tuple] = []
    total = len(cases) + len(strategy)
    for rank, doc in enumerate(cases + strategy):
        text = " ".join(
            str(doc.get(f) or "")
            for f in ("command", "abstracted", "raw_output", "task", "findings", "next_step")
        )
        overlap = len(q_tokens & _tokens(text))
        recency_bonus = (total - rank) / max(1, total)  # 0..1, newer = higher
        score = overlap + recency_bonus
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, doc in scored[: max(1, int(k))]:
        snippet = doc.get("abstracted") or doc.get("findings") or doc.get("raw_output") or ""
        out.append(
            {
                "command": doc.get("command", ""),
                "task": doc.get("task", ""),
                "snippet": str(snippet)[:_RECALL_SNIPPET_CHARS],
                "score": round(float(score), 3),
            }
        )
    return out
