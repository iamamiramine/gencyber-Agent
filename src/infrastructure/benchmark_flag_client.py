"""Delegate submission validation to the workbench (server-only).

The agent holds no benchmark/challenge identity. To validate a submitted flag it
asks the workbench — which owns challenge materialization and therefore the
ground truth — to check the candidate against the challenge bound to *this
session* (the binding is registered when the frontend materializes the
challenge, keyed by the same ``session_id`` the agent runs under).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional, Tuple


def _workbench_base() -> str:
    return (
        os.environ.get("CHALLENGE_TOOLKIT_URL")
        or os.environ.get("GENCYBER_WORKBENCH_URL")
        or "http://localhost:8080"
    ).rstrip("/")


def validate_submission_remote(
    *,
    session_id: str,
    candidate: str,
) -> Tuple[Optional[bool], Optional[str]]:
    """Ask the workbench whether ``candidate`` solves this session's challenge.

    Returns ``(accepted, reason)``:
      - ``(True, None)``  — the workbench confirmed the flag.
      - ``(False, reason)`` — the workbench rejected it (mismatch / no challenge
        bound to the session / no ground-truth oracle for that benchmark).
      - ``(None, reason)`` — the workbench could not be reached or returned no
        verdict; the caller treats this as "could not validate" (i.e. not a win).
    """
    if not session_id:
        return None, "no session id to resolve the active challenge"

    url = f"{_workbench_base()}/benchmarks/validate-submission"
    body = json.dumps({"session_id": session_id, "candidate": candidate}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, ValueError):
        return None, "workbench validation unavailable"

    accepted = data.get("accepted")
    reason = data.get("reason")
    if isinstance(accepted, bool):
        return accepted, reason
    return None, reason or "workbench returned no verdict"
