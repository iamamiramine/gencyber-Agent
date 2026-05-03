"""
Legacy hook: leaderboard HTTP endpoints were removed from the challenge / workbench API.
"""

from __future__ import annotations

from typing import Any, Dict


def _state_excerpt(state: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "query",
        "submitted_goal",
        "script_output",
        "generative_agent_response",
        "reasoning_task_tree",
        "reasoning_cycle_count",
        "should_stop",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        v = state.get(k)
        if v is None:
            continue
        s = v if isinstance(v, str) else str(v)
        if len(s) > 12000:
            s = s[:6000] + "\n…\n" + s[-6000:]
        out[k] = s
    return out


def record_leaderboard_run(
    leaderboard_cfg: Dict[str, Any],
    api_response: Dict[str, Any],
    full_state: Dict[str, Any],
) -> None:
    _ = (leaderboard_cfg, api_response, _state_excerpt(full_state if isinstance(full_state, dict) else {}))
    return
