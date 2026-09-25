"""Tests for the submit-goal tools.

Validation is now **per-tool**:
  - ``NYUCTFSubmitGoalTool`` delegates the accept/reject decision to the workbench
    (``validate_submission_remote``) — the agent holds no ground-truth flag. We
    monkeypatch that remote call so the test needs no running workbench.
  - ``BaselineSubmitGoalTool`` is evidence-based: it accepts a value that appears
    verbatim in the latest real tool output (``script_output``) and rejects an
    answer with no supporting evidence.

Runnable both as a script and under pytest:

    PYTHONPATH=src python3 tests/test_submit_validator.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import infrastructure.benchmark_flag_client as flag_client
from core.tools.submit_goal_tool import (
    BaselineSubmitGoalTool,
    NYUCTFSubmitGoalTool,
    build_submit_goal_tool,
)


class _patched_remote:
    """Monkeypatch ``validate_submission_remote`` for the duration of a ``with``."""

    def __init__(self, fn):
        self._fn = fn
        self._orig = None

    def __enter__(self):
        self._orig = flag_client.validate_submission_remote
        flag_client.validate_submission_remote = self._fn
        return self

    def __exit__(self, *exc):
        flag_client.validate_submission_remote = self._orig
        return False


def _accept_only(good: str):
    def _fn(*, session_id, candidate):
        if candidate == good:
            return True, None
        return False, "flag rejected by workbench"

    return _fn


def test_nyuctf_accepts_workbench_confirmed_flag():
    with _patched_remote(_accept_only("flag{a}")):
        out = NYUCTFSubmitGoalTool(session_id="s1")(
            {"submitted_goal": "flag{a}", "session_id": "s1", "script_output": ""}
        )
    assert out["submission_verified"] is True
    assert out["submitted_goal"] == "flag{a}"


def test_nyuctf_rejects_workbench_rejected_flag():
    with _patched_remote(_accept_only("flag{a}")):
        out = NYUCTFSubmitGoalTool(session_id="s1")(
            {"submitted_goal": "flag{b}", "session_id": "s1", "script_output": "x"}
        )
    assert out["submission_verified"] is False
    assert out["submitted_goal"] is None
    assert "[SUBMIT REJECTED]" in out["script_output"]


def test_nyuctf_unreachable_workbench_rejects():
    def _unreachable(*, session_id, candidate):
        return None, "workbench validation unavailable"

    with _patched_remote(_unreachable):
        out = NYUCTFSubmitGoalTool(session_id="s1")(
            {"submitted_goal": "flag{a}", "session_id": "s1", "script_output": ""}
        )
    assert out["submission_verified"] is False


def test_baseline_accepts_recovered_value():
    out = BaselineSubmitGoalTool()(
        {"submitted_goal": "flag{ok}", "script_output": "we found flag{ok} here"}
    )
    assert out["submission_verified"] is True
    assert out["submitted_goal"] == "flag{ok}"


def test_baseline_rejects_unsupported_value():
    out = BaselineSubmitGoalTool()(
        {"submitted_goal": "flag{guess}", "script_output": "nothing relevant"}
    )
    assert out["submission_verified"] is False
    assert out["submitted_goal"] is None
    assert "[SUBMIT REJECTED]" in out["script_output"]


def test_placeholder_rejected_by_any_tool():
    out = BaselineSubmitGoalTool()(
        {"submitted_goal": "flag{example}", "script_output": "flag{example}"}
    )
    assert out["submission_verified"] is False


def test_build_selects_nyuctf_then_falls_back():
    assert isinstance(
        build_submit_goal_tool(["nyuctf_submit_goal"], session_id="s1"),
        NYUCTFSubmitGoalTool,
    )
    assert isinstance(
        build_submit_goal_tool(["execute_script"], session_id="s1"),
        BaselineSubmitGoalTool,
    )
    assert isinstance(
        build_submit_goal_tool(None, session_id="s1"),
        BaselineSubmitGoalTool,
    )


def _main() -> int:
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            passed += 1
            print(f"  OK  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
