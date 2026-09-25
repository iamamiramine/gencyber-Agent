"""Standalone tests for CTF workflow hardening modules.

    PYTHONPATH=src python3 tests/test_workflow_hardening.py

Submission validation is per-tool: the NYU CTF tool delegates accept/reject to the
workbench (monkeypatched here so no workbench is needed), and the baseline tool
accepts only a value present verbatim in real tool output.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import infrastructure.benchmark_flag_client as flag_client
from core.tools.submit_goal_tool import (
    BaselineSubmitGoalTool,
    NYUCTFSubmitGoalTool,
    build_submit_goal_tool,
    looks_like_placeholder,
)

_passed = 0
_failed = 0


def check(name: str, cond: bool) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  OK  {name}")
    else:
        _failed += 1
        print(f"FAIL  {name}")


def main() -> int:
    # NYU CTF submit tool delegates the comparison to the workbench. Monkeypatch the
    # remote call so the test is hermetic.
    def fake_remote(*, session_id, candidate):
        if candidate == "flag{a}":
            return True, None
        return False, "flag rejected by workbench"

    orig = flag_client.validate_submission_remote
    flag_client.validate_submission_remote = fake_remote
    try:
        tool = NYUCTFSubmitGoalTool(session_id="s1")
        out = tool({"submitted_goal": "flag{a}", "session_id": "s1", "script_output": ""})
        check("nyuctf accepts workbench-confirmed flag", out["submission_verified"] is True)

        out2 = tool({"submitted_goal": "flag{b}", "session_id": "s1", "script_output": "x"})
        check("nyuctf rejects workbench-rejected flag", out2["submission_verified"] is False)
    finally:
        flag_client.validate_submission_remote = orig

    # Baseline (no-oracle) tool: accept a recovered value, reject an unsupported one.
    b_ok = BaselineSubmitGoalTool()(
        {"submitted_goal": "flag{ok}", "script_output": "recovered flag{ok}"}
    )
    check("baseline accepts recovered value", b_ok["submission_verified"] is True)

    b_no = BaselineSubmitGoalTool()(
        {"submitted_goal": "flag{guess}", "script_output": "nothing"}
    )
    check("baseline rejects unsupported value", b_no["submission_verified"] is False)

    # Placeholder guard applies before any tool-specific comparison.
    check("placeholder guard flags template", looks_like_placeholder("flag{example}"))
    check("placeholder guard passes real value", not looks_like_placeholder("flag{abc123}"))

    # Tool selection from the UI-attached tools list.
    check(
        "build picks nyuctf tool",
        isinstance(build_submit_goal_tool(["nyuctf_submit_goal"]), NYUCTFSubmitGoalTool),
    )
    check(
        "build falls back to baseline",
        isinstance(build_submit_goal_tool([]), BaselineSubmitGoalTool),
    )

    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
