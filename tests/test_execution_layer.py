"""Standalone tests for script execution output composition.

Run with:

    PYTHONPATH=src python3 tests/test_execution_layer.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.tools.script_execution_tool import compose_script_output_for_agent

_passed = 0


def check(name, cond):
    global _passed
    assert cond, f"FAILED: {name}"
    _passed += 1
    print(f"  ok - {name}")


def test_compose_script_output():
    # Faithful to gencyber-0: raw stdout only, no exit/cwd annotation.
    ok = compose_script_output_for_agent(
        {"stdout": "hello", "exit_code": 0, "success": True}
    )
    check("stdout returned verbatim", ok == "hello")

    # Infra/transport error: stdout empty, message surfaced from stderr fallback.
    err = compose_script_output_for_agent(
        {"stdout": "", "stderr": "boom", "exit_code": 1, "success": False}
    )
    check("stderr fallback", err == "boom")

    empty = compose_script_output_for_agent({})
    check("empty stays empty", empty == "")


def test_linux_playbook_present():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "config/prompts/playbooks"
    pb = root / "generative_agent" / "linux_playbook.xml"
    check("generative linux playbook exists", pb.is_file())
    check("generative has HexToBinary", "HexToBinary" in pb.read_text(encoding="utf-8"))


def main():
    print("test_execution_layer:")
    test_compose_script_output()
    test_linux_playbook_present()
    print(f"\nAll execution-layer tests passed: {_passed}")


if __name__ == "__main__":
    main()
