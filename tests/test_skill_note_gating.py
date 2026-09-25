"""Tests for the skill-note gating hardening fixes.

    PYTHONPATH=src python3 tests/test_skill_note_gating.py

Covers four fixes layered on top of the deep planner's specialist subagents:
  1. read_skill_note duplicate guard (don't re-inject a note already loaded this run).
  2. anchor handling (`name.md#section` resolves to `name.md`).
  3. planner write_todos loop-breaker (identical plans without delegating).
  4. wrong-note steering re-arm (many commands after a note read => re-consult).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from application.langgraph.helpers.skills_helper import (
    _count_prior_note_reads,
    _normalize_note_name,
    make_read_skill_note_tool,
)
from application.langgraph.models.deep_generative_workflow import (
    _MAX_TODOS_STEERS,
    _REARM_AFTER_EXECUTIONS,
    _skill_note_gate_msg,
    _skill_note_rearm_msg,
    _count_text_marker,
    _executions_since_last_note_read,
    _skill_note_gate,
    _todos_signature,
    _trailing_identical_write_todos,
    FlagGateMiddleware,
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


def ai(tool_name, args, tid):
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": tid}])


def read_note_call(note, tid):
    return ai("read_skill_note", {"note": note}, tid)


def exec_call(cmd, tid):
    return ai("execute_script", {"command": cmd}, tid)


def todos_call(items, tid):
    return ai("write_todos", {"todos": [{"content": c} for c in items]}, tid)


# --- Fix 2: anchor / name normalization ------------------------------------------

def test_normalize() -> None:
    check("normalize adds .md", _normalize_note_name("stream-ciphers") == "stream-ciphers.md")
    check("normalize keeps .md", _normalize_note_name("stream-ciphers.md") == "stream-ciphers.md")
    check(
        "normalize strips #anchor",
        _normalize_note_name("classic-ciphers.md#vigenere") == "classic-ciphers.md",
    )
    check(
        "normalize strips dir + anchor",
        _normalize_note_name("notes/classic-ciphers.md#x") == "classic-ciphers.md",
    )
    check("normalize empty -> ''", _normalize_note_name("   ") == "")
    check("normalize bare anchor -> ''", _normalize_note_name("#section") == "")


# --- Fix 1: duplicate read guard --------------------------------------------------

def test_count_prior_reads() -> None:
    msgs = [
        read_note_call("classic-ciphers.md", "a"),
        ToolMessage(content="...", tool_call_id="a"),
        read_note_call("classic-ciphers.md#vigenere", "b"),  # same note via anchor
    ]
    # excluding the current call ("b"), one prior read of the same normalized note.
    check(
        "prior reads counts normalized matches across anchor",
        _count_prior_note_reads(msgs, "classic-ciphers.md", "b") == 1,
    )
    check(
        "prior reads excludes current id",
        _count_prior_note_reads(msgs, "classic-ciphers.md", "a") == 1,
    )
    check(
        "prior reads zero for a different note",
        _count_prior_note_reads(msgs, "stream-ciphers.md", "b") == 0,
    )


def test_read_skill_note_tool() -> None:
    with tempfile.TemporaryDirectory() as d:
        skill_dir = Path(d)
        (skill_dir / "stream-ciphers.md").write_text("LFSR keystream recovery", encoding="utf-8")
        (skill_dir / "classic-ciphers.md").write_text("vigenere etc", encoding="utf-8")
        tool = make_read_skill_note_tool(skill_dir, ["classic-ciphers.md", "stream-ciphers.md"])
        fn = tool.func  # underlying closure, bypassing Injected* for a direct unit call

        # First read loads the file content.
        out1 = fn(note="stream-ciphers.md", state={"messages": []}, tool_call_id="c1")
        check("read loads content", "LFSR keystream recovery" in out1)

        # Anchor form resolves to the same file.
        out_anchor = fn(
            note="stream-ciphers.md#galois", state={"messages": []}, tool_call_id="c2"
        )
        check("read resolves #anchor to file", "LFSR keystream recovery" in out_anchor)

        # Duplicate read (same note already in history) is refused with a pointer.
        hist = [
            read_note_call("stream-ciphers.md", "h1"),
            ToolMessage(content="LFSR keystream recovery", tool_call_id="h1"),
            read_note_call("stream-ciphers.md", "c3"),
        ]
        out_dup = fn(note="stream-ciphers.md", state={"messages": hist}, tool_call_id="c3")
        check("duplicate read returns ALREADY LOADED", "[ALREADY LOADED]" in out_dup)
        check("duplicate read does NOT re-inject body", "LFSR keystream recovery" not in out_dup)

        # A *different* note is still loadable even after another was read.
        out_other = fn(note="classic-ciphers.md", state={"messages": hist}, tool_call_id="c4")
        check("different note still loads", "vigenere etc" in out_other)

        # Unknown note -> not found with the available list.
        out_nf = fn(note="does-not-exist.md", state={"messages": []}, tool_call_id="c5")
        check("missing note reports available", "No such note" in out_nf)


# --- Fix 4: wrong-note steering / re-arm ------------------------------------------

def test_executions_since_note() -> None:
    check("no note read -> None", _executions_since_last_note_read([exec_call("ls", "1")]) is None)
    msgs = [
        exec_call("file chal", "1"),
        read_note_call("classic-ciphers.md", "2"),
        ToolMessage(content="...", tool_call_id="2"),
        exec_call("cat a", "3"),
        exec_call("cat b", "4"),
    ]
    check("counts execs after last note", _executions_since_last_note_read(msgs) == 2)


def test_skill_note_gate() -> None:
    # No note read at all -> initial gate message.
    check("gate blocks with no note", _skill_note_gate([exec_call("ls", "1")]) == _skill_note_gate_msg())

    # Note read, few execs since -> gate passes (None).
    ok = [read_note_call("stream-ciphers.md", "1"), ToolMessage(content="x", tool_call_id="1")]
    check("gate passes right after a note read", _skill_note_gate(ok) is None)

    # Note read, then many execs without solution -> re-arm.
    rearm = [read_note_call("classic-ciphers.md", "1"), ToolMessage(content="x", tool_call_id="1")]
    rearm += [exec_call(f"cmd{i}", f"e{i}") for i in range(_REARM_AFTER_EXECUTIONS)]
    check("gate re-arms after many execs", _skill_note_gate(rearm) == _skill_note_rearm_msg())

    # A fresh (re-)read clears the re-arm even if it's the same note (duplicate guard
    # makes that cheap), because execs-since resets to 0.
    rearm_cleared = rearm + [read_note_call("classic-ciphers.md", "rr")]
    check("re-read clears re-arm", _skill_note_gate(rearm_cleared) is None)


# --- Fix 3: planner write_todos loop-breaker --------------------------------------

def test_todos_signature_and_run() -> None:
    a = todos_call(["recon", "exploit"], "1")
    b = todos_call(["recon", "exploit"], "2")
    c = todos_call(["recon", "decrypt"], "3")
    check("identical todos share signature", _todos_signature(a.tool_calls[0]) == _todos_signature(b.tool_calls[0]))
    check("changed todos differ", _todos_signature(a.tool_calls[0]) != _todos_signature(c.tool_calls[0]))

    # Two identical write_todos turns in a row.
    msgs = [
        a, ToolMessage(content="ok", tool_call_id="1"),
        b, ToolMessage(content="ok", tool_call_id="2"),
    ]
    check("trailing identical run == 2", _trailing_identical_write_todos(msgs) == 2)

    # A changed plan breaks the run (newest turn is unique).
    msgs2 = msgs + [c, ToolMessage(content="ok", tool_call_id="3")]
    check("changed last plan breaks run", _trailing_identical_write_todos(msgs2) == 1)

    # A task delegation breaks the run entirely.
    deleg = msgs + [ai("task", {"subagent_type": "ctf-crypto"}, "t1")]
    check("delegation breaks run", _trailing_identical_write_todos(deleg) == 0)


def test_todos_loop_steer() -> None:
    mw = FlagGateMiddleware()
    a = todos_call(["recon", "exploit"], "1")
    b = todos_call(["recon", "exploit"], "2")

    # One write_todos turn: no steer.
    one = {"messages": [a, ToolMessage(content="ok", tool_call_id="1")]}
    check("single plan does not steer", mw._todos_loop_steer(one) is None)

    # Two identical: steer injected.
    two = {"messages": [a, ToolMessage(content="ok", tool_call_id="1"),
                        b, ToolMessage(content="ok", tool_call_id="2")]}
    res = mw._todos_loop_steer(two)
    check("repeated plan steers", res is not None and isinstance(res["messages"][0], HumanMessage))
    check("steer carries marker", res is not None and "[PLAN LOOP]" in res["messages"][0].content)

    # Bounded: once the cap of markers is already present, stop steering.
    capped = dict(two)
    capped["messages"] = list(two["messages"]) + [
        HumanMessage(content="[PLAN LOOP] nudge") for _ in range(_MAX_TODOS_STEERS)
    ]
    check("steering is bounded", mw._todos_loop_steer(capped) is None)
    check("marker counter works", _count_text_marker(capped["messages"], "[PLAN LOOP]") == _MAX_TODOS_STEERS)


def main() -> int:
    test_normalize()
    test_count_prior_reads()
    test_read_skill_note_tool()
    test_executions_since_note()
    test_skill_note_gate()
    test_todos_signature_and_run()
    test_todos_loop_steer()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
