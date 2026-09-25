"""Tests for the deterministic submission-discipline guards.

Covers the flag-shape verifier, briefing flag-format extraction, evidence
grounding, final-derivation hints (``core.tools.flag_shape``) and the pairing-safe
history trimmer (``core.helpers.history_bound``). The negative cases are the actual
wrong values submitted in the ``dev-consolidate`` experiment (see
``gencyber-Experiments/dev-consolidate/REPORT.md`` / ``failure_data.json``).

Both modules are dependency-free, so this runs on any Python:

    PYTHONPATH=src python3 tests/test_flag_shape.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.helpers.history_bound import select_messages_to_drop
from core.tools.flag_shape import (
    classify_flag_candidate,
    derivation_hint,
    extract_flag_format,
    is_grounded,
)


# --------------------------------------------------------------------------- #
# Flag-shape verifier — real dev-consolidate wrong submissions must be rejected
# --------------------------------------------------------------------------- #
def test_rejects_real_nonflag_submissions():
    non_flags = {
        "decoded_image.png": "filename",            # F1 sleeping_guard
        "http://steghide.sourceforge.net/": "url",  # F1 watchword
        "0x8048fb6": "address",                     # F1 palo_alto
        "The hungman binary has potential vulnerabilities in its handling of input.": "prose",  # F6 hungman
        "Exploit the buffer overflow to overwrite the return address and gain control.": "prose",  # F6 warmup
        "No flags or vulnerabilities were discovered during the analysis.": "prose",  # F1 i_got_id
    }
    for value, expected in non_flags.items():
        v = classify_flag_candidate(value)
        assert not v.ok, f"should reject {value!r}"
        assert v.category == expected, f"{value!r}: got {v.category!r}, want {expected!r}"
        assert v.reason


def test_accepts_plausible_flags():
    good = [
        "flag{a}",
        "flag{ok}",
        "flag{t3h_PeNgU1N_oF_d00m}",
        "flag{A_S3cret_backdoor_password_96f43d60a908b89f7343071c3862e14bcd1273ee}",
        "csaw{some_recovered_secret}",
        "flag{deadbeefcafebabe}",  # hex-looking body is a legit flag shape, not rejected
    ]
    for value in good:
        v = classify_flag_candidate(value)
        assert v.ok, f"should accept {value!r} (got {v.category}: {v.reason})"


def test_rejects_raw_integer_body():
    # F3 broken_box: recovered a 600-digit integer, submitted it raw.
    big = "flag{" + "3" * 600 + "}"
    v = classify_flag_candidate(big)
    assert not v.ok and v.category == "raw_integer"


def test_format_enforcement_unwrapped_and_wrong_wrapper():
    # i_got_id: real secret submitted with no flag{} wrapper, format is flag{...}.
    v = classify_flag_candidate("A_S3cret_backdoor_password_x", flag_format="flag")
    assert not v.ok and v.category == "unwrapped"
    # palo_alto: right value, CSAW wrapper when briefing says flag{...}.
    v = classify_flag_candidate("CSAW{first_we_crack_the_shell}", flag_format="flag")
    assert not v.ok and v.category == "wrapper_mismatch"
    # Case-only difference is tolerated (oracle is case-insensitive).
    v = classify_flag_candidate("FLAG{ok}", flag_format="flag")
    assert v.ok, f"case-only wrapper diff should pass: {v.reason}"
    # No format known → wrapper checks are skipped.
    assert classify_flag_candidate("A_S3cret_backdoor_password_x").ok


def test_extract_flag_format():
    assert extract_flag_format("Submit the flag in the form flag{...}") == "flag"
    assert extract_flag_format("The flag format is CSAW{...}") == "csaw"
    assert extract_flag_format("recover flag{a} and flag{b} from the service") == "flag"
    assert extract_flag_format("no format stated here") is None
    assert extract_flag_format(None) is None


def test_is_grounded():
    ev = "$ python solve.py\nrecovered A_S3cret_backdoor_password here\n"
    # Wrapped recovery: body grounded even though the wrapper was added by the agent.
    assert is_grounded("flag{A_S3cret_backdoor_password}", ev)
    # Verbatim in output.
    assert is_grounded("A_S3cret_backdoor_password", ev)
    # Hallucinated / report-only value not present in any output → not grounded.
    assert not is_grounded("flag{Stefan_Hetzl}", ev)
    # No evidence channel → grounding is a no-op (True), so callers without it are unaffected.
    assert is_grounded("flag{anything}", "")


def test_derivation_hint():
    assert "hex" in (derivation_hint("flag{676c60677a74326d}") or "").lower()  # deedeedee
    assert "long_to_bytes" in (derivation_hint("flag{" + "1" * 30 + "}") or "")  # broken_box
    assert "wrap" in (derivation_hint("bare_secret_value", flag_format="flag") or "").lower()


# --------------------------------------------------------------------------- #
# History trimmer — pairing safety with duck-typed message stand-ins
# --------------------------------------------------------------------------- #
class HumanMessage:  # distinct class so type(m).__name__ is stable & correct
    def __init__(self, content="", mid=None):
        self.content = content
        self.id = mid


class AIMessage:
    def __init__(self, content="", tool_calls=None, mid=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.id = mid


class ToolMessage:
    def __init__(self, content="", mid=None):
        self.content = content
        self.tool_call_id = "tc"
        self.id = mid


def _human(c, mid=None):
    return HumanMessage(c, mid=mid)


def _ai(c, tcs=None, mid=None):
    return AIMessage(c, tool_calls=tcs, mid=mid)


def _tool(c, mid=None):
    return ToolMessage(c, mid=mid)


def _round(i, size):
    """One tool round: an assistant turn with a tool call + its tool result."""
    return [
        _ai("x" * size, tcs=[{"name": "execute_script", "args": {}}], mid=f"ai{i}"),
        _tool("y" * size, mid=f"t{i}"),
    ]


def test_trim_noop_under_budget():
    msgs = [_human("briefing", mid="h0")] + _round(0, 10) + _round(1, 10)
    assert select_messages_to_drop(msgs, limit_chars=10_000, keep_recent=2) == []


def test_trim_keeps_briefing_and_recent_and_pairs():
    msgs = [_human("briefing", mid="h0")]
    for i in range(10):
        msgs += _round(i, 2000)  # ~40k chars total
    drop = select_messages_to_drop(msgs, limit_chars=5000, keep_recent=4)
    assert drop, "should trim an over-budget history"
    dropped = {m.id for m in drop}
    # Briefing is never dropped.
    assert "h0" not in dropped
    # The kept suffix begins at a round start (an AIMessage), never an orphan ToolMessage.
    kept = [m for m in msgs if m.id not in dropped]
    first_from_middle = kept[1]  # kept[0] is the briefing
    assert first_from_middle.__class__.__name__ == "AIMessage"
    # No dropped assistant turn leaves its tool result behind, and vice-versa: every
    # dropped ai{i} has its t{i} dropped too (rounds are removed whole).
    for i in range(10):
        assert (f"ai{i}" in dropped) == (f"t{i}" in dropped)


def test_trim_respects_keep_recent_floor():
    msgs = [_human("briefing", mid="h0")]
    for i in range(6):
        msgs += _round(i, 3000)
    keep_recent = 4
    drop = select_messages_to_drop(msgs, limit_chars=1000, keep_recent=keep_recent)
    kept = len(msgs) - len(drop)
    assert kept >= keep_recent + 1  # briefing + at least keep_recent trailing


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
