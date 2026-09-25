"""Standalone tests for Phase 4 skill-section retrieval (bloat source #1).

Covers the pure parsing/ranking/condensing helpers in ``skills_helper`` and the
body-read-aware skill-note gate helpers in ``deep_generative_workflow``. No external
services required.

Run with:

    PYTHONPATH=src python3 tests/test_skill_retrieval.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from application.langgraph.helpers import skills_helper as sh

_passed = 0


def check(name, cond):
    global _passed
    assert cond, f"FAILED: {name}"
    _passed += 1
    print(f"  ok - {name}")


SAMPLE = """\
intro text before any heading
spanning two lines

# Detect CTFd
how to detect it
```
curl /api/v1
```

## Sub of detect
nested content

# Authentication
login flow here
token=abc

# SQL Injection
sqlmap -u url
"""


def test_parse_sections():
    intro, sections = sh._parse_sections(SAMPLE)
    check("intro captured", intro.startswith("intro text"))
    titles = [s["title"] for s in sections]
    check("top-level + sub headings parsed", titles == [
        "Detect CTFd", "Sub of detect", "Authentication", "SQL Injection"
    ])
    detect = sections[0]
    check("section slug", detect["slug"] == "detect-ctfd")
    # 'Detect CTFd' body must include its subsection (next heading is same-level '#').
    check("section includes its subsection", "nested content" in detect["body"])
    check("section stops at next same-level heading", "login flow" not in detect["body"])


def test_parse_no_headings():
    intro, sections = sh._parse_sections("just a flat note, no headings at all")
    check("no-heading: whole text is intro", intro.startswith("just a flat"))
    check("no-heading: zero sections", sections == tuple())


def test_find_section():
    _intro, sections = sh._parse_sections(SAMPLE)
    check("exact slug match", sh._find_section(sections, "authentication")["title"] == "Authentication")
    check("anchor-style match", sh._find_section(sections, "sql-injection")["title"] == "SQL Injection")
    check("title substring match", sh._find_section(sections, "Detect")["slug"] == "detect-ctfd")
    check("no match → None", sh._find_section(sections, "nonexistent-xyz") is None)
    check("empty query → None", sh._find_section(sections, "") is None)


def test_render_outline():
    intro, sections = sh._parse_sections(SAMPLE)
    out = sh._render_outline("x.md", intro, sections)
    check("outline lists section slugs", "detect-ctfd" in out and "authentication" in out)
    check("outline does NOT include section bodies", "sqlmap -u url" not in out)
    check("outline mentions how to load a section", "section=" in out)


def test_condense_index():
    body = (
        "# Web techniques\n"
        "Lots of prose that should be dropped to save context. " * 20 + "\n"
        "- SQL injection: see [sqli.md](sqli.md)\n"
        "More dropped prose here. " * 20 + "\n"
        "## XSS\n"
        "see [xss.md](xss.md) for details\n"
    )
    condensed = sh._condense_index(body, cap=10000)
    check("condense keeps headings", "# Web techniques" in condensed and "## XSS" in condensed)
    check("condense keeps note pointers", "sqli.md" in condensed and "xss.md" in condensed)
    check("condense drops prose", "dropped prose" not in condensed)
    check("condense smaller than original", len(condensed) < len(body))
    # cap is enforced
    capped = sh._condense_index(body, cap=40)
    check("condense respects cap", len(capped) <= 40 + 60)  # + truncation notice
    # fail-open: nothing keepable → empty (caller falls back to full)
    check("condense empty when no headings/links", sh._condense_index("just prose\nmore prose") == "")


def test_embedded_guide_modes(monkeypatch_env):
    skill = sh.CtfSkill(
        name="ctf-web",
        description="d",
        body="# Tech\nprose prose prose\n- thing: [a.md](a.md)\n",
        skill_dir=sh.Path("/tmp/ctf-web"),
        notes=["a.md", "b.md"],
    )
    monkeypatch_env("SKILL_INDEX_MODE", "full")
    label, guide = sh._embedded_guide(skill)
    check("full mode returns whole body", guide == skill.body and "INDEX ONLY" in label)

    monkeypatch_env("SKILL_INDEX_MODE", "header")
    label, guide = sh._embedded_guide(skill)
    check("header mode condenses", "a.md" in guide and "prose prose prose" not in guide)
    check("header mode label", "condensed" in label.lower())

    monkeypatch_env("SKILL_INDEX_MODE", "search")
    label, guide = sh._embedded_guide(skill)
    check("search mode omits index, names search_skill", "search_skill" in guide)
    check("search mode lists notes", "a.md" in guide and "b.md" in guide)

    monkeypatch_env("SKILL_INDEX_MODE", "bogus")
    label, guide = sh._embedded_guide(skill)
    check("unknown mode falls back to full", guide == skill.body)


def test_tool_schemas():
    rsn = sh.make_read_skill_note_tool(sh.Path("/tmp/ctf-web"), ["a.md"])
    check("read_skill_note model args = note+section",
          set(rsn.args.keys()) == {"note", "section"})
    srch = sh.make_search_skill_tool(sh.Path("/tmp/ctf-web"), ["a.md"])
    check("search_skill model arg = query only", set(srch.args.keys()) == {"query"})
    check("search_skill has a useful description", len(srch.description) > 20)


def test_compose_prompt_adapts(monkeypatch_env):
    skill = sh.CtfSkill(
        name="ctf-web", description="d",
        body="# Tech\n[a.md](a.md)\n", skill_dir=sh.Path("/tmp/ctf-web"), notes=["a.md"],
    )
    monkeypatch_env("SKILL_SECTION_RETRIEVAL", "false")
    p = sh.compose_subagent_prompt("BASE", skill)
    check("legacy prompt references read_skill_note('<file>.md')", "read_skill_note('<file>.md')" in p)
    check("legacy prompt has no section step", "section='<slug>'" not in p)

    monkeypatch_env("SKILL_SECTION_RETRIEVAL", "true")
    p = sh.compose_subagent_prompt("BASE", skill)
    check("section prompt teaches outline→section flow", "section='<slug>'" in p)
    check("section prompt mentions search_skill", "search_skill(" in p)
    check("section gate note: outline peek does not count", "outline peek" in p)


# ---- gate helpers (deep_generative_workflow) ------------------------------- #
def _msg(*tool_calls):
    return SimpleNamespace(tool_calls=list(tool_calls))


def _tc(name, **args):
    return {"name": name, "args": args, "id": "x"}


def test_gate_consulting_reads(monkeypatch_env):
    from application.langgraph.models import deep_generative_workflow as dg

    outline_read = _msg(_tc("read_skill_note", note="a.md"))
    section_read = _msg(_tc("read_skill_note", note="a.md", section="sqli"))
    anchor_read = _msg(_tc("read_skill_note", note="a.md#sqli"))

    # Section retrieval OFF: any read counts (legacy).
    monkeypatch_env("SKILL_SECTION_RETRIEVAL", "false")
    check("off: outline read counts", dg._is_consulting_note_read(outline_read.tool_calls[0]))
    check("off: gate satisfied by any read",
          dg._skill_note_gate([outline_read]) is None)

    # Section retrieval ON: only a section/body read counts.
    monkeypatch_env("SKILL_SECTION_RETRIEVAL", "true")
    check("on: outline read does NOT count",
          not dg._is_consulting_note_read(outline_read.tool_calls[0]))
    check("on: section read counts",
          dg._is_consulting_note_read(section_read.tool_calls[0]))
    check("on: anchor read counts",
          dg._is_consulting_note_read(anchor_read.tool_calls[0]))
    check("on: gate still locked after outline-only",
          dg._skill_note_gate([outline_read]) is not None)
    check("on: gate unlocked after a section read",
          dg._skill_note_gate([section_read]) is None)


def test_gate_messages_match_mode(monkeypatch_env):
    """Regression: the steering text must name the action that actually clears the gate.

    Under section retrieval, only a SECTION read unlocks the gate, so the message must
    instruct `section='<slug>'` / `search_skill(...)`. A message that says only
    `read_skill_note('<file>.md')` would send an obedient agent to an outline peek that
    never clears — the unbreakable thrash loop we observed live.
    """
    from application.langgraph.models import deep_generative_workflow as dg

    monkeypatch_env("SKILL_SECTION_RETRIEVAL", "true")
    g, r = dg._skill_note_gate_msg(), dg._skill_note_rearm_msg()
    check("on: initial gate names section= form", "section='<slug>'" in g)
    check("on: initial gate mentions search_skill", "search_skill(" in g)
    check("on: re-arm names a section-loading action",
          "section='<slug>'" in r and "search_skill(" in r)
    check("on: re-arm says outline does not count", "outline peek does NOT count" in r)

    monkeypatch_env("SKILL_SECTION_RETRIEVAL", "false")
    g, r = dg._skill_note_gate_msg(), dg._skill_note_rearm_msg()
    check("off: gate keeps legacy whole-note wording",
          "read_skill_note('<file>.md')" in g and "section=" not in g)
    check("off: re-arm keeps legacy whole-note wording",
          "read_skill_note('<file>.md')" in r and "section=" not in r)


def test_gate_rearm(monkeypatch_env):
    from application.langgraph.models import deep_generative_workflow as dg

    monkeypatch_env("SKILL_SECTION_RETRIEVAL", "true")
    msgs = [_msg(_tc("read_skill_note", note="a.md", section="sqli"))]
    msgs += [_msg(_tc("execute_script", command=f"c{i}")) for i in range(dg._REARM_AFTER_EXECUTIONS)]
    check("re-arm triggers after many execs since consult",
          dg._skill_note_gate(msgs) is not None)
    # a fresh section read clears it
    msgs.append(_msg(_tc("read_skill_note", note="a.md", section="lfi")))
    check("re-arm cleared by re-consult", dg._skill_note_gate(msgs) is None)


def test_real_note_parses():
    """A real shipped note must parse into multiple sections (sanity on live data)."""
    skills = sh.discover_ctf_skills()
    parsed_any = False
    for skill in skills:
        for note in skill.notes:
            path = skill.skill_dir / note
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            _intro, sections = sh._parse_sections(text)
            if len(sections) >= 2:
                parsed_any = True
                break
        if parsed_any:
            break
    check("at least one shipped note parses into >=2 sections", parsed_any)


def _run():
    # tiny env-patch helper that restores afterwards
    saved = {}

    def patch(key, val):
        if key not in saved:
            saved[key] = os.environ.get(key)
        os.environ[key] = val

    try:
        test_parse_sections()
        test_parse_no_headings()
        test_find_section()
        test_render_outline()
        test_condense_index()
        test_embedded_guide_modes(patch)
        test_tool_schemas()
        test_compose_prompt_adapts(patch)
        test_gate_consulting_reads(patch)
        test_gate_messages_match_mode(patch)
        test_gate_rearm(patch)
        test_real_note_parses()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print(f"\nALL PASSED ({_passed} checks)")


if __name__ == "__main__":
    _run()
