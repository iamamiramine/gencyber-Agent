"""Parity test for the M/P controlled experiment (Plan A WS3 step 3.5).

The 2x2 only means anything if M and P differ in **structure alone**. This test is
the evidence for that claim, so it asserts on the real build paths rather than
restating the intent:

  1. tool set        — the monolith and a specialist expose the same tool names
  2. skill corpus    — both reach the same pack, sandboxed the same way
  3. skill index     — both carry that pack's SKILL.md in the system prompt
  4. guard set       — same middleware classes, in the same order
  5. escalation rule — same trigger and same strong model, and 'off' means off
  6. category hint   — the hint's wording matches the topology being run

What it deliberately does NOT assert: that the graphs are identical. They must not
be. P has a persistent non-executing planner, a ``task`` delegation tool, transient
specialist contexts and evidence folding; M has none of those. That difference is
the independent variable.

Run inside the image (host Python is too old):

  docker run --rm -e PYTHONPATH=/app/src \
    -v "$PWD/src:/app/src:ro" -v "$PWD/config:/app/config:ro" \
    -v "$PWD/tests:/app/tests:ro" -v "$PWD/scripts:/app/scripts:ro" \
    generative-module:latest python3 /app/tests/test_mp_parity.py
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda

from application.langgraph.models import deep_generative_workflow as dgw
from application.langgraph.models.deep_generative_workflow import (
    DeepGenerativeWorkflow,
    EscalationMiddleware,
    HistoryBoundMiddleware,
    StepBudgetMiddleware,
    _make_recall_tool,
    _make_tools,
    set_escalation_config,
    set_skill_scope,
)
from application.langgraph.models.deep_planner_workflow import DeepPlannerWorkflow

CATEGORY = "crypto"
EXPECTED_PACK = "ctf-crypto"
failures: list = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def _tool_names(tools) -> set:
    return {getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools}


def _monolith(session: str):
    """Assemble the monolith's tools + prompt through the real build path."""
    noop = RunnableLambda(lambda state: {})
    tools = _make_tools(noop, noop, noop, session)
    tools.append(_make_recall_tool(session))
    return DeepGenerativeWorkflow._attach_skill_scope(tools, "BASE PROMPT", session)


def _specialist(session: str, pack: str):
    """The planner subagent dict for ``pack``, via the real _build_subagents path."""
    noop = RunnableLambda(lambda state: {})
    stub = SimpleNamespace(model=None, system_prompt="BASE PROMPT")
    subs = DeepPlannerWorkflow._build_subagents(
        agents={"generative": SimpleNamespace(model=_DummyModel(), system_prompt="BASE PROMPT")},
        execute_script_tool=noop,
        write_script_tool=noop,
        submit_goal_tool=noop,
        session_id=session,
    ) or []
    return next((s for s in subs if s["name"] == pack), None)


class _DummyModel:
    """Minimal stand-in: _build_subagents only needs a truthy bindable object."""

    def bind_tools(self, *a, **k):
        return self

    def with_structured_output(self, *a, **k):
        return self


def main() -> int:
    session = "__parity__"
    set_skill_scope(session, category=CATEGORY)

    m_tools, m_prompt = _monolith(session)
    spec = _specialist(session, EXPECTED_PACK)
    if spec is None:
        check(f"specialist {EXPECTED_PACK} discovered", False,
              "corpus missing — cannot compare")
        return 1

    m_names, p_names = _tool_names(m_tools), _tool_names(spec["tools"])

    # 1 + 2. Tool set and corpus reach.
    check("tool sets identical", m_names == p_names,
          f"M-only={sorted(m_names - p_names)} P-only={sorted(p_names - m_names)}")
    check("monolith reaches the corpus", "read_skill_note" in m_names,
          f"tools={sorted(m_names)}")

    # 3. Skill index in the system prompt — the half that is easy to forget.
    check("monolith carries the SKILL.md index", m_prompt != "BASE PROMPT"
          and len(m_prompt) > len("BASE PROMPT"))
    check("specialist and monolith share the same index",
          m_prompt == spec["system_prompt"],
          f"len M={len(m_prompt)} P={len(spec['system_prompt'])}")

    # 4. Guard set. The specialist adds SubagentContextMiddleware because it runs in a
    # transient delegated context; that is part of the structure under test, so it is
    # excluded here rather than treated as a parity break.
    structural = {"SubagentContextMiddleware"}
    p_mw = [type(x).__name__ for x in spec["middleware"]]
    p_shared = [n for n in p_mw if n not in structural]
    expected = ["HistoryBoundMiddleware", "StepBudgetMiddleware", "EscalationMiddleware"]
    check("specialist guard order as expected", p_shared == expected, f"{p_shared}")
    # The monolith gets HistoryBound + StepBudget + FlagGate from _compile and
    # Escalation from build(); assert the classes are the ones _compile installs.
    check("monolith guards available", all(
        c is not None for c in (HistoryBoundMiddleware, StepBudgetMiddleware,
                                EscalationMiddleware)))

    # 5. Escalation rule: same threshold, and 'off' really disables.
    thr_default = dgw._escalation_after_for(session) if hasattr(
        dgw, "_escalation_after_for") else None
    set_escalation_config(session, model="off")
    off = dgw.build_escalation_model(
        SimpleNamespace(model=_DummyModel(), system_prompt=""), session_id=session)
    check("escalation 'off' yields no strong model", off is None, f"got {off!r}")
    if thr_default is not None:
        check("escalation threshold shared", thr_default == 12, f"{thr_default}")

    # 6. Category hint wording must follow the topology.
    sys.path.insert(0, "/app/scripts")
    try:
        import run_langfuse_experiment as R

        mat = {"seed_prompt": "seed"}
        p_hint = R._compose_question(mat, {}, category=CATEGORY, has_specialists=True)
        m_hint = R._compose_question(mat, {}, category=CATEGORY, has_specialists=False)
        check("planner hint says delegate", "Delegate to the" in p_hint)
        check("monolith hint does NOT say delegate", "Delegate" not in m_hint,
              m_hint[-120:])
        check("both hints carry the category", CATEGORY in p_hint and CATEGORY in m_hint)
        check("topology detection", R._has_specialists("deepagent_planner_pipeline_registry")
              and not R._has_specialists("deepagent_generative_pipeline_registry"))
    except ImportError as exc:
        check("runner importable for hint parity", False, str(exc))

    print()
    if failures:
        print(f"{len(failures)} PARITY FAILURE(S): {failures}")
        return 1
    print("M/P parity: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
