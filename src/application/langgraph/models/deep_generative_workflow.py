"""DeepAgents-backed generative workflow.

A drop-in alternative to :class:`WorkflowGraph` (the baseline structured-output
ReAct graph). Instead of routing on which structured-output field the agent set,
this workflow runs the LangChain *DeepAgents* native tool-calling loop with three
typed tools that reuse the EXACT same workbench-backed implementations the baseline
uses (passed in by the service as state-callables):

  - ``execute_script(command)``  → run one shell command in the workbench PTY
  - ``write_script(content, language)`` → save a script to the shared volume
  - ``submit_goal(flag)`` → validate a recovered flag

Routing parity with the baseline is enforced by :class:`FlagGateMiddleware`: the
graph never ends until a submission is accepted; a model turn with no tool calls
loops back (the ``recursion_limit`` is the only hard ceiling), and acceptance ends
the run immediately. State keys ``generative_agent_response`` / ``submitted_goal`` /
``submission_verified`` are surfaced so the existing service / API response and the
NDJSON stream work unchanged.

The default DeepAgents virtual filesystem + sandbox ``execute`` tools and the
general-purpose subagent are dropped via a registered ``HarnessProfile`` — this
agent talks to the REAL workbench, not a virtual FS.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Annotated, Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.graph import DeepAgentState
from langchain.agents.middleware.types import AgentMiddleware, hook_config

from application.langgraph.helpers import memory_fold
from application.langgraph.helpers.skills_helper import (
    compose_subagent_prompt,
    discover_ctf_skills,
    make_read_skill_note_tool,
    make_search_skill_tool,
    section_retrieval_enabled,
)
from core.helpers.history_bound import select_messages_to_drop
from core.tools.flag_shape import (
    classify_flag_candidate,
    derivation_hint,
    extract_flag_format,
    is_grounded,
)
from infrastructure.observability import langfuse_tracer
from infrastructure.repository import agent_memory_repository as agent_memory
from infrastructure.repository.mongodb_repository import get_mongodb_client

logger = logging.getLogger(__name__)

_DEFAULT_RECURSION_LIMIT = int(os.getenv("GENCYBER_RECURSION_LIMIT", "300"))

# Global per-challenge action budget. ``recursion_limit`` only bounds a SINGLE graph
# invocation; in the planner topology the planner re-delegates via ``task`` and each
# specialist subagent runs its own ReAct loop with its own recursion budget, so the
# real per-challenge work is (planner delegations) × (per-specialist steps) — which is
# effectively unbounded. This budget is a single counter shared by the planner and
# every specialist (keyed by ``session_id``); once the total number of concrete actions
# (execute_script / write_script / submit_goal) reaches it, the tools stop doing real
# work and ``StepBudgetMiddleware`` ends the run deterministically. 0 (default) disables
# the cap, preserving legacy behavior when the env var is unset.
_STEP_BUDGET = int(os.getenv("GENCYBER_STEP_BUDGET", "0"))
_step_counts: Dict[str, int] = {}


def _reset_step_budget(session_id: str) -> None:
    """Zero the per-challenge action counter at the start of a fresh run."""
    _step_counts[session_id] = 0


def _bump_step_budget(session_id: str) -> int:
    """Record one concrete action and return the new running total for this run."""
    _step_counts[session_id] = _step_counts.get(session_id, 0) + 1
    return _step_counts[session_id]


def _steps_used(session_id: str) -> int:
    return _step_counts.get(session_id, 0)


def _step_budget_exhausted(session_id: str) -> bool:
    """True once the run has spent its whole action budget (no-op when disabled)."""
    return _STEP_BUDGET > 0 and _steps_used(session_id) >= _STEP_BUDGET


_STEP_BUDGET_TOOL_MSG = (
    "[STEP BUDGET EXHAUSTED] This run has used its entire per-challenge action budget "
    f"of {_STEP_BUDGET} actions. No further commands, scripts, or submissions will be "
    "executed. Stop working and end now."
)

# Flags already submitted-and-rejected this run, per session_id (shared by the
# planner and every specialist). This is the submission analogue of the
# execute_script duplicate guard: a weak model otherwise re-fires an IDENTICAL
# rejected flag many times (28x observed on one run), burning the whole budget.
# Re-submitting a value already known to be wrong can never succeed, so we
# short-circuit it and steer the model to recover a DIFFERENT value.
_rejected_flags: Dict[str, set] = {}

# Per-run diagnostics for the eval harness, keyed by session_id. These make the
# Langfuse scores trustworthy: the old ``attempted_submission`` metric read
# ``submitted_goal``, which submit_goal nulls on rejection (and the _prefer_new
# reducer preserves the null), so it could only ever be 1 on an ACCEPTED flag —
# it silently equalled ``solved``. We now count real attempts directly.
_submit_attempts: Dict[str, int] = {}          # total submissions that reached validation
_submitted_flags: Dict[str, set] = {}          # distinct values submitted (accepted or not)
_placeholder_submits: Dict[str, int] = {}      # submissions rejected as placeholder/template
_shape_rejects: Dict[str, int] = {}            # submissions blocked by the flag-shape guard
_ungrounded_rejects: Dict[str, int] = {}       # submissions blocked as not-in-tool-output
_delegations: Dict[str, List[str]] = {}        # full planner delegation sequence (specialists)
_escalated: Dict[str, bool] = {}               # did the run escalate to the strong model
# Flag format ("flag", "csaw", …) extracted once per run from the briefing, so the
# shape guard can enforce the correct wrapper without re-parsing on every submit.
_flag_formats: Dict[str, Optional[str]] = {}

# Distinct-guess cap (consolidated-experiment F4): compute/puzzle challenges spray
# dozens of format variants of one answer instead of computing once and verifying
# (coinslot 29 distinct, regexpire 14 distinct / 113 submit calls). Once this many
# DISTINCT values have been submitted and rejected this run, block further brand-new
# guesses and steer the model to verify a candidate against the challenge's own
# checker/service before submitting. 0 disables the cap.
_MAX_DISTINCT_FLAGS = int(os.getenv("GENCYBER_MAX_DISTINCT_FLAGS", "10"))


# ---- skill scope (resource-matched monolith, M0/M1) -----------------------
# The planner reaches the corpus through its specialists: each one is built over a
# single ``ctf-*`` pack, so a specialist sees exactly one pack's SKILL.md index and
# can lazily open only that pack's notes. A monolith has no specialists, so to be
# resource-MATCHED rather than merely resource-comparable it must get the same
# access: one pack, chosen by the challenge's category hint.
#
# Category-scoped (not all-category) is deliberate. All-category would hand the
# monolith strictly more corpus than any single specialist ever sees, which makes a
# planner win unfalsifiable ("your control had more information and still lost") and
# a planner loss uninterpretable. Scoping to the hinted category means M and P differ
# only in structure: persistent non-executing planner, delegation, transient
# specialist contexts and evidence folding.
_skill_scope: Dict[str, str] = {}              # session_id -> challenge category

# Category hint as the benchmark reports it -> skill pack directory name. The NYU
# dataset uses the short forms on the left; the corpus uses the ``ctf-*`` names on
# the right. Unmapped categories leave the monolith with no pack, which is logged
# rather than raised: a challenge in an uncovered category must still run, it just
# has no matched corpus to withhold or grant.
_CATEGORY_TO_PACK: Dict[str, str] = {
    "crypto": "ctf-crypto",
    "cry": "ctf-crypto",
    "rev": "ctf-reverse",
    "reverse": "ctf-reverse",
    "pwn": "ctf-pwn",
    "misc": "ctf-misc",
    "msc": "ctf-misc",
    "web": "ctf-web",
    "forensics": "ctf-forensics",
    "for": "ctf-forensics",
    "osint": "ctf-osint",
    "malware": "ctf-malware",
    "ai-ml": "ctf-ai-ml",
    "ai_ml": "ctf-ai-ml",
}


def set_skill_scope(session_id: Optional[str], *, category: Optional[str] = None) -> None:
    """Record the challenge category for ``session_id`` (no-op on blank id/category).

    Called from ``LangGraphService.init_workflow`` alongside
    :func:`set_escalation_config`. Only the monolith build reads it; the planner
    ignores it, because its specialists are already one-pack-per-agent.
    """
    if not session_id or category is None:
        return
    cat = str(category).strip().lower()
    if cat:
        _skill_scope[session_id] = cat


def _pack_for_session(session_id: Optional[str]) -> Optional[str]:
    return _CATEGORY_TO_PACK.get(_skill_scope.get(session_id or "", ""))


def _reset_submission_tracking(session_id: str) -> None:
    """Clear all per-run submission/diagnostic tracking at the start of a fresh run."""
    _rejected_flags.pop(session_id, None)
    _submit_attempts.pop(session_id, None)
    _submitted_flags.pop(session_id, None)
    _placeholder_submits.pop(session_id, None)
    _shape_rejects.pop(session_id, None)
    _ungrounded_rejects.pop(session_id, None)
    _delegations.pop(session_id, None)
    _escalated.pop(session_id, None)
    _flag_formats.pop(session_id, None)


def _record_submission_attempt(session_id: str, flag: str) -> None:
    """Count one submission attempt + its distinct value (shared by every reject path
    and the oracle path) so the eval diagnostics see every real attempt, not only the
    ones that reached the oracle."""
    _submit_attempts[session_id] = _submit_attempts.get(session_id, 0) + 1
    _submitted_flags.setdefault(session_id, set()).add(_normalize_flag(flag))


def _flag_format_for(session_id: str, query: Optional[str]) -> Optional[str]:
    """The briefing's flag wrapper for this run (cached; parsed once from the query)."""
    if session_id not in _flag_formats:
        try:
            _flag_formats[session_id] = extract_flag_format(query)
        except Exception:  # pragma: no cover - defensive
            _flag_formats[session_id] = None
    return _flag_formats[session_id]


def run_diagnostics(session_id: str) -> Dict[str, Any]:
    """Snapshot the per-run diagnostics for the eval harness (see :func:`_extract_output`)."""
    return {
        "submission_attempts": _submit_attempts.get(session_id, 0),
        "distinct_flags_tried": len(_submitted_flags.get(session_id, set())),
        "placeholder_submissions": _placeholder_submits.get(session_id, 0),
        "shape_rejected_submissions": _shape_rejects.get(session_id, 0),
        "ungrounded_submissions": _ungrounded_rejects.get(session_id, 0),
        "delegations": list(_delegations.get(session_id, [])),
        "escalated": bool(_escalated.get(session_id, False)),
    }


def _normalize_flag(flag: str) -> str:
    """Whitespace-insensitive key for comparing two submitted flag strings."""
    return " ".join(str(flag or "").split())


def _flag_already_rejected(session_id: str, flag: str) -> bool:
    return _normalize_flag(flag) in _rejected_flags.get(session_id, set())


def _record_rejected_flag(session_id: str, flag: str) -> None:
    _rejected_flags.setdefault(session_id, set()).add(_normalize_flag(flag))


def _distinct_rejected_count(session_id: str) -> int:
    return len(_rejected_flags.get(session_id, set()))

# Cap any single tool observation / accumulated evidence fed back to the model so a
# huge dump cannot blow up the context window (mirrors LLM_SCRIPT_OUTPUT_MAX_CHARS).
_MAX_EVIDENCE_CHARS = int(os.getenv("LLM_SCRIPT_OUTPUT_MAX_CHARS", "12000"))

# Master switch for the externalized memory subsystem (function-split stores +
# cheap-LLM fold + fold-card injection). When OFF, the legacy flat
# ``execution_evidence`` accumulation + injection path runs unchanged, so the two
# behaviours can be A/B'd against the Langfuse NYU CTF datasets.
# See docs/agent-memory-design.md.
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").strip().lower() not in (
    "0",
    "false",
    "no",
)
# How many recent fold cards to inject into a freshly-spawned specialist.
_FOLD_INJECT_N = int(os.getenv("MEMORY_FOLD_INJECT_N", "6"))
# Top-k prior observations returned by the recall_evidence pull tool (Piece 3).
_RECALL_K = int(os.getenv("MEMORY_RECALL_K", "5"))

# --- Model escalation ladder ------------------------------------------------
# Specialists run on the cheap recon model by default; when one stalls (spends a
# chunk of the action budget with no accepted flag) or produces a bad submission
# (placeholder / rejected), EscalationMiddleware swaps in a stronger model for the
# rest of that specialist's run. The weak model is fine for recon and note-reading;
# the strong model is spent only where the cheap one is stuck — where actually
# applying an exploit/decryption technique is the blocker.
#   GENCYBER_ESCALATION_MODEL        strong model id (empty/"none"/"off" disables)
#   GENCYBER_ESCALATION_AFTER_ACTIONS action count that trips a stall escalation
_ESCALATION_MODEL = os.getenv("GENCYBER_ESCALATION_MODEL", "openai/gpt-5-mini").strip()
_ESCALATION_AFTER_ACTIONS = int(os.getenv("GENCYBER_ESCALATION_AFTER_ACTIONS", "12"))

# Per-run escalation overrides, keyed by session_id. Set from the init_workflow
# request (see LangGraphService.init_workflow → set_escalation_config) so the
# escalation target is *chosen per run by the operator*, not by the agent, and never
# baked into the container image. Absent keys fall back to the env defaults above, so
# existing runs behave identically. Fields: model, after_actions, provider, base_url.
_escalation_overrides: Dict[str, Dict[str, Any]] = {}


def set_escalation_config(
    session_id: Optional[str],
    *,
    model: Optional[str] = None,
    after_actions: Optional[int] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Record the per-run escalation target for ``session_id`` (no-op on blank id).

    Only non-empty values override; everything else keeps the env default. Passing
    ``model="off"`` (or none/0/false) disables escalation for that run only.
    """
    if not session_id:
        return
    ov = _escalation_overrides.setdefault(session_id, {})
    if model is not None and str(model).strip():
        ov["model"] = str(model).strip()
    if after_actions is not None:
        try:
            ov["after_actions"] = int(after_actions)
        except (TypeError, ValueError):
            pass
    if provider is not None and str(provider).strip():
        ov["provider"] = str(provider).strip()
    if base_url is not None and str(base_url).strip():
        ov["base_url"] = str(base_url).strip()


def _escalation_model_for(session_id: Optional[str]) -> str:
    ov = _escalation_overrides.get(session_id or "", {})
    return str(ov.get("model") or _ESCALATION_MODEL or "").strip()


def _escalation_after_actions_for(session_id: Optional[str]) -> int:
    ov = _escalation_overrides.get(session_id or "", {})
    val = ov.get("after_actions")
    return int(val) if val is not None else _ESCALATION_AFTER_ACTIONS


def _escalation_enabled_for(session_id: Optional[str]) -> bool:
    model = _escalation_model_for(session_id)
    return bool(model) and model.lower() not in ("none", "off", "0", "false")


def _escalation_enabled() -> bool:
    return bool(_ESCALATION_MODEL) and _ESCALATION_MODEL.lower() not in ("none", "off", "0", "false")


def build_escalation_model(agent: Any, session_id: Optional[str] = None) -> Optional[Any]:
    """Build the strong escalation model from the worker agent's own params, with the
    model name overridden to the per-run target (``set_escalation_config``) or the
    ``GENCYBER_ESCALATION_MODEL`` env default. Returns None (escalation disabled) when
    unset/off or on any failure — the run then behaves exactly as before.

    The escalation model inherits the worker's provider/base_url by default so a
    fully-local run escalates local→local; a run may instead point escalation at a
    different provider (e.g. cheap local base → strong cloud model) via the
    ``escalation_provider`` / ``escalation_base_url`` init fields.
    """
    if not _escalation_enabled_for(session_id):
        return None
    strong_name = _escalation_model_for(session_id)
    override = _escalation_overrides.get(session_id or "", {})
    try:
        from application.langgraph.helpers.langraph_helpers import create_llm

        base_params = getattr(agent, "model_params", None)
        pipeline_params = getattr(agent, "pipeline_params", None)
        raw = dict(getattr(agent, "model_config_raw", {}) or {})
        if base_params is None or pipeline_params is None:
            return None
        try:
            strong_params = base_params.model_copy(
                update={"model_name": strong_name, "model_path": strong_name}
            )
        except Exception:
            import copy

            strong_params = copy.copy(base_params)
            setattr(strong_params, "model_name", strong_name)
            setattr(strong_params, "model_path", strong_name)
        raw["model_name"] = strong_name
        raw["model_path"] = strong_name
        # Optional per-run provider/endpoint for the strong model (else inherit base).
        if override.get("provider"):
            raw["provider"] = override["provider"]
        if override.get("base_url"):
            raw["base_url"] = override["base_url"]
        model = create_llm(
            model_params=strong_params,
            pipeline_params=pipeline_params,
            model_config_raw=raw,
        )
        logger.info("escalation model ready: %s", strong_name)
        return model
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("escalation model build failed (%s); escalation disabled", e)
        return None

# DeepAgents builtins we drop: the virtual filesystem tools and the sandbox
# ``execute`` (which only works with a SandboxBackend). We bring our own tools that
# hit the real workbench PTY.
_EXCLUDED_BUILTIN_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}
)

# Profile that drops the builtin FS/sandbox tools and disables the general-purpose
# subagent. Registration is global, additive and idempotent; we register under the
# provider keys our models resolve to.
_HARNESS_PROFILE = HarnessProfile(
    excluded_tools=_EXCLUDED_BUILTIN_TOOLS,
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
)
_PROFILE_KEYS = ("openai", "ollama")
_profiles_registered = False

_NUDGE = (
    "You did not call a tool and the goal is not yet recovered and accepted. Do not "
    "stop. Take the next concrete step: run a command with execute_script, save a "
    "script with write_script, or, if you already recovered the value from real tool "
    "output, call submit_goal."
)


def _register_profiles() -> None:
    """Register the tool-exclusion profile (idempotent)."""
    global _profiles_registered
    if _profiles_registered:
        return
    for key in _PROFILE_KEYS:
        try:
            register_harness_profile(key, _HARNESS_PROFILE)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("register_harness_profile(%s) failed: %s", key, e)
    _profiles_registered = True


def _prefer_new(old: Any, new: Any) -> Any:
    """Reducer for every baseline channel below.

    The native tool-calling loop — and parallel ``task`` delegations — can emit
    several tool calls in a SINGLE model turn, whose ``Command`` updates all land in
    the same LangGraph super-step. A plain channel (``LastValue``) rejects more than
    one write per step (``InvalidUpdateError: can receive only one value per step``),
    so each custom channel needs a reducer. Semantics: the newest meaningful value
    wins, but a ``None``/empty write never erases an existing value — so an idle
    sibling tool or subagent that returns nothing can't wipe what another just
    recovered. ``False`` is a meaningful value (a rejected submission), so it still
    registers.
    """
    return new if new not in (None, "") else old


class DeepGenerativeState(DeepAgentState):
    """DeepAgents state + the baseline channels the service/API/stream expect.

    Every custom channel carries the :func:`_prefer_new` reducer (declared via a
    top-level ``Annotated`` so LangGraph actually picks it up) — otherwise concurrent
    tool calls / parallel subagent delegations in one super-step raise
    ``InvalidUpdateError``.
    """

    submitted_goal: Annotated[Optional[str], _prefer_new]
    submission_verified: Annotated[Optional[bool], _prefer_new]
    submission_rejection_reason: Annotated[Optional[str], _prefer_new]
    script_output: Annotated[Optional[str], _prefer_new]
    execution_evidence: Annotated[Optional[str], _prefer_new]
    generative_agent_response: Annotated[Optional[str], _prefer_new]
    command: Annotated[Optional[str], _prefer_new]
    write_script: Annotated[Optional[str], _prefer_new]
    query: Annotated[Optional[str], _prefer_new]
    session_id: Annotated[Optional[str], _prefer_new]
    # Last specialist the planner delegated to (and the task it handed off), so the
    # orchestration decision is observable in the API/stream — the DeepAgents
    # ``task`` tool is otherwise silent.
    active_subagent: Annotated[Optional[str], _prefer_new]
    active_subagent_task: Annotated[Optional[str], _prefer_new]


def _cap(text: str) -> str:
    if len(text) <= _MAX_EVIDENCE_CHARS:
        return text
    return text[:_MAX_EVIDENCE_CHARS] + "\n…(truncated)"


def _append_evidence(prev: str, new: str) -> str:
    combined = (prev + "\n" + new) if prev else new
    if len(combined) > _MAX_EVIDENCE_CHARS:
        combined = combined[-_MAX_EVIDENCE_CHARS:]
    return combined


# A weak model can fall into a loop, re-issuing the SAME read-only command many times
# instead of acting on output it already has (the "soliloquizing" failure EnIGMA
# describes). Allow each distinct command to actually execute at most this many times
# within one agent loop; further identical calls are short-circuited with a corrective
# message instead of being re-run.
_MAX_IDENTICAL_RUNS = 2


def _normalize_command(command: str) -> str:
    """Whitespace-insensitive key for comparing two command strings."""
    return " ".join(str(command or "").split())


def _count_prior_executions(
    messages: List[Any], command_norm: str, current_id: Optional[str]
) -> int:
    """How many times ``command_norm`` was already requested via ``execute_script``
    earlier in THIS agent loop, excluding the current tool call.

    Scans the agent's own message history (``InjectedState``), so the count is scoped
    to the current subagent run — a fresh ``task`` delegation starts clean.
    """
    n = 0
    for m in messages or []:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") != "execute_script":
                continue
            if current_id is not None and tc.get("id") == current_id:
                continue
            prior = (tc.get("args") or {}).get("command")
            if prior and _normalize_command(prior) == command_norm:
                n += 1
    return n


# A weak model also re-saves the SAME script body many times (42 identical writes
# observed on one run) instead of running the path it already saved or fixing the
# script. Allow each distinct body to be written at most this many times; further
# identical writes are short-circuited with a corrective message.
_MAX_IDENTICAL_WRITES = 2


def _content_key(content: str) -> str:
    """Whitespace-insensitive key for comparing two script bodies."""
    return " ".join(str(content or "").split())


def _count_prior_writes(
    messages: List[Any], content: str, current_id: Optional[str]
) -> int:
    """How many times this exact script body was already saved via ``write_script``
    earlier in THIS agent loop (excludes the current call). Scoped to the current
    subagent run via ``InjectedState`` messages, like :func:`_count_prior_executions`.
    """
    key = _content_key(content)
    n = 0
    for m in messages or []:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") != "write_script":
                continue
            if current_id is not None and tc.get("id") == current_id:
                continue
            prior = (tc.get("args") or {}).get("content")
            if prior and _content_key(prior) == key:
                n += 1
    return n


def _count_tool_calls(messages: List[Any], tool_name: str) -> int:
    """How many times ``tool_name`` was called earlier in THIS agent loop.

    Scoped to the current subagent run (``InjectedState`` messages), so a fresh
    ``task`` delegation starts the count at zero. Used by the skill-note gate to
    check whether a specialist has consulted at least one deep note before it is
    allowed to write a solution script or submit a flag.
    """
    n = 0
    for m in messages or []:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") == tool_name:
                n += 1
    return n


# Message returned when a specialist tries to write a solution / submit before it
# has consulted any skill note. Mirrors the loop-breaker: the action is refused and
# the model is steered to the missing prerequisite instead of being executed blind.
#
# IMPORTANT (Phase 4): when section retrieval is on, a section/body read — NOT an
# outline peek — is what satisfies the gate (see ``_is_consulting_note_read``). The
# steering text MUST therefore name the section form, or an obedient agent that calls
# ``read_skill_note('x.md')`` gets only the outline and the gate never clears (an
# unbreakable thrash loop). The two messages below branch on the retrieval mode so the
# instructed action is always the one that actually unlocks the gate.
_SKILL_NOTE_GATE_MSG_LEGACY = (
    "[SKILL NOTE REQUIRED] You have not read any skill note yet this run. The skill "
    "guide in your instructions is only a one-line INDEX — the actual working "
    "code/method for your technique lives in a deep note. Before you write a solution "
    "script or submit a flag, call `read_skill_note('<file>.md')` for the note that "
    "matches this challenge's technique (the notes available to you are listed in your "
    "<CtfSpecialization> instructions, and `[name.md](name.md)` references in the guide "
    "point at them). Read the closest-matching note now, then retry."
)

_SKILL_NOTE_GATE_MSG_SECTION = (
    "[SKILL NOTE REQUIRED] You have not loaded any skill-note SECTION yet this run. "
    "Your instructions carry only a condensed INDEX; the working code lives in a deep "
    "note's sections. Before you write a solution or submit a flag you must load a "
    "section body — an outline peek does NOT count. Either call "
    "`search_skill('<technique you now see>')` to locate the right section, or call "
    "`read_skill_note('<file>.md', section='<slug>')` (get slugs from the outline via "
    "`read_skill_note('<file>.md')` first). Load the matching section now, then retry."
)


def _skill_note_gate_msg() -> str:
    return (
        _SKILL_NOTE_GATE_MSG_SECTION
        if section_retrieval_enabled()
        else _SKILL_NOTE_GATE_MSG_LEGACY
    )

# Re-arm threshold: even after a note was read, if this many execute_script commands
# have run SINCE the last read_skill_note without a solution being written, the early
# pick probably didn't match the challenge — re-gate write_script/submit so the
# specialist re-consults a (possibly different) note instead of forcing a wrong
# technique. Re-reading the SAME note is cheap (the duplicate guard returns a pointer),
# so the correct path satisfies this with one cheap call.
_REARM_AFTER_EXECUTIONS = 6

_SKILL_NOTE_REARM_MSG_LEGACY = (
    "[RE-CONSULT SKILL NOTE] You read a skill note earlier, but have since run many "
    "commands without writing a solution — a sign the note you picked may not match "
    "this challenge's real technique. Look at what your recon revealed (file types, "
    "source, magic bytes, headers) and call `read_skill_note('<file>.md')` for the note "
    "that matches the SPECIFIC technique you now see (it is fine to read a different "
    "note). Then retry."
)

_SKILL_NOTE_REARM_MSG_SECTION = (
    "[RE-CONSULT SKILL NOTE] You loaded a section earlier, but have since run many "
    "commands without writing a solution — a sign it did not match this challenge's "
    "real technique. Do NOT retry the same action. Look at what your recon revealed "
    "(file types, source, magic bytes, headers) and load a DIFFERENT section body: call "
    "`search_skill('<technique you now see>')` or "
    "`read_skill_note('<file>.md', section='<slug>')`. An outline peek does NOT count — "
    "you must load an actual section. Then retry."
)


def _skill_note_rearm_msg() -> str:
    return (
        _SKILL_NOTE_REARM_MSG_SECTION
        if section_retrieval_enabled()
        else _SKILL_NOTE_REARM_MSG_LEGACY
    )


def _is_consulting_note_read(tc: Dict[str, Any]) -> bool:
    """Whether a ``read_skill_note`` tool call actually *consulted* a note's content.

    With Phase 4 section retrieval on, a ``read_skill_note`` call WITHOUT a ``section``
    only returns the note's outline (a table of contents) — that is navigation, not
    consulting the working code, so it must NOT satisfy the gate. A call WITH a
    ``section`` (including an ``name.md#anchor`` on the filename) loads a section body
    and DOES satisfy it. When section retrieval is off, every read loads the whole note
    and counts — preserving the legacy behavior exactly.
    """
    if tc.get("name") != "read_skill_note":
        return False
    if not section_retrieval_enabled():
        return True
    args = tc.get("args") or {}
    if (args.get("section") or "").strip():
        return True
    note = args.get("note") or ""
    return "#" in note  # section encoded as a filename anchor


def _count_consulting_note_reads(messages: List[Any]) -> int:
    """How many gate-satisfying note reads happened this run (see _is_consulting_note_read)."""
    n = 0
    for m in messages or []:
        for tc in getattr(m, "tool_calls", None) or []:
            if _is_consulting_note_read(tc):
                n += 1
    return n


def _executions_since_last_note_read(messages: List[Any]) -> Optional[int]:
    """``execute_script`` calls issued AFTER the most recent CONSULTING ``read_skill_note``.

    Returns ``None`` if no note was consulted this run. Drives the gate re-arm: a
    specialist that read a note but then ran many recon commands without committing a
    solution likely picked the wrong note and must re-consult before writing/submitting.
    """
    last_note_idx = None
    for i, m in enumerate(messages or []):
        for tc in getattr(m, "tool_calls", None) or []:
            if _is_consulting_note_read(tc):
                last_note_idx = i
    if last_note_idx is None:
        return None
    n = 0
    for m in (messages or [])[last_note_idx + 1:]:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") == "execute_script":
                n += 1
    return n


def _skill_note_gate(messages: List[Any]) -> Optional[str]:
    """Steering message if a specialist must (re)consult a skill note before it may
    write a solution or submit, else ``None``.

    Two gates: (1) no note read at all this run, and (2) a note was read but many
    execute_script commands have run since with no solution written (likely the wrong
    note — re-arm). Both are satisfied by a (re-)read of the matching note. With section
    retrieval on, only a section/body read counts (an outline peek does not unlock).
    """
    if _count_consulting_note_reads(messages) == 0:
        return _skill_note_gate_msg()
    execs_since = _executions_since_last_note_read(messages)
    if execs_since is not None and execs_since >= _REARM_AFTER_EXECUTIONS:
        return _skill_note_rearm_msg()
    return None


def _make_tools(
    execute_script_tool: Callable[..., Any],
    write_script_tool: Callable[..., Any],
    submit_goal_tool: Callable[..., Any],
    session_id: str,
    *,
    require_skill_note: bool = False,
    model: Any = None,
) -> List[Any]:
    """Build the three native tools, reusing the passed workbench state-callables.

    Each passed tool is a state-callable (``RunnableLambda``) that the baseline graph
    invokes with a partial state dict; we invoke them the same way and translate the
    result into a ``Command`` state update plus the ``ToolMessage`` the model reads.

    ``require_skill_note`` turns on the skill-note gate (specialist subagents only):
    ``write_script`` and ``submit_goal`` refuse to run until the model has called
    ``read_skill_note`` at least once this run, forcing the specialist to ground its
    solution in the deep note rather than the always-present one-line index. The
    planner and the standalone generative worker build their tools with this off, so
    they are never gated (and the planner has no ``read_skill_note`` tool to satisfy
    it with).
    """

    @tool
    def execute_script(
        command: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        """Run ONE non-interactive shell command in the persistent workbench terminal and return its raw output."""
        # Loop-breaker: refuse to re-run a command that has already been executed
        # several times in this loop. A weak model otherwise repeats the same
        # read-only inspection indefinitely instead of acting on output it already
        # has. The prior output is still in the message history, so we point back to
        # it and force a different next action instead of hitting the PTY again.
        # The budget is charged BEFORE the guards, because a guard-blocked call still
        # costs a model turn and therefore still costs money. Charging it after meant a
        # duplicate-spamming run spent its turns without ever advancing the counter:
        # measured 394 of 570 calls duplicate-blocked, so only ~176 of a 340 budget was
        # ever charged and the graph recursion limit ended the run instead. That breaks
        # the M/P match, since the two topologies spam duplicates at different rates and
        # recursion would then bind them at different real-action counts.
        sid = state.get("session_id") or session_id
        if _step_budget_exhausted(sid):
            print("EXECUTE_SCRIPT_TOOL[blocked-step-budget]", flush=True)
            return Command(
                update={
                    "command": None,
                    "messages": [ToolMessage(_STEP_BUDGET_TOOL_MSG, tool_call_id=tool_call_id)],
                }
            )
        _bump_step_budget(sid)

        command_norm = _normalize_command(command)
        prior_runs = _count_prior_executions(
            state.get("messages") or [], command_norm, tool_call_id
        )
        if prior_runs >= _MAX_IDENTICAL_RUNS:
            warn = (
                f"[DUPLICATE COMMAND BLOCKED] `{command}` has already been run "
                f"{prior_runs} times in this run; its output is already in your "
                "context above. Repeating it yields no new information and wastes the "
                "step budget. Take a DIFFERENT next action now: inspect a different "
                "file or region, read the whole file at once instead of slicing it, "
                "write a script with write_script to process what you already have, "
                "or — if you already recovered the answer from earlier output — call "
                "submit_goal. Do not run this command again."
            )
            print(
                f"EXECUTE_SCRIPT_TOOL[blocked-duplicate] command={command!r} "
                f"prior_runs={prior_runs}",
                flush=True,
            )
            return Command(
                update={
                    "command": None,
                    "messages": [ToolMessage(warn, tool_call_id=tool_call_id)],
                }
            )
        result = execute_script_tool.invoke({"command": command}) or {}
        out = result.get("script_output") or ""
        # Record the command alongside its output so the accumulated evidence reads
        # as a transcript: a re-delegated specialist can then see WHICH commands ran
        # (not just their output) and avoid repeating them.
        transcript = f"$ {command}\n{out}" if out else f"$ {command}\n(no output)"
        evidence = _append_evidence(state.get("execution_evidence") or "", transcript)
        # Memory subsystem: persist the raw output to the case store and (for verbose
        # output) put only a compressed observation into the message history, so the
        # in-run context stops growing one full tool dump at a time. Fail-open — on
        # any error this falls back to the legacy _cap(out) message verbatim.
        observation = _cap(out)
        if MEMORY_ENABLED:
            sid = state.get("session_id") or session_id
            subagent = state.get("active_subagent")
            abstracted = memory_fold.abstract_observation(model, command, out)
            agent_memory.record_case(sid, subagent, command, out, abstracted)
            observation = _cap(abstracted)
        return Command(
            update={
                "script_output": out,
                "execution_evidence": evidence,
                "command": None,
                "messages": [ToolMessage(observation, tool_call_id=tool_call_id)],
            }
        )

    @tool
    def write_script(
        content: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
        language: str = "py",
    ) -> Command:
        """Save a full script body to the shared workbench volume and return its absolute path. language is the file extension (py|sh|bash)."""
        # Loop-breaker: refuse to re-save a script body already written several times
        # this run. A weak model otherwise re-writes an identical script instead of
        # running the saved path or fixing it. Point it back at the saved copy.
        prior_writes = _count_prior_writes(
            state.get("messages") or [], content, tool_call_id
        )
        # Charged before the guards: a guard-blocked write still costs a model turn.
        sid = state.get("session_id") or session_id
        if _step_budget_exhausted(sid):
            print("WRITE_SCRIPT_TOOL[blocked-step-budget]", flush=True)
            return Command(
                update={
                    "write_script": None,
                    "messages": [ToolMessage(_STEP_BUDGET_TOOL_MSG, tool_call_id=tool_call_id)],
                }
            )
        _bump_step_budget(sid)

        if prior_writes >= _MAX_IDENTICAL_WRITES:
            warn = (
                f"[DUPLICATE SCRIPT BLOCKED] You have already saved this exact script "
                f"{prior_writes} times this run; saving it again changes nothing and "
                "wastes the step budget. Either RUN the path you already saved with "
                "execute_script, or MODIFY the script (fix the bug or switch technique) "
                "before saving. Do not save an identical script again."
            )
            print(
                f"WRITE_SCRIPT_TOOL[blocked-duplicate] prior_writes={prior_writes}",
                flush=True,
            )
            return Command(
                update={
                    "write_script": None,
                    "messages": [ToolMessage(warn, tool_call_id=tool_call_id)],
                }
            )
        if require_skill_note:
            gate_msg = _skill_note_gate(state.get("messages") or [])
            if gate_msg is not None:
                print(
                    "WRITE_SCRIPT_TOOL[blocked-skill-note] steering to read_skill_note",
                    flush=True,
                )
                return Command(
                    update={
                        "write_script": None,
                        "messages": [ToolMessage(gate_msg, tool_call_id=tool_call_id)],
                    }
                )
        result = (
            write_script_tool.invoke(
                {"write_script": content, "write_script_language": language}
            )
            or {}
        )
        out = result.get("script_output") or ""
        transcript = f"$ write_script ({language}) -> {out}" if out else f"$ write_script ({language})"
        evidence = _append_evidence(state.get("execution_evidence") or "", transcript)
        if MEMORY_ENABLED:
            sid = state.get("session_id") or session_id
            agent_memory.record_case(
                sid, state.get("active_subagent"), f"write_script ({language})", out, out
            )
        return Command(
            update={
                "script_output": out,
                "execution_evidence": evidence,
                "write_script": None,
                "messages": [ToolMessage(_cap(out), tool_call_id=tool_call_id)],
            }
        )

    @tool
    def submit_goal(
        flag: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        """Submit the recovered final answer for validation. Returns whether it was accepted or rejected (with a reason). Only call this with a value you actually recovered from real tool output."""
        sid = state.get("session_id") or session_id
        # Charged up front so a guard-blocked submission still costs budget: it costs a
        # model turn either way. Never short-circuited on the budget, though —
        # validation is cheap and an accepted flag is the win condition, which must not
        # be refused on a budget boundary. StepBudgetMiddleware ends the run after.
        _bump_step_budget(sid)
        # Loop-breaker: never re-validate a flag already rejected this run. A weak
        # model otherwise re-submits an IDENTICAL rejected value many times (28x
        # observed), burning the whole budget. A value already known wrong cannot
        # become right, so short-circuit and steer toward recovering a new value.
        if _flag_already_rejected(sid, flag):
            tried = _distinct_rejected_count(sid)
            msg = (
                f"[DUPLICATE SUBMISSION BLOCKED] '{flag}' was already submitted and "
                f"rejected this run — re-submitting it cannot succeed and wastes the "
                f"step budget (you have tried {tried} distinct value(s) already). Do NOT "
                "submit this value again. Recover a DIFFERENT value from real tool "
                "output (re-read the source/ciphertext, apply the correct transform) and "
                "submit only a value your own commands actually produced."
            )
            print(f"SUBMIT_GOAL_TOOL[blocked-duplicate] flag={flag!r}", flush=True)
            return Command(
                update={
                    "submitted_goal": None,
                    "messages": [ToolMessage(msg, tool_call_id=tool_call_id)],
                }
            )

        fmt = _flag_format_for(sid, state.get("query"))

        # Flag-shape guard (consolidated-experiment F1/F5/F6/W4): reject values that
        # are obviously not a flag — a filename, URL, memory address, report/plan prose,
        # the wrong wrapper, or an un-converted huge integer — BEFORE spending an oracle
        # call. This is the single highest-leverage fix identified in the failure
        # reports; it would have blocked most of the 377 wrong submissions.
        shape = classify_flag_candidate(flag, flag_format=fmt)
        if not shape.ok:
            _record_submission_attempt(sid, flag)
            _record_rejected_flag(sid, flag)  # so an identical re-submit is dup-blocked
            _shape_rejects[sid] = _shape_rejects.get(sid, 0) + 1
            print(
                f"SUBMIT_GOAL_TOOL[blocked-shape:{shape.category}] flag={flag!r}",
                flush=True,
            )
            return Command(
                update={
                    "submitted_goal": None,
                    "submission_verified": False,
                    "messages": [
                        ToolMessage(
                            f"[SUBMIT REJECTED] {shape.reason}",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        # Evidence-grounding guard (F6 report-as-flag, W5 hallucination, and the
        # recall_evidence laundering of a fabricated flag, H4): the value — or its
        # unwrapped body — must appear verbatim in this run's real command output.
        # A legitimately wrapped recovery (recover the secret, submit flag{secret})
        # passes because the body is grounded; a value the model invented, read from a
        # note, or lifted from a specialist's prose report does not.
        evidence = state.get("execution_evidence") or state.get("script_output") or ""
        # Ground against the full transcript AND the latest observation, so a value
        # just recovered is never missed if the capped evidence tail has rolled over.
        grounding_text = "\n".join(
            t for t in (state.get("execution_evidence"), state.get("script_output")) if t
        )
        if not is_grounded(flag, grounding_text):
            _record_submission_attempt(sid, flag)
            _record_rejected_flag(sid, flag)
            _ungrounded_rejects[sid] = _ungrounded_rejects.get(sid, 0) + 1
            print(f"SUBMIT_GOAL_TOOL[blocked-ungrounded] flag={flag!r}", flush=True)
            return Command(
                update={
                    "submitted_goal": None,
                    "submission_verified": False,
                    "messages": [
                        ToolMessage(
                            "[SUBMIT REJECTED] This value did not appear in any real "
                            "command output this run, so it cannot be a value you "
                            "recovered. Do NOT submit a flag taken from a skill note, a "
                            "specialist's report, or your own reasoning — first run a "
                            "command or script that actually PRINTS the exact secret, "
                            "then submit what that output shows.",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        # Distinct-guess cap (F4): stop the spray-submit spiral on compute/puzzle
        # challenges. Once enough distinct values have been rejected, a brand-new guess
        # is almost never the fix — a precise derivation is. Block it and steer to
        # verify locally. Re-submitting an already-tried value is handled above.
        distinct = _submitted_flags.get(sid, set())
        if (
            _MAX_DISTINCT_FLAGS > 0
            and _normalize_flag(flag) not in distinct
            and len(distinct) >= _MAX_DISTINCT_FLAGS
        ):
            print(
                f"SUBMIT_GOAL_TOOL[blocked-guess-cap] distinct={len(distinct)}",
                flush=True,
            )
            return Command(
                update={
                    "submitted_goal": None,
                    "messages": [
                        ToolMessage(
                            f"[SUBMIT REJECTED] You have already submitted "
                            f"{len(distinct)} distinct values this run and all were "
                            "rejected. Submitting more variants will not help — the "
                            "answer needs a precise derivation, not more guesses. Stop "
                            "guessing: compute the answer once and, if the challenge "
                            "exposes a service/checker, verify your candidate against it "
                            "before submitting; otherwise report the exact blocker.",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )
        if require_skill_note:
            gate_msg = _skill_note_gate(state.get("messages") or [])
            if gate_msg is not None:
                print(
                    "SUBMIT_GOAL_TOOL[blocked-skill-note] steering to read_skill_note",
                    flush=True,
                )
                return Command(
                    update={
                        "submitted_goal": None,
                        "messages": [ToolMessage(gate_msg, tool_call_id=tool_call_id)],
                    }
                )
        # Per-run diagnostics for the eval harness (real attempt counting).
        _record_submission_attempt(sid, flag)
        result = (
            submit_goal_tool.invoke(
                {
                    "submitted_goal": flag,
                    "script_output": evidence,
                    "session_id": session_id,
                    "flag_format": fmt,
                }
            )
            or {}
        )
        if bool(result.get("submission_verified")):
            msg = f"[SUBMIT ACCEPTED] '{flag}' was accepted. The task is solved."
            return Command(
                update={
                    "submitted_goal": flag,
                    "submission_verified": True,
                    "submission_rejection_reason": None,
                    "command": None,
                    "write_script": None,
                    "messages": [ToolMessage(msg, tool_call_id=tool_call_id)],
                }
            )
        reason = result.get("submission_rejection_reason") or "submission rejected"
        # Remember this rejected value so an identical re-submission is blocked above.
        _record_rejected_flag(sid, flag)
        if "placeholder" in reason.lower():
            _placeholder_submits[sid] = _placeholder_submits.get(sid, 0) + 1
        if MEMORY_ENABLED:
            agent_memory.record_feedback(
                state.get("session_id") or session_id, flag, reason
            )
        # Final-derivation hint (F3/W3): the value was grounded in real output and
        # correctly shaped but still wrong — often the last transform was skipped
        # (raw hex/int/base64 body, or a missing wrapper). Point at the likely fix so
        # a near-miss (broken_box, deedeedee, i_got_id) becomes a solve.
        hint = derivation_hint(flag, flag_format=fmt)
        msg = (
            f"[SUBMIT REJECTED] Reason: {reason}\n"
            + (f"{hint}\n" if hint else "")
            + "Keep investigating; recover the value from real tool output before "
            "submitting again."
        )
        return Command(
            update={
                "submitted_goal": None,
                "submission_verified": False,
                "submission_rejection_reason": reason,
                "messages": [ToolMessage(msg, tool_call_id=tool_call_id)],
            }
        )

    return [execute_script, write_script, submit_goal]


def _make_recall_tool(session_id: str) -> Any:
    """Build the on-demand evidence-recall tool (Piece 3, pull memory).

    A specialist calls ``recall_evidence(query)`` to fetch the top-k most relevant
    prior observations/strategy cards for THIS session, ranked by recency + keyword
    overlap, instead of receiving the whole transcript up front. Returned as a
    separate factory (not part of ``_make_tools``) so the existing three-tool
    unpacking contract callers rely on is untouched. Read-only and ungated — pulling
    context is a recon-like action the specialist should do freely before acting.
    """

    @tool
    def recall_evidence(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        """Search this run's accumulated evidence (prior command output + specialist findings) for the top matches to a short query, e.g. 'AES key' or 'open ports'. Use this to recover a detail you need instead of re-running broad reconnaissance."""
        sid = state.get("session_id") or session_id
        if not MEMORY_ENABLED:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            "[recall_evidence] memory subsystem disabled; nothing to recall.",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )
        hits = agent_memory.recall(sid, query, _RECALL_K)
        langfuse_tracer.record_event(
            "memory-recall", query=query, hits=len(hits)
        )
        if not hits:
            body = (
                f"[recall_evidence] No prior evidence matched {query!r}. "
                "Gather it yourself with execute_script."
            )
        else:
            lines = []
            for i, h in enumerate(hits, 1):
                label = h.get("command") or h.get("task") or "(evidence)"
                lines.append(f"{i}. [{label}] {h.get('snippet', '')}")
            body = (
                f"[recall_evidence] Top {len(hits)} matches for {query!r}:\n"
                + "\n".join(lines)
            )
        return Command(
            update={"messages": [ToolMessage(_cap(body), tool_call_id=tool_call_id)]}
        )

    return recall_evidence


# A weak planner can loop, re-emitting an IDENTICAL write_todos plan turn after turn
# without ever delegating (`task`) or finishing — the planning analogue of the
# execute_script duplicate loop. When the same list reappears this many times in a row
# we steer the planner to act on the plan instead of re-writing it.
_MAX_IDENTICAL_TODOS = 2
# Cap how many steers we inject per run, so a truly stuck planner adds at most this many
# nudges before the recursion_limit ends the run (parity: recursion_limit is the hard
# ceiling).
_MAX_TODOS_STEERS = 3
_TODOS_LOOP_MARKER = "[PLAN LOOP]"
_TODOS_LOOP_STEER = (
    "[PLAN LOOP] You have written the same to-do list several times without acting on "
    "it — re-planning is not progress. Stop calling write_todos and take a concrete "
    "step now: delegate the next item to a specialist with the `task` tool, or, if a "
    "value has already been recovered from real tool output, submit it with "
    "submit_goal. Do not emit the same plan again."
)


def _todos_signature(tool_call: Dict[str, Any]) -> str:
    """Stable key for a write_todos call: its ordered todo ``content`` strings joined."""
    todos = (tool_call.get("args") or {}).get("todos") or []
    parts = [
        str((t or {}).get("content", "")).strip()
        for t in todos
        if isinstance(t, dict)
    ]
    return "||".join(parts)


def _trailing_identical_write_todos(messages: List[Any]) -> int:
    """Length of the trailing run of IDENTICAL write_todos turns (scanning newest
    first), broken by any ``task`` delegation or other tool turn. 0 if the most recent
    tool-calling turn was not write_todos. Detects a planner stuck re-planning.
    """
    run = 0
    sig = None
    for m in reversed(messages or []):
        tcs = getattr(m, "tool_calls", None)
        if not tcs:
            continue
        names = [tc.get("name") for tc in tcs]
        if "task" in names:
            break  # a delegation means the planner made progress
        wt = next((tc for tc in tcs if tc.get("name") == "write_todos"), None)
        if wt is None:
            break  # some other tool turn — not the write_todos loop
        cur = _todos_signature(wt)
        if sig is None:
            sig, run = cur, 1
        elif cur == sig:
            run += 1
        else:
            break
    return run


def _count_text_marker(messages: List[Any], marker: str) -> int:
    """How many messages contain ``marker`` in their text content."""
    n = 0
    for m in messages or []:
        content = getattr(m, "content", "")
        text = content if isinstance(content, str) else str(content)
        if marker in text:
            n += 1
    return n


# --- Context-window bounding --------------------------------------------------
# Diagnosed failure (consolidated-experiment): on a long spiral (regexpire, 867
# steps / 113 submit calls) the planner's message history grew unbounded and blew
# past gpt-4o-mini's 128k context, returning an HTTP 400 that killed the run. This
# middleware trims the OLDEST complete tool-rounds out of the history before each
# model call once it exceeds a char budget, always keeping the initial briefing and
# a recent suffix. Trimming is pairing-safe (see core.helpers.history_bound):
# it only ever cuts at a round boundary, so no OpenAI "tool_call without response"
# error is possible. 0 disables it (legacy behaviour).
_MAX_HISTORY_CHARS = int(os.getenv("GENCYBER_MAX_HISTORY_CHARS", "120000"))
_HISTORY_KEEP_RECENT = int(os.getenv("GENCYBER_HISTORY_KEEP_RECENT", "12"))


class HistoryBoundMiddleware(AgentMiddleware):
    """Trim the oldest tool-rounds from an agent's history before each model call.

    Attached to the planner and the standalone generative worker (and specialists),
    it keeps long runs from exceeding the model's context window (the regexpire
    128k-context HTTP 400). Fail-open: any error leaves the history untouched.
    """

    def _trim(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            messages = state.get("messages") or []
            drop = select_messages_to_drop(
                messages,
                limit_chars=_MAX_HISTORY_CHARS,
                keep_recent=_HISTORY_KEEP_RECENT,
            )
            if not drop:
                return None
            print(
                f"HISTORY_BOUND[trim] dropping {len(drop)} of {len(messages)} messages",
                flush=True,
            )
            return {"messages": [RemoveMessage(id=m.id) for m in drop]}
        except Exception as e:  # pragma: no cover - must never break a run
            print(f"HISTORY_BOUND[error] {e}", flush=True)
            return None

    def before_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        return self._trim(state)

    async def abefore_model(
        self, state: Dict[str, Any], runtime: Any = None
    ) -> Optional[Dict[str, Any]]:
        return self._trim(state)


class StepBudgetMiddleware(AgentMiddleware):
    """Deterministically end an agent once the shared per-challenge budget is spent.

    Attached to BOTH the planner and every specialist subagent so it can stop each
    graph the moment the global action budget (shared across all of them via
    ``session_id``) is exhausted. This is what actually bounds per-challenge work: the
    tools stop doing real work at the budget, and this hook then ends whatever graph is
    currently running instead of letting it spin against its own ``recursion_limit``
    emitting budget-exhausted tool calls (a specialist has no enforced recursion limit,
    so without this it would loop forever after the budget trips).

    A no-op when the budget is disabled (``GENCYBER_STEP_BUDGET`` unset / 0).
    """

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self._session_id = session_id

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        sid = state.get("session_id") or self._session_id
        if _step_budget_exhausted(sid):
            print(
                f"STEP_BUDGET[end] session={sid} used={_steps_used(sid)}/{_STEP_BUDGET}",
                flush=True,
            )
            return {"jump_to": "end"}
        return None


class EscalationMiddleware(AgentMiddleware):
    """Swap in a stronger model for the rest of a specialist's run once it stalls.

    Trigger (any of, per shared session_id): a placeholder submission, a rejected
    submission, or spending >= ``GENCYBER_ESCALATION_AFTER_ACTIONS`` actions without an
    accepted flag. Once tripped it is sticky for the run (``_escalated``), so the strong
    model is used for every subsequent model call in that specialist. Uses the langchain
    ``wrap_model_call`` hook with ``ModelRequest.override(model=...)`` — the framework
    re-binds tools to the swapped model, so tool-calling continues unchanged. A no-op when
    escalation is disabled or the strong model failed to build (``strong`` is None).
    """

    def __init__(self, session_id: str, strong_model: Any) -> None:
        super().__init__()
        self._session_id = session_id
        self._strong = strong_model

    def _triggered(self, state: Dict[str, Any]) -> tuple:
        sid = (state or {}).get("session_id") or self._session_id
        if _escalated.get(sid):
            return sid, True
        after_actions = _escalation_after_actions_for(sid)
        trip = (
            _placeholder_submits.get(sid, 0) > 0
            or _distinct_rejected_count(sid) > 0
            or (after_actions > 0 and _steps_used(sid) >= after_actions)
        )
        return sid, trip

    def _apply(self, request: Any) -> Any:
        if self._strong is None:
            return request
        state = getattr(request, "state", None)
        if not isinstance(state, dict):
            try:
                state = dict(state or {})
            except Exception:
                state = {}
        sid, trip = self._triggered(state)
        if not trip:
            return request
        if not _escalated.get(sid):
            _escalated[sid] = True
            strong_name = _escalation_model_for(sid)
            print(
                f"ESCALATION[fired] session={sid} model={strong_name} "
                f"steps={_steps_used(sid)} rejects={_distinct_rejected_count(sid)}",
                flush=True,
            )
            langfuse_tracer.record_event("escalation", session=sid, model=strong_name)
        try:
            return request.override(model=self._strong)
        except Exception:
            # Older ModelRequest without override(): mutate in place (dataclass is mutable).
            try:
                request.model = self._strong
            except Exception:
                logger.debug("could not swap model for escalation", exc_info=True)
            return request

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._apply(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._apply(request))


class FlagGateMiddleware(AgentMiddleware):
    """Enforce baseline routing parity: end only on an accepted submission.

    - Mirrors the latest assistant text into ``generative_agent_response`` so the
      existing API response / stream keep working.
    - When a submission has been accepted, force the run to end.
    - Otherwise, if the model turn produced no tool calls, append a nudge and loop
      back to the model — the graph never ends while the goal is unrecovered; the
      ``recursion_limit`` is the only hard ceiling (matching the baseline).
    """

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        messages = state.get("messages") or []
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )

        updates: Dict[str, Any] = {}
        if last_ai is not None:
            content = last_ai.content
            updates["generative_agent_response"] = (
                content if isinstance(content, str) else str(content)
            )
            # Observability: surface the planner's routing decision. The planner
            # delegates by calling the DeepAgents ``task`` tool, which is otherwise
            # silent — record which specialist it picked (and the handed-off task) so
            # the choice is visible in both run logs and the API/stream/UI.
            for tc in getattr(last_ai, "tool_calls", None) or []:
                name = tc.get("name")
                if name == "task":
                    args = tc.get("args") or {}
                    subagent = args.get("subagent_type")
                    task_desc = (args.get("description") or "")[:300]
                    updates["active_subagent"] = subagent
                    updates["active_subagent_task"] = task_desc
                    # Record the FULL delegation sequence (not just the last) so the eval
                    # can score routing over the whole run — the old correct_subagent read
                    # only the final active_subagent and hid mid-run category-hopping.
                    _sid = state.get("session_id")
                    if _sid and subagent:
                        _delegations.setdefault(_sid, []).append(subagent)
                    print(
                        f"PLANNER_DELEGATION subagent_type={subagent!r} "
                        f"description={task_desc[:200]!r}",
                        flush=True,
                    )
                    # Middleware hooks aren't runnables; the callback bus can't see the
                    # planner's delegation choice. Surface it as a trace event.
                    langfuse_tracer.record_event(
                        "planner-delegation",
                        subagent_type=subagent,
                        task_description=task_desc,
                    )
                elif name == "write_todos":
                    # Observability for the planner's planning step: write_todos is
                    # auto-provided by DeepAgents and is otherwise silent. Logging it
                    # lets us see WHEN the planner plans vs. dives straight into work.
                    todos = (tc.get("args") or {}).get("todos") or []
                    preview = [
                        str((t or {}).get("content", ""))[:60]
                        for t in todos
                        if isinstance(t, dict)
                    ][:8]
                    print(
                        f"WRITE_TODOS count={len(todos)} items={preview}",
                        flush=True,
                    )
                    langfuse_tracer.record_event(
                        "planner-write-todos",
                        count=len(todos),
                        items=preview,
                    )

        if state.get("submission_verified"):
            updates["jump_to"] = "end"
            return updates

        if last_ai is not None and not getattr(last_ai, "tool_calls", None):
            updates["messages"] = [HumanMessage(content=_NUDGE)]
            updates["jump_to"] = "model"

        return updates or None

    def _todos_loop_steer(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Break a planner re-emitting an identical write_todos plan without acting.

        Injected at ``before_model`` (not ``after_model``): by then the previous
        write_todos tool call already has its ToolMessage, so appending a HumanMessage
        keeps the AIMessage→ToolMessage pairing valid. Bounded by ``_MAX_TODOS_STEERS``
        so a stuck planner can't fill the context with nudges before the recursion limit.
        """
        messages = state.get("messages") or []
        if _trailing_identical_write_todos(messages) < _MAX_IDENTICAL_TODOS:
            return None
        if _count_text_marker(messages, _TODOS_LOOP_MARKER) >= _MAX_TODOS_STEERS:
            return None
        print("PLAN_LOOP[steer] planner repeated identical write_todos", flush=True)
        return {"messages": [HumanMessage(content=_TODOS_LOOP_STEER)]}

    def before_model(
        self, state: Dict[str, Any], runtime: Any = None
    ) -> Optional[Dict[str, Any]]:
        return self._todos_loop_steer(state)

    async def abefore_model(
        self, state: Dict[str, Any], runtime: Any = None
    ) -> Optional[Dict[str, Any]]:
        return self._todos_loop_steer(state)


# --- Planner consolidation nudge -------------------------------------------
# Diagnosed failure: the planner delegates endless recon (~28 task calls/challenge)
# and never consolidates what specialists already recovered into a submission — it
# burns the budget and thrashes to the recursion limit with zero submissions. This
# middleware (planner-only) periodically surfaces the accumulated evidence and steers
# the planner to SUBMIT a recovered value, or delegate only the specific missing step,
# instead of ordering another survey. It is purely advisory: NO stronger model, NO
# hard delegation cap, and it never force-ends the run — just escalating nudges,
# bounded so they can't flood the context.
_CONSOLIDATE_AFTER_DELEGATIONS = int(os.getenv("GENCYBER_CONSOLIDATE_AFTER_DELEGATIONS", "4"))
_MAX_CONSOLIDATE_STEERS = int(os.getenv("GENCYBER_MAX_CONSOLIDATE_STEERS", "5"))
_CONSOLIDATE_EVIDENCE_CHARS = int(os.getenv("GENCYBER_CONSOLIDATE_EVIDENCE_CHARS", "4000"))
_CONSOLIDATE_MARKER = "[CONSOLIDATE]"


def _count_task_calls(messages: List[Any]) -> int:
    """How many ``task`` delegations the planner has made this run."""
    n = 0
    for m in messages or []:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") == "task":
                n += 1
    return n


class ConsolidationMiddleware(AgentMiddleware):
    """Nudge the planner to turn recovered evidence into a submission.

    Attached to the planner only. After every ``_CONSOLIDATE_AFTER_DELEGATIONS``
    delegations without an accepted flag, injects the accumulated
    ``execution_evidence`` tail plus a steer to submit a recovered value (or delegate
    only the one missing step) rather than ordering more reconnaissance. Bounded by
    ``_MAX_CONSOLIDATE_STEERS``; advisory only — no hard cap, never force-ends.
    """

    def _steer(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if state.get("submission_verified"):
            return None
        messages = state.get("messages") or []
        tasks = _count_task_calls(messages)
        steers = _count_text_marker(messages, _CONSOLIDATE_MARKER)
        if steers >= _MAX_CONSOLIDATE_STEERS:
            return None
        # Fire the Nth steer only after N*threshold delegations, so the nudge escalates
        # with continued delegation instead of firing on every turn.
        if tasks < _CONSOLIDATE_AFTER_DELEGATIONS * (steers + 1):
            return None
        evidence = (state.get("execution_evidence") or "").strip()
        ev_tail = evidence[-_CONSOLIDATE_EVIDENCE_CHARS:] if evidence else ""
        body = (
            f"{_CONSOLIDATE_MARKER} You have delegated {tasks} step(s) without submitting. "
            "STOP ordering more reconnaissance and CONSOLIDATE what your specialists have "
            "already recovered. Review the evidence below: if it contains the flag — or a "
            "value that, in this challenge's answer format, IS the flag — submit it now with "
            "submit_goal. Do NOT delegate again merely to re-derive something you already "
            "have. Only if a SPECIFIC recovery step is genuinely still missing, delegate that "
            "ONE step and hand over the concrete facts already discovered — never another "
            "broad survey."
            + (
                f"\n\n<RecoveredEvidence>\n{ev_tail}\n</RecoveredEvidence>"
                if ev_tail
                else "\n\n(No command evidence has accumulated yet. If repeated delegations "
                "keep returning nothing, the current approach is wrong — direct a clearly "
                "different technique.)"
            )
        )
        print(f"CONSOLIDATE[steer] tasks={tasks} steer#{steers + 1}", flush=True)
        return {"messages": [HumanMessage(content=body)]}

    def before_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        return self._steer(state)

    async def abefore_model(
        self, state: Dict[str, Any], runtime: Any = None
    ) -> Optional[Dict[str, Any]]:
        return self._steer(state)


_PRIOR_EVIDENCE_MARKER = "<PriorEvidence>"


# Per-message preview length for the SUBAGENT_INPUT dump. Long enough to recognize
# each block (task description, PriorEvidence, the loaded note) without flooding logs.
_INPUT_PREVIEW_CHARS = 240


def _one_line(text: Any, n: int = _INPUT_PREVIEW_CHARS) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[:n] + "…"


def _just_loaded_skill_note(messages: List[Any]) -> bool:
    """True iff the model is about to react to a freshly loaded skill note: the most
    recent assistant turn called ``read_skill_note`` and its tool result is now the
    tail of the message list. Fires once per read (the next assistant turn replaces
    the trailing AIMessage), so it does not spam every subsequent model call.
    """
    if not messages or not isinstance(messages[-1], ToolMessage):
        return False
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return False
    return any(
        (tc.get("name") == "read_skill_note")
        for tc in (getattr(last_ai, "tool_calls", None) or [])
    )


def _log_subagent_input_after_skill(request: Any) -> None:
    """Print the literal model input (system prompt size + full message stack) on the
    turn right after a specialist loads a skill note, so we can confirm the note
    content actually reached the model's context window. Observability only — never
    raises into the run.
    """
    try:
        messages = list(getattr(request, "messages", None) or [])
        if not _just_loaded_skill_note(messages):
            return
        sys_msg = getattr(request, "system_message", None)
        sys_content = getattr(sys_msg, "content", sys_msg)
        sys_chars = len(
            sys_content if isinstance(sys_content, str) else str(sys_content or "")
        )
        state = getattr(request, "state", None)
        sub = state.get("active_subagent") if isinstance(state, dict) else None
        print(
            f"SUBAGENT_INPUT[after-skill-load] subagent={sub!r} "
            f"system_message_chars={sys_chars} messages={len(messages)}",
            flush=True,
        )
        for i, m in enumerate(messages):
            content = getattr(m, "content", "")
            text = content if isinstance(content, str) else str(content)
            kind = m.__class__.__name__
            extra = ""
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                extra = " tool_calls=" + str([tc.get("name") for tc in tcs])
            name = getattr(m, "name", None)
            if isinstance(m, ToolMessage) and name:
                extra = f" name={name!r}"
            print(
                f"  [{i}] {kind} chars={len(text)}{extra} :: {_one_line(text)}",
                flush=True,
            )
    except Exception as e:  # pragma: no cover - observability must not break the run
        print(f"SUBAGENT_INPUT[log-error] {e}", flush=True)


class SubagentContextMiddleware(AgentMiddleware):
    """Carry prior findings into a freshly-spawned specialist, and fold its run back.

    Every DeepAgents ``task`` delegation spins up a specialist with a *fresh* message
    history — it sees only the planner's ``description``, never what prior delegations
    discovered. Without this, a re-delegated specialist re-runs the same recon from
    scratch.

    Two memory modes (see docs/agent-memory-design.md):

    * ``MEMORY_ENABLED`` (default) — at the start of each subagent run we inject the
      most recent **fold cards** (compact ``task → findings → next_step`` summaries
      from the strategy store), NOT the raw transcript. When the run finishes,
      ``after_agent`` folds the whole sub-trajectory into a new card. This keeps
      injected context to a few hundred chars instead of a 12k raw dump.
    * Legacy — inject ``_cap(execution_evidence)`` (the full accumulated transcript)
      once per run, the original behaviour, used when memory is disabled for A/B.
    """

    def __init__(self, model: Any = None) -> None:
        super().__init__()
        self._model = model

    def _injection(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        messages = state.get("messages") or []
        # Inject only once per subagent run: the injected message persists in this
        # run's history, so later turns see the marker and skip. A *new* delegation
        # starts with fresh messages (no marker) and re-injects the latest context.
        for m in messages:
            content = getattr(m, "content", "")
            text = content if isinstance(content, str) else str(content)
            if _PRIOR_EVIDENCE_MARKER in text:
                return None

        if MEMORY_ENABLED:
            block = self._fold_card_block(state)
        else:
            block = self._legacy_evidence_block(state)
        if block is None:
            return None
        return {"messages": [HumanMessage(content=block)]}

    def _fold_card_block(self, state: Dict[str, Any]) -> Optional[str]:
        session_id = state.get("session_id")
        cards = agent_memory.recent_folds(session_id, _FOLD_INJECT_N)
        if not cards:
            return None
        rendered = []
        # Oldest-first so the newest card reads last (closest to the model's cursor).
        for c in reversed(cards):
            parts = []
            if c.get("task"):
                parts.append(f"task: {c['task']}")
            if c.get("findings"):
                parts.append(f"findings: {c['findings']}")
            if c.get("artifacts"):
                parts.append(f"artifacts: {c['artifacts']}")
            if c.get("next_step"):
                parts.append(f"next_step: {c['next_step']}")
            if parts:
                rendered.append("- " + "\n  ".join(parts))
        if not rendered:
            return None
        body = "\n".join(rendered)
        return (
            f"{_PRIOR_EVIDENCE_MARKER}\n"
            "Compact summaries of what earlier specialists already accomplished this "
            "run. Do NOT repeat their reconnaissance — build on these findings and take "
            "the next step toward the goal. If you need the exact command output behind "
            "a finding, recover it yourself rather than re-running broad surveys:\n\n"
            f"{_cap(body)}\n"
            "</PriorEvidence>"
        )

    def _legacy_evidence_block(self, state: Dict[str, Any]) -> Optional[str]:
        evidence = state.get("execution_evidence") or ""
        if not evidence.strip():
            return None
        return (
            f"{_PRIOR_EVIDENCE_MARKER}\n"
            "Commands already run earlier in this run, with their output. Do NOT repeat "
            "this reconnaissance — read it and take the next step toward recovering the "
            "goal (e.g. implement the decryption/exploit, not another file/type survey):\n\n"
            f"{_cap(evidence)}\n"
            "</PriorEvidence>"
        )

    def _fold(self, state: Dict[str, Any]) -> None:
        """Fold this finished sub-trajectory into a strategy card (Piece 2b)."""
        if not MEMORY_ENABLED:
            return
        session_id = state.get("session_id")
        if not session_id:
            return
        task = state.get("active_subagent_task")
        card = memory_fold.fold_subtrajectory(
            self._model, task, state.get("messages") or []
        )
        if not card:
            return
        card["subagent"] = state.get("active_subagent") or ""
        card["task"] = task or ""
        agent_memory.record_fold(session_id, card)
        langfuse_tracer.record_event(
            "memory-fold",
            subagent=card.get("subagent"),
            findings_chars=len(card.get("findings") or ""),
            next_step_chars=len(card.get("next_step") or ""),
        )

    def before_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        return self._injection(state)

    async def abefore_model(
        self, state: Dict[str, Any], runtime: Any = None
    ) -> Optional[Dict[str, Any]]:
        return self._injection(state)

    def after_agent(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        self._fold(state)
        return None

    async def aafter_agent(
        self, state: Dict[str, Any], runtime: Any = None
    ) -> Optional[Dict[str, Any]]:
        # _fold makes a blocking LLM call; run it off the event loop so concurrent
        # streams are not stalled while a sub-trajectory is folded.
        import asyncio

        await asyncio.to_thread(self._fold, state)
        return None

    # Observability: dump the literal model input on the turn right after a skill
    # note is loaded, so the run logs prove the note content reached the model's
    # context (not just that read_skill_note ran). Pass-through wrappers — they only
    # print, then delegate to the real model call unchanged.
    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        _log_subagent_input_after_skill(request)
        return handler(request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        _log_subagent_input_after_skill(request)
        return await handler(request)


class DeepGenerativeWorkflow:
    """Builder + thin runtime adapter around a compiled DeepAgents graph.

    Exposes the same surface the service drives on the baseline ``WorkflowGraph``:
    ``graph`` (compiled), ``session_id``, ``recursion_limit``,
    ``_build_initial_state(query)`` and ``invoke(query=...)``.
    """

    REQUIRED_AGENTS: frozenset = frozenset({"generative"})

    def __init__(
        self,
        session_id: str,
        *,
        graph: Any,
        recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
    ) -> None:
        self.session_id = session_id
        self.graph = graph
        self.recursion_limit = recursion_limit

    def _build_initial_state(self, query: str) -> Dict[str, Any]:
        now = datetime.now()
        return {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "session_id": self.session_id,
            "timestamp": now,
            "created_at": now,
            "submitted_goal": None,
            "submission_verified": None,
            "submission_rejection_reason": None,
            "script_output": None,
            "execution_evidence": "",
            "generative_agent_response": None,
            "command": None,
            "write_script": None,
        }

    def invoke(self, query: str) -> Dict[str, Any]:
        logger.info("Invoking deep generative workflow query=%s", query)
        # Fresh per-challenge action budget (shared by the planner + all specialists
        # that run under this session_id).
        _reset_step_budget(self.session_id)
        # Fresh per-challenge rejected-flag set (submission loop-breaker).
        _reset_submission_tracking(self.session_id)
        state_dict = self._build_initial_state(query=query)
        try:
            with langfuse_tracer.traced_run(
                self.session_id,
                name=type(self).__name__,
                tags=[type(self).__name__, "invoke"],
            ) as handler:
                config: Dict[str, Any] = {
                    "configurable": {"thread_id": self.session_id},
                    "recursion_limit": self.recursion_limit,
                }
                if handler is not None:
                    config["callbacks"] = [handler]
                final_state = self.graph.invoke(state_dict, config=config)
                langfuse_tracer.record_output(final_state)
            print("FINAL_STATE", final_state, flush=True)
            logger.info("Deep generative workflow completed")
            return final_state
        except Exception as e:
            logger.error("Error invoking deep generative workflow: %s", e)
            return {
                "error": str(e),
                "generated_response": f"I encountered an error processing your request: {str(e)}",
            }

    @classmethod
    def _compile(
        cls,
        *,
        model: Any,
        system_prompt: str,
        tools: List[Any],
        checkpointer: Any,
        subagents: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
        extra_middleware: Optional[List[Any]] = None,
    ) -> Any:
        _register_profiles()
        # StepBudgetMiddleware runs first so an exhausted budget ends the graph before
        # FlagGateMiddleware's no-tool-call nudge can loop it back to the model. It is a
        # no-op when the budget is disabled or ``session_id`` is absent (topology).
        middleware: List[Any] = []
        # Trim first so a long history is bounded before any other before_model hook
        # (consolidation injection, todos steer) reads or appends to it.
        middleware.append(HistoryBoundMiddleware())
        if session_id is not None:
            middleware.append(StepBudgetMiddleware(session_id))
        middleware.append(FlagGateMiddleware())
        # Caller-supplied middleware (e.g. the planner's ConsolidationMiddleware).
        if extra_middleware:
            middleware.extend(extra_middleware)
        kwargs: Dict[str, Any] = dict(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            state_schema=DeepGenerativeState,
            middleware=middleware,
            checkpointer=checkpointer,
        )
        if subagents:
            kwargs["subagents"] = subagents
        return create_deep_agent(**kwargs)

    @classmethod
    def _attach_skill_scope(
        cls,
        tools: List[Any],
        system_prompt: str,
        session_id: Optional[str],
    ) -> tuple:
        """Give the monolith the SAME corpus access one planner specialist has.

        A specialist is built over exactly one ``ctf-*`` pack: that pack's SKILL.md is
        its system-prompt index, and ``read_skill_note`` is sandboxed to that pack's
        notes. To be resource-MATCHED the monolith needs both halves, scoped by the
        challenge's category hint (:func:`set_skill_scope`).

        Both halves matter and for different reasons. Without the tool the monolith
        cannot reach the corpus at all, so a planner win would partly be a corpus win.
        Without the SKILL.md index it can reach the corpus but does not know what is in
        it, which is a subtler version of the same confound.

        Returns ``(tools, system_prompt)`` unchanged when no category was recorded, so
        non-benchmark runs and the baseline registry are unaffected.
        """
        pack = _pack_for_session(session_id)
        if not pack:
            logger.info("monolith skill scope: none (no category hint for this run)")
            return tools, system_prompt
        skill = next((s for s in discover_ctf_skills() if s.name == pack), None)
        if skill is None:
            logger.warning("monolith skill scope %s not found in corpus", pack)
            return tools, system_prompt
        tools = list(tools)
        tools.append(make_read_skill_note_tool(skill.skill_dir, skill.notes))
        if section_retrieval_enabled():
            tools.append(make_search_skill_tool(skill.skill_dir, skill.notes))
        logger.info("monolith skill scope: %s (%d notes)", pack, len(skill.notes))
        return tools, compose_subagent_prompt(system_prompt, skill)

    @classmethod
    def build(
        cls,
        *,
        session_id: str,
        agents: Dict[str, Any],
        execute_script_tool: Callable[..., Any],
        write_script_tool: Callable[..., Any],
        submit_goal_tool: Callable[..., Any],
    ) -> "DeepGenerativeWorkflow":
        agent = agents["generative"]
        model = getattr(agent, "model", None)
        if model is None:
            raise ValueError(
                "DeepGenerativeWorkflow requires the 'generative' agent to expose a "
                "tool-bindable .model (use agent_class_key: DeepGenerativeAgent)"
            )
        system_prompt = getattr(agent, "system_prompt", "") or ""

        tools = _make_tools(
            execute_script_tool, write_script_tool, submit_goal_tool, session_id,
            model=model,
        )
        tools.append(_make_recall_tool(session_id))

        tools, system_prompt = cls._attach_skill_scope(tools, system_prompt, session_id)

        # ---- escalation parity (M1) ----
        # Identical sticky ladder to a specialist's: same trigger (placeholder
        # submission, rejected submission, or the action threshold), same strong model,
        # same per-episode stickiness. ``build_escalation_model`` returns None when the
        # run set ``--escalation-model off``, which is exactly M0 — so M0 and M1 differ
        # by the runner flag alone, not by a code path.
        strong_model = build_escalation_model(agent, session_id=session_id)
        extra_middleware = [EscalationMiddleware(session_id, strong_model)]

        mongo_client = get_mongodb_client()
        db_name = os.getenv("MONGODB_DATABASE", "gencyber")
        checkpointer = MongoDBSaver(mongo_client, db_name=db_name)

        compiled = cls._compile(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            checkpointer=checkpointer,
            session_id=session_id,
            extra_middleware=extra_middleware,
        )
        return cls(session_id, graph=compiled)

    @classmethod
    def topology(cls) -> Dict[str, Any]:
        """Serialize the compiled deep agent shape (no MongoDB, no side effects)."""
        from application.langgraph.models.langraph_model import serialize_graph_topology
        from langchain_openai import ChatOpenAI

        dummy_model = ChatOpenAI(
            model="openai/gpt-4o-mini",
            api_key="dummy",
            base_url="https://openrouter.ai/api/v1",
        )
        noop = RunnableLambda(lambda state: {})
        tools = _make_tools(noop, noop, noop, "__topology__")
        compiled = cls._compile(
            model=dummy_model,
            system_prompt="topology",
            tools=tools,
            checkpointer=False,
        )
        return serialize_graph_topology(compiled)
