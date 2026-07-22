"""Validate and finalize ``submitted_goal``.

Submission validation is **per-tool**, not derived from any benchmark metadata on
the graph state. The UI explicitly selects which submit-goal tool to attach to the
workflow (a tool id passed at init); the active workflow instantiates that one when
the graph is built (see ``LangGraphService._build_workflow``).

Two validation strategies:
  - **Ground-truth (workbench-validated)** — :class:`NYUCTFSubmitGoalTool`,
    selected by the ``nyuctf_submit_goal`` tool id. The agent holds no flag and no
    challenge identity: it POSTs the candidate (plus its ``session_id``) to the
    workbench, which owns challenge materialization and therefore the ground truth,
    and the workbench returns accept/reject. Mirrors nyuctf_agents' ``CheckFlag``,
    but with the comparison living where the challenge does.
  - **Evidence-based (no oracle)** — :class:`BaselineSubmitGoalTool`, the default
    when no ground-truth tool is selected. There is no oracle, so a value the agent
    genuinely recovered — one present verbatim in real tool output
    (``script_output``) — is accepted, while an answer with no supporting evidence
    is rejected. This is the original single-agent baseline behaviour.

New submit tools register themselves in ``SUBMIT_TOOL_REGISTRY`` below under their
UI tool id (e.g. ``bandit_submit_goal``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type


@dataclass(frozen=True)
class SubmitValidationResult:
    accepted: bool
    reason: Optional[str] = None


# Whole-value placeholders the agent must never submit (lowercased, exact match).
_PLACEHOLDER_EXACT = frozenset(
    {
        "flag", "flag{}", "flag{...}", "example", "test", "todo", "tbd",
        "placeholder", "changeme", "redacted", "none", "null", "n/a",
        "foo", "bar", "sample", "yourflag", "your_flag",
    }
)

# Substrings that mark a value as a template rather than a recovered secret.
_PLACEHOLDER_SUBSTR = (
    "example_flag", "example flag", "your_flag_here", "your flag here",
    "yourflaghere", "flag_here", "flaghere", "insert_flag", "insert flag",
    "sample_flag", "dummy", "placeholder", "redacted", "<flag>", "fill_in",
)


def looks_like_placeholder(candidate: str) -> bool:
    """Deterministic guard against submitting a template/placeholder value.
    """
    norm = (candidate or "").strip().lower()
    if not norm:
        return False  # empty is handled separately as "empty submission"
    # Template markers — ellipsis / angle-bracket placeholders like ``flag{...}`` or ``<flag>``.
    if "..." in norm or "<" in norm or ">" in norm:
        return True
    if norm in _PLACEHOLDER_EXACT:
        return True
    if any(sub in norm for sub in _PLACEHOLDER_SUBSTR):
        return True
    # ``flag{example}`` / ``flag{your_flag}`` / ``flag{placeholder}`` style wrappers.
    if re.fullmatch(r"flag\{(?:example|your[_ ]?flag|placeholder|test|todo|xxx+|\.\.\.)\}", norm):
        return True
    return False


# Below this length a substring match against raw output is too weak to trust as
# evidence (a 1-2 char candidate matches almost any output).
_MIN_EVIDENCE_MATCH_LEN = 3


def validate_baseline_submission(
    candidate: str,
    *,
    evidence_text: str,
) -> SubmitValidationResult:
    """No-oracle fallback validator (baseline-style).

    Accept when the candidate appears verbatim (case-insensitive substring) in the
    latest real tool output (``script_output``) — i.e. it is a value the agent
    genuinely recovered from execution.

    Otherwise reject: with no ground-truth flag, an answer unsupported by any
    recovered evidence is treated as a guess. (The shared placeholder guard in
    :meth:`BaseSubmitGoalTool.__call__` has already run before this.)
    """
    submitted = (candidate or "").strip()
    if not submitted:
        return SubmitValidationResult(accepted=False, reason="empty submission")

    norm = submitted.lower()
    if len(submitted) >= _MIN_EVIDENCE_MATCH_LEN and norm in (evidence_text or "").lower():
        return SubmitValidationResult(accepted=True)

    return SubmitValidationResult(
        accepted=False,
        reason=(
            "submitted value was not found in any real command output — recover it "
            "from actual tool execution before submitting"
        ),
    )


def format_rejection_feedback(reason: str, candidate: str) -> str:
    return (
        "[SUBMIT REJECTED]\n"
        f"Reason: {reason}\n"
        f"Rejected candidate: {candidate!r}\n"
        "Continue investigating; submit only after the flag appears in execution output."
    )


class BaseSubmitGoalTool:
    """Abstract LangGraph node callable for goal submission validation.

    Concrete subclasses implement :meth:`_validate` against their own rules. The
    ``__call__`` shell handles the common state contract (read ``submitted_goal``,
    emit ``submission_verified`` / ``submission_rejection_reason``, prepend rejection
    feedback to ``script_output`` so the next observation turn sees it).

    ``session_id`` is captured at build time so workbench-backed validators can tell
    the workbench which session's challenge to check the candidate against.
    """

    name: str = "submit_goal"
    label: str = "submit_goal"

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id

    def _validate(self, state: Dict[str, Any], submitted: str) -> SubmitValidationResult:
        raise NotImplementedError

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        candidate = state.get("submitted_goal")
        if candidate is None or not str(candidate).strip():
            return {
                "script_output": "[submit_goal] No ``submitted_goal`` in state.",
                "submitted_goal": None,
            }

        submitted = str(candidate).strip()
        # Deterministic placeholder guard, applied before any tool-specific
        # comparison and uniformly across all submit tools. Stops the
        # "submitted flag{example_flag}" false-win regardless of validator.
        if looks_like_placeholder(submitted):
            validation = SubmitValidationResult(
                accepted=False,
                reason=(
                    "submission looks like a placeholder/template value, not a flag "
                    "recovered from real tool output — perform the actual recovery first"
                ),
            )
        else:
            # Deterministic flag-shape guard (consolidated-experiment F1/F5/F6/W4):
            # reject values that are obviously not a flag — a filename, URL, address,
            # multi-sentence report prose, or the wrong wrapper — before any oracle
            # call. ``flag_format`` is optional (read from state when the caller
            # supplied it); when absent, only format-agnostic shape checks run.
            from core.tools.flag_shape import classify_flag_candidate

            shape = classify_flag_candidate(
                submitted, flag_format=state.get("flag_format")
            )
            if not shape.ok:
                validation = SubmitValidationResult(
                    accepted=False, reason=shape.reason
                )
            else:
                validation = self._validate(state, submitted)

        print(
            f"SUBMIT_GOAL_TOOL[{self.label}] candidate_len={len(submitted)} "
            f"accepted={validation.accepted} reason={validation.reason!r}",
            flush=True,
        )

        if validation.accepted:
            return {
                "submitted_goal": submitted,
                "submission_verified": True,
                "submission_rejection_reason": None,
                "command": None,
                "write_script": None,
                "write_script_language": None,
            }

        feedback = format_rejection_feedback(
            validation.reason or "rejected", submitted
        )
        prev = state.get("script_output") or ""
        return {
            "submitted_goal": None,
            "submission_verified": False,
            "submission_rejection_reason": validation.reason,
            "script_output": f"{feedback}\n\n{prev}".strip(),
            "command": None,
            "write_script": None,
            "write_script_language": None,
        }


class NYUCTFSubmitGoalTool(BaseSubmitGoalTool):
    """NYU CTF flag submission validator — delegates to the workbench.

    The agent holds no ``expected_flag`` and no benchmark/challenge identity. It
    POSTs the candidate (with the run's ``session_id``) to the workbench, which
    looks up the challenge bound to that session, resolves the ground-truth flag
    (mirroring ``nyuctf_agents`` ``CheckFlag``), compares, and returns accept/reject.
    Setting ``submission_verified=True`` ends the graph just like ``environment.solved
    = True`` ends the reference agent's loop. An unreachable workbench / unbound
    session is treated as "could not validate" (rejected, not a win).
    """

    label = "nyuctf_submit_goal"

    def _validate(self, state: Dict[str, Any], submitted: str) -> SubmitValidationResult:
        from infrastructure.benchmark_flag_client import validate_submission_remote

        session_id = str(state.get("session_id") or self.session_id or "")
        accepted, reason = validate_submission_remote(
            session_id=session_id,
            candidate=submitted,
        )
        if accepted:
            return SubmitValidationResult(accepted=True)
        return SubmitValidationResult(
            accepted=False,
            reason=reason or "flag rejected by workbench",
        )


class BaselineSubmitGoalTool(BaseSubmitGoalTool):
    """No-oracle fallback validator (the original single-agent baseline behaviour).

    Attached when no ground-truth submit tool is selected in the UI. There is no
    oracle to compare against, so — exactly as the original single-agent baseline
    did — we accept a value the agent genuinely recovered: one present verbatim in
    real tool output. A value with no supporting evidence is rejected (the rejection
    feedback is injected into ``script_output`` so the next observation turn sees it),
    which keeps the guard against ending the graph on a fabricated/guessed answer.
    """

    label = "baseline_submit_goal"

    def _validate(self, state: Dict[str, Any], submitted: str) -> SubmitValidationResult:
        evidence_text = str(state.get("script_output") or "")
        return validate_baseline_submission(
            submitted,
            evidence_text=evidence_text,
        )


# Backwards-compatible alias — older imports / references to the reject-everything
# tool now resolve to the evidence-based baseline validator.
NoSubmitGoalTool = BaselineSubmitGoalTool


# Registry of available submit tools by UI tool id. Add more by writing a
# ``BaseSubmitGoalTool`` subclass with the tool-specific validator and registering
# it here under the id the UI sends (e.g. ``"bandit_submit_goal"``).
SUBMIT_TOOL_REGISTRY: Dict[str, Type[BaseSubmitGoalTool]] = {
    "nyuctf_submit_goal": NYUCTFSubmitGoalTool,
    "baseline_submit_goal": BaselineSubmitGoalTool,
}


def build_submit_goal_tool(
    tools: Optional[List[str]],
    session_id: Optional[str] = None,
) -> BaseSubmitGoalTool:
    """Construct the submit-goal tool the UI selected for this workflow.

    ``tools`` is the explicit tool list the UI attached at init. The first id in it
    that names a registered submit tool wins. When none is present we fall back to
    :class:`BaselineSubmitGoalTool`: with no ground-truth oracle we accept a
    genuinely-recovered value (baseline behaviour) instead of rejecting every
    submission, while still blocking fabricated/guessed answers.
    """
    for tool_id in tools or []:
        cls = SUBMIT_TOOL_REGISTRY.get(str(tool_id).strip().lower())
        if cls is not None:
            return cls(session_id=session_id)
    return BaselineSubmitGoalTool(session_id=session_id)


# Backwards-compatible alias — older imports of ``SubmitGoalTool`` still work, now
# resolving to the workbench-validated NYU CheckFlag-style implementation. New
# callers should use :func:`build_submit_goal_tool` instead.
SubmitGoalTool = NYUCTFSubmitGoalTool
