#!/usr/bin/env python3
"""Run a Langfuse experiment over CTF challenges — Part 3 of the Langfuse program.

For every challenge it materializes the challenge in the workbench, brings up its
services, runs the agent IN-PROCESS to completion, then tears the services down —
and Langfuse records each run as one experiment item (a full trace via the Part 1
tracing) scored by the Part 4 evaluators (``infrastructure.observability.langfuse_eval``).

Pass/fail ground truth is the **workbench oracle**: the agent submits via the
``nyuctf_submit_goal`` tool, the workbench checks the candidate against the flag
bound to the run's ``session_id`` (registered at materialize), and the run's
``submission_verified`` is what ``solved`` scores. The agent never holds the flag.

Run it inside the agent container (it imports the agent graph and needs the
workbench reachable + Langfuse credentials), e.g.::

    PYTHONPATH=src LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
        LANGFUSE_BASE_URL=http://langfuse-web:3000 \
        CHALLENGE_TOOLKIT_URL=http://gencyber-workbench:80 \
        python3 scripts/run_langfuse_experiment.py run \
            --split development --limit 5 \
            --registry deepagent_planner_pipeline_registry \
            --model openai/gpt-4o-mini --run-name planner-gpt4omini-dev5

``--model`` is the ONLY knob needed to switch models: pass an OpenRouter slug
(``openai/gpt-4o-mini``, ``qwen/qwen-2.5-coder-32b-instruct``) or a model served by
gencyber-Engine (``Qwen/Qwen2.5-1.5B-Instruct`` — any name in
``config/models/engine_models.yaml``). gencyber-Agent decides OpenRouter vs the local
engine from the name; ``--provider`` is only for forcing an unusual case.

Subcommands:
  * ``sync`` — enumerate challenges and upsert them as a Langfuse dataset.
  * ``run``  — run the experiment (syncs the dataset first unless ``--local``).

``--dry-run`` enumerates and prints the items without contacting Langfuse or
running any agent.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as `python3 scripts/run_langfuse_experiment.py` without PYTHONPATH=src.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Light imports only at module load — heavy agent modules (langchain/langgraph) are
# imported lazily inside _build_service so enumeration/dry-run work without them.
from infrastructure.observability import langfuse_tracer
from infrastructure.observability.langfuse_eval import (
    DEFAULT_ITEM_EVALUATORS,
    DEFAULT_RUN_EVALUATORS,
    make_reasoning_quality_evaluator,
)

logger = logging.getLogger("gencyber.experiment")

_SUBMIT_TOOL = "nyuctf_submit_goal"  # selects the workbench-oracle submit variant


# ---------------------------------------------------------------------------
# Workbench HTTP (mirrors infrastructure.benchmark_flag_client base resolution)
# ---------------------------------------------------------------------------
def _workbench_base() -> str:
    return (
        os.environ.get("CHALLENGE_TOOLKIT_URL")
        or os.environ.get("GENCYBER_WORKBENCH_URL")
        or "http://localhost:8080"
    ).rstrip("/")


_HTTP_TIMEOUT = int(os.getenv("EXPERIMENT_HTTP_TIMEOUT", "600"))


def _get(base: str, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{base}{path}?{qs}" if qs else f"{base}{path}"
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(base: str, path: str, payload: Dict[str, Any], *, timeout: int = _HTTP_TIMEOUT) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# urllib.parse is needed by _get; import here to keep the top tidy.
import urllib.parse  # noqa: E402


# ---------------------------------------------------------------------------
# Challenge enumeration -> experiment items
# ---------------------------------------------------------------------------
def enumerate_items(
    base: str, benchmark: str, split: str, *, limit: Optional[int], only: Optional[List[str]]
) -> List[Dict[str, Any]]:
    meta = _get(base, "/benchmarks/metadata", {"benchmark": benchmark, "split": split})
    block = (meta.get("splits") or {}).get(split) or {}
    items: List[Dict[str, Any]] = []
    for c in block.get("challenges") or []:
        cid = c.get("challenge_id")
        if not cid or (only and cid not in only):
            continue
        items.append(
            {
                "input": {
                    "benchmark": benchmark,
                    "split": split,
                    "challenge_id": cid,
                    "category": c.get("category"),
                    "name": c.get("challenge"),
                },
                "expected_output": None,
                "metadata": {
                    "category": c.get("category"),
                    "name": c.get("challenge"),
                    "year": c.get("year"),
                    "event": c.get("event"),
                },
            }
        )
    return items[:limit] if limit else items


def _item_input(item: Any) -> Dict[str, Any]:
    return item.input if hasattr(item, "input") else item["input"]


def _dataset_item_cid(item: Any) -> Optional[str]:
    """challenge_id of a Langfuse DatasetItem.

    Items are created with ``id=challenge_id`` and ``input.challenge_id`` (see
    ``sync_dataset``); prefer the input field and fall back to the item id.
    """
    inp = getattr(item, "input", None)
    if isinstance(inp, dict) and inp.get("challenge_id"):
        return inp["challenge_id"]
    return getattr(item, "id", None)


# ---------------------------------------------------------------------------
# In-process agent run
# ---------------------------------------------------------------------------
def _has_specialists(registry_id: Optional[str]) -> bool:
    """Does this registry build a planner with ctf-* specialist subagents?

    Drives the wording of the category hint (``_compose_question``). Only the planner
    registry has specialists to delegate to; the deep *generative* registry (M0/M1)
    and the baseline single-agent registry do not.
    """
    return "planner" in (registry_id or "").lower()


def _resolve_registry_path(registry_id: Optional[str]) -> Path:
    base = os.getenv("REGISTRY_PATH", "config/pipeline/default_pipeline_registry.yaml")
    if not registry_id:
        return Path(base)
    from application.pipeline.helpers import pipeline_helper as ph

    reg_dir = Path(base).resolve().parent
    return Path(ph.resolve_pipeline_registry_yaml(reg_dir, registry_id))


def _apply_model_override(
    rt: Dict[str, Any], *, model_name: str, provider: Optional[str]
) -> Dict[str, Any]:
    """Override every agent's model fields (mirrors the controller's helper)."""
    from application.pipeline.helpers import pipeline_helper as ph

    resolved = ph.resolve_model_path(model_name)
    new_mp, new_raw, new_pp = {}, {}, {}
    for agent, raw in rt["model_config_raw"].items():
        rc = dict(raw)
        rc["model_name"] = model_name
        rc["model_path"] = resolved
        if provider is not None:
            rc["provider"] = provider
        new_raw[agent] = rc
        new_mp[agent] = ph.build_model_params(rc)
    for agent, pp in rt["pipeline_params"].items():
        try:
            new_pp[agent] = pp.model_copy(update={"model_name": model_name})
        except Exception:
            new_pp[agent] = pp
    merged = dict(rt)
    merged.update(model_params=new_mp, model_config_raw=new_raw, pipeline_params=new_pp)
    return merged


def _apply_per_agent_models(rt: Dict[str, Any], agent_models: Dict[str, str]) -> Dict[str, Any]:
    """Override specific agents' model fields by registry name (others keep their config).

    Enables tiered runs (e.g. planner=27B, generative/subagent=8B) without editing YAML.
    Model routing (OpenRouter vs gencyber-Engine) is still decided by the name.
    """
    from application.pipeline.helpers import pipeline_helper as ph

    new_raw = dict(rt["model_config_raw"])
    new_mp = dict(rt["model_params"])
    new_pp = dict(rt["pipeline_params"])
    for agent, model in agent_models.items():
        if not model or agent not in rt["model_config_raw"]:
            continue
        rc = dict(rt["model_config_raw"][agent])
        rc["model_name"] = model
        rc["model_path"] = ph.resolve_model_path(model)
        new_raw[agent] = rc
        new_mp[agent] = ph.build_model_params(rc)
        try:
            new_pp[agent] = rt["pipeline_params"][agent].model_copy(update={"model_name": model})
        except Exception:
            new_pp[agent] = rt["pipeline_params"][agent]
    merged = dict(rt)
    merged.update(model_params=new_mp, model_config_raw=new_raw, pipeline_params=new_pp)
    return merged


def _build_service(
    registry_id: Optional[str],
    model_name: Optional[str],
    provider: Optional[str],
    session_id: str,
    escalation_model: Optional[str] = None,
    escalation_after_actions: Optional[int] = None,
    agent_models: Optional[Dict[str, str]] = None,
    challenge_category: Optional[str] = None,
):
    """Initialize a LangGraphService for one run (replicates the init endpoint)."""
    from application.langgraph.services.langgraph_service import LangGraphService
    from application.pipeline.services.pipeline_service import PipelineConfigService
    from core.helpers.terminal_session_client import ensure_terminal_session

    rt = PipelineConfigService(registry_path=str(_resolve_registry_path(registry_id))).load_runtime_config()
    if model_name:
        rt = _apply_model_override(rt, model_name=model_name, provider=provider)
    if agent_models:
        rt = _apply_per_agent_models(rt, agent_models)

    service = LangGraphService()
    resp = service.init_workflow(
        session_id=session_id,
        agent_definitions=rt["agent_definitions"],
        model_params=rt["model_params"],
        pipeline_params=rt["pipeline_params"],
        generation_configs=rt["generation_configs"],
        model_config_raw=rt["model_config_raw"],
        history_keys=rt["history_keys"],
        tools=[_SUBMIT_TOOL],
        graph_key=rt.get("graph"),
        # Per-run escalation target (name routes to engine/OpenRouter automatically).
        # None → falls back to the GENCYBER_ESCALATION_MODEL env default.
        escalation_model=escalation_model,
        escalation_after_actions=escalation_after_actions,
        # Scopes the monolith's skill access to this challenge's pack, so M sees the
        # same corpus one planner specialist would. The planner ignores it.
        challenge_category=challenge_category,
    )
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"init_workflow failed: {resp['error']}")
    # The init endpoint also provisions the workbench PTY; the oracle binding is by
    # session_id, so this must use the same session_id we materialized under.
    ensure_terminal_session(session_id)
    return service


def _compose_question(
    mat: Dict[str, Any],
    svc: Dict[str, Any],
    category: Optional[str] = None,
    *,
    has_specialists: bool = True,
) -> str:
    """Build the briefing from the workbench seed prompt + reachable endpoints.

    When the benchmark exposes the challenge ``category`` we append it as an explicit
    hint so the agent does not mis-classify the challenge (observed 36%
    mis-delegation, with a wasted ``ctf-osint`` triage step). Only the category is
    added — never the challenge/file name (prompt-hygiene rule). For non-benchmark use
    where no category is known, this is simply omitted.

    The hint's *wording* must match the topology, or the control is not matched. The
    planner is told to delegate; a monolith has nobody to delegate to, so telling it
    to would be an instruction it cannot follow — a silent handicap dressed up as
    parity. ``has_specialists=False`` swaps in the equivalent wording for a single
    agent: same category information, same emphasis, no delegation verb.
    """
    seed = (mat.get("seed_prompt") or "").strip()
    if not seed:
        chal = mat.get("challenge") or {}
        seed = (
            f"Solve this {chal.get('category', '')} CTF challenge: "
            f"{chal.get('name', '')}.\n{chal.get('description', '')}"
        ).strip()
    endpoints = (svc or {}).get("endpoints") or []
    if endpoints:
        def _fmt(e: Dict[str, Any]) -> str:
            return e.get("url") or f"{e.get('host')}:{e.get('port')}"

        lines = "\n".join(f"- {_fmt(e)}" for e in endpoints)
        seed += f"\n\nReachable challenge services:\n{lines}"
    cat = (category or "").strip()
    if cat:
        if has_specialists:
            seed += (
                f"\n\nChallenge category (routing hint): {cat}. Delegate to the "
                f"specialist whose remit matches this category first; only switch "
                f"categories if the evidence you gather clearly contradicts it."
            )
        else:
            seed += (
                f"\n\nChallenge category (routing hint): {cat}. Treat this as a {cat} "
                f"challenge and start from the techniques that category implies; only "
                f"change course if the evidence you gather clearly contradicts it."
            )
    return seed


def _reasoning_text(state: Dict[str, Any]) -> str:
    raw = state.get("generative_agent_response")
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            r = obj.get("reasoning")
            if isinstance(r, list):
                return "\n".join(str(x) for x in r)
            if isinstance(r, str):
                return r
        except Exception:
            pass
        return raw
    ev = state.get("execution_evidence")
    return ev if isinstance(ev, str) else ""


def _count_commands(state: Dict[str, Any]) -> Optional[int]:
    """Best-effort command count from the accumulated evidence transcript.

    Only the deep workflows accumulate ``execution_evidence`` (a ``$ cmd`` / output
    transcript); the baseline keeps commands in ephemeral chat history, so this
    returns None there — command-level detail still lives in the Langfuse trace.
    """
    ev = state.get("execution_evidence")
    if isinstance(ev, str) and ev:
        return sum(1 for line in ev.splitlines() if line.lstrip().startswith("$ "))
    return None


def _run_diagnostics(session_id: str) -> Dict[str, Any]:
    """Per-run diagnostics from the deep workflow (real attempt counts, delegation
    sequence, escalation). Best-effort: empty when the deep workflow isn't importable
    (e.g. baseline registry) so the harness still runs."""
    try:
        from application.langgraph.models.deep_generative_workflow import run_diagnostics

        return run_diagnostics(session_id)
    except Exception:
        return {}


def _extract_output(final_state: Any, session_id: str, svc: Dict[str, Any]) -> Dict[str, Any]:
    fs = final_state if isinstance(final_state, dict) else {}
    out = {
        "submission_verified": bool(fs.get("submission_verified")),
        "submitted_goal": fs.get("submitted_goal"),
        "submission_rejection_reason": fs.get("submission_rejection_reason"),
        "active_subagent": fs.get("active_subagent"),
        "active_subagent_task": fs.get("active_subagent_task"),
        "reasoning": _reasoning_text(fs),
        "num_commands": _count_commands(fs),
        "services_status": (svc or {}).get("status"),
        "error": fs.get("error"),
        "session_id": session_id,
    }
    # Merge trustworthy per-run diagnostics (submission_attempts, distinct_flags_tried,
    # placeholder_submissions, delegations, escalated) for the fixed evaluators.
    out.update(_run_diagnostics(session_id))
    return out


def _session_id(run_name: str, challenge_id: str) -> str:
    raw = f"exp-{run_name}-{challenge_id}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)[:120]


def make_task(
    *,
    base: str,
    registry_id: Optional[str],
    model_name: Optional[str],
    provider: Optional[str],
    run_name: str,
    start_services: bool,
    escalation_model: Optional[str] = None,
    escalation_after_actions: Optional[int] = None,
    agent_models: Optional[Dict[str, str]] = None,
):
    """Build the per-item task: materialize → start → run agent → stop."""

    # Shared across items in this run. Two provider-health failure modes abort the run
    # early so a systemic outage can't silently null dozens of challenges (a real
    # OpenRouter connection outage once wiped 34/57 this way — and later 178/200 —
    # each failing instantly with 0 attempts):
    #   * credits exhausted (HTTP 402) — every remaining item would 402 the same way;
    #   * repeated connection errors — the provider is unreachable. Each item first
    #     absorbs a blip itself: on a connection error it retries the SAME challenge
    #     with exponential backoff (``_CONN_RETRIES`` times) before giving up. Only a
    #     challenge that STILL connection-errors after all its retries counts toward
    #     the abort guard, which trips after ``_CONN_ABORT_AFTER`` such consecutive
    #     challenges (a single healthy item resets the streak). So aborting now means
    #     a genuinely sustained outage, not a brief hiccup.
    # Re-running the experiment when the provider is healthy naturally resumes the
    # unskipped items (session_id is deterministic per run-name+challenge).
    run_state = {"aborted": False, "abort_reason": "", "conn_fails": 0}
    _CONN_ABORT_AFTER = int(os.getenv("EXPERIMENT_CONN_ABORT_AFTER", "3"))
    # Per-challenge connection-error backoff retry: wait out a transient provider blip
    # instead of burning the challenge. delay = min(base * 2**attempt, max) seconds.
    _CONN_RETRIES = max(0, int(os.getenv("EXPERIMENT_CONN_RETRIES", "5")))
    _CONN_BACKOFF_BASE = float(os.getenv("EXPERIMENT_CONN_BACKOFF_BASE", "30"))
    _CONN_BACKOFF_MAX = float(os.getenv("EXPERIMENT_CONN_BACKOFF_MAX", "300"))

    def _is_credit_error(err: Any) -> bool:
        e = str(err or "").lower()
        return "insufficient credits" in e or "code: 402" in e or "error code: 402" in e

    def _is_connection_error(err: Any) -> bool:
        e = str(err or "").lower()
        return (
            "connection error" in e
            or "apiconnectionerror" in e
            or "connection reset" in e
            or "connection aborted" in e
            or "max retries exceeded" in e
            or "temporarily unavailable" in e
        )

    def task(*, item: Any, **_: Any) -> Dict[str, Any]:
        inp = _item_input(item)
        benchmark, split, cid = inp["benchmark"], inp["split"], inp["challenge_id"]
        session_id = _session_id(run_name, cid)

        if run_state["aborted"]:
            logger.warning("challenge=%s skipped: run aborted (%s)", cid, run_state["abort_reason"])
            return {
                "submission_verified": False,
                "skipped": True,
                "error": f"skipped: run aborted earlier ({run_state['abort_reason']})",
                "session_id": session_id,
            }

        mat = _post(
            base,
            "/benchmarks/materialize",
            {"benchmark": benchmark, "split": split, "challenge_id": cid, "session_id": session_id},
        )
        svc: Dict[str, Any] = {"status": "skipped"}
        project_name = None
        try:
            if start_services:
                svc = _post(
                    base,
                    "/benchmarks/start-challenge-services",
                    {
                        "benchmark": benchmark,
                        "split": split,
                        "challenge_id": cid,
                        "written_root": mat.get("written_root"),
                    },
                )
                project_name = svc.get("project_name")

            # Per-challenge backoff retry. A transient provider blip surfaces as a
            # connection error and fast-fails the item with ~0 work; rather than
            # counting it toward the abort guard immediately, wait it out and retry the
            # SAME challenge. A fresh service gives a fresh LLM client (clean
            # connection); the session_id is unchanged so the flag-oracle binding holds
            # and the graph resumes from its checkpoint. Credit (402) errors and any
            # non-connection result (a solve or a genuine failure) are final and break
            # out immediately — only connection errors are retried.
            out: Optional[Dict[str, Any]] = None
            for attempt in range(_CONN_RETRIES + 1):
                service = _build_service(
                    registry_id, model_name, provider, session_id,
                    escalation_model=escalation_model,
                    escalation_after_actions=escalation_after_actions,
                    agent_models=agent_models,
                    challenge_category=inp.get("category"),
                )
                try:
                    final_state = service.workflow.invoke(
                        query=_compose_question(
                            mat, svc,
                            category=inp.get("category"),
                            has_specialists=_has_specialists(registry_id),
                        )
                    )
                    out = _extract_output(final_state, session_id, svc)
                except Exception as exc:  # invoke raised instead of returning an error state
                    if not _is_connection_error(exc):
                        raise
                    out = {
                        "submission_verified": False,
                        "error": str(exc),
                        "session_id": session_id,
                    }
                err = out.get("error")
                if _is_credit_error(err) or not _is_connection_error(err):
                    break  # final: credits won't recover by waiting; a real result stands
                if attempt < _CONN_RETRIES:
                    delay = min(_CONN_BACKOFF_BASE * (2 ** attempt), _CONN_BACKOFF_MAX)
                    logger.warning(
                        "challenge=%s connection error (attempt %d/%d): %s — retrying in %.0fs",
                        cid, attempt + 1, _CONN_RETRIES + 1, str(err)[:80], delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "challenge=%s connection error persisted after %d attempts: %s",
                        cid, _CONN_RETRIES + 1, str(err)[:80],
                    )

            err = out.get("error")
            if _is_credit_error(err):
                run_state["aborted"] = True
                run_state["abort_reason"] = "LLM credits exhausted (HTTP 402)"
                logger.error(
                    "challenge=%s hit LLM credit exhaustion (HTTP 402); remaining "
                    "challenges will be skipped. Top up credits and re-run to resume.",
                    cid,
                )
            elif _is_connection_error(err):
                # Only reached after the item exhausted all its backoff retries, so this
                # is a sustained outage, not a blip.
                run_state["conn_fails"] += 1
                logger.error(
                    "challenge=%s connection error after %d retries (%d consecutive): %s",
                    cid,
                    _CONN_RETRIES,
                    run_state["conn_fails"],
                    str(err)[:80],
                )
                if run_state["conn_fails"] >= _CONN_ABORT_AFTER:
                    run_state["aborted"] = True
                    run_state["abort_reason"] = (
                        f"{run_state['conn_fails']} consecutive challenges connection-failed "
                        "after backoff retries (LLM provider appears down)"
                    )
                    logger.error(
                        "aborting run: %s. Remaining challenges will be skipped — "
                        "re-run when the provider is healthy to resume.",
                        run_state["abort_reason"],
                    )
            else:
                # A healthy item clears the streak so a brief blip never aborts the run.
                run_state["conn_fails"] = 0
            logger.info(
                "challenge=%s solved=%s attempts=%s",
                cid,
                out["submission_verified"],
                out.get("submission_attempts", 0),
            )
            return out
        finally:
            if project_name:
                try:
                    _post(
                        base,
                        "/benchmarks/stop-challenge-services",
                        {
                            "benchmark": benchmark,
                            "split": split,
                            "challenge_id": cid,
                            "project_name": project_name,
                        },
                    )
                except Exception as e:
                    logger.warning("stop-challenge-services failed for %s: %s", cid, e)

    return task


# ---------------------------------------------------------------------------
# Langfuse dataset sync
# ---------------------------------------------------------------------------
def sync_dataset(client: Any, name: str, items: List[Dict[str, Any]], benchmark: str, split: str) -> None:
    try:
        client.create_dataset(
            name=name,
            description=f"{benchmark} {split} CTF challenges (gencyber experiment harness)",
            metadata={"benchmark": benchmark, "split": split},
        )
    except Exception as e:
        logger.info("create_dataset(%s): %s (likely already exists)", name, e)
    for it in items:
        try:
            client.create_dataset_item(
                dataset_name=name,
                input=it["input"],
                expected_output=it["expected_output"],
                metadata=it["metadata"],
                id=it["input"]["challenge_id"],
            )
        except Exception as e:
            logger.warning("create_dataset_item(%s): %s", it["input"]["challenge_id"], e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--benchmark", default="nyuctf")
    p.add_argument("--split", default="development", choices=["development", "test", "bandit", "krypton"])
    p.add_argument("--limit", type=int, default=None, help="cap number of challenges")
    p.add_argument("--challenge", action="append", dest="only", help="restrict to these challenge_id(s); repeatable")
    p.add_argument("--dataset-name", default=None, help="default: gencyber-<benchmark>-<split>")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sync", help="upsert challenges as a Langfuse dataset")
    _add_common(ps)
    ps.add_argument("--dry-run", action="store_true", help="print items; do not contact Langfuse")

    pr = sub.add_parser("run", help="run the experiment over the challenges")
    _add_common(pr)
    pr.add_argument("--run-name", required=True, help="experiment run name (groups scores in Langfuse)")
    pr.add_argument("--registry", default=None, help="pipeline registry id (selects baseline vs deep planner)")
    pr.add_argument("--model", default=None, help="model override applied to every agent; OpenRouter slug or a gencyber-Engine model name (agent routes by name)")
    pr.add_argument("--provider", default=None, help="force provider (openrouter/openai/ollama); normally unnecessary — the model name decides")
    pr.add_argument("--escalation-model", default=None, help="per-run escalation target (model name; routes to engine/OpenRouter by name; 'off' disables). Default: GENCYBER_ESCALATION_MODEL env")
    pr.add_argument("--escalation-after-actions", type=int, default=None, help="actions-without-a-flag that trip a stall escalation (env default 12)")
    pr.add_argument("--planner-model", default=None, help="tiered runs: model for the planner/orchestrator agent (overrides its config)")
    pr.add_argument("--subagent-model", default=None, help="tiered runs: model for the generative/subagent (overrides its config)")
    pr.add_argument("--local", action="store_true", help="run off freshly-enumerated data, not a Langfuse dataset")
    pr.add_argument("--no-services", action="store_true", help="skip start/stop of challenge docker services")
    pr.add_argument("--judge", action="store_true", help="include the LLM reasoning-quality evaluator")
    pr.add_argument("--max-concurrency", type=int, default=1, help="parallel items (keep 1: shared workbench PTY/ports)")
    pr.add_argument("--dry-run", action="store_true", help="print items; do not contact Langfuse or run agents")

    args = ap.parse_args()
    base = _workbench_base()
    dataset_name = args.dataset_name or f"gencyber-{args.benchmark}-{args.split}"

    items = enumerate_items(base, args.benchmark, args.split, limit=args.limit, only=args.only)
    if not items:
        logger.error("no challenges enumerated for %s/%s", args.benchmark, args.split)
        return 1
    logger.info("enumerated %d challenge(s) from %s", len(items), base)

    if getattr(args, "dry_run", False):
        for it in items:
            print(json.dumps(it["input"]))
        print(f"\n{len(items)} items (dry-run; dataset='{dataset_name}').")
        return 0

    if not langfuse_tracer.is_enabled():
        logger.error("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — aborting.")
        return 1
    client = langfuse_tracer.client()
    if client is None:
        logger.error("Langfuse client unavailable (auth/SDK). Aborting.")
        return 1

    if args.cmd == "sync":
        sync_dataset(client, dataset_name, items, args.benchmark, args.split)
        print(f"Synced {len(items)} items into dataset '{dataset_name}'.")
        return 0

    # cmd == "run"
    evaluators = list(DEFAULT_ITEM_EVALUATORS)
    if args.judge or os.getenv("GENCYBER_EVAL_LLM_JUDGE"):
        evaluators.append(make_reasoning_quality_evaluator())
    agent_models: Dict[str, str] = {}
    if args.planner_model:
        agent_models["planner"] = args.planner_model
    if args.subagent_model:
        agent_models["generative"] = args.subagent_model

    task = make_task(
        base=base,
        registry_id=args.registry,
        model_name=args.model,
        provider=args.provider,
        run_name=args.run_name,
        start_services=not args.no_services,
        escalation_model=args.escalation_model,
        escalation_after_actions=args.escalation_after_actions,
        agent_models=agent_models or None,
    )
    run_metadata = {
        "model": args.model or "registry-default",
        "planner_model": args.planner_model or "config-default",
        "subagent_model": args.subagent_model or "config-default",
        "escalation_model": args.escalation_model or "env-default",
        "registry": args.registry or "default",
        "benchmark": args.benchmark,
        "split": args.split,
    }
    common = dict(
        name=args.run_name,
        description=f"gencyber {args.benchmark}/{args.split} — {run_metadata['registry']} / {run_metadata['model']}",
        task=task,
        evaluators=evaluators,
        run_evaluators=DEFAULT_RUN_EVALUATORS,
        max_concurrency=args.max_concurrency,
        metadata=run_metadata,
    )
    if args.local:
        result = client.run_experiment(data=items, **common)
    else:
        sync_dataset(client, dataset_name, items, args.benchmark, args.split)
        ds = client.get_dataset(dataset_name)
        if args.only:
            # A dataset-based ``run_experiment`` iterates ALL items in the dataset, so the
            # ``--challenge`` filter (which only shrank the enumerated ``items`` used for the
            # upsert above) would otherwise be silently ignored here — the whole dataset
            # would run. Replicate what ``Dataset.run_experiment`` does internally
            # (``client.run_experiment(data=dataset.items, _dataset_version=...)``) but over
            # the filtered subset, so the run stays a proper, comparable dataset run while
            # executing exactly the requested challenges.
            only_set = set(args.only)
            present = {_dataset_item_cid(it) for it in ds.items}
            missing = only_set - present
            if missing:
                logger.warning(
                    "--challenge id(s) not present in dataset '%s' (skipped): %s",
                    dataset_name, ", ".join(sorted(missing)),
                )
            selected = [it for it in ds.items if _dataset_item_cid(it) in only_set]
            if not selected:
                logger.error("no dataset items matched --challenge; aborting.")
                return 1
            logger.info(
                "running %d of %d dataset item(s) (filtered by --challenge)",
                len(selected), len(ds.items),
            )
            result = client.run_experiment(
                data=selected, _dataset_version=ds.version, **common
            )
        else:
            result = ds.run_experiment(**common)

    try:
        print(result.format())
    except Exception:
        print(result)
    client.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
