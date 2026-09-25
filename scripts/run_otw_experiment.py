#!/usr/bin/env python3
"""Run an agent over an OverTheWire wargame (Bandit or Krypton), chained and in order,
with Langfuse tracing — the OTW analogue of ``run_langfuse_experiment.py``.

Why a dedicated runner: OTW levels are **chained**. The password the agent validates for
level N is the SSH login that unlocks level N+1, and OTW passwords are dynamic (they
rotate — they cannot be pre-seeded, only discovered). Levels must therefore run strictly
in order, and a level the agent cannot solve leaves the rest **locked** (unreachable).
This runner enforces that order and stops a game at the first unsolved/locked level, so
the result measures the *autonomous depth reached*. Bandit starts at level 0, Krypton at
level 1 (level 0 is the website's base64 hand-off, not an SSH challenge); terminal levels
are excluded by the benchmark catalog.

Validation is SSH-based and lives on the workbench: the agent submits a candidate
password via ``submit_goal``; the workbench logs into the next account with it over SSH —
a successful login means solved. There is no local flag file (and Unix permissions stop
the agent reading ahead, so unlike NYU CTF there is no flag-leak vector).

Deep planner (with escalation):
  PYTHONPATH=/app python3 scripts/run_otw_experiment.py --split bandit \\
      --run-name otw-bandit-deepplanner \\
      --registry deepagent_planner_pipeline_registry \\
      --model openai/gpt-4o-mini --escalation-model openai/gpt-5-mini

Single-agent baseline (no escalation):
  PYTHONPATH=/app python3 scripts/run_otw_experiment.py --split bandit \\
      --run-name otw-bandit-baseline --registry default_pipeline_registry \\
      --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import logging
import re
from typing import Any, Dict

import run_langfuse_experiment as R
from infrastructure.observability import langfuse_tracer

logger = logging.getLogger("otw")


def _level_key(cid: str) -> int:
    m = re.search(r"(\d+)$", str(cid))
    return int(m.group(1)) if m else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True, choices=["bandit", "krypton"], help="which OTW game")
    ap.add_argument("--run-name", required=True, help="experiment run name (groups traces in Langfuse via thread_id exp-<run>-<level>)")
    ap.add_argument("--registry", default=None, help="pipeline registry id (deepagent_planner_pipeline_registry | default_pipeline_registry)")
    ap.add_argument("--model", default=None, help="model override applied to every agent")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--escalation-model", default=None, help="'off' disables; default GENCYBER_ESCALATION_MODEL")
    ap.add_argument("--escalation-after-actions", type=int, default=None)
    ap.add_argument("--planner-model", default=None)
    ap.add_argument("--subagent-model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap number of levels")
    ap.add_argument("--challenge", action="append", dest="only", default=None, help="restrict to these level id(s), e.g. bandit3; repeatable")
    ap.add_argument("--dry-run", action="store_true", help="list the levels that would run, then exit")
    args = ap.parse_args()

    base = R._workbench_base()
    items = R.enumerate_items(base, "otw", args.split, limit=args.limit, only=args.only)
    if not items:
        logger.error("no OTW levels enumerated for split=%s", args.split)
        return 1
    # Strict level order is REQUIRED for chaining (level N unlocks N+1).
    items.sort(key=lambda it: _level_key(it["input"]["challenge_id"]))
    logger.info("enumerated %d %s level(s): %s", len(items), args.split,
                ", ".join(it["input"]["challenge_id"] for it in items))

    if args.dry_run:
        return 0

    if not langfuse_tracer.is_enabled():
        logger.error("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — aborting.")
        return 1
    client = langfuse_tracer.client()

    agent_models: Dict[str, str] = {}
    if args.planner_model:
        agent_models["planner"] = args.planner_model
    if args.subagent_model:
        agent_models["generative"] = args.subagent_model

    # Reuse the exact per-level machinery from the NYU CTF harness: materialize → run the
    # workflow → submit_goal (SSH-validated on the workbench) → log → teardown.
    task = R.make_task(
        base=base,
        registry_id=args.registry,
        model_name=args.model,
        provider=args.provider,
        run_name=args.run_name,
        start_services=False,  # OTW has no docker services to start/stop
        escalation_model=args.escalation_model,
        escalation_after_actions=args.escalation_after_actions,
        agent_models=agent_models or None,
    )

    solved = 0
    attempted = 0
    for it in items:
        cid = it["input"]["challenge_id"]
        sid = R._session_id(args.run_name, cid)
        # Chain gate: is this level unlocked yet? materialize resolves its entry password
        # from the prior level's validated answer; a "locked" status means the previous
        # level was not solved, so this and every later level are unreachable.
        mat = R._post(
            base,
            "/benchmarks/materialize",
            {"benchmark": "otw", "split": args.split, "challenge_id": cid, "session_id": sid},
        )
        if (mat or {}).get("status") == "locked":
            logger.info(
                "challenge=%s solved=False attempts=0 locked=1 "
                "(previous level unsolved — stopping %s chain here)",
                cid, args.split,
            )
            break
        attempted += 1
        out = task(item=it)
        if out.get("submission_verified"):
            solved += 1
        else:
            logger.info(
                "challenge=%s not solved — chain stops here (remaining %s levels unreachable)",
                cid, args.split,
            )
            break

    logger.info(
        "OTW %s complete: solved %d / %d attempted (autonomous depth reached).",
        args.split, solved, attempted,
    )
    try:
        client.flush()
    except Exception:  # pragma: no cover - best effort
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
