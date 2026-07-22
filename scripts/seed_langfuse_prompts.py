#!/usr/bin/env python3
"""Seed Langfuse with every prompt asset gencyber-Agent fetches at runtime.

Each asset is pushed under the name scheme ``gencyber/<relpath-without-ext>`` (see
:func:`infrastructure.observability.langfuse_prompts.prompt_name_for`) so the
runtime's ``get_prompt(name, label="production")`` resolves the same files it would
otherwise read from disk. Run once after standing up a self-hosted Langfuse instance,
and re-run to publish prompt edits (each run creates a new version under the label)::

    PYTHONPATH=src LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
        LANGFUSE_BASE_URL=http://localhost:3000 \
        python3 scripts/seed_langfuse_prompts.py [--base-path config] [--dry-run]

Classification mirrors the runtime loaders exactly:
  * **Templated** (compiled with ``{{vars}}`` at runtime): the top-level
    ``*_system.xml`` base prompts and ``context/*.xml``. Placeholder names are
    normalized to the ``{{double-brace}}`` form Langfuse ``.compile()`` expects.
  * **Leaf** (fetched raw): playbooks, snippets, and skill notes. ``SKILL.md`` is
    pushed frontmatter-stripped — the runtime only ever uses its body, and the
    frontmatter ``description`` is parsed locally for subagent routing.

Assets the runtime never fetches (``templates/``, ``skills/scripts/``) are skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python3 scripts/seed_langfuse_prompts.py` without PYTHONPATH=src.
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infrastructure.observability import langfuse_tracer
from infrastructure.observability.langfuse_prompts import (
    PROMPT_LABEL,
    prompt_name_for,
    to_double_brace,
)


def _strip_frontmatter(text: str) -> str:
    """Return the markdown body with a leading ``---`` YAML frontmatter block removed.

    Mirrors ``skills_helper._split_frontmatter`` so the seeded SKILL.md body matches
    the runtime's local fallback byte-for-byte. Kept inline (not imported) so this
    script has no langchain/langgraph dependency and runs anywhere.
    """
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return text
    return "\n".join(lines[end + 1:]).strip()

# Every placeholder a templated asset may carry. ``compile()`` leaves any unbound
# ``{{...}}`` (literal flag-format examples, JSON, SSTI) intact, so over-listing is
# safe; this just guarantees these names are in ``{{double-brace}}`` form.
TEMPLATE_VARS = (
    "playbooks_section",
    "snippets_section",
    "context_section",
    "shell_context",
)

# Subtrees under prompts/ the runtime never fetches as managed text.
SKIP_DIRS = {"templates", "scripts"}


def classify(path: Path, prompts_root: Path):
    """Return ``(kind, text)`` for a seedable asset, or ``None`` to skip it.

    ``kind`` is ``"templated"`` (carries ``{{var}}`` placeholders, compiled at
    runtime) or ``"leaf"`` (fetched raw). ``text`` is the exact payload to push.
    """
    rel = path.relative_to(prompts_root)
    parts = rel.parts
    if any(p in SKIP_DIRS for p in parts):
        return None
    suffix = path.suffix.lower()

    # Templated base system prompts live at the prompts root as ``*_system.xml``.
    if len(parts) == 1 and suffix == ".xml" and path.stem.endswith("_system"):
        return ("templated", path.read_text(encoding="utf-8"))
    # Templated context templates (e.g. linux_context.xml carries {{shell_context}}).
    if parts[0] == "context" and suffix == ".xml":
        return ("templated", path.read_text(encoding="utf-8"))
    # Leaf playbooks / snippets (XML, fetched raw and concatenated).
    if parts[0] in ("playbooks", "snippets") and suffix == ".xml":
        return ("leaf", path.read_text(encoding="utf-8"))
    # Leaf skill assets (markdown). SKILL.md is served frontmatter-stripped.
    if parts[0] == "skills" and suffix == ".md":
        raw = path.read_text(encoding="utf-8")
        if path.name.lower() == "skill.md":
            return ("leaf", _strip_frontmatter(raw))
        return ("leaf", raw)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-path",
        default="config",
        help="root holding prompts/ (default: config)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="classify and print what would be seeded without contacting Langfuse",
    )
    args = ap.parse_args()

    prompts_root = Path(args.base_path) / "prompts"
    if not prompts_root.is_dir():
        print(f"prompts root not found: {prompts_root}", file=sys.stderr)
        return 1

    client = None
    if not args.dry_run:
        if not langfuse_tracer.is_enabled():
            print(
                "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — aborting "
                "(use --dry-run to preview).",
                file=sys.stderr,
            )
            return 1
        client = langfuse_tracer.client()
        if client is None:
            print("Langfuse client unavailable (auth/SDK). Aborting.", file=sys.stderr)
            return 1

    seeded = skipped = 0
    for path in sorted(prompts_root.rglob("*")):
        if not path.is_file():
            continue
        result = classify(path, prompts_root)
        if result is None:
            skipped += 1
            continue
        kind, text = result
        if kind == "templated":
            text = to_double_brace(text, TEMPLATE_VARS)
        name = prompt_name_for(path)
        if args.dry_run:
            print(f"[dry-run] {kind:9} {name}  ({len(text)} chars)")
            seeded += 1
            continue
        client.create_prompt(
            name=name,
            type="text",
            prompt=text,
            labels=[PROMPT_LABEL],
        )
        print(f"seeded {kind:9} {name}")
        seeded += 1

    verb = "previewed" if args.dry_run else "seeded"
    print(f"\nDone. {seeded} prompts {verb}, {skipped} files skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
