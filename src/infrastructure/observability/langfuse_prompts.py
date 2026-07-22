"""Langfuse prompt management — fetch managed prompt text with local-file fallback.

Langfuse is the source of truth for every prompt asset (base system prompts,
playbooks, snippets, context, skill packs + notes). At runtime we fetch the managed
version from Langfuse; if Langfuse is disabled, unreachable, or the prompt is missing,
we fall back to the on-disk file so the agent always runs.

Templating uses Langfuse ``{{double-brace}}`` variables:
  - **Templated** assets (base ``*_system.xml`` + ``context/*.xml``) carry our
    placeholders (``{{playbooks_section}}`` etc.) and are compiled with the SDK's
    ``prompt.compile(**vars)`` (offline: an equivalent local substituter).
  - **Leaf** assets (playbooks, snippets, skill packs + notes) are plain text and are
    fetched RAW — never compiled — so literal ``{{...}}`` payloads inside skill notes
    (e.g. SSTI ``{{7*7}}``) are preserved verbatim.

Prompt names mirror the on-disk layout: a file at ``config/prompts/<rel>.<ext>`` is
stored as ``gencyber/<rel>`` (posix, no extension), so the mapping is reversible and
the seed script and runtime agree without a separate manifest.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from infrastructure.observability import langfuse_tracer

logger = logging.getLogger(__name__)

NAME_PREFIX = "gencyber"
PROMPT_LABEL = os.getenv("LANGFUSE_PROMPT_LABEL", "production")

# A Langfuse variable is ``{{ word }}`` (letters/digits/underscore). JSON braces
# (``{{"k": 1}}``) and SSTI payloads (``{{7*7}}``) don't match this and are left as-is.
_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def compile_local(text: str, variables: Optional[Dict[str, Any]]) -> str:
    """Offline equivalent of Langfuse ``prompt.compile`` for ``{{var}}`` templates.

    Substitutes only the provided keys; any other ``{{...}}`` is left untouched,
    matching Langfuse's behaviour for unbound variables.
    """
    if not variables:
        return text

    def _repl(m: "re.Match[str]") -> str:
        key = m.group(1)
        return str(variables[key]) if key in variables else m.group(0)

    return _VAR_RE.sub(_repl, text)


def to_double_brace(text: str, names: tuple[str, ...]) -> str:
    """Rewrite the named ``{single}`` placeholders to ``{{double}}`` (idempotent).

    Only the exact placeholder names are converted; everything else — including
    literal ``flag{...}`` and any pre-existing ``{{...}}`` — is left untouched. Used by
    the seed script so the Langfuse copy of a templated asset uses Langfuse syntax.
    """
    out = text
    for name in names:
        out = out.replace("{{" + name + "}}", "{" + name + "}")  # de-dupe if already run
        out = out.replace("{" + name + "}", "{{" + name + "}}")
    return out


def prompt_name_for(path: Path) -> str:
    """Map a prompt-asset file path to its Langfuse prompt name.

    ``.../config/prompts/playbooks/generative_agent/foo.xml`` →
    ``gencyber/playbooks/generative_agent/foo``.
    """
    parts = Path(path).resolve().parts
    if "prompts" in parts:
        idx = len(parts) - 1 - parts[::-1].index("prompts")
        rel = Path(*parts[idx + 1:])
    else:
        rel = Path(Path(path).name)
    return f"{NAME_PREFIX}/" + rel.with_suffix("").as_posix()


def get_managed_text(
    *,
    name: str,
    fallback_text: str,
    variables: Optional[Dict[str, Any]] = None,
    templated: bool = False,
) -> str:
    """Return the managed prompt text for ``name``, falling back to ``fallback_text``.

    Leaf assets (``templated=False``) are returned raw. Templated assets are compiled
    with ``variables``. Any Langfuse error (disabled, unreachable, missing prompt)
    degrades to the local fallback, compiled locally when templated.
    """
    c = langfuse_tracer.client() if langfuse_tracer.is_enabled() else None
    if c is not None:
        try:
            prompt = c.get_prompt(name, label=PROMPT_LABEL)
            if templated:
                return prompt.compile(**(variables or {}))
            return prompt.prompt
        except Exception:
            logger.debug("Langfuse get_prompt(%s) failed; using local file", name, exc_info=True)
    return compile_local(fallback_text, variables) if templated else fallback_text


def managed_text_for_file(
    path: Path,
    *,
    variables: Optional[Dict[str, Any]] = None,
    templated: bool = False,
) -> str:
    """Convenience: resolve a prompt asset by file path (name derived from the path)."""
    try:
        local = Path(path).read_text(encoding="utf-8").strip() if Path(path).exists() else ""
    except Exception:
        logger.warning("Could not read prompt asset %s", path, exc_info=True)
        local = ""
    return get_managed_text(
        name=prompt_name_for(path),
        fallback_text=local,
        variables=variables,
        templated=templated,
    )
