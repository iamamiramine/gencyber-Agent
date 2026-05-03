"""
Centralized system prompt loading and formatting for all agents.
Loads base prompts from {base_path}/prompts, optionally injects playbooks, snippets, and context
from {base_path}/prompts/playbooks, snippets, and context based on flags.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default root for prompts and prompt assets (relative to cwd when the app runs)
DEFAULT_BASE_PATH = "config"


def _prompts_dir(base_path: str) -> Path:
    return Path(base_path) / "prompts"


def _read_file(path: Path, default: str = "") -> str:
    """Read file content or return default if missing."""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
    return default


def _replace_named_placeholders(template: str, values: Dict[str, Any]) -> str:
    """
    Substitute only ``{key}`` tokens for keys present in ``values``.

    Unlike ``str.format``, arbitrary literal braces (e.g. ``flag{...}`` in examples) are left
    untouched. Keys are applied longest-first so ``{shell_context_extra}`` wins over ``{shell_context}``.
    """
    out = template
    for k, v in sorted(values.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        if not isinstance(k, str):
            continue
        token = "{" + k + "}"
        if token in out:
            out = out.replace(token, "" if v is None else str(v))
    return out


def load_playbooks_section(base_path: str, subdir: str = "playbooks") -> str:
    """Load and concatenate all XML files from {base_path}/prompts/playbooks (e.g. linux_playbook.xml)."""
    root = _prompts_dir(base_path) / subdir
    if not root.exists():
        return ""
    parts = []
    for f in sorted(root.glob("*.xml")):
        content = _read_file(f)
        if content:
            parts.append(content)
    return "\n\n".join(parts) if parts else ""


def load_snippets_section(base_path: str, subdir: str = "snippets") -> str:
    """Load and concatenate all XML files from {base_path}/prompts/snippets (e.g. linux_snippets.xml)."""
    root = _prompts_dir(base_path) / subdir
    if not root.exists():
        return ""
    parts = []
    for f in sorted(root.glob("*.xml")):
        content = _read_file(f)
        if content:
            parts.append(content)
    return "\n\n".join(parts) if parts else ""


def load_context_section(
    base_path: str,
    subdir: str = "context",
    context_vars: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Load context template from {base_path}/prompts/context (e.g. linux_context.xml),
    then format with context_vars (e.g. {"shell_context": "..."}).
    """
    root = _prompts_dir(base_path) / subdir
    if not root.exists():
        return ""
    parts = []
    for f in sorted(root.glob("*.xml")):
        content = _read_file(f)
        if content:
            if context_vars:
                content = _replace_named_placeholders(content, context_vars)
            parts.append(content)
    return "\n\n".join(parts) if parts else ""


def load_system_prompt_for_agent(
    agent_name: str,
    base_path: str = DEFAULT_BASE_PATH,
    *,
    load_snippets: bool = False,
    load_playbooks: bool = False,
    load_context: bool = False,
    context_vars: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Load the system prompt for a given agent from {base_path}/prompts/{agent_name}_system.xml.
    Optionally inject playbooks, snippets, and context from under {base_path}/prompts/ when flags are True.
    Placeholders in the base prompt: {playbooks_section}, {snippets_section}, {context_section}.
    If a section is not loaded, its placeholder is replaced with an empty string.
    """
    prompts_dir = _prompts_dir(base_path)
    prompt_file = prompts_dir / f"{agent_name}_system.xml"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    base_content = prompt_file.read_text(encoding="utf-8")

    # Build substitution dict for optional sections
    context_vars = context_vars or {}
    subs: Dict[str, str] = {
        "playbooks_section": load_playbooks_section(base_path) if load_playbooks else "",
        "snippets_section": load_snippets_section(base_path) if load_snippets else "",
        "context_section": load_context_section(base_path, context_vars=context_vars) if load_context else "",
    }

    # If base prompt uses old-style {shell_context} only (no section placeholders), support that too
    if "playbooks_section" not in base_content and "snippets_section" not in base_content and "context_section" not in base_content:
        if context_vars:
            return _replace_named_placeholders(base_content, context_vars)
        return base_content

    return _replace_named_placeholders(base_content, subs)
