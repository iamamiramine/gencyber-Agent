"""
Centralized system prompt loading and formatting for all agents.
Loads base prompts from {base_path}/prompts, optionally injects playbooks, snippets, and context
from {base_path}/prompts/playbooks, snippets, and context based on flags.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from infrastructure.observability.langfuse_prompts import managed_text_for_file

logger = logging.getLogger(__name__)

# Default root for prompts and prompt assets (relative to cwd when the app runs)
DEFAULT_BASE_PATH = "config"


def _prompts_dir(base_path: str) -> Path:
    return Path(base_path) / "prompts"


def _load_xml_dir(root: Path) -> list[str]:
    """Load every ``*.xml`` file in ``root`` (non-recursive) as managed prompt text.

    Each file resolves to its Langfuse-managed version (leaf asset, fetched raw) and
    falls back to the on-disk content when Langfuse is disabled/unreachable.
    """
    if not root.exists() or not root.is_dir():
        return []
    parts: list[str] = []
    for f in sorted(root.glob("*.xml")):
        content = managed_text_for_file(f)
        if content:
            parts.append(content)
    return parts


# Some agents reuse another agent's prompt-asset pack instead of duplicating the XML
# files. The DeepAgents generative worker (prompt_key ``deep_generative_agent``) shares
# the baseline generative agent's playbooks/snippets — crucially the EnIGMA interactive-
# tools (IAT) guide — so the planner's specialist subagents learn the
# ``debug_start`` / ``connect_start`` interfaces instead of reaching for raw ``gdb`` /
# ``nc`` (which open their own REPL and wedge the workbench sentinel PTY).
_AGENT_ASSET_ALIASES: Dict[str, str] = {
    "deep_generative_agent": "generative_agent",
}


def _load_agent_assets(root: Path, agent_name: Optional[str]) -> list[str]:
    """Load shared (``root``) assets, then this agent's per-agent assets, then any
    aliased agent's assets (see :data:`_AGENT_ASSET_ALIASES`).

    The alias lets an agent inherit another agent's asset directory without copying
    files, keeping a single source of truth for shared guidance.

    Directories are de-duplicated by resolved path, so when a per-agent directory is
    itself a symlink to the aliased directory (an alternative, filesystem-based way to
    share a pack) the same files are not loaded twice.
    """
    seen: set = set()
    parts: list[str] = []

    def _add(directory: Path) -> None:
        try:
            key = directory.resolve()
        except OSError:
            key = directory
        if key in seen:
            return
        seen.add(key)
        parts.extend(_load_xml_dir(directory))

    _add(root)
    if agent_name:
        _add(root / agent_name)
        alias = _AGENT_ASSET_ALIASES.get(agent_name)
        if alias and alias != agent_name:
            _add(root / alias)
    return parts


def load_playbooks_section(
    base_path: str,
    *,
    agent_name: Optional[str] = None,
    subdir: str = "playbooks",
) -> str:
    """Load playbook XML files for this agent.

    Files are sourced from these locations and concatenated in order:

      1. ``{base_path}/prompts/{subdir}/*.xml`` — **shared** playbooks loaded by every
         agent (e.g. ``linux_playbook.xml``). Keeps backwards compatibility with the
         original flat layout.
      2. ``{base_path}/prompts/{subdir}/{agent_name}/*.xml`` — **per-agent** playbooks
         that only this agent sees. Lets domain knowledge / examples / command catalogs
         live with the agent that consumes them, instead of being baked into the system
         prompt.
      3. ``{base_path}/prompts/{subdir}/{alias}/*.xml`` — playbooks inherited from an
         aliased agent (see :data:`_AGENT_ASSET_ALIASES`), so e.g. the DeepAgents
         generative worker reuses the baseline generative agent's pack.
    """
    root = _prompts_dir(base_path) / subdir
    parts = _load_agent_assets(root, agent_name)
    return "\n\n".join(parts) if parts else ""


def load_snippets_section(
    base_path: str,
    *,
    agent_name: Optional[str] = None,
    subdir: str = "snippets",
) -> str:
    """Load snippet XML files for this agent.

    Same shared-then-per-agent(-then-alias) layering as :func:`load_playbooks_section`.
    Shared files at the root of ``{subdir}``; per-agent files under
    ``{subdir}/{agent_name}/``; inherited files under ``{subdir}/{alias}/``.
    """
    root = _prompts_dir(base_path) / subdir
    parts = _load_agent_assets(root, agent_name)
    return "\n\n".join(parts) if parts else ""


def load_context_section(
    base_path: str,
    subdir: str = "context",
    context_vars: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Load context template from {base_path}/prompts/context (e.g. linux_context.xml),
    then format with context_vars (e.g. {"shell_context": "..."}).

    Context files are **templated** assets: they carry ``{{shell_context}}`` style
    placeholders, so each resolves to its Langfuse-managed version compiled with
    ``context_vars`` (offline: the local ``{{var}}`` substituter), falling back to the
    on-disk file when Langfuse is disabled/unreachable.
    """
    root = _prompts_dir(base_path) / subdir
    if not root.exists():
        return ""
    parts = []
    for f in sorted(root.glob("*.xml")):
        content = managed_text_for_file(f, variables=context_vars, templated=True)
        if content:
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
    Placeholders in the base prompt: {{playbooks_section}}, {{snippets_section}}, {{context_section}}.
    If a section is not loaded, its placeholder is replaced with an empty string.

    The base prompt is a **templated** asset: it resolves to its Langfuse-managed
    version compiled with the section/context variables (offline: the local ``{{var}}``
    substituter), falling back to the on-disk file when Langfuse is disabled/unreachable.
    Section and context vars are passed together so the same call handles both the new
    ``{{*_section}}`` placeholders and any legacy ``{{shell_context}}`` style vars.
    """
    prompts_dir = _prompts_dir(base_path)
    prompt_file = prompts_dir / f"{agent_name}_system.xml"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    # Build substitution dict for optional sections.
    # ``agent_name`` is the prompt_key (e.g. "planner_agent", "recon_agent_adaptive").
    # When set, playbooks_section / snippets_section also load files from the matching
    # subdirectory (e.g. playbooks/planner_agent/*.xml) so per-agent domain knowledge
    # stays out of the general system prompt.
    context_vars = context_vars or {}
    subs: Dict[str, str] = {
        "playbooks_section": (
            load_playbooks_section(base_path, agent_name=agent_name)
            if load_playbooks
            else ""
        ),
        "snippets_section": (
            load_snippets_section(base_path, agent_name=agent_name)
            if load_snippets
            else ""
        ),
        "context_section": (
            load_context_section(base_path, context_vars=context_vars)
            if load_context
            else ""
        ),
    }

    # Compile the base prompt with section + context vars in one pass. Unbound
    # ``{{...}}`` (e.g. literal flag-format examples / JSON) are left intact by both
    # Langfuse ``.compile()`` and the local substituter.
    return managed_text_for_file(
        prompt_file,
        variables={**context_vars, **subs},
        templated=True,
    )
