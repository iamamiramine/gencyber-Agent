"""Skill store and loader for the reasoning agent's progressive-disclosure skills.

This is the domain-knowledge backbone for the ReasoningGen workflow. It implements
the *progressive disclosure* pattern (a.k.a. "Agent Skills"): instead of stuffing
every technique playbook into the system prompt, the reasoning agent sees only a
lightweight **catalog** (Tier 1) and pulls in full instructions (Tier 2) or a
specific supporting file (Tier 3) on demand via tool nodes.

A *skill* is a directory under the skills root containing:

- ``SKILL.md`` (required) — YAML frontmatter (``name``, ``description``, ``tags``…)
  plus a markdown body of instructions for the LLM.
- Optional supporting ``.md`` files referenced by SKILL.md.

The :class:`SkillStore`:

1. **Discovery** — ``scan()`` walks the tree for ``SKILL.md`` files and parses only
   the YAML frontmatter (fast), so the catalog can be built without reading every
   body.
2. **Lazy content loading** — ``load(name)`` reads + caches the full body only when
   the reasoning agent requests it.
3. **On-demand file reading** — ``read_supporting_file(name, filename)`` reads a
   single supporting file, with directory-traversal protection.

The module exposes a cached singleton via :func:`get_skill_store` pointed at
``config/prompts/skills`` (override with the ``GENCYBER_SKILLS_DIR`` env var).

This module is intentionally dependency-light (only PyYAML) so it can be imported
and unit-tested without langchain / langgraph installed.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Split a SKILL.md into YAML frontmatter (group 1) and markdown body (group 2).
# re.DOTALL lets the capture groups span multiple lines.
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class SkillMetadata:
    """Lightweight metadata parsed from a SKILL.md's YAML frontmatter."""

    name: str
    description: str
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: Optional[Path] = None) -> "SkillMetadata":
        tags = data.get("tags", []) or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        return cls(
            name=str(data.get("name", "unknown")),
            description=str(data.get("description", "")),
            version=str(data.get("version", "1.0")),
            tags=list(tags),
            path=path,
        )


@dataclass
class ParsedSkill:
    """A fully parsed skill — metadata plus the markdown body (no frontmatter)."""

    metadata: SkillMetadata
    content: str


def _parse_skill_content(text: str, file_path: Optional[Path] = None) -> ParsedSkill:
    match = FRONTMATTER_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"Invalid skill file (missing YAML frontmatter): {file_path}")
    frontmatter_yaml, body = match.groups()
    try:
        data = yaml.safe_load(frontmatter_yaml) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in frontmatter: {e}") from e
    metadata = SkillMetadata.from_dict(data, path=file_path)
    return ParsedSkill(metadata=metadata, content=body.strip())


def parse_skill_file(file_path: Path) -> ParsedSkill:
    if not file_path.exists():
        raise FileNotFoundError(f"Skill file not found: {file_path}")
    return _parse_skill_content(file_path.read_text(encoding="utf-8"), file_path)


def parse_metadata_only(file_path: Path) -> SkillMetadata:
    if not file_path.exists():
        raise FileNotFoundError(f"Skill file not found: {file_path}")
    match = FRONTMATTER_PATTERN.match(file_path.read_text(encoding="utf-8").strip())
    if not match:
        raise ValueError(f"Invalid skill file (missing YAML frontmatter): {file_path}")
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in frontmatter: {e}") from e
    return SkillMetadata.from_dict(data, path=file_path)


def read_skill_supporting_file(skill_dir: Path, filename: str) -> str:
    """Read a supporting file from a skill folder, blocking directory traversal.

    The filename may include subdirectories (``resources/guides/x.md``) but must not
    escape ``skill_dir`` — the LLM controls this argument, so the path is validated.
    """
    if ".." in filename or filename.startswith("/"):
        raise ValueError(f"Invalid filename: {filename}")
    file_path = (skill_dir / filename).resolve()
    if not str(file_path).startswith(str(skill_dir.resolve())):
        raise ValueError(f"Invalid filename: {filename}")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    return file_path.read_text(encoding="utf-8")


class SkillStore:
    """Two-level cache over a skills directory: metadata (catalog) + full content."""

    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)
        self._metadata_cache: Dict[str, SkillMetadata] = {}
        self._content_cache: Dict[str, ParsedSkill] = {}
        self._scanned = False

    def scan(self) -> Dict[str, SkillMetadata]:
        """Discover skills, parsing only YAML frontmatter (fast). Idempotent."""
        if self._scanned and self._metadata_cache:
            return self._metadata_cache
        self._metadata_cache.clear()
        self._content_cache.clear()
        if not self.skills_dir.exists():
            logger.warning("Skills directory does not exist: %s", self.skills_dir)
            self._scanned = True
            return self._metadata_cache
        for skill_file in sorted(self.skills_dir.rglob("SKILL.md")):
            try:
                metadata = parse_metadata_only(skill_file)
                skill_name = skill_file.parent.name
                if metadata.name and metadata.name != "unknown":
                    skill_name = metadata.name
                metadata.path = skill_file
                self._metadata_cache[skill_name] = metadata
            except Exception as e:  # noqa: BLE001 — one bad skill must not break scan
                logger.warning("Failed to parse skill at %s: %s", skill_file, e)
        self._scanned = True
        logger.info("Scanned %d skills from %s", len(self._metadata_cache), self.skills_dir)
        return self._metadata_cache

    def load(self, skill_name: str) -> Optional[ParsedSkill]:
        """Lazy-load + cache the full SKILL.md body. ``None`` if the skill is unknown."""
        if skill_name in self._content_cache:
            return self._content_cache[skill_name]
        if not self._scanned:
            self.scan()
        metadata = self._metadata_cache.get(skill_name)
        if not metadata or not metadata.path:
            logger.warning("Skill not found: %s", skill_name)
            return None
        try:
            parsed = parse_skill_file(metadata.path)
            self._content_cache[skill_name] = parsed
            return parsed
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load skill %s: %s", skill_name, e)
            return None

    def read_supporting_file(self, skill_name: str, filename: str) -> str:
        if not self._scanned:
            self.scan()
        metadata = self._metadata_cache.get(skill_name)
        if not metadata or not metadata.path:
            raise ValueError(f"Skill not found: {skill_name}")
        return read_skill_supporting_file(metadata.path.parent, filename)

    def list_supporting_files(self, skill_name: str) -> List[str]:
        """List supporting ``.md`` files (relative paths) for a skill, recursively."""
        if not self._scanned:
            self.scan()
        metadata = self._metadata_cache.get(skill_name)
        if not metadata or not metadata.path:
            return []
        skill_dir = metadata.path.parent
        return sorted(
            str(f.relative_to(skill_dir))
            for f in skill_dir.rglob("*.md")
            if f.is_file() and f.name != "SKILL.md"
        )

    def get_skill_names(self) -> List[str]:
        if not self._scanned:
            self.scan()
        return list(self._metadata_cache.keys())

    def get_skill_catalog(self) -> str:
        """Render the Tier-1 XML catalog injected into the reasoning agent's prompt."""
        if not self._scanned:
            self.scan()
        if not self._metadata_cache:
            return "No skills available."
        lines: List[str] = []
        for name, metadata in sorted(self._metadata_cache.items()):
            lines.append("<skill>")
            lines.append(f"  <name>{name}</name>")
            lines.append(f"  <description>{metadata.description}</description>")
            if metadata.tags:
                lines.append(f"  <tags>{', '.join(metadata.tags)}</tags>")
            files = self.list_supporting_files(name)
            if files:
                lines.append(f"  <supporting_files>{', '.join(files)}</supporting_files>")
            lines.append("</skill>")
        return "\n".join(lines)

    def invalidate(self) -> None:
        """Force a rescan on next access (useful when skills are edited at runtime)."""
        self._metadata_cache.clear()
        self._content_cache.clear()
        self._scanned = False


def _default_skills_dir() -> Path:
    """Resolve the default skills root: env override, else ``config/prompts/skills``.

    The agent runs with the repo root as cwd (configs use ``./config/`` paths), so the
    relative default resolves correctly. We also fall back to a path anchored on this
    file's location so imports outside that cwd still find the bundled skills.
    """
    env = os.environ.get("GENCYBER_SKILLS_DIR", "").strip()
    if env:
        return Path(env)
    cwd_default = Path("config/prompts/skills")
    if cwd_default.exists():
        return cwd_default
    # src/core/helpers/skill_store.py → repo root is three parents up from src/.
    anchored = Path(__file__).resolve().parents[3] / "config" / "prompts" / "skills"
    return anchored


_STORE: Optional[SkillStore] = None


def get_skill_store() -> SkillStore:
    """Return the process-wide cached :class:`SkillStore`, scanning on first use."""
    global _STORE
    if _STORE is None:
        store = SkillStore(_default_skills_dir())
        store.scan()
        _STORE = store
    return _STORE
