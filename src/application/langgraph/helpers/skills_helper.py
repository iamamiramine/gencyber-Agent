"""Discover CTF skill packs and turn them into deep-agent subagent specs.

Each attack category lives under ``{base_path}/prompts/skills/ctf-<category>/`` as a
self-contained *skill pack*:

  - ``SKILL.md`` — YAML frontmatter (``name`` + ``description``) followed by a base
    prompt. The ``description`` says *when to use / not use* this category; we surface
    it verbatim as the subagent's description so the planner's ``task`` tool can route
    to the right specialist.
  - one ``*.md`` *note* per sub-topic — deeper technique references the specialist
    loads ON DEMAND (lazy) via the ``read_skill_note`` tool, instead of bloating every
    prompt with every technique.

Two skill packs are **shared guidance, not subagents**: ``solve-challenge`` (the
triage/dispatcher guide — belongs to the planner) and ``ctf-writeup`` (post-solve
documentation). They are excluded from subagent discovery and exposed separately via
:func:`load_shared_guidance`.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from infrastructure.observability.langfuse_prompts import (
    get_managed_text,
    managed_text_for_file,
    prompt_name_for,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_PATH = "config"

# Skill packs under prompts/skills/ that are NOT attack subagents.
# ``solve-challenge`` is the planner's triage guide; ``ctf-writeup`` is post-solve docs.
_SHARED_GUIDANCE = frozenset({"ctf-writeup", "solve-challenge"})


# --------------------------------------------------------------------------- #
# Phase 4 — skill-section retrieval (bloat source #1). See
# docs/agent-skill-retrieval-design.md. All knobs are env-driven and DEFAULT-OFF
# so the legacy whole-note / full-index behavior is byte-for-byte preserved unless
# explicitly enabled, allowing clean A/B on the Langfuse NYU CTF datasets.
# --------------------------------------------------------------------------- #
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def section_retrieval_enabled() -> bool:
    """Master switch: serve note *sections* on demand instead of whole notes."""
    return _env_bool("SKILL_SECTION_RETRIEVAL", False)


def _section_max_chars() -> int:
    try:
        return int(os.getenv("SKILL_NOTE_SECTION_MAX_CHARS", "4000"))
    except ValueError:
        return 4000


def _index_mode() -> str:
    """How much of the SKILL.md index to embed in the system prompt.

    ``full`` (default, legacy) embeds the whole index; ``header`` condenses it to a
    heading-outline + note-pointer list; ``search`` embeds a minimal stub and relies
    on the ``search_skill`` tool to surface relevant sections on demand.
    """
    mode = (os.getenv("SKILL_INDEX_MODE", "full") or "full").strip().lower()
    return mode if mode in ("full", "header", "search") else "full"


def _search_k() -> int:
    try:
        return int(os.getenv("SKILL_SEARCH_K", "4"))
    except ValueError:
        return 4


# ----- markdown heading parsing (pure + cached; notes are static assets) ----- #
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _slugify(text: str) -> str:
    """GitHub-style anchor slug for a heading title."""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s


@lru_cache(maxsize=512)
def _parse_sections(text: str) -> Tuple[str, Tuple[Dict[str, Any], ...]]:
    """Split markdown into (intro, sections).

    Each section is ``{level, title, slug, body}`` where ``body`` is the heading line
    plus everything until the next heading of the SAME OR HIGHER level (so subsections
    are included). ``intro`` is any text before the first heading. Pure and cached on
    the note text, so re-parsing across turns/specialists is free.
    """
    lines = text.splitlines()
    heads: List[Tuple[int, int, str]] = []
    for i, ln in enumerate(lines):
        m = _HEADING_RE.match(ln)
        if m:
            heads.append((i, len(m.group(1)), m.group(2).strip()))
    if not heads:
        return text.strip(), tuple()
    sections: List[Dict[str, Any]] = []
    for pos, (idx, level, title) in enumerate(heads):
        end = len(lines)
        for idx2, level2, _t in heads[pos + 1:]:
            if level2 <= level:
                end = idx2
                break
        body = "\n".join(lines[idx:end]).strip()
        sections.append({"level": level, "title": title, "slug": _slugify(title), "body": body})
    intro = "\n".join(lines[: heads[0][0]]).strip()
    return intro, tuple(sections)


def _render_outline(note: str, intro: str, sections: Tuple[Dict[str, Any], ...]) -> str:
    """A compact table-of-contents for a note: section slugs the model can request."""
    if not sections:
        return (
            f"[NOTE {note}] This note has no subsections; it is short enough to read "
            f"whole — call read_skill_note('{note}', section='all')."
        )
    toc = "\n".join(
        f"  - {s['slug']}  ({'#' * s['level']} {s['title']})" for s in sections
    )
    intro_snip = (intro[:400] + " …") if len(intro) > 400 else intro
    return (
        f"[OUTLINE of {note}] {len(sections)} sections. This is only a table of "
        f"contents — to load the working code for a technique, call "
        f"read_skill_note('{note}', section='<slug>') with one of the slugs below "
        f"(or section='all' to read the whole note).\n"
        + (f"\nIntro: {intro_snip}\n" if intro_snip else "")
        + f"\nSections:\n{toc}"
    )


def _find_section(
    sections: Tuple[Dict[str, Any], ...], query: str
) -> Optional[Dict[str, Any]]:
    """Best-effort match of a requested section selector to a parsed section."""
    if not query:
        return None
    q = _slugify(query)
    ql = query.strip().lower()
    for s in sections:  # exact slug
        if s["slug"] == q:
            return s
    for s in sections:  # slug containment (either direction)
        if q and (q in s["slug"] or s["slug"] in q):
            return s
    for s in sections:  # title substring
        if ql and ql in s["title"].lower():
            return s
    return None


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _emit_event(name: str, **metadata: Any) -> None:
    """Best-effort Langfuse event (mirrors memory-fold / memory-recall). Never raises."""
    try:
        from infrastructure.observability import langfuse_tracer

        langfuse_tracer.record_event(name, **metadata)
    except Exception:  # pragma: no cover - observability must never break a run
        pass


@dataclass
class CtfSkill:
    """One attack-category skill pack resolved into subagent ingredients."""

    name: str  # e.g. "ctf-web" — used as the subagent_type the planner selects
    description: str  # frontmatter description: when to use / not use this category
    body: str  # SKILL.md base prompt (frontmatter stripped)
    skill_dir: Path  # absolute dir, the read tool is sandboxed to this
    notes: List[str] = field(default_factory=list)  # lazy-loadable sub-topic filenames


def _skills_root(base_path: str) -> Path:
    return Path(base_path) / "prompts" / "skills"


def _split_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Split a leading ``---`` YAML frontmatter block from the markdown body.

    Minimal, dependency-free: only ``key: value`` scalar lines are parsed (enough for
    ``name`` and ``description``). ``partition(':')`` keeps colons inside the value.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text
    meta: Dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    body = "\n".join(lines[end + 1:]).strip()
    return meta, body


def _list_notes(skill_dir: Path) -> List[str]:
    """All sub-topic note filenames in the pack (everything but SKILL.md), sorted."""
    return sorted(
        f.name for f in skill_dir.glob("*.md") if f.name.lower() != "skill.md"
    )


def discover_ctf_skills(base_path: str = DEFAULT_BASE_PATH) -> List[CtfSkill]:
    """Discover every attack-category skill pack (the 9 ``ctf-*`` subagents).

    Skips :data:`_SHARED_GUIDANCE` and any dir without a readable ``SKILL.md`` with a
    non-empty frontmatter ``description`` (the description is required for routing).
    """
    root = _skills_root(base_path)
    if not root.exists() or not root.is_dir():
        logger.warning("skills root not found: %s", root)
        return []

    skills: List[CtfSkill] = []
    for skill_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = skill_dir.name
        if not name.startswith("ctf-") or name in _SHARED_GUIDANCE:
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        try:
            raw = skill_md.read_text(encoding="utf-8")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("could not read %s: %s", skill_md, e)
            continue
        meta, body = _split_frontmatter(raw)
        description = meta.get("description", "").strip()
        if not description:
            logger.warning("skill %s has no frontmatter description; skipping", name)
            continue
        # The SKILL.md body is a **leaf** asset: served from its Langfuse-managed
        # version (frontmatter-stripped, fetched raw), falling back to the locally
        # parsed body when Langfuse is disabled/unreachable. Frontmatter (the routing
        # ``description``) is never pushed to Langfuse, so it stays parsed locally.
        body = get_managed_text(
            name=prompt_name_for(skill_md),
            fallback_text=body,
            templated=False,
        )
        skills.append(
            CtfSkill(
                name=name,
                description=description,
                body=body,
                skill_dir=skill_dir.resolve(),
                notes=_list_notes(skill_dir),
            )
        )
    return skills


def load_shared_guidance(name: str, base_path: str = DEFAULT_BASE_PATH) -> str:
    """Return the body (frontmatter stripped) of a shared-guidance skill pack.

    Used by the planner to embed triage logic (``solve-challenge``) and writeup
    conventions (``ctf-writeup``). Returns ``""`` if the pack is missing.
    """
    skill_md = _skills_root(base_path) / name / "SKILL.md"
    if not skill_md.exists():
        return ""
    try:
        _meta, body = _split_frontmatter(skill_md.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("could not read shared guidance %s: %s", name, e)
        return ""
    # Leaf asset: prefer the Langfuse-managed (frontmatter-stripped) body, falling
    # back to the locally parsed body when Langfuse is disabled/unreachable.
    return get_managed_text(
        name=prompt_name_for(skill_md),
        fallback_text=body,
        templated=False,
    )


def _normalize_note_name(note: str) -> str:
    """Resolve a requested note to its bare ``<file>.md`` form.

    Strips surrounding whitespace, any directory component (no path traversal), and a
    trailing ``#section`` anchor — markdown links read ``[txt](name.md#heading)`` and a
    weak model sometimes passes the anchor as if it were the filename. Adds the ``.md``
    extension when missing. Returns ``""`` when no usable name was given.
    """
    safe = os.path.basename(note.strip())
    safe = safe.split("#", 1)[0].strip()  # a '#heading' is a section, not a note
    if not safe:
        return ""
    if not safe.lower().endswith(".md"):
        safe += ".md"
    return safe


def _count_prior_note_reads(
    messages: List[Any],
    note_norm: str,
    current_id: Optional[str],
    section_norm: Optional[str] = None,
) -> int:
    """How many times the same note (and, when section retrieval is on, the same
    *section*) was already requested via ``read_skill_note`` earlier in THIS subagent
    run (excluding the current tool call). Compared on the normalized ``<file>.md`` name
    (and slugified section) so anchors / path prefixes can't fool the guard.

    When ``section_norm`` is given, a prior read of the SAME note but a DIFFERENT
    section does not count — reading a new section of an already-seen note is useful,
    not a wasteful re-read.
    """
    n = 0
    for m in messages or []:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") != "read_skill_note":
                continue
            if current_id is not None and tc.get("id") == current_id:
                continue
            args = tc.get("args") or {}
            prior = args.get("note")
            if not prior or _normalize_note_name(prior) != note_norm:
                continue
            if section_norm is not None:
                prior_sec = _slugify((args.get("section") or "").split("#", 1)[-1])
                if prior_sec != section_norm:
                    continue
            n += 1
    return n


def make_read_skill_note_tool(skill_dir: Path, notes: List[str]) -> Any:
    """Build a ``read_skill_note`` tool sandboxed to one skill pack directory.

    Lazy loading: a specialist reads a deep-dive note by filename only when it needs
    the detailed techniques. ``_normalize_note_name`` strips any directory component
    (no path traversal) and any ``#section`` anchor, and a duplicate guard refuses to
    re-load a note already read this run (its content is already in context) — both
    mirror the execute_script loop-breaker so a weak model can't bloat its context by
    re-reading the same tens-of-KB note over and over.

    Phase 4 (``SKILL_SECTION_RETRIEVAL=true``): instead of returning the whole note,
    the tool returns the note's heading OUTLINE when no ``section`` is given and only
    the requested SECTION's body (capped at ``SKILL_NOTE_SECTION_MAX_CHARS``) when a
    section slug is given — so the model ingests ~1–3k chars per technique instead of
    a 15k+ note. ``section='all'`` still returns the whole note. The skill-note gate
    (see :mod:`deep_generative_workflow`) only counts a *section/body* read as
    "consulted", so an outline peek alone cannot unlock ``write_script``.
    """
    resolved_dir = skill_dir.resolve()
    skill_name = resolved_dir.name
    available = ", ".join(notes) if notes else "(none)"

    @tool
    def read_skill_note(
        note: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        section: str = "",
    ) -> str:
        """Read a deep-dive technique note for THIS skill. Pass the '<file>.md' name (e.g. 'sql-injection.md'). When section retrieval is on, call it first WITHOUT a section to get the note's outline (its section slugs), then call it again with section='<slug>' to load that section's working code; pass section='all' to read the whole note. Only notes belonging to this skill are accessible."""
        # Observability: read_skill_note is otherwise silent, so without these prints we
        # cannot tell from the run logs whether a specialist actually consulted a note
        # (the whole reason it was spawned), which one, or whether it spun re-reading
        # the same one — mirrors WRITE_SCRIPT_TOOL / PLANNER_DELEGATION.
        norm = _normalize_note_name(note)
        if not norm:
            print(
                f"READ_SKILL_NOTE[no-arg] skill={skill_name!r} requested={note!r}",
                flush=True,
            )
            return f"Specify a note filename. Available notes: {available}"

        # A model often encodes the section as an anchor on the filename
        # (`name.md#heading`); honor that when no explicit ``section`` was passed.
        sec_sel = (section or "").strip()
        if not sec_sel and "#" in note:
            sec_sel = note.split("#", 1)[1].strip()

        target = (resolved_dir / norm).resolve()
        if target.parent != resolved_dir or not target.exists():
            print(
                f"READ_SKILL_NOTE[not-found] skill={skill_name!r} note={norm!r}",
                flush=True,
            )
            return f"No such note '{norm}'. Available notes: {available}"

        try:
            # Leaf asset: the note's Langfuse-managed version (fetched raw, so literal
            # ``{{...}}`` SSTI payloads survive verbatim), falling back to the on-disk
            # file when Langfuse is disabled/unreachable. The sandbox check above
            # guarantees ``target`` is inside this pack before we resolve its name.
            content = managed_text_for_file(target)
        except Exception as e:  # pragma: no cover - defensive
            print(
                f"READ_SKILL_NOTE[error] skill={skill_name!r} note={norm!r} err={e}",
                flush=True,
            )
            return f"Could not read note '{norm}': {e}"

        # ---- Legacy whole-note path (section retrieval disabled) ---------------- #
        if not section_retrieval_enabled():
            prior = _count_prior_note_reads(
                state.get("messages") or [], norm, tool_call_id
            )
            if prior >= 1:
                print(
                    f"READ_SKILL_NOTE[duplicate] skill={skill_name!r} note={norm!r} "
                    f"prior={prior}",
                    flush=True,
                )
                return (
                    f"[ALREADY LOADED] You already read '{norm}' earlier this run; its "
                    f"full content is in your context above — re-reading adds nothing. "
                    f"Act on it now: implement the attack with write_script, or read a "
                    f"DIFFERENT note if it did not match. Available notes: {available}"
                )
            print(
                f"READ_SKILL_NOTE[loaded] skill={skill_name!r} note={norm!r} "
                f"chars={len(content)}",
                flush=True,
            )
            return content

        # ---- Phase 4 section-retrieval path ------------------------------------- #
        intro, sections = _parse_sections(content)

        # No section requested → return the cheap outline (does NOT satisfy the gate).
        if not sec_sel:
            print(
                f"READ_SKILL_NOTE[outline] skill={skill_name!r} note={norm!r} "
                f"sections={len(sections)}",
                flush=True,
            )
            _emit_event(
                "skill-section-read",
                skill=skill_name,
                note=norm,
                section="(outline)",
                out_chars=0,
            )
            return _render_outline(norm, intro, sections)

        # section='all' (or a note with no headings) → whole note, capped.
        if sec_sel.strip().lower() == "all" or not sections:
            body = content
            sec_label = "all"
        else:
            match = _find_section(sections, sec_sel)
            if match is None:
                print(
                    f"READ_SKILL_NOTE[no-section] skill={skill_name!r} note={norm!r} "
                    f"requested={sec_sel!r}",
                    flush=True,
                )
                return (
                    f"No section matching '{sec_sel}' in '{norm}'.\n\n"
                    + _render_outline(norm, intro, sections)
                )
            body = match["body"]
            sec_label = match["slug"]

        sec_norm = _slugify(sec_sel)
        prior = _count_prior_note_reads(
            state.get("messages") or [], norm, tool_call_id, section_norm=sec_norm
        )
        if prior >= 1:
            print(
                f"READ_SKILL_NOTE[duplicate] skill={skill_name!r} note={norm!r} "
                f"section={sec_label!r} prior={prior}",
                flush=True,
            )
            return (
                f"[ALREADY LOADED] You already read section '{sec_label}' of '{norm}' "
                f"this run; it is in your context above. Act on it now with write_script, "
                f"or read a DIFFERENT section/note. Outline:\n"
                + _render_outline(norm, intro, sections)
            )

        cap = _section_max_chars()
        truncated = len(body) > cap
        out = body[:cap] + ("\n…[section truncated]" if truncated else "")
        print(
            f"READ_SKILL_NOTE[section] skill={skill_name!r} note={norm!r} "
            f"section={sec_label!r} chars={len(out)} (full={len(body)})",
            flush=True,
        )
        _emit_event(
            "skill-section-read",
            skill=skill_name,
            note=norm,
            section=sec_label,
            out_chars=len(out),
            full_chars=len(body),
        )
        return out

    return read_skill_note


def make_search_skill_tool(skill_dir: Path, notes: List[str]) -> Any:
    """Build a ``search_skill(query)`` tool sandboxed to one skill pack (Phase 4b-B).

    Keyword-ranks every note SECTION in the pack against the query (challenge details
    + technique) and returns the top-``SKILL_SEARCH_K`` locators with a short snippet,
    so the specialist can find the right ``read_skill_note(note, section)`` target
    without loading whole notes. Read-only and stateless; it does NOT satisfy the
    skill-note gate (it is selection/recon, not consulting the working code).
    """
    resolved_dir = skill_dir.resolve()
    skill_name = resolved_dir.name

    @tool
    def search_skill(query: str) -> str:
        """Search THIS skill's deep notes for the sections most relevant to a query (e.g. the challenge's technique, file type, or error). Returns ranked 'note.md#section' locators with a snippet; then call read_skill_note(note, section='<slug>') to load the best match. Use this when unsure which note/section applies."""
        q = _tokens(query)
        if not q:
            return "Provide a search query (e.g. the technique or challenge detail)."
        scored: List[Tuple[int, str, str, str]] = []
        for fname in notes:
            target = (resolved_dir / fname).resolve()
            if target.parent != resolved_dir or not target.exists():
                continue
            try:
                content = managed_text_for_file(target)
            except Exception:
                continue
            _intro, sections = _parse_sections(content)
            for s in sections:
                overlap = len(q & _tokens(s["title"] + " " + s["body"]))
                # Title hits weigh more than body hits.
                title_hits = len(q & _tokens(s["title"]))
                score = overlap + 2 * title_hits
                if score > 0:
                    snippet = re.sub(r"\s+", " ", s["body"]).strip()[:160]
                    scored.append((score, fname, s["slug"], snippet))
        if not scored:
            return (
                f"No sections in {skill_name} matched '{query}'. Available notes: "
                + (", ".join(notes) if notes else "(none)")
            )
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: _search_k()]
        print(
            f"SEARCH_SKILL skill={skill_name!r} query={query!r} hits={len(scored)} "
            f"returned={len(top)}",
            flush=True,
        )
        _emit_event("skill-search", skill=skill_name, query=query, hits=len(scored))
        lines = [
            f"{i + 1}. {fname}#{slug}  — {snip}"
            for i, (_score, fname, slug, snip) in enumerate(top)
        ]
        return (
            f"[search_skill] Top {len(top)} sections for {query!r} (load one with "
            f"read_skill_note('<note>', section='<slug>')):\n" + "\n".join(lines)
        )

    return search_skill


def _condense_index(body: str, cap: int = 2500) -> str:
    """Programmatically shrink a SKILL.md index to its navigational skeleton.

    Keeps heading lines and any line that references a ``.md`` note (the technique →
    note pointers), drops prose, and caps the result. Used by ``SKILL_INDEX_MODE=header``
    to cut the always-on system-prompt cost (some indexes are 25–39k chars) without
    hand-editing the nine ``SKILL.md`` files. Fail-open: if condensing yields nothing
    usable, the caller falls back to the full body.
    """
    keep: List[str] = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _HEADING_RE.match(s) or ".md" in s:
            keep.append(ln.rstrip())
    text = "\n".join(keep).strip()
    if not text:
        return ""
    if len(text) > cap:
        text = text[:cap].rstrip() + "\n…[index truncated — use search_skill to find sections]"
    return text


def _embedded_guide(skill: CtfSkill) -> Tuple[str, str]:
    """Pick the SKILL.md content embedded in the system prompt, per ``SKILL_INDEX_MODE``.

    Returns ``(label, guide_text)``. ``full`` (default) is the legacy whole body;
    ``header`` is the condensed skeleton; ``search`` is a minimal stub that defers to
    the ``search_skill`` tool. Any mode fails open to the full body.
    """
    mode = _index_mode()
    if mode == "header":
        condensed = _condense_index(skill.body)
        if condensed:
            return ("INDEX OUTLINE — condensed; use search_skill / read_skill_note", condensed)
    elif mode == "search":
        notes = ", ".join(skill.notes) if skill.notes else "(none)"
        return (
            "INDEX OMITTED — use search_skill(query) to find the right section",
            f"This skill's deep notes are not inlined to save context. Notes available: "
            f"{notes}.\nCall search_skill('<your challenge's technique / file type>') to "
            f"rank the most relevant sections, then read_skill_note(note, section='<slug>').",
        )
    return ("INDEX ONLY — load the matching note for the real method", skill.body)


def compose_subagent_prompt(base_prompt: str, skill: CtfSkill) -> str:
    """Layer a skill pack's specialization on top of the shared worker base prompt.

    The SKILL.md body is injected as a one-line *index* only: it tells the specialist
    which techniques exist, not how to run them. The working code / step-by-step
    methods live in the deep notes, loaded on demand with ``read_skill_note``. The
    prompt frames the note as a mandatory step (and the runtime gate in
    :mod:`deep_generative_workflow` enforces it by blocking the first
    ``write_script`` / ``submit_goal`` until a note has been read), because weak
    models otherwise treat the always-present index as "enough" and never descend
    into the note that actually contains the exploit.

    Phase 4: the embedded index can be condensed (``SKILL_INDEX_MODE``) and notes can
    be served section-by-section (``SKILL_SECTION_RETRIEVAL``); the workflow
    instructions adapt to whichever modes are active so the model uses the right tool
    calls.
    """
    notes = ", ".join(skill.notes) if skill.notes else "(none)"
    label, guide = _embedded_guide(skill)

    sectioned = section_retrieval_enabled()
    if sectioned:
        # Two-step lazy read: outline first, then the specific section's code.
        consult_step = (
            f"2. Identify the SPECIFIC technique the challenge uses. Call "
            f"`read_skill_note('<file>.md')` (no section) to see that note's OUTLINE of "
            f"sections, then `read_skill_note('<file>.md', section='<slug>')` to load the "
            f"working code for the matching section — BEFORE you write any exploit. Use "
            f"`search_skill('<technique/file-type>')` first if you are unsure which "
            f"note/section applies. (`section='all'` reads a whole note when needed.)\n"
        )
        gate_note = (
            "consulted at least one note SECTION (an outline peek alone does not count)"
        )
    else:
        consult_step = (
            f"2. Identify the SPECIFIC technique the challenge uses, find its entry in the "
            f"index below, and call `read_skill_note('<file>.md')` for the note it points "
            f"to — BEFORE you write any exploit/decryption script. In the guide, "
            f"`[name.md](name.md)` is a note filename; a `#section` suffix (e.g. "
            f"`(name.md#heading)`) is a heading WITHIN that note, not a separate note — "
            f"pass only the `<file>.md` part.\n"
        )
        gate_note = "consulted at least one note"

    specialization = (
        f"\n\n<CtfSpecialization category=\"{skill.name}\">\n"
        f"You are operating as the {skill.name} specialist.\n\n"
        f"HOW TO USE THIS SKILL — read this first:\n"
        f"The SKILL GUIDE below is only an INDEX of techniques. It tells you "
        f"what EXISTS, not how to do it. The actual working code, step-by-step methods, "
        f"and parameter details live in the deep notes, which you load with the "
        f"`read_skill_note(note)` tool. The index alone is NOT enough to solve the "
        f"challenge — treat it as a table of contents, not a solution.\n\n"
        f"How to work (progressive disclosure — use the notes like a reference manual):\n"
        f"1. Inspect the ACTUAL challenge FIRST: read the source/handout files "
        f"(cat/strings/xxd them, check magic bytes and headers) and the planner's task "
        f"description. Do NOT pick a note from the file type or a one-word hint alone — "
        f"'crypto'/'cipher' does not tell you WHICH technique applies.\n"
        f"{consult_step}"
        f"3. Implement the attack using the note's code/method as your template — copy its "
        f"working code and adapt it to this challenge with write_script, then run it. If a "
        f"note does not match what you found in step 1, consult the closest OTHER note "
        f"instead of forcing a wrong technique.\n\n"
        f"Notes available for this skill: {notes}. Consulting the matching note is the "
        f"difference between guessing and solving — the index only names techniques, the "
        f"note has the actual code. You have properly {gate_note} only once you load and "
        f"act on its content. Reading a note is cheap and you may read several, so "
        f"when unsure read the closest match first, and re-consult a different note if many "
        f"commands pass without a working solution (a sign the first pick was wrong). You "
        f"are never blocked from acting: read when it helps, then implement.\n\n"
        f"Stay focused on recovering the goal with this category's techniques. If you "
        f"become confident the challenge's true category is different, say so explicitly "
        f"in your final report so the planner can re-delegate.\n\n"
        f"--- SKILL GUIDE ({label}) ---\n"
        f"{guide}\n"
        f"</CtfSpecialization>"
    )
    return (base_prompt or "") + specialization
