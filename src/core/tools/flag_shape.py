"""Deterministic flag-shape discrimination for submission discipline.

The consolidated-experiment failure taxonomy (dev-consolidate) reduced most wrong
submissions to a single root cause: the model submits values that are **not
flag-shaped** — a filename (``decoded_image.png``), a URL from a man page
(``http://steghide.sourceforge.net/``), a memory address (``0x8048fb6``), a
challenge title, its own multi-sentence analysis/report prose, a value with the
wrong wrapper (``CSAW{...}`` / no wrapper when the format is ``flag{...}``), or an
un-derived intermediate (a raw 600-digit integer instead of the ``long_to_bytes``
result). See ``gencyber-Experiments/dev-consolidate/REPORT.md`` (F1/F3/F5/F6) and
``APPROACH_ANALYSIS.md`` (W3/W4).

This module is the deterministic pre-oracle guard both reports recommend as the
single highest-leverage fix: it classifies obvious non-flags, returns a corrective
steer instead of burning an oracle call, and derives targeted "finish the last
step" hints for the near-miss (F3) cases.

Pure and dependency-free (stdlib ``re`` only) so it is unit-testable without the
graph or a running workbench.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShapeVerdict:
    """Result of the deterministic flag-shape check.

    ``ok`` is False when the candidate is confidently not a flag (a pre-oracle
    reject). ``reason`` is the corrective steer surfaced to the model. ``category``
    is a short machine label for diagnostics/metrics.
    """

    ok: bool
    reason: Optional[str] = None
    category: str = "ok"


# File extensions that mark a bare token as a filename (F1: sleeping_guard submitted
# ``decoded_image.png``). Kept broad but anchored to a full-string match so a flag
# that merely contains one of these as a substring is never rejected.
_FILE_EXT = (
    r"png|jpe?g|gif|bmp|svg|webp|zip|tar|gz|tgz|bz2|xz|7z|rar|txt|log|bin|dat|"
    r"exe|elf|out|o|so|a|dll|pdf|docx?|xlsx?|pptx?|csv|pcap|pcapng|cap|py|pyc|"
    r"sh|bash|c|cc|cpp|h|hpp|rs|go|js|ts|md|json|yaml|yml|xml|html?|css|wav|mp3|"
    r"mp4|avi|jar|apk|war|class|img|iso|vmdk|db|sqlite"
)

_URL_RE = re.compile(r"^\s*(?:[a-z][a-z0-9+.\-]*://|www\.)", re.IGNORECASE)
_ADDR_RE = re.compile(r"^\s*0x[0-9a-fA-F]{4,}\s*$")
_FILENAME_RE = re.compile(
    r"^[\w][\w .()\-]*\.(?:" + _FILE_EXT + r")\s*$", re.IGNORECASE
)
# A flag token: some short alphanumeric prefix, then a ``{...}`` body.
_WRAPPER_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]{0,15})\s*\{(.*)\}", re.DOTALL)
_LEADING_WRAPPER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]{0,15})\s*\{")

# A candidate with this many whitespace-separated words and no flag wrapper is prose
# (F6: hungman submitted its binary-analysis paragraphs; warmup its exploit plan;
# i_got_id its "no flags found" report).
_PROSE_MIN_WORDS = 7
# A shorter run of words still reads as prose if it carries sentence punctuation.
_PROSE_SENTENCE_MIN_WORDS = 5
_SENTENCE_RE = re.compile(r"[.:;!?]\s")


def extract_flag_format(text: Optional[str]) -> Optional[str]:
    """Best-effort recovery of the flag *wrapper* prefix stated in a briefing.

    Returns the lowercased prefix (e.g. ``"flag"``, ``"csaw"``) when the briefing
    states the format, else ``None``. Priority order:

      1. An explicit template like ``flag{...}`` / ``FLAG{FLAG}`` / ``csaw{...}``.
      2. The most common ``prefix{...}`` occurrence in the text.

    Conservative by design: when it cannot confidently name a single format it
    returns ``None`` and the caller skips all wrapper enforcement.
    """
    if not text:
        return None
    # (1) explicit template with an obvious placeholder body.
    tmpl = re.findall(
        r"\b([A-Za-z][A-Za-z0-9_]{1,15})\s*\{\s*"
        r"(?:\.\.\.|xxx+|[A-Za-z0-9_ ]{0,20})\s*\}",
        text,
    )
    placeholder_bodies = {"...", "xxx", "flag", "your_flag", "placeholder", ""}
    for pref in tmpl:
        # A template like flag{...}/flag{FLAG} is a strong signal.
        return pref.lower()
    # (2) fall back to the most frequent prefix{...} in the text.
    occ = re.findall(r"\b([A-Za-z][A-Za-z0-9_]{1,15})\s*\{[^}\n]{0,60}\}", text)
    if not occ:
        return None
    counts: dict = {}
    for pref in occ:
        counts[pref.lower()] = counts.get(pref.lower(), 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0]


def _body_of(candidate: str) -> str:
    """The inner ``{...}`` body of a wrapped candidate, else the whole candidate."""
    m = _WRAPPER_RE.search(candidate)
    return m.group(2).strip() if m else candidate.strip()


def _wrapper_prefix(candidate: str) -> Optional[str]:
    m = _LEADING_WRAPPER_RE.match(candidate)
    return m.group(1) if m else None


def classify_flag_candidate(
    candidate: str, *, flag_format: Optional[str] = None
) -> ShapeVerdict:
    """Classify whether ``candidate`` is plausibly a flag.

    Rejects (``ok=False``) only high-precision non-flag classes so a genuine flag —
    including a legitimately derived one (a long underscored secret, a hex token, a
    wrapped recovered value) — is never blocked:

      * URLs, bare filenames, bare memory addresses (F1)
      * multi-sentence prose / report text with no flag wrapper (F6)
      * a wrapper that contradicts the briefing's stated format, or no wrapper at
        all when the format requires one (F5)
      * an un-derived huge decimal integer body (F3: broken_box)

    Case-only wrapper differences (``FLAG`` vs ``flag``) are NOT rejected — the
    workbench oracle is case-tolerant. Everything else passes to the oracle.
    """
    norm = (candidate or "").strip()
    if not norm:
        return ShapeVerdict(False, "empty submission", "empty")

    if _URL_RE.match(norm):
        return ShapeVerdict(
            False,
            "That looks like a URL, not a flag. A flag is the specific secret token "
            "the challenge hides — never a URL, man-page link, or documentation "
            "address. Recover the actual flag from tool output.",
            "url",
        )

    if _ADDR_RE.match(norm):
        return ShapeVerdict(
            False,
            "That looks like a memory address, not a flag. An address is a step "
            "toward an exploit, not the answer. Continue the exploit and recover the "
            "flag it yields.",
            "address",
        )

    m = _WRAPPER_RE.search(norm)
    has_wrapper = m is not None
    if not has_wrapper and _FILENAME_RE.match(norm):
        return ShapeVerdict(
            False,
            "That looks like a filename, not a flag. Do not submit the name of a file "
            "you produced — open/inspect it and submit the secret it contains.",
            "filename",
        )

    # Prose / report text: no flag wrapper and reads like sentences. This is the
    # dominant F6 pattern (submitting the specialist's report, an exploit plan, or a
    # "nothing found" summary). Underscored secrets (no spaces) are never prose.
    if not has_wrapper:
        words = norm.split()
        looks_sentence = bool(_SENTENCE_RE.search(norm)) and len(words) >= _PROSE_SENTENCE_MIN_WORDS
        if "\n" in norm or len(words) >= _PROSE_MIN_WORDS or looks_sentence:
            return ShapeVerdict(
                False,
                "That reads like analysis/report text, not a flag. Never submit your "
                "own summary, plan, or a 'nothing found' note — extract only the exact "
                "flag token (matching the challenge's format) from real tool output.",
                "prose",
            )

    # Format enforcement (only when the briefing confidently stated a format).
    fmt = (flag_format or "").strip().lower()
    if fmt:
        got = _wrapper_prefix(norm)
        if got is None:
            return ShapeVerdict(
                False,
                f"The challenge's flag format is {fmt}{{...}}, but this value has no "
                f"{fmt}{{}} wrapper. If you recovered the secret, wrap it: "
                f"{fmt}{{<recovered value>}} and submit that.",
                "unwrapped",
            )
        if got.lower() != fmt:
            return ShapeVerdict(
                False,
                f"Wrong flag wrapper: the briefing states the format is {fmt}{{...}}, "
                f"but you submitted a {got}{{}} wrapper. Re-wrap the recovered value "
                f"as {fmt}{{...}}.",
                "wrapper_mismatch",
            )

    # Un-derived intermediate: a body that is a very long decimal integer is almost
    # never the flag text — it is a number that still needs converting (F3 broken_box
    # submitted a raw 600-digit integer instead of long_to_bytes(...)).
    body = _body_of(norm)
    if re.fullmatch(r"\d{40,}", body):
        return ShapeVerdict(
            False,
            f"The flag body is a {len(body)}-digit integer — that is an un-converted "
            "number, not flag text. Convert it (e.g. long_to_bytes for a big int, or "
            "the challenge's own decoding) and submit the resulting readable value.",
            "raw_integer",
        )

    return ShapeVerdict(True, None, "ok")


def is_grounded(candidate: str, evidence_text: Optional[str]) -> bool:
    """Whether ``candidate`` was actually recovered from real tool output.

    True if the candidate — or its unwrapped ``{...}`` body — appears verbatim
    (case-insensitive) in ``evidence_text`` (the accumulated command-output
    transcript). Accepting a match on the *body* lets a legitimately wrapped
    recovery through (i_got_id: recover ``A_S3cret_...`` from output, submit
    ``flag{A_S3cret_...}``) while still blocking a value that was never observed —
    the F6 report-as-flag and W5 hallucination cases, and the ``recall_evidence``
    laundering of a fabricated flag (H4).

    Returns True when there is no evidence to check against, so callers that lack an
    evidence channel (e.g. the base tool's NYU path) are unaffected.
    """
    ev = (evidence_text or "").lower()
    if not ev:
        return True
    cand = (candidate or "").strip().lower()
    if len(cand) >= 4 and cand in ev:
        return True
    body = _body_of(candidate).strip().lower()
    return len(body) >= 4 and body in ev


def derivation_hint(candidate: str, *, flag_format: Optional[str] = None) -> Optional[str]:
    """A targeted "you may have missed the last transform" hint for a rejected value.

    Maps the *shape* of a rejected candidate to the final-derivation step it most
    likely still needs (F3/W3): hex → decode, big int → long_to_bytes, base64-ish →
    decode, wrong/again wrapper → normalize. Appended to the oracle's rejection
    feedback so a near-miss becomes a solve. Returns ``None`` when nothing applies.
    """
    body = _body_of(candidate)
    # Decimal first: digits are also valid hex, so an all-digits body must be treated
    # as an integer (long_to_bytes), not as hex.
    if re.fullmatch(r"\d{20,}", body):
        return (
            "The value is a long integer — if it is a number that must become text, "
            "apply long_to_bytes(...) (or the challenge's decoding) before submitting."
        )
    if re.fullmatch(r"[0-9a-fA-F]{16,}", body) and len(body) % 2 == 0:
        return (
            "The value looks like raw hex — if it is un-decoded, convert it first "
            "(e.g. bytes.fromhex(...)) and submit the decoded text."
        )
    if re.fullmatch(r"[A-Za-z0-9+/]{16,}={0,2}", body) and len(body) % 4 == 0:
        return (
            "The value looks base64-encoded — try decoding it (base64 -d) and submit "
            "the decoded result if that is the real secret."
        )
    fmt = (flag_format or "").strip().lower()
    if fmt and _wrapper_prefix(candidate) is None:
        return f"Remember the flag format is {fmt}{{...}} — wrap the recovered value."
    return None
