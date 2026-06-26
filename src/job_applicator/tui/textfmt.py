"""Readable rendering of scraped job descriptions.

Board scrapers return the detail-pane ``inner_text`` verbatim: paragraphs are sometimes one
long line, sometimes hard-wrapped mid-sentence; blank-line spacing is inconsistent; section
headers run into the text. This reflows that into something readable WITHOUT a fragile parser —
a few robust rules (per the project's "robust rules beat a fragile parser" stance):

1. Reflow only **lowercase-continuation** lines into the previous line — that is the signature
   of a hard-wrapped sentence; a list item or new sentence starts uppercase, so items are NOT
   merged into a wall of text.
2. Collapse runs of blank lines to a single blank (fixes the single-vs-double spacing drift).
3. Bold a small set of **known section headers** and pad them with one blank line, giving every
   posting consistent hierarchy regardless of how the board emitted it.

Returns Rich-safe markup: the text content is escaped here, and the only markup added is the
header ``[bold]`` — so callers must NOT re-escape the result.
"""

from __future__ import annotations

from rich.markup import escape

# Lowercased prefixes that mark a section header when they appear on a short line of their own.
# Matched by startswith so "What You'll Do" / "About This Role" / "Key Responsibilities" hit.
_HEADER_PREFIXES = (
    "about",
    "responsibilities",
    "key responsibilities",
    "requirements",
    "qualifications",
    "minimum qualifications",
    "preferred qualifications",
    "what you",
    "what we",
    "who you",
    "who we",
    "the role",
    "your role",
    "role description",
    "job description",
    "job overview",
    "overview",
    "summary",
    "skills",
    "experience",
    "education",
    "benefits",
    "perks",
    "compensation",
    "salary",
    "duties",
    "nice to have",
    "what's in it",
    "why join",
    "our team",
)
_HEADER_MAX_LEN = 60  # a real header is short; this guards a sentence that merely starts "About…"
# A known-prefix line longer than this is a sentence ("Experience in a …"), not a header.
_PREFIX_MAX_LEN = 42
# Words that don't count toward "is this Title Case" (so "Incident Response & Ownership" qualifies).
_CONNECTORS = {"of", "the", "and", "to", "in", "for", "a", "an", "with", "on", "at", "or", "&"}


def _looks_like_heading(s: str) -> bool:
    """A short, terminal-punctuation-free line whose every content word is capitalized — the
    signature of a section sub-header ("Incident Response & Ownership") the known list misses.
    Strict (ALL content words capitalized) to keep prose fragments from being bolded."""
    words = s.split()
    if not (2 <= len(words) <= 7):
        return False
    content = [w for w in words if w.lower() not in _CONNECTORS]
    return len(content) >= 2 and all(w[:1].isupper() for w in content)


def _is_header(line: str) -> bool:
    s = line.strip().rstrip(":")
    if not s or len(s) > _HEADER_MAX_LEN or s[-1:] in ".!?,;":
        return False
    low = s.lower()
    if len(s) <= _PREFIX_MAX_LEN and any(
        low == p or low.startswith(p + " ") or low.startswith(p + "'") for p in _HEADER_PREFIXES
    ):
        return True
    return _looks_like_heading(s)


def format_job_description(raw: str) -> str:
    """Reflow + lightly structure a scraped description into readable Rich markup (escaped)."""
    if not raw:
        return ""
    # Pass 1: classify each line, reflowing hard-wrapped sentences. A line starting lowercase
    # continues the previous line ONLY when that previous line is body text (never a header or
    # across a blank) — so a header is never glued to the line beneath it.
    tokens: list[tuple[str, str]] = []  # ("header"|"text"|"blank", content)
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if not s:
            tokens.append(("blank", ""))
        elif _is_header(s):
            tokens.append(("header", s.rstrip(":")))
        elif tokens and tokens[-1][0] == "text" and s[:1].islower():
            tokens[-1] = ("text", f"{tokens[-1][1]} {s}")
        else:
            tokens.append(("text", s))
    # Pass 2: render — collapse blank runs, pad headers with a blank, escape the body text.
    out: list[str] = []

    def blank() -> None:
        if out and out[-1] != "":
            out.append("")

    for kind, content in tokens:
        if kind == "blank":
            blank()
        elif kind == "header":
            blank()
            out.append(f"[bold]{escape(content)}[/bold]")
            out.append("")
        else:
            out.append(escape(content))
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)
