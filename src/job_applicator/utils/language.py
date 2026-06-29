"""Output-language resolution for generated documents (cover letters + tailored résumés).

The app generates a CV and a cover letter in a single, consistent language so one application
never mixes languages (the Québec reality: a French job description should yield a French
packet). The language is configured by ``[llm] language`` (``auto`` | ``en`` | ``fr``); ``auto``
detects the job description's language so the packet mirrors the posting.

Detection is a deliberately small FR/EN heuristic (the user's two languages), not a dependency:
French has accented characters and French-only function words that English never carries, so
the two separate cleanly. The resolved language is logged by the callers so a misdetect — which
would otherwise send a real application in the wrong language — is catchable.
"""

from __future__ import annotations

import re

# French function words that English does not share (spaced so " des " doesn't match "designed").
_FR_WORDS: tuple[str, ...] = (
    " le ",
    " la ",
    " les ",
    " un ",
    " une ",
    " des ",
    " du ",
    " et ",
    " ou ",
    " pour ",
    " avec ",
    " dans ",
    " sur ",
    " vous ",
    " nous ",
    " votre ",
    " notre ",
    " est ",
    " sont ",
    " aux ",
    " qui ",
    " que ",
    " ainsi ",
    " au sein ",
    " en tant que ",
    " sécurité ",
    " réseau ",
    " expérience ",
    " compétences ",
    " poste ",
    " entreprise ",
    " équipe ",
)
# Accented characters are a strong French signal — English job posts essentially never carry them.
_FR_ACCENT_RE = re.compile(r"[àâäçéèêëîïôùûü]")
_EN_WORDS: tuple[str, ...] = (
    " the ",
    " and ",
    " for ",
    " with ",
    " you ",
    " your ",
    " our ",
    " is ",
    " are ",
    " this ",
    " that ",
    " will ",
    " skills ",
    " security ",
    " network ",
    " experience ",
    " team ",
    " role ",
)

_LANG_NAMES = {"en": "English", "fr": "French"}


def detect_language(text: str) -> str:
    """Return ``'fr'`` or ``'en'`` for a job description (heuristic; defaults to ``'en'``)."""
    t = f" {text.lower()} "
    fr = sum(t.count(w) for w in _FR_WORDS) + 2 * len(_FR_ACCENT_RE.findall(t))
    en = sum(t.count(w) for w in _EN_WORDS)
    return "fr" if fr > en else "en"


def resolve_output_language(setting: str, text: str) -> str:
    """Resolve the configured ``language`` setting to a full language name for a prompt.

    ``auto`` detects from *text* (the job description); ``en``/``fr`` (or ``english``/``french``)
    force. Returns ``"English"`` or ``"French"`` (unknown settings fall back to ``auto`` detection).
    """
    s = setting.strip().lower()
    if s.startswith("fr"):
        return "French"
    if s.startswith("en"):
        return "English"
    return _LANG_NAMES[detect_language(text)]
