"""
dataset/cleaner.py
Text normalisation and noise removal for SADEV training data.

Role in pipeline:
  data/raw/ → cleaner → data/interim/

What it does:
  - Strips URLs, extra whitespace, invisible characters
  - Normalises Hinglish punctuation and spacing
  - Removes AI meta-language that leaked through generation
  - Checks language code consistency
  - Does NOT change labels — label correction is validator.py's job

Design:
  - Every operation is reversible in theory (keeps original text in '_original_text')
  - Rejection is a last resort — prefer fixing over dropping
  - All rules are O(n) regex, fast enough for 100k+ records
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ── Noise patterns ────────────────────────────────────────────────────────

_URL          = re.compile(r"https?://\S+|www\.\S+")
_HTML_ENT     = re.compile(r"&[a-z]+;|&#\d+;")
_EXCESS_SPACE = re.compile(r"[ \t]{2,}")
_EXCESS_NEWLN = re.compile(r"\n{3,}")
_ZERO_WIDTH   = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_ELLIPSIS_PAD = re.compile(r"\s*\.\.\.\s*")
_MULTIPLE_PUNC= re.compile(r"([?!.,]){3,}")  # ??? → ??  ...  → ...  kept at 2

# AI meta-language that sometimes leaks through generation prompts
_META_LANGUAGE = re.compile(
    r"\b(as a user|generate an example|here is an example|training data|"
    r"llm|gpt|claude|sample message|example message|annotation note|annotator)\b",
    re.IGNORECASE,
)

# Common Hinglish markers — used for language detection heuristics
_HINGLISH_MARKERS = re.compile(
    r"\b(hai|hun|hain|meri|mujhe|main|aur|lekin|kya|nahi|bhi|toh|bahut|"
    r"ghar|mummy|papa|chahta|chahti|karna|karun|izzat|shaadi|rishta|"
    r"yaar|dost|matlab|kyunki|isliye|phir|accha|theek|hota|lagta|rehta)\b",
    re.IGNORECASE,
)
_ENGLISH_MARKERS = re.compile(
    r"\b(the|a|an|is|are|my|your|we|they|this|and|but|not|feel|think|want|"
    r"been|have|I'm|it's|don't|can't|won't)\b",
)


@dataclass
class CleanResult:
    text: str
    original_text: str
    language: str
    language_corrected: bool
    dropped: bool
    drop_reason: str


class TextCleaner:
    """
    Cleans individual text records.

    Args:
        min_length: minimum character length after cleaning (default: 8)
        max_length: maximum character length (default: 1200)
        strip_urls: remove URLs (default: True)
    """

    def __init__(
        self,
        min_length: int = 8,
        max_length: int = 1200,
        strip_urls: bool = True,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.strip_urls = strip_urls

    def clean(self, text: str, language: str = "en") -> CleanResult:
        """
        Clean a single text record.
        Returns CleanResult with drop=True if the record should be discarded.
        """
        original = text
        lang_corrected = False

        # ── Drop immediately if AI meta-language present ──────────────────
        if _META_LANGUAGE.search(text):
            return CleanResult(
                text=text, original_text=original,
                language=language, language_corrected=False,
                dropped=True, drop_reason="meta_language_detected",
            )

        # ── Unicode normalisation (NFC) ───────────────────────────────────
        text = unicodedata.normalize("NFC", text)

        # ── Strip zero-width and invisible characters ─────────────────────
        text = _ZERO_WIDTH.sub("", text)

        # ── HTML entities ────────────────────────────────────────────────
        text = _HTML_ENT.sub(" ", text)

        # ── URLs ─────────────────────────────────────────────────────────
        if self.strip_urls:
            text = _URL.sub("[link]", text)

        # ── Punctuation normalisation ─────────────────────────────────────
        text = _MULTIPLE_PUNC.sub(lambda m: m.group(1) * 2, text)
        text = _ELLIPSIS_PAD.sub("...", text)

        # ── Whitespace normalisation ──────────────────────────────────────
        text = _EXCESS_SPACE.sub(" ", text)
        text = _EXCESS_NEWLN.sub("\n\n", text)
        text = text.strip()

        # ── Length checks ─────────────────────────────────────────────────
        if len(text) < self.min_length:
            return CleanResult(
                text=text, original_text=original,
                language=language, language_corrected=False,
                dropped=True, drop_reason="too_short",
            )
        if len(text) > self.max_length:
            text = text[: self.max_length].rstrip()
            # Truncation is not a drop — just a trim

        # ── Language code sanity check ────────────────────────────────────
        language, lang_corrected = self._check_language(text, language)

        return CleanResult(
            text=text,
            original_text=original,
            language=language,
            language_corrected=lang_corrected,
            dropped=False,
            drop_reason="",
        )

    def clean_batch(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Clean a list of record dicts (must have 'text' and 'language' keys).
        Returns (accepted, rejected) lists.
        Each accepted record gets '_original_text' and '_language_corrected' keys.
        """
        accepted, rejected = [], []
        for rec in records:
            text = str(rec.get("text", ""))
            lang = str(rec.get("language", "en"))
            result = self.clean(text, lang)
            if result.dropped:
                rejected.append({**rec, "_drop_reason": result.drop_reason})
            else:
                accepted.append({
                    **rec,
                    "text": result.text,
                    "language": result.language,
                    "_original_text": result.original_text,
                    "_language_corrected": result.language_corrected,
                })
        return accepted, rejected

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_language(self, text: str, declared: str) -> tuple[str, bool]:
        """
        Heuristic language consistency check.
        If declared is 'en' but text has strong Hinglish markers, upgrade to 'hi-en-mix'.
        If declared is 'hi-en-mix' but text has no Hindi markers at all, downgrade to 'en'.
        Returns (corrected_language, was_corrected).
        """
        hindi_matches   = len(_HINGLISH_MARKERS.findall(text))
        english_matches = len(_ENGLISH_MARKERS.findall(text))

        if declared == "en" and hindi_matches >= 3:
            return "hi-en-mix", True

        if declared == "hi-en-mix" and hindi_matches == 0 and english_matches >= 3:
            return "en", True

        return declared, False
