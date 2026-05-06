"""
local_pipeline/sadev_formatter.py
Parser for the new SADEV output format:

  ---RESPONSE---
  <natural conversational reply>
  ---METADATA---
  emotion=...
  intent=...
  stressor=...
  urgency=...
  risk=...
  strategy=...

Three-strategy parsing — degrades gracefully:
  1. Exact   — both delimiters present
  2. Partial — only ---RESPONSE--- found; metadata extracted from tail lines
  3. Fallback — no delimiters; treat whole text as response, parse metadata from any `key=val` lines

The formatter also:
  - Strips hollow affirmations from response start
  - Enforces word cap (120 words)
  - Validates and normalises metadata fields
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SadevResponse:
    response: str                    # the clean natural response text
    emotion: str    = "neutral"
    intent: str     = "unknown"
    stressor: str   = "none"
    urgency: int    = 1
    risk: int       = 1
    strategy: list[str] = field(default_factory=list)
    parse_strategy: str = "exact"    # "exact" | "partial" | "fallback"


_HOLLOW = re.compile(
    r"^(absolutely[,!]?|great question[!]?|of course[!]?|sure[!]?|"
    r"certainly[!]?|definitely[!]?|that'?s great[!]?|"
    r"great[!]?|wow[!]?|awesome[!]?|i see[!.]?|"
    r"that makes sense[!.]?|i hear you[!.]?)\s*",
    re.IGNORECASE,
)

_RESPONSE_DELIM = re.compile(r"---RESPONSE---\s*", re.IGNORECASE)
_METADATA_DELIM = re.compile(r"---METADATA---\s*", re.IGNORECASE)

# Matches key=value lines anywhere in text
_KV_RE = re.compile(
    r"^(emotion|intent|stressor|urgency|risk|strategy)\s*=\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


class SadevFormatter:

    def parse(self, raw: str, fallback_signals: dict | None = None) -> SadevResponse:
        """
        Parse LLM output into a SadevResponse.

        fallback_signals: pre-computed signals from EmotionDetector/SafetyLayer.
        Used to fill metadata if the LLM omits or garbles it.
        """
        text = raw.strip()
        fb = fallback_signals or {}

        # ── Strategy 1: Both delimiters present ───────────────────────────
        r_match = _RESPONSE_DELIM.search(text)
        m_match = _METADATA_DELIM.search(text)

        if r_match and m_match and r_match.start() < m_match.start():
            response_raw = text[r_match.end():m_match.start()].strip()
            metadata_raw = text[m_match.end():].strip()
            parse_strategy = "exact"

        # ── Strategy 2: Only ---RESPONSE--- found ─────────────────────────
        elif r_match:
            tail = text[r_match.end():].strip()
            # Split on first blank line or key=value block
            kv_start = _KV_RE.search(tail)
            if kv_start:
                response_raw = tail[:kv_start.start()].strip()
                metadata_raw = tail[kv_start.start():].strip()
            else:
                response_raw = tail
                metadata_raw = ""
            parse_strategy = "partial"

        # ── Strategy 3: No delimiters at all ──────────────────────────────
        else:
            # Anything that looks like metadata is at the bottom
            kv_start = _KV_RE.search(text)
            if kv_start:
                response_raw = text[:kv_start.start()].strip()
                metadata_raw = text[kv_start.start():].strip()
            else:
                response_raw = text
                metadata_raw = ""
            parse_strategy = "fallback"

        # ── Parse metadata first (needed for risk-aware clean) ────────────
        meta = self._parse_metadata(metadata_raw)

        # ── Clean response ─────────────────────────────────────────────────
        try:
            _risk_for_cap = int(meta.get("risk") or fb.get("risk", 1))
        except (ValueError, TypeError):
            _risk_for_cap = fb.get("risk", 1)
        response = self._clean_response(response_raw, risk=_risk_for_cap)

        # ── Resolve final field values ─────────────────────────────────────
        # Fill missing fields from fallback signals
        emotion  = meta.get("emotion")  or fb.get("emotion",  "neutral")
        intent   = meta.get("intent")   or fb.get("intent",   "unknown")
        stressor = meta.get("stressor") or fb.get("stressor", "none")
        strategy_raw = meta.get("strategy", "")
        strategy = [s.strip() for s in strategy_raw.split(",") if s.strip()]

        try:
            urgency = int(meta.get("urgency") or fb.get("urgency", 1))
        except (ValueError, TypeError):
            urgency = fb.get("urgency", 1)

        try:
            risk = int(meta.get("risk") or fb.get("risk", 1))
        except (ValueError, TypeError):
            risk = fb.get("risk", 1)

        return SadevResponse(
            response=response,
            emotion=emotion,
            intent=intent,
            stressor=stressor,
            urgency=max(1, min(5, urgency)),
            risk=max(1, min(5, risk)),
            strategy=strategy,
            parse_strategy=parse_strategy,
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _clean_response(self, text: str, risk: int = 1) -> str:
        # Strip section markers that leaked through
        text = re.sub(r"\[(?:VALIDATION|REFLECTION|QUESTION)\]", "", text, flags=re.IGNORECASE)
        # Strip hollow affirmations from start
        text = _HOLLOW.sub("", text).strip()
        # Remove "I " at very start (rule: don't start with "I")
        text = re.sub(r"^I\s+", "", text)
        # Deduplicate adjacent blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        # Word cap: 150 for crisis (risk >= 4), 120 otherwise
        cap = 150 if risk >= 4 else 120
        words = text.split()
        if len(words) > cap:
            text = " ".join(words[:cap]).rstrip(",:;") + "..."
        return text

    def _parse_metadata(self, raw: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for match in _KV_RE.finditer(raw):
            key = match.group(1).lower()
            val = match.group(2).strip()
            result[key] = val
        return result