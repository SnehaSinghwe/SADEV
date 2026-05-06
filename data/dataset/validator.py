"""
dataset/validator.py
Label correction and schema enforcement for SADEV training data.

Role in pipeline:
  data/interim/ → validator → data/processed/

Two jobs:
  1. SCHEMA ENFORCEMENT — required fields, valid enum values, type coercion
  2. LABEL CORRECTION — fix logically wrong labels using domain rules

Label correction rules (in priority order):
  - Suicidal / self-harm language → always crisis, urgency=5
  - Idiom/metaphor ("kill this exam") → NOT crisis, urgency capped at 2
  - Crisis category → urgency must be >= 4
  - High distress category → urgency must be >= 3
  - Neutral category → urgency must be 1
  - Urgency 4-5 with non-crisis/non-distress category → escalate to high_distress

These rules prevent the model from learning incorrect patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Taxonomy ──────────────────────────────────────────────────────────────

VALID_INTENTS = {
    "venting_distress", "anxiety_panic", "sadness_low_mood", "overwhelm_burnout",
    "loneliness", "career_job_stress", "academic_pressure", "financial_stress",
    "family_conflict", "relationship_issue", "marriage_pressure", "social_reputation",
    "conflict_exposure", "fear_of_consequence", "advice_seeking", "decision_conflict",
    "planning_next_steps", "sleep_issue", "focus_issue", "productivity_block",
    "hopelessness", "emotional_exhaustion", "self_harm_intent", "suicidal_thoughts",
    "greeting", "casual_chat",
}

VALID_CATEGORIES = {
    "emotional", "life_stressor", "social_stress", "problem_solving",
    "behavioral", "high_distress", "crisis", "neutral",
}

VALID_LANGUAGES = {"hi-en-mix", "en", "hi", "bn", "bn-en-mix"}

# Intent → expected category (for correction)
INTENT_TO_CATEGORY = {
    "venting_distress":    "emotional",
    "anxiety_panic":       "emotional",
    "sadness_low_mood":    "emotional",
    "loneliness":          "emotional",
    "emotional_exhaustion":"high_distress",
    "overwhelm_burnout":   "high_distress",
    "hopelessness":        "high_distress",
    "career_job_stress":   "life_stressor",
    "academic_pressure":   "life_stressor",
    "financial_stress":    "life_stressor",
    "family_conflict":     "social_stress",
    "relationship_issue":  "social_stress",
    "marriage_pressure":   "social_stress",
    "social_reputation":   "social_stress",
    "conflict_exposure":   "social_stress",
    "fear_of_consequence": "social_stress",
    "advice_seeking":      "problem_solving",
    "decision_conflict":   "problem_solving",
    "planning_next_steps": "problem_solving",
    "sleep_issue":         "behavioral",
    "focus_issue":         "behavioral",
    "productivity_block":  "behavioral",
    "self_harm_intent":    "crisis",
    "suicidal_thoughts":   "crisis",
    "greeting":            "neutral",
    "casual_chat":         "neutral",
}

# ── Text patterns ─────────────────────────────────────────────────────────

# Crisis signals: explicit ideation / risk language
_CRISIS_SIGNALS = re.compile(
    r"\b(end my life|end it all|kill myself|want to die|don't want to live|"
    r"suicidal|no reason to live|can't go on|hurt myself|cut myself|"
    r"thinking about suicide|thinking of ending|overdose on|take my life|"
    r"marna chahta|mar jaun|zindagi nahi chahiye|khatam kar lun)\b",
    re.IGNORECASE,
)

# Safe metaphors that look like crisis but aren't
_SAFE_METAPHORS = re.compile(
    r"\b(kill(ing)?\s+(it|this|exam|test|interview|presentation)|"
    r"dying\s+of\s+(laughter|boredom|embarrassment|hunger)|"
    r"dead\s+(tired|serious|inside\s+jokes)|"
    r"murder(ing)?\s+(it|this)|"
    r"i\s+could\s+kill\s+for|"
    r"this\s+is\s+killing\s+me\s+(with\s+)?(laughter|boredom))\b",
    re.IGNORECASE,
)

# Distress signals (not crisis, but elevated)
_DISTRESS_SIGNALS = re.compile(
    r"\b(can't take it|breaking down|falling apart|losing my mind|"
    r"completely lost|nothing matters|what's the point|"
    r"toot gaya|toot raha|haar gaya|haar raha|roke nahi)\b",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    record: dict
    corrections: list[str] = field(default_factory=list)
    dropped: bool = False
    drop_reason: str = ""

    @property
    def was_corrected(self) -> bool:
        return bool(self.corrections)


class Validator:
    """
    Validates and corrects labels for SADEV training records.

    Args:
        strict_schema:  if True, drop records with missing required fields
                        if False, fill missing fields with defaults
    """

    REQUIRED_FIELDS = {"text", "intent", "category", "urgency"}

    def __init__(self, strict_schema: bool = False):
        self.strict_schema = strict_schema

    def validate(self, record: dict) -> ValidationResult:
        """Validate and correct a single record."""
        rec = dict(record)
        corrections: list[str] = []

        # ── Schema: required fields ───────────────────────────────────────
        missing = self.REQUIRED_FIELDS - rec.keys()
        if missing:
            if self.strict_schema:
                return ValidationResult(rec, dropped=True, drop_reason=f"missing_fields:{','.join(missing)}")
            # Fill defaults
            defaults = {"text": "", "intent": "casual_chat", "category": "neutral", "urgency": 1}
            for f in missing:
                rec[f] = defaults[f]
                corrections.append(f"filled_default:{f}")

        # ── Type coercion ──────────────────────────────────────────────────
        text = str(rec.get("text", "")).strip()
        rec["text"] = text
        if not text:
            return ValidationResult(rec, dropped=True, drop_reason="empty_text")

        try:
            urgency = max(1, min(5, int(rec.get("urgency", 2))))
        except (ValueError, TypeError):
            urgency = 2
            corrections.append("urgency_coerced_to_int")
        rec["urgency"] = urgency

        intent   = str(rec.get("intent", "")).strip()
        category = str(rec.get("category", "")).strip()
        language = str(rec.get("language", "en")).strip()

        # ── Schema: enum validation ───────────────────────────────────────
        if intent not in VALID_INTENTS:
            corrections.append(f"intent_invalid:{intent!r}")
            intent = "casual_chat"
        rec["intent"] = intent

        if category not in VALID_CATEGORIES:
            # Try to fix from intent
            fixed_cat = INTENT_TO_CATEGORY.get(intent, "emotional")
            corrections.append(f"category_fixed_from_intent:{category!r}→{fixed_cat!r}")
            category = fixed_cat
        rec["category"] = category

        if language not in VALID_LANGUAGES:
            corrections.append(f"language_defaulted:{language!r}→en")
            language = "en"
        rec["language"] = language

        # ── Label correction ──────────────────────────────────────────────
        rec, corrections = self._correct_labels(rec, corrections, text, intent, category, urgency)

        return ValidationResult(record=rec, corrections=corrections)

    def validate_batch(
        self, records: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Validate a batch of records.
        Returns (clean, corrected, dropped):
          - clean:     passed without any changes
          - corrected: passed but with label corrections applied
          - dropped:   failed schema and cannot be recovered
        """
        clean, corrected, dropped = [], [], []
        for rec in records:
            result = self.validate(rec)
            if result.dropped:
                dropped.append({**result.record, "_drop_reason": result.drop_reason})
            elif result.was_corrected:
                corrected.append({
                    **result.record,
                    "_corrections": result.corrections,
                })
            else:
                clean.append(result.record)
        return clean, corrected, dropped

    # ── Label correction logic ────────────────────────────────────────────

    def _correct_labels(
        self,
        rec: dict,
        corrections: list[str],
        text: str,
        intent: str,
        category: str,
        urgency: int,
    ) -> tuple[dict, list[str]]:
        """Apply domain-rule-based label corrections. Returns updated (rec, corrections)."""

        is_safe_metaphor = bool(_SAFE_METAPHORS.search(text))
        is_crisis_signal = bool(_CRISIS_SIGNALS.search(text))

        # ── Rule 1: Crisis signals → always crisis, urgency=5 ────────────
        if is_crisis_signal and not is_safe_metaphor:
            if category != "crisis":
                corrections.append(f"category_crisis_promoted:{category!r}→crisis")
                rec["category"] = "crisis"
                category = "crisis"
            if intent not in ("self_harm_intent", "suicidal_thoughts"):
                corrections.append(f"intent_crisis_promoted:{intent!r}→suicidal_thoughts")
                rec["intent"] = "suicidal_thoughts"
                intent = "suicidal_thoughts"
            if urgency < 5:
                corrections.append(f"urgency_crisis_promoted:{urgency}→5")
                rec["urgency"] = 5

        # ── Rule 2: Safe metaphors must NOT be crisis ─────────────────────
        elif is_safe_metaphor and category == "crisis":
            corrections.append("category_metaphor_demoted:crisis→life_stressor")
            rec["category"] = "life_stressor"
            category = "life_stressor"
            if urgency > 2:
                corrections.append(f"urgency_metaphor_capped:{urgency}→2")
                rec["urgency"] = 2

        # ── Rule 3: Category-urgency floor constraints ────────────────────
        if category == "crisis" and rec["urgency"] < 4:
            corrections.append(f"urgency_crisis_floor:{rec['urgency']}→4")
            rec["urgency"] = 4

        if category == "high_distress" and rec["urgency"] < 3:
            corrections.append(f"urgency_distress_floor:{rec['urgency']}→3")
            rec["urgency"] = 3

        if category == "neutral" and rec["urgency"] > 1:
            corrections.append(f"urgency_neutral_cap:{rec['urgency']}→1")
            rec["urgency"] = 1

        # ── Rule 4: High urgency (4-5) on non-distress category → escalate
        if rec["urgency"] >= 4 and category not in ("crisis", "high_distress"):
            # Only escalate if there's real distress language
            if _DISTRESS_SIGNALS.search(text):
                corrections.append(f"category_high_urgency_escalated:{category!r}→high_distress")
                rec["category"] = "high_distress"

        # ── Rule 5: Intent-category alignment ────────────────────────────
        expected_cat = INTENT_TO_CATEGORY.get(rec["intent"])
        if expected_cat and rec["category"] != expected_cat:
            # Don't downgrade crisis if text has distress signals
            if not (is_crisis_signal and expected_cat != "crisis"):
                corrections.append(f"category_intent_aligned:{rec['category']!r}→{expected_cat!r}")
                rec["category"] = expected_cat

        return rec, corrections
