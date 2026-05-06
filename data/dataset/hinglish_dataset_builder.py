"""
dataset/hinglish_dataset_builder.py
Generates culturally-grounded Hinglish / Hindi emotional expression data
for fine-tuning the emotion detector and testing the pipeline.

Two modes:
  --generate   Call Claude API to synthesise labelled examples
  --check      Quality-check an existing CSV
  --export     Export to CSV ready for annotation

The schema mirrors labelling_schema.json exactly so data feeds directly
into the MuRIL training pipeline.

Usage:
    # Generate 50 family-pressure examples in Hinglish
    python -m dataset.hinglish_dataset_builder --generate --intent express_family_pressure --n 50

    # Fill gaps across all classes
    python -m dataset.hinglish_dataset_builder --generate --fill_gaps

    # Quality check
    python -m dataset.hinglish_dataset_builder --check --input data/hinglish_seed.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path


# ── Schema ─────────────────────────────────────────────────────────────────

INTENTS = [
    "express_family_pressure", "express_academic_pressure",
    "express_social_judgment", "express_marriage_pressure",
    "express_identity_conflict", "express_financial_stress",
    "seek_validation", "vent_frustration", "ask_for_advice",
    "share_positive_news", "crisis_signal", "unclear",
]

EMOTIONS = [
    "conflict", "guilt", "anxiety", "shame", "anger", "grief",
    "loneliness", "fear", "inadequacy", "identity_confusion", "relief",
    "hope", "neutral", "overwhelm", "resentment", "helplessness",
    "love", "frustration", "confusion", "sadness",
]

STRESSOR_TYPES = [
    "academic_pressure", "family_career_expectations", "social_judgment",
    "marriage_pressure", "intergenerational_conflict", "identity_conflict",
    "financial_stress", "multiple", "none",
]

LANGUAGES = ["hi-en-mix", "en", "hi", "bn-en-mix", "bn"]

MIN_TARGETS = {
    "express_family_pressure": 500, "express_academic_pressure": 500,
    "express_social_judgment": 400, "express_marriage_pressure": 400,
    "express_identity_conflict": 300, "express_financial_stress": 300,
    "seek_validation": 400, "vent_frustration": 400,
    "ask_for_advice": 300, "share_positive_news": 200,
    "crisis_signal": 300, "unclear": 200,
}

INTENT_STRESSOR_MAP = {
    "express_family_pressure":   "family_career_expectations",
    "express_academic_pressure": "academic_pressure",
    "express_social_judgment":   "social_judgment",
    "express_marriage_pressure": "marriage_pressure",
    "express_identity_conflict": "identity_conflict",
    "express_financial_stress":  "financial_stress",
}

CSV_FIELDS = [
    "id", "text", "language", "script", "intent", "emotions",
    "urgency_level", "stressor_type", "cultural_markers_present",
    "contains_code_mix", "annotator_id", "annotator_confidence",
    "second_annotator_id", "agreed", "source", "split",
]

# ── Metaphor / unsafe patterns ─────────────────────────────────────────────

_METAPHORS = re.compile(
    r"kill(ing)?\s+(it|this|exam|test)|dying\s+of\s+laughter|"
    r"dead\s+(tired|serious)|murder(ing)?\s+(it|this)",
    re.IGNORECASE,
)
_UNSAFE_CRISIS = re.compile(
    r"(step\s+by\s+step|instructions\s+for).{0,50}(kill|harm|suicide|overdose)|"
    r"took\s+\d+\s+pills",
    re.IGNORECASE,
)
_TEMPLATED = re.compile(
    r"(write\s+a\s+message|generate\s+an\s+example|here\s+is\s+an\s+example|"
    r"as\s+a\s+user|training\s+data|annotation|llm|gpt|claude)",
    re.IGNORECASE,
)
_HINGLISH_MARKERS = re.compile(
    r"\b(hai|hun|hain|meri|mujhe|main|aur|lekin|kya|nahi|bhi|toh|bahut|"
    r"ghar|mummy|papa|chahta|chahti|karna|karun|izzat|shaadi|rishta)\b",
    re.IGNORECASE,
)
_ENGLISH_MARKERS = re.compile(
    r"\b(the|a|an|is|are|my|your|we|they|this|and|but|not|feel|think|want)\b",
    re.IGNORECASE,
)


# ── Generation ─────────────────────────────────────────────────────────────

def _build_generation_prompt(
    intent: str,
    language: str,
    n: int,
    stressor_type: str,
    urgency: int | None = None,
) -> tuple[str, str]:
    lang_map = {
        "hi-en-mix": "Hinglish — mix Hindi and English WITHIN individual sentences in Roman script.",
        "en":        "Informal South Asian English (Indian register, colloquial).",
        "hi":        "Hindi in Devanagari script.",
    }
    lang_instruction = lang_map.get(language, lang_map["en"])
    urgency_note = ""
    if urgency:
        scale = {1: "general stress", 2: "elevated distress, hopeless phrasing",
                 3: "clear hopelessness — NO direct self-harm language",
                 4: "direct self-harm ideation", 5: "immediate crisis"}
        urgency_note = f"\nUrgency target: {urgency}/5 — {scale.get(urgency, '')}"

    system = f"""You are a specialist NLU training data generator for SADEV, a South Asian mental health AI.
Generate realistic synthetic examples of what a real South Asian person would type to a mental health chatbot.
Each example must sound authentic — typed informally, code-mixed, with typos fine.

Schema rules:
- intent: exactly one from the SADEV schema
- emotions: 1–4, comma-separated, only from the SADEV emotion vocabulary
- urgency: 1–5 scale
- contains_code_mix: true only if two languages mixed WITHIN a single sentence
- cultural_markers_present: any of log_kya_kahenge, izzat, ghar_ki_izzat, JEE, NEET, arranged_marriage, beta_padh_le

CRITICAL: Never generate methods, step-by-step instructions, or detailed plans for self-harm.
CRITICAL: Never write AI meta-language ("as a user", "generate", "example message").
METAPHOR RULE: "I want to kill this exam" = urgency 1, NOT crisis.
"""
    user = f"""Generate exactly {n} unique synthetic training examples.

Target:
- intent: {intent}
- language: {language} — {lang_instruction}
- stressor_type: {stressor_type}{urgency_note}

Output ONLY a JSON array (no markdown, no preamble):
[
  {{
    "text": "<message exactly as user would type>",
    "language": "{language}",
    "script": "latin_roman",
    "intent": "{intent}",
    "emotions": "<comma-separated>",
    "urgency_level": <1-5>,
    "stressor_type": "{stressor_type}",
    "cultural_markers_present": "<comma-separated or empty>",
    "contains_code_mix": <true|false>
  }}
]

Diversity: vary length (1–5 sentences), formality, vocabulary. No two examples identical."""
    return system, user


def _call_claude(system: str, user: str, api_key: str) -> str:
    import urllib.request
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "temperature": 0.9,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
        return data["content"][0]["text"].strip()


def _parse_response(raw: str, intent: str, language: str, stressor: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
        if not m:
            return []
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return []
    if not isinstance(items, list):
        return []

    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if len(text) < 10:
            continue
        emotions_raw = item.get("emotions", "neutral")
        if isinstance(emotions_raw, list):
            emotions_raw = ",".join(emotions_raw)
        valid_em = [e.strip() for e in str(emotions_raw).split(",") if e.strip() in EMOTIONS]
        if not valid_em:
            valid_em = ["neutral"]
        rows.append({
            "text": text,
            "language": item.get("language", language),
            "script": item.get("script", "latin_roman"),
            "intent": item.get("intent", intent),
            "emotions": ",".join(valid_em[:4]),
            "urgency_level": max(1, min(5, int(item.get("urgency_level", 1)))),
            "stressor_type": item.get("stressor_type", stressor),
            "cultural_markers_present": str(item.get("cultural_markers_present", "") or "").strip(),
            "contains_code_mix": str(item.get("contains_code_mix", False)).lower() in ("true", "1"),
            "annotator_id": "synthetic_claude",
            "annotator_confidence": 0.85,
            "second_annotator_id": "",
            "agreed": "",
            "source": "synthetic_claude",
            "split": "train",
        })
    return rows


# ── Quality check ──────────────────────────────────────────────────────────

def _fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.lower().strip())
    norm = re.sub(r"[^\w\s]", "", norm)
    return hashlib.md5(norm.encode()).hexdigest()


def quality_check(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (accepted, rejected)."""
    accepted, rejected = [], []
    seen_fps: set[str] = set()
    seen_texts: list[str] = []

    for row in rows:
        text = str(row.get("text", ""))
        reason = ""

        if len(text) < 10:
            reason = "too_short"
        elif len(text) > 1500:
            reason = "too_long"
        elif _TEMPLATED.search(text):
            reason = "templated_text"
        elif _UNSAFE_CRISIS.search(text):
            reason = "unsafe_crisis_content"
        elif _METAPHORS.search(text) and int(row.get("urgency_level", 1)) >= 3:
            reason = "metaphor_crisis_mismatch"
        elif row.get("intent") not in INTENTS:
            reason = "invalid_intent"
        else:
            fp = _fingerprint(text)
            if fp in seen_fps:
                reason = "exact_duplicate"
            else:
                # Near-duplicate check
                words = set(text.lower().split())
                for prev in seen_texts[-200:]:
                    prev_words = set(prev.lower().split())
                    if words and prev_words:
                        overlap = len(words & prev_words) / max(len(words), len(prev_words))
                        if overlap >= 0.85:
                            reason = "near_duplicate"
                            break

        if reason:
            rejected.append({**row, "rejection_reason": reason})
        else:
            seen_fps.add(_fingerprint(text))
            seen_texts.append(text)
            accepted.append(row)

    return accepted, rejected


# ── CSV helpers ────────────────────────────────────────────────────────────

def _assign_ids(rows: list[dict], output_path: str, split: str = "train") -> list[dict]:
    prefix = f"sadev_{split}_"
    max_idx = 0
    if Path(output_path).exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rid = row.get("id", "")
                    if rid.startswith(prefix):
                        try:
                            max_idx = max(max_idx, int(rid.replace(prefix, "")))
                        except ValueError:
                            pass
        except Exception:
            pass
    for i, row in enumerate(rows):
        row["id"] = f"{prefix}{max_idx + i + 1:04d}"
        row["split"] = split
    return rows


def _append_csv(rows: list[dict], path: str) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(path).exists() or Path(path).stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def _count_by_intent(path: str) -> dict[str, int]:
    counts = {i: 0 for i in INTENTS}
    if not Path(path).exists():
        return counts
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = row.get("intent", "")
            if k in counts:
                counts[k] += 1
    return counts


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SADEV Hinglish dataset builder")
    p.add_argument("--generate",   action="store_true")
    p.add_argument("--check",      action="store_true")
    p.add_argument("--fill_gaps",  action="store_true")
    p.add_argument("--intent",     choices=INTENTS)
    p.add_argument("--language",   default="hi-en-mix", choices=LANGUAGES)
    p.add_argument("--n",          type=int, default=20)
    p.add_argument("--urgency",    type=int, choices=[1, 2, 3, 4, 5])
    p.add_argument("--output",     default="data/hinglish_dataset.csv")
    p.add_argument("--input",      default=None)
    p.add_argument("--api_key",    default=os.getenv("ANTHROPIC_API_KEY", ""))
    args = p.parse_args()

    if args.check:
        path = args.input or args.output
        if not Path(path).exists():
            print(f"File not found: {path}")
            return
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        accepted, rejected = quality_check(rows)
        print(f"\nQuality check: {len(rows)} total | {len(accepted)} accepted | {len(rejected)} rejected")
        reasons = {}
        for r in rejected:
            reasons[r.get("rejection_reason", "unknown")] = reasons.get(r.get("rejection_reason", "unknown"), 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<40} {count}")
        return

    if not args.api_key:
        print("Set ANTHROPIC_API_KEY or pass --api_key")
        return

    if args.fill_gaps:
        counts = _count_by_intent(args.output)
        gaps = [(intent, MIN_TARGETS[intent] - counts[intent])
                for intent in INTENTS if counts[intent] < MIN_TARGETS[intent]]
        print(f"Gaps to fill: {len(gaps)} intents")
        for intent, needed in sorted(gaps, key=lambda x: -x[1]):
            print(f"  {intent:<40} need {needed} more")
        for intent, needed in gaps:
            if intent == "crisis_signal" and needed > 50:
                needed = 50  # cap crisis per run
            stressor = INTENT_STRESSOR_MAP.get(intent, "none")
            sys_p, usr_p = _build_generation_prompt(intent, args.language, min(needed, 20), stressor)
            try:
                raw = _call_claude(sys_p, usr_p, args.api_key)
                rows = _parse_response(raw, intent, args.language, stressor)
                accepted, _ = quality_check(rows)
                accepted = _assign_ids(accepted, args.output)
                w = _append_csv(accepted, args.output)
                print(f"  {intent}: wrote {w} rows")
                time.sleep(1.5)
            except Exception as e:
                print(f"  {intent}: error — {e}")
        return

    if args.generate:
        if not args.intent:
            print("--intent required with --generate")
            return
        stressor = INTENT_STRESSOR_MAP.get(args.intent, "none")
        sys_p, usr_p = _build_generation_prompt(
            args.intent, args.language, args.n, stressor, args.urgency
        )
        print(f"Generating {args.n} examples for {args.intent} / {args.language}...")
        raw = _call_claude(sys_p, usr_p, args.api_key)
        rows = _parse_response(raw, args.intent, args.language, stressor)
        accepted, rejected = quality_check(rows)
        accepted = _assign_ids(accepted, args.output)
        w = _append_csv(accepted, args.output)
        print(f"Generated {len(rows)} | Accepted {len(accepted)} | Rejected {len(rejected)} | Wrote {w}")
        for row in accepted[:3]:
            print(f"  [{row['intent']}|{row['language']}] {row['text'][:80]}")


if __name__ == "__main__":
    main()
