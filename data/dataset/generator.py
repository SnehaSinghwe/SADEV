"""
dataset/generator.py
LLM-powered batch training data generator for SADEV.

Role in pipeline:
  prompt → generator → data/raw/batch_N.jsonl

Design principles:
  - Calls Ollama (local) or Anthropic API (cloud)
  - Outputs raw JSONL — no cleaning here (cleaner.py handles that)
  - Retries on parse failure, up to MAX_RETRIES
  - Each batch is saved atomically so partial runs are resumable
  - Tracks generation metadata (model, timestamp, prompt hash) for audit

Usage:
  from dataset.generator import DatasetGenerator
  gen = DatasetGenerator(backend="ollama")
  batch = gen.generate_batch(intent="venting_distress", n=20)
  gen.save_batch(batch, "data/raw/batch_1.jsonl")
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

# ── Taxonomy (mirrors hinglish_dataset_builder.py) ────────────────────────

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

# Maps intent → (category, urgency_range, language_hint)
INTENT_CONFIG: dict[str, dict] = {
    "venting_distress":    {"category": "emotional",     "urgency": (2, 3), "lang": "hi-en-mix"},
    "anxiety_panic":       {"category": "emotional",     "urgency": (2, 4), "lang": "hi-en-mix"},
    "sadness_low_mood":    {"category": "emotional",     "urgency": (2, 3), "lang": "hi-en-mix"},
    "overwhelm_burnout":   {"category": "high_distress", "urgency": (3, 4), "lang": "en"},
    "loneliness":          {"category": "emotional",     "urgency": (2, 3), "lang": "hi-en-mix"},
    "career_job_stress":   {"category": "life_stressor", "urgency": (2, 3), "lang": "en"},
    "academic_pressure":   {"category": "life_stressor", "urgency": (2, 3), "lang": "hi-en-mix"},
    "financial_stress":    {"category": "life_stressor", "urgency": (2, 4), "lang": "hi-en-mix"},
    "family_conflict":     {"category": "social_stress", "urgency": (2, 3), "lang": "hi-en-mix"},
    "relationship_issue":  {"category": "social_stress", "urgency": (2, 3), "lang": "en"},
    "marriage_pressure":   {"category": "social_stress", "urgency": (2, 3), "lang": "hi-en-mix"},
    "social_reputation":   {"category": "social_stress", "urgency": (2, 3), "lang": "hi-en-mix"},
    "conflict_exposure":   {"category": "social_stress", "urgency": (2, 3), "lang": "hi-en-mix"},
    "fear_of_consequence": {"category": "social_stress", "urgency": (2, 3), "lang": "en"},
    "advice_seeking":      {"category": "problem_solving","urgency": (1, 2), "lang": "en"},
    "decision_conflict":   {"category": "problem_solving","urgency": (1, 2), "lang": "en"},
    "planning_next_steps": {"category": "problem_solving","urgency": (1, 2), "lang": "en"},
    "sleep_issue":         {"category": "behavioral",    "urgency": (2, 3), "lang": "en"},
    "focus_issue":         {"category": "behavioral",    "urgency": (2, 3), "lang": "en"},
    "productivity_block":  {"category": "behavioral",    "urgency": (2, 3), "lang": "hi-en-mix"},
    "hopelessness":        {"category": "high_distress", "urgency": (3, 4), "lang": "hi-en-mix"},
    "emotional_exhaustion":{"category": "high_distress", "urgency": (3, 4), "lang": "en"},
    "self_harm_intent":    {"category": "crisis",        "urgency": (4, 5), "lang": "en"},
    "suicidal_thoughts":   {"category": "crisis",        "urgency": (5, 5), "lang": "en"},
    "greeting":            {"category": "neutral",       "urgency": (1, 1), "lang": "hi-en-mix"},
    "casual_chat":         {"category": "neutral",       "urgency": (1, 1), "lang": "en"},
}

MAX_RETRIES = 3
_JSON_RE = re.compile(r"\[.*?\]", re.DOTALL)


# ── Data record ───────────────────────────────────────────────────────────

@dataclass
class RawRecord:
    text: str
    intent: str
    category: str
    urgency: int
    language: str = "hi-en-mix"
    # Generation metadata — stripped before training, kept in raw/
    _gen_model: str = ""
    _gen_batch_id: str = ""
    _gen_prompt_hash: str = ""
    _gen_ts: str = ""

    def to_jsonl_dict(self) -> dict:
        return {
            "text":     self.text,
            "intent":   self.intent,
            "category": self.category,
            "urgency":  self.urgency,
            "language": self.language,
            "_meta": {
                "model":        self._gen_model,
                "batch_id":     self._gen_batch_id,
                "prompt_hash":  self._gen_prompt_hash,
                "generated_at": self._gen_ts,
            },
        }


# ── Prompt builder ────────────────────────────────────────────────────────

def _build_prompt(intent: str, n: int, language: str) -> tuple[str, str]:
    cfg = INTENT_CONFIG.get(intent, {})
    cat = cfg.get("category", "emotional")
    u_lo, u_hi = cfg.get("urgency", (2, 3))

    lang_map = {
        "hi-en-mix": "Hinglish — mix Hindi and English WITHIN sentences (Roman script). "
                     "e.g. 'yaar I'm so stressed about placement season'",
        "en":        "Informal South Asian English (Indian register, colloquial). "
                     "e.g. 'my parents are putting so much pressure on me for this exam'",
        "hi":        "Hindi in Roman script only. e.g. 'mujhe bahut takleef ho rahi hai aajkal'",
    }
    lang_instruction = lang_map.get(language, lang_map["en"])

    # Crisis safety constraint
    crisis_note = ""
    if cat == "crisis":
        crisis_note = (
            "\nCRITICAL SAFETY RULE: Never include methods, step-by-step instructions, "
            "or detailed plans for self-harm. Ideation only — never facilitation."
        )

    system = f"""You are a specialist NLU training-data generator for SADEV — a mental health AI for Indian users.
Generate realistic, natural messages that a real Indian person would type in a chat app.
Messages should sound like WhatsApp — casual, authentic, sometimes with typos, never formal.
{crisis_note}
DO NOT include any AI meta-language ("as a user", "generate", "example", "training data")."""

    user = f"""Generate exactly {n} unique training examples.

TARGET:
  intent: {intent}
  category: {cat}
  urgency range: {u_lo}–{u_hi} (out of 5)
  language: {language} — {lang_instruction}

URGENCY SCALE:
  1 = casual / neutral
  2 = mild stress or discomfort
  3 = moderate distress
  4 = high distress — significant emotional weight
  5 = crisis — immediate danger signals

Output ONLY a JSON array. No markdown, no explanation, no preamble:
[
  {{
    "text": "<the message>",
    "intent": "{intent}",
    "category": "{cat}",
    "urgency": <{u_lo}-{u_hi}>,
    "language": "{language}"
  }}
]

Vary: length (5–150 words), formality, vocabulary. No two examples identical.
Include authentic Indian context: parents, exams, JEE/NEET, izzat, arranged marriage, job pressure, etc."""
    return system, user


# ── Backends ──────────────────────────────────────────────────────────────

def _call_ollama(system: str, user: str, model: str = "gemma2",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 300) -> str:
    """
    Call Ollama using streaming so the HTTP connection stays alive
    for the full generation — avoids read timeout on slow local models.
    timeout is per-chunk (not total), so 300s is very generous.
    """
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": True,                          # ← streaming keeps connection alive
        "options": {"temperature": 0.9, "num_predict": 2048},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                chunks.append(token)
            if chunk.get("done", False):
                break
    return "".join(chunks).strip()


def _call_anthropic(system: str, user: str, api_key: str,
                    model: str = "claude-sonnet-4-20250514") -> str:
    import urllib.request
    payload = json.dumps({
        "model": model,
        "max_tokens": 2048,
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


# ── Parser ────────────────────────────────────────────────────────────────

def _parse_records(raw: str, intent: str, category: str,
                   batch_id: str, model_name: str, prompt_hash: str) -> list[RawRecord]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_RE.search(cleaned)
        if not m:
            return []
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return []
    if not isinstance(items, list):
        return []

    ts = datetime.utcnow().isoformat()
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if len(text) < 5:
            continue
        records.append(RawRecord(
            text=text,
            intent=str(item.get("intent", intent)),
            category=str(item.get("category", category)),
            urgency=max(1, min(5, int(item.get("urgency", 2)))),
            language=str(item.get("language", "en")),
            _gen_model=model_name,
            _gen_batch_id=batch_id,
            _gen_prompt_hash=prompt_hash,
            _gen_ts=ts,
        ))
    return records


# ── Main class ────────────────────────────────────────────────────────────

class DatasetGenerator:
    """
    Generates raw training batches via Ollama (default) or Anthropic API.

    Args:
        backend:     "ollama" | "anthropic"
        model:       model name for the chosen backend
        ollama_url:  Ollama base URL (ignored for anthropic backend)
        api_key:     Anthropic API key (ignored for ollama backend)
        timeout:     per-chunk read timeout in seconds (Ollama only, default 300)
        n_per_call:  max examples to request per LLM call — large n is auto-chunked
                     (default 10, keeps each call fast and avoids timeouts)
    """

    def __init__(
        self,
        backend: Literal["ollama", "anthropic"] = "ollama",
        model: str = "gemma2",
        ollama_url: str = "http://localhost:11434",
        api_key: str = "",
        timeout: int = 300,
        n_per_call: int = 10,
    ):
        self.backend    = backend
        self.model      = model
        self.ollama_url = ollama_url
        self.api_key    = api_key
        self.timeout    = timeout
        self.n_per_call = n_per_call   # auto-chunk large requests

    def generate_batch(
        self,
        intent: str,
        n: int = 20,
        language: str | None = None,
    ) -> list[RawRecord]:
        """
        Generate n examples for the given intent.
        Automatically chunks into calls of n_per_call to avoid timeouts.
        Returns RawRecord list — not yet cleaned or validated.
        """
        if intent not in VALID_INTENTS:
            raise ValueError(f"Unknown intent: {intent!r}. Valid: {sorted(VALID_INTENTS)}")

        cfg = INTENT_CONFIG.get(intent, {})
        lang = language or cfg.get("lang", "en")
        cat  = cfg.get("category", "emotional")

        # Auto-chunk: split n into smaller calls
        all_records: list[RawRecord] = []
        remaining = n
        while remaining > 0:
            chunk_n = min(remaining, self.n_per_call)
            records = self._generate_chunk(intent, chunk_n, lang, cat)
            all_records.extend(records)
            remaining -= chunk_n
            if remaining > 0:
                time.sleep(0.5)   # brief pause between chunks

        return all_records

    def _generate_chunk(
        self,
        intent: str,
        n: int,
        lang: str,
        cat: str,
    ) -> list[RawRecord]:
        """Generate a single chunk of n examples with retry logic."""
        system, user = _build_prompt(intent, n, lang)
        prompt_hash = hashlib.md5((system + user).encode()).hexdigest()[:8]
        batch_id    = str(uuid.uuid4())[:8]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self.backend == "anthropic":
                    if not self.api_key:
                        raise ValueError("api_key required for anthropic backend")
                    raw = _call_anthropic(system, user, self.api_key, self.model)
                else:
                    raw = _call_ollama(system, user, self.model, self.ollama_url, self.timeout)

                records = _parse_records(raw, intent, cat, batch_id, self.model, prompt_hash)
                if records:
                    return records

                print(f"  [generator] attempt {attempt}: parse yielded 0 records, retrying...")
                time.sleep(2)

            except Exception as e:
                print(f"  [generator] attempt {attempt} error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(3)
        return []

    def generate_multi_batch(
        self,
        intent_list: list[str],
        n_per_intent: int = 20,
        delay_secs: float = 1.5,
    ) -> dict[str, list[RawRecord]]:
        """
        Generate batches for multiple intents in sequence.
        Returns {intent: [RawRecord, ...]}
        """
        results: dict[str, list[RawRecord]] = {}
        for intent in intent_list:
            print(f"  [generator] generating {n_per_intent}x '{intent}'...")
            batch = self.generate_batch(intent, n=n_per_intent)
            results[intent] = batch
            print(f"  [generator] got {len(batch)} records for '{intent}'")
            time.sleep(delay_secs)
        return results

    @staticmethod
    def save_batch(records: list[RawRecord], path: str) -> int:
        """Save records to a JSONL file. Returns count written."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
                written += 1
        return written

    @staticmethod
    def load_batch(path: str) -> list[dict]:
        """Load a raw JSONL batch file."""
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records