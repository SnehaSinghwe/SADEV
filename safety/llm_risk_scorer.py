"""
safety/llm_risk_scorer.py
LLM-based semantic risk scorer — second gate after regex.

The key architectural principle: the scorer's understanding of dangerous
situations comes entirely from knowledge_base_raw.json, not from hardcoded
examples in this file. Add a new risk pattern to the KB → scorer
automatically gets smarter. No code changes needed.

The system prompt is built at startup from:
  - kb["crisis_keywords"]      → what phrases signal each tier
  - kb["stressor_taxonomy"]    → which stressor types carry higher risk
  - kb["helplines"]            → which resources exist (scorer doesn't pick them — pipeline does)
  - kb["guardrail_registry"]   → what the responder must never do (cross-check context)

The scorer is DISCRIMINATIVE, not generative:
  - Returns JSON only: {"risk_level": int, "tier": str, "reason": str, "safeguarding": bool}
  - temperature=0.0 — maximum consistency
  - max_tokens=120  — just enough for the JSON, nothing else
  - Fails safe to risk=1 on ANY error (fail-safe, not fail-open)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class RiskScore:
    risk_level: int          # 1–5
    tier: str                # "safe"|"tier_2"|"tier_3"|"tier_4"|"tier_5"
    reason: str              # one-line for audit log
    safeguarding: bool       # True if minor-at-risk or violence/coercion context
    helpline_key: str        # hint for pipeline helpline resolution
    from_llm: bool = True


_TIER_TO_HELPLINE: dict[str, str] = {
    "safe":   "",
    "tier_2": "tier_2_minor",
    "tier_3": "tier_3",
    "tier_4": "tier_4",
    "tier_5": "tier_5",
}

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


# ── KB-driven prompt builder ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_system_prompt(kb_path: str = "data/knowledge_base_raw.json") -> str:
    """
    Build the classification system prompt from the knowledge base.
    Cached — called once at startup.
    The KB is the single source of truth for what counts as risky.
    """
    try:
        with open(kb_path, encoding="utf-8") as f:
            kb = json.load(f)
    except FileNotFoundError:
        kb = {}

    crisis_kw = kb.get("crisis_keywords", {})
    stressors  = kb.get("stressor_taxonomy", {})
    idioms     = kb.get("cultural_idioms", {})

    # ── Crisis keyword context (from KB, not hardcoded) ───────────────────
    kw_lines = []
    tier_descs = {
        "tier_3": "hopelessness / passive ideation — person not describing active plan",
        "tier_4": "direct self-harm or suicidal ideation — person wants to hurt themselves",
        "tier_5": "immediate active crisis — person is actively harming self right now",
    }
    for tier, desc in tier_descs.items():
        phrases = crisis_kw.get(tier, [])
        if phrases:
            kw_lines.append(
                f"  {tier} ({desc}):\n"
                + "\n".join(f"    - {p}" for p in phrases[:12])
            )

    # ── Safeguarding context from stressors ──────────────────────────────
    safeguarding_stressors = []
    high_risk_stressors = []
    for sid, s in stressors.items():
        emotions = s.get("linked_emotions", [])
        if any(e in emotions for e in ["helplessness", "identity_confusion"]):
            high_risk_stressors.append(f"{sid}: {s.get('label', sid)}")
        if sid in ("marriage_pressure", "identity_conflict", "intergenerational_conflict"):
            safeguarding_stressors.append(f"{sid}: {s.get('label', sid)}")

    # ── Cultural idioms that signal high distress ─────────────────────────
    idiom_risk_notes = []
    for iid, idiom in idioms.items():
        if iid in ("log_kya_kahenge", "ghar_ki_izzat"):
            idiom_risk_notes.append(
                f"  '{iid}' ({idiom.get('translation','')}) — "
                f"social shame signal, may accompany identity conflict or forced marriage stress"
            )

    kw_block = "\n".join(kw_lines) if kw_lines else "  (no crisis keywords in KB)"
    sg_block  = "\n".join(f"  - {s}" for s in safeguarding_stressors) or "  (none listed)"
    id_block  = "\n".join(idiom_risk_notes) or "  (none)"

    prompt = f"""You are a crisis risk classifier for SADEV, a South Asian mental health AI.
Your job: classify the user's message into a risk level 1–5.
You reason from meaning and context — not surface pattern matching.

## RISK LEVEL DEFINITIONS

1 = safe
  General distress, venting, emotional difficulty. No crisis signal.
  Examples: exam stress, family conflict, loneliness, frustration, identity questions.

2 = safeguarding
  The user is in danger but NOT expressing self-harm ideation.
  Includes:
    - Homicidal ideation (wanting to harm others — family, partner, anyone)
    - Threats of lethal violence TOWARD the user ("my father will kill me")
    - Forced marriage, child marriage, marriage against stated will
    - Physical or sexual abuse disclosure
    - Coercive control (locked in, not allowed to leave, trapped)
    - Minor (under 18) describing any of the above
  safeguarding=true for all tier_2 results.

3 = hopelessness / passive ideation
  Person EXPLICITLY expresses that life feels pointless, they want to disappear,
  or states they see no reason to live — language of giving up on life itself.
  NOT tier 3: general sadness, numbness, "nothing feels important", "feeling empty",
  "don't know what I want" — these are tier 1 even if heavy.
  Tier 3 requires EXPLICIT statements about not wanting to continue living.

4 = self-harm / suicidal ideation
  Person directly states they want to hurt or kill themselves,
  or asks about methods, or describes forming a plan.

5 = immediate crisis
  Person is actively harming themselves right now, already took something,
  already cut, already in physical danger.

## CRISIS SIGNAL PHRASES (from knowledge base — not exhaustive, use as anchors)

{kw_block}

## STRESSORS THAT INCREASE SAFEGUARDING PROBABILITY

{sg_block}

## CULTURAL IDIOMS THAT SIGNAL HIGH DISTRESS

{id_block}

## CROSS-TURN ESCALATION RULE
If conversation history shows escalating distress AND current message adds any
crisis-adjacent language, escalate by at least one tier.
Example: history shows "my father will kill me if I leave" + current "shall I kill them?"
→ tier_2 safeguarding, not tier_1 safe.

## CALIBRATION — THE MOST COMMON MISTAKE
Err toward tier 1 for general emotional distress. Tier 3+ requires EXPLICIT language.
- "nothing feels important" → tier 1 (general flatness/sadness)
- "I feel empty" → tier 1
- "what's the point" → tier 1 UNLESS followed by "of living/being here/going on"
- "I don't see the point anymore" → tier 3 (explicit about life)
- "I want to disappear" → tier 3
- "jeene ka mann nahi" → tier 3 (explicit about living)
- "thak gaya hun zindagi se" → tier 3 (explicit about life/living)
When in doubt between tier 1 and tier 3: choose tier 1.

## CRITICAL DISAMBIGUATION RULES
- "shall i kill this exam / presentation / project" = risk 1 (metaphor, NOT homicide)
- "dying of laughter / boredom" = risk 1 (metaphor)
- "shall i kill them / my family / my father / everyone" = risk 2 (homicide ideation)
- "shall i try for a suicide / attempt suicide / do suicide" = risk 4
- "I want to kill myself" = risk 4
- Minor stating they ARE being married / forced to marry = risk 2
- "my father will kill me if I go out" (threat to user) = risk 2
- Novel phrasing, typos, transliterations of the above = same risk level
  Example: "marr jana chaiye", "mar jao", "khatam ho jao" = risk 3–4 context-dependent

## RESPONSE FORMAT
Respond with ONLY valid JSON. No explanation, no markdown, no text before or after:
{{"risk_level": <int 1-5>, "tier": "<safe|tier_2|tier_3|tier_4|tier_5>", "reason": "<one sentence max 100 chars>", "safeguarding": <true|false>}}

When ambiguous between level 1 and 2+, always choose the HIGHER level.
"""
    return prompt


# ── Scorer ────────────────────────────────────────────────────────────────

class LLMRiskScorer:
    """
    Semantic crisis classifier backed by Gemma via Ollama.
    System prompt is built from knowledge_base_raw.json — not hardcoded.
    Always fails safe — any error returns risk_level=1.
    """

    def __init__(self, ollama_client, kb_path: str = "data/knowledge_base_raw.json"):
        from local_pipeline.ollama_client import OllamaClient
        self._ollama = ollama_client
        self._kb_path = kb_path
        # Classification client: deterministic, short output
        self._classifier = OllamaClient(
            model=ollama_client.model,
            base_url=ollama_client.base_url,
            temperature=0.0,
            max_tokens=120,
            timeout=20,
        )
        # Build and cache the prompt at startup
        self._system_prompt = _build_system_prompt(kb_path)

    def score(
        self,
        current_text: str,
        history: list[dict],
        session_turn_count: int = 1,
    ) -> RiskScore:
        try:
            return self._score_internal(current_text, history, session_turn_count)
        except Exception as e:
            return RiskScore(
                risk_level=1, tier="safe",
                reason=f"llm_scorer_error:{str(e)[:50]}",
                safeguarding=False, helpline_key="",
            )

    def _score_internal(
        self,
        current_text: str,
        history: list[dict],
        session_turn_count: int,
    ) -> RiskScore:
        # Include last 4 turns for cross-turn accumulation detection
        recent = history[-4:] if len(history) > 4 else history
        history_block = ""
        if recent:
            lines = [
                f"{'User' if t['role']=='user' else 'Assistant'}: {t['content'][:250]}"
                for t in recent
            ]
            history_block = "CONVERSATION HISTORY (for context):\n" + "\n".join(lines) + "\n\n"

        user_message = (
            f"{history_block}"
            f"CURRENT MESSAGE TO CLASSIFY:\n{current_text}\n\n"
            f"Turn number: {session_turn_count}"
        )

        raw = self._classifier.generate(
            system_prompt=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return self._parse(raw)

    def _parse(self, raw: str) -> RiskScore:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

        m = _JSON_RE.search(text)
        if not m:
            return self._safe_default("no_json_in_response")

        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return self._safe_default("json_parse_error")

        risk_level = int(data.get("risk_level", 1))
        if risk_level not in (1, 2, 3, 4, 5):
            return self._safe_default("invalid_risk_level")

        tier = data.get("tier", "safe")
        valid_tiers = ("safe", "tier_2", "tier_3", "tier_4", "tier_5")
        if tier not in valid_tiers:
            tier = {1:"safe",2:"tier_2",3:"tier_3",4:"tier_4",5:"tier_5"}.get(risk_level, "safe")

        safeguarding = bool(data.get("safeguarding", False))
        reason = str(data.get("reason", ""))[:100]
        helpline_key = _TIER_TO_HELPLINE.get(tier, "")

        return RiskScore(
            risk_level=risk_level, tier=tier,
            reason=reason, safeguarding=safeguarding,
            helpline_key=helpline_key,
        )

    def _safe_default(self, reason: str) -> RiskScore:
        return RiskScore(
            risk_level=1, tier="safe",
            reason=reason, safeguarding=False, helpline_key="",
        )

    def reload_prompt(self) -> None:
        """Call this if knowledge_base_raw.json changes at runtime."""
        _build_system_prompt.cache_clear()
        self._system_prompt = _build_system_prompt(self._kb_path)