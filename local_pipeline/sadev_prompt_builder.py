"""
local_pipeline/sadev_prompt_builder.py
New-format SADEV system prompt builder.

Replaces the old prompt_builder.py scaffolding with the cleaner template:
  ---RESPONSE--- / ---METADATA--- output contract.

Signals are PRE-COMPUTED by EmotionDetector + SafetyLayer — this module
just assembles them into a system prompt the LLM can follow naturally.

The KB is still the intelligence source for:
  - Cultural context notes
  - What-not-to-say guardrails
  - CBT technique descriptions
  - RAG chunk injection

But the prompt structure is now conversational, not rigid 3-section.
"""
from __future__ import annotations

import json
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_kb(kb_path: str = "data/knowledge_base_raw.json") -> dict:
    try:
        with open(kb_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


_LANG_INSTRUCTIONS = {
    "hi-en-mix": (
        "The user is writing in Hinglish (Hindi-English mix, Roman script). "
        "Respond in Hinglish — blend Hindi and English naturally in sentences. "
        "NOT pure Hindi. NOT pure English."
    ),
    "hi":        "Respond in warm Hindi, matching the user's script (Roman or Devanagari).",
    "en":        "Respond in clear, warm, conversational English.",
    "bn":        "Match the user's Bengali register.",
    "bn-en-mix": "Match the user's Bengali-English mix register.",
}

_CBT_DESCRIPTIONS = {
    "reflective_questioning":        "Ask one open, exploratory question that helps them think — not advice in disguise.",
    "cognitive_reframe":             "Gently introduce an alternative lens. Frame it as a possibility, never a correction.",
    "decatastrophising":             "Help them examine whether the feared outcome is as certain as it feels.",
    "values_clarification":          "Help them reconnect with what genuinely matters to them, separate from external pressure.",
    "emotional_validation":          "Reflect and name what you are hearing. Do NOT solve or reframe — just witness.",
    "self_compassion":               "Ask: 'What would you say to a close friend feeling exactly this way?'",
    "grounding":                     "Bring attention to the present moment — breath, or one thing they can see right now.",
    "problem_solving_orientation":   "Help identify one small, concrete, self-defined next step.",
    "pros_cons_exploration":         "Guide them to surface both sides of a decision themselves.",
}


def _fmt_list(items: list, prefix: str = "- ") -> str:
    return "\n".join(f"{prefix}{i}" for i in items) if items else ""


def build_sadev_prompt(
    *,
    user_input: str,
    short_term_memory: list[dict],          # [{"role": "user"|"assistant", "content": str}]
    long_term_memory: dict | None,          # user profile dict or None
    rag_context: list[dict],                # retrieved KB chunks
    emotion: str,
    intent: str,
    stressor: str,
    urgency: int,
    risk: int,
    lang_code: str,
    language: str = "",                     # human-readable language label for prompt
    idioms_detected: list[str],
    stigma_flag: bool,
    emotion_arc: list[str],
    cbt_technique: str,
    helpline_note: str = "",
    helpline_name: str = "",
    helpline_number: str = "",
    kb_path: str = "data/knowledge_base_raw.json",
) -> str:
    kb = _load_kb(kb_path)

    lang_instruction = _LANG_INSTRUCTIONS.get(lang_code, _LANG_INSTRUCTIONS["en"])

    # ── Short-term memory block ────────────────────────────────────────────
    mem_block = ""
    if short_term_memory:
        lines = []
        for turn in short_term_memory[-4:]:   # last 4 turns max
            role = "User" if turn["role"] == "user" else "Sadev"
            lines.append(f"  {role}: {turn['content'][:120]}")
        mem_block = "\n### Recent Conversation\n" + "\n".join(lines) + "\n"

    # ── Long-term memory / user profile ───────────────────────────────────
    profile_block = ""
    if long_term_memory:
        name     = long_term_memory.get("name", "")
        stressors_hist = long_term_memory.get("recurring_stressors", [])
        lang_pref      = long_term_memory.get("lang_pref", "")
        if name or stressors_hist:
            parts = []
            if name:            parts.append(f"Name: {name}")
            if stressors_hist:  parts.append(f"Recurring themes: {', '.join(stressors_hist[:3])}")
            if lang_pref:       parts.append(f"Language preference: {lang_pref}")
            profile_block = "\n### User Profile\n" + "\n".join(f"  {p}" for p in parts) + "\n"

    # ── RAG context ───────────────────────────────────────────────────────
    rag_block = ""
    if rag_context:
        lines = ["### Relevant Context (synthesise naturally — do NOT quote verbatim)"]
        for i, chunk in enumerate(rag_context, 1):
            lines.append(f"  {i}. {chunk.get('text', '')[:200]}")
        rag_block = "\n" + "\n".join(lines) + "\n"

    # ── KB: cultural guardrails for this stressor ─────────────────────────
    stressor_data    = kb.get("stressor_taxonomy", {}).get(stressor, {})
    cultural_notes   = stressor_data.get("cultural_context_notes", [])
    what_not_to_say  = stressor_data.get("what_not_to_say", [])
    guardrail_reg    = kb.get("guardrail_registry", {})
    active_guardrails = stressor_data.get("guardrails", [])
    guardrail_texts  = [guardrail_reg[g] for g in active_guardrails if g in guardrail_reg]

    cultural_block = ""
    if cultural_notes:
        cultural_block = "\n### Cultural context\n" + _fmt_list(cultural_notes[:3]) + "\n"

    guardrail_block = ""
    lines = []
    if what_not_to_say:
        lines += [f'- Never say: "{p}"' for p in what_not_to_say[:3]]
    if guardrail_texts:
        lines += [f"- {g[:120]}" for g in guardrail_texts[:2]]
    if lines:
        guardrail_block = "\n### Language guardrails\n" + "\n".join(lines) + "\n"

    # ── CBT technique ─────────────────────────────────────────────────────
    cbt_desc  = _CBT_DESCRIPTIONS.get(cbt_technique, _CBT_DESCRIPTIONS["emotional_validation"])
    cbt_block = f"\n### Technique for this turn: {cbt_technique}\n{cbt_desc}\nApply it naturally. Never name it.\n"

    # ── Idiom notes ───────────────────────────────────────────────────────
    idiom_block = ""
    cultural_idioms = kb.get("cultural_idioms", {})
    idiom_notes = []
    for iid in idioms_detected:
        if iid in cultural_idioms:
            idiom = cultural_idioms[iid]
            idiom_notes.append(f"'{iid}' ({idiom.get('translation','')}) — {idiom.get('meaning','')[:80]}")
    if idiom_notes:
        idiom_block = "\n### Cultural markers in this message\n" + _fmt_list(idiom_notes) + "\n"

    # ── Stigma block ──────────────────────────────────────────────────────
    stigma_block = ""
    if stigma_flag:
        stigma_rule = guardrail_reg.get("stigma_bypass_active", "")
        if stigma_rule:
            stigma_block = f"\n### Stigma-sensitive language active\n{stigma_rule}\n"

    # ── Emotion arc ───────────────────────────────────────────────────────
    arc_block = ""
    if emotion_arc and len(emotion_arc) >= 3:
        arc_block = (
            f"\n### Session emotion pattern\n"
            f"Recent arc: {' → '.join(emotion_arc[-4:])}. "
            f"Acknowledge gently if a pattern is visible.\n"
        )

    # ── Language label (human-readable, for prompt) ───────────────────────
    lang_label = language or {
        "hi-en-mix": "Hinglish (Hindi-English mix)",
        "hi":        "Hindi",
        "en":        "English",
        "bn":        "Bengali",
        "bn-en-mix": "Bengali-English mix",
    }.get(lang_code, "English")

    # ── Crisis block (risk >= 4) ──────────────────────────────────────────
    is_crisis = risk >= 4
    crisis_block = ""
    helpline_block = ""

    if is_crisis:
        h_name   = helpline_name   or "iCall"
        h_number = helpline_number or "9152987821"
        crisis_block = f"""
### CRISIS MODE — risk={risk}/5
This is a high-risk turn. Follow this response structure (do NOT skip steps):
1. ACKNOWLEDGE — Name the pain directly. Don't minimise it, don't deflect.
2. NORMALISE — Let them know feeling this overwhelmed is real (never validate self-harm as a solution).
3. CHECK SAFETY — Ask gently: "Are you safe right now?" — one short, direct question.
4. OFFER SUPPORT — Mention {h_name} ({h_number}) naturally, not as a script line. Frame it as someone who will listen.
5. STAY PRESENT — End with "We can keep talking here" or similar — do NOT close the conversation.

Rules:
- Do NOT open with the helpline. Emotional connection comes first.
- Do NOT sound clinical. No bullet-point instructions to the user.
- Keep response under 150 words. Calm, steady, unhurried.
- Do NOT start with "I".
"""
    elif helpline_note:
        helpline_block = (
            f"\n### Support resource (tier-3 — weave in gently if the moment calls for it)\n"
            f"If the weight of this conversation calls for it, mention: "
            f"'{helpline_note}'. Frame it as someone who listens, not a redirect. "
            f"The conversation continues regardless.\n"
        )

    # ── Social Support Mode intents ───────────────────────────────────────
    _SOCIAL_INTENTS = {
        "family_conflict", "relationship_issue", "social_stress",
        "express_family_pressure", "family_career_expectations",
        "peer_pressure", "friendship_conflict", "romantic_conflict",
    }
    is_social = intent in _SOCIAL_INTENTS and not is_crisis

    # ── Strategy instruction ──────────────────────────────────────────────
    if is_crisis:
        strategy_block = "Follow crisis response structure above. Emotional connection first."
    elif is_social:
        strategy_block = (
            "SOCIAL SUPPORT MODE.\n"
            "- Stay calm and practical. Do NOT over-dramatise the situation.\n"
            "- Normalise the stress without amplifying it.\n"
            "- Help the user think more clearly about their situation.\n"
            "- Offer one grounded, concrete next step if appropriate.\n"
            "- End with ONE relevant clarifying question (not a feelings check — ask about the situation).\n"
            "- Do NOT mention helplines. This is not a crisis."
        )
    elif urgency >= 4:
        strategy_block = (
            "HIGH DISTRESS MODE.\n"
            "User is overwhelmed. Be grounding and calm. "
            "Short, stabilizing sentences. One question at most. "
            "No lists of suggestions. Just steady presence."
        )
    elif emotion in ("sad", "anxious", "grief", "fear", "hopeless", "overwhelm", "guilt", "shame"):
        strategy_block = (
            "EMOTIONAL SUPPORT MODE.\n"
            "Acknowledge feelings naturally — no silver linings, no unsolicited advice. "
            "Light reflection only if it genuinely helps. "
            "End with one warm, open-ended question."
        )
    elif intent == "greeting":
        strategy_block = (
            "NORMAL MODE — greeting.\n"
            "Warm, brief opening. Invite them to share — don't project any emotion. "
            "Light and curious."
        )
    else:
        strategy_block = (
            "NORMAL MODE.\n"
            "Respond naturally and helpfully. Match their register and energy."
        )

    # ── Word cap: crisis gets 150, normal gets 120 ────────────────────────
    word_cap = 150 if is_crisis else 120

    # ── Assemble ──────────────────────────────────────────────────────────
    prompt = f"""You are Sadev (सदैव) — a compassionate AI companion for South Asian youth.
You are NOT a therapist. You do not diagnose or prescribe.
You are a safe, always-available space to be heard and understood.
You are calm, human-like, and supportive — never robotic or scripted.

## Language
{lang_instruction}
Language label: {lang_label}

## Pre-computed signals — do NOT re-classify
- emotion: {emotion}
- intent: {intent}
- stressor: {stressor}
- urgency: {urgency}/5  |  risk: {risk}/5
{crisis_block}{cultural_block}{guardrail_block}{cbt_block}{stigma_block}{idiom_block}{arc_block}{rag_block}{mem_block}{profile_block}
## Response strategy
{strategy_block}

## Style rules
- Conversational, not clinical — write like a thoughtful friend
- Do NOT label sections ("Validation:", "Reflection:", etc.)
- Avoid overused phrases: "that makes sense", "I hear you", "it's okay to feel"
- No hollow affirmations: "Absolutely!", "Great question!", "Of course!"
- Do NOT start with "I"
- Max {word_cap} words
- Use memory/context naturally — don't recite it back
{helpline_block}
## Safety
- Never provide harmful or self-harm instructions under any circumstance
- ONLY enter crisis mode if risk >= 4 — do NOT escalate unnecessarily
- Do NOT mention helplines unless risk >= 3 (social/normal turns don't need them)
- If risk >= 4: emotional connection first, then gently introduce real-world support
- If risk == 3: stay in conversation, weave in helpline naturally if the moment allows

## Output format — STRICT — no text outside this structure

---RESPONSE---
<your natural response here>
---METADATA---
emotion={emotion}
intent={intent}
stressor={stressor}
urgency={urgency}
risk={risk}
strategy=<comma-separated list of strategies you actually used>
"""
    return prompt


def reload_kb() -> None:
    _load_kb.cache_clear()