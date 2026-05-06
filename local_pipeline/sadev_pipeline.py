"""
local_pipeline/sadev_pipeline.py
Drop-in replacement pipeline using the new prompt format.

Identical stage ordering to pipeline.py:
  1. Regex safety gate      (SafetyLayer)
  2. LLM semantic scoring   (LLMRiskScorer, optional)
  3. Emotion detection      (EmotionDetector)
  4. RAG retrieval          (RAGRetriever)
  5. Prompt assembly        (sadev_prompt_builder.build_sadev_prompt)
  6. LLM generation         (OllamaClient)
  7. Response parsing       (SadevFormatter)

Key differences from pipeline.py:
  - Prompt uses ---RESPONSE--- / ---METADATA--- contract (no rigid [VALIDATION] scaffold)
  - SadevFormatter parses the new format with graceful degradation
  - long_term_memory passed through (loaded externally — this class just accepts it)
  - PipelineResult reused unchanged for chat.py compatibility
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from safety.safety_layer import SafetyLayer, SafetyResult
from safety.llm_risk_scorer import LLMRiskScorer, RiskScore
from emotion.emotion_detector import EmotionDetector, EmotionResult
from rag.retriever import RAGRetriever
from local_pipeline.sadev_prompt_builder import build_sadev_prompt
from local_pipeline.ollama_client import OllamaClient
from local_pipeline.sadev_formatter import SadevFormatter, SadevResponse


# ── Session ───────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list[Turn] = field(default_factory=list)
    emotion_arc: list[str] = field(default_factory=list)
    last_stressor: str = "none"
    turn_count: int = 0


# ── PipelineResult (kept identical to pipeline.py for chat.py compat) ─────

@dataclass
class PipelineResult:
    session_id: str
    response_text: str
    follow_up: str
    validation_part: str
    reflection_part: str
    question_part: str
    emotion_detected: str
    intent_detected: str
    stressor_detected: str
    urgency_level: int
    risk_level: int
    risk_action: str
    helpline_name: Optional[str] = None
    helpline_number: Optional[str] = None
    rag_chunks_used: int = 0
    model_used: str = "gemma2"
    crisis_exit: bool = False
    risk_source: str = "none"
    safeguarding: bool = False
    strategy_used: list[str] = field(default_factory=list)
    parse_strategy: str = "exact"


_HELPLINES = {
    "tier_5":       {"name": "Vandrevala Foundation", "number": "1860-2662-345", "hours": "24/7"},
    "tier_4":       {"name": "iCall",                 "number": "9152987821",    "hours": "Mon–Sat 8am–10pm"},
    "tier_3":       {"name": "iCall",                 "number": "9152987821",    "hours": "Mon–Sat 8am–10pm"},
    "tier_2_minor": {"name": "CHILDLINE",             "number": "1098",          "hours": "24/7"},
    "tier_2_adult": {"name": "iCall",                 "number": "9152987821",    "hours": "Mon–Sat 8am–10pm"},
}


def _requires_hard_exit(risk_level: int) -> bool:
    return risk_level in (2, 4, 5)


# ── Pipeline ──────────────────────────────────────────────────────────────

class SadevPipeline:

    def __init__(
        self,
        ollama_model: str = "gemma2",
        ollama_url: str = "http://localhost:11434",
        index_path: str = "data/vector_index",
        kb_path: str = "data/knowledge_base_raw.json",
        top_k: int = 3,
        llm_scoring: bool = True,
        long_term_memory: dict | None = None,
    ):
        self.safety      = SafetyLayer()
        self.emotion     = EmotionDetector()
        self.rag         = RAGRetriever(index_path=index_path, kb_path=kb_path, top_k=top_k)
        self.ollama      = OllamaClient(model=ollama_model, base_url=ollama_url)
        self.formatter   = SadevFormatter()
        self.kb_path     = kb_path
        self._sessions: dict[str, Session] = {}
        self.llm_scoring = llm_scoring
        self.risk_scorer = LLMRiskScorer(self.ollama, kb_path=kb_path) if llm_scoring else None
        # Long-term memory: user profile dict, keyed by session_id (optional)
        # Caller can pass a pre-loaded profile or update per session
        self._long_term_memory: dict[str, dict] = {}
        if long_term_memory:
            self._long_term_memory["default"] = long_term_memory

    # ── Public API ────────────────────────────────────────────────────────

    def chat(
        self,
        raw_text: str,
        session_id: Optional[str] = None,
        long_term_memory: dict | None = None,
    ) -> PipelineResult:
        """
        Process one turn and return a PipelineResult.

        long_term_memory: per-call override (user profile dict).
        If None, uses session-level or pipeline-level profile.
        """
        session = self._get_or_create_session(session_id)
        session.turn_count += 1

        # Resolve long-term memory
        ltm = (
            long_term_memory
            or self._long_term_memory.get(session.session_id)
            or self._long_term_memory.get("default")
        )

        # ── STAGE 1: Regex safety gate ────────────────────────────────────
        regex_result: SafetyResult = self.safety.check(raw_text)
        risk_level_so_far = regex_result.risk_level
        risk_source = "regex" if regex_result.is_crisis else "none"

        if regex_result.is_crisis and _requires_hard_exit(regex_result.risk_level):
            return self._hard_exit(
                raw_text, session,
                regex_result.risk_level,
                regex_result.helpline_name,
                regex_result.helpline_number,
                risk_source="regex",
                safeguarding=regex_result.safeguarding,
            )

        # ── STAGE 2: LLM semantic scoring ─────────────────────────────────
        llm_risk_level   = 1
        llm_safeguarding = False
        llm_helpline_key = ""

        if self.risk_scorer is not None:
            history_dicts = [
                {"role": t.role, "content": t.content}
                for t in session.history[-6:]
            ]
            llm_score: RiskScore = self.risk_scorer.score(
                current_text=raw_text,
                history=history_dicts,
                session_turn_count=session.turn_count,
            )
            llm_risk_level   = llm_score.risk_level
            llm_safeguarding = llm_score.safeguarding
            llm_helpline_key = llm_score.helpline_key

            if _requires_hard_exit(llm_score.risk_level):
                h_name, h_number = self._resolve_helpline(
                    llm_score.helpline_key, llm_score.safeguarding, raw_text
                )
                return self._hard_exit(
                    raw_text, session,
                    llm_score.risk_level, h_name, h_number,
                    risk_source="llm",
                    safeguarding=llm_score.safeguarding,
                )

        effective_risk = max(risk_level_so_far, llm_risk_level)
        tier3_active   = effective_risk == 3

        # ── STAGE 3: Emotion detection ────────────────────────────────────
        emotion_result: EmotionResult = self.emotion.detect(raw_text)
        session.emotion_arc.append(emotion_result.primary_emotion)
        session.last_stressor = emotion_result.stressor_type

        urgency = emotion_result.urgency_level
        cbt     = emotion_result.cbt_technique
        if tier3_active:
            urgency = 3

        # ── STAGE 4: RAG retrieval ────────────────────────────────────────
        rag_query = (
            f"{emotion_result.stressor_type} "
            f"{emotion_result.primary_emotion} "
            f"{raw_text[:200]}"
        )
        rag_chunks = self.rag.retrieve(
            query=rag_query,
            stressor_filter=emotion_result.stressor_type,
        )

        # ── STAGE 5: Prompt assembly ──────────────────────────────────────
        history_dicts = [
            {"role": t.role, "content": t.content}
            for t in session.history[-6:]
        ]
        # Resolve helpline for tier-3
        helpline_note = ""
        h_name_t3, h_num_t3 = "", ""
        if tier3_active:
            _h = _HELPLINES["tier_3"]
            h_name_t3  = _h["name"]
            h_num_t3   = _h["number"]
            helpline_note = f"{h_name_t3} ({h_num_t3})"

        # Resolve helpline for crisis (risk >= 4) — pipeline already hard-exits above,
        # but pass through for safety in case scoring changes between stages
        h_name_crisis, h_num_crisis = "", ""
        if effective_risk >= 4:
            h_name_crisis, h_num_crisis = self._resolve_helpline(
                llm_helpline_key, llm_safeguarding, raw_text
            )

        system_prompt = build_sadev_prompt(
            user_input=raw_text,
            short_term_memory=history_dicts,
            long_term_memory=ltm,
            rag_context=rag_chunks,
            emotion=emotion_result.primary_emotion,
            intent=emotion_result.intent,
            stressor=emotion_result.stressor_type,
            urgency=urgency,
            risk=effective_risk,
            lang_code=emotion_result.lang_code,
            language=getattr(emotion_result, "language_label", ""),
            idioms_detected=emotion_result.idioms_detected,
            stigma_flag=emotion_result.stigma_flag,
            emotion_arc=session.emotion_arc,
            cbt_technique=cbt,
            helpline_note=helpline_note,
            helpline_name=h_name_crisis,
            helpline_number=h_num_crisis,
            kb_path=self.kb_path,
        )

        # ── STAGE 6: Generation ───────────────────────────────────────────
        messages = history_dicts + [{"role": "user", "content": raw_text}]
        raw_llm_output = self.ollama.generate(
            system_prompt=system_prompt,
            messages=messages,
        )

        # ── STAGE 7: Parse with new formatter ────────────────────────────
        fallback_signals = {
            "emotion":  emotion_result.primary_emotion,
            "intent":   emotion_result.intent,
            "stressor": emotion_result.stressor_type,
            "urgency":  urgency,
            "risk":     effective_risk,
        }
        parsed: SadevResponse = self.formatter.parse(raw_llm_output, fallback_signals)

        session.history.append(Turn("user", raw_text))
        session.history.append(Turn("assistant", parsed.response))

        return PipelineResult(
            session_id=session.session_id,
            response_text=parsed.response,
            follow_up="",                  # no longer a separate field — it's woven in
            validation_part=parsed.response,
            reflection_part="",
            question_part="",
            emotion_detected=parsed.emotion,
            intent_detected=parsed.intent,
            stressor_detected=parsed.stressor,
            urgency_level=parsed.urgency,
            risk_level=parsed.risk,
            risk_action=(
                "crisis_calm_conversation" if tier3_active else
                "gentle_checkin"           if urgency == 2 else
                "none"
            ),
            rag_chunks_used=len(rag_chunks),
            model_used=self.ollama.model,
            crisis_exit=False,
            risk_source=(
                "llm"   if llm_risk_level >= 3 else
                "regex" if regex_result.is_crisis else
                "none"
            ),
            safeguarding=False,
            strategy_used=parsed.strategy,
            parse_strategy=parsed.parse_strategy,
        )

    # ── Long-term memory management ───────────────────────────────────────

    def set_user_profile(self, session_id: str, profile: dict) -> None:
        """Store a user profile for this session. Call before chat()."""
        self._long_term_memory[session_id] = profile

    def update_profile_from_session(self, session_id: str) -> None:
        """
        Derive and update user profile from session history.
        Merges new stressors into recurring_stressors list.
        """
        session = self._sessions.get(session_id)
        if not session:
            return
        profile = self._long_term_memory.get(session_id, {})

        # Track recurring stressors
        stressors = list(set(
            profile.get("recurring_stressors", []) +
            [t for t in session.emotion_arc if t != "neutral"]
        ))
        profile["recurring_stressors"] = stressors[-10:]   # keep last 10
        self._long_term_memory[session_id] = profile

    # ── Hard exit (unchanged from pipeline.py) ────────────────────────────

    def _hard_exit(
        self,
        raw_text: str,
        session: Session,
        risk_level: int,
        helpline_name: str,
        helpline_number: str,
        risk_source: str,
        safeguarding: bool = False,
    ) -> PipelineResult:
        response = self._hard_exit_response(
            risk_level, helpline_name, helpline_number, safeguarding
        )
        session.history.append(Turn("user", raw_text))
        session.history.append(Turn("assistant", response))
        return PipelineResult(
            session_id=session.session_id,
            response_text=response,
            follow_up="",
            validation_part=response,
            reflection_part="",
            question_part="",
            emotion_detected="grief" if not safeguarding else "fear",
            intent_detected="crisis_signal",
            stressor_detected="none",
            urgency_level=risk_level,
            risk_level=risk_level,
            risk_action=(
                "crisis_overlay" if risk_level == 5 else
                "urgent_prompt"  if risk_level == 4 else
                "safeguarding"
            ),
            helpline_name=helpline_name,
            helpline_number=helpline_number,
            crisis_exit=True,
            risk_source=risk_source,
            safeguarding=safeguarding,
        )

    def _hard_exit_response(
        self,
        risk_level: int,
        helpline_name: str,
        helpline_number: str,
        safeguarding: bool,
    ) -> str:
        name   = helpline_name   or "iCall"
        number = helpline_number or "9152987821"

        if safeguarding and helpline_name == "CHILDLINE":
            return (
                f"Aap jo share kar rahe hain woh bahut serious hai — "
                f"aur aapko help lene ka poora haq hai.\n\n"
                f"Please abhi CHILDLINE ko call karo: {number}. "
                f"Yeh 24/7 available hai, free hai, aur minors ke liye specifically hai."
            )
        if safeguarding:
            return (
                f"Jo situation aap describe kar rahe hain woh unsafe hai "
                f"aur aapko support chahiye.\n\n"
                f"Please {name} se baat karo: {number}."
            )
        if risk_level == 5:
            return (
                f"Aap akele nahi hain. Main yahan hun.\n\n"
                f"Abhi please {name} ko call karo: {number}. Woh 24/7 available hain."
            )
        return (
            f"Jo aap feel kar rahe hain woh bahut heavy hai. "
            f"Aapko ye akele carry nahi karna hai.\n\n"
            f"Please {name} se baat karo: {number}."
        )

    def _resolve_helpline(
        self,
        helpline_key: str,
        safeguarding: bool,
        text: str,
    ) -> tuple[str, str]:
        import re
        if safeguarding:
            is_minor = bool(re.search(
                r"\b(1[0-7]|fifteen|sixteen|seventeen|fourteen|thirteen|"
                r"twelve|eleven|ten)\s+years?\s+old\b",
                text, re.IGNORECASE,
            ))
            key = "tier_2_minor" if is_minor else "tier_2_adult"
        else:
            key = helpline_key or "tier_3"
        h = _HELPLINES.get(key, _HELPLINES["tier_3"])
        return h["name"], h["number"]

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def _get_or_create_session(self, session_id: Optional[str]) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        s = Session(session_id=session_id or str(uuid.uuid4()))
        self._sessions[s.session_id] = s
        return s