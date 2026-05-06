"""
emotion/emotion_detector.py
Heuristic emotion + intent detector optimised for South Asian languages.

Designed to be replaced by MuRIL when training is complete (same interface).
Extends the patterns from the SADEV NLU stub with:
  - Greeting / check-in detection (prevents false vent_frustration on "hi")
  - Casual-bhai guard ("aur bhai kya haal" is NOT family pressure)
  - Guilt + love co-occurrence detection (key South Asian emotional pattern)
  - Stigma-language detection
  - Richer Hinglish idiom detection
  - Lang code inference

Returns EmotionResult which maps directly to the fields expected by
prompt_builder.build_system_prompt().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class EmotionResult:
    intent: str
    primary_emotion: str
    secondary_emotion: str | None
    emotions_all: list[str]
    urgency_level: int
    stressor_type: str
    tone_mode: str
    lang_code: str
    sentiment_score: float
    idioms_detected: list[str]
    stigma_flag: bool
    cbt_technique: str
    entities_persons: list[str] = field(default_factory=list)
    entities_roles: list[str] = field(default_factory=list)


class EmotionDetector:
    """
    Keyword-map based emotion + intent detector.
    Zero model weights. Zero latency. ~75-80% accuracy for common cases.
    """

    # ── Script / language signals ──────────────────────────────────────────
    _DEVANAGARI = re.compile(r"[\u0900-\u097F]")
    _BENGALI_SCRIPT = re.compile(r"[\u0980-\u09FF]")
    _HINGLISH_MARKERS = {
        "hun", "hai", "hain", "tha", "thi", "the", "meri", "mera", "mujhe",
        "main", "aur", "par", "lekin", "kyun", "kya", "nahi", "bhi", "toh",
        "abhi", "bahut", "ghar", "mummy", "papa", "chahta", "chahti", "karna",
        "karun", "izzat", "shaadi", "rishta", "yaar", "bhai", "didi", "samajh",
        "lagta", "sunke", "kitna", "woh", "agar", "sirf", "unhe", "sabse",
        "mast", "theek", "badhiya", "mazey", "chill",
    }

    # ── Greeting / check-in detection (FIRST priority) ────────────────────
    # Short inputs that are social openers, not distress signals
    # Two-pass greeting detection:
    # 1. _GREETING_OPENER fires when text STARTS with a greeting word
    # 2. _GREETING_EXACT fires on Hinglish check-ins and standalone short phrases
    # Combined in detect() with a word-count guard (<=6 words = greeting, not venting)
    _GREETING_OPENER = re.compile(
        r"^(hi+|hello+|hey+|heyy+|hii+|namaste|namaskar|howdy|sup|"
        r"good\s+(morning|afternoon|evening|night)|"
        r"wassup|yo\s|greetings)\b",
        re.IGNORECASE,
    )
    _GREETING_EXACT = re.compile(
        r"^("
        r"aur\s+(bhai|yaar|dost|ji)?\s*[!?.]?|"
        r"kya\s+haal(\s+hai)?|kaise\s+ho|kaise\s+hain|"
        r"what'?s?\s+up|"
        r"mast\s+hun[!.]*|theek\s+hun[!.]*|sab\s+badhiya[!.]*|all\s+good[!.]*|"
        r"bas\s+aisa\s+hi[!.]*|chill\s+hun[!.]*|mazey\s+mein[!.]*|"
        r"mai\s+to\s+mast\s+hoon[!.]*|main\s+mast\s+hun[!.]*|"
        r"mast\s+hoon[!.]*|badhiya\s+hoon[!.]*|ekdum\s+mast[!.]*|"
        r"kuch\s+nahi\s+bas[!.]*|bas\s+yun\s+hi[!.]*)[\s!?.]*$",
        re.IGNORECASE,
    )
    # Positive/neutral short check-in (user is doing fine)
    _POSITIVE_CHECKIN = re.compile(
        r"^(i'?m?\s+(fine|good|great|okay|ok|alright|doing\s+well|doing\s+fine)|"
        r"i\s+am\s+(fine|good|okay)|"
        r"mai\s+to\s+mast|main\s+theek|sab\s+theek|"
        r"mast\s+hoon|badhiya\s+hoon|ekdum\s+badhiya|"
        r"khush\s+hun|bahut\s+achha|life\s+is\s+good)[\s!?.]*$",
        re.IGNORECASE,
    )

    # ── Intent signals ─────────────────────────────────────────────────────
    # NOTE: bare "bhai" is NOT in _FAMILY_SIG — it fires too often as address word
    _FAMILY_SIG = re.compile(
        r"\b(maa|papa|parents|family|ghar|mummy|didi|doctor\s+banu|engineer|"
        r"ghar\s+wale|gharwale|beta|beti|expectations|career\s+pressure|"
        r"ghar\s+waalon|gharwaalon|bacche|baap|maa\s+baap|ma\s+baap|"
        r"mere\s+bhai|chota\s+bhai|bade\s+bhai|bhai\s+ne|bhai\s+ko)\b",
        re.IGNORECASE,
    )
    _ACADEMIC_SIG = re.compile(
        r"\b(JEE|NEET|UPSC|exam|marks|result|college|rank|percentile|boards|"
        r"fail|grade|score|studies|padhai|coaching|topper|AIIMS|IIT|12th|10th|"
        r"fees|admission|entrance|competitive)\b",
        re.IGNORECASE,
    )
    _MARRIAGE_SIG = re.compile(
        r"\b(shaadi|marriage|rishta|arranged|inter.?caste|dowry|divorce|"
        r"love\s+marriage|wedding|byah|vivah|shadi)\b",
        re.IGNORECASE,
    )
    _IDENTITY_SIG = re.compile(
        r"\b(gay|queer|transgender|non.?binary|LGBTQ|atheist|religion|faith|"
        r"sexuality|identity|coming\s+out|double\s+life|bisexual|lesbian|"
        r"dharm|aastha|nafs)\b",
        re.IGNORECASE,
    )
    _FINANCIAL_SIG = re.compile(
        r"\b(loan|EMI|scholarship|job|income|money|broke|debt|afford|"
        r"remittance|breadwinner|support\s+family|paisa|paise|gareeb|"
        r"karz|rozgaar|naukri|business)\b",
        re.IGNORECASE,
    )
    _SOCIAL_SIG = re.compile(
        r"\b(log\s+kya\s+kahenge|izzat|reputation|society|relatives|"
        r"neighbours|community|honour|ashamed|judge|gossip|log\s+sochenge|"
        r"padosi|rishtedaar|samaj|log\s+dekhenge)\b",
        re.IGNORECASE,
    )
    _POSITIVE_SIG = re.compile(
        r"\b(got\s+the\s+job|cleared|passed|scholarship|promotion|happy|excited|"
        r"finally|achievement|congrats|success|made\s+it|khushi|khush|"
        r"mila|mil\s+gaya|ho\s+gaya|qualify)\b",
        re.IGNORECASE,
    )
    _ADVICE_SIG = re.compile(
        r"\b(what\s+should\s+I\s+do|kya\s+karun|how\s+do\s+I|please\s+help|"
        r"suggest|advice|tell\s+me|guide\s+me|koi\s+solution|batao|"
        r"kya\s+karna\s+chahiye|samjhao)\b",
        re.IGNORECASE,
    )
    _VALIDATION_SIG = re.compile(
        r"\b(am\s+I\s+overreacting|is\s+this\s+normal|am\s+I\s+wrong|sahi\s+hun|"
        r"theek\s+hun|kya\s+main|do\s+you\s+think\s+I\s+should|galat\s+hun|"
        r"sahi\s+kar\s+raha|behave\s+kar\s+raha)\b",
        re.IGNORECASE,
    )

    # ── Emotion signals ────────────────────────────────────────────────────
    _ANXIETY_SIG = re.compile(
        r"\b(worried|anxious|nervous|scared|fear|darr|tension|pressure|stress|"
        r"panic|overthinking|ghabrahat|dara|dar\s+lag|chinta)\b",
        re.IGNORECASE,
    )
    _GUILT_SIG = re.compile(
        r"\b(guilt|guilty|mere\s+wajah\s+se|meri\s+galti|I\s+feel\s+bad|"
        r"disappointing|let\s+them\s+down|sharm|paap|kasoor\s+mera)\b",
        re.IGNORECASE,
    )
    _LOVE_SIG = re.compile(
        r"\b(love\s+them|pyaar\s+karta|care\s+karta|chahta\s+hun\s+unhe|"
        r"unse\s+pyaar|unka\s+bura\s+nahi\s+chahta|maa\s+se\s+pyaar|"
        r"family\s+ko\s+love|I\s+love\s+my|apno\s+se\s+pyaar)\b",
        re.IGNORECASE,
    )
    _ANGER_SIG = re.compile(
        r"\b(angry|frustrated|furious|gusse|hate|irritated|fed\s+up|enough|"
        r"bhot\s+gussa|bahut\s+gussa|naraaz|krodh|nafrat)\b",
        re.IGNORECASE,
    )
    _HOPELESS_SIG = re.compile(
        r"\b(hopeless|pointless|no\s+point|give\s+up|can't\s+anymore|exhausted|"
        r"thak\s+gaya|thak\s+gayi|haarna|haar\s+gaya|haar\s+jaana|haarna\s+chahta|"
        r"haarna\s+chahti|nirasha|haar\s+maan|umeed\s+nahi|koi\s+umeed|"
        r"sab\s+khatam|khatam\s+ho\s+gaya|bas\s+ho\s+gaya)\b",
        re.IGNORECASE,
    )
    _SHAME_SIG = re.compile(
        r"\b(ashamed|sharam|embarrassed|humiliated|sharminda|sar\s+nahi\s+utha|"
        r"muh\s+nahi\s+dikha)\b",
        re.IGNORECASE,
    )
    _LONELINESS_SIG = re.compile(
        r"\b(lonely|alone|akela|akeli|no\s+one|koi\s+nahi|akelapan|"
        r"isolated|no\s+friends|dost\s+nahi)\b",
        re.IGNORECASE,
    )
    _OVERWHELM_SIG = re.compile(
        r"\b(overwhelmed|too\s+much|bahut\s+zyada|sambhal\s+nahi|"
        r"handle\s+nahi|cope\s+nahi|sab\s+kuch\s+ek\s+saath|"
        r"itna\s+sab|bore\s+ho\s+gayi)\b",
        re.IGNORECASE,
    )

    # ── Stigma signals ─────────────────────────────────────────────────────
    _STIGMA_SIG = re.compile(
        r"\b(not\s+crazy|not\s+mental|not\s+sick|just\s+stressed|"
        r"don't\s+need\s+therapy|not\s+a\s+therapy|not\s+a\s+mental|"
        r"paagal\s+nahi\s+hun|pagal\s+nahi|mental\s+nahi|"
        r"main\s+paagal\s+thodi\s+hun|mujhe\s+psychiatrist)\b",
        re.IGNORECASE,
    )

    # ── Cultural idioms ────────────────────────────────────────────────────
    _IDIOMS: dict[str, re.Pattern] = {
        "log_kya_kahenge": re.compile(r"log\s+kya\s+kahenge|log\s+sochenge|log\s+dekhenge", re.IGNORECASE),
        "izzat":           re.compile(r"\bizzat\b", re.IGNORECASE),
        "ghar_ki_izzat":   re.compile(r"ghar\s+ki\s+izzat|ghar\s+ka\s+naam", re.IGNORECASE),
        "beta_padh_le":    re.compile(r"beta\s+padh\s+le|beta\s+pehle\s+padh|padhai\s+pe\s+dhyan", re.IGNORECASE),
        "sab_theek_ho_jayega": re.compile(r"sab\s+theek\s+ho\s+jayega|sab\s+thik\s+ho\s+jayega", re.IGNORECASE),
        "hamare_zamane_mein": re.compile(r"hamare\s+zamane\s+mein|hamare\s+time\s+mein", re.IGNORECASE),
        "JEE":             re.compile(r"\bJEE\b"),
        "NEET":            re.compile(r"\bNEET\b"),
        "arranged_marriage": re.compile(r"arranged\s+marriage|arranged\s+shaadi|rishta\s+dhoondhna", re.IGNORECASE),
    }

    # ── Sentiment word counts ──────────────────────────────────────────────
    _NEG_WORDS = re.compile(
        r"\b(stress|pressure|worried|anxious|sad|fail|bad|hard|difficult|"
        r"cant|can't|nahi|hopeless|terrible|awful|worst|dard|takleef|"
        r"mushkil|pareshaan|bura|zyada|problem|tense|dar)\b",
        re.IGNORECASE,
    )
    _POS_WORDS = re.compile(
        r"\b(happy|good|great|excited|passed|cleared|success|wonderful|"
        r"achha|theek|sahi|love|khushi|mila|appreciate|grateful|"
        r"positive|better|improve|achhe|mast|badhiya|mazey)\b",
        re.IGNORECASE,
    )

    # ── Role entities ──────────────────────────────────────────────────────
    _ROLES = ["doctor", "engineer", "lawyer", "IAS", "IPS", "teacher", "artist",
              "MBA", "CA", "professor", "government", "civil servant"]
    _PERSONS = ["maa", "papa", "mummy", "dad", "didi", "nani",
                "dadi", "dada", "nana", "chacha", "chachi", "mausi", "mama"]

    # ── CBT technique selection ────────────────────────────────────────────
    _CBT_MAP: dict[tuple[str, str], str] = {
        ("express_family_pressure",  "conflict"):      "reflective_questioning",
        ("express_family_pressure",  "guilt"):         "self_compassion",
        ("express_family_pressure",  "anxiety"):       "cognitive_reframe",
        ("express_academic_pressure","anxiety"):       "decatastrophising",
        ("express_academic_pressure","shame"):         "self_compassion",
        ("express_academic_pressure","inadequacy"):    "values_clarification",
        ("express_social_judgment",  "shame"):         "cognitive_reframe",
        ("express_marriage_pressure","conflict"):      "reflective_questioning",
        ("express_identity_conflict","fear"):          "emotional_validation",
        ("express_identity_conflict","shame"):         "self_compassion",
        ("express_identity_conflict","loneliness"):    "emotional_validation",
        ("express_financial_stress", "anxiety"):       "problem_solving_orientation",
        ("express_financial_stress", "overwhelm"):     "emotional_validation",
        ("seek_validation",          "confusion"):     "reflective_questioning",
        ("vent_frustration",         "frustration"):   "emotional_validation",
        ("vent_frustration",         "anger"):         "emotional_validation",
        ("ask_for_advice",           "confusion"):     "pros_cons_exploration",
        ("crisis_signal",            "grief"):         "grounding",
        ("crisis_signal",            "helplessness"):  "grounding",
        ("share_positive_news",      "relief"):        "reflective_questioning",
        ("greeting",                 "neutral"):       "reflective_questioning",
    }

    def detect(self, text: str) -> EmotionResult:
        """Run all heuristics and return an EmotionResult."""
        t = text.lower().strip()

        lang_code   = self._detect_lang(text, t)
        idioms      = [k for k, p in self._IDIOMS.items() if p.search(text)]
        stigma_flag = bool(self._STIGMA_SIG.search(text))

        # Sentiment score
        neg = len(self._NEG_WORDS.findall(t))
        pos = len(self._POS_WORDS.findall(t))
        sentiment = max(-1.0, min(1.0, (pos - neg) * 0.2))

        # Entities
        roles   = [r for r in self._ROLES   if r.lower() in t]
        persons = [p for p in self._PERSONS if p in t]

        # ── GREETING / CHECK-IN (highest priority after safety) ──────────
        # Two-pass: opener (hi/hello/hey) with short word count, OR exact Hinglish check-in
        word_count = len(t.split())
        is_greeting = (
            (self._GREETING_OPENER.match(t) and word_count <= 6)
            or self._GREETING_EXACT.match(t)
            or self._POSITIVE_CHECKIN.match(t)
        )
        if is_greeting:
            return EmotionResult(
                intent="greeting",
                primary_emotion="neutral",
                secondary_emotion=None,
                emotions_all=["neutral"],
                urgency_level=1,
                stressor_type="none",
                tone_mode="indirect_validating",
                lang_code=lang_code,
                sentiment_score=round(sentiment, 2),
                idioms_detected=idioms,
                stigma_flag=stigma_flag,
                cbt_technique="reflective_questioning",
                entities_persons=persons,
                entities_roles=roles,
            )

        # ── Standard intent classification ────────────────────────────────
        intent, stressor = self._classify_intent(text, t)

        # Urgency (no crisis here — safety layer handles that)
        urgency = 2 if bool(self._HOPELESS_SIG.search(t)) else 1

        # Emotions
        emotions = self._classify_emotions(intent, text, t, urgency)
        primary   = emotions[0] if emotions else "neutral"
        secondary = emotions[1] if len(emotions) > 1 else None

        # Tone mode
        tone = self._select_tone(intent, primary, urgency)

        # CBT technique
        cbt = self._CBT_MAP.get((intent, primary), "emotional_validation")

        return EmotionResult(
            intent=intent,
            primary_emotion=primary,
            secondary_emotion=secondary,
            emotions_all=emotions,
            urgency_level=urgency,
            stressor_type=stressor,
            tone_mode=tone,
            lang_code=lang_code,
            sentiment_score=round(sentiment, 2),
            idioms_detected=idioms,
            stigma_flag=stigma_flag,
            cbt_technique=cbt,
            entities_persons=persons,
            entities_roles=roles,
        )

    def _detect_lang(self, text: str, t: str) -> str:
        if self._DEVANAGARI.search(text):
            return "hi"
        if self._BENGALI_SCRIPT.search(text):
            return "bn"
        words = set(t.split())
        if len(words & self._HINGLISH_MARKERS) >= 2:
            return "hi-en-mix"
        return "en"

    def _classify_intent(self, text: str, t: str) -> tuple[str, str]:
        if self._POSITIVE_SIG.search(t):
            return "share_positive_news", "none"
        if self._ADVICE_SIG.search(t):
            if self._FAMILY_SIG.search(t):   return "ask_for_advice", "family_career_expectations"
            if self._ACADEMIC_SIG.search(t): return "ask_for_advice", "academic_pressure"
            return "ask_for_advice", "none"
        if self._VALIDATION_SIG.search(t):
            return "seek_validation", "none"
        if self._IDENTITY_SIG.search(t):
            return "express_identity_conflict", "identity_conflict"
        if self._MARRIAGE_SIG.search(t):
            return "express_marriage_pressure", "marriage_pressure"
        if self._FINANCIAL_SIG.search(t):
            return "express_financial_stress", "financial_stress"
        if self._SOCIAL_SIG.search(t):
            return "express_social_judgment", "social_judgment"
        if self._ACADEMIC_SIG.search(t):
            return "express_academic_pressure", "academic_pressure"
        if self._FAMILY_SIG.search(t):
            return "express_family_pressure", "family_career_expectations"
        if self._ANGER_SIG.search(t):
            return "vent_frustration", "none"
        return "vent_frustration", "none"

    def _classify_emotions(
        self, intent: str, text: str, t: str, urgency: int
    ) -> list[str]:
        if intent == "share_positive_news":
            return ["relief", "hope"]

        base = {
            "express_family_pressure":   ["conflict", "guilt"],
            "express_academic_pressure": ["anxiety", "fear"],
            "express_identity_conflict": ["fear", "loneliness", "conflict"],
            "express_marriage_pressure": ["conflict", "anxiety"],
            "express_social_judgment":   ["shame", "anxiety"],
            "express_financial_stress":  ["anxiety", "overwhelm"],
            "vent_frustration":          ["frustration"],
            "seek_validation":           ["confusion", "anxiety"],
            "ask_for_advice":            ["confusion", "anxiety"],
        }.get(intent, ["neutral"])
        emotions = list(base)

        if self._GUILT_SIG.search(t) and "guilt" not in emotions:
            emotions.append("guilt")
        if self._LOVE_SIG.search(t) and "love" not in emotions:
            emotions.append("love")
        if self._ANXIETY_SIG.search(t) and "anxiety" not in emotions:
            emotions.append("anxiety")
        if self._SHAME_SIG.search(t) and "shame" not in emotions:
            emotions.append("shame")
        if self._LONELINESS_SIG.search(t) and "loneliness" not in emotions:
            emotions.append("loneliness")
        if self._OVERWHELM_SIG.search(t) and "overwhelm" not in emotions:
            emotions.append("overwhelm")
        if self._ANGER_SIG.search(t) and "anger" not in emotions:
            emotions.append("anger")
        if urgency >= 2 and "inadequacy" not in emotions and intent == "express_academic_pressure":
            emotions.append("inadequacy")

        return emotions[:4]

    def _select_tone(self, intent: str, primary_emotion: str, urgency: int) -> str:
        if urgency >= 3:
            return "crisis_calm"
        if intent == "share_positive_news":
            return "celebratory"
        if intent == "ask_for_advice":
            return "direct_supportive"
        tone_map = {
            "express_identity_conflict":  "gentle_reframe",
            "express_marriage_pressure":  "gentle_reframe",
            "express_financial_stress":   "direct_supportive",
        }
        return tone_map.get(intent, "indirect_validating")
