"""
safety/safety_layer.py
Rule-based safety layer. Runs on raw text before ANY other stage.

Tier model:
  5 — immediate, active crisis        → helpline + crisis overlay, LLM never called
  4 — direct self-harm / suicidal     → helpline + urgent prompt, LLM never called
  3 — hopelessness / passive ideation → helpline surface, LLM never called
  2 — safeguarding (minor at risk,    → safeguarding response, LLM never called
       violence ideation, coercive
       control, child marriage)
  1 — safe                            → pipeline continues normally

Metaphor guard always runs FIRST — returns risk=1 immediately on match.

Tier 2 — SAFEGUARDING — is a new tier added after a real conversation exposed:
  - "shall i kill them instead?" → was passing as risk=1
  - "shall i try for a homicide?" → was passing as risk=1
  - "shall i try for a suicide?" → was passing as risk=1 (phrasing gap)
  - child marriage disclosure → was passing as risk=1
  - "my father will kill me" (threat of violence toward user) → was risk=1

Safeguarding (tier 2) does NOT assume the person wants to die.
It assumes they are in danger or describing danger to others.
Response: calm, safety-focused, CHILDLINE / NCPCR for minors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SafetyResult:
    is_crisis: bool
    risk_level: int           # 1–5
    tier: str                 # "safe"|"tier_2"|"tier_3"|"tier_4"|"tier_5"
    matched_pattern: str = ""
    helpline_name: str = ""
    helpline_number: str = ""
    helpline_hours: str = ""
    safeguarding: bool = False   # True when minor-at-risk or violence context


class SafetyLayer:
    """
    Single-call safety gate. Returns immediately on first match.
    Never raises — degrades to safe on unexpected error.

    Check order:
      1. Metaphor guard  (exits safe)
      2. Tier 5          (immediate active crisis)
      3. Tier 4          (direct self-harm / suicidal ideation)
      4. Tier 3          (hopelessness / passive ideation)
      5. Tier 2          (safeguarding: violence ideation, minor at risk,
                          coercive control, child marriage, threats to user)
    """

    # ── Metaphor guard ─────────────────────────────────────────────────────
    _METAPHORS = [re.compile(p, re.IGNORECASE) for p in [
        r"kill(ing)?\s+(it|this|that|the|my|these|those)\s+"
        r"(exam|test|presentation|interview|quiz|viva|class|project|assignment|game|match)",
        r"dying\s+of\s+(laughter|boredom|excitement|hunger|thirst|cuteness|embarrassment)",
        r"dead\s+(tired|serious|cert|honest|broke)",
        r"murder(ing)?\s+(it|this|the|that)",
        r"i('m| am)\s+(literally\s+|figuratively\s+)?dying\s+(lol|lmao|haha|😂|🤣)",
        r"killing\s+me\s+(softly|gently)?$",
        r"i\s+(could|would)\s+(literally\s+|figuratively\s+)?kill\s+for",
        r"suicide\s+(squad|mission|door|note\s+joke|meme)",
        r"want\s+to\s+(shoot|kill)\s+(myself\s+)?(for|because\s+of)?\s+(fun|laughter|embarrassment)",
        r"killing\s+it\s+at",
    ]]

    # ── Tier 5: Immediate active crisis ────────────────────────────────────
    _TIER_5 = [re.compile(p, re.IGNORECASE) for p in [
        r"\balready\s+took\s+(pills|tablets|medicines|drugs)\b",
        r"\balready\s+hurt\s+(myself|yourself)\b",
        r"\bi('m| am)\s+(currently\s+|right\s+now\s+|already\s+)?"
        r"(bleeding|cutting\s+myself|hurting\s+myself)\b",
        r"\bdoing\s+it\s+tonight\b",
        r"\bi\s+have\s+a\s+(plan|method)\b",
        r"\bgoli\s+le\s+lunga\b",
        r"\babhi\s+kuch\s+kar\s+lunga\b",
        r"\bbleeding\s*(now|right\s+now|already)?\b",
        r"\btook\s+(the\s+)?pills\b",
        r"\bpehle\s+se\s+kar\s+liya\b",
    ]]

    # ── Tier 4: Direct self-harm / suicidal ideation ────────────────────────
    _TIER_4 = [re.compile(p, re.IGNORECASE) for p in [
        r"\bi\s+want\s+to\s+(hurt|harm|kill|end)\s+myself\b",
        r"\bi('m| am)\s+going\s+to\s+(hurt|harm|kill|end)\s+myself\b",
        r"\bwant\s+to\s+die\b",
        r"\bkill\s+myself\b",
        r"\bend\s+my\s+life\b",
        r"\bsuicidal\b",
        r"\bself[.\s-]?harm(ing)?\b",
        r"\bcutting\s+(myself|my\s+(arms?|wrists?|legs?|skin))\b",
        # Hinglish / Hindi
        r"\bmar\s+jan(a|e)\s+chah(ta|ti)\s+(hun|hai|hoon)\b",
        r"\bmar\s+jaana\s+chahti\s+hun\b",
        r"\bkhud\s+ko\s+hurt\s+(karna|kar)\s+(chahta|chahti)\b",
        r"\bnahi\s+rahunga\b",
        r"\bzindagi\s+khatam\s+(kar\s+dunga|karna\s+chahta)\b",
        r"\bjaan\s+de\s+dunga\b",
        r"\bjaan\s+dena\s+chahta\b",
        r"\bkhud\s+ko\s+khatam\s+karna\b",
        r"\bmujhe\s+marr?\s+jana\s+chahi?ye\b",
        r"\bmai(n)?\s+marr?\s+jaana\s+chahta\b",
        # Phrasing variants — "try for a suicide", "attempt suicide"
        r"\b(try|attempt|thinking\s+of|consider)\s+(a\s+)?suicide\b",
        r"\bsuicide\s+(attempt|try|kar\s+leta|kar\s+leti)\b",
        r"\bkhatam\s+kar\s+leta\s+hun\b",
        r"\bkhatam\s+kar\s+leti\s+hun\b",
        # "shall i try for a suicide" — modal question form
        r"\bshall\s+i\s+(try|attempt|do)\s+(a\s+)?suicide\b",
        r"\bshould\s+i\s+(try|attempt|do)\s+(a\s+)?suicide\b",
        r"\bcan\s+i\s+(try|attempt|do)\s+(a\s+)?suicide\b",
    ]]

    # ── Tier 3: Hopelessness / passive ideation ─────────────────────────────
    _TIER_3 = [re.compile(p, re.IGNORECASE) for p in [
        r"\bdon'?t\s+want\s+to\s+(live|be\s+here|exist|go\s+on)\b",
        r"\bno\s+(point|reason|purpose)\s+(in\s+|to\s+)?(living|going\s+on|trying|anything)\b",
        r"\bcan'?t\s+go\s+on\b",
        r"\bwant\s+it\s+all\s+to\s+stop\b",
        r"\bwish\s+(i\s+were|i\s+was|i'?d)\s+(dead|never\s+born|not\s+here|gone)\b",
        r"\bkhatam\s+karna\s+chahta\s+(hun|hun\s+main)\b",
        r"\bnahi\s+rehna\s+chahta\b",
        r"\bjeene\s+ka\s+mann\s+nahi\b",
        r"\bsab\s+khatam\b",
        r"\bthak\s+gaya\s+(hun|hoon)\s+(zindagi\s+se|sab\s+se|sab\s+kuch\s+se)\b",
        r"\bfinished\s+with\s+everything\b",
        r"\bnothing\s+left\s+(to\s+live\s+for|for\s+me)\b",
        r"\beverything\s+feels\s+(hopeless|pointless|meaningless)\b",
        r"\bdon'?t\s+see\s+the\s+point\s+anymore\b",
        r"\bjust\s+want\s+to\s+disappear\b",
        r"\bwish\s+i\s+could\s+disappear\b",
        r"\bmarna\s+chahta\s+hun\b",
        r"\bjina\s+nahi\s+chahta\b",
        r"\bfeel\s+(so\s+)?(hopeless|worthless|pointless)\b",
        r"\blife\s+(is\s+)?(not\s+worth|meaningless|pointless)\b",
    ]]

    # ── Tier 2: Safeguarding ────────────────────────────────────────────────
    # Covers scenarios where the person is in danger but NOT expressing
    # self-harm ideation — or is expressing violence-toward-others ideation.
    # These need a different response: safety-focused, not crisis-grief-focused.
    # Includes: violence ideation toward others, threats received, child marriage,
    # forced marriage, coercive control, minor disclosing abuse.
    _TIER_2 = [re.compile(p, re.IGNORECASE) for p in [
        # ── Violence ideation toward others ──────────────────────────────
        r"\b(shall|should|can|could|will|would)\s+i\s+(try|do|commit|attempt)?\s*"
        r"(a\s+)?(homicide|murder|kill\s+(them|him|her|my\s+family|my\s+parents|"
        r"my\s+father|my\s+mother|everyone))\b",
        r"\bkill\s+(them|him|her|my\s+family|my\s+parents|my\s+father|my\s+mother|everyone)\b",
        r"\b(want\s+to|going\s+to|thinking\s+of)\s+(kill|murder|hurt)\s+"
        r"(them|him|her|my\s+(family|parents|father|mother|husband|wife))\b",
        r"\bhomicide\b",
        r"\bmurder\s+(them|him|her|my|someone)\b",
        # ── Threats of lethal violence toward user ────────────────────────
        r"\b(will|would|might|going\s+to)\s+kill\s+me\b",
        r"\b(father|dad|mother|mum|maa|papa|husband|in.?laws?)\s+"
        r"(will|would|might|going\s+to)\s+kill\s+me\b",
        r"\bthey('?ll|\s+will)\s+kill\s+me\b",
        r"\bwoh\s+mujhe\s+maar\s+(denge|dalenge)\b",
        r"\bmujhe\s+maar\s+(denge|dalenge|dera)\b",
        # ── Child / forced marriage ───────────────────────────────────────
        r"\b(getting|being|forced)\s+(married|to\s+marry)\s+(tomorrow|today|next\s+week|soon|against)\b",
        r"\barranged\s+marriage.{0,40}(forced|don'?t\s+want|against\s+my\s+will|not\s+ready)\b",
        r"\bforced\s+(to\s+)?(marry|marriage|shaadi|wedding)\b",
        r"\bdon'?t\s+want\s+to\s+(get\s+)?marry\b",
        r"\b(married|marrying)\s+against\s+my\s+will\b",
        r"\bchild\s+(marriage|bride|groom)\b",
        r"\bmarrying\s+a\s+\d+\s+year.?old\b",
        r"\bmarried\s+to\s+a\s+\d+\s+year.?old\b",
        r"\b(shaadi|byah|vivah).{0,30}(kal|aaj|parso|kal\s+hai|ho\s+rahi\s+hai)\b",
        # ── Coercive control / abuse disclosure ──────────────────────────
        r"\b(physically|sexually)\s+(abused|abusing|hurt)\s+(by|me)\b",
        r"\bhe'?s?\s+(beating|hitting|hurting)\s+me\b",
        r"\b(locked|trapped|stuck)\s+(in|inside|at)\s+(my\s+)?(home|house|room)\b",
        r"\b(can'?t|cannot)\s+(leave|escape|get\s+out)\b.{0,40}\b(home|house|family|them)\b",
        r"\bnot\s+(allowed|permitted)\s+to\s+(go\s+out|leave|study|work)\b",
        r"\bgharon\s+se\s+bahar\s+nahi\s+(nikalne\s+dete|aane\s+dete)\b",
        # ── Minor-specific distress ───────────────────────────────────────
        r"\bi'?m\s+(only\s+)?(1[0-7]|ten|eleven|twelve|thirteen|fourteen|fifteen|"
        r"sixteen|seventeen)\s+years?\s+old.{0,60}(marry|marriage|abuse|hurt|kill|escape|run)\b",
        r"\b(1[0-7])\s+years?\s+old.{0,60}(forced|married|abuse|escape|run\s+away)\b",
    ]]

    # ── Helplines ──────────────────────────────────────────────────────────
    _HELPLINES = {
        "tier_5":       {"name": "Vandrevala Foundation", "number": "1860-2662-345", "hours": "24/7"},
        "tier_4":       {"name": "iCall",                 "number": "9152987821",    "hours": "Mon–Sat 8am–10pm"},
        "tier_3":       {"name": "iCall",                 "number": "9152987821",    "hours": "Mon–Sat 8am–10pm"},
        "tier_2_minor": {"name": "CHILDLINE",             "number": "1098",          "hours": "24/7"},
        "tier_2_adult": {"name": "iCall",                 "number": "9152987821",    "hours": "Mon–Sat 8am–10pm"},
    }

    def check(self, text: str) -> SafetyResult:
        try:
            return self._check_internal(text)
        except Exception:
            return SafetyResult(is_crisis=False, risk_level=1, tier="safe")

    def _check_internal(self, text: str) -> SafetyResult:
        # 1. Metaphor guard — safe exit
        for p in self._METAPHORS:
            if p.search(text):
                return SafetyResult(is_crisis=False, risk_level=1, tier="safe")

        # 2. Tier 5 — immediate
        for p in self._TIER_5:
            if p.search(text):
                h = self._HELPLINES["tier_5"]
                return SafetyResult(
                    is_crisis=True, risk_level=5, tier="tier_5",
                    matched_pattern=p.pattern[:80],
                    helpline_name=h["name"], helpline_number=h["number"],
                    helpline_hours=h["hours"],
                )

        # 3. Tier 4 — direct ideation
        for p in self._TIER_4:
            if p.search(text):
                h = self._HELPLINES["tier_4"]
                return SafetyResult(
                    is_crisis=True, risk_level=4, tier="tier_4",
                    matched_pattern=p.pattern[:80],
                    helpline_name=h["name"], helpline_number=h["number"],
                    helpline_hours=h["hours"],
                )

        # 4. Tier 3 — hopelessness
        for p in self._TIER_3:
            if p.search(text):
                h = self._HELPLINES["tier_3"]
                return SafetyResult(
                    is_crisis=True, risk_level=3, tier="tier_3",
                    matched_pattern=p.pattern[:80],
                    helpline_name=h["name"], helpline_number=h["number"],
                    helpline_hours=h["hours"],
                )

        # 5. Tier 2 — safeguarding
        for p in self._TIER_2:
            if p.search(text):
                # Detect if minor mentioned anywhere in the full text
                is_minor = bool(re.search(
                    r"\b(1[0-7]|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)"
                    r"\s+years?\s+old\b|"
                    r"\bminor\b|"
                    r"\bchild\b",
                    text, re.IGNORECASE
                ))
                h_key = "tier_2_minor" if is_minor else "tier_2_adult"
                h = self._HELPLINES[h_key]
                return SafetyResult(
                    is_crisis=True, risk_level=2, tier="tier_2",
                    matched_pattern=p.pattern[:80],
                    helpline_name=h["name"], helpline_number=h["number"],
                    helpline_hours=h["hours"],
                    safeguarding=True,
                )

        return SafetyResult(is_crisis=False, risk_level=1, tier="safe")
