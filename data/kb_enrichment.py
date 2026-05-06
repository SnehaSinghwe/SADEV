"""
Enriches knowledge_base_raw.json in-place with the 6 missing fields per stressor.
Run once: python -m data.kb_enrichment
Backs up the original first.
"""
import json, shutil, os, datetime

KB_PATH = "data/knowledge_base_raw.json"

ENRICHMENT = {
  "academic_pressure": {
    "cultural_context_notes": [
      "JEE and NEET are not just exams — they represent years of family sacrifice, financial investment, and social identity. Failure carries collective shame, not just personal disappointment.",
      "The coaching industry (Kota, Allen, FIITJEE) creates a parallel pressure ecosystem where children live away from home under extreme conditions from age 14-17.",
      "Rank is public in India — classmates, relatives, and neighbours often know results, making academic outcomes a community event.",
      "A single exam attempt can determine a person's entire career trajectory in the Indian system, creating catastrophising that is not irrational given the actual stakes."
    ],
    "escalation_markers": [
      "mentions of repeat attempts (second drop, third attempt)",
      "financial language alongside failure (fees wasted, coaching money)",
      "family shame language (muh nahi dikhaunga, sar nahi utha sakta)",
      "comparisons to siblings or cousins who succeeded",
      "statements about parents' sacrifices being wasted"
    ],
    "co_emotion_patterns": {
      "anxiety+shame": "Fear of the upcoming result combined with anticipatory shame — validate both separately",
      "guilt+love": "User loves parents and feels guilty for not meeting expectations — do not conflate these",
      "shame+anger": "Anger at the system or parents combined with shame — validate anger before shame"
    },
    "response_templates": {
      "validation_en": "That pressure — knowing how much is riding on this — is real and enormous.",
      "validation_hinglish": "Ye pressure jo aap feel kar rahe hain — itna sab kuch ek exam pe laga hai — woh bilkul real hai.",
      "reflection_en": "In a system where a single number carries so much weight, it can be hard to remember that the number is measuring one very specific thing — not your worth, not your effort, not your future.",
      "reflection_hinglish": "Jab ek number pe itna kuch depend kare, toh apni value alag se dekhna bahut mushkil ho jaata hai.",
      "question_en": "What does success look like to you, separate from the rank?",
      "question_hinglish": "Agar JEE/NEET ka number side mein rakh dein — aap khud ke liye kya banana chahte hain?"
    },
    "what_not_to_say": [
      "It's just an exam",
      "Grades don't define you (before validation)",
      "There are other options (before acknowledging the loss)",
      "You can try again (minimises the current pain)",
      "Your parents just want the best for you (deflects from user's own pain)"
    ],
    "reflection_frames": [
      "The exam system in India carries collective identity, not just individual ambition",
      "Dropping a year is a significant sacrifice that deserves acknowledgment, not just reframing",
      "Shame about results is often internalised community judgment, not just personal disappointment"
    ]
  },
  "family_career_expectations": {
    "cultural_context_notes": [
      "Career choice in South Asian families is often a collective family decision, not an individual one — children are expected to optimise for family social capital.",
      "The STEM/medicine/law hierarchy is deeply entrenched. Arts, humanities, and creative fields carry implicit shame in many families.",
      "Parents who sacrificed for migration or financial stability often project their own thwarted ambitions onto children.",
      "The concept of 'doing well for the family' can make individual ambition feel selfish rather than legitimate."
    ],
    "escalation_markers": [
      "language about hiding interests from parents",
      "secret parallel pursuits (music practice hidden, art portfolio hidden)",
      "ultimatums from parents ('if you take arts, we won't support you')",
      "mentions of siblings who complied and the implicit comparison",
      "physical symptoms of anxiety (not eating, not sleeping) tied to career pressure"
    ],
    "co_emotion_patterns": {
      "guilt+love": "User loves family and feels guilty for wanting something different — this co-occurrence is the defining South Asian emotional pattern",
      "conflict+helplessness": "Pulled between two legitimate needs — user's own identity and family belonging",
      "anger+shame": "Anger at the constraint combined with shame for being angry at people who love them"
    },
    "response_templates": {
      "validation_en": "Wanting something different from what your family has planned — and loving them at the same time — is one of the most painful places to be.",
      "validation_hinglish": "Apni family se pyaar karna aur phir bhi apne liye kuch alag chahna — yeh ek bahut mushkil jagah hai.",
      "reflection_en": "The pressure you're feeling isn't about anyone being wrong — it's what happens when two real things pull in opposite directions.",
      "reflection_hinglish": "Yeh kheencha-tani tab hoti hai jab do sachchi cheezein alag disha mein kheenchti hain.",
      "question_en": "What would you choose, if you knew they would still love you regardless?",
      "question_hinglish": "Agar aap jaante ki woh phir bhi aapko utna hi pyaar karenge — toh aap kya chunte?"
    },
    "what_not_to_say": [
      "Just follow your passion",
      "Your life is your own",
      "You need to stand up to them",
      "They're being controlling",
      "You'll regret not pursuing what you love"
    ],
    "reflection_frames": [
      "Career conflict in South Asian families is rarely about control for its own sake — it's usually about fear, love, and social survival",
      "The user's dilemma is legitimate on both sides — neither their desire nor their family's concern is wrong",
      "Guilt in this context is often a sign of how deeply the user loves their family, not a character flaw"
    ]
  },
  "social_judgment": {
    "cultural_context_notes": [
      "'Log kya kahenge' (what will people say) is not irrational — in many South Asian communities, social standing genuinely affects family prospects including marriages, business relationships, and social access.",
      "The extended family network functions as a surveillance and accountability system — actions of one member are experienced as reflecting on the whole family.",
      "Social media has intensified this — family achievements and failures are now broadcast to hundreds of relatives and community members.",
      "First-generation immigrant communities often maintain stricter social norms than families in the home country, as identity is preserved through cultural adherence."
    ],
    "escalation_markers": [
      "specific people mentioned (phuppo said, uncle knows, neighbours found out)",
      "language about hiding or concealing something",
      "fear of marriage prospects being affected",
      "family reputation language (naam kharab, izzat chali gayi)"
    ],
    "co_emotion_patterns": {
      "shame+anger": "Ashamed of themselves AND angry at the community standard — both need space",
      "fear+loneliness": "Fear of judgment AND feeling unable to talk to anyone about it"
    },
    "response_templates": {
      "validation_en": "The weight of community opinion is real — it's not just in your head.",
      "validation_hinglish": "Log kya sochenge — yeh sirf aapke dimag mein nahi hai, yeh ek real pressure hai.",
      "reflection_en": "When community is woven into your sense of safety and belonging, its judgment carries a different kind of weight than just opinion.",
      "question_hinglish": "Is mein aapka khud ka kya mann karta hai — agar log ki awaaz ek pal ke liye band ho jaaye?"
    },
    "what_not_to_say": [
      "Who cares what people think",
      "Live your own life",
      "Those people don't matter",
      "You're giving them too much power"
    ],
    "reflection_frames": [
      "Community judgment in South Asian contexts affects real outcomes — it is not simply a cognitive distortion to be challenged",
      "The tension is between individual authenticity and collective belonging — both are legitimate needs"
    ]
  },
  "marriage_pressure": {
    "cultural_context_notes": [
      "Marriage in South Asian families is a family project, not solely a personal one — parents often feel responsible for their children's marital status.",
      "Age pressure intensifies particularly for women — unmarried women in their late 20s face intense scrutiny.",
      "The arranged marriage process involves multiple family members, public viewings, and community networks — it is not a private negotiation.",
      "LGBTQ+ individuals face compounded pressure — the expectation of heterosexual marriage on top of identity suppression.",
      "Dowry pressure, even when illegal, remains real in many communities and can be a source of significant financial and emotional stress."
    ],
    "escalation_markers": [
      "mentions of specific age milestones ('I'm 27 and they're panicking')",
      "rishta viewings happening without consent",
      "mentions of LGBTQ+ identity alongside marriage pressure — highest risk combination",
      "financial pressure attached to marriage (dowry, reception costs)",
      "pressure to settle for a match they're not comfortable with"
    ],
    "co_emotion_patterns": {
      "conflict+anxiety": "Pulled between family's timeline and own unreadiness",
      "shame+loneliness": "Unable to tell family the real reason they're resistant to marriage",
      "fear+grief": "Grieving a future they can't have while fearing the one being imposed"
    },
    "response_templates": {
      "validation_en": "The pressure to marry on someone else's timeline — when you don't feel ready — is exhausting.",
      "validation_hinglish": "Jab shaadi kisi aur ki timeline pe karna pade aur aap khud ready na hon — yeh bahut thaka dene wala hota hai.",
      "reflection_en": "Marriage is one of the few things families treat as a collective project in ways that can leave very little room for your own voice.",
      "question_en": "What would 'ready' look like for you, if you could define it yourself?"
    },
    "what_not_to_say": [
      "Just tell them you're not ready",
      "You need to set boundaries with your parents",
      "Arranged marriages can work out — many couples are happy",
      "You should try meeting the person before deciding",
      "Have you tried talking to them honestly?"
    ],
    "reflection_frames": [
      "Marriage pressure intersects with identity, autonomy, and belonging — it is rarely a single issue",
      "LGBTQ+ individuals in this context face a categorically different situation that requires extra care and safety-first thinking"
    ]
  },
  "intergenerational_conflict": {
    "cultural_context_notes": [
      "Intergenerational conflict in South Asian families often centres on clashing frameworks of obligation vs autonomy — parents raised in collectivist contexts meeting children shaped by individualist environments.",
      "Migration creates a specific version of this: first-generation immigrants who sacrificed for their children may feel that independence is ingratitude.",
      "Communication styles differ significantly — older South Asian parents may not have frameworks for emotional conversation and may revert to authority instead.",
      "The concept of 'answering back' as disrespect means that open disagreement is often not available as a tool."
    ],
    "escalation_markers": [
      "language about not being allowed to speak ('they don't listen', 'talking is pointless')",
      "physical household conflict (shouting, objects thrown)",
      "threats of estrangement or being thrown out",
      "user has stopped communicating with family entirely"
    ],
    "co_emotion_patterns": {
      "anger+guilt": "Angry at parents AND guilty for being angry — validate both",
      "helplessness+love": "Loves family but feels unable to change anything — acknowledge the impasse",
      "grief+resentment": "Grieving a relationship that isn't what they wanted while resenting the constraint"
    },
    "response_templates": {
      "validation_hinglish": "Jab aap pyaar bhi karte hain aur samajh nahi bhi aata — dono ek saath — toh bahut akela feel hota hai.",
      "reflection_en": "Parents and children in your situation are often both right about what they feel, and both unable to fully see the other's world.",
      "question_en": "What would you most want them to understand, if they could really hear it?"
    },
    "what_not_to_say": [
      "You need to stand up for yourself",
      "That's emotional manipulation",
      "They're being toxic",
      "You don't owe them that",
      "Have you tried therapy with them?"
    ],
    "reflection_frames": [
      "Neither the parent nor the child is simply wrong — the conflict is structural, rooted in different formative worlds",
      "The user's anger is valid AND their love is valid — do not resolve this tension by taking sides"
    ]
  },
  "identity_conflict": {
    "cultural_context_notes": [
      "LGBTQ+ identity in South Asian contexts carries unique risks — Section 377 was struck down in 2018 in India but social and family persecution continues.",
      "Religious and cultural identity conflicts (leaving faith, interfaith relationships, atheism) carry community and family consequences that can include ostracism.",
      "Third-culture or diaspora identity — feeling neither fully South Asian nor fully belonging to the country of birth — is a specific and common source of distress.",
      "Coming out is not a single event in South Asian contexts — it may need to be done to multiple family members, each carrying different risks.",
      "Physical safety must always be the first question when LGBTQ+ identity is combined with family or marriage pressure."
    ],
    "escalation_markers": [
      "mentions of being 'found out' or fear of discovery",
      "LGBTQ+ identity + marriage pressure in same message — highest risk combination",
      "language about double life or hiding",
      "isolation language (no one knows, I'm completely alone in this)",
      "religious guilt language alongside identity (paap, dharm, Allah)"
    ],
    "co_emotion_patterns": {
      "shame+fear": "Core combination — shame about identity and fear of consequences",
      "loneliness+grief": "Grieving the family relationship they cannot have while being isolated",
      "love+conflict": "Loving family members who would reject them if they knew — painful and common"
    },
    "response_templates": {
      "validation_en": "Carrying something this significant — alone — takes an enormous amount of energy.",
      "reflection_en": "The gap between who you are and what feels safe to show is one of the most exhausting things a person can hold.",
      "question_en": "Is there any space in your life right now where you get to be fully yourself?",
      "safety_check_question": "Before anything else — are you physically safe at home right now?"
    },
    "what_not_to_say": [
      "You should come out to them",
      "They'll probably come around eventually",
      "Have you considered a support group?",
      "Love is love — your family should accept you",
      "You deserve to be your authentic self (without safety check first)"
    ],
    "reflection_frames": [
      "Physical safety is the first frame — before any exploration of identity expression",
      "The user's silence is often a survival strategy, not cowardice",
      "Isolation in this context is a real material condition, not just a feeling"
    ]
  },
  "financial_stress": {
    "cultural_context_notes": [
      "Provider obligation in South Asian families often falls on eldest sons in particular — financial support of parents is expected, not optional.",
      "Remittance pressure on diaspora members can mean significant portions of income go to extended family in the home country.",
      "Discussion of financial difficulty is taboo in many South Asian families — admitting struggle carries shame.",
      "Education loans in India have created a generation of young professionals starting careers with significant debt while also expected to support family."
    ],
    "escalation_markers": [
      "mentions of debt alongside family provider role",
      "hiding financial situation from family",
      "physical stress symptoms tied to financial worry",
      "language about not being able to meet basic needs"
    ],
    "co_emotion_patterns": {
      "shame+anxiety": "Ashamed of the difficulty AND anxious about the consequences",
      "helplessness+guilt": "Unable to provide at the expected level AND feeling guilty about it"
    },
    "response_templates": {
      "validation_en": "Carrying the weight of financial responsibility — for yourself and for others — is exhausting in a way that's hard to explain to people who haven't felt it.",
      "validation_hinglish": "Apne liye bhi aur doosron ke liye bhi — yeh bojh uthana bahut heavy hota hai.",
      "question_en": "What would 'enough' look like for you, just for yourself, right now?"
    },
    "what_not_to_say": [
      "Have you considered a financial advisor?",
      "You should talk to your family about this",
      "Things will get better",
      "Money isn't everything"
    ],
    "reflection_frames": [
      "Provider obligation in South Asian families is a real structural expectation, not a cognitive distortion",
      "Financial shame prevents help-seeking — creating the language of 'stress' rather than 'financial difficulty' can reduce stigma"
    ]
  }
}

SAFEGUARDING_CONTEXT = {
  "forced_marriage": {
    "definition": "Marriage arranged without free and full consent of one or both parties. Includes child marriage (under 18), marriage under threat or coercion, and pressure that removes meaningful choice.",
    "risk_level": 2,
    "indicators": [
      "marriage happening imminently against stated will",
      "minor (under 18) describing upcoming marriage",
      "threats of violence or ostracism for refusing",
      "being kept from leaving to prevent refusal",
      "partner chosen without any consultation"
    ],
    "response_principle": "Safety first. Do not explore feelings before establishing that the person is safe. Surface CHILDLINE (1098) for minors immediately.",
    "helpline_primary": "CHILDLINE (1098) for minors — 24/7 free",
    "helpline_secondary": "iCall (9152987821) for adults"
  },
  "homicidal_ideation": {
    "definition": "Thoughts of killing or seriously harming another person. Distinct from self-harm ideation. Requires safeguarding response, not crisis calm.",
    "risk_level": 2,
    "indicators": [
      "explicit statement of intent to kill or harm a specific person",
      "asking about methods of harming others",
      "in context of being trapped or threatened — may be desperate ideation, not plan",
      "distinguish from: anger metaphors ('I could kill him' in venting), fiction, hypotheticals"
    ],
    "response_principle": "Do not validate or engage with the plan. Acknowledge the desperation underneath it. Surface support resources immediately.",
    "note": "In the context of a trapped person (forced marriage, domestic violence), homicidal ideation is often an expression of extreme desperation, not a genuine plan. The safeguarding need is still the priority."
  },
  "threats_to_user": {
    "definition": "Another person has threatened to physically harm or kill the user.",
    "risk_level": 2,
    "indicators": [
      "explicit statement that someone will kill or hurt them if they do X",
      "history of physical violence in the household",
      "fear of returning home"
    ],
    "response_principle": "Treat as active safeguarding concern. Do not suggest family conversation. Ask if they are safe right now.",
    "helpline_primary": "iCall (9152987821)",
    "helpline_secondary": "Police (100) if immediate danger"
  }
}

RESPONSE_QUALITY_ANCHORS = {
  "what_good_validation_looks_like": [
    "Names the specific emotion the user expressed, not a generic one",
    "Uses the user's own language register (Hinglish if they wrote Hinglish)",
    "Does not pivot to anything else before the user feels heard",
    "Avoids hollow affirmations: 'Absolutely!', 'That makes so much sense!'",
    "Does not reframe, minimize, or silver-line in the validation section"
  ],
  "what_good_reflection_looks_like": [
    "Names the cultural or systemic weight behind what the user is experiencing",
    "Does not lecture or moralize",
    "Introduces one genuinely new lens, not a restatement of the validation",
    "Is specific to the stressor, not generic ('life can be hard')",
    "Is 2-3 sentences maximum"
  ],
  "what_good_question_looks_like": [
    "Is genuinely open — cannot be answered yes/no",
    "Does not contain hidden advice ('Have you considered talking to them?')",
    "Is exactly one question",
    "Uses the user's language register",
    "Opens new territory rather than summarizing what was already said"
  ],
  "common_failure_modes": {
    "generic_frustration_default": "Assuming frustration when nothing in the input signals it — check intent before assuming negative emotion",
    "advice_as_question": "Embedding advice in a question form: 'Have you thought about...?' 'Why not try...?'",
    "premature_reframe": "Introducing a silver lining before validation is complete",
    "cultural_flattening": "Treating South Asian stressors as equivalent to Western ones — exam pressure is not just 'stress'",
    "hollow_affirmation": "Starting responses with 'Absolutely!', 'That makes so much sense!', 'Great question!'"
  }
}

if __name__ == "__main__":
    print(f"Backing up to {KB_PATH}.bak")
    shutil.copy(KB_PATH, KB_PATH + ".bak")
    
    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)
    
    # Enrich each stressor
    for sid, enrichment in ENRICHMENT.items():
        if sid in kb["stressor_taxonomy"]:
            kb["stressor_taxonomy"][sid].update(enrichment)
            print(f"  enriched: {sid}")
    
    # Add new top-level sections
    kb["safeguarding_context"] = SAFEGUARDING_CONTEXT
    kb["response_quality_anchors"] = RESPONSE_QUALITY_ANCHORS
    
    # Add missing crisis keywords
    kb["crisis_keywords"].setdefault("tier_2", [])
    tier2_kw = [
        "shall i kill them", "shall i kill my father", "shall i kill my family",
        "homicide", "murder them", "kill my parents",
        "my father will kill me", "they will kill me", "woh mujhe maar denge",
        "forced to marry", "getting married tomorrow against",
        "child marriage", "married to a 9 year old"
    ]
    for kw in tier2_kw:
        if kw not in kb["crisis_keywords"]["tier_2"]:
            kb["crisis_keywords"]["tier_2"].append(kw)

    # Enrich cultural idioms with risk context
    idiom_risk = {
        "log_kya_kahenge": {"risk_amplifier": True, "escalates_stressors": ["social_judgment", "marriage_pressure", "identity_conflict"]},
        "izzat":           {"risk_amplifier": True, "escalates_stressors": ["social_judgment", "marriage_pressure"]},
        "ghar_ki_izzat":   {"risk_amplifier": True, "escalates_stressors": ["social_judgment", "marriage_pressure", "identity_conflict"]},
    }
    for idiom_id, extra in idiom_risk.items():
        if idiom_id in kb["cultural_idioms"]:
            kb["cultural_idioms"][idiom_id].update(extra)

    kb["_meta"]["last_enriched"] = datetime.date.today().isoformat()
    kb["_meta"]["enrichment_version"] = "2.0"
    
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    
    print(f"\nKB enriched and saved.")
    print(f"Stressors enriched: {len(ENRICHMENT)}")
    print(f"New sections added: safeguarding_context, response_quality_anchors")
    print(f"Tier-2 crisis keywords added: {len(tier2_kw)}")
