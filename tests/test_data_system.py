"""
tests/test_data_system.py
Tests for the SADEV data + learning system.

Covers:
  - dataset/generator.py   — prompt building, record parsing, batch saving
  - dataset/cleaner.py     — text cleaning, language detection, rejection rules
  - dataset/deduplicator.py— exact and cross-batch deduplication
  - dataset/validator.py   — schema enforcement, label correction rules
  - dataset/pipeline.py    — end-to-end in-memory pipeline
  - models/multitask_model.py — label maps, ordinal urgency helpers

No LLM or GPU required — all tests are offline.
"""
import json
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from dataset.cleaner      import TextCleaner
from dataset.deduplicator import Deduplicator, _fingerprint
from dataset.validator    import Validator, INTENT_TO_CATEGORY
from dataset.pipeline     import DataPipeline, _save_jsonl, _load_jsonl
from dataset.generator    import (
    DatasetGenerator, _build_prompt, _parse_records,
    VALID_INTENTS, INTENT_CONFIG, RawRecord,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_record(**kwargs) -> dict:
    base = {
        "text":     "yaar bahut stress ho raha hai exams se",
        "intent":   "academic_pressure",
        "category": "life_stressor",
        "urgency":  3,
        "language": "hi-en-mix",
    }
    base.update(kwargs)
    return base


def make_records(n: int, **kwargs) -> list[dict]:
    return [make_record(text=f"Sample text number {i} about stress", **kwargs)
            for i in range(n)]


# ── Generator tests ───────────────────────────────────────────────────────

class TestGenerator:

    def test_valid_intents_non_empty(self):
        assert len(VALID_INTENTS) >= 20

    def test_all_intents_have_config(self):
        for intent in VALID_INTENTS:
            assert intent in INTENT_CONFIG, f"Missing config for intent: {intent}"

    def test_intent_config_has_required_keys(self):
        for intent, cfg in INTENT_CONFIG.items():
            assert "category" in cfg
            assert "urgency"  in cfg
            assert "lang"     in cfg

    def test_build_prompt_returns_tuple(self):
        system, user = _build_prompt("venting_distress", n=5, language="hi-en-mix")
        assert isinstance(system, str) and len(system) > 50
        assert isinstance(user, str)   and len(user) > 50

    def test_build_prompt_contains_intent(self):
        _, user = _build_prompt("academic_pressure", n=10, language="en")
        assert "academic_pressure" in user

    def test_build_prompt_crisis_safety_note(self):
        system, _ = _build_prompt("suicidal_thoughts", n=5, language="en")
        assert "method" in system.lower() or "step-by-step" in system.lower()

    def test_parse_records_valid_json_array(self):
        raw = json.dumps([
            {"text": "yaar stress ho raha hai", "intent": "venting_distress",
             "category": "emotional", "urgency": 2, "language": "hi-en-mix"},
        ])
        records = _parse_records(raw, "venting_distress", "emotional", "b1", "test", "abc")
        assert len(records) == 1
        assert isinstance(records[0], RawRecord)
        assert records[0].text == "yaar stress ho raha hai"

    def test_parse_records_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps([
            {"text": "I feel so lost", "intent": "sadness_low_mood",
             "category": "emotional", "urgency": 3, "language": "en"},
        ]) + "\n```"
        records = _parse_records(raw, "sadness_low_mood", "emotional", "b1", "test", "abc")
        assert len(records) == 1

    def test_parse_records_urgency_clamped(self):
        raw = json.dumps([
            {"text": "I'm okay", "intent": "greeting",
             "category": "neutral", "urgency": 99, "language": "en"},
        ])
        records = _parse_records(raw, "greeting", "neutral", "b1", "test", "abc")
        assert records[0].urgency == 5  # clamped to max

    def test_parse_records_skips_short_text(self):
        raw = json.dumps([
            {"text": "hi", "intent": "greeting", "category": "neutral", "urgency": 1, "language": "en"},
            {"text": "I have been feeling really overwhelmed by everything lately.",
             "intent": "venting_distress", "category": "emotional", "urgency": 3, "language": "en"},
        ])
        records = _parse_records(raw, "greeting", "neutral", "b1", "test", "abc")
        assert len(records) == 1

    def test_parse_records_invalid_json_returns_empty(self):
        records = _parse_records("NOT JSON AT ALL !!!", "greeting", "neutral", "b1", "test", "abc")
        assert records == []

    def test_save_and_load_batch(self):
        records = [
            RawRecord(text="Sample text", intent="greeting", category="neutral",
                      urgency=1, language="en", _gen_model="test", _gen_batch_id="b1",
                      _gen_prompt_hash="abc", _gen_ts="2025-01-01"),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            DatasetGenerator.save_batch(records, path)
            loaded = DatasetGenerator.load_batch(path)
            assert len(loaded) == 1
            assert loaded[0]["text"] == "Sample text"
        finally:
            os.unlink(path)


# ── Cleaner tests ─────────────────────────────────────────────────────────

class TestTextCleaner:

    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_clean_removes_urls(self):
        result = self.cleaner.clean("Check this https://example.com for info", "en")
        assert "https://example.com" not in result.text
        assert not result.dropped

    def test_clean_removes_zero_width_chars(self):
        text = "hello\u200bworld"
        result = self.cleaner.clean(text, "en")
        assert "\u200b" not in result.text

    def test_clean_normalises_excess_space(self):
        result = self.cleaner.clean("too   many    spaces here", "en")
        assert "  " not in result.text

    def test_clean_drops_meta_language(self):
        result = self.cleaner.clean("As a user I would say training data example", "en")
        assert result.dropped
        assert result.drop_reason == "meta_language_detected"

    def test_clean_drops_too_short(self):
        result = self.cleaner.clean("hi", "en")
        assert result.dropped
        assert result.drop_reason == "too_short"

    def test_clean_truncates_too_long(self):
        long_text = "word " * 400  # ~2000 chars
        result = self.cleaner.clean(long_text, "en")
        assert not result.dropped
        assert len(result.text) <= 1200

    def test_clean_limits_repeated_punctuation(self):
        result = self.cleaner.clean("WHY!!!!! is this happening????", "en")
        assert "!!!!!" not in result.text
        assert "????" not in result.text

    def test_language_upgraded_to_hinglish(self):
        text = "yaar mujhe bahut stress ho raha hai aur kuch samajh nahi aata"
        result = self.cleaner.clean(text, language="en")
        assert result.language == "hi-en-mix"
        assert result.language_corrected

    def test_language_downgraded_from_hinglish(self):
        text = "I have been feeling really overwhelmed and anxious about the future lately."
        result = self.cleaner.clean(text, language="hi-en-mix")
        assert result.language == "en"
        assert result.language_corrected

    def test_clean_batch_splits_accepted_rejected(self):
        records = [
            {"text": "I feel really stressed about my job situation", "language": "en"},
            {"text": "hi", "language": "en"},  # too short → rejected
            {"text": "as a user I would type training data", "language": "en"},  # meta → rejected
        ]
        accepted, rejected = self.cleaner.clean_batch(records)
        assert len(accepted) == 1
        assert len(rejected) == 2

    def test_original_text_preserved(self):
        text = "I feel   really   stressed"
        records = [{"text": text, "language": "en"}]
        accepted, _ = self.cleaner.clean_batch(records)
        assert accepted[0]["_original_text"] == text
        assert accepted[0]["text"] != text  # was normalised


# ── Deduplicator tests ────────────────────────────────────────────────────

class TestDeduplicator:

    def setup_method(self):
        self.deduper = Deduplicator(semantic=False)

    def test_fingerprint_normalises(self):
        fp1 = _fingerprint("Hello, World!")
        fp2 = _fingerprint("hello world")
        assert fp1 == fp2

    def test_exact_dedup_removes_duplicate(self):
        records = [
            make_record(text="I am so stressed about exams"),
            make_record(text="I am so stressed about exams"),  # exact duplicate
        ]
        result = self.deduper.deduplicate(records)
        assert len(result.kept) == 1
        assert result.exact_removed == 1

    def test_exact_dedup_keeps_unique(self):
        records = [
            make_record(text="I am so stressed about exams"),
            make_record(text="My parents are pressuring me about marriage"),
        ]
        result = self.deduper.deduplicate(records)
        assert len(result.kept) == 2
        assert result.exact_removed == 0

    def test_cross_batch_dedup(self):
        existing = {_fingerprint("I am stressed about my job")}
        records = [make_record(text="I am stressed about my job")]
        result = self.deduper.deduplicate(records, existing_fingerprints=existing)
        assert len(result.kept) == 0
        assert result.exact_removed == 1

    def test_dedup_result_total_input(self):
        records = make_records(10)
        result = self.deduper.deduplicate(records)
        assert result.total_input == 10

    def test_compute_fingerprints_returns_set(self):
        records = make_records(5)
        fps = self.deduper.compute_fingerprints(records)
        assert isinstance(fps, set)
        assert len(fps) <= 5  # could be fewer if texts are similar


# ── Validator tests ───────────────────────────────────────────────────────

class TestValidator:

    def setup_method(self):
        self.v = Validator(strict_schema=False)

    def test_valid_record_passes_clean(self):
        result = self.v.validate(make_record())
        assert not result.dropped
        assert not result.was_corrected

    # ── Crisis promotion ──────────────────────────────────────────────────

    def test_crisis_signal_promotes_category(self):
        rec = make_record(text="I want to end my life, I can't go on",
                          intent="sadness_low_mood", category="emotional", urgency=3)
        result = self.v.validate(rec)
        assert result.record["category"] == "crisis"
        assert result.record["urgency"] == 5

    def test_crisis_signal_promotes_intent(self):
        rec = make_record(text="Main marna chahta hun, koi raasta nahi",
                          intent="venting_distress", category="emotional", urgency=3)
        result = self.v.validate(rec)
        assert result.record["intent"] == "suicidal_thoughts"

    def test_crisis_urgency_already_5_no_correction(self):
        rec = make_record(text="I want to end my life",
                          intent="suicidal_thoughts", category="crisis", urgency=5)
        result = self.v.validate(rec)
        assert result.record["urgency"] == 5
        # Should not add urgency correction since it's already 5
        urgency_corrections = [c for c in result.corrections if "urgency_crisis_promoted" in c]
        assert len(urgency_corrections) == 0

    # ── Safe metaphor protection ──────────────────────────────────────────

    def test_safe_metaphor_not_promoted_to_crisis(self):
        rec = make_record(text="I'm killing this exam today!",
                          intent="academic_pressure", category="life_stressor", urgency=1)
        result = self.v.validate(rec)
        assert result.record["category"] != "crisis"

    def test_safe_metaphor_demotes_from_crisis(self):
        rec = make_record(text="Killing it in this interview",
                          intent="career_job_stress", category="crisis", urgency=4)
        result = self.v.validate(rec)
        assert result.record["category"] == "life_stressor"
        assert result.record["urgency"] <= 2

    def test_dying_of_laughter_not_crisis(self):
        rec = make_record(text="dying of laughter at this meme rn lol",
                          intent="casual_chat", category="neutral", urgency=1)
        result = self.v.validate(rec)
        assert result.record["category"] != "crisis"

    # ── Urgency floor/ceiling constraints ─────────────────────────────────

    def test_crisis_category_urgency_floor_4(self):
        rec = make_record(intent="self_harm_intent", category="crisis", urgency=2)
        result = self.v.validate(rec)
        assert result.record["urgency"] >= 4

    def test_high_distress_urgency_floor_3(self):
        rec = make_record(intent="hopelessness", category="high_distress", urgency=1)
        result = self.v.validate(rec)
        assert result.record["urgency"] >= 3

    def test_neutral_urgency_capped_at_1(self):
        rec = make_record(intent="greeting", category="neutral", urgency=3)
        result = self.v.validate(rec)
        assert result.record["urgency"] == 1

    # ── Schema enforcement ────────────────────────────────────────────────

    def test_invalid_intent_corrected(self):
        rec = make_record(intent="completely_invalid_intent_xyz")
        result = self.v.validate(rec)
        assert not result.dropped
        assert result.record["intent"] == "casual_chat"

    def test_missing_urgency_defaults(self):
        rec = make_record()
        del rec["urgency"]
        result = self.v.validate(rec)
        assert not result.dropped
        assert isinstance(result.record["urgency"], int)

    def test_empty_text_dropped(self):
        rec = make_record(text="")
        result = self.v.validate(rec)
        assert result.dropped

    def test_string_urgency_coerced(self):
        rec = make_record(urgency="3")
        result = self.v.validate(rec)
        assert result.record["urgency"] == 3

    def test_batch_split(self):
        records = [
            make_record(text="I want to end my life", intent="casual_chat",
                        category="neutral", urgency=1),                          # crisis → corrected
            make_record(),                                                        # clean
            make_record(text="", intent="greeting", category="neutral"),         # dropped
        ]
        clean, corrected, dropped = self.v.validate_batch(records)
        assert len(dropped) == 1
        assert len(corrected) == 1
        assert len(clean) == 1


# ── Pipeline tests ────────────────────────────────────────────────────────

class TestDataPipeline:

    def setup_method(self):
        self.pipeline = DataPipeline(semantic_dedup=False)

    def _make_raw_records(self, n=10, **kwargs) -> list[dict]:
        recs = []
        for i in range(n):
            recs.append({
                "text":     f"I have been feeling really stressed about everything number {i}",
                "intent":   "venting_distress",
                "category": "emotional",
                "urgency":  3,
                "language": "en",
                "_meta": {"model": "test", "batch_id": "b1", "prompt_hash": "abc", "generated_at": "2025-01-01"},
            })
        return recs

    def test_pipeline_runs_on_clean_input(self):
        records = self._make_raw_records(10)
        final, stats = self.pipeline.run_from_records(records, verbose=False)
        assert len(final) > 0
        assert stats.raw_loaded == 10

    def test_pipeline_deduplicates(self):
        text = "I feel very overwhelmed by everything going on in my life right now"
        records = [
            {"text": text, "intent": "overwhelm_burnout",
             "category": "high_distress", "urgency": 3, "language": "en"},
            {"text": text, "intent": "overwhelm_burnout",
             "category": "high_distress", "urgency": 3, "language": "en"},
        ]
        final, stats = self.pipeline.run_from_records(records, verbose=False)
        assert stats.dedup_exact == 1
        assert len(final) == 1

    def test_pipeline_strips_meta_keys(self):
        records = self._make_raw_records(3)
        final, _ = self.pipeline.run_from_records(records, verbose=False)
        for rec in final:
            for key in rec:
                assert not key.startswith("_"), f"Meta key leaked into output: {key}"

    def test_pipeline_corrects_crisis_labels(self):
        records = [{
            "text":     "I want to end my life, I can't do this anymore",
            "intent":   "venting_distress",
            "category": "emotional",
            "urgency":  2,
            "language": "en",
        }]
        final, stats = self.pipeline.run_from_records(records, verbose=False)
        assert len(final) == 1
        assert final[0]["category"] == "crisis"
        assert final[0]["urgency"] == 5
        assert stats.validate_corrected == 1

    def test_pipeline_drops_short_text(self):
        records = [
            {"text": "hi", "intent": "greeting", "category": "neutral", "urgency": 1, "language": "en"},
        ]
        _, stats = self.pipeline.run_from_records(records, verbose=False)
        assert stats.clean_dropped == 1
        assert stats.final_written == 0

    def test_pipeline_drops_meta_language(self):
        records = [{
            "text":     "As a user I would generate this training data example here",
            "intent":   "casual_chat",
            "category": "neutral",
            "urgency":  1,
            "language": "en",
        }]
        _, stats = self.pipeline.run_from_records(records, verbose=False)
        assert stats.clean_dropped == 1

    def test_pipeline_stats_consistent(self):
        records = self._make_raw_records(20)
        _, stats = self.pipeline.run_from_records(records, verbose=False)
        # after_clean + clean_dropped should equal raw_loaded
        assert stats.after_clean + stats.clean_dropped == stats.raw_loaded
        # after_validate should equal after_dedup - validate_dropped
        assert stats.after_validate == stats.after_dedup - stats.validate_dropped


# ── JSONL I/O tests ───────────────────────────────────────────────────────

class TestJSONLIO:

    def test_save_and_load_roundtrip(self):
        records = [
            {"text": "test one", "intent": "greeting", "urgency": 1},
            {"text": "test two", "intent": "venting_distress", "urgency": 3},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            _save_jsonl(records, path)
            loaded = _load_jsonl(path)
            assert len(loaded) == 2
            assert loaded[0]["text"] == "test one"
        finally:
            os.unlink(path)

    def test_save_append_mode(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            _save_jsonl([{"a": 1}], path)
            _save_jsonl([{"b": 2}], path, append=True)
            loaded = _load_jsonl(path)
            assert len(loaded) == 2
        finally:
            os.unlink(path)


# ── Model label map tests (no GPU needed) ─────────────────────────────────

class TestModelLabelMaps:

    def test_intent_label_count(self):
        from models.multitask_model import INTENT_LABELS
        assert len(INTENT_LABELS) == 26

    def test_category_label_count(self):
        from models.multitask_model import CATEGORY_LABELS
        assert len(CATEGORY_LABELS) == 8

    def test_intent_to_idx_bidirectional(self):
        from models.multitask_model import INTENT_TO_IDX, IDX_TO_INTENT
        for intent, idx in INTENT_TO_IDX.items():
            assert IDX_TO_INTENT[idx] == intent

    def test_category_to_idx_bidirectional(self):
        from models.multitask_model import CATEGORY_TO_IDX, IDX_TO_CATEGORY
        for cat, idx in CATEGORY_TO_IDX.items():
            assert IDX_TO_CATEGORY[idx] == cat

    def test_urgency_to_ordinal_targets(self):
        from models.multitask_model import urgency_to_ordinal_targets
        t = urgency_to_ordinal_targets(3)
        vals = list(t) if not isinstance(t, list) else t
        assert [int(v) for v in vals] == [1, 1, 0, 0]

    def test_urgency_1_ordinal(self):
        from models.multitask_model import urgency_to_ordinal_targets
        t = urgency_to_ordinal_targets(1)
        vals = list(t) if not isinstance(t, list) else t
        assert [int(v) for v in vals] == [0, 0, 0, 0]

    def test_urgency_5_ordinal(self):
        from models.multitask_model import urgency_to_ordinal_targets
        t = urgency_to_ordinal_targets(5)
        vals = list(t) if not isinstance(t, list) else t
        assert [int(v) for v in vals] == [1, 1, 1, 1]
