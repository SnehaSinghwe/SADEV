# SADEV — Data Pipeline Documentation

> **Scope:** This document covers the data generation, processing, and pipeline infrastructure only.  
> Frontend, chat interface, and deployment docs are out of scope here.

---

## Project Structure (Data Layer)

```
SADEV_LOCAL1/
├── data/
│   ├── raw/                        # Raw batches from generator (unvalidated)
│   ├── interim/                    # Partially cleaned / in-progress splits
│   └── processed/                  # Final, model-ready datasets
│
├── data/dataset/
│   ├── hinglish_dataset_builder.py # Core Hinglish data schema & builder logic
│   ├── generator.py                # Batch data generation (LLM-assisted or rule-based)
│   ├── validator.py                # Schema + quality validation checks
│   ├── cleaner.py                  # Normalization, noise removal, formatting fixes
│   ├── deduplicator.py             # Exact + fuzzy deduplication
│   └── pipeline.py                 # Main orchestrator — runs full data pipeline
│
├── scripts/
│   ├── run_generation.py           # Entry point: generate raw batches
│   └── run_data_pipeline.py        # Entry point: validate → clean → dedup → export
```

---

## Data Flow

```
[generator.py]
      │  raw JSON batches
      ▼
  data/raw/
      │
      ▼
[validator.py]       ← schema checks, label sanity, language detection
      │  valid samples only
      ▼
[cleaner.py]         ← normalize text, fix encoding, strip noise
      │
      ▼
  data/interim/
      │
      ▼
[deduplicator.py]    ← exact hash dedup + optional MinHash fuzzy dedup
      │
      ▼
  data/processed/    ← final train/val/test splits, ready for model
```

---

## Module Descriptions

### `generator.py`
Generates raw Hinglish conversational samples. Supports:
- LLM-assisted generation via API (configurable model/prompt)
- Rule-based templating for controlled scenarios
- Batch output format: `data/raw/batch_<timestamp>.jsonl`

**Key config params:**
```python
BATCH_SIZE = 500          # samples per batch
TARGET_LANG = "hinglish"  # hinglish | hindi | english
EMOTION_LABELS = [...]    # emotion tags to include
SAFETY_TAGS = [...]       # safety scenario flags
```

---

### `validator.py`
Validates each sample against the schema defined in `hinglish_dataset_builder.py`.

Checks performed:
- Required fields present (`text`, `label`, `lang_tag`, `source`)
- Label is within allowed set
- Text length within bounds (min/max tokens)
- Language tag consistency (basic script detection)
- No null/empty fields

Outputs: valid samples forwarded, invalid samples logged to `data/interim/rejected_<timestamp>.jsonl`

---

### `cleaner.py`
Normalizes and cleans validated samples.

Operations:
- Unicode normalization (NFC)
- Devanagari/Latin script normalization
- Remove HTML tags, extra whitespace, control characters
- Standardize punctuation
- Strip PII patterns (phone numbers, emails) — regex-based

---

### `deduplicator.py`
Two-stage deduplication:

| Stage | Method | Use Case |
|-------|--------|----------|
| Exact | SHA-256 hash on normalized text | Catches copy-paste duplicates |
| Fuzzy | MinHash + LSH (datasketch) | Catches near-duplicate paraphrases |

**Config:**
```python
FUZZY_THRESHOLD = 0.85    # Jaccard similarity threshold
NUM_PERM = 128            # MinHash permutations (accuracy vs speed tradeoff)
```

---

### `pipeline.py`
Orchestrates the full data pipeline in sequence:

```python
# Internal execution order
load_raw_batches()
  → validate_all()
  → clean_all()
  → deduplicate()
  → split_train_val_test(ratios=[0.8, 0.1, 0.1])
  → export_to_processed()
```

Produces final outputs in `data/processed/`:
```
data/processed/
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── metadata.json       # sample counts, label distribution, dedup stats
```

---

### `hinglish_dataset_builder.py`
Defines the canonical data schema and builder utilities.

**Sample schema:**
```json
{
  "id": "uuid",
  "text": "Yaar, kya scene hai aaj?",
  "lang_tag": "hinglish",
  "emotion": "curiosity",
  "safety_flag": null,
  "intent": "casual_inquiry",
  "source": "generated",
  "quality_score": 0.91
}
```

---

## Entry Point Scripts

### Generate raw data
```bash
python scripts/run_generation.py \
  --batch-size 500 \
  --num-batches 10 \
  --output-dir data/raw/
```

### Run full data pipeline
```bash
python scripts/run_data_pipeline.py \
  --input-dir data/raw/ \
  --output-dir data/processed/ \
  --fuzzy-dedup \
  --split 0.8 0.1 0.1
```

---

## Data Quality Targets

| Metric | Target |
|--------|--------|
| Post-dedup retention | > 70% of generated |
| Validation pass rate | > 85% per batch |
| Label balance (per class) | ± 15% of mean |
| Avg quality score | > 0.80 |

If a batch consistently fails below these thresholds, check generator prompt config or label definitions in `hinglish_dataset_builder.py`.

---

## Dependencies

```
datasketch          # MinHash / LSH deduplication
langdetect          # language tag verification
ftfy                # unicode/encoding fixes
tqdm                # progress bars
jsonlines           # .jsonl read/write
```

Install:
```bash
pip install datasketch langdetect ftfy tqdm jsonlines
```

---

## Notes

- All intermediate outputs in `data/interim/` are safe to delete after pipeline completes.
- `data/raw/` batches are kept for reproducibility — don't delete unless storage is a constraint.
- `metadata.json` in `data/processed/` is required by the training scripts in `models/train.py` — always regenerate it when rerunning the pipeline.
- The pipeline is idempotent: rerunning on the same raw input produces the same processed output (given fixed fuzzy threshold and seed).
