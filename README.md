# SADEV — Build Guide
**Sadev (सदैव) — Always.** Culturally adaptive AI mental health support for South Asian youth.

---

## Architecture Overview

```
User message
     │
     ▼
[1] Input Capture    :8001  — language detect, PII scrub, normalise
     │
     ├──────────────────────────┐
     ▼                          ▼  (concurrent — asyncio.gather)
[2] NLU              :8002  [5] Risk Detection  :8005
    intent, emotion,            keyword match,
    urgency, entities           sentiment trajectory
     │                          │
     ▼                          │ (if urgency > 1 → second risk check)
[3] CCE              :8003      │
    stressor, tone,             │
    guardrails, framing         │
     │                          │
     ▼                          │
[3.5] Retrieval      :8007      │   (RAG — cultural KB vector search)
    top-k chunks                │
    context block               │
     │                          │
     ▼                          │
[4] Response Gen     :8004 ◄────┘
    LLM + CBT + nudge
    (injected: framing, guardrails, RAG context, risk level)
     │
     ▼
[6] Storage          :8006  — session summary (no raw content, no PII)
     │
     ▼
Response delivered
```

> **The one rule above all others:** Risk detection (stage 5) receives every message the moment it enters the gateway — before NLU completes, before CCE, before response generation. It runs via `asyncio.gather`. If you ever find risk detection waiting on any other service, the architecture is broken.

---

## Quick Start — Full Local Stack (Ollama)

SADEV runs entirely offline using Ollama as the LLM backend. No Anthropic API key required for local development.

### Prerequisites
- Python 3.10+
- Ollama installed and running
- MongoDB (local or Atlas)
- Firebase project (or emulator)

### Step 1 — Install dependencies

```bash
git clone https://github.com/your-org/sadev
cd sadev
pip install -r requirements.txt --break-system-packages

cp .env.example .env
nano .env   # fill in SERVICE_KEY, Firebase path, Mongo URI
```

### Step 2 — Pull and serve the local LLM

```bash
# Pull the model (one-time, ~5 GB)
ollama pull gemma2

# Keep this running in a separate terminal
ollama serve
```

Ollama serves at `http://localhost:11434` by default.

| Model | Use case |
|-------|----------|
| `gemma2` | Recommended default |
| `gemma2:2b` | Lower memory |
| `llama3.1:8b` | Higher quality synthetic data |

---

## Data Pipeline

### Project layout

```
SADEV_LOCAL1/
├── data/
│   ├── raw/                        # Raw batches from generator (unvalidated)
│   ├── interim/                    # Partially cleaned / rejected logs
│   └── processed/                  # Final model-ready datasets + RAG chunks
│
├── data/dataset/
│   ├── hinglish_dataset_builder.py # Canonical schema & builder utilities
│   ├── generator.py                # Batch generation (LLM-assisted + rule-based)
│   ├── validator.py                # Schema, label, language-tag validation
│   ├── cleaner.py                  # Normalization, encoding fixes, PII strip
│   ├── deduplicator.py             # Exact hash + MinHash fuzzy dedup
│   └── pipeline.py                 # Orchestrator — runs the full sequence
│
├── scripts/
│   ├── run_generation.py           # Entry point: generate raw batches
│   └── run_data_pipeline.py        # Entry point: validate → clean → dedup → export
```

### Data flow

```
[generator.py]
      │  raw JSONL batches → data/raw/
      ▼
[validator.py]       ← schema checks, label sanity, language detection
      │  valid samples only; rejects → data/interim/rejected_<ts>.jsonl
      ▼
[cleaner.py]         ← unicode normalise, fix encoding, strip PII, standardise punct
      │
      ▼
  data/interim/      ← partially processed, safe to delete after pipeline completes
      │
      ▼
[deduplicator.py]    ← exact SHA-256 dedup → MinHash fuzzy dedup
      │
      ▼
  data/processed/    ← train.jsonl / val.jsonl / test.jsonl + metadata.json
```

### Step 3 — Generate raw data

```bash
# Auto-fill all intent classes below minimum thresholds
python scripts/run_generation.py --fill-gaps --n-per-intent 30 --n-per-call 10

# Single class, specific language
python scripts/run_generation.py --intent crisis_signal --language hi-en-mix --n 50

# Dry run — preview prompts, no LLM calls
python scripts/run_generation.py --intent seek_validation --n 5 --dry-run

# Specific urgency level
python scripts/run_generation.py --intent express_academic_pressure --urgency 2 --n 20
```

> **Crisis cap:** Crisis examples (urgency 4–5) are capped at 50 per run and generated in a separate restricted pass with additional safety instructions in the prompt.

### Step 4 — Run the data pipeline

```bash
python scripts/run_data_pipeline.py \
    --input-dir data/raw/ \
    --output-dir data/processed/ \
    --fuzzy-dedup \
    --split 0.8 0.1 0.1
```

**Pipeline stages (in order):**
1. Schema validation + label sanity (`validator.py`)
2. Text normalization + PII strip (`cleaner.py`)
3. Exact dedup (SHA-256) → fuzzy dedup (MinHash, threshold 0.85) (`deduplicator.py`)
4. Stratified train/val/test split
5. Export `processed_chunks.json` for the RAG retrieval service

**Outputs:**

| File | Description |
|------|-------------|
| `data/processed/train.jsonl` | Training set (quality-filtered) |
| `data/processed/val.jsonl` | Validation set |
| `data/processed/test.jsonl` | Held-out test set |
| `data/processed/metadata.json` | Sample counts, label distribution, dedup stats |
| `data/processed/processed_chunks.json` | Chunked knowledge base for RAG |
| `data/interim/rejected_<ts>.jsonl` | Rows that failed validation, with reasons |

Report-only mode (no writes):

```bash
python data/dataset/validator.py \
    --input data/raw/ \
    --report-only
```

**Data quality targets:**

| Metric | Target |
|--------|--------|
| Post-dedup retention | > 70% of generated |
| Validation pass rate | > 85% per batch |
| Label balance (per class) | ± 15% of mean |
| Avg quality score | > 0.80 |

If a batch consistently fails below these thresholds, check generator prompt config or label definitions in `hinglish_dataset_builder.py`.

### Sample schema (`hinglish_dataset_builder.py`)

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

### Deduplicator config

```python
FUZZY_THRESHOLD = 0.85    # Jaccard similarity cutoff
NUM_PERM = 128            # MinHash permutations — accuracy vs speed tradeoff
```

### Data pipeline dependencies

```bash
pip install datasketch langdetect ftfy tqdm jsonlines
```

---

## NLU Dataset Annotation

Before MuRIL training, 4,200+ examples must be labelled using the schema in `data/labelling_schema.json`.

**Target distribution:**
- 40% Hinglish (hi-en-mix), 30% English, 15% Hindi, 5% Bengali-English, 10% other
- Minimum 300 `crisis_signal` examples — oversample this class
- Inter-annotator agreement target: Cohen's Kappa > 0.75

---

## Step 5 — Train the NLU Model

```bash
# Fine-tune MuRIL, evaluate on dev after each epoch, final eval on test
python scripts/run_training.py --eval-after-train
```

Full options:

```bash
python ml/train.py \
    --data_path data/processed/train.jsonl \
    --dev_path  data/processed/val.jsonl \
    --output_dir ml/checkpoints \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5
```

Evaluate a saved checkpoint:

```bash
python ml/evaluate.py \
    --checkpoint_dir ml/checkpoints/best_model \
    --test_csv data/processed/test.jsonl
```

**Training targets:**

| Metric | Gate |
|--------|------|
| Macro F1 (all intents) | > 0.78 |
| `crisis_signal` recall | **> 0.90 — mandatory** |
| False negative rate (urgency 4–5) | < 0.05 |

Training output will show `✓ GATE PASSED` or `✗ GATE FAILED`. The model will not be marked deployment-ready unless the crisis recall gate passes.

---

## Step 6 — Build the RAG Index

```bash
# Embed the cultural knowledge base into a FAISS index (~2 min, one-time)
python -m services.retrieval.index_builder
```

Rebuild whenever `data/knowledge_base_raw.json` changes.

---

## Step 7 — Run the Project

```bash
# Start all 7 services + gateway
python chat.py
```

- Gateway: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

Individual services:

```bash
uvicorn services.nlu.main:app --port 8002 --reload
uvicorn services.risk_detection.main:app --port 8005 --reload
```

---

## NLU Stub Mode

The NLU service ships with `NLU_STUB_MODE=true`. The stub uses keyword heuristics to return plausible intent/emotion/urgency — the full pipeline (RAG, CCE, response generation, risk detection) works end-to-end before MuRIL training is complete.

To switch to the real model after the gate passes:

```bash
# .env
NLU_STUB_MODE=false
MURIL_ENDPOINT=http://localhost:8090
```

```bash
uvicorn ml.serve:app --port 8090
# then restart services
```

---

## Service Map

| Port | Service | File |
|------|---------|------|
| 8000 | Gateway (orchestrator) | `gateway.py` |
| 8001 | Input capture | `services/input_capture/main.py` |
| 8002 | NLU | `services/nlu/main.py` |
| 8003 | Cultural context engine | `services/cce/main.py` |
| 8004 | Response generator | `services/response_gen/main.py` |
| 8005 | Risk detection | `services/risk_detection/main.py` |
| 8006 | Storage | `services/storage/main.py` |
| 8007 | Retrieval (RAG) | `services/retrieval/main.py` |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SERVICE_KEY` | ✓ | Shared secret for inter-service auth |
| `ANTHROPIC_API_KEY` | — | Claude API key (cloud LLM mode only) |
| `OLLAMA_BASE_URL` | — | Ollama endpoint (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | — | Ollama model name (default: `gemma2`) |
| `FIREBASE_CREDENTIALS_PATH` | ✓ | Path to Firebase service account JSON |
| `FIREBASE_PROJECT_ID` | ✓ | Firebase project ID |
| `MONGO_URI` | ✓ | MongoDB Atlas or local connection string |
| `MONGO_DB` | ✓ | MongoDB database name |
| `NLU_STUB_MODE` | — | `true` (default) during MuRIL training |
| `MURIL_ENDPOINT` | When stub=false | Fine-tuned model serving URL |
| `EMBEDDING_MODEL` | — | Sentence transformer model name |
| `RETRIEVAL_TOP_K` | — | Knowledge base chunks per query (default: 3) |
| `KNOWLEDGE_BASE_PATH` | — | Path to `knowledge_base_raw.json` |
| `VECTOR_INDEX_PATH` | — | Path to FAISS index directory |
| `LLM_MODEL` | — | Cloud model string (default: `claude-sonnet-4-20250514`) |
| `SESSION_RETENTION_DAYS` | — | Default 30 — user-configurable |

---

## Data Files

| File | Description |
|------|-------------|
| `data/knowledge_base_raw.json` | Cultural stressor taxonomy, guardrails, tone modes, idioms, helplines |
| `data/labelling_schema.json` | NLU annotation schema for annotators |
| `data/seed_dataset.csv` | 120 pre-labelled examples for annotator calibration |
| `data/raw/` | Raw generated batches (keep for reproducibility) |
| `data/interim/` | Partially processed + rejection logs (safe to delete post-pipeline) |
| `data/processed/train.jsonl` | Final training set |
| `data/processed/val.jsonl` | Final validation set |
| `data/processed/test.jsonl` | Final test set |
| `data/processed/metadata.json` | Stats — required by `models/train.py`, regenerate after each pipeline run |
| `data/processed/processed_chunks.json` | Chunked knowledge base for RAG (generated by pipeline) |
| `data/vector_index/` | FAISS index files (generated by index_builder) |

---

## Testing

```bash
# All unit tests (no running services needed)
make test

# Per-service
make test-nlu
make test-cce
make test-retrieval

# With coverage
make test-coverage

# End-to-end (requires running services)
make test-e2e
```

---

## Key Design Decisions

**`gateway.py`** — `asyncio.gather(nlu_coro, risk_prelim_coro)` is the parallel fan-out. Do not convert to sequential. If NLU urgency > 1, a second risk check runs with the full signal — the only time risk runs twice.

**`services/nlu/stub.py`** — Heuristic classification using keyword patterns. Metaphor guard runs first: "kill this exam" never triggers crisis. Swap for `muril.py` endpoint when training gate passes.

**`services/risk_detection/keyword_check.py`** — Three tiers of keyword patterns (tier 3/4/5) with explicit metaphor safeguard. Runs on raw un-normalised text. Crisis keywords must never pass through normalisation before detection.

**`services/cce/rules.py`** — All cultural logic (guardrail selection, CBT technique mapping, idiom detection, framing assembly) lives here as pure functions. No model weights. Auditable and editable without retraining.

**`services/retrieval/`** — FAISS flat inner-product index over knowledge base chunks. Embeddings are L2-normalised (inner product = cosine similarity). Built once, loaded into memory at startup. Retrieval failure never blocks response generation.

**`services/storage/main.py`** — Writes session summaries (emotion arc, stressor type, CBT technique) — never raw conversation text. PII is scrubbed at ingestion (stage 1). 30-day rolling retention runs as a background task.

---

## Crisis Safety Guarantees

- Risk detection is always parallel — never sequential, never gated behind response generation
- Raw text is always used for crisis detection — normalisation runs after, not before
- Metaphor guard prevents false positives — "kill this exam" is urgency 1
- All sessions are audit-logged — risk level + timestamp, no content
- Helplines are always real — verify iCall, Vandrevala Foundation, AASRA numbers before every production release
- Crisis overlay is undismissable — level 5 requires explicit acknowledgement before conversation continues

---

*SADEV — Sadev (सदैव) means Always. A constant companion.*
