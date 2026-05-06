# SADEV — Build Guide

**Sadev (सदैव) — Always.**
Culturally adaptive AI mental health support for South Asian youth.

---

## Architecture overview

SADEV is a 7-service microservice pipeline with RAG-augmented response generation:

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
Response delivered to frontend
```

**The one rule above all others:** Risk detection (stage 5) receives every message the moment it enters the gateway — before NLU completes, before CCE, before response generation. It runs via `asyncio.gather`. If you ever find risk detection waiting on any other service, the architecture is broken.

---

## Quick start — full local stack (Ollama)

SADEV can run entirely offline using [Ollama](https://ollama.com) as the LLM backend for both dataset generation and response generation. No Anthropic API key required for local development.

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- MongoDB (local or Atlas) — for audit log
- Firebase project (or emulator) — for session summaries

---

### Step 1 — Install dependencies

```bash
git clone https://github.com/your-org/sadev
cd sadev
pip install -r requirements.txt --break-system-packages
```

Copy and fill in environment variables:
```bash
cp .env.example .env
nano .env   # fill in SERVICE_KEY, Firebase path, Mongo URI
```

---

### Step 2 — Pull and serve the local LLM

```bash
# Pull the model (one-time, ~5 GB)
ollama pull gemma2

# Start the Ollama server (keep this running in a separate terminal)
ollama serve
```

Ollama will serve at `http://localhost:11434` by default. SADEV's generation scripts talk to this endpoint.

> **Model options:** `gemma2` is the recommended default. For lower memory usage try `gemma2:2b`. For higher quality synthetic data try `llama3.1:8b`.

---

### Step 3 — Generate the NLU training dataset

```bash
# Auto-fill all intent classes below their minimum thresholds
# --fill-gaps   : only generate what's missing
# --n-per-intent: examples to generate per class per language
# --n-per-call  : batch size per LLM API call (lower = more reliable)
python scripts/run_generation.py --fill-gaps --n-per-intent 30 --n-per-call 10
```

Generated data is written to `annotation/output/train.csv` (and `dev.csv`, `test.csv`).

**Useful flags:**
```bash
# Single class, specific language
python scripts/run_generation.py --intent crisis_signal --language hi-en-mix --n 50

# Dry run — preview prompts without calling the model
python scripts/run_generation.py --intent seek_validation --n 5 --dry-run

# Generate for a specific urgency level
python scripts/run_generation.py --intent express_academic_pressure --urgency 2 --n 20
```

> **Crisis cap:** Crisis examples (urgency 4–5) are capped at 50 per run and generated in a separate restricted pass with extra safety instructions baked into the prompt.

---

### Step 4 — Process and quality-check the data

```bash
# Run the full data pipeline:
#   1. Quality checks (schema validation, metaphor guard, duplicate removal)
#   2. Stratified train/dev/test splits
#   3. Write processed_chunks.json for the RAG retrieval service
python scripts/run_data_pipeline.py
```

This produces:
- `annotation/output/train_clean.csv` — training set (quality-filtered)
- `annotation/output/audit.csv` — rejected rows with rejection reasons
- `data/processed_chunks.json` — chunked knowledge base for RAG

**Report-only mode (no writes):**
```bash
python annotation/synthetic/quality_check.py \
    --input annotation/output/train.csv \
    --report_only
```

---

### Step 5 — Train the NLU model

```bash
# Fine-tune MuRIL and evaluate on dev set after each epoch
# --eval-after-train runs final evaluation on the held-out test set
python scripts/run_training.py --eval-after-train
```

Training produces a checkpoint at `ml/checkpoints/best_model/`.

**Critical gate:** The model will not be marked deployment-ready unless `crisis_signal recall >= 0.90`. Training output will clearly show `✓ GATE PASSED` or `✗ GATE FAILED`.

**Full training options:**
```bash
python ml/train.py \
    --data_path annotation/output/train_clean.csv \
    --dev_path  annotation/output/dev_clean.csv \
    --output_dir ml/checkpoints \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5
```

**Evaluate a saved checkpoint:**
```bash
python ml/evaluate.py \
    --checkpoint_dir ml/checkpoints/best_model \
    --test_csv annotation/output/test_clean.csv
```

---

### Step 6 — Build the RAG index

```bash
# Embed the cultural knowledge base into a FAISS vector index (one-time, ~2 min)
python -m services.retrieval.index_builder
```

Rebuild whenever `data/knowledge_base_raw.json` changes.

---

### Step 7 — Run the project

```bash
# Start the full pipeline — all 7 services + gateway
python chat.py
```

The gateway is live at **http://localhost:8000**
Swagger UI: **http://localhost:8000/docs**

To run services individually (e.g. for debugging):
```bash
uvicorn services.nlu.main:app --port 8002 --reload
uvicorn services.risk_detection.main:app --port 8005 --reload
# ... etc
```

---

## NLU stub mode

The NLU service ships in **stub mode** (`NLU_STUB_MODE=true`).

The stub uses keyword heuristics to return plausible intent/emotion/urgency output. This allows the full pipeline — including RAG, CCE, response generation, and risk detection — to work end-to-end immediately, before MuRIL training is complete.

**To switch to the real model** (after gate is passed):

1. Set in `.env`:
   ```
   NLU_STUB_MODE=false
   MURIL_ENDPOINT=http://localhost:8090
   ```
2. Start model server:
   ```bash
   uvicorn ml.serve:app --port 8090
   ```
3. Restart services

---

## Frontend quick start

```bash
cd frontend
npm install
npx expo start
```

In `frontend/.env`:
```
EXPO_PUBLIC_GATEWAY_URL=http://localhost:8000
EXPO_PUBLIC_SERVICE_KEY=<same as SERVICE_KEY in backend .env>
```

---

## Testing

```bash
# All unit tests (no running services needed)
make test

# Specific service tests
make test-nlu
make test-cce
make test-retrieval

# With coverage
make test-coverage

# End-to-end (requires running services)
make test-e2e
```

---

## Service map

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

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SERVICE_KEY` | ✓ | Shared secret for inter-service auth |
| `ANTHROPIC_API_KEY` | — | Claude API key (only needed for cloud LLM mode) |
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

## Data files

| File | Description |
|------|-------------|
| `data/knowledge_base_raw.json` | Cultural stressor taxonomy, guardrails, tone modes, idioms, helplines |
| `data/labelling_schema.json` | NLU annotation schema for annotators |
| `data/seed_dataset.csv` | 120 pre-labelled examples for annotator calibration |
| `data/processed_chunks.json` | Chunked knowledge base (generated by `scripts/run_data_pipeline.py`) |
| `data/vector_index/` | FAISS index files (generated by `python -m services.retrieval.index_builder`) |

---

## NLU dataset annotation

Before MuRIL can be trained, 4,200+ examples must be labelled using the schema in `data/labelling_schema.json`. See `SADEV_NLU_Training_Data_Guide.docx` for the full annotation guide.

**Target distribution:**
- 40% Hinglish (hi-en-mix), 30% English, 15% Hindi, 5% Bengali-English, other 10%
- Minimum 300 crisis_signal examples — oversample this class
- Inter-annotator agreement target: Cohen's Kappa > 0.75

**Training targets:**
- Macro F1 > 0.78 across all intent classes
- **crisis_signal recall > 0.90 — mandatory gate before production deployment**
- False negative rate on urgency 4–5 < 0.05

---

## Key design decisions

**`gateway.py`**
`asyncio.gather(nlu_coro, risk_prelim_coro)` is the parallel fan-out. Do not convert this to sequential calls. If NLU finds urgency > 1, a second risk check runs with the full signal — this is the only time risk runs twice.

**`services/nlu/stub.py`**
Heuristic classification using keyword patterns. Metaphor guard runs first — "kill this exam" never triggers crisis. Swap for `muril.py` endpoint when training is complete.

**`services/risk_detection/keyword_check.py`**
Three tiers of keyword patterns (tier 3/4/5) with explicit metaphor safeguard. Runs on raw (un-normalised) text. Crisis keywords must never be processed through normalisation before detection.

**`services/cce/rules.py`**
All cultural logic — guardrail selection, CBT technique mapping, idiom detection, framing guide assembly — lives here as pure functions. No model weights. Auditable and editable without retraining.

**`services/retrieval/`**
FAISS flat inner-product index over knowledge base chunks. Embeddings are L2-normalised, so inner product = cosine similarity. Index is built once and loaded into memory at service startup. Retrieval failure never blocks response generation.

**`services/storage/main.py`**
Writes session summaries (emotion arc, stressor type, CBT technique) — never raw conversation text. PII is scrubbed at ingestion (stage 1), not here. 30-day rolling retention runs as a background task.

---

## Crisis safety guarantees

1. **Risk detection is always parallel** — never sequential, never gated behind response generation
2. **Raw text is always used for crisis detection** — normalisation runs after, not before
3. **Metaphor guard prevents false positives** — "kill this exam" is urgency 1
4. **All sessions are audit-logged** — risk level + timestamp, no content
5. **Helplines are always real** — verify iCall, Vandrevala Foundation, AASRA numbers before every production release
6. **Crisis overlay is undismissable** — level 5 requires explicit acknowledgement before conversation continues

---

*SADEV — Sadev (सदैव) means Always. A constant companion.*
