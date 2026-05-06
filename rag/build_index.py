"""
rag/build_index.py
Build the FAISS vector index from knowledge_base_raw.json.

Run once before using RAGRetriever:
    python -m rag.build_index

Or with custom paths:
    python -m rag.build_index --kb data/knowledge_base_raw.json --out data/vector_index
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np


def chunk_stressor(stressor_id: str, stressor: dict) -> list[dict]:
    chunks = []
    label = stressor.get("label", stressor_id)
    desc  = stressor.get("description", "")
    chunks.append({
        "text": f"{label}: {desc}",
        "source": "stressor_taxonomy",
        "stressor_type": stressor_id,
        "chunk_type": "description",
    })
    for lang, phrases in stressor.get("signal_phrases", {}).items():
        if phrases:
            combined = "; ".join(phrases[:12])
            chunks.append({
                "text": f"Signal phrases ({lang}) for {stressor_id}: {combined}",
                "source": "signal_phrases",
                "stressor_type": stressor_id,
                "chunk_type": "signal_phrases",
            })
    emotions = stressor.get("linked_emotions", [])
    if emotions:
        chunks.append({
            "text": f"Common emotions in {stressor_id}: {', '.join(emotions)}",
            "source": "emotion_map",
            "stressor_type": stressor_id,
            "chunk_type": "emotions",
        })
    guardrails = stressor.get("guardrails", [])
    if guardrails:
        chunks.append({
            "text": f"Guardrails for {stressor_id}: {'; '.join(guardrails)}",
            "source": "guardrails",
            "stressor_type": stressor_id,
            "chunk_type": "guardrails",
        })
    return chunks


def chunk_guardrail_registry(registry: dict) -> list[dict]:
    return [
        {"text": f"Guardrail '{gid}': {desc}", "source": "guardrail_registry",
         "stressor_type": None, "chunk_type": "guardrail", "guardrail_id": gid}
        for gid, desc in registry.items()
    ]


def chunk_idioms(idioms: dict) -> list[dict]:
    chunks = []
    for idiom_id, idiom in idioms.items():
        text = (
            f"South Asian idiom '{idiom_id}' ({idiom.get('translation', '')}): "
            f"{idiom.get('meaning', '')} "
            f"Linked to: {', '.join(idiom.get('linked_stressors', []))}."
        )
        chunks.append({"text": text, "source": "cultural_idioms",
                        "stressor_type": None, "chunk_type": "idiom"})
    return chunks


def chunk_tone_modes(tone_modes: dict) -> list[dict]:
    chunks = []
    for tone_id, tone in tone_modes.items():
        desc   = tone.get("description", "")
        when   = ", ".join(tone.get("when_to_use", []))
        prefer = "; ".join(tone.get("linguistic_markers", {}).get("prefer", []))
        avoid  = "; ".join(tone.get("linguistic_markers", {}).get("avoid", []))
        text = f"Tone {tone_id}: {desc}. When: {when}. Prefer: {prefer}. Avoid: {avoid}."
        chunks.append({"text": text, "source": "tone_modes",
                        "stressor_type": None, "chunk_type": "tone_mode"})
    return chunks


def build_index(
    kb_path: str = "data/knowledge_base_raw.json",
    out_path: str = "data/vector_index",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunks_path: str = "data/processed_chunks.json",
) -> int:
    print(f"[build_index] Loading KB from {kb_path}")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    all_chunks: list[dict] = []
    for sid, stressor in kb.get("stressor_taxonomy", {}).items():
        all_chunks.extend(chunk_stressor(sid, stressor))
    all_chunks.extend(chunk_tone_modes(kb.get("tone_modes", {})))
    all_chunks.extend(chunk_guardrail_registry(kb.get("guardrail_registry", {})))
    all_chunks.extend(chunk_idioms(kb.get("cultural_idioms", {})))

    print(f"[build_index] Generated {len(all_chunks)} chunks")

    os.makedirs(os.path.dirname(chunks_path) or ".", exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("[build_index] Embedding (this may take ~2 minutes)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=True)

    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    os.makedirs(out_path, exist_ok=True)
    faiss.write_index(index, os.path.join(out_path, "index.faiss"))
    with open(os.path.join(out_path, "metadata.pkl"), "wb") as f:
        pickle.dump({"metadata": all_chunks, "dim": dim}, f)

    print(f"[build_index] Saved {index.ntotal} vectors to {out_path}")
    return index.ntotal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb",     default="data/knowledge_base_raw.json")
    parser.add_argument("--out",    default="data/vector_index")
    parser.add_argument("--chunks", default="data/processed_chunks.json")
    parser.add_argument("--model",  default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()
    n = build_index(args.kb, args.out, args.model, args.chunks)
    print(f"[build_index] Done. {n} vectors indexed.")
