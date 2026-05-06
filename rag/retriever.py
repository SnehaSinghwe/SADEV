"""
rag/retriever.py
Local RAG retriever. Loads the FAISS index built by index_builder.py
and serves top-k chunks for injection into the system prompt.

This is a thin wrapper around the existing vector_store.py and embedder.py
logic, adapted to work without the microservice HTTP layer.

Usage:
    retriever = RAGRetriever(index_path="data/vector_index", kb_path="data/knowledge_base_raw.json")
    chunks = retriever.retrieve("academic pressure JEE exam", stressor_filter="academic_pressure")

The index must be built first:
    python -m rag.build_index
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Optional

import numpy as np


class RAGRetriever:
    """
    Loads a pre-built FAISS index and metadata.
    Falls back gracefully to an empty list if index is not built yet.
    """

    def __init__(
        self,
        index_path: str = "data/vector_index",
        kb_path: str = "data/knowledge_base_raw.json",
        top_k: int = 3,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.index_path = index_path
        self.kb_path = kb_path
        self.top_k = top_k
        self.embedding_model = embedding_model
        self._index = None
        self._metadata: list[dict] = []
        self._model = None
        self._load()

    def _load(self) -> None:
        index_file = os.path.join(self.index_path, "index.faiss")
        meta_file  = os.path.join(self.index_path, "metadata.pkl")
        if not (os.path.exists(index_file) and os.path.exists(meta_file)):
            print(f"[RAGRetriever] Index not found at {self.index_path}. "
                  f"Run `python -m rag.build_index` first.")
            return
        try:
            import faiss
            self._index = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                data = pickle.load(f)
            self._metadata = data.get("metadata", [])
            print(f"[RAGRetriever] Loaded {len(self._metadata)} chunks from {self.index_path}")
        except ImportError:
            print("[RAGRetriever] faiss-cpu not installed. pip install faiss-cpu --break-system-packages")
        except Exception as e:
            print(f"[RAGRetriever] Load failed: {e}")

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError("sentence-transformers required: pip install sentence-transformers")
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0]

    def retrieve(
        self,
        query: str,
        stressor_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Return top-k relevant chunks for a query string.
        Optional stressor_filter narrows to chunks for that stressor type.
        Returns empty list if index is not loaded.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        k = top_k or self.top_k
        try:
            import faiss
            q = self.embed(query).reshape(1, -1).astype("float32")
            fetch_k = min(k * 3, self._index.ntotal)
            scores, indices = self._index.search(q, fetch_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                chunk = {**self._metadata[idx], "score": float(score)}
                if stressor_filter and chunk.get("stressor_type") not in (stressor_filter, None):
                    continue
                results.append(chunk)

            return results[:k]
        except Exception as e:
            print(f"[RAGRetriever] Retrieval error: {e}")
            return []

    def format_context(self, chunks: list[dict]) -> str:
        """Format chunks as a readable block for the system prompt."""
        if not chunks:
            return ""
        lines = ["## Retrieved cultural context (do NOT quote directly):", ""]
        for i, chunk in enumerate(chunks, 1):
            src  = chunk.get("source", "kb")
            text = chunk.get("text", "")
            lines.append(f"{i}. [{src}] {text}")
        return "\n".join(lines)

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
