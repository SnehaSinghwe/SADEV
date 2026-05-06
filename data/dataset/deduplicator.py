"""
dataset/deduplicator.py
Two-level deduplication for SADEV training data.

Role in pipeline:
  data/interim/ → deduplicator → (still interim, deduplicated)

Level 1 — Exact deduplication:
  - MD5 fingerprint of normalised text (lowercase, no punctuation, collapsed spaces)
  - O(n) hash-set lookup

Level 2 — Semantic deduplication (optional, embedding-based):
  - Computes sentence embeddings using a lightweight model
  - Removes examples with cosine similarity > threshold (default 0.92)
  - Falls back gracefully if sentence-transformers is not installed

Design:
  - Deduplication is within-batch AND cross-batch (pass existing_fingerprints)
  - Both levels are independently toggleable
  - Returns (deduplicated, removed) with removal reasons
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_NORM_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _fingerprint(text: str) -> str:
    """Normalise and hash text for exact dedup."""
    norm = _NORM_RE.sub("", text.lower())
    norm = _SPACE_RE.sub(" ", norm).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


# ── Semantic dedup (optional) ─────────────────────────────────────────────

def _try_import_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        return SentenceTransformer, np
    except ImportError:
        return None, None


@dataclass
class DedupResult:
    kept: list[dict] = field(default_factory=list)
    removed: list[dict] = field(default_factory=list)
    exact_removed: int = 0
    semantic_removed: int = 0
    total_input: int = 0


class Deduplicator:
    """
    Two-level deduplicator.

    Args:
        semantic:          enable embedding-based semantic dedup
        semantic_threshold: cosine similarity above which records are considered duplicates
        embed_model:       sentence-transformers model name
        batch_size:        embedding batch size (affects memory)
    """

    def __init__(
        self,
        semantic: bool = False,
        semantic_threshold: float = 0.92,
        embed_model: str = "paraphrase-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        self.semantic           = semantic
        self.semantic_threshold = semantic_threshold
        self.embed_model        = embed_model
        self.batch_size         = batch_size
        self._embedder          = None
        self._np                = None

        if semantic:
            SentenceTransformer, np = _try_import_embedder()
            if SentenceTransformer is None:
                print(
                    "[deduplicator] WARNING: sentence-transformers not installed. "
                    "Semantic dedup disabled. Install with: pip install sentence-transformers"
                )
                self.semantic = False
            else:
                self._embedder = SentenceTransformer(embed_model)
                self._np = np

    def deduplicate(
        self,
        records: list[dict],
        existing_fingerprints: set[str] | None = None,
    ) -> DedupResult:
        """
        Deduplicate records.

        Args:
            records:               list of record dicts (must have 'text' key)
            existing_fingerprints: fingerprints already in the dataset (for cross-batch dedup)

        Returns:
            DedupResult with .kept and .removed lists
        """
        result = DedupResult(total_input=len(records))
        seen_fps: set[str] = set(existing_fingerprints or set())

        # ── Level 1: Exact dedup ──────────────────────────────────────────
        after_exact: list[dict] = []
        for rec in records:
            fp = _fingerprint(str(rec.get("text", "")))
            if fp in seen_fps:
                result.removed.append({**rec, "_dedup_reason": "exact_duplicate"})
                result.exact_removed += 1
            else:
                seen_fps.add(fp)
                after_exact.append({**rec, "_fingerprint": fp})

        if not self.semantic or not after_exact:
            result.kept = after_exact
            return result

        # ── Level 2: Semantic dedup ───────────────────────────────────────
        result.kept = self._semantic_dedup(after_exact, result)
        return result

    def compute_fingerprints(self, records: list[dict]) -> set[str]:
        """Compute fingerprint set for a list of records (for cross-batch tracking)."""
        return {_fingerprint(str(r.get("text", ""))) for r in records}

    # ── Internal ──────────────────────────────────────────────────────────

    def _semantic_dedup(self, records: list[dict], result: DedupResult) -> list[dict]:
        np = self._np
        texts = [str(r.get("text", "")) for r in records]

        # Embed in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            embs = self._embedder.encode(chunk, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(embs)
        embeddings = np.vstack(all_embeddings)

        kept_indices: list[int] = []
        kept_embeddings: list = []

        for i, (rec, emb) in enumerate(zip(records, embeddings)):
            if not kept_embeddings:
                kept_indices.append(i)
                kept_embeddings.append(emb)
                continue

            # Cosine similarities against all kept embeddings
            kept_matrix = np.vstack(kept_embeddings)
            sims = kept_matrix @ emb    # embeddings are already L2-normalised
            max_sim = float(sims.max())

            if max_sim >= self.semantic_threshold:
                result.removed.append({
                    **rec,
                    "_dedup_reason": "semantic_duplicate",
                    "_sim_score": round(max_sim, 4),
                })
                result.semantic_removed += 1
            else:
                kept_indices.append(i)
                kept_embeddings.append(emb)

        return [records[i] for i in kept_indices]
