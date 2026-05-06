"""
dataset/pipeline.py
Main data processing orchestrator for SADEV.

Role: ties generator → cleaner → deduplicator → validator together.

Flow:
  data/raw/*.jsonl
    → TextCleaner            → data/interim/cleaned.jsonl
    → Deduplicator           → data/interim/deduped.jsonl
    → Validator              → data/processed/clean_dataset.jsonl
                               data/processed/corrections.jsonl  (label-corrected records)
                               data/processed/rejected.jsonl     (dropped records)

Design:
  - Each stage writes a checkpoint to data/interim/ so failures are resumable
  - Stats are printed at every stage
  - All file I/O is append-safe (safe to re-run on new raw batches)
  - The pipeline can be run in full or stage-by-stage
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from data.dataset.cleaner import TextCleaner
from data.dataset.deduplicator import Deduplicator
from data.dataset.validator import Validator


# ── Paths ─────────────────────────────────────────────────────────────────

DEFAULT_RAW_DIR      = "data/raw"
DEFAULT_INTERIM_DIR  = "data/interim"
DEFAULT_PROCESSED_DIR= "data/processed"


# ── Stats ─────────────────────────────────────────────────────────────────

@dataclass
class PipelineStats:
    raw_loaded:        int = 0
    after_clean:       int = 0
    clean_dropped:     int = 0
    after_dedup:       int = 0
    dedup_exact:       int = 0
    dedup_semantic:    int = 0
    after_validate:    int = 0
    validate_corrected:int = 0
    validate_dropped:  int = 0
    final_written:     int = 0

    def report(self) -> str:
        lines = [
            "──── Data Pipeline Stats ────────────────",
            f"  Raw loaded:         {self.raw_loaded:>6}",
            f"  After cleaning:     {self.after_clean:>6}  (dropped {self.clean_dropped})",
            f"  After dedup:        {self.after_dedup:>6}  (exact {self.dedup_exact} | semantic {self.dedup_semantic})",
            f"  After validation:   {self.after_validate:>6}  (corrected {self.validate_corrected} | dropped {self.validate_dropped})",
            f"  Final written:      {self.final_written:>6}",
            "─────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ── JSONL I/O ─────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [pipeline] skip line {i} in {path}: {e}")
    return records


def _save_jsonl(records: list[dict], path: str, append: bool = False) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    written = 0
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def _load_raw_dir(raw_dir: str) -> list[dict]:
    """Load all *.jsonl files from raw_dir."""
    records = []
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    files = sorted(raw_path.glob("*.jsonl"))
    for fpath in files:
        batch = _load_jsonl(str(fpath))
        print(f"  [pipeline] loaded {len(batch):>4} records from {fpath.name}")
        records.extend(batch)
    return records


def _strip_meta_keys(records: list[dict]) -> list[dict]:
    """Strip internal _meta, _original_text etc. keys before final output."""
    strip_prefix = ("_",)
    return [
        {k: v for k, v in rec.items() if not k.startswith(strip_prefix)}
        for rec in records
    ]


# ── Pipeline ──────────────────────────────────────────────────────────────

class DataPipeline:
    """
    Full data processing pipeline: raw JSONL → clean training-ready JSONL.

    Args:
        raw_dir:             directory with raw generated batches
        interim_dir:         directory for checkpoint files
        processed_dir:       directory for final outputs
        semantic_dedup:      enable embedding-based semantic deduplication
        semantic_threshold:  cosine similarity threshold for semantic dedup
        strict_schema:       drop (vs. fix) records with schema violations
    """

    def __init__(
        self,
        raw_dir:            str   = DEFAULT_RAW_DIR,
        interim_dir:        str   = DEFAULT_INTERIM_DIR,
        processed_dir:      str   = DEFAULT_PROCESSED_DIR,
        semantic_dedup:     bool  = False,
        semantic_threshold: float = 0.92,
        strict_schema:      bool  = False,
    ):
        self.raw_dir       = raw_dir
        self.interim_dir   = interim_dir
        self.processed_dir = processed_dir

        self.cleaner   = TextCleaner()
        self.deduper   = Deduplicator(semantic=semantic_dedup, semantic_threshold=semantic_threshold)
        self.validator = Validator(strict_schema=strict_schema)

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> PipelineStats:
        """
        Run the full pipeline: raw → interim → processed.
        Safe to run repeatedly on the same raw_dir (appends to processed/).
        """
        stats = PipelineStats()

        # ── Stage 1: Load raw ─────────────────────────────────────────────
        if verbose:
            print("[pipeline] Stage 1: Loading raw data...")
        records = _load_raw_dir(self.raw_dir)
        stats.raw_loaded = len(records)
        if not records:
            print("[pipeline] No raw data found. Run scripts/run_generation.py first.")
            return stats
        if verbose:
            print(f"[pipeline] Loaded {stats.raw_loaded} raw records.")

        # ── Stage 2: Clean ────────────────────────────────────────────────
        if verbose:
            print("[pipeline] Stage 2: Cleaning...")
        accepted, rejected = self.cleaner.clean_batch(records)
        stats.after_clean   = len(accepted)
        stats.clean_dropped = len(rejected)
        _save_jsonl(accepted, f"{self.interim_dir}/cleaned.jsonl")
        _save_jsonl(rejected, f"{self.interim_dir}/clean_rejected.jsonl")
        if verbose:
            print(f"[pipeline] Clean: {stats.after_clean} kept, {stats.clean_dropped} dropped.")

        # ── Stage 3: Deduplicate ──────────────────────────────────────────
        if verbose:
            print("[pipeline] Stage 3: Deduplicating...")

        # Load existing processed fingerprints for cross-batch dedup
        existing_fps = self._load_existing_fingerprints()
        dedup_result = self.deduper.deduplicate(accepted, existing_fingerprints=existing_fps)
        stats.after_dedup    = len(dedup_result.kept)
        stats.dedup_exact    = dedup_result.exact_removed
        stats.dedup_semantic = dedup_result.semantic_removed
        _save_jsonl(dedup_result.kept,    f"{self.interim_dir}/deduped.jsonl")
        _save_jsonl(dedup_result.removed, f"{self.interim_dir}/dedup_removed.jsonl")
        if verbose:
            print(
                f"[pipeline] Dedup: {stats.after_dedup} kept, "
                f"{stats.dedup_exact} exact dupes, "
                f"{stats.dedup_semantic} semantic dupes removed."
            )

        # ── Stage 4: Validate & correct labels ───────────────────────────
        if verbose:
            print("[pipeline] Stage 4: Validating labels...")
        clean, corrected, dropped = self.validator.validate_batch(dedup_result.kept)
        stats.after_validate     = len(clean) + len(corrected)
        stats.validate_corrected = len(corrected)
        stats.validate_dropped   = len(dropped)

        # Merge clean + corrected for output
        final = clean + corrected
        _save_jsonl(dropped,   f"{self.processed_dir}/rejected.jsonl", append=True)
        _save_jsonl(corrected, f"{self.processed_dir}/corrections.jsonl", append=True)

        # Strip internal keys before writing final output
        final_clean = _strip_meta_keys(final)
        written = _save_jsonl(final_clean, f"{self.processed_dir}/clean_dataset.jsonl", append=True)
        stats.final_written = written

        if verbose:
            print(
                f"[pipeline] Validate: {stats.after_validate} valid "
                f"({stats.validate_corrected} corrected, {stats.validate_dropped} dropped)."
            )
            print(f"[pipeline] Final: {stats.final_written} records written to processed/.")
            print(stats.report())

        return stats

    def run_from_records(
        self,
        records: list[dict],
        verbose: bool = True,
    ) -> tuple[list[dict], PipelineStats]:
        """
        Run pipeline on an in-memory list of records (e.g. from DatasetGenerator).
        Returns (final_records, stats) without touching disk (interim only).
        Useful for testing or streaming generation.
        """
        stats = PipelineStats(raw_loaded=len(records))

        # Clean
        accepted, rejected_c = self.cleaner.clean_batch(records)
        stats.after_clean   = len(accepted)
        stats.clean_dropped = len(rejected_c)

        # Dedup
        dedup_result = self.deduper.deduplicate(accepted)
        stats.after_dedup    = len(dedup_result.kept)
        stats.dedup_exact    = dedup_result.exact_removed
        stats.dedup_semantic = dedup_result.semantic_removed

        # Validate
        clean, corrected, dropped_v = self.validator.validate_batch(dedup_result.kept)
        stats.after_validate     = len(clean) + len(corrected)
        stats.validate_corrected = len(corrected)
        stats.validate_dropped   = len(dropped_v)

        final = _strip_meta_keys(clean + corrected)
        stats.final_written = len(final)

        if verbose:
            print(stats.report())

        return final, stats

    # ── Category distribution ─────────────────────────────────────────────

    def distribution_report(self, path: str | None = None) -> dict[str, int]:
        """Print and return category distribution of the final dataset."""
        dataset_path = path or f"{self.processed_dir}/clean_dataset.jsonl"
        if not Path(dataset_path).exists():
            print("[pipeline] No processed dataset found yet.")
            return {}
        records = _load_jsonl(dataset_path)
        from collections import Counter
        cats = Counter(r.get("category", "unknown") for r in records)
        intents = Counter(r.get("intent", "unknown") for r in records)
        total = len(records)
        print(f"\nDataset: {total} records")
        print("By category:")
        for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {cat:<25} {n:>5}  ({n/total*100:.1f}%)")
        print("By intent (top 10):")
        for intent, n in intents.most_common(10):
            print(f"  {intent:<35} {n:>5}  ({n/total*100:.1f}%)")
        return dict(cats)

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_existing_fingerprints(self) -> set[str]:
        """Load fingerprints of records already in processed/ for cross-batch dedup."""
        dataset_path = f"{self.processed_dir}/clean_dataset.jsonl"
        if not Path(dataset_path).exists():
            return set()
        records = _load_jsonl(dataset_path)
        return self.deduper.compute_fingerprints(records)
