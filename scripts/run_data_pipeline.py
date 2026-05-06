"""
scripts/run_data_pipeline.py
Entry point: run the full data processing pipeline (clean → dedup → validate → save).

Usage:
  # Full pipeline (raw/ → processed/)
  python scripts/run_data_pipeline.py

  # With semantic deduplication enabled (needs sentence-transformers)
  python scripts/run_data_pipeline.py --semantic-dedup

  # Only show distribution of existing processed dataset
  python scripts/run_data_pipeline.py --report-only

  # Strict schema: drop (not fix) invalid records
  python scripts/run_data_pipeline.py --strict

  # Custom paths
  python scripts/run_data_pipeline.py --raw-dir data/raw --processed-dir data/processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset.pipeline import DataPipeline


def main():
    parser = argparse.ArgumentParser(
        description="SADEV Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-dir",           default="data/raw")
    parser.add_argument("--interim-dir",        default="data/interim")
    parser.add_argument("--processed-dir",      default="data/processed")
    parser.add_argument("--semantic-dedup",     action="store_true",
                        help="Enable embedding-based semantic deduplication")
    parser.add_argument("--semantic-threshold", type=float, default=0.92,
                        help="Cosine similarity threshold for semantic dedup (default 0.92)")
    parser.add_argument("--strict",            action="store_true",
                        help="Strict schema: drop invalid records instead of fixing them")
    parser.add_argument("--report-only",       action="store_true",
                        help="Only print distribution report, no processing")
    parser.add_argument("--quiet",             action="store_true",
                        help="Suppress verbose stage output")
    args = parser.parse_args()

    pipeline = DataPipeline(
        raw_dir=args.raw_dir,
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        semantic_dedup=args.semantic_dedup,
        semantic_threshold=args.semantic_threshold,
        strict_schema=args.strict,
    )

    if args.report_only:
        pipeline.distribution_report()
        return

    stats = pipeline.run(verbose=not args.quiet)

    if stats.final_written == 0:
        print("[run_data_pipeline] No data was written. Check data/raw/ for raw batches.")
        sys.exit(1)

    # Also print distribution
    print()
    pipeline.distribution_report()


if __name__ == "__main__":
    main()
