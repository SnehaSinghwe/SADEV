"""
scripts/run_generation.py
Entry point: generate training data batches using the DatasetGenerator.

Usage examples:
  # Generate 20 examples for a single intent (Ollama, default)
  python scripts/run_generation.py --intent venting_distress --n 20

  # Generate across all intents (fills gaps toward target counts)
  python scripts/run_generation.py --fill-gaps --n-per-intent 30

  # Use Anthropic API instead of Ollama
  python scripts/run_generation.py --backend anthropic --intent academic_pressure --n 50

  # Specify output batch file
  python scripts/run_generation.py --intent loneliness --n 40 --batch-name batch_3.jsonl

Environment:
  ANTHROPIC_API_KEY  — required only for --backend anthropic
  OLLAMA_URL         — default http://localhost:11434
  OLLAMA_MODEL       — default gemma2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset.generator import DatasetGenerator, VALID_INTENTS, INTENT_CONFIG

# Target counts per intent for --fill-gaps mode
TARGET_COUNTS: dict[str, int] = {
    "venting_distress":    500,
    "anxiety_panic":       500,
    "sadness_low_mood":    500,
    "overwhelm_burnout":   400,
    "loneliness":          400,
    "career_job_stress":   500,
    "academic_pressure":   500,
    "financial_stress":    400,
    "family_conflict":     400,
    "relationship_issue":  400,
    "marriage_pressure":   400,
    "social_reputation":   300,
    "conflict_exposure":   300,
    "fear_of_consequence": 300,
    "advice_seeking":      400,
    "decision_conflict":   400,
    "planning_next_steps": 300,
    "sleep_issue":         300,
    "focus_issue":         300,
    "productivity_block":  300,
    "hopelessness":        400,
    "emotional_exhaustion":400,
    "self_harm_intent":    200,   # capped intentionally
    "suicidal_thoughts":   200,   # capped intentionally
    "greeting":            200,
    "casual_chat":         200,
}


def _count_existing(raw_dir: str) -> dict[str, int]:
    """Count existing records per intent from raw/."""
    import json
    counts = {i: 0 for i in VALID_INTENTS}
    for fpath in Path(raw_dir).glob("*.jsonl"):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    intent = rec.get("intent", "")
                    if intent in counts:
                        counts[intent] += 1
                except json.JSONDecodeError:
                    continue
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="SADEV Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--intent",      choices=sorted(VALID_INTENTS),
                        help="Intent to generate examples for")
    parser.add_argument("--n",           type=int, default=20,
                        help="Number of examples to generate (per intent)")
    parser.add_argument("--language",    default=None,
                        choices=["hi-en-mix", "en", "hi", "bn", "bn-en-mix"],
                        help="Override default language for intent")
    parser.add_argument("--fill-gaps",   action="store_true",
                        help="Generate for all intents below target count")
    parser.add_argument("--backend",     default="ollama", choices=["ollama", "anthropic"],
                        help="LLM backend to use")
    parser.add_argument("--model",       default=None,
                        help="Model name (default: gemma2 for ollama, "
                             "claude-sonnet-4-20250514 for anthropic)")
    parser.add_argument("--ollama-url",  default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--api-key",     default=os.getenv("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--raw-dir",     default="data/raw",
                        help="Output directory for raw JSONL batches")
    parser.add_argument("--batch-name",  default=None,
                        help="Output batch filename (auto-generated if not set)")
    parser.add_argument("--n-per-intent",type=int, default=20,
                        help="Examples per intent in --fill-gaps mode")
    parser.add_argument("--n-per-call",  type=int, default=10,
                        help="Max examples per LLM call — large n is auto-chunked (default: 10)")
    parser.add_argument("--timeout",     type=int, default=300,
                        help="Per-chunk read timeout in seconds for Ollama (default: 300)")
    parser.add_argument("--delay",       type=float, default=1.5,
                        help="Seconds between API calls (default: 1.5)")
    args = parser.parse_args()

    # Resolve model defaults
    if args.model is None:
        args.model = "gemma2" if args.backend == "ollama" else "claude-sonnet-4-20250514"

    gen = DatasetGenerator(
        backend=args.backend,
        model=args.model,
        ollama_url=args.ollama_url,
        api_key=args.api_key,
        timeout=args.timeout,
        n_per_call=args.n_per_call,
    )

    Path(args.raw_dir).mkdir(parents=True, exist_ok=True)

    # ── Mode: single intent ───────────────────────────────────────────────
    if args.intent and not args.fill_gaps:
        print(f"[run_generation] Generating {args.n}x '{args.intent}'...")
        batch = gen.generate_batch(args.intent, n=args.n, language=args.language)
        if not batch:
            print("[run_generation] Generation failed — no records returned.")
            sys.exit(1)
        batch_name = args.batch_name or f"batch_{args.intent}.jsonl"
        out_path   = f"{args.raw_dir}/{batch_name}"
        written    = gen.save_batch(batch, out_path)
        print(f"[run_generation] Wrote {written} records → {out_path}")
        # Preview
        for rec in batch[:3]:
            print(f"  [{rec.intent}|{rec.language}] {rec.text[:80]}")

    # ── Mode: fill gaps ───────────────────────────────────────────────────
    elif args.fill_gaps:
        existing = _count_existing(args.raw_dir)
        gaps = [
            (intent, TARGET_COUNTS[intent] - existing.get(intent, 0))
            for intent in VALID_INTENTS
            if existing.get(intent, 0) < TARGET_COUNTS.get(intent, 0)
        ]
        gaps.sort(key=lambda x: -x[1])
        print(f"[run_generation] Filling gaps for {len(gaps)} intents:")
        for intent, needed in gaps:
            print(f"  {intent:<35} need {needed:>4} more")

        total_written = 0
        import time
        for intent, needed in gaps:
            n = min(needed, args.n_per_intent)
            print(f"\n[run_generation] Generating {n}x '{intent}'...")
            batch = gen.generate_batch(intent, n=n, language=args.language)
            if batch:
                out_path = f"{args.raw_dir}/batch_{intent}.jsonl"
                written  = gen.save_batch(batch, out_path)
                total_written += written
                print(f"  → Wrote {written} records")
            else:
                print(f"  → FAILED: no records returned")
            time.sleep(args.delay)

        print(f"\n[run_generation] Done. Total written: {total_written}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()