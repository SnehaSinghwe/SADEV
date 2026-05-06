"""
scripts/run_training.py
Entry point: train SADEVClassifier and/or evaluate a checkpoint.

Usage:
  # Train from scratch
  python scripts/run_training.py

  # Train with custom config
  python scripts/run_training.py --epochs 15 --lr 3e-5 --encoder google/muril-base-cased

  # Evaluate an existing checkpoint
  python scripts/run_training.py --eval-only --checkpoint models/checkpoints/sadev_v1_best.pt

  # Train then evaluate
  python scripts/run_training.py --eval-after-train

  # Use MiniLM (faster, lighter)
  python scripts/run_training.py --encoder sentence-transformers/paraphrase-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.train import Trainer, TrainConfig
from models.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(
        description="SADEV Model Training & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--data",       default="data/processed/clean_dataset.jsonl",
                        help="Path to processed training data")
    parser.add_argument("--val-split",  type=float, default=0.1)

    # Model
    parser.add_argument("--encoder",    default="google/muril-base-cased",
                        help="HuggingFace encoder model ID")
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--freeze",     type=int,   default=4,
                        help="Number of encoder layers to freeze")

    # Training
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--patience",   type=int,   default=3)
    parser.add_argument("--no-fp16",    action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--run-name",   default="sadev_v1",
                        help="Name prefix for checkpoint files")
    parser.add_argument("--output-dir", default="models/checkpoints")

    # Mode
    parser.add_argument("--eval-only",       action="store_true",
                        help="Skip training, evaluate checkpoint only")
    parser.add_argument("--eval-after-train", action="store_true",
                        help="Run evaluation after training completes")
    parser.add_argument("--checkpoint",      default=None,
                        help="Checkpoint path for --eval-only mode")
    parser.add_argument("--eval-samples",    type=int, default=None,
                        help="Max samples for evaluation (default: all)")
    parser.add_argument("--device",          default="cpu",
                        choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    # ── Validate paths ────────────────────────────────────────────────────
    if not args.eval_only and not Path(args.data).exists():
        print(f"[run_training] ERROR: Training data not found: {args.data}")
        print("  Run scripts/run_generation.py then scripts/run_data_pipeline.py first.")
        sys.exit(1)

    # ── Eval-only mode ────────────────────────────────────────────────────
    if args.eval_only:
        ckpt = args.checkpoint or f"{args.output_dir}/{args.run_name}_best.pt"
        if not Path(ckpt).exists():
            print(f"[run_training] ERROR: Checkpoint not found: {ckpt}")
            sys.exit(1)
        ev = Evaluator(ckpt, device=args.device)
        ev.evaluate(args.data, max_samples=args.eval_samples)
        return

    # ── Training ──────────────────────────────────────────────────────────
    config = TrainConfig(
        data_path=args.data,
        val_split=args.val_split,
        encoder_name=args.encoder,
        dropout=args.dropout,
        freeze_layers=args.freeze,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        fp16=not args.no_fp16,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )

    trainer = Trainer(config=config)
    best_ckpt = trainer.train()

    if args.eval_after_train and best_ckpt and Path(best_ckpt).exists():
        print(f"\n[run_training] Running evaluation on best checkpoint: {best_ckpt}")
        ev = Evaluator(best_ckpt, device=args.device)
        ev.evaluate(args.data, max_samples=args.eval_samples)


if __name__ == "__main__":
    main()
