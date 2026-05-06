"""
models/evaluate.py
Evaluation suite for SADEVClassifier.

Produces:
  - Overall accuracy for intent, category, urgency (MAE + within-1 accuracy)
  - Per-class F1, precision, recall for intent and category
  - Confusion matrix (intent, category)
  - Urgency distribution comparison (predicted vs true)
  - Error analysis: worst-performing intent pairs + examples

Usage:
  python scripts/run_training.py --eval-only
  or:
  from models.evaluate import Evaluator
  ev = Evaluator("models/checkpoints/sadev_v1_best.pt")
  ev.evaluate("data/processed/clean_dataset.jsonl")
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from models.multitask_model import (
    SADEVInference,
    INTENT_LABELS, CATEGORY_LABELS,
    IDX_TO_INTENT, IDX_TO_CATEGORY,
    INTENT_TO_IDX, CATEGORY_TO_IDX,
)


class Evaluator:
    """
    Full evaluation suite.

    Args:
        checkpoint_path: path to .pt checkpoint file
        device:          "cpu" | "cuda" | "mps"
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for evaluation.")
        self.inference = SADEVInference.load(checkpoint_path, device=device)

    def evaluate(
        self,
        data_path: str,
        max_samples: int | None = None,
        print_confusion: bool = True,
        print_per_class: bool = True,
        print_errors: bool = True,
        n_error_examples: int = 5,
    ) -> dict:
        """
        Run full evaluation on a JSONL dataset.
        Returns metrics dict.
        """
        records = self._load(data_path, max_samples)
        print(f"[eval] Evaluating on {len(records)} records...")

        texts = [r["text"] for r in records]
        true_intents   = [r.get("intent",   "casual_chat") for r in records]
        true_categories= [r.get("category", "neutral")     for r in records]
        true_urgencies = [max(1, min(5, int(r.get("urgency", 1)))) for r in records]

        # Batch inference
        preds = self.inference.predict_batch(texts)
        pred_intents    = [p.intent   for p in preds]
        pred_categories = [p.category for p in preds]
        pred_urgencies  = [p.urgency  for p in preds]

        # ── Intent metrics ────────────────────────────────────────────────
        intent_acc = sum(p == t for p, t in zip(pred_intents, true_intents)) / len(records)
        intent_f1  = self._macro_f1(true_intents, pred_intents, INTENT_LABELS)

        # ── Category metrics ──────────────────────────────────────────────
        cat_acc = sum(p == t for p, t in zip(pred_categories, true_categories)) / len(records)
        cat_f1  = self._macro_f1(true_categories, pred_categories, CATEGORY_LABELS)

        # ── Urgency metrics ───────────────────────────────────────────────
        abs_errors   = [abs(p - t) for p, t in zip(pred_urgencies, true_urgencies)]
        urgency_mae  = sum(abs_errors) / len(abs_errors)
        urgency_w1   = sum(1 for e in abs_errors if e <= 1) / len(abs_errors)

        metrics = {
            "n_samples":    len(records),
            "intent_acc":   round(intent_acc, 4),
            "intent_f1":    round(intent_f1,  4),
            "category_acc": round(cat_acc,    4),
            "category_f1":  round(cat_f1,     4),
            "urgency_mae":  round(urgency_mae, 3),
            "urgency_w1":   round(urgency_w1,  4),  # within-1 accuracy
        }

        # ── Print report ──────────────────────────────────────────────────
        self._print_summary(metrics)

        if print_per_class:
            print("\n── Intent F1 (per class) ────────────────────────────────")
            self._print_per_class_f1(true_intents, pred_intents, INTENT_LABELS)
            print("\n── Category F1 (per class) ──────────────────────────────")
            self._print_per_class_f1(true_categories, pred_categories, CATEGORY_LABELS)

        if print_confusion:
            print("\n── Category Confusion Matrix ─────────────────────────────")
            self._print_confusion(true_categories, pred_categories, CATEGORY_LABELS)

        if print_errors:
            print(f"\n── Error Analysis (worst {n_error_examples} intent mismatches) ───")
            self._print_errors(texts, true_intents, pred_intents, n_error_examples)

        urgency_dist = self._urgency_distribution(true_urgencies, pred_urgencies)
        print("\n── Urgency Distribution ─────────────────────────────────")
        for level in range(1, 6):
            t = urgency_dist["true"].get(level, 0)
            p = urgency_dist["pred"].get(level, 0)
            print(f"  Level {level}: true={t:>4}  pred={p:>4}")

        return metrics

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _load(path: str, max_samples: int | None) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if max_samples:
            import random
            random.shuffle(records)
            records = records[:max_samples]
        return records

    @staticmethod
    def _macro_f1(true: list[str], pred: list[str], labels: list[str]) -> float:
        f1s = []
        for label in labels:
            tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        return sum(f1s) / len(f1s) if f1s else 0.0

    @staticmethod
    def _print_summary(metrics: dict):
        print("\n════ SADEV Classifier Evaluation ════════════════════════")
        print(f"  Samples:      {metrics['n_samples']}")
        print(f"  Intent  Acc:  {metrics['intent_acc']:.4f}   Macro-F1: {metrics['intent_f1']:.4f}")
        print(f"  Category Acc: {metrics['category_acc']:.4f}   Macro-F1: {metrics['category_f1']:.4f}")
        print(f"  Urgency MAE:  {metrics['urgency_mae']:.3f}    Within-1: {metrics['urgency_w1']:.4f}")
        print("═════════════════════════════════════════════════════════")

    @staticmethod
    def _print_per_class_f1(true, pred, labels):
        rows = []
        for label in labels:
            tp = sum(1 for t, p in zip(true, pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(true, pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(true, pred) if t == label and p != label)
            support = sum(1 for t in true if t == label)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            rows.append((label, prec, rec, f1, support))
        rows.sort(key=lambda x: x[3])
        print(f"  {'Label':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
        print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*6} {'─'*5}")
        for label, prec, rec, f1, n in rows:
            print(f"  {label:<35} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {n:>5}")

    @staticmethod
    def _print_confusion(true, pred, labels):
        """Compact confusion matrix — only rows/cols with errors."""
        # Build matrix
        mat = defaultdict(lambda: defaultdict(int))
        for t, p in zip(true, pred):
            mat[t][p] += 1
        # Print header
        short = [l[:10] for l in labels]
        print("  " + " " * 14 + " ".join(f"{s:>10}" for s in short))
        for t_label, t_short in zip(labels, short):
            row = [mat[t_label].get(p_label, 0) for p_label in labels]
            if sum(row) == 0:
                continue
            print(f"  {t_short:<14}" + " ".join(f"{v:>10}" for v in row))

    @staticmethod
    def _print_errors(texts, true, pred, n: int):
        """Show n examples where intent was misclassified."""
        errors = [
            (text, t, p)
            for text, t, p in zip(texts, true, pred)
            if t != p
        ]
        # Sort by most common error pair
        pair_count = Counter((t, p) for _, t, p in errors)
        shown = set()
        count = 0
        for (t, p), _ in pair_count.most_common():
            if count >= n:
                break
            example = next((text for text, ti, pi in errors if ti == t and pi == p), None)
            if example and (t, p) not in shown:
                shown.add((t, p))
                print(f"  TRUE={t}  PRED={p}")
                print(f"  Text: {example[:100]}")
                print()
                count += 1

    @staticmethod
    def _urgency_distribution(true, pred) -> dict:
        from collections import Counter
        return {
            "true": dict(Counter(true)),
            "pred": dict(Counter(pred)),
        }
