"""
models/train.py
Training loop for SADEVClassifier.

Features:
  - Multi-task loss: weighted sum of intent CE + category CE + urgency ordinal BCE
  - Gradient clipping (prevents exploding gradients with transformer fine-tuning)
  - Learning rate warmup + linear decay (standard for transformer fine-tuning)
  - Early stopping on validation loss
  - Checkpoint saving: best model + latest epoch
  - Class-weight balancing for imbalanced intents (crisis is rare)
  - Mixed precision training (fp16) if GPU available

Usage:
  python scripts/run_training.py
  or:
  from models.train import Trainer
  trainer = Trainer(data_path="data/processed/clean_dataset.jsonl")
  trainer.train()
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from models.multitask_model import (
    SADEVClassifier,
    INTENT_TO_IDX, CATEGORY_TO_IDX,
    urgency_to_ordinal_targets,
    INTENT_LABELS, CATEGORY_LABELS,
)

# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    data_path:          str   = "data/processed/clean_dataset.jsonl"
    val_split:          float = 0.1           # fraction held out for validation
    max_length:         int   = 128           # tokeniser max tokens

    # Model
    encoder_name:       str   = "google/muril-base-cased"
    dropout:            float = 0.3
    freeze_layers:      int   = 4             # freeze bottom N encoder layers

    # Training
    epochs:             int   = 10
    batch_size:         int   = 32
    lr:                 float = 2e-5
    weight_decay:       float = 0.01
    warmup_ratio:       float = 0.1           # fraction of total steps for warmup
    grad_clip:          float = 1.0
    fp16:               bool  = True          # use mixed precision if GPU available

    # Loss weights
    intent_loss_weight:   float = 1.0
    category_loss_weight: float = 0.7
    urgency_loss_weight:  float = 0.5

    # Early stopping
    patience:           int   = 3            # stop if val loss doesn't improve

    # Output
    output_dir:         str   = "models/checkpoints"
    run_name:           str   = "sadev_v1"

    # Reproducibility
    seed:               int   = 42


# ── Dataset ───────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class SADEVDataset(Dataset):
        def __init__(self, records: list[dict], tokenizer, max_length: int = 128):
            self.records    = records
            self.tokenizer  = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> dict:
            rec = self.records[idx]
            text = str(rec.get("text", ""))
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            intent_idx   = INTENT_TO_IDX.get(rec.get("intent",   "casual_chat"), 25)
            category_idx = CATEGORY_TO_IDX.get(rec.get("category", "neutral"),   7)
            urgency      = max(1, min(5, int(rec.get("urgency", 1))))

            return {
                "input_ids":      encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "intent_label":   torch.tensor(intent_idx,   dtype=torch.long),
                "category_label": torch.tensor(category_idx, dtype=torch.long),
                "urgency_target": urgency_to_ordinal_targets(urgency),
            }


# ── Trainer ───────────────────────────────────────────────────────────────

class Trainer:
    """
    Multi-task trainer for SADEVClassifier.

    Args:
        data_path: path to clean_dataset.jsonl
        config:    TrainConfig instance (uses defaults if None)
    """

    def __init__(
        self,
        data_path: str | None = None,
        config: TrainConfig | None = None,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers are required for training.\n"
                "Install: pip install torch transformers"
            )
        self.config = config or TrainConfig()
        if data_path:
            self.config.data_path = data_path

        self._set_seed(self.config.seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        print(f"[trainer] Device: {self.device}")

    def train(self) -> str:
        """
        Run training. Returns path to the best checkpoint.
        """
        cfg = self.config

        # ── Load and split data ───────────────────────────────────────────
        records = self._load_records(cfg.data_path)
        random.shuffle(records)
        n_val   = max(1, int(len(records) * cfg.val_split))
        val_recs, train_recs = records[:n_val], records[n_val:]
        print(f"[trainer] Train: {len(train_recs)} | Val: {len(val_recs)}")

        # ── Tokenizer + datasets ──────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
        train_ds  = SADEVDataset(train_recs, tokenizer, cfg.max_length)
        val_ds    = SADEVDataset(val_recs,   tokenizer, cfg.max_length)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=self.device.type == "cuda")
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=0)

        # ── Model ─────────────────────────────────────────────────────────
        model = SADEVClassifier(
            encoder_name=cfg.encoder_name,
            dropout=cfg.dropout,
            freeze_encoder_layers=cfg.freeze_layers,
        ).to(self.device)

        # ── Class weights for intent (crisis is rare → upweight) ──────────
        intent_weights = self._compute_intent_weights(train_recs)
        intent_weights = intent_weights.to(self.device)

        # ── Loss functions ────────────────────────────────────────────────
        intent_loss_fn   = nn.CrossEntropyLoss(weight=intent_weights)
        category_loss_fn = nn.CrossEntropyLoss()
        urgency_loss_fn  = nn.BCEWithLogitsLoss()  # ordinal regression

        # ── Optimizer + scheduler ─────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        total_steps  = len(train_loader) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Mixed precision scaler
        use_fp16 = cfg.fp16 and self.device.type == "cuda"
        scaler   = torch.cuda.amp.GradScaler() if use_fp16 else None

        # ── Training loop ─────────────────────────────────────────────────
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        best_val_loss   = float("inf")
        patience_count  = 0
        best_ckpt_path  = ""

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(
                model, train_loader, optimizer, scheduler,
                intent_loss_fn, category_loss_fn, urgency_loss_fn,
                cfg, scaler, use_fp16,
            )
            val_loss, val_metrics = self._evaluate(
                model, val_loader,
                intent_loss_fn, category_loss_fn, urgency_loss_fn,
                cfg,
            )
            elapsed = time.time() - t0

            print(
                f"[trainer] Epoch {epoch:>2}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"intent_acc={val_metrics['intent_acc']:.3f} | "
                f"category_acc={val_metrics['category_acc']:.3f} | "
                f"urgency_mae={val_metrics['urgency_mae']:.2f} | "
                f"{elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_ckpt_path = f"{cfg.output_dir}/{cfg.run_name}_best.pt"
                self._save_checkpoint(model, optimizer, epoch, val_loss, cfg, best_ckpt_path)
                print(f"[trainer] ✓ New best model saved: {best_ckpt_path}")
            else:
                patience_count += 1
                if patience_count >= cfg.patience:
                    print(f"[trainer] Early stopping at epoch {epoch} (patience={cfg.patience})")
                    break

        print(f"[trainer] Training complete. Best val_loss={best_val_loss:.4f}")
        return best_ckpt_path

    # ── Epoch helpers ─────────────────────────────────────────────────────

    def _train_epoch(
        self, model, loader, optimizer, scheduler,
        intent_fn, category_fn, urgency_fn, cfg, scaler, use_fp16,
    ) -> float:
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if use_fp16:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(model, batch, intent_fn, category_fn, urgency_fn, cfg)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = self._compute_loss(model, batch, intent_fn, category_fn, urgency_fn, cfg)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate(self, model, loader, intent_fn, category_fn, urgency_fn, cfg):
        import torch
        model.eval()
        total_loss = 0.0
        intent_correct = category_correct = total = 0
        urgency_abs_err = 0.0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out   = model(batch["input_ids"], batch["attention_mask"])

                loss = self._compute_loss_from_out(out, batch, intent_fn, category_fn, urgency_fn, cfg)
                total_loss += loss.item()

                # Accuracy
                pred_intent   = out["intent_logits"].argmax(dim=-1)
                pred_category = out["category_logits"].argmax(dim=-1)
                pred_urgency  = (torch.sigmoid(out["urgency_logits"]) > 0.5).sum(dim=-1) + 1
                true_urgency  = batch["urgency_target"].sum(dim=-1) + 1

                intent_correct   += (pred_intent   == batch["intent_label"]).sum().item()
                category_correct += (pred_category == batch["category_label"]).sum().item()
                urgency_abs_err  += (pred_urgency - true_urgency).abs().float().sum().item()
                total            += batch["intent_label"].size(0)

        return total_loss / len(loader), {
            "intent_acc":   intent_correct   / total,
            "category_acc": category_correct / total,
            "urgency_mae":  urgency_abs_err  / total,
        }

    def _compute_loss(self, model, batch, intent_fn, category_fn, urgency_fn, cfg):
        out = model(batch["input_ids"], batch["attention_mask"])
        return self._compute_loss_from_out(out, batch, intent_fn, category_fn, urgency_fn, cfg)

    def _compute_loss_from_out(self, out, batch, intent_fn, category_fn, urgency_fn, cfg):
        l_intent   = intent_fn(out["intent_logits"],   batch["intent_label"])
        l_category = category_fn(out["category_logits"], batch["category_label"])
        l_urgency  = urgency_fn(out["urgency_logits"],  batch["urgency_target"])
        return (
            cfg.intent_loss_weight   * l_intent   +
            cfg.category_loss_weight * l_category +
            cfg.urgency_loss_weight  * l_urgency
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_records(path: str) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if not records:
            raise RuntimeError(f"No records loaded from {path}")
        print(f"[trainer] Loaded {len(records)} records from {path}")
        return records

    @staticmethod
    def _compute_intent_weights(records: list[dict]) -> "torch.Tensor":
        from collections import Counter
        counts = Counter(r.get("intent", "casual_chat") for r in records)
        total  = sum(counts.values())
        n_cls  = len(INTENT_LABELS)
        weights = []
        for label in INTENT_LABELS:
            c = counts.get(label, 1)
            weights.append(total / (n_cls * c))
        return torch.tensor(weights, dtype=torch.float)

    @staticmethod
    def _save_checkpoint(model, optimizer, epoch, val_loss, cfg, path: str):
        import torch
        torch.save({
            "epoch":            epoch,
            "val_loss":         val_loss,
            "encoder_name":     cfg.encoder_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "config":           cfg.__dict__,
        }, path)

    @staticmethod
    def _set_seed(seed: int):
        import torch
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
