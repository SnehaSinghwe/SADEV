"""
models/multitask_model.py
SADEV multi-task text classifier.

Architecture:
  Input text
    → Transformer encoder (MuRIL or MiniLM)
    → Shared pooled representation
    → Three independent classification heads:
        1. intent_head   → 26 intent classes
        2. category_head → 8 category classes
        3. urgency_head  → 5 urgency levels (ordinal regression)

Design decisions:
  - Shared encoder + separate heads = efficient, avoids 3 independent models
  - Urgency uses ordinal loss (not cross-entropy) since levels are ordered
  - MuRIL is preferred for Hinglish (trained on 17 Indian languages)
  - Falls back to MiniLM if MuRIL unavailable (lighter, still decent)
  - All heads use dropout for regularisation

Inference output matches EmotionResult fields so local_pipeline/sadev_pipeline.py
can swap in classifier output with zero code changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── Label maps ────────────────────────────────────────────────────────────

INTENT_LABELS = [
    "venting_distress", "anxiety_panic", "sadness_low_mood", "overwhelm_burnout",
    "loneliness", "career_job_stress", "academic_pressure", "financial_stress",
    "family_conflict", "relationship_issue", "marriage_pressure", "social_reputation",
    "conflict_exposure", "fear_of_consequence", "advice_seeking", "decision_conflict",
    "planning_next_steps", "sleep_issue", "focus_issue", "productivity_block",
    "hopelessness", "emotional_exhaustion", "self_harm_intent", "suicidal_thoughts",
    "greeting", "casual_chat",
]

CATEGORY_LABELS = [
    "emotional", "life_stressor", "social_stress", "problem_solving",
    "behavioral", "high_distress", "crisis", "neutral",
]

URGENCY_LABELS = [1, 2, 3, 4, 5]  # ordinal

INTENT_TO_IDX   = {l: i for i, l in enumerate(INTENT_LABELS)}
CATEGORY_TO_IDX = {l: i for i, l in enumerate(CATEGORY_LABELS)}
IDX_TO_INTENT   = {i: l for l, i in INTENT_TO_IDX.items()}
IDX_TO_CATEGORY = {i: l for l, i in CATEGORY_TO_IDX.items()}

# Preferred encoder models (tried in order)
_ENCODER_CANDIDATES = [
    "google/muril-base-cased",         # Best for Hinglish / Indian languages
    "sentence-transformers/paraphrase-MiniLM-L6-v2",  # Lightweight fallback
    "bert-base-multilingual-cased",    # Last resort
]


@dataclass
class ClassifierOutput:
    """Output of a single inference pass."""
    intent: str
    category: str
    urgency: int
    intent_confidence: float
    category_confidence: float
    urgency_confidence: float
    intent_logits: "Optional[list[float]]" = None
    category_logits: "Optional[list[float]]" = None


# ── Model definition ──────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class SADEVClassifier(nn.Module):
        """
        Multi-task transformer classifier.

        Args:
            encoder_name: HuggingFace model ID for the shared encoder
            dropout:      dropout probability for all heads
            freeze_encoder_layers: number of encoder layers to freeze during training
                                   (0 = train all, useful when data is small)
        """

        def __init__(
            self,
            encoder_name: str = "google/muril-base-cased",
            dropout: float = 0.3,
            freeze_encoder_layers: int = 0,
        ):
            super().__init__()
            self.encoder_name = encoder_name
            self.encoder = AutoModel.from_pretrained(encoder_name)
            hidden_size = self.encoder.config.hidden_size

            # Optionally freeze early encoder layers
            if freeze_encoder_layers > 0:
                for i, layer in enumerate(self.encoder.encoder.layer):
                    if i < freeze_encoder_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

            # Shared projection layer (reduces dim, adds non-linearity)
            self.shared_proj = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Head 1: intent (26 classes)
            self.intent_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, len(INTENT_LABELS)),
            )

            # Head 2: category (8 classes)
            self.category_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, len(CATEGORY_LABELS)),
            )

            # Head 3: urgency (5 ordinal levels)
            # Uses 4 binary classifiers for ordinal regression: P(y > k) for k=1..4
            self.urgency_head = nn.Sequential(
                nn.Linear(512, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, len(URGENCY_LABELS) - 1),  # 4 thresholds
            )

        def forward(
            self,
            input_ids: "torch.Tensor",
            attention_mask: "torch.Tensor",
            token_type_ids: "Optional[torch.Tensor]" = None,
        ) -> dict[str, "torch.Tensor"]:
            encoder_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                encoder_kwargs["token_type_ids"] = token_type_ids

            outputs = self.encoder(**encoder_kwargs)
            # CLS token representation
            cls_repr = outputs.last_hidden_state[:, 0, :]
            shared   = self.shared_proj(cls_repr)

            return {
                "intent_logits":   self.intent_head(shared),
                "category_logits": self.category_head(shared),
                "urgency_logits":  self.urgency_head(shared),   # shape: (B, 4)
            }

        @classmethod
        def from_pretrained_encoder(cls, encoder_name: str | None = None, **kwargs) -> "SADEVClassifier":
            """Try encoder candidates in order until one loads successfully."""
            candidates = ([encoder_name] if encoder_name else []) + _ENCODER_CANDIDATES
            for name in candidates:
                try:
                    model = cls(encoder_name=name, **kwargs)
                    print(f"[model] Loaded encoder: {name}")
                    return model
                except Exception as e:
                    print(f"[model] Could not load {name}: {e}")
            raise RuntimeError("No encoder could be loaded. Check HuggingFace connectivity.")


# ── Urgency: ordinal regression helpers ───────────────────────────────────

def urgency_logits_to_level(logits: "torch.Tensor") -> "torch.Tensor":
    """
    Convert 4 ordinal threshold logits to urgency level (1-5).
    P(y > k) = sigmoid(logit_k), then sum P(y > k) for k=1..4 and add 1.
    This is the standard ordinal regression decoding.
    """
    import torch
    probs = torch.sigmoid(logits)           # P(y > k) for k = 1..4
    return (probs > 0.5).sum(dim=-1) + 1   # urgency in {1, 2, 3, 4, 5}


def urgency_to_ordinal_targets(urgency: int):
    """
    Convert urgency level (1-5) to ordinal binary targets.
    urgency=3 → [1, 1, 0, 0]  (y>1, y>2, not y>3, not y>4)
    Returns a torch.Tensor if torch is available, else a plain list.
    """
    targets = [1 if urgency > k else 0 for k in range(1, 5)]
    if _TORCH_AVAILABLE:
        import torch
        return torch.tensor(targets, dtype=torch.float)
    return targets


# ── Inference wrapper ─────────────────────────────────────────────────────

class SADEVInference:
    """
    Inference wrapper for SADEVClassifier.
    Handles tokenisation, batching, and output formatting.

    Usage:
        model = SADEVInference.load("models/sadev_v1.pt")
        result = model.predict("yaar I'm so stressed about placement season")
    """

    def __init__(self, model: "SADEVClassifier", tokenizer, device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for inference.")
        self.model     = model.eval()
        self.tokenizer = tokenizer
        self.device    = torch.device(device)
        self.model.to(self.device)

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "SADEVInference":
        """Load a saved checkpoint."""
        import torch
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")
        ckpt = torch.load(checkpoint_path, map_location=device)
        encoder_name = ckpt.get("encoder_name", "google/muril-base-cased")
        model = SADEVClassifier(encoder_name=encoder_name)
        model.load_state_dict(ckpt["model_state_dict"])
        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        print(f"[inference] Loaded checkpoint: {checkpoint_path}")
        return cls(model, tokenizer, device)

    def predict(self, text: str, return_logits: bool = False) -> ClassifierOutput:
        """Predict intent, category, urgency for a single text."""
        import torch
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**encoding)

        intent_logits   = out["intent_logits"][0]
        category_logits = out["category_logits"][0]
        urgency_logits  = out["urgency_logits"][0]

        intent_probs   = torch.softmax(intent_logits, dim=-1)
        category_probs = torch.softmax(category_logits, dim=-1)

        intent_idx   = intent_probs.argmax().item()
        category_idx = category_probs.argmax().item()
        urgency_val  = urgency_logits_to_level(urgency_logits.unsqueeze(0)).item()

        return ClassifierOutput(
            intent=IDX_TO_INTENT[intent_idx],
            category=IDX_TO_CATEGORY[category_idx],
            urgency=int(urgency_val),
            intent_confidence=round(float(intent_probs[intent_idx]), 4),
            category_confidence=round(float(category_probs[category_idx]), 4),
            urgency_confidence=round(float(torch.sigmoid(urgency_logits).max()), 4),
            intent_logits=intent_logits.tolist() if return_logits else None,
            category_logits=category_logits.tolist() if return_logits else None,
        )

    def predict_batch(self, texts: list[str]) -> list[ClassifierOutput]:
        """Predict for a batch of texts."""
        import torch
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**encoding)

        results = []
        for i in range(len(texts)):
            ip = torch.softmax(out["intent_logits"][i], dim=-1)
            cp = torch.softmax(out["category_logits"][i], dim=-1)
            ii = ip.argmax().item()
            ci = cp.argmax().item()
            uv = urgency_logits_to_level(out["urgency_logits"][i].unsqueeze(0)).item()
            results.append(ClassifierOutput(
                intent=IDX_TO_INTENT[ii],
                category=IDX_TO_CATEGORY[ci],
                urgency=int(uv),
                intent_confidence=round(float(ip[ii]), 4),
                category_confidence=round(float(cp[ci]), 4),
                urgency_confidence=round(float(torch.sigmoid(out["urgency_logits"][i]).max()), 4),
            ))
        return results
