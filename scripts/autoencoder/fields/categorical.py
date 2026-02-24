from typing import List, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseField

class _CategoryStateMixin:
    """Shared get_state/set_state for Multi/SingleCategoryField."""
    category_list: List[str]
    category_set: set
    category_counts: Dict[str, int]

    def get_state(self):
        state = super().get_state()
        state.update({
            "category_list": list(self.category_list or []),
            "category_counts": dict(self.category_counts or {}),
        })
        return state

    def set_state(self, state):
        super().set_state(state)
        self.category_list = list(state.get("category_list", []) or [])
        self.category_set = set(self.category_list)
        self.category_counts = dict(state.get("category_counts", {}) or {})


class MultiCategoryField(_CategoryStateMixin, BaseField):
    def __init__(self, name: str, optional: bool = False):
        super().__init__(name, optional)
        self.category_set = set()
        self.category_counts: Dict[str, int] = {}
        self.category_list: List[str] = []

    def _get_input_shape(self):
        return (len(self.category_list),) if self.category_list else (0,)

    def _get_output_shape(self):
        return (len(self.category_list),) if self.category_list else (0,)

    def get_base_padding_value(self):
        return torch.zeros((len(self.category_list),), dtype=torch.float32)

    def get_flag_padding_value(self):
        return torch.tensor([1.0], dtype=torch.float32)

    def _accumulate_stats(self, raw_value):
        if raw_value:
            cats = raw_value if isinstance(raw_value, list) else [raw_value]
            for c in set(cats):
                c = str(c)
                if c:
                    self.category_set.add(c)
                    self.category_counts[c] = self.category_counts.get(c, 0) + 1

    def _finalize_stats(self):
        self.category_list = sorted(self.category_set)

    def _transform(self, raw_value):
        v = torch.zeros((len(self.category_list),), dtype=torch.float32)
        cats = raw_value if isinstance(raw_value, list) else [raw_value] if raw_value is not None else []
        for c in cats:
            try:
                idx = self.category_list.index(str(c))
                v[idx] = 1.0
            except ValueError:
                pass
        return v

    def transform(self, raw_value):
        return self._transform(raw_value if raw_value is not None else [])

    def transform_target(self, raw_value):
        return self._transform(raw_value if raw_value is not None else [])

    def render_prediction(self, prediction_tensor: torch.Tensor) -> str:
        # Predictions are logits -> Apply Sigmoid
        probs = torch.sigmoid(prediction_tensor)
        return self.to_string(probs.detach().cpu().numpy())

    def render_ground_truth(self, target_tensor: torch.Tensor) -> str:
        # Targets are 0.0 or 1.0 -> Pass through
        return self.to_string(target_tensor.detach().cpu().numpy())

    def to_string(self, values: np.ndarray, threshold: float = 0.5) -> str:
        # Takes Probabilities or Binary targets
        probs = np.asarray(values).flatten().astype(float)
        chosen = [(c, p) for c, p in zip(self.category_list, probs) if p >= threshold]
        
        # If no categories passed threshold, pick the highest probability one
        # (Only if the input isn't all zeros, which might happen in empty GT)
        if not chosen and np.max(probs) > 0:
            i = int(np.argmax(probs))
            chosen = [(self.category_list[i], probs[i])]
            
        return " ".join(f"{c}:{p:.2f}" for c, p in chosen)

    def build_encoder(self, latent_dim: int) -> nn.Module:
        d = max(8, max(1, len(self.category_list) // 4))
        return nn.Sequential(
            nn.Linear(len(self.category_list), d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

    def build_decoder(self, latent_dim: int) -> nn.Module:
        d = max(8, max(1, len(self.category_list) // 4))
        return nn.Sequential(
            nn.Linear(latent_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, len(self.category_list)),
        )

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred, target)

    def print_stats(self):
        return


class SingleCategoryField(_CategoryStateMixin, BaseField):
    def __init__(self, name: str, optional: bool = False):
        super().__init__(name, optional)
        self.category_set = set()
        self.category_counts: Dict[str, int] = {}
        self.category_list: List[str] = []

    def _get_input_shape(self):
        return (1,)

    def _get_output_shape(self):
        return (1,)

    def _accumulate_stats(self, raw_value):
        if raw_value is not None:
            s = str(raw_value)
            if s:
                self.category_set.add(s)
                self.category_counts[s] = self.category_counts.get(s, 0) + 1

    def _finalize_stats(self):
        self.category_list = sorted(list(self.category_set))

    def _transform(self, raw_value):
        if raw_value is None:
            if not self.optional:
                raise ValueError(f"Field '{self.name}' is not optional, but received None.")
            return torch.tensor([0], dtype=torch.long)
        s = str(raw_value)
        try:
            idx = self.category_list.index(s)
        except ValueError:
            idx = 0
        return torch.tensor([idx], dtype=torch.long)

    def transform_target(self, raw_value):
        return self._transform(raw_value)

    def transform(self, raw_value):
        return self._transform(raw_value)

    def render_prediction(self, prediction_tensor: torch.Tensor) -> str:
        # Prediction: (B, NumClasses) logits -> Argmax
        idx = torch.argmax(prediction_tensor, dim=-1)
        return self.to_string(idx.detach().cpu().numpy())

    def render_ground_truth(self, target_tensor: torch.Tensor) -> str:
        # Target: (B, 1) index
        return self.to_string(target_tensor.detach().cpu().numpy())

    def to_string(self, values: np.ndarray) -> str:
        vec = values.flatten()
        if vec.size == 1:
            idx = int(vec[0])
        else:
            idx = int(np.argmax(vec))
            
        if 0 <= idx < len(self.category_list):
            return self.category_list[idx]
        return "[Unknown]"

    def build_encoder(self, latent_dim: int) -> nn.Module:
        emb_dim = max(1, len(self.category_list) // 4) or 1
        return nn.Sequential(
            nn.Embedding(num_embeddings=max(1, len(self.category_list)), embedding_dim=max(1, emb_dim)),
            nn.Flatten(),
            nn.Linear(max(1, emb_dim), max(1, emb_dim)),
        )

    def build_decoder(self, latent_dim: int) -> nn.Module:
        out_dim = max(1, len(self.category_list))
        return nn.Linear(latent_dim, out_dim)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(pred, target.long().squeeze(-1))

    def print_stats(self):
        return