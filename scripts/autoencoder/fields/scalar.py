from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseField
from .scaling import Scaling

class ScalarField(BaseField):
    def __init__(
        self,
        name: str,
        scaling: Scaling = Scaling.NONE,
        clip_max=None,
        optional: bool = False,
    ):
        super().__init__(name, optional)
        self.scaling = scaling
        self.clip_max = clip_max
        self.n = 0
        self.sum_ = 0.0
        self.sum_sq = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.mean_val = 0.0
        self.std_val = 1.0

    def _get_input_shape(self):
        return (1,)

    def _get_output_shape(self):
        return (1,)

    def get_base_padding_value(self):
        zero_val = 0.0
        if self.scaling == Scaling.NONE:
            t = self.scaling.none_transform(zero_val)
        elif self.scaling == Scaling.NORMALIZE:
            min_v = self.min_val if self.min_val != float("inf") else 0.0
            max_v = self.max_val if self.max_val != float("-inf") else 0.0
            t = self.scaling.normalize_transform(zero_val, min_val=min_v, max_val=max_v)
        elif self.scaling == Scaling.STANDARDIZE:
            mean_v = self.mean_val if self.n > 0 else 0.0
            std_v = self.std_val if self.n > 0 else 1.0
            t = self.scaling.standardize_transform(zero_val, mean_val=mean_v, std_val=std_v)
        elif self.scaling == Scaling.LOG:
            t = self.scaling.log_transform(zero_val)
        return torch.tensor([t], dtype=torch.float32)

    def get_flag_padding_value(self):
        return torch.tensor([1.0], dtype=torch.float32)

    def _accumulate_stats(self, raw_value):
        if raw_value is not None:
            try:
                v = float(raw_value)
                self.n += 1
                self.sum_ += v
                self.sum_sq += v * v
                if v < self.min_val:
                    self.min_val = v
                if v > self.max_val:
                    self.max_val = v
            except ValueError:
                pass

    def _finalize_stats(self):
        if self.n > 0:
            self.mean_val = self.sum_ / self.n
            var = (self.sum_sq / self.n) - (self.mean_val ** 2)
            self.std_val = np.sqrt(max(0.0, var)) if max(0.0, var) > 1e-12 else 1.0
            if self.std_val == 0:
                self.std_val = 1.0
        else:
            self.min_val = 0.0
            self.max_val = 0.0
            self.mean_val = 0.0
            self.std_val = 1.0

    def _transform(self, raw_value):
        try:
            x = float(raw_value)
        except (ValueError, TypeError):
            x = 0.0
        if self.clip_max is not None:
            x = min(x, self.clip_max)
        if self.scaling == Scaling.NONE:
            t = self.scaling.none_transform(x)
        elif self.scaling == Scaling.NORMALIZE:
            t = self.scaling.normalize_transform(x, min_val=self.min_val, max_val=self.max_val)
        elif self.scaling == Scaling.STANDARDIZE:
            t = self.scaling.standardize_transform(x, mean_val=self.mean_val, std_val=self.std_val)
        elif self.scaling == Scaling.LOG:
            t = self.scaling.log_transform(x)
        return torch.tensor([t], dtype=torch.float32)

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        v = float(predicted_main.flatten()[0])
        if self.scaling == Scaling.NONE:
            v = self.scaling.none_untransform(v)
        elif self.scaling == Scaling.NORMALIZE:
            v = self.scaling.normalize_untransform(v, min_val=self.min_val, max_val=self.max_val)
        elif self.scaling == Scaling.STANDARDIZE:
            v = self.scaling.standardize_untransform(v, mean_val=self.mean_val, std_val=self.std_val)
        elif self.scaling == Scaling.LOG:
            v = self.scaling.log_untransform(v)
        return f"{v:.2f}"

    def build_encoder(self, latent_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.LayerNorm(8),
            nn.Linear(8, 8),
            nn.GELU(),
            nn.LayerNorm(8),
        )

    def build_decoder(self, latent_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.GELU(),
            nn.LayerNorm(8),
            nn.Linear(8, 8),
            nn.GELU(),
            nn.LayerNorm(8),
            nn.Linear(8, 1),
        )

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(pred, target)

    def print_stats(self):
        return
