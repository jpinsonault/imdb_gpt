from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseField

class BooleanField(BaseField):
    def __init__(self, name: str, use_bce_loss: bool = True, optional: bool = False):
        super().__init__(name, optional)
        self.use_bce_loss = use_bce_loss
        self.count_total = 0
        self.count_ones = 0

    def _get_input_shape(self):
        return (1,)

    def _get_output_shape(self):
        return (1,)

    def get_base_padding_value(self):
        return torch.tensor([0.0], dtype=torch.float32)

    def get_flag_padding_value(self):
        return torch.tensor([1.0], dtype=torch.float32)

    def _accumulate_stats(self, raw_value):
        if raw_value is not None:
            try:
                v = float(raw_value)
                v = 1.0 if v == 1.0 else 0.0
                self.count_total += 1
                if v == 1.0:
                    self.count_ones += 1
            except ValueError:
                pass

    def _finalize_stats(self):
        return

    def _transform(self, raw_value):
        try:
            v = float(raw_value)
        except (ValueError, TypeError):
            v = 0.0
        v = 1.0 if v == 1.0 else 0.0
        return torch.tensor([v], dtype=torch.float32)

    def render_prediction(self, prediction_tensor: torch.Tensor) -> str:
        # Convert logits/tanh to 0.0-1.0 probability range
        if self.use_bce_loss:
            val = torch.sigmoid(prediction_tensor)
        else:
            # Tanh output (-1 to 1) -> (0 to 1) approx or just check sign
            val = (torch.tanh(prediction_tensor) + 1) / 2
        return self.to_string(val.detach().cpu().numpy())

    def render_ground_truth(self, target_tensor: torch.Tensor) -> str:
        return self.to_string(target_tensor.detach().cpu().numpy())

    def to_string(self, values: np.ndarray) -> str:
        p = float(values.flatten()[0])
        return "True" if p >= 0.5 else "False"

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
        if self.use_bce_loss:
            return F.binary_cross_entropy_with_logits(pred, target)
        return F.mse_loss(torch.tanh(pred), target)

    def print_stats(self):
        return