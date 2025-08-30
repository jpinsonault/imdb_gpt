from typing import Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseField

class NumericDigitCategoryField(BaseField):
    def __init__(self, name: str, base: int = 10, fraction_digits: int = 0, optional: bool = False):
        super().__init__(name, optional)
        self.base = base
        self.fraction_digits = fraction_digits
        self.data_points = []
        self.has_negative = False
        self.has_nan = False
        self.integer_digits = None
        self.total_positions = None

    def _accumulate_stats(self, raw_value):
        if raw_value is None:
            self.has_nan = True
        else:
            try:
                v = float(raw_value)
                if math.isnan(v):
                    self.has_nan = True
                else:
                    if v < 0:
                        self.has_negative = True
                    self.data_points.append(v)
            except (ValueError, TypeError):
                self.has_nan = True

    def _finalize_stats(self):
        if not self.data_points:
            self.integer_digits = 1
        else:
            abs_ints = [int(math.floor(abs(v))) for v in self.data_points]
            max_int = max(abs_ints)
            if max_int > 0:
                needed = int(math.floor(math.log(max_int, self.base))) + 1
            else:
                needed = 0
            self.integer_digits = needed or 1
        self.total_positions = (
            (1 if self.has_nan else 0)
            + (1 if self.has_negative else 0)
            + self.integer_digits
            + self.fraction_digits
        )

    def _get_input_shape(self):
        if self.total_positions is None:
            self._finalize_stats()
        return (self.total_positions,)

    def _get_output_shape(self):
        return self._get_input_shape()

    def get_base_padding_value(self):
        return torch.zeros((self._get_input_shape()[0],), dtype=torch.long)

    def get_flag_padding_value(self):
        return torch.tensor([1.0], dtype=torch.float32)

    def _transform(self, raw_value):
        if self.total_positions is None:
            self._finalize_stats()

        is_nan = False
        raw = 0.0
        if raw_value is None:
            is_nan = True
        else:
            try:
                v = float(raw_value)
                if math.isnan(v):
                    is_nan = True
                else:
                    raw = v
            except (ValueError, TypeError):
                is_nan = True

        if is_nan:
            seq = [1] + [0] * (self.total_positions - 1) if self.has_nan else [0] * self.total_positions
            return torch.tensor(seq, dtype=torch.long)

        seq = []
        if self.has_nan:
            seq.append(0)
        if self.has_negative:
            seq.append(1 if raw < 0 else 0)

        abs_val = abs(raw)
        ipart = int(math.floor(abs_val))
        if self.integer_digits > 0:
            int_digits = self._int_to_digits(ipart, self.integer_digits)
        else:
            int_digits = []
        seq.extend(int_digits)

        if self.fraction_digits > 0:
            frac = abs_val - ipart
            scaled = int(round(frac * (self.base ** self.fraction_digits)))
            scaled = min(scaled, self.base ** self.fraction_digits - 1)
            seq.extend(self._int_to_digits(scaled, self.fraction_digits))

        if len(seq) < self.total_positions:
            seq += [0] * (self.total_positions - len(seq))

        return torch.tensor(seq, dtype=torch.long)

    def _int_to_digits(self, value, num_digits):
        digits = []
        for _ in range(num_digits):
            digits.append(value % self.base)
            value //= self.base
        return digits[::-1]

    def to_string(self, predicted_tensor: np.ndarray, flag_tensor: Optional[np.ndarray] = None) -> str:
        if self.total_positions is None or self.base is None or self.integer_digits is None:
            self._finalize_stats()
        arr = np.asarray(predicted_tensor)
        if arr.ndim >= 2 and arr.shape[-1] == self.base:
            arr = np.argmax(arr, axis=-1)
        digits = arr.flatten().astype(int).tolist()
        idx = 0
        if self.has_nan:
            if digits[idx] == 1:
                return "NaN"
            idx += 1
        negative = False
        if self.has_negative:
            negative = bool(digits[idx] == 1)
            idx += 1
        int_val = 0
        for _ in range(self.integer_digits):
            int_val = int_val * self.base + digits[idx]
            idx += 1
        frac_val = 0
        for _ in range(self.fraction_digits):
            frac_val = frac_val * self.base + digits[idx]
            idx += 1
        s = f"{int_val}"
        if self.fraction_digits > 0:
            s += "." + f"{frac_val:0{self.fraction_digits}d}"
        return ("-" if negative else "") + s

    def build_encoder(self, latent_dim: int) -> nn.Module:
        positions = self._get_input_shape()[0]
        return _DigitsEncoder(base=self.base, positions=positions)

    def build_decoder(self, latent_dim: int) -> nn.Module:
        positions = self._get_input_shape()[0]
        return _DigitsDecoder(base=self.base, positions=positions, latent_dim=latent_dim)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, P, V = pred.shape
        return F.cross_entropy(pred.reshape(B * P, V), target.reshape(B * P).long())

    def print_stats(self):
        return


class _DigitsEncoder(nn.Module):
    def __init__(self, base: int, positions: int):
        super().__init__()
        self.base = base
        self.positions = positions
        self.emb = nn.Embedding(base, base)
        self.mlp = nn.Sequential(
            nn.Linear(positions * base, positions * base),
            nn.GELU(),
            nn.LayerNorm(positions * base),
            nn.Linear(positions * base, positions * base),
            nn.GELU(),
            nn.LayerNorm(positions * base),
        )

    def forward(self, x):
        x = self.emb(x)
        x = x.flatten(1)
        return self.mlp(x)


class _DigitsDecoder(nn.Module):
    def __init__(self, base: int, positions: int, latent_dim: int):
        super().__init__()
        self.base = base
        self.positions = positions
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, positions * base),
            nn.GELU(),
            nn.LayerNorm(positions * base),
            nn.Linear(positions * base, positions * base),
            nn.GELU(),
            nn.LayerNorm(positions * base),
        )

    def forward(self, z):
        x = self.mlp(z)
        x = x.view(z.size(0), self.positions, self.base)
        return x
