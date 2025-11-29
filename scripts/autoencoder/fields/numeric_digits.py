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
        self.base = int(base)
        self.fraction_digits = int(fraction_digits)
        self.data_points = []
        self.has_negative = False
        self.has_nan = False
        self.integer_digits: Optional[int] = None
        self.total_positions: Optional[int] = None
        self.mask_index: Optional[int] = None
        self.vocab_size: Optional[int] = None

    def _accumulate_stats(self, raw_value):
        if raw_value is None:
            self.has_nan = True
            return
        try:
            v = float(raw_value)
        except (ValueError, TypeError):
            self.has_nan = True
            return
        if math.isnan(v):
            self.has_nan = True
            return
        if v < 0:
            self.has_negative = True
        self.data_points.append(v)

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

        self.mask_index = self.base
        self.vocab_size = self.base + 1

    def _ensure_finalized(self):
        if self.total_positions is None or self.vocab_size is None or self.mask_index is None:
            self._finalize_stats()

    def _get_input_shape(self):
        self._ensure_finalized()
        return (self.total_positions,)

    def _get_output_shape(self):
        return self._get_input_shape()

    def get_base_padding_value(self):
        self._ensure_finalized()
        return torch.full((self.total_positions,), int(self.mask_index), dtype=torch.long)

    def get_flag_padding_value(self):
        return torch.tensor([1.0], dtype=torch.float32)

    def _encode_nan(self):
        self._ensure_finalized()
        if self.has_nan:
            seq = [1]
            rest = self.total_positions - 1
            if rest > 0:
                seq.extend([0] * rest)
            return torch.tensor(seq, dtype=torch.long)
        return self.get_base_padding_value()

    def _encode_numeric(self, v: float):
        self._ensure_finalized()

        seq = []

        if self.has_nan:
            seq.append(0)

        if self.has_negative:
            seq.append(1 if v < 0 else 0)

        abs_val = abs(v)
        ipart = int(math.floor(abs_val))

        if self.integer_digits > 0:
            int_digits = self._int_to_digits(ipart, self.integer_digits)
            seq.extend(int_digits)

        if self.fraction_digits > 0:
            frac = abs_val - ipart
            scale = self.base ** self.fraction_digits
            scaled = int(round(frac * scale))
            if scaled >= scale:
                scaled = scale - 1
            frac_digits = self._int_to_digits(scaled, self.fraction_digits)
            seq.extend(frac_digits)

        if len(seq) < self.total_positions:
            seq.extend([0] * (self.total_positions - len(seq)))
        elif len(seq) > self.total_positions:
            seq = seq[: self.total_positions]

        return torch.tensor(seq, dtype=torch.long)

    def _int_to_digits(self, value, num_digits):
        digits = []
        for _ in range(num_digits):
            digits.append(value % self.base)
            value //= self.base
        return digits[::-1]

    def _is_missing(self, raw_value) -> bool:
        if raw_value is None:
            return True
        try:
            v = float(raw_value)
        except (ValueError, TypeError):
            return True
        if math.isnan(v):
            return True
        return False

    def _transform(self, raw_value):
        self._ensure_finalized()

        if self._is_missing(raw_value):
            if self.has_nan:
                return self._encode_nan()
            else:
                return self.get_base_padding_value()

        try:
            v = float(raw_value)
        except (ValueError, TypeError):
            if self.has_nan:
                return self._encode_nan()
            return self.get_base_padding_value()

        if math.isnan(v):
            if self.has_nan:
                return self._encode_nan()
            return self.get_base_padding_value()

        return self._encode_numeric(v)

    def transform(self, raw_value):
        return self._transform(raw_value)

    def transform_target(self, raw_value):
        return self._transform(raw_value)

    def render_prediction(self, prediction_tensor: torch.Tensor) -> str:
        # Prediction: (B, P, Vocab) -> Argmax
        if prediction_tensor.ndim >= 2 and prediction_tensor.shape[-1] in (self.base, self.vocab_size):
            indices = torch.argmax(prediction_tensor, dim=-1)
            return self.to_string(indices.detach().cpu().numpy())
        return self.to_string(prediction_tensor.detach().cpu().numpy())

    def render_ground_truth(self, target_tensor: torch.Tensor) -> str:
        # Target: (B, P) Indices
        return self.to_string(target_tensor.detach().cpu().numpy())

    def to_string(self, values: np.ndarray) -> str:
        self._ensure_finalized()

        arr = np.asarray(values)
        if arr.ndim > 1:
            arr = arr.flatten()

        digits = arr.astype(int).tolist()
        if not digits:
            return ""

        if all(d == self.mask_index for d in digits):
            return ""

        idx = 0

        if self.has_nan:
            if idx < len(digits):
                if digits[idx] == 1:
                    return "NaN"
            idx += 1

        negative = False
        if self.has_negative and idx < len(digits):
            negative = bool(digits[idx] == 1)
            idx += 1

        int_val = 0
        for _ in range(self.integer_digits):
            if idx >= len(digits):
                break
            d = digits[idx]
            if d == self.mask_index:
                d = 0
            d = max(0, min(self.base - 1, d))
            int_val = int_val * self.base + d
            idx += 1

        frac_val = 0
        for _ in range(self.fraction_digits):
            if idx >= len(digits):
                break
            d = digits[idx]
            if d == self.mask_index:
                d = 0
            d = max(0, min(self.base - 1, d))
            frac_val = frac_val * self.base + d
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
        self._ensure_finalized()
        B, P, V = pred.shape
        mask_id = int(self.mask_index) if self.mask_index is not None else -100
        return F.cross_entropy(
            pred.reshape(B * P, V),
            target.reshape(B * P).long(),
            ignore_index=mask_id,
        )

    def print_stats(self):
        return

    def encode_with_flag(self, raw_value):
        x = self._transform(raw_value)
        missing = 1.0 if self._is_missing(raw_value) else 0.0
        f = torch.tensor([missing], dtype=torch.float32)
        return x, f


class _DigitsEncoder(nn.Module):
    def __init__(self, base: int, positions: int):
        super().__init__()
        self.base = int(base)
        self.positions = int(positions)
        self.vocab_size = self.base + 1
        self.mask_index = self.vocab_size - 1
        emb_dim = self.base
        self.emb = nn.Embedding(self.vocab_size, emb_dim, padding_idx=self.mask_index)
        hidden = self.positions * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, x):
        x = self.emb(x)
        x = x.flatten(1)
        return self.mlp(x)


class _DigitsDecoder(nn.Module):
    def __init__(self, base: int, positions: int, latent_dim: int):
        super().__init__()
        self.base = int(base)
        self.positions = int(positions)
        self.vocab_size = self.base + 1
        hidden = self.positions * self.vocab_size
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, z):
        x = self.mlp(z)
        x = x.view(z.size(0), self.positions, self.vocab_size)
        return x