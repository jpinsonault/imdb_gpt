import math
import logging
from enum import Enum
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .character_tokenizer import CharacterTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Scaling(Enum):
    NONE = 1
    NORMALIZE = 2
    STANDARDIZE = 3
    LOG = 4

    def none_transform(self, x, **kwargs):
        return x

    def none_untransform(self, x, **kwargs):
        return x

    def normalize_transform(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val) if max_val > min_val else 0.0

    def normalize_untransform(self, x, min_val, max_val):
        return x * (max_val - min_val) + min_val

    def standardize_transform(self, x, mean_val, std_val):
        return (x - mean_val) / std_val if std_val != 0 else 0.0

    def standardize_untransform(self, x, mean_val, std_val):
        return x * std_val + mean_val

    def log_transform(self, x):
        return np.log1p(x)

    def log_untransform(self, x):
        return np.expm1(x)


SPECIAL_PAD = "\u200C"
SPECIAL_START = "\u200D"
SPECIAL_END = "\u200E"
SPECIAL_SEP = "\u200F"


class BaseField:
    def __init__(self, name: str, optional: bool = False):
        self.name = name
        self.optional = optional
        self._stats_finalized = False

    def _get_input_shape(self):
        raise NotImplementedError

    def _get_output_shape(self):
        raise NotImplementedError

    def _accumulate_stats(self, raw_value):
        raise NotImplementedError

    def _finalize_stats(self):
        raise NotImplementedError

    def _transform(self, raw_value):
        raise NotImplementedError

    def build_encoder(self, latent_dim: int) -> nn.Module:
        raise NotImplementedError

    def build_decoder(self, latent_dim: int) -> nn.Module:
        raise NotImplementedError

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        raise NotImplementedError

    def print_stats(self):
        raise NotImplementedError

    def get_base_padding_value(self):
        raise NotImplementedError

    def get_flag_padding_value(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        return self._get_input_shape()

    @property
    def output_shape(self):
        return self._get_output_shape()

    @property
    def weight(self):
        return self._get_weight()

    def _get_weight(self):
        return 1.0

    def accumulate_stats(self, raw_value):
        self._accumulate_stats(raw_value)

    def finalize_stats(self):
        if self._stats_finalized:
            return
        self._finalize_stats()
        self._stats_finalized = True

    def stats_finalized(self) -> bool:
        return self._stats_finalized

    def transform(self, raw_value):
        if raw_value is None:
            if not self.optional:
                raise ValueError(f"Field '{self.name}' is not optional, but received None.")
            return self.get_base_padding_value()
        return self._transform(raw_value)

    def transform_target(self, raw_value):
        return self.transform(raw_value)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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
        pass

    def _transform(self, raw_value):
        try:
            v = float(raw_value)
        except (ValueError, TypeError):
            v = 0.0
        v = 1.0 if v == 1.0 else 0.0
        return torch.tensor([v], dtype=torch.float32)

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        p = float(predicted_main.flatten()[0])
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
        pass


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
        return nn.Identity()

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
        return F.mse_loss(pred, target)

    def print_stats(self):
        pass


class MultiCategoryField(BaseField):
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

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None, threshold: float = 0.5) -> str:
        probs = predicted_main.flatten().astype(float)
        chosen = [(c, p) for c, p in zip(self.category_list, probs) if p >= threshold]
        if not chosen and len(probs) > 0:
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
        pass


class SingleCategoryField(BaseField):
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
        if not self.category_list:
            logging.warning(f"No categories found for SingleCategoryField '{self.name}'.")

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

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        vec = predicted_main.flatten()
        idx = int(np.argmax(vec))
        if 0 <= idx < len(self.category_list):
            return self.category_list[idx]
        return "[Unknown]"

    def build_encoder(self, latent_dim: int) -> nn.Module:
        emb_dim = max(1, len(self.category_list) // 4) or 1
        return nn.Sequential(
            nn.Embedding(num_embeddings=max(1, len(self.category_list)), embedding_dim=max(1, emb_dim)),
            nn.Flatten(),
        )

    def build_decoder(self, latent_dim: int) -> nn.Module:
        out_dim = max(1, len(self.category_list))
        return nn.Linear(latent_dim, out_dim)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(pred, target.long().squeeze(-1))

    def print_stats(self):
        pass


class TextField(BaseField):
    def __init__(
        self,
        name: str,
        max_length: Optional[int] = None,
        downsample_steps: int = 2,
        base_size: int = 48,
        num_blocks_per_step: List[int] = [2, 2],
        optional: bool = False,
    ):
        super().__init__(name, optional=optional)
        self.user_max_length = max_length
        self.downsample_steps = downsample_steps
        self.base_size = base_size
        self.num_blocks_per_step = num_blocks_per_step
        self.texts: List[str] = []
        self.dynamic_max_len: int = 0
        self.tokenizer: Optional[CharacterTokenizer] = None
        self.max_length: Optional[int] = None
        self.pad_token_id: Optional[int] = None
        self.null_token_id: Optional[int] = None
        self.avg_raw_length: Optional[float] = None
        self.avg_token_count: Optional[float] = None
        self.avg_chars_saved: Optional[float] = None
        self.compression_ratio: Optional[float] = None

    def _get_input_shape(self):
        if self.max_length is None:
            raise ValueError("TextField stats not finalized. Call finalize_stats() first.")
        return (self.max_length,)

    def _get_output_shape(self):
        if self.max_length is None:
            raise ValueError("TextField stats not finalized. Call finalize_stats() first.")
        return (self.max_length,)

    def _accumulate_stats(self, raw_value):
        if raw_value is not None:
            s = str(raw_value)
            if s:
                self.texts.append(s)

    def _finalize_stats(self):
        special_tokens = ["<unk>", SPECIAL_PAD, SPECIAL_START, SPECIAL_END, SPECIAL_SEP]
        self.tokenizer = CharacterTokenizer(special_tokens=special_tokens)
        self.tokenizer.train(self.texts if self.texts else [])

        self.pad_token_id = self.tokenizer.token_to_id(SPECIAL_PAD)
        max_tokens = 0
        total_raw = 0
        total_tokens = 0
        n = len(self.texts)
        if n > 0:
            for t in self.texts:
                ids = self.tokenizer.encode(t)
                total_tokens += len(ids)
                total_raw += len(t)
                if len(ids) > max_tokens:
                    max_tokens = len(ids)
            self.avg_raw_length = total_raw / n
            self.avg_token_count = total_tokens / n
            self.avg_chars_saved = self.avg_raw_length - self.avg_token_count
            self.compression_ratio = (self.avg_raw_length / self.avg_token_count) if self.avg_token_count else None
        else:
            self.avg_raw_length = 0.0
            self.avg_token_count = 0.0
            self.avg_chars_saved = 0.0
            self.compression_ratio = None

        self.dynamic_max_len = max_tokens
        eff = max_tokens
        if self.user_max_length is not None:
            self.max_length = self.user_max_length
        else:
            self.max_length = eff
        self.max_length = max(1, self.max_length)

        multiple = 2 ** self.downsample_steps
        if multiple > 1:
            adj = max(multiple, self.max_length)
            rounded = ((adj + multiple - 1) // multiple) * multiple
            self.max_length = rounded

    def _transform(self, raw_value):
        txt = str(raw_value)
        token_ids = self.tokenizer.encode(txt)
        cur = len(token_ids)
        if cur < self.max_length:
            token_ids += [self.pad_token_id] * (self.max_length - cur)
        else:
            token_ids = token_ids[: self.max_length]
        return torch.tensor(token_ids, dtype=torch.long)

    def get_base_padding_value(self):
        if self.pad_token_id is None or self.max_length is None:
            raise RuntimeError("TextField stats not finalized. Call finalize_stats() first.")
        return torch.tensor([self.pad_token_id] * self.max_length, dtype=torch.long)

    def get_flag_padding_value(self):
        return torch.tensor([1.0], dtype=torch.float32)

    def to_string(self, predicted_main: np.ndarray, predicted_flag: Optional[np.ndarray] = None) -> str:
        arr = np.asarray(predicted_main)
        if arr.ndim >= 2 and self.tokenizer is not None and arr.shape[-1] == self.tokenizer.get_vocab_size():
            arr = np.argmax(arr, axis=-1)
        if arr.ndim > 1:
            arr = arr.flatten()
        ids = arr.astype(int).tolist()
        toks = [self.tokenizer.id_to_token(i) for i in ids]
        out = []
        for t in toks:
            if t == SPECIAL_END:
                break
            if t in (SPECIAL_START, SPECIAL_PAD):
                continue
            out.append(t)
        return "".join(out)

    def build_encoder(self, latent_dim: int) -> nn.Module:
        vocab = self.tokenizer.get_vocab_size()
        ch = self.base_size
        return _TextEncoder(vocab=vocab, max_len=self.max_length, ch=ch, latent_dim=latent_dim)

    def build_decoder(self, latent_dim: int) -> nn.Module:
        vocab = self.tokenizer.get_vocab_size()
        ch = self.base_size
        return _TextDecoder(vocab=vocab, max_len=self.max_length, ch=ch, latent_dim=latent_dim)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pad_token_id is None:
            raise RuntimeError("TextField not finalized")
        B, L, V = pred.shape
        pred2 = pred.reshape(B * L, V)
        tgt2 = target.reshape(B * L)
        return F.cross_entropy(pred2, tgt2, ignore_index=int(self.pad_token_id))

    def print_stats(self):
        pass


class _TextEncoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, ch: int, latent_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, ch)
        self.conv = nn.Conv1d(ch, ch * 2, kernel_size=5, stride=2, padding=2)
        self.out = nn.Linear((max_len // 2) * (ch * 2), latent_dim)

    def forward(self, x):
        x = self.emb(x)          # B,L,C
        x = x.transpose(1, 2)    # B,C,L
        x = F.gelu(self.conv(x)) # B,2C,L/2
        x = x.flatten(1)
        return self.out(x)


class _TextDecoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, ch: int, latent_dim: int):
        super().__init__()
        self.seq_half = max_len // 2
        self.ch = ch
        self.fc = nn.Linear(latent_dim, self.seq_half * ch)
        self.deconv = nn.ConvTranspose1d(ch, ch, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.head = nn.Conv1d(ch, vocab, kernel_size=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.ch, self.seq_half)
        x = F.gelu(self.deconv(x))
        x = self.head(x)
        x = x.transpose(1, 2)
        return x
    

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
        pass



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
        x = self.emb(x)           # B,P,base
        x = x.flatten(1)          # B,P*base
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
