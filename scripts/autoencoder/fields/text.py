from typing import List, Optional
import numpy as np
from scripts.autoencoder.fields.constants import SPECIAL_END, SPECIAL_PAD, SPECIAL_SEP, SPECIAL_START, SPECIAL_UNK
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseField
from ..character_tokenizer import CharacterTokenizer

class TextField(BaseField):
    def __init__(
        self,
        name: str,
        max_length: Optional[int] = None,
        downsample_steps: int = 1,
        base_size: int = 48,
        num_blocks_per_step: List[int] = [2],
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
            raise ValueError("TextField stats not finalized")
        return (self.max_length,)

    def _get_output_shape(self):
        if self.max_length is None:
            raise ValueError("TextField stats not finalized")
        return (self.max_length,)

    def _accumulate_stats(self, raw_value):
        if raw_value is not None:
            s = str(raw_value)
            if s:
                self.texts.append(s)

    def _finalize_stats(self):
        special_tokens = [SPECIAL_UNK, SPECIAL_PAD, SPECIAL_START, SPECIAL_END, SPECIAL_SEP]
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

        effective_steps = 1
        multiple = 2 ** effective_steps
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
            raise RuntimeError("TextField stats not finalized")
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
        return


class _TextEncoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, ch: int, latent_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, ch)
        self.conv = nn.Conv1d(ch, ch * 2, kernel_size=5, stride=2, padding=2)
        self.out = nn.Linear((max_len // 2) * (ch * 2), latent_dim)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = F.gelu(self.conv(x))
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
