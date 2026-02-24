from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from scripts.autoencoder.fields import (
    ScalarField,
    NumericDigitCategoryField,
    Scaling as ScalarFieldScaling,
)
from .config import TaskSpec, RepConfig


# ---- Scalar encoding ----

def build_scalar_fields(task: TaskSpec, cfg: RepConfig, train_x: torch.Tensor) -> List[ScalarField]:
    fields: List[ScalarField] = []
    scaling = cfg.params["scaling"]
    for i in range(task.arity):
        f = ScalarField(name=f"x{i}", scaling=scaling, optional=False)
        for v in train_x[:, i].detach().cpu().tolist():
            f.accumulate_stats(v)
        f.finalize_stats()
        fields.append(f)
    return fields


def encode_scalar_batch(fields: List[ScalarField], x: torch.Tensor) -> torch.Tensor:
    cols = []
    x_cpu = x.detach().cpu()
    for i, f in enumerate(fields):
        vals = [f.transform(float(v)).item() for v in x_cpu[:, i].tolist()]
        cols.append(torch.tensor(vals, dtype=torch.float32).unsqueeze(-1))
    return torch.cat(cols, dim=-1)


# ---- Digit encoding ----

def build_digit_fields(task: TaskSpec, cfg: RepConfig, train_x: torch.Tensor) -> List[NumericDigitCategoryField]:
    base = int(cfg.params["base"])
    frac = int(cfg.params["fraction_digits"])
    fields: List[NumericDigitCategoryField] = []
    for i in range(task.arity):
        f = NumericDigitCategoryField(
            name=f"x{i}",
            base=base,
            fraction_digits=frac,
            optional=False,
        )
        for v in train_x[:, i].detach().cpu().tolist():
            f.accumulate_stats(v)
        f.finalize_stats()
        fields.append(f)
    return fields


def encode_digit_batch(fields: List[NumericDigitCategoryField], x: torch.Tensor) -> torch.Tensor:
    """
    Returns LongTensor of shape [batch, arity, positions].
    """
    x_cpu = x.detach().cpu()
    all_cols = []
    for i, f in enumerate(fields):
        rows = []
        for v in x_cpu[:, i].tolist():
            d = f.transform(float(v))
            if not isinstance(d, torch.Tensor):
                d = torch.tensor(d, dtype=torch.long)
            rows.append(d.unsqueeze(0))
        col = torch.cat(rows, dim=0)  # [B, positions]
        all_cols.append(col.unsqueeze(1))  # [B, 1, positions]
    return torch.cat(all_cols, dim=1).long()  # [B, arity, positions]


# ---- Adapters & head ----

class ScalarAdapter(nn.Module):
    def __init__(self, in_dim: int, rep_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, rep_dim),
            nn.GELU(),
            nn.LayerNorm(rep_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DigitAdapter(nn.Module):
    def __init__(self, arity: int, positions: int, base: int, rep_dim: int, emb_dim: int = 8):
        super().__init__()
        # +1 to allow e.g. PAD or sign bucket robustly
        self.embedding = nn.Embedding(base + 1, emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(arity * positions * emb_dim, rep_dim),
            nn.GELU(),
            nn.LayerNorm(rep_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, arity, positions]
        emb = self.embedding(x)          # [B, arity, positions, emb_dim]
        flat = emb.reshape(emb.size(0), -1)
        return self.proj(flat)


class TaskHead(nn.Module):
    def __init__(self, rep_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---- Full model ----

class NumericRepModel(nn.Module):
    def __init__(self, task: TaskSpec, rep: RepConfig, rep_dim: int, device: torch.device):
        super().__init__()
        self.task = task
        self.rep = rep
        self.rep_dim = rep_dim
        self.device = device

        self.scalar_fields: Optional[List[ScalarField]] = None
        self.digit_fields: Optional[List[NumericDigitCategoryField]] = None
        self.adapter: Optional[nn.Module] = None
        self.head = TaskHead(rep_dim=rep_dim, out_dim=task.out_dim).to(device)

    # Representation init (fit fields from train_x)

    def init_from_train(self, train_x: torch.Tensor):
        if self.rep.kind == "scalar":
            self.scalar_fields = build_scalar_fields(self.task, self.rep, train_x)
            self.adapter = ScalarAdapter(self.task.arity, self.rep_dim).to(self.device)

        elif self.rep.kind == "digits":
            base = int(self.rep.params["base"])
            self.digit_fields = build_digit_fields(self.task, self.rep, train_x)

            # Infer number of positions from one example
            sample_val = float(train_x[0, 0].item())
            sample_digits = self.digit_fields[0].transform(sample_val)
            if not isinstance(sample_digits, torch.Tensor):
                sample_digits = torch.tensor(sample_digits, dtype=torch.long)
            positions = int(sample_digits.numel())

            self.adapter = DigitAdapter(
                arity=self.task.arity,
                positions=positions,
                base=base,
                rep_dim=self.rep_dim,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown rep kind: {self.rep.kind}")

    # Encoding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.rep.kind == "scalar":
            if self.scalar_fields is None:
                raise RuntimeError("Scalar fields not initialized")
            enc = encode_scalar_batch(self.scalar_fields, x).to(self.device)
        else:
            if self.digit_fields is None:
                raise RuntimeError("Digit fields not initialized")
            digits = encode_digit_batch(self.digit_fields, x).to(self.device)
            enc = digits
        return self.adapter(enc)

    # Forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep_vec = self.encode(x)
        return self.head(rep_vec)
