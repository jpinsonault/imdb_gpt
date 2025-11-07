from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Literal

import torch

from scripts.autoencoder.fields import (
    Scaling as ScalarFieldScaling,
)

# Core specs

@dataclass
class TaskSpec:
    name: str
    arity: int
    kind: Literal["regression", "classification"]
    sampler: Callable[[int, torch.device], torch.Tensor]
    target_fn: Callable[[torch.Tensor], torch.Tensor]
    out_dim: int
    display_name: Optional[str] = None

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.name


@dataclass
class RepConfig:
    name: str
    kind: Literal["scalar", "digits"]
    params: Dict


@dataclass
class TrainConfig:
    train_size: int = 8_000
    val_size: int = 1_000
    test_size: int = 2_000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    rep_dim: int = 64
    seed: int = 1337
    n_seeds: int = 1


# ---- Tasks ----

def _uniform_xy(n, device, low=-100.0, high=100.0, arity=2):
    return (high - low) * torch.rand(n, arity, device=device) + low


def _make_default_tasks() -> List[TaskSpec]:
    def add_sampler(n, device):
        return _uniform_xy(n, device, -100.0, 100.0, 2)

    def add_target(x):
        return x.sum(dim=-1, keepdim=True)

    def mul_sampler(n, device):
        return _uniform_xy(n, device, -20.0, 20.0, 2)

    def mul_target(x):
        return (x[:, 0] * x[:, 1]).unsqueeze(-1)

    def sin_sampler(n, device):
        return 20.0 * torch.rand(n, 1, device=device) - 10.0

    def sin_target(x):
        return torch.sin(x)

    def gt_sampler(n, device):
        return _uniform_xy(n, device, -100.0, 100.0, 2)

    def gt_target(x):
        return (x[:, 0] > x[:, 1]).long()

    def parity_sampler(n, device):
        vals = torch.randint(0, 1024, (n, 1), device=device).float()
        return vals

    def parity_target(x):
        v = x.long().squeeze(-1)
        parity = torch.zeros_like(v)
        tmp = v.clone()
        while True:
            parity ^= (tmp & 1)
            tmp >>= 1
            if torch.all(tmp == 0):
                break
        return parity

    return [
        TaskSpec(
            name="add",
            arity=2,
            kind="regression",
            sampler=add_sampler,
            target_fn=add_target,
            out_dim=1,
            display_name="Addition: x + y",
        ),
        TaskSpec(
            name="mul",
            arity=2,
            kind="regression",
            sampler=mul_sampler,
            target_fn=mul_target,
            out_dim=1,
            display_name="Multiplication: x * y",
        ),
        TaskSpec(
            name="sin",
            arity=1,
            kind="regression",
            sampler=sin_sampler,
            target_fn=sin_target,
            out_dim=1,
            display_name="Sine: sin(x)",
        ),
        TaskSpec(
            name="gt",
            arity=2,
            kind="classification",
            sampler=gt_sampler,
            target_fn=gt_target,
            out_dim=2,
            display_name="Comparison: x > y",
        ),
        TaskSpec(
            name="parity",
            arity=1,
            kind="classification",
            sampler=parity_sampler,
            target_fn=parity_target,
            out_dim=2,
            display_name="Integer parity",
        ),
    ]


def _make_full_tasks() -> List[TaskSpec]:
    # Placeholder for richer suites if you expand later
    return _make_default_tasks()


def get_tasks(full: bool) -> List[TaskSpec]:
    return _make_full_tasks() if full else _make_default_tasks()


# ---- Representations ----

def _make_default_reps() -> List[RepConfig]:
    return [
        RepConfig(
            name="scalar_standardize",
            kind="scalar",
            params={"scaling": ScalarFieldScaling.STANDARDIZE},
        ),
        RepConfig(
            name="scalar_minmax",
            kind="scalar",
            params={"scaling": ScalarFieldScaling.NORMALIZE},
        ),
        RepConfig(
            name="digits_b10_f0",
            kind="digits",
            params={"base": 10, "fraction_digits": 0},
        ),
        RepConfig(
            name="digits_b10_f2",
            kind="digits",
            params={"base": 10, "fraction_digits": 2},
        ),
    ]


def _make_full_reps() -> List[RepConfig]:
    reps: List[RepConfig] = [
        RepConfig("scalar_none", "scalar", {"scaling": ScalarFieldScaling.NONE}),
        RepConfig("scalar_standardize", "scalar", {"scaling": ScalarFieldScaling.STANDARDIZE}),
        RepConfig("scalar_minmax", "scalar", {"scaling": ScalarFieldScaling.NORMALIZE}),
        RepConfig("scalar_log", "scalar", {"scaling": ScalarFieldScaling.LOG}),
    ]
    for base in (2, 4, 5, 10, 16):
        for frac in (0, 2):
            reps.append(
                RepConfig(
                    name=f"digits_b{base}_f{frac}",
                    kind="digits",
                    params={"base": base, "fraction_digits": frac},
                )
            )
    return reps


def get_reps(full: bool) -> List[RepConfig]:
    return _make_full_reps() if full else _make_default_reps()
