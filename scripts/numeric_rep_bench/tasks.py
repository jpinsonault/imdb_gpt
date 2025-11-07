# scripts/numeric_rep_bench/tasks.py

import math
import random
from typing import Callable, Dict, List, Tuple

import torch

from .config import TaskSpec

def _regime_sampler_small(d: int, n: int) -> torch.Tensor:
    return 2.0 * torch.rand(n, d) - 1.0

def _regime_sampler_medium(d: int, n: int) -> torch.Tensor:
    return 20.0 * torch.rand(n, d) - 10.0

def _regime_sampler_large(d: int, n: int) -> torch.Tensor:
    return 2e4 * torch.rand(n, d) - 1e4

def _regime_sampler_mixed(d: int, n: int) -> torch.Tensor:
    x = torch.empty(n, d)
    mask = torch.rand(n, d) < 0.5
    x[mask] = 2.0 * torch.rand(mask.sum()) - 1.0
    x[~mask] = 2e4 * torch.rand((~mask).sum()) - 1e4
    return x

def _regime_sampler_near_zero(d: int, n: int) -> torch.Tensor:
    return 2e-6 * torch.rand(n, d) - 1e-6

REGIME_SAMPLERS: Dict[str, Callable[[int, int], torch.Tensor]] = {
    "small": _regime_sampler_small,
    "medium": _regime_sampler_medium,
    "large": _regime_sampler_large,
    "mixed": _regime_sampler_mixed,
    "near_zero": _regime_sampler_near_zero,
}

REGIMES_ORDERED: List[str] = [
    "small",
    "medium",
    "large",
    "mixed",
    "near_zero",
]

def sample_task_regime(
    task: TaskSpec,
    regime: str,
    n: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if task.name == "parity":
        x = task.sampler(n, device)
        y = task.target_fn(x)
        return x.to(device), y.to(device)

    if regime not in REGIME_SAMPLERS:
        raise ValueError(f"unknown regime: {regime}")

    sampler = REGIME_SAMPLERS[regime]
    x = sampler(task.arity, n).to(device)
    y = task.target_fn(x)
    return x.to(device), y.to(device)
