"""Tests for training-loop callbacks: FocalLoss, LR scheduler, GracefulStopper,
and the global→local target-construction logic from compute_side_loss."""

import math
import signal

import pytest
import torch
import torch.nn.functional as F

from scripts.train_simple_set import FocalLoss, GracefulStopper, make_lr_scheduler


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class TestFocalLoss:
    def test_gamma_zero_matches_weighted_bce(self):
        """With gamma=0 focal loss reduces to alpha-weighted BCE."""
        alpha = 0.25
        fl = FocalLoss(alpha=alpha, gamma=0.0, reduction="mean")

        logits = torch.randn(8, 10)
        targets = torch.zeros(8, 10)
        targets[:, :3] = 1.0

        result = fl(logits, targets)
        expected = alpha * F.binary_cross_entropy_with_logits(logits, targets)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_confident_prediction_low_loss(self):
        """A very confident correct prediction should have near-zero focal loss."""
        fl = FocalLoss(alpha=0.25, gamma=2.0)

        # Large positive logit for target=1 → high pt → small (1-pt)^gamma
        logits = torch.tensor([[10.0]])
        targets = torch.tensor([[1.0]])
        loss_high_conf = fl(logits, targets)

        logits_low = torch.tensor([[0.0]])
        loss_low_conf = fl(logits_low, targets)

        assert loss_high_conf < loss_low_conf

    def test_reduction_none_shape(self):
        fl = FocalLoss(alpha=1.0, gamma=2.0, reduction="none")
        logits = torch.randn(4, 5)
        targets = torch.zeros(4, 5)
        result = fl(logits, targets)
        assert result.shape == (4, 5)

    def test_reduction_sum_gt_mean(self):
        fl_mean = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
        fl_sum = FocalLoss(alpha=1.0, gamma=2.0, reduction="sum")
        logits = torch.randn(4, 5)
        targets = torch.zeros(4, 5)
        assert fl_sum(logits, targets) > fl_mean(logits, targets)


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

class TestLRScheduler:
    def _get_lr(self, scheduler):
        return scheduler.get_last_lr()[0]

    def test_warmup_starts_low(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = make_lr_scheduler(opt, total_steps=1000, schedule="cosine",
                                  warmup_steps=100, warmup_ratio=0.0, min_factor=0.05)
        # Step 0: LR = 1/100 = 0.01
        assert self._get_lr(sched) == pytest.approx(1.0 / 100)

    def test_warmup_reaches_full_lr(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = make_lr_scheduler(opt, total_steps=1000, schedule="cosine",
                                  warmup_steps=100, warmup_ratio=0.0, min_factor=0.05)
        for _ in range(99):
            sched.step()
        # At step 99 (100th step): LR = 100/100 = 1.0
        assert self._get_lr(sched) == pytest.approx(1.0)

    def test_cosine_midpoint(self):
        """At the midpoint of the decay phase, cosine should give ~(1 + min_factor)/2."""
        min_factor = 0.05
        total = 1000
        warmup = 100
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = make_lr_scheduler(opt, total_steps=total, schedule="cosine",
                                  warmup_steps=warmup, warmup_ratio=0.0, min_factor=min_factor)
        midpoint = warmup + (total - warmup) // 2
        for _ in range(midpoint):
            sched.step()

        expected = min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * 0.5))
        assert self._get_lr(sched) == pytest.approx(expected, abs=0.01)

    def test_end_hits_min_factor(self):
        min_factor = 0.05
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = make_lr_scheduler(opt, total_steps=200, schedule="cosine",
                                  warmup_steps=0, warmup_ratio=0.0, min_factor=min_factor)
        for _ in range(200):
            sched.step()
        assert self._get_lr(sched) == pytest.approx(min_factor, abs=0.01)

    def test_linear_schedule_fallback(self):
        """Non-cosine schedule should use linear decay."""
        min_factor = 0.1
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = make_lr_scheduler(opt, total_steps=100, schedule="linear",
                                  warmup_steps=0, warmup_ratio=0.0, min_factor=min_factor)
        # Step to midpoint
        for _ in range(50):
            sched.step()
        # Linear: max(min_factor, 1 - (1 - min_factor) * 0.5) = max(0.1, 0.55) = 0.55
        assert self._get_lr(sched) == pytest.approx(0.55, abs=0.02)

    def test_none_total_steps_returns_none(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        result = make_lr_scheduler(opt, total_steps=None, schedule="cosine",
                                   warmup_steps=0, warmup_ratio=0.0, min_factor=0.05)
        assert result is None


# ---------------------------------------------------------------------------
# GracefulStopper
# ---------------------------------------------------------------------------

def test_graceful_stopper():
    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    try:
        stopper = GracefulStopper()
        assert not stopper.stop
        stopper._handler(signal.SIGINT, None)
        assert stopper.stop
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


# ---------------------------------------------------------------------------
# Target construction from global→local mapping
# (replicates the core logic inside compute_side_loss)
# ---------------------------------------------------------------------------

def _build_targets(padded, mapping, vocab_size):
    """Extract the target-building logic from compute_side_loss for testing."""
    batch_size = padded.shape[0]
    targets = torch.zeros(batch_size, vocab_size)

    mask = padded != -1
    rows, cols = torch.nonzero(mask, as_tuple=True)

    if rows.numel() > 0:
        glob = padded[rows, cols].long()
        loc = mapping[glob]
        valid = loc != -1
        if valid.any():
            rows_v, loc_v = rows[valid], loc[valid]
            targets[rows_v, loc_v] = 1.0

    return targets


class TestTargetConstruction:
    def test_basic_mapping(self):
        """Global indices correctly mapped to local positions in target matrix."""
        # 6 global items, 4 mapped to local: globals 1→0, 2→1, 4→2, 5→3
        mapping = torch.tensor([-1, 0, 1, -1, 2, 3])

        # Batch of 2 samples, up to 3 targets, -1 = padding
        padded = torch.tensor([
            [1, 4, -1],  # sample 0: globals 1, 4
            [2, 5, 1],   # sample 1: globals 2, 5, 1
        ])

        targets = _build_targets(padded, mapping, vocab_size=4)

        expected = torch.tensor([
            [1.0, 0.0, 1.0, 0.0],  # locals 0 and 2
            [1.0, 1.0, 0.0, 1.0],  # locals 0, 1, and 3
        ])
        assert torch.equal(targets, expected)

    def test_all_padding(self):
        """All-padding row produces zero target."""
        mapping = torch.tensor([0, 1, 2])
        padded = torch.tensor([[-1, -1, -1]])
        targets = _build_targets(padded, mapping, vocab_size=3)
        assert targets.sum() == 0.0

    def test_unmapped_global_ignored(self):
        """Global indices that map to -1 (unmapped) are silently skipped."""
        mapping = torch.tensor([-1, -1, 0])  # only global 2 → local 0
        padded = torch.tensor([
            [0, 1, 2],  # globals 0 and 1 unmapped, only 2 maps
        ])
        targets = _build_targets(padded, mapping, vocab_size=1)
        expected = torch.tensor([[1.0]])
        assert torch.equal(targets, expected)

    def test_duplicate_targets_no_error(self):
        """Same global appearing twice in one sample still sets target to 1."""
        mapping = torch.tensor([0, 1])
        padded = torch.tensor([[0, 0, 1]])  # global 0 twice
        targets = _build_targets(padded, mapping, vocab_size=2)
        expected = torch.tensor([[1.0, 1.0]])
        assert torch.equal(targets, expected)
