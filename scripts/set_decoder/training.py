# scripts/set_decoder/training.py

import sqlite3
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ProjectConfig
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.autoencoder.fields import (
    ScalarField,
    BooleanField,
    MultiCategoryField,
    SingleCategoryField,
    TextField,
    NumericDigitCategoryField,
)
from scripts.autoencoder.run_logger import build_run_logger
from scripts.set_decoder.model import SetDecoder
from scripts.set_decoder.data import TitlePeopleSetDataset, collate_set_decoder


def _hungarian(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = cost.detach().cpu().clone()
    n_rows, n_cols = c.shape
    n = max(n_rows, n_cols)
    if n_rows != n_cols:
        pad = torch.zeros((n, n), dtype=c.dtype)
        pad[:n_rows, :n_cols] = c
        c = pad

    u = torch.zeros(n)
    v = torch.zeros(n)
    p = torch.full((n,), -1, dtype=torch.long)
    way = torch.full((n,), -1, dtype=torch.long)

    for i in range(1, n):
        p[0] = i
        j0 = 0
        minv = torch.full((n,), float("inf"))
        used = torch.zeros(n, dtype=torch.bool)
        while True:
            used[j0] = True
            i0 = p[j0].item()
            delta = float("inf")
            j1 = 0
            for j in range(1, n):
                if not used[j]:
                    cur = c[i0, j].item() - u[i0].item() - v[j].item()
                    if cur < minv[j].item():
                        minv[j] = cur
                        way[j] = j0
                    if minv[j].item() < delta:
                        delta = minv[j].item()
                        j1 = j
            for j in range(n):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == -1:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_of = torch.full((n,), -1, dtype=torch.long)
    for j in range(1, n):
        i = p[j]
        if i >= 0:
            row_of[i] = j

    rows = []
    cols = []
    for i in range(n_rows):
        j = row_of[i].item()
        if 0 <= j < n_cols:
            rows.append(i)
            cols.append(j)

    if not rows:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        )

    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


def _field_loss_one(field, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if isinstance(field, TextField):
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
        B = pred.shape[0]
        L = pred.shape[1]
        V = pred.shape[2]
        pad_id = int(getattr(field, "pad_token_id", 0) or 0)
        loss = F.cross_entropy(
            pred.view(B * L, V),
            tgt.view(B * L),
            ignore_index=pad_id,
            reduction="none",
        )
        loss = loss.view(B, L)
        mask = (tgt.view(B, L) != pad_id).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (loss * mask).sum(dim=1) / denom

    if isinstance(field, NumericDigitCategoryField):
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
        B = pred.shape[0]
        P = pred.shape[1]
        V = pred.shape[2]
        mask_id = int(field.mask_index) if field.mask_index is not None else -1000
        loss = F.cross_entropy(
            pred.view(B * P, V),
            tgt.view(B * P).long(),
            ignore_index=mask_id,
            reduction="none",
        )
        loss = loss.view(B, P)
        return loss.mean(dim=1)

    if isinstance(field, ScalarField):
        diff = pred.view(-1) - tgt.view(-1)
        return (diff * diff).mean().unsqueeze(0)

    if isinstance(field, BooleanField):
        if getattr(field, "use_bce_loss", True):
            loss = F.binary_cross_entropy_with_logits(
                pred.view(-1),
                tgt.view(-1),
                reduction="none",
            ).mean()
            return loss.unsqueeze(0)
        diff = torch.tanh(pred.view(-1)) - tgt.view(-1)
        return (diff * diff).mean().unsqueeze(0)

    if isinstance(field, MultiCategoryField):
        loss = F.binary_cross_entropy_with_logits(
            pred.view(-1),
            tgt.view(-1),
            reduction="none",
        ).mean()
        return loss.unsqueeze(0)

    if isinstance(field, SingleCategoryField):
        logits = pred.view(1, -1)
        target = tgt.view(-1).long()
        if target.numel() == 0:
            return torch.zeros(1, device=pred.device)
        loss = F.cross_entropy(logits, target, reduction="none").mean()
        return loss.unsqueeze(0)

    diff = (pred - tgt).view(-1)
    return (diff * diff).mean().unsqueeze(0)


def _compute_cost_matrices(
    people_ae,
    z_slots: torch.Tensor,
    Z_gt: torch.Tensor,
    Y_gt_fields: List[torch.Tensor],
    mask: torch.Tensor,
    w_latent: float,
    w_recon: float,
):
    B, N, D = z_slots.shape

    dec_device = next(people_ae.decoder.parameters()).device
    slot_device = z_slots.device

    z_flat = z_slots.view(B * N, D).to(dec_device)
    dec_out = people_ae.decoder(z_flat)

    dec_per_field: List[torch.Tensor] = []
    for pred in dec_out:
        shape = pred.shape[1:]
        dec_per_field.append(pred.view(B, N, *shape))

    C_lat_list: List[torch.Tensor | None] = []
    C_rec_list: List[torch.Tensor | None] = []
    C_match_list: List[torch.Tensor | None] = []

    Z_gt = Z_gt.to(slot_device)
    Y_gt_fields = [y.to(dec_device) for y in Y_gt_fields]

    for b in range(B):
        k_b = int(mask[b].sum().item())
        if k_b == 0:
            C_lat_list.append(None)
            C_rec_list.append(None)
            C_match_list.append(None)
            continue

        z_pred_b = z_slots[b]
        z_gt_b = Z_gt[b, :k_b]

        diff = z_pred_b.unsqueeze(1) - z_gt_b.unsqueeze(0)
        C_lat = (diff * diff).sum(dim=-1)

        C_rec = torch.zeros_like(C_lat, device=dec_device)

        for fi, field in enumerate(people_ae.fields):
            pred_f = dec_per_field[fi][b]
            tgt_f_all = Y_gt_fields[fi][b]
            for i in range(N):
                for j in range(k_b):
                    loss_ij = _field_loss_one(
                        field,
                        pred_f[i],
                        tgt_f_all[j],
                    )[0]
                    C_rec[i, j] += loss_ij.to(C_rec.dtype)

        C_lat = C_lat.to(dec_device)

        C_match = (w_latent * C_lat + w_recon * C_rec).detach()

        C_lat_list.append(C_lat.to(slot_device))
        C_rec_list.append(C_rec.to(slot_device))
        C_match_list.append(C_match)

    return C_match_list, C_lat_list, C_rec_list


class SetReconstructionLogger:
    def __init__(
        self,
        model: SetDecoder,
        movie_ae,
        people_ae,
        db_path: str,
        num_slots: int,
        interval_steps: int = 200,
        num_samples: int = 4,
        table_width: int = 48,
    ):
        self.model = model
        self.movie_ae = movie_ae
        self.people_ae = people_ae
        self.db_path = db_path
        self.num_slots = int(num_slots)
        self.interval = int(interval_steps)
        self.num_samples = int(num_samples)
        self.table_width = int(table_width)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._cur = self._conn.cursor()

    def _movie_title(self, tconst: str) -> str:
        r = self._cur.execute(
            "SELECT primaryTitle,startYear FROM titles WHERE tconst = ? LIMIT 1",
            (tconst,),
        ).fetchone()
        if r is None:
            return tconst
        title = str(r[0] or "")
        year = str(r[1] or "")
        return f"{title} ({year})" if year else title

    def step(
        self,
        global_step: int,
        sample_tconsts: List[str],
        z_movies: torch.Tensor,
        mask: torch.Tensor,
        run_logger,
    ):
        if global_step % self.interval != 0:
            return

        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            z_movies_d = z_movies.to(device)
            z_slots, presence_logits = self.model(z_movies_d)
            probs = torch.sigmoid(presence_logits)

        lines: List[str] = []
        b = min(len(sample_tconsts), self.num_samples)

        for i in range(b):
            tconst = sample_tconsts[i]
            title = self._movie_title(tconst)
            lines.append("=" * self.table_width)
            lines.append(title)
            lines.append("-" * self.table_width)

            k_b = int(mask[i].sum().item())
            lines.append(f"true_count={k_b}")

            lines.append("predicted_slots:")
            prob_i = probs[i].detach().cpu()
            z_i = z_slots[i].detach().cpu()
            order = torch.argsort(prob_i, descending=True)
            for rank, slot in enumerate(order[: self.num_slots]):
                p = float(prob_i[slot].item())
                lines.append(
                    f"  slot {rank}: p={p:.3f} norm={z_i[slot].norm().item():.3f}"
                )

        text = "\n".join(lines)
        if run_logger and hasattr(run_logger, "add_text"):
            run_logger.add_text("set_decoder/recon", text, global_step)
        else:
            print(text)

        self.model.train()


def build_set_decoder_trainer(
    cfg: ProjectConfig,
    db_path: str,
):
    mov_ae, per_ae = _load_frozen_autoencoders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mov_ae.encoder.to("cpu").eval()
    per_ae.encoder.to("cpu").eval()
    mov_ae.decoder.to(device).eval()
    per_ae.decoder.to(device).eval()

    for p in mov_ae.encoder.parameters():
        p.requires_grad_(False)
    for p in per_ae.encoder.parameters():
        p.requires_grad_(False)
    for p in mov_ae.decoder.parameters():
        p.requires_grad_(False)
    for p in per_ae.decoder.parameters():
        p.requires_grad_(False)

    num_slots = int(getattr(cfg, "set_decoder_slots", 10))
    movie_limit = getattr(cfg, "set_decoder_movie_limit", None)
    if movie_limit is not None:
        movie_limit = int(movie_limit)
        if movie_limit <= 0:
            movie_limit = None

    ds = TitlePeopleSetDataset(
        db_path=db_path,
        movie_ae=mov_ae,
        people_ae=per_ae,
        num_slots=num_slots,
        movie_limit=movie_limit,
    )

    from torch.utils.data import DataLoader

    num_workers = int(getattr(cfg, "num_workers", 0) or 0)
    prefetch_factor = None if num_workers == 0 else max(
        1, int(getattr(cfg, "prefetch_factor", 2) or 2)
    )
    pin = bool(torch.cuda.is_available())

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_set_decoder,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
    )

    latent_dim = int(getattr(mov_ae, "latent_dim", cfg.latent_dim))

    model = SetDecoder(
        latent_dim=latent_dim,
        num_slots=num_slots,
        hidden_mult=float(getattr(cfg, "set_decoder_hidden_mult", 2.0)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(cfg, "set_decoder_lr", 3e-4)),
        weight_decay=float(getattr(cfg, "set_decoder_weight_decay", 1e-2)),
    )

    run_logger = build_run_logger(cfg)

    recon_logger = SetReconstructionLogger(
        model=model,
        movie_ae=mov_ae,
        people_ae=per_ae,
        db_path=db_path,
        num_slots=num_slots,
        interval_steps=int(
            getattr(cfg, "set_decoder_callback_interval", 500)
        ),
        num_samples=int(
            getattr(cfg, "set_decoder_recon_samples", 3)
        ),
        table_width=int(
            getattr(cfg, "set_decoder_table_width", 60)
        ),
    )

    loss_cfg = {
        "w_latent": float(getattr(cfg, "set_decoder_w_latent", 1.0)),
        "w_recon": float(getattr(cfg, "set_decoder_w_recon", 1.0)),
        "w_presence": float(getattr(cfg, "set_decoder_w_presence", 1.0)),
        "w_null": float(getattr(cfg, "set_decoder_w_null", 0.1)),
    }

    return model, opt, loader, mov_ae, per_ae, run_logger, recon_logger, loss_cfg