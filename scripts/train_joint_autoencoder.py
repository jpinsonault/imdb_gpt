# scripts/train_joint_autoencoder.py

from __future__ import annotations
import argparse
import logging
import math
import signal
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ProjectConfig, project_config
from scripts.autoencoder.mapping_samplers import LossLedger
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.joint_edge_sampler import make_edge_sampler, EdgeEpochDataset
from scripts.autoencoder.print_model import print_model_summary
from scripts.autoencoder.run_logger import build_run_logger
from scripts.autoencoder.training_callbacks.training_callbacks import JointReconstructionLogger
from scripts.autoencoder.fields import (
    TextField,
    MultiCategoryField,
    BooleanField,
    ScalarField,
    SingleCategoryField,
    NumericDigitCategoryField,
)
from scripts.autoencoder.share_policy import SharePolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


def info_nce_components(movie_z: torch.Tensor, person_z: torch.Tensor, temperature: float):
    logits = cosine_similarity(movie_z, person_z) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    row_ce = F.cross_entropy(logits, labels, reduction="none")
    col_ce = F.cross_entropy(logits.t(), labels, reduction="none")
    nce_per = 0.5 * (row_ce + col_ce)
    nce_mean = nce_per.mean()
    return nce_mean, nce_per


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int | None,
    schedule: str,
    warmup_steps: int,
    warmup_ratio: float,
    min_factor: float,
):
    if total_steps is None:
        return None

    total_steps = max(1, int(total_steps))
    schedule = (schedule or "").lower()
    if schedule not in ("cosine", "linear"):
        return None

    base_warmup = max(0, int(warmup_steps))
    ratio = float(warmup_ratio)
    if ratio > 0.0:
        frac_warmup = int(total_steps * ratio)
    else:
        frac_warmup = 0

    w_steps = max(base_warmup, frac_warmup)
    w_steps = min(w_steps, total_steps - 1) if total_steps > 1 else 0
    min_factor = float(min_factor)

    def cosine_lambda(step: int) -> float:
        s = int(step)
        if w_steps > 0 and s < w_steps:
            return float(s + 1) / float(w_steps)
        if s >= total_steps:
            return min_factor
        if w_steps >= total_steps:
            return min_factor
        t = float(s - w_steps) / float(total_steps - w_steps)
        t = max(0.0, min(1.0, t))
        decay = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_factor + (1.0 - min_factor) * decay

    def linear_lambda(step: int) -> float:
        s = int(step)
        if w_steps > 0 and s < w_steps:
            return float(s + 1) / float(w_steps)
        if s >= total_steps:
            return min_factor
        if w_steps >= total_steps:
            return min_factor
        t = float(s - w_steps) / float(total_steps - w_steps)
        t = max(0.0, min(1.0, t))
        return max(min_factor, 1.0 - (1.0 - min_factor) * t)

    if schedule == "cosine":
        lr_lambda = cosine_lambda
    else:
        lr_lambda = linear_lambda

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class _TypeFiLM(nn.Module):
    def __init__(self, type_dim: int, latent_dim: int, hidden_mult: float = 1.0):
        super().__init__()
        self.type_dim = int(type_dim)
        self.latent_dim = int(latent_dim)
        h = max(self.latent_dim, int(self.latent_dim * hidden_mult))
        self.net = nn.Sequential(
            nn.Linear(self.type_dim, h),
            nn.GELU(),
            nn.Linear(h, 2 * self.latent_dim),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, z: torch.Tensor, type_onehot: torch.Tensor) -> torch.Tensor:
        params = self.net(type_onehot)
        gamma, beta = params.chunk(2, dim=-1)
        return z * (1.0 + gamma) + beta


class _TypedEncoder(nn.Module):
    def __init__(
        self,
        base_encoder: nn.Module,
        film: _TypeFiLM,
        type_index: int,
        type_dim: int,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.film = film
        self.type_index = int(type_index)
        self.type_dim = int(type_dim)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        z = self.base_encoder(xs)
        if not isinstance(z, torch.Tensor):
            raise RuntimeError("Base encoder must return a Tensor")
        b = z.size(0)
        device = z.device
        type_onehot = z.new_zeros(b, self.type_dim)
        type_onehot[:, self.type_index] = 1.0
        return self.film(z, type_onehot)


class JointAutoencoder(nn.Module):
    def __init__(self, movie_ae: TitlesAutoencoder, person_ae: PeopleAutoencoder):
        super().__init__()
        self.movie_ae = movie_ae
        self.person_ae = person_ae

        d = int(getattr(movie_ae, "latent_dim", 64))
        self.latent_dim = d
        self.type_dim = 2

        self.film = _TypeFiLM(
            type_dim=self.type_dim,
            latent_dim=self.latent_dim,
            hidden_mult=1.0,
        )

        self.type_head = nn.Linear(self.latent_dim, self.type_dim)
        with torch.no_grad():
            self.type_head.weight.zero_()
            self.type_head.bias.zero_()

        self.mov_enc_base = movie_ae.encoder
        self.per_enc_base = person_ae.encoder

        self.mov_enc = _TypedEncoder(
            base_encoder=self.mov_enc_base,
            film=self.film,
            type_index=0,
            type_dim=self.type_dim,
        )
        self.per_enc = _TypedEncoder(
            base_encoder=self.per_enc_base,
            film=self.film,
            type_index=1,
            type_dim=self.type_dim,
        )

        self.mov_dec = movie_ae.decoder
        self.per_dec = person_ae.decoder

        self.movie_ae.encoder = self.mov_enc
        self.person_ae.encoder = self.per_enc

    def forward(
        self,
        movie_in: List[torch.Tensor],
        person_in: List[torch.Tensor],
        movie_type: torch.Tensor | None = None,
        person_type: torch.Tensor | None = None,
    ):
        m_z = self.mov_enc(movie_in)
        p_z = self.per_enc(person_in)

        m_rec = self.mov_dec(m_z)
        p_rec = self.per_dec(p_z)

        return m_z, p_z, m_rec, p_rec


def _reduce_per_sample_mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    diff = pred - tgt
    if diff.dim() > 1:
        return diff.pow(2).reshape(diff.size(0), -1).mean(dim=1)
    return diff.pow(2)


def _per_sample_field_loss(field, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3:
        if tgt.dim() == 3:
            tgt = tgt.argmax(dim=-1)
        if hasattr(field, "tokenizer"):
            B, L, V = pred.shape
            pad_id = int(getattr(field, "pad_token_id", 0) or 0)
            loss_flat = F.cross_entropy(
                pred.reshape(B * L, V),
                tgt.reshape(B * L),
                ignore_index=pad_id,
                reduction="none",
            )
            loss_bt = loss_flat.reshape(B, L)
            mask_tok = (tgt.reshape(B, L) != pad_id).float()
            denom = mask_tok.sum(dim=1).clamp_min(1.0)
            return (loss_bt * mask_tok).sum(dim=1) / denom
        if hasattr(field, "base"):
            B, P, V = pred.shape
            loss_flat = F.cross_entropy(
                pred.reshape(B * P, V),
                tgt.reshape(B * P).long(),
                reduction="none",
            )
            return loss_flat.reshape(B, P).mean(dim=1)

    if isinstance(field, ScalarField):
        return _reduce_per_sample_mse(pred, tgt)

    if isinstance(field, BooleanField):
        if getattr(field, "use_bce_loss", True):
            loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            if loss.dim() > 1:
                loss = loss.reshape(loss.size(0), -1).mean(dim=1)
            return loss
        return _reduce_per_sample_mse(torch.tanh(pred), tgt)

    if isinstance(field, MultiCategoryField):
        loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        if loss.dim() > 1:
            loss = loss.reshape(loss.size(0), -1).mean(dim=1)
        return loss

    if isinstance(field, SingleCategoryField):
        B, C = pred.shape[0], pred.shape[-1]
        t = tgt.long().squeeze(-1)
        return F.cross_entropy(pred.reshape(B, C), t.reshape(B), reduction="none")

    return _reduce_per_sample_mse(pred, tgt)


def build_joint_trainer(
    config: ProjectConfig,
    warm: bool,
    db_path: Path,
):
    fresh_cfg = ProjectConfig(**vars(config))
    fresh_cfg.use_cache = False
    fresh_cfg.refresh_cache = True

    movie_ae = TitlesAutoencoder(fresh_cfg)
    people_ae = PeopleAutoencoder(fresh_cfg)

    if warm:
        raise NotImplementedError("Warm start is not implemented yet.")
    else:
        movie_ae.accumulate_stats()
        people_ae.accumulate_stats()

        policy = (
            SharePolicy()
            .group("text_all", TextField)
            .group("year_digits", NumericDigitCategoryField)
        )
        policy.apply(movie_ae, people_ae)

        movie_ae.finalize_stats()
        people_ae.finalize_stats()

        movie_ae.build_autoencoder()
        people_ae.build_autoencoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    movie_ae.model.to(device)
    people_ae.model.to(device)

    loss_logger = LossLedger()

    edge_gen = make_edge_sampler(
        db_path=str(db_path),
        movie_ae=movie_ae,
        person_ae=people_ae,
        batch_size=config.batch_size,
        refresh_batches=config.refresh_batches,
        boost=config.weak_edge_boost,
        loss_logger=loss_logger,
    )

    num_workers = int(getattr(config, "num_workers", 0) or 0)
    cfg_pf = int(getattr(config, "prefetch_factor", 2) or 0)
    prefetch_factor = None if num_workers == 0 else max(1, cfg_pf)
    pin = bool(torch.cuda.is_available())

    ds_epoch = EdgeEpochDataset(edge_gen)
    loader = DataLoader(
        ds_epoch,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda batch: _collate_edge(batch),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
    )

    joint = JointAutoencoder(movie_ae, people_ae).to(device)
    total_edges = len(edge_gen.edges)
    logging.info(
        f"joint trainer ready device={device} edges={total_edges} "
        f"bs={config.batch_size} workers={num_workers} mode=epoch"
    )
    return joint, loader, loss_logger, movie_ae, people_ae, total_edges


def _collate_edge(batch):
    ms, ps, eids = zip(*batch)
    m_cols = list(zip(*ms))
    p_cols = list(zip(*ps))
    M = [torch.stack(col, dim=0) for col in m_cols]
    P = [torch.stack(col, dim=0) for col in p_cols]
    e = torch.tensor(eids, dtype=torch.long)
    return M, P, e


def _train_step(
    joint_model: JointAutoencoder,
    M: List[torch.Tensor],
    P: List[torch.Tensor],
    eids: torch.Tensor,
    mov_ae: TitlesAutoencoder,
    per_ae: PeopleAutoencoder,
    opt: torch.optim.Optimizer,
    logger: LossLedger,
    temperature: float,
    nce_weight: float,
    type_loss_weight: float,
    device: torch.device,
):
    m_z, p_z, m_rec, p_rec = joint_model(M, P)

    rec_loss = 0.0
    field_losses_movie: Dict[str, float] = {}
    field_losses_person: Dict[str, float] = {}
    rec_per_sample = torch.zeros(M[0].size(0), device=device)

    for f, pred, tgt in zip(mov_ae.fields, m_rec, M):
        per_s = _per_sample_field_loss(f, pred, tgt) * float(f.weight)
        rec_per_sample = rec_per_sample + per_s
        field_losses_movie[f.name] = float(per_s.mean().detach().cpu().item())
        rec_loss = rec_loss + per_s.mean()

    for f, pred, tgt in zip(per_ae.fields, p_rec, P):
        per_s = _per_sample_field_loss(f, pred, tgt) * float(f.weight)
        rec_per_sample = rec_per_sample + per_s
        field_losses_person[f.name] = float(per_s.mean().detach().cpu().item())
        rec_loss = rec_loss + per_s.mean()

    nce_mean, nce_per = info_nce_components(m_z, p_z, temperature=temperature)

    total_per_sample = rec_per_sample + nce_weight * nce_per

    type_loss = torch.tensor(0.0, device=device)

    if type_loss_weight > 0.0:
        b = m_z.size(0)

        movie_logits = joint_model.type_head(m_z)
        person_logits = joint_model.type_head(p_z)

        movie_labels = torch.zeros(b, dtype=torch.long, device=device)
        person_labels = torch.ones(b, dtype=torch.long, device=device)

        loss_movie = F.cross_entropy(movie_logits, movie_labels)
        loss_person = F.cross_entropy(person_logits, person_labels)

        type_loss = 0.5 * (loss_movie + loss_person)

    total = rec_loss + nce_weight * nce_mean + type_loss_weight * type_loss

    opt.zero_grad()
    total.backward()
    opt.step()

    total_val = float(total.detach().cpu().item())
    rec_val = float(rec_loss.detach().cpu().item())
    nce_val = float(nce_mean.detach().cpu().item())
    type_val = float(type_loss.detach().cpu().item()) if type_loss_weight > 0.0 else 0.0

    batch_min = float(total_per_sample.min().detach().cpu().item())
    batch_max = float(total_per_sample.max().detach().cpu().item())

    for eid, edgeloss in zip(
        eids.detach().cpu().tolist(),
        total_per_sample.detach().cpu().tolist(),
    ):
        logger.add(int(eid), float(edgeloss))

    return (
        total_val,
        rec_val,
        nce_val,
        type_val,
        batch_min,
        batch_max,
        field_losses_movie,
        field_losses_person,
    )


def main(config: ProjectConfig):
    parser = argparse.ArgumentParser(
        description="Train movie-person joint embedding autoencoder"
    )
    parser.add_argument("--warm", action="store_true")
    args = parser.parse_args()

    data_dir = Path(config.data_dir)
    db_path = data_dir / "imdb.db"

    joint_model, loader, logger, mov_ae, per_ae, total_edges = build_joint_trainer(
        config=config,
        warm=args.warm,
        db_path=db_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M_sample, P_sample, _ = next(iter(loader))
    M_sample = [m.to(device) for m in M_sample]
    P_sample = [p.to(device) for p in P_sample]
    print_model_summary(joint_model, [M_sample, P_sample])

    opt = torch.optim.AdamW(
        joint_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = (total_edges + config.batch_size - 1) // config.batch_size
    total_steps = int(config.epochs) * int(max(1, steps_per_epoch))

    sched = make_lr_scheduler(
        optimizer=opt,
        total_steps=total_steps,
        schedule=config.lr_schedule,
        warmup_steps=config.lr_warmup_steps,
        warmup_ratio=config.lr_warmup_ratio,
        min_factor=config.lr_min_factor,
    )

    temperature = config.nce_temp
    nce_weight = config.nce_weight
    type_w = float(getattr(config, "latent_type_loss_weight", 0.0))
    save_interval = config.save_interval
    flush_interval = config.flush_interval
    batch_size = config.batch_size

    run_logger = build_run_logger(config)
    if run_logger.run_dir:
        logging.info(f"tensorboard logdir: {run_logger.run_dir}")

    jr_interval = config.callback_interval
    joint_recon = JointReconstructionLogger(
        joint_model,
        mov_ae,
        per_ae,
        str(db_path),
        interval_steps=jr_interval,
        num_samples=4,
        table_width=38,
    )


    stop_flag = {"stop": False}

    def _handle_sigint(signum, frame):
        stop_flag["stop"] = True
        logging.info("interrupt received; will finish current step and save")

    signal.signal(signal.SIGINT, _handle_sigint)

    if torch.cuda.is_available():
        logging.info(
            f"cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}"
        )

    joint_model.train()
    global_step = 0
    edges_seen = 0
    epochs = int(getattr(config, "epochs", 1) or 1)

    for epoch in range(epochs):
        pbar = tqdm(loader, unit="batch", dynamic_ncols=True)
        pbar.set_description(f"epoch {epoch+1}/{epochs}")
        for M, P, eids in pbar:
            iter_start = time.perf_counter()

            M = [m.to(device, non_blocking=True) for m in M]
            P = [p.to(device, non_blocking=True) for p in P]
            eids = eids.to(device)

            (
                total_val,
                rec_val,
                nce_val,
                type_val,
                batch_min,
                batch_max,
                field_losses_movie,
                field_losses_person,
            ) = _train_step(
                joint_model=joint_model,
                M=M,
                P=P,
                eids=eids,
                mov_ae=mov_ae,
                per_ae=per_ae,
                opt=opt,
                logger=logger,
                temperature=temperature,
                nce_weight=nce_weight,
                type_loss_weight=type_w,
                device=device,
            )

            if sched is not None:
                sched.step()

            iter_time = time.perf_counter() - iter_start

            run_logger.add_scalars(
                total_val,
                rec_val,
                nce_val,
                type_val,
                iter_time,
                opt,
            )
            run_logger.add_field_losses("loss/movie", field_losses_movie)
            run_logger.add_field_losses("loss/person", field_losses_person)
            run_logger.add_extremes(batch_min, batch_max)
            run_logger.tick(
                total_val,
                rec_val,
                nce_val,
                type_val,
                iter_time,
            )

            edges_seen += eids.size(0)
            global_step += 1

            pbar.set_postfix(
                edges=f"{min(edges_seen, total_edges)}/{total_edges}",
                loss=f"{total_val:.4f}",
                rec=f"{rec_val:.4f}",
                nce=f"{nce_val:.4f}",
                cls=f"{type_val:.4f}",
                min=f"{batch_min:.4f}",
                max=f"{batch_max:.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
                it_s=f"{iter_time:.3f}",
            )

            JointReconstructionLogger.on_batch_end(
                joint_recon,
                global_step - 1,
            )

            if (global_step % flush_interval) == 0:
                logger.flush()

            if (global_step % save_interval) == 0:
                mov_ae.save_model()
                per_ae.save_model()
                logging.info(
                    f"checkpoint saved at step {global_step}"
                )

            if stop_flag["stop"]:
                break

        if stop_flag["stop"]:
            break

    logger.flush()
    mov_ae.save_model()
    per_ae.save_model()

    out = Path(project_config.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(joint_model.state_dict(), out / "JointMoviePersonAE_final.pt")
    run_logger.close()


if __name__ == "__main__":
    main(project_config)
