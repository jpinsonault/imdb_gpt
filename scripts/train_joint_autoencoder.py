# scripts/train_joint_autoencoder.py

from __future__ import annotations
import argparse
import logging
import math
import signal
import time
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ProjectConfig, project_config
from scripts.autoencoder.mapping_samplers import LossLedger
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.joint_edge_sampler import EdgeTensorCacheDataset
from scripts.autoencoder.print_model import print_model_summary
from scripts.autoencoder.run_logger import build_run_logger
from scripts.autoencoder.joint_autoencoder.training_callbacks import JointReconstructionLogger
from scripts.autoencoder.fields import (
    TextField,
    MultiCategoryField,
    BooleanField,
    ScalarField,
    SingleCategoryField,
    NumericDigitCategoryField,
)
from scripts.autoencoder.precompute_joint_cache import ensure_joint_tensor_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
    last_epoch: int = -1,
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

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


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
        z = self.film(z, type_onehot)
        z = F.normalize(z, dim=-1)
        return z


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
    movie_ae = TitlesAutoencoder(config)
    people_ae = PeopleAutoencoder(config)

    if warm:
        raise NotImplementedError("Warm start is not implemented yet.")
    else:
        movie_ae.accumulate_stats()
        people_ae.accumulate_stats()

        movie_ae.finalize_stats()
        people_ae.finalize_stats()

        movie_ae.build_autoencoder()
        people_ae.build_autoencoder()

    # -------------------------------------------------------------------------
    # SELF-HEALING CACHE CHECK
    # -------------------------------------------------------------------------
    cache_path = ensure_joint_tensor_cache(
        config,
        db_path,
        movie_ae.fields,
        people_ae.fields,
    )

    must_rebuild = False
    try:
        payload = torch.load(cache_path, map_location="cpu")
        
        # Check movie fields
        if len(payload["movie"]) != len(movie_ae.fields):
            logging.warning("Cache validation: Number of movie fields mismatch.")
            must_rebuild = True
        else:
            for i, f in enumerate(movie_ae.fields):
                # cache tensor shape is (NumEdges, Features...)
                cache_shape = payload["movie"][i].shape[1:]
                if cache_shape != f.input_shape:
                    logging.warning(
                        f"Cache mismatch for field '{f.name}': "
                        f"Cache {cache_shape} != Model {f.input_shape}"
                    )
                    must_rebuild = True
                    break

        # Check person fields
        if not must_rebuild:
            if len(payload["person"]) != len(people_ae.fields):
                logging.warning("Cache validation: Number of person fields mismatch.")
                must_rebuild = True
            else:
                for i, f in enumerate(people_ae.fields):
                    cache_shape = payload["person"][i].shape[1:]
                    if cache_shape != f.input_shape:
                        logging.warning(
                            f"Cache mismatch for field '{f.name}': "
                            f"Cache {cache_shape} != Model {f.input_shape}"
                        )
                        must_rebuild = True
                        break

    except Exception as e:
        logging.warning(f"Cache validation check failed ({e}). Forcing rebuild.")
        must_rebuild = True

    if must_rebuild:
        logging.info("!!! CACHE INCONSISTENCY DETECTED !!!")
        logging.info("Invalidating tensor cache and JSON stats to ensure synchronization.")
        
        if cache_path.exists():
            os.remove(cache_path)
            logging.info(f"Deleted {cache_path}")

        movie_ae._drop_cache()
        people_ae._drop_cache()

        movie_ae = TitlesAutoencoder(config)
        people_ae = PeopleAutoencoder(config)

        movie_ae.accumulate_stats()
        people_ae.accumulate_stats()
        movie_ae.finalize_stats()
        people_ae.finalize_stats()

        movie_ae.build_autoencoder()
        people_ae.build_autoencoder()

        cache_path = ensure_joint_tensor_cache(
            config,
            db_path,
            movie_ae.fields,
            people_ae.fields,
        )
        logging.info("Cache rebuild complete. Resuming training.")
    # -------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    movie_ae.model.to(device)
    people_ae.model.to(device)

    loss_logger = LossLedger()

    num_workers = int(getattr(config, "num_workers", 0) or 0)
    cfg_pf = int(getattr(config, "prefetch_factor", 2) or 0)
    prefetch_factor = None if num_workers == 0 else max(1, cfg_pf)
    pin = bool(torch.cuda.is_available())

    ds_epoch = EdgeTensorCacheDataset(str(cache_path))
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
    total_edges = len(ds_epoch)
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


def save_checkpoint(
    model_dir: Path,
    joint_model: JointAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    config: ProjectConfig,
):
    """
    Saves a resume-capable checkpoint including model weights, optimizer/scheduler state,
    and the project configuration.
    """
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the config for reference/verification
        with open(model_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=4)

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": joint_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "rng_state_pytorch": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_random": random.getstate(),
        }

        # Atomic save is preferred but simple direct save here
        torch.save(state, model_dir / "training_state.pt")
        logging.info(f"Saved training state to {model_dir / 'training_state.pt'}")
    except Exception as e:
        logging.error(f"Failed to save training state: {e}")


def main(config: ProjectConfig):
    parser = argparse.ArgumentParser(
        description="Train movie-person joint embedding autoencoder"
    )
    parser.add_argument("--warm", action="store_true")
    parser.add_argument(
        "--new-run", 
        action="store_true", 
        help="Ignore existing checkpoint and start fresh"
    )
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

    # Initialize variables for training state
    start_epoch = 0
    global_step = 0
    
    # -------------------------------------------------------------------------
    # RESUME LOGIC
    # -------------------------------------------------------------------------
    checkpoint_path = Path(config.model_dir) / "training_state.pt"
    
    if checkpoint_path.exists() and not args.new_run:
        logging.info(f"Found checkpoint at {checkpoint_path}. Resuming...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            joint_model.load_state_dict(checkpoint["model_state_dict"])
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            
            start_epoch = checkpoint["epoch"]
            global_step = checkpoint["global_step"]
            
            # Restore RNG states to ensure data augmentation/shuffling continuity if possible
            torch.set_rng_state(checkpoint["rng_state_pytorch"])
            np.random.set_state(checkpoint["rng_state_numpy"])
            random.setstate(checkpoint["rng_state_random"])
            
            logging.info(f"Resumed from Epoch {start_epoch}, Step {global_step}")
            
            # Since we are restarting the epoch loop (retreading data), 
            # we do not seek the dataloader. The global_step remains valid for
            # LR scheduling purposes.
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}. Starting fresh.")
    elif args.new_run and checkpoint_path.exists():
        logging.info("--new-run flag detected. Ignoring existing checkpoint.")

    # -------------------------------------------------------------------------

    # Scheduler creation needs to happen after we know if we loaded state or not,
    # because if we loaded, we load the scheduler state dict.
    # Note: We pass -1 as last_epoch initially, then load state dict if available.
    sched = make_lr_scheduler(
        optimizer=opt,
        total_steps=total_steps,
        schedule=config.lr_schedule,
        warmup_steps=config.lr_warmup_steps,
        warmup_ratio=config.lr_warmup_ratio,
        min_factor=config.lr_min_factor,
        last_epoch=-1
    )
    
    if checkpoint_path.exists() and not args.new_run:
         if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] and sched:
             sched.load_state_dict(checkpoint["scheduler_state_dict"])

    temperature = config.nce_temp
    nce_weight = config.nce_weight
    type_w = float(getattr(config, "latent_type_loss_weight", 0.0))
    save_interval = config.save_interval
    flush_interval = config.flush_interval
    batch_size = config.batch_size

    run_logger = build_run_logger(config)
    if run_logger.run_dir:
        logging.info(f"tensorboard logdir: {run_logger.run_dir}")
        # Sync logger step
        run_logger.step = global_step

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
    edges_seen = 0 # This resets on resume as we restart the epoch iteration
    epochs = int(getattr(config, "epochs", 1) or 1)

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(loader, unit="batch", dynamic_ncols=True)
        pbar.set_description(f"epoch {epoch+1}/{epochs}")
        
        # NOTE: On resume, we restart the epoch from the beginning of the dataloader.
        # This retreads data seen in the interrupted epoch, but ensures robust resuming.
        
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
                # Save regular model checkpoints (weights only)
                mov_ae.save_model()
                per_ae.save_model()
                
                # Save Resume State (weights + optimizer + scheduler + config)
                save_checkpoint(
                    model_dir=Path(config.model_dir),
                    joint_model=joint_model,
                    optimizer=opt,
                    scheduler=sched,
                    epoch=epoch, # Save current epoch index (will start here on resume)
                    global_step=global_step,
                    config=config
                )
                logging.info(f"checkpoint saved at step {global_step}")

            if stop_flag["stop"]:
                break

        # End of epoch save
        save_checkpoint(
            model_dir=Path(config.model_dir),
            joint_model=joint_model,
            optimizer=opt,
            scheduler=sched,
            epoch=epoch + 1, # Next start epoch
            global_step=global_step,
            config=config
        )

        if stop_flag["stop"]:
            break

    logger.flush()
    mov_ae.save_model()
    per_ae.save_model()

    out = Path(project_config.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Save final model
    try:
        torch.save(joint_model.state_dict(), out / "JointMoviePersonAE_final.pt")
    except Exception as e:
        logging.error(f"Failed to save final model: {e}")
        
    run_logger.close()


if __name__ == "__main__":
    main(project_config)