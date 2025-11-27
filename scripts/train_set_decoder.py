# scripts/train_set_decoder.py

import argparse
import logging
import json
import random
import signal
import math
import numpy as np
import time
from pathlib import Path
from dataclasses import asdict

import torch
from tqdm import tqdm

from config import project_config, ensure_dirs, ProjectConfig
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.autoencoder.run_logger import build_run_logger
from scripts.set_decoder.model import SetDecoder
from scripts.set_decoder.training import _compute_cost_matrices
from scripts.set_decoder.recon_logger import SetReconstructionLogger
from scripts.precompute_set_cache import ensure_set_decoder_cache
from scripts.set_decoder.data import CachedSetDataset, collate_set_decoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def make_lr_scheduler(optimizer, total_steps, schedule, warmup_steps, warmup_ratio, min_factor, last_epoch=-1):
    if total_steps is None: return None
    total_steps = max(1, int(total_steps))
    schedule = (schedule or "").lower()
    if schedule not in ("cosine", "linear"): return None

    base_warmup = max(0, int(warmup_steps))
    ratio = float(warmup_ratio)
    frac_warmup = int(total_steps * ratio) if ratio > 0.0 else 0
    w_steps = max(base_warmup, frac_warmup)
    w_steps = min(w_steps, total_steps - 1) if total_steps > 1 else 0
    min_factor = float(min_factor)

    def cosine_lambda(step):
        s = int(step)
        if w_steps > 0 and s < w_steps: return float(s + 1) / float(w_steps)
        if s >= total_steps: return min_factor
        t = float(s - w_steps) / float(total_steps - w_steps)
        decay = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_factor + (1.0 - min_factor) * decay

    def linear_lambda(step):
        s = int(step)
        if w_steps > 0 and s < w_steps: return float(s + 1) / float(w_steps)
        if s >= total_steps: return min_factor
        t = float(s - w_steps) / float(total_steps - w_steps)
        return max(min_factor, 1.0 - (1.0 - min_factor) * t)

    lr_lambda = cosine_lambda if schedule == "cosine" else linear_lambda
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

def save_checkpoint(model_dir, model, optimizer, scheduler, epoch, global_step, config, best_loss):
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "set_decoder_config.json", "w") as f:
            json.dump(asdict(config), f, indent=4)
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_loss": best_loss,
            "rng_state_pytorch": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_random": random.getstate(),
        }
        torch.save(state, model_dir / "set_decoder_state.pt")
        logging.info(f"Saved training state to {model_dir / 'set_decoder_state.pt'}")
    except Exception as e:
        logging.error(f"Failed to save training state: {e}")

def build_set_decoder_trainer(cfg: ProjectConfig, db_path: str):
    mov_ae, per_ae = _load_frozen_autoencoders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mov_ae.decoder.to(device).eval()
    per_ae.decoder.to(device).eval()
    mov_ae.encoder.to("cpu")
    per_ae.encoder.to("cpu")

    for m in [mov_ae.encoder, mov_ae.decoder, per_ae.encoder, per_ae.decoder]:
        for p in m.parameters(): p.requires_grad_(False)

    cache_path = ensure_set_decoder_cache(cfg)
    num_slots = int(getattr(cfg, "set_decoder_slots", 10))
    ds = CachedSetDataset(str(cache_path), num_slots=num_slots)

    from torch.utils.data import DataLoader
    num_workers = int(getattr(cfg, "num_workers", 0) or 0)
    
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_set_decoder,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
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
    
    # [FIX] Updated signature: removed db_path
    recon_logger = SetReconstructionLogger(
        model=model, movie_ae=mov_ae, people_ae=per_ae,
        num_slots=num_slots,
        interval_steps=int(getattr(cfg, "set_decoder_callback_interval", 500)),
        num_samples=int(getattr(cfg, "set_decoder_recon_samples", 3)),
        table_width=int(getattr(cfg, "set_decoder_table_width", 60)),
    )

    loss_cfg = {
        "w_latent": float(getattr(cfg, "set_decoder_w_latent", 1.0)),
        "w_recon": float(getattr(cfg, "set_decoder_w_recon", 1.0)),
        "w_presence": float(getattr(cfg, "set_decoder_w_presence", 1.0)),
        "w_null": float(getattr(cfg, "set_decoder_w_null", 0.1)),
    }

    return model, opt, loader, mov_ae, per_ae, run_logger, recon_logger, loss_cfg

def main():
    parser = argparse.ArgumentParser(description="Train set decoder")
    parser.add_argument("--new-run", action="store_true", help="Start fresh")
    args = parser.parse_args()

    cfg = project_config
    ensure_dirs(cfg)
    db_path = cfg.db_path

    model, opt, loader, mov_ae, per_ae, run_logger, recon_logger, loss_cfg = build_set_decoder_trainer(cfg, db_path=db_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = int(getattr(cfg, "set_decoder_epochs", 3))
    save_interval = int(getattr(cfg, "set_decoder_save_interval", 1000))
    flush_interval = int(getattr(cfg, "flush_interval", 250))
    w_latent, w_recon = loss_cfg["w_latent"], loss_cfg["w_recon"]
    w_presence, w_null = loss_cfg["w_presence"], loss_cfg["w_null"]

    sched = make_lr_scheduler(opt, len(loader) * num_epochs, cfg.lr_schedule, cfg.lr_warmup_steps, cfg.lr_warmup_ratio, cfg.lr_min_factor)

    start_epoch = 0
    global_step = 0
    best_loss = None
    checkpoint_path = Path(cfg.model_dir) / "set_decoder_state.pt"

    if checkpoint_path.exists() and not args.new_run:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch, global_step = ckpt["epoch"], ckpt["global_step"]
            best_loss = ckpt.get("best_loss")
            if ckpt["scheduler_state_dict"] and sched: sched.load_state_dict(ckpt["scheduler_state_dict"])
            logging.info(f"Resumed from Epoch {start_epoch}, Step {global_step}")
        except Exception as e:
            logging.error(f"Failed resume: {e}")

    if run_logger and run_logger.run_dir: run_logger.step = global_step

    stop_flag = {"stop": False}
    def _sig(s, f): stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _sig)

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(loader, dynamic_ncols=True, desc=f"set-dec epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            z_movies, Z_gt, mask, Y_gt_fields = batch
            z_movies = z_movies.to(device, non_blocking=True)
            Z_gt = Z_gt.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            Y_gt_fields = [y.to(device, non_blocking=True) for y in Y_gt_fields]

            model.train()
            opt.zero_grad()
            z_slots, presence_logits = model(z_movies)

            # [FIX] Hungarian import inside loop to avoid top-level issues if any
            from scripts.set_decoder.training import _compute_cost_matrices, _hungarian

            C_match_list, C_lat_list, C_rec_list = _compute_cost_matrices(
                per_ae, z_slots, Z_gt, Y_gt_fields, mask, w_latent, w_recon
            )

            total_latent_loss = torch.zeros((), device=device)
            total_recon_loss = torch.zeros((), device=device)
            total_presence_loss = torch.zeros((), device=device)
            total_null_loss = torch.zeros((), device=device)
            
            matched_cnt, pres_cnt, null_cnt = 0, 0, 0
            recalls, precisions, cards = [], [], []

            for b in range(z_slots.shape[0]):
                k_b = int(mask[b].sum().item())
                logits_b = presence_logits[b]
                probs_b = torch.sigmoid(logits_b)
                pred_on = int((probs_b > 0.5).sum().item())

                rows, cols = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
                if k_b > 0 and C_match_list[b] is not None:
                    # Pass the cost matrix for this sample
                    rows, cols = _hungarian(C_match_list[b])
                    
                    # C_match_list is (N, k_b). 
                    # Rows are pred indices, Cols are target indices.
                    # _hungarian returns valid pairs.
                    # No need to filter cols < k_b explicitly because the matrix passed 
                    # into hungarian was already sliced to k_b cols in _compute_cost_matrices.

                # Presence Loss
                tgt_pres = torch.zeros_like(logits_b)
                if rows.numel() > 0:
                    tgt_pres[rows.to(device)] = 1.0
                
                total_presence_loss += torch.nn.functional.binary_cross_entropy_with_logits(logits_b, tgt_pres)
                pres_cnt += 1

                # Content Loss
                if rows.numel() > 0:
                    r_idx, c_idx = rows.to(device), cols.to(device)
                    # Accumulate losses from the precomputed matrices
                    total_latent_loss += C_lat_list[b][r_idx, c_idx].mean()
                    total_recon_loss += C_rec_list[b][r_idx, c_idx].mean()
                    matched_cnt += 1
                
                # Null Loss
                if rows.numel() < z_slots.shape[1]:
                    mask_null = torch.ones(z_slots.shape[1], dtype=torch.bool, device=device)
                    if rows.numel() > 0: mask_null[rows.to(device)] = False
                    total_null_loss += z_slots[b, mask_null].pow(2).mean()
                    null_cnt += 1

                # Metrics
                matched_n = rows.numel()
                recalls.append(matched_n / max(k_b, 1) if k_b > 0 else (1.0 if pred_on==0 else 0.0))
                precisions.append(matched_n / max(pred_on, 1) if pred_on > 0 else (1.0 if k_b==0 else 0.0))
                cards.append(abs(pred_on - k_b))

            loss = (
                w_latent * (total_latent_loss / max(1, matched_cnt)) +
                w_recon * (total_recon_loss / max(1, matched_cnt)) +
                w_presence * (total_presence_loss / max(1, pres_cnt)) +
                w_null * (total_null_loss / max(1, null_cnt))
            )

            loss.backward()
            opt.step()
            if sched: sched.step()

            m_rec = sum(recalls)/len(recalls)
            m_prec = sum(precisions)/len(precisions)
            m_err = sum(cards)/len(cards)

            if run_logger:
                run_logger.add_scalar("set_decoder/loss", float(loss), global_step)
                run_logger.add_scalar("set_decoder/recall", m_rec, global_step)
                run_logger.tick()

            pbar.set_postfix(loss=f"{loss.item():.4f}", rec=f"{m_rec:.3f}", prec=f"{m_prec:.3f}", err=f"{m_err:.2f}")

            if (global_step+1) % save_interval == 0:
                save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch, global_step, cfg, None)
                torch.save(model.state_dict(), Path(cfg.model_dir) / "SetDecoder.pt")

            if hasattr(loader.dataset, "movies"):
                sample_tconsts = loader.dataset.movies[: z_movies.size(0)]
            else:
                sample_tconsts = [""] * z_movies.size(0)

            if recon_logger:
                recon_logger.step(global_step, z_movies.detach().cpu(), mask.detach().cpu(), run_logger, sample_tconsts)

            global_step += 1
            if stop_flag["stop"]: break
        
        save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch+1, global_step, cfg, None)
        if stop_flag["stop"]: break

    torch.save(model.state_dict(), Path(cfg.model_dir) / "SetDecoder_final.pt")
    if run_logger: run_logger.close()

if __name__ == "__main__":
    main()