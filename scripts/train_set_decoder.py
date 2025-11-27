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
from scripts.autoencoder.run_logger import RunLogger
from scripts.set_decoder.model import SetDecoder
from scripts.set_decoder.recon_logger import SetReconstructionLogger
from scripts.precompute_set_cache import ensure_set_decoder_cache
from scripts.set_decoder.data import CachedSetDataset, collate_set_decoder
from scripts.set_decoder.training import compute_cost_and_losses, match_batch_hungarian

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
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
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
    
    # FiLM + Deep Supervision Model
    model = SetDecoder(
        latent_dim=latent_dim,
        num_slots=num_slots,
        hidden_mult=float(getattr(cfg, "set_decoder_hidden_mult", 2.0)),
        num_layers=int(getattr(cfg, "set_decoder_layers", 4)),
        num_heads=int(getattr(cfg, "set_decoder_heads", 4)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(cfg, "set_decoder_lr", 3e-4)),
        weight_decay=float(getattr(cfg, "set_decoder_weight_decay", 1e-2)),
    )

    run_logger = RunLogger(cfg.tensorboard_dir, "set_decoder", cfg)
    
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
    device = next(model.parameters()).device
    
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
            iter_start = time.perf_counter()

            z_movies, Z_gt, mask, Y_gt_fields = batch
            z_movies = z_movies.to(device, non_blocking=True)
            Z_gt = Z_gt.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            Y_gt_fields = [y.to(device, non_blocking=True) for y in Y_gt_fields]

            model.train()
            opt.zero_grad()
            
            # Outputs is a list of tuples [(z, p), (z, p)...] for each layer
            layer_outputs = model(z_movies)
            
            total_loss = torch.tensor(0.0, device=device)
            
            # Metrics to log (will grab from final layer)
            log_metrics = {}

            # Iterate through all layers (Deep Supervision)
            for i, (z_slots, presence_logits) in enumerate(layer_outputs):
                
                # 1. Compute Costs & Losses for THIS layer
                C_match, C_lat_full, C_rec_full, lengths = compute_cost_and_losses(
                    per_ae, z_slots, Z_gt, Y_gt_fields, mask, w_latent, w_recon
                )

                # 2. Match for THIS layer (Independent matching per layer guides flow)
                b_idx, r_idx, c_idx = match_batch_hungarian(C_match, lengths)
                
                # 3. Calculate layer loss
                matched_count = b_idx.numel()
                denom = max(1, matched_count)
                
                loss_latent_l = torch.zeros((), device=device)
                loss_recon_l = torch.zeros((), device=device)
                
                if matched_count > 0:
                    loss_latent_l = C_lat_full[b_idx, r_idx, c_idx].sum() / denom
                    loss_recon_l = C_rec_full[b_idx, r_idx, c_idx].sum() / denom

                # Presence
                target_pres = torch.zeros_like(presence_logits)
                if matched_count > 0:
                    target_pres[b_idx, r_idx] = 1.0
                
                loss_presence_l = torch.nn.functional.binary_cross_entropy_with_logits(presence_logits, target_pres)

                # Null
                mask_unmatched = torch.ones((z_slots.shape[0], z_slots.shape[1]), dtype=torch.bool, device=device)
                if matched_count > 0:
                    mask_unmatched[b_idx, r_idx] = False
                loss_null_l = z_slots[mask_unmatched].pow(2).mean()
                if torch.isnan(loss_null_l): loss_null_l = torch.tensor(0.0, device=device)

                layer_loss = (
                    w_latent * loss_latent_l +
                    w_recon * loss_recon_l +
                    w_presence * loss_presence_l +
                    w_null * loss_null_l
                )
                
                total_loss = total_loss + layer_loss

                # If this is the final layer, capture metrics for logging
                if i == len(layer_outputs) - 1:
                    total_targets = lengths.sum().item()
                    total_preds = (torch.sigmoid(presence_logits) > 0.5).float().sum().item()
                    
                    log_metrics['loss_total'] = layer_loss.item() # Log final layer's contribution for clarity
                    log_metrics['loss_latent'] = loss_latent_l.item()
                    log_metrics['loss_recon'] = loss_recon_l.item()
                    log_metrics['loss_presence'] = loss_presence_l.item()
                    log_metrics['loss_null'] = loss_null_l.item()
                    
                    log_metrics['recall'] = matched_count / max(1, total_targets)
                    log_metrics['precision'] = matched_count / max(1, total_preds)
                    log_metrics['card_error'] = abs(total_preds - total_targets) / z_movies.shape[0]
                    
                    # Top-1 Loss Check
                    with torch.no_grad():
                        top1_slots = presence_logits.argmax(dim=1)
                        is_top1 = (r_idx == top1_slots[b_idx])
                        top1_loss = 0.0
                        if is_top1.any():
                            top1_loss = C_rec_full[b_idx[is_top1], r_idx[is_top1], c_idx[is_top1]].mean().item()
                        log_metrics['loss_recon_top1'] = top1_loss

            # Average loss over layers to keep magnitude consistent
            total_loss = total_loss / len(layer_outputs)

            total_loss.backward()
            opt.step()
            if sched: sched.step()

            iter_time = time.perf_counter() - iter_start

            if run_logger:
                run_logger.add_scalar("loss/total", float(total_loss), global_step)
                run_logger.add_scalar("loss/latent", log_metrics['loss_latent'], global_step)
                run_logger.add_scalar("loss/recon", log_metrics['loss_recon'], global_step)
                run_logger.add_scalar("loss/presence", log_metrics['loss_presence'], global_step)
                run_logger.add_scalar("loss/null", log_metrics['loss_null'], global_step)
                run_logger.add_scalar("loss/recon_top1", log_metrics['loss_recon_top1'], global_step)
                
                run_logger.add_scalar("metric/recall", log_metrics['recall'], global_step)
                run_logger.add_scalar("metric/precision", log_metrics['precision'], global_step)
                run_logger.add_scalar("metric/card_error", log_metrics['card_error'], global_step)
                
                run_logger.add_scalar("time/iter_sec", iter_time, global_step)
                run_logger.tick()

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}", 
                rec=f"{log_metrics['recall']:.3f}", 
                prec=f"{log_metrics['precision']:.3f}"
            )

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