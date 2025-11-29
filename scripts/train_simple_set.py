# scripts/train_simple_set.py

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
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import project_config, ensure_dirs, ProjectConfig
from scripts.autoencoder.run_logger import RunLogger
from scripts.autoencoder.print_model import print_model_summary
from scripts.simple_set.model import HybridSetModel
from scripts.simple_set.data import HybridSetDataset, collate_hybrid_set
from scripts.simple_set.precompute import ensure_hybrid_cache
from scripts.simple_set.recon import HybridSetReconLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def asymmetric_loss(logits, targets, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
    p = torch.sigmoid(logits)
    pos_weight = (1 - p).pow(gamma_pos)
    loss_pos = -targets * torch.log(p.clamp(min=eps)) * pos_weight
    
    p_neg = p.clone()
    p_neg[p < clip] = 0.0 
    neg_weight = p_neg.pow(gamma_neg)
    loss_neg = -(1 - targets) * torch.log((1 - p).clamp(min=eps)) * neg_weight
    return (loss_pos + loss_neg).sum(dim=1).mean()

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

def save_checkpoint(model_dir, model, optimizer, scheduler, epoch, global_step, config):
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "hybrid_set_config.json", "w") as f:
            json.dump(asdict(config), f, indent=4)
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "rng_state_pytorch": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_random": random.getstate(),
        }
        torch.save(state, model_dir / "hybrid_set_state.pt")
        logging.info(f"Saved training state to {model_dir / 'hybrid_set_state.pt'}")
    except Exception as e:
        logging.error(f"Failed to save training state: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Set (Metadata->People) Decoder")
    parser.add_argument("--new-run", action="store_true", help="Start fresh")
    args = parser.parse_args()

    cfg = project_config
    ensure_dirs(cfg)
    
    # 1. Prepare Data
    cache_path = ensure_hybrid_cache(cfg)
    ds = HybridSetDataset(str(cache_path), cfg)
    
    num_people = ds.num_people
    
    # Custom collate needs reference to dataset to slice big tensors
    def _collate(batch_indices):
        return collate_hybrid_set(batch_indices, ds)

    loader = DataLoader(
        ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
        pin_memory=False # Manual pinning in collate if needed, but big tensors usually pinned if mmap
    )

    logging.info(f"Data loaded: {len(ds)} Movies, {num_people} People.")

    # 2. Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    model = HybridSetModel(
        fields=ds.fields,
        num_people=num_people,
        latent_dim=int(cfg.hybrid_set_latent_dim),
        hidden_dim=int(cfg.hybrid_set_hidden_dim),
        output_rank=int(cfg.hybrid_set_output_rank),
        depth=int(cfg.hybrid_set_depth),
        dropout=float(cfg.hybrid_set_dropout)
    ).to(device)

    # --- Print Model Summary ---
    # Create dummy inputs from the dataset to trace the model
    dummy_indices = [0, 1]
    dummy_inputs, _, _, _ = _collate(dummy_indices)
    dummy_inputs = [x.to(device) for x in dummy_inputs]
    
    print("\n" + "="*40)
    print("      HYBRID SET MODEL ARCHITECTURE      ")
    print("="*40)
    print_model_summary(model, [dummy_inputs])
    print("="*40 + "\n")
    # ---------------------------

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.hybrid_set_lr),
        weight_decay=float(cfg.hybrid_set_weight_decay)
    )

    # 3. Training Setup
    run_logger = RunLogger(cfg.tensorboard_dir, "hybrid_set", cfg)
    recon_logger = HybridSetReconLogger(
        dataset=ds,
        interval_steps=int(cfg.hybrid_set_recon_interval)
    )

    num_epochs = int(cfg.hybrid_set_epochs)
    sched = make_lr_scheduler(opt, len(loader) * num_epochs, cfg.lr_schedule, cfg.lr_warmup_steps, cfg.lr_warmup_ratio, cfg.lr_min_factor)

    start_epoch = 0
    global_step = 0
    checkpoint_path = Path(cfg.model_dir) / "hybrid_set_state.pt"

    if checkpoint_path.exists() and not args.new_run:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt["global_step"]
            if ckpt["scheduler_state_dict"] and sched:
                sched.load_state_dict(ckpt["scheduler_state_dict"])
            logging.info(f"Resumed from Epoch {start_epoch}, Step {global_step}")
        except Exception as e:
            logging.error(f"Failed resume: {e}")

    if run_logger and run_logger.run_dir: run_logger.step = global_step

    stop_flag = {"stop": False}
    def _sig(s, f): stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _sig)
    
    w_bce = float(cfg.hybrid_set_w_bce)
    w_count = float(cfg.hybrid_set_w_count)
    save_int = int(cfg.hybrid_set_save_interval)

    # 4. Loop
    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(loader, dynamic_ncols=True, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            iter_start = time.perf_counter()
            
            # Unpack
            batch_inputs, multi_hot_targets, count_targets, _ = batch
            
            batch_inputs = [x.to(device, non_blocking=True) for x in batch_inputs]
            multi_hot_targets = multi_hot_targets.to(device, non_blocking=True)
            count_targets = count_targets.to(device, non_blocking=True)
            
            model.train()
            opt.zero_grad()
            
            logits, pred_counts = model(batch_inputs)
            
            loss_set = asymmetric_loss(logits, multi_hot_targets)
            loss_count = F.mse_loss(pred_counts, count_targets)
            
            total_loss = w_bce * loss_set + w_count * loss_count
            
            total_loss.backward()
            opt.step()
            if sched: sched.step()
            
            iter_time = time.perf_counter() - iter_start
            
            # Metrics
            with torch.no_grad():
                pos_mask = (multi_hot_targets > 0.5)
                if pos_mask.sum() > 0:
                    pos_prob = torch.sigmoid(logits)[pos_mask].mean().item()
                else:
                    pos_prob = 0.0
                probs = torch.sigmoid(logits)
                neg_prob = probs[~pos_mask].mean().item()
            
            if run_logger:
                run_logger.add_scalar("loss/total", total_loss.item(), global_step)
                run_logger.add_scalar("loss/set_asymmetric", loss_set.item(), global_step)
                run_logger.add_scalar("loss/count", loss_count.item(), global_step)
                run_logger.add_scalar("metric/pos_prob", pos_prob, global_step)
                run_logger.add_scalar("metric/neg_prob", neg_prob, global_step)
                run_logger.add_scalar("time/iter_sec", iter_time, global_step)
                run_logger.tick()
                
            recon_logger.step(global_step, model, batch_inputs, multi_hot_targets, count_targets, run_logger)
            
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}", 
                set=f"{loss_set.item():.4f}", 
                cnt=f"{loss_count.item():.4f}",
                pos=f"{pos_prob:.2f}"
            )
            
            global_step += 1
            
            if global_step % save_int == 0:
                save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch, global_step, cfg)
                torch.save(model.state_dict(), Path(cfg.model_dir) / "HybridSetModel.pt")

            if stop_flag["stop"]: break
        
        save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch+1, global_step, cfg)
        if stop_flag["stop"]: break

    torch.save(model.state_dict(), Path(cfg.model_dir) / "HybridSetModel_final.pt")
    if run_logger: run_logger.close()

if __name__ == "__main__":
    main()