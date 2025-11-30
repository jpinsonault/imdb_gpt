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

from config import project_config, ensure_dirs
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run", action="store_true")
    args = parser.parse_args()
    cfg = project_config
    ensure_dirs(cfg)
    
    cache_path = ensure_hybrid_cache(cfg)
    ds = HybridSetDataset(str(cache_path), cfg)
    
    def _collate(batch_indices):
        return collate_hybrid_set(batch_indices, ds)

    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=int(cfg.num_workers), collate_fn=_collate,
        pin_memory=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HybridSetModel(
        fields=ds.fields,
        num_people=ds.num_people,
        heads_config=cfg.hybrid_set_heads,
        latent_dim=int(cfg.hybrid_set_latent_dim),
        hidden_dim=int(cfg.hybrid_set_hidden_dim),
        base_output_rank=int(cfg.hybrid_set_output_rank),
        depth=int(cfg.hybrid_set_depth),
        dropout=float(cfg.hybrid_set_dropout)
    ).to(device)

    # Dummy forward for summary
    dummy_idxs = [0]
    inputs, _, _, _ = _collate(dummy_idxs)
    inputs = [x.to(device) for x in inputs]
    print_model_summary(model, [inputs])

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.hybrid_set_lr), weight_decay=float(cfg.hybrid_set_weight_decay))
    run_logger = RunLogger(cfg.tensorboard_dir, "hybrid_set", cfg)
    recon_logger = HybridSetReconLogger(ds, interval_steps=int(cfg.hybrid_set_recon_interval))

    num_epochs = int(cfg.hybrid_set_epochs)
    sched = make_lr_scheduler(opt, len(loader)*num_epochs, cfg.lr_schedule, cfg.lr_warmup_steps, cfg.lr_warmup_ratio, cfg.lr_min_factor)

    start_epoch, global_step = 0, 0
    ckpt_path = Path(cfg.model_dir) / "hybrid_set_state.pt"
    if ckpt_path.exists() and not args.new_run:
        try:
            c = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(c["model_state_dict"])
            opt.load_state_dict(c["optimizer_state_dict"])
            start_epoch = c["epoch"]
            global_step = c["global_step"]
            if c["scheduler_state_dict"] and sched: sched.load_state_dict(c["scheduler_state_dict"])
            logging.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logging.error(f"Resume failed: {e}")
    
    if run_logger.run_dir: run_logger.step = global_step
    stop_flag = {"stop": False}
    signal.signal(signal.SIGINT, lambda s,f: stop_flag.update({"stop": True}))

    w_bce, w_count, w_recon = float(cfg.hybrid_set_w_bce), float(cfg.hybrid_set_w_count), 1.0

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(loader, dynamic_ncols=True, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            iter_start = time.perf_counter()
            # unpack: coords_dict contains tensors of shape (N, 2) [row, col]
            inputs, coords_dict, count_targets, _ = batch
            
            inputs = [x.to(device, non_blocking=True) for x in inputs]
            
            model.train()
            opt.zero_grad()
            
            logits_dict, counts_dict, recon_outputs = model(inputs)
            
            total_set_loss = 0.0
            total_count_loss = 0.0
            
            batch_size = inputs[0].size(0)
            
            for head_name in logits_dict.keys():
                # --- FAST TARGET CONSTRUCTION ON GPU ---
                # 1. Get sparse coords
                coords = coords_dict.get(head_name)
                
                # 2. Build Dense Target on Device
                # Initialize zeros (B, NumPeople)
                target_dense = torch.zeros(
                    batch_size, ds.num_people, 
                    device=device, dtype=torch.float32
                )
                
                if coords is not None and coords.size(0) > 0:
                    coords = coords.to(device, non_blocking=True)
                    # Use index_put or scatter. Scatter is usually safe.
                    # dim=0 (row), dim=1 (col) -> (N, 2)
                    # We want target[coords[:,0], coords[:,1]] = 1.0
                    target_dense.index_put_(
                        (coords[:, 0], coords[:, 1]), 
                        torch.tensor(1.0, device=device)
                    )
                
                # 3. Counts
                t_cnt = count_targets.get(head_name)
                if t_cnt is None:
                    t_cnt = torch.zeros(batch_size, 1, device=device)
                else:
                    t_cnt = t_cnt.to(device, non_blocking=True)
                    
                total_set_loss += asymmetric_loss(logits_dict[head_name], target_dense)
                total_count_loss += F.mse_loss(counts_dict[head_name], t_cnt)
            
            # Recon Loss
            recon_loss = 0.0
            for f, p, t in zip(ds.fields, recon_outputs, inputs):
                recon_loss += f.compute_loss(p, t) * float(f.weight)
                
            loss = w_bce * total_set_loss + w_count * total_count_loss + w_recon * recon_loss
            loss.backward()
            opt.step()
            if sched: sched.step()
            
            iter_time = time.perf_counter() - iter_start
            
            if run_logger:
                run_logger.add_scalar("loss/total", loss.item(), global_step)
                run_logger.add_scalar("loss/set", total_set_loss.item(), global_step)
                run_logger.add_scalar("loss/recon", recon_loss.item(), global_step)
                run_logger.add_scalar("time/iter", iter_time, global_step)
                run_logger.tick()
            
            # Pass sparse coords to recon logger (requires update in recon logger to handle sparse if needed, 
            # OR we pass the dense we just built)
            # We must be careful not to hold onto the massive dense tensors.
            # The Recon logger usually runs infrequently. 
            # We can re-build a *small sample* dense tensor inside the logger if needed.
            
            # Update: We need to adapt the logger call slightly because we changed `multi_targets` signature
            recon_logger.step(global_step, model, inputs, coords_dict, count_targets, run_logger)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", set=f"{total_set_loss.item():.4f}")
            global_step += 1
            
            if global_step % int(cfg.hybrid_set_save_interval) == 0:
                save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch, global_step, cfg)
            
            if stop_flag["stop"]: break
        if stop_flag["stop"]: break
        
        save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch+1, global_step, cfg)
        
    if run_logger: run_logger.close()

if __name__ == "__main__":
    main()