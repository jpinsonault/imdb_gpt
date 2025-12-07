# scripts/train_simple_set.py

import argparse
import logging
import json
import signal
import math
import time
import sys
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from tqdm import tqdm

from config import project_config, ensure_dirs
from scripts.autoencoder.run_logger import RunLogger
from scripts.autoencoder.print_model import print_model_summary
from scripts.simple_set.precompute import ensure_hybrid_cache
from scripts.simple_set.model import HybridSetModel
from scripts.simple_set.data import HybridSetDataset, FastInfiniteLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GracefulStopper:
    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, sig, frame):
        print("\n[GracefulStopper] Signal received! Finishing current step and saving...")
        self.stop = True


def make_lr_scheduler(optimizer, total_steps, schedule, warmup_steps, warmup_ratio, min_factor, last_epoch=-1):
    if total_steps is None: return None
    total_steps = max(1, int(total_steps))
    ratio = float(warmup_ratio)
    frac_warmup = int(total_steps * ratio)
    w_steps = max(int(warmup_steps), frac_warmup)
    
    def lambda_fn(step):
        s = int(step)
        if s < w_steps:
            return float(s + 1) / float(w_steps)
        progress = float(s - w_steps) / float(max(1, total_steps - w_steps))
        progress = max(0.0, min(1.0, progress))
        if schedule == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * decay
        else:
            return max(min_factor, 1.0 - (1.0 - min_factor) * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn, last_epoch=last_epoch)


def save_checkpoint(model_dir, model, optimizer, scheduler, epoch, global_step, config):
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / "hybrid_set_config.json"
        state_path = model_dir / "hybrid_set_state.pt"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=4)
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        state = {"epoch": int(epoch), "global_step": int(global_step), "model_state_dict": model_state}
        torch.save(state, state_path)
        logging.info(f"Saved training state to {state_path}")
    except Exception as e:
        logging.error(f"Failed to save training state: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    args = parser.parse_args()

    cfg = project_config
    ensure_dirs(cfg)
    if args.no_resume: cfg.refresh_cache = False

    cache_path = ensure_hybrid_cache(cfg)
    ds = HybridSetDataset(str(cache_path), cfg)
    loader = FastInfiniteLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridSetModel(
        fields=ds.fields,
        num_people=ds.num_people,
        heads_config=cfg.hybrid_set_heads,
        head_vocab_sizes=ds.head_vocab_sizes,
        head_groups_config=cfg.hybrid_set_head_groups,  # Added this line
        latent_dim=cfg.hybrid_set_latent_dim,
        hidden_dim=cfg.hybrid_set_hidden_dim,
        dropout=cfg.hybrid_set_dropout,
        num_movies=len(ds),
        hybrid_set_logit_scale=cfg.hybrid_set_logit_scale
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.hybrid_set_lr,
        weight_decay=cfg.hybrid_set_weight_decay,
    )

    # Print Summary
    sample_inputs_cpu, _, sample_idx = next(loader)
    print("\n" + "=" * 50)
    print_model_summary(model, {"field_tensors": [x.to(device) for x in sample_inputs_cpu], "batch_indices": sample_idx})
    print("=" * 50 + "\n")

    mapping_tensors = {}
    if hasattr(ds, "head_mappings"):
        for name, t in ds.head_mappings.items():
            mapping_tensors[name] = t.to(device)

    # Pre-calc counts for logging
    head_true_counts = {}
    if hasattr(ds, "heads_padded"):
        for head_name, padded in ds.heads_padded.items():
            mask = padded != -1
            head_true_counts[head_name] = mask.sum(dim=1, keepdim=True)

    scaler = torch.amp.GradScaler("cuda")
    from scripts.autoencoder.run_logger import RunLogger
    from scripts.simple_set.recon import HybridSetReconLogger
    run_logger = RunLogger(cfg.tensorboard_dir, "hybrid_set", cfg)
    recon_logger = HybridSetReconLogger(ds, interval_steps=int(cfg.hybrid_set_recon_interval))

    num_epochs = int(cfg.hybrid_set_epochs)
    batches_per_epoch = len(loader)
    sched = make_lr_scheduler(optimizer, batches_per_epoch * num_epochs, cfg.lr_schedule, cfg.lr_warmup_steps, cfg.lr_warmup_ratio, cfg.lr_min_factor)

    start_epoch, global_step = 0, 0
    ckpt_path = Path(cfg.model_dir) / "hybrid_set_state.pt"
    if ckpt_path.exists() and not args.no_resume:
        try:
            c = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(c["model_state_dict"])
            start_epoch = c.get("epoch", 0)
            global_step = c.get("global_step", 0)
            logging.info(f"Resumed from epoch {start_epoch}")
        except: pass

    # --- FIX: Proper Signal Handling ---
    stopper = GracefulStopper()

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(range(batches_per_epoch), dynamic_ncols=True, desc=f"Epoch {epoch+1}")
        for _ in pbar:
            # Check for stop signal at the start of iteration
            if stopper.stop:
                pbar.close()
                break

            inputs_cpu, heads_padded_cpu, indices_cpu = next(loader)
            inputs = [x.to(device, non_blocking=True) for x in inputs_cpu]
            
            model.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits_dict, recon_table, _ = model(inputs, indices_cpu)

                # Recon Loss
                recon_loss = 0.0
                for f, p, t in zip(ds.fields, recon_table, inputs):
                    recon_loss += f.compute_loss(p, t) * f.weight

                # Set Loss
                set_loss_total = 0.0
                collect = (global_step + 1) % recon_logger.every == 0
                coords_log, counts_log = {}, {}

                for head_name, logits in logits_dict.items():
                    raw_padded = heads_padded_cpu[head_name].to(device, non_blocking=True)
                    mask = raw_padded != -1
                    rows, cols = torch.nonzero(mask, as_tuple=True)
                    
                    targets = torch.zeros_like(logits)
                    if rows.numel() > 0:
                        glob = raw_padded[rows, cols].long()
                        loc = mapping_tensors[head_name][glob]
                        valid = loc != -1
                        if valid.any():
                            rows_v, loc_v = rows[valid], loc[valid]
                            targets[rows_v, loc_v] = 1.0
                            if collect:
                                ds_rows = indices_cpu.to(device)[rows_v]
                                coords_log[head_name] = torch.stack([ds_rows, loc_v], dim=1).cpu()

                    if collect and head_name in head_true_counts:
                         counts_log[head_name] = head_true_counts[head_name]

                    # Standard BCE With Logits
                    set_loss_total += F.binary_cross_entropy_with_logits(logits, targets)

                loss = cfg.hybrid_set_w_bce * set_loss_total + cfg.hybrid_set_w_recon * recon_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if sched: sched.step()

            if run_logger:
                run_logger.add_scalar("loss/total", loss.item(), global_step)
                run_logger.add_scalar("loss/set", set_loss_total.item(), global_step)
                run_logger.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], global_step)
                run_logger.tick()

            if collect:
                recon_logger.step(global_step, model, inputs, indices_cpu, coords_log, counts_log, run_logger)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1
            
            if global_step % cfg.hybrid_set_save_interval == 0:
                save_checkpoint(Path(cfg.model_dir), model, optimizer, sched, epoch, global_step, cfg)
        
        if stopper.stop:
            print("[GracefulStopper] Training loop exited. Final save...")
            break
            
        save_checkpoint(Path(cfg.model_dir), model, optimizer, sched, epoch + 1, global_step, cfg)
    
    save_checkpoint(Path(cfg.model_dir), model, optimizer, sched, epoch if stopper.stop else num_epochs, global_step, cfg)
    if run_logger: run_logger.close()
    print("Done.")

if __name__ == "__main__":
    main()