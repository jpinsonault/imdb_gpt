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
from torch.cuda.amp import autocast, GradScaler

from config import project_config, ensure_dirs
from scripts.autoencoder.run_logger import RunLogger
from scripts.autoencoder.print_model import print_model_summary
from scripts.simple_set.model import HybridSetModel
from scripts.simple_set.data import HybridSetDataset, FastInfiniteLoader
from scripts.simple_set.precompute import ensure_hybrid_cache
from scripts.simple_set.recon import HybridSetReconLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ... [Keep compute_sampled_asymmetric_loss and make_lr_scheduler exactly as they were] ...
# (Omitting them here for brevity as they haven't changed, but include them in your file)

def compute_sampled_asymmetric_loss(
    embedding_batch,    
    head_layer,         
    positive_coords,    
    num_negatives=2048, 
    gamma_neg=4, 
    gamma_pos=1, 
    clip=0.05, 
    eps=1e-8
):
    device = embedding_batch.device
    batch_size = embedding_batch.shape[0]
    num_people_subset = head_layer.out_features
    
    loss_pos = torch.tensor(0.0, device=device)
    num_pos_edges = 0
    
    pos_person_indices = None
    pos_batch_indices = None

    if positive_coords.shape[0] > 0:
        pos_batch_indices = positive_coords[:, 0]
        pos_person_indices = positive_coords[:, 1]
        num_pos_edges = pos_batch_indices.shape[0]
        
        pos_embs = embedding_batch[pos_batch_indices]
        pos_weights = head_layer.weight[pos_person_indices]
        pos_biases = head_layer.bias[pos_person_indices] if head_layer.bias is not None else 0.0
        
        pos_logits = (pos_embs * pos_weights).sum(dim=1) + pos_biases
        
        p_pos = torch.sigmoid(pos_logits)
        pos_weight_factor = (1 - p_pos).pow(gamma_pos)
        loss_pos_raw = -torch.log(p_pos.clamp(min=eps)) * pos_weight_factor
        loss_pos = loss_pos_raw.sum() / max(1.0, float(num_pos_edges))

    neg_person_indices = torch.randint(0, num_people_subset, (num_negatives,), device=device)
    neg_weights = head_layer.weight[neg_person_indices]
    neg_biases = head_layer.bias[neg_person_indices] if head_layer.bias is not None else 0.0
    
    neg_logits = embedding_batch @ neg_weights.t() + neg_biases
    
    p_neg = torch.sigmoid(neg_logits)
    p_neg_clipped = p_neg.clone()
    p_neg_clipped[p_neg < clip] = 0.0
    neg_weight_factor = p_neg_clipped.pow(gamma_neg)
    loss_neg_matrix = -torch.log((1 - p_neg).clamp(min=eps)) * neg_weight_factor
    
    if pos_person_indices is not None:
        collisions = (pos_person_indices.unsqueeze(1) == neg_person_indices.unsqueeze(0))
        if collisions.any():
            match_pos_idx, match_neg_idx = torch.nonzero(collisions, as_tuple=True)
            affected_batch_idxs = pos_batch_indices[match_pos_idx]
            loss_neg_matrix[affected_batch_idxs, match_neg_idx] = 0.0

    loss_neg = loss_neg_matrix.sum() / batch_size
    return loss_pos + loss_neg

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
        if w_steps >= total_steps: return min_factor
        t = float(s - w_steps) / float(total_steps - w_steps)
        t = max(0.0, min(1.0, t))
        decay = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_factor + (1.0 - min_factor) * decay

    def linear_lambda(step):
        s = int(step)
        if w_steps > 0 and s < w_steps: return float(s + 1) / float(w_steps)
        if s >= total_steps: return min_factor
        if w_steps >= total_steps: return min_factor
        t = float(s - w_steps) / float(total_steps - w_steps)
        t = max(0.0, min(1.0, t))
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
            "scaler_state_dict": None,
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
    
    # E2E Training requires fresh cache of RAW fields if forcing new run
    if args.new_run:
        cfg.refresh_cache = True

    cache_path = ensure_hybrid_cache(cfg)
    ds = HybridSetDataset(str(cache_path), cfg)
    
    loader = FastInfiniteLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    batches_per_epoch = len(loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HybridSetModel(
        fields=ds.fields,
        num_people=ds.num_people,
        heads_config=cfg.hybrid_set_heads,
        head_vocab_sizes=ds.head_vocab_sizes,
        latent_dim=int(cfg.hybrid_set_latent_dim),
        hidden_dim=int(cfg.hybrid_set_hidden_dim),
        base_output_rank=int(cfg.hybrid_set_output_rank),
        depth=int(cfg.hybrid_set_depth),
        dropout=float(cfg.hybrid_set_dropout)
    ).to(device)

    # --- Print Model & Sizes ---
    sample_inputs_cpu, _, _ = next(loader)
    sample_inputs = [x.to(device) for x in sample_inputs_cpu]

    print("\n" + "="*50)
    print_model_summary(model, [sample_inputs])
    
    print("\n=== HybridSetModel Dimensions (End-to-End) ===")
    print(f"  Input:                 Raw Fields")
    print(f"  Learned Latent Dim:    {cfg.hybrid_set_latent_dim}")
    print(f"  Trunk Hidden Dim:      {cfg.hybrid_set_hidden_dim}")
    print(f"  Trunk Depth:           {cfg.hybrid_set_depth}")
    print(f"  Global Vocab:          {ds.num_people}")
    print("  Heads:")
    for name in model.people_bottlenecks.keys():
        rank = model.people_bottlenecks[name].out_features
        out_dim = model.people_expansions[name].out_features
        print(f"    - {name:<12} | Rank: {rank:<4} | Vocab: {out_dim}")
    print("="*50 + "\n")
    # ---------------------------

    mapping_tensors = {}
    if hasattr(ds, "head_mappings"):
        for name, t in ds.head_mappings.items():
            mapping_tensors[name] = t.to(device)

    scaler = GradScaler()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.hybrid_set_lr), weight_decay=float(cfg.hybrid_set_weight_decay))
    run_logger = RunLogger(cfg.tensorboard_dir, "hybrid_set", cfg)
    recon_logger = HybridSetReconLogger(ds, interval_steps=int(cfg.hybrid_set_recon_interval))

    num_epochs = int(cfg.hybrid_set_epochs)
    
    sched = make_lr_scheduler(
        opt, 
        total_steps=batches_per_epoch * num_epochs, 
        schedule=cfg.lr_schedule, 
        warmup_steps=cfg.lr_warmup_steps, 
        warmup_ratio=cfg.lr_warmup_ratio, 
        min_factor=cfg.lr_min_factor
    )

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
            if "scaler_state_dict" in c and c["scaler_state_dict"]:
                scaler.load_state_dict(c["scaler_state_dict"])
            logging.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logging.error(f"Resume failed: {e}")
    
    if run_logger.run_dir: run_logger.step = global_step
    stop_flag = {"stop": False}
    signal.signal(signal.SIGINT, lambda s,f: stop_flag.update({"stop": True}))

    w_bce, w_count, w_recon = float(cfg.hybrid_set_w_bce), float(cfg.hybrid_set_w_count), 1.0
    NUM_NEG_SAMPLES = 2048

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(range(batches_per_epoch), dynamic_ncols=True, desc=f"Epoch {epoch+1}")
        
        for _ in pbar:
            iter_start = time.perf_counter()
            
            # 1. Fetch data (RAW FIELDS)
            inputs_cpu, heads_padded_cpu, _ = next(loader)
            
            # 2. Transfer to GPU
            inputs = [x.to(device, non_blocking=True) for x in inputs_cpu]
            
            model.train()
            opt.zero_grad(set_to_none=True)
            
            with autocast():
                # E2E: Pass raw inputs to model. 
                # Model encodes them -> Z -> Trunk -> Heads
                embeddings_dict, counts_dict, recon_outputs = model(inputs, return_embeddings=True)
                
                total_set_loss = 0.0
                total_count_loss = 0.0
                
                # Decoder Regularization Loss: Compare decoded fields to inputs
                recon_loss = 0.0
                for f, p, t in zip(ds.fields, recon_outputs, inputs):
                    recon_loss += f.compute_loss(p, t) * float(f.weight)
                
                collect_coords_for_log = (global_step + 1) % recon_logger.every == 0
                coords_dict_for_log = {}
                count_targets_for_log = {}
                head_metrics = {}

                for head_name in embeddings_dict.keys():
                    raw_padded = heads_padded_cpu.get(head_name)
                    
                    if raw_padded is not None:
                        raw_padded = raw_padded.to(device, non_blocking=True)
                        mask = (raw_padded != -1)
                        t_cnt = mask.sum(dim=1, keepdim=True).float()
                        
                        # Count Loss
                        c_loss = F.mse_loss(counts_dict[head_name], t_cnt)
                        total_count_loss += c_loss
                        head_metrics[f"{head_name}_count"] = c_loss.detach()
                        
                        if collect_coords_for_log:
                            count_targets_for_log[head_name] = t_cnt

                        # Set Loss
                        rows, cols = torch.nonzero(mask, as_tuple=True)
                        global_person_ids = raw_padded[rows, cols].long()
                        
                        if global_person_ids.numel() > 0:
                            if head_name in mapping_tensors:
                                local_person_ids = mapping_tensors[head_name][global_person_ids]
                                valid_mask = (local_person_ids != -1)
                                
                                if valid_mask.any():
                                    final_rows = rows[valid_mask]
                                    final_locals = local_person_ids[valid_mask]
                                    
                                    pos_coords = torch.stack([final_rows, final_locals], dim=1)
                                    
                                    head_layer = model.people_expansions[head_name]
                                    loss_head = compute_sampled_asymmetric_loss(
                                        embedding_batch=embeddings_dict[head_name],
                                        head_layer=head_layer,
                                        positive_coords=pos_coords,
                                        num_negatives=NUM_NEG_SAMPLES
                                    )
                                    total_set_loss += loss_head
                                    head_metrics[f"{head_name}_bce"] = loss_head.detach()

                                    if collect_coords_for_log:
                                        coords_dict_for_log[head_name] = pos_coords.cpu() 

                loss = w_bce * total_set_loss + w_count * total_count_loss + w_recon * recon_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            if sched: sched.step()
            
            iter_time = time.perf_counter() - iter_start
            
            if run_logger:
                run_logger.add_scalar("loss/total", loss.item(), global_step)
                run_logger.add_scalar("loss/set_total", total_set_loss.item(), global_step)
                run_logger.add_scalar("loss/count_total", total_count_loss.item(), global_step)
                run_logger.add_scalar("loss/recon", recon_loss.item(), global_step)
                run_logger.add_scalar("time/iter", iter_time, global_step)
                current_lr = opt.param_groups[0]['lr']
                run_logger.add_scalar("lr/group_0", current_lr, global_step)
                for k, v in head_metrics.items():
                    run_logger.add_scalar(f"loss_heads/{k}", v.item(), global_step)
                run_logger.tick()
            
            if collect_coords_for_log:
                recon_logger.step(global_step, model, inputs, coords_dict_for_log, count_targets_for_log, run_logger)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", set=f"{total_set_loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
            global_step += 1
            
            if global_step % int(cfg.hybrid_set_save_interval) == 0:
                save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch, global_step, cfg)
            
            if stop_flag["stop"]: break
        if stop_flag["stop"]: break
        
        save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch+1, global_step, cfg)
        
    if run_logger: run_logger.close()

if __name__ == "__main__":
    main()