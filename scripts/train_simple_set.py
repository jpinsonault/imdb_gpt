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


def make_lr_scheduler(optimizer, total_steps, schedule, warmup_steps, warmup_ratio, min_factor, last_epoch=-1):
    if total_steps is None:
        return None
    total_steps = max(1, int(total_steps))
    schedule = (schedule or "").lower()
    if schedule not in ("cosine", "linear"):
        return None

    base_warmup = max(0, int(warmup_steps))
    ratio = float(warmup_ratio)
    frac_warmup = int(total_steps * ratio) if ratio > 0.0 else 0
    w_steps = max(base_warmup, frac_warmup)
    w_steps = min(w_steps, total_steps - 1) if total_steps > 1 else 0
    min_factor = float(min_factor)

    def cosine_lambda(step):
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

    def linear_lambda(step):
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

    lr_lambda = cosine_lambda if schedule == "cosine" else linear_lambda
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def save_checkpoint(model_dir, model, optimizer, scheduler, epoch, global_step, config):
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / "hybrid_set_config.json"
        state_path = model_dir / "hybrid_set_state.pt"

        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=4)

        model_state = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        }

        state = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": model_state,
        }

        torch.save(state, state_path)
        logging.info(f"Saved training state to {state_path}")
    except Exception as e:
        logging.error(f"Failed to save training state: {e}")


def binary_focal_loss_with_logits(logits, targets, gamma):
    if gamma <= 0.0:
        return F.binary_cross_entropy_with_logits(logits, targets)
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    loss = modulating * ce_loss
    return loss.mean()


def build_focus_mask(logits, pos_mask, topk_negatives):
    if topk_negatives <= 0:
        return pos_mask
    if logits.numel() == 0:
        return pos_mask

    b, v = logits.shape
    if v == 0:
        return pos_mask

    k = min(int(topk_negatives), v)

    with torch.no_grad():
        logits_detached = logits.detach()
        if logits_detached.dtype in (torch.float16, torch.bfloat16):
            logits_work = logits_detached.float()
        else:
            logits_work = logits_detached

        logits_neg = logits_work.masked_fill(pos_mask, float("-inf"))
        _, topk_idx = torch.topk(logits_neg, k, dim=1)

    topk_mask = torch.zeros_like(pos_mask)
    topk_mask.scatter_(1, topk_idx, True)

    focus_mask = pos_mask | topk_mask
    return focus_mask


def mass_loss_from_logits(logits, true_counts):
    pred_mass = torch.sigmoid(logits).sum(dim=-1, keepdim=True)
    pred_log = torch.log1p(pred_mass)
    true_log = torch.log1p(true_counts)
    return F.mse_loss(pred_log, true_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint and start fresh")
    parser.add_argument("--debug", action="store_true", help="Enable anomaly detection")
    args = parser.parse_args()

    if args.debug:
        logging.warning("DEBUG MODE: Enabling anomaly detection")
        torch.autograd.set_detect_anomaly(True)

    cfg = project_config
    ensure_dirs(cfg)

    if args.no_resume:
        cfg.refresh_cache = False

    cache_path = ensure_hybrid_cache(cfg)
    ds = HybridSetDataset(str(cache_path), cfg)

    num_movies = len(ds)
    logging.info(f"Dataset contains {num_movies} items.")

    loader = FastInfiniteLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    batches_per_epoch = len(loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridSetModel(
        fields=ds.fields,
        num_people=ds.num_people,
        heads_config=cfg.hybrid_set_heads,
        head_vocab_sizes=ds.head_vocab_sizes,
        head_group_offsets=ds.head_group_offsets,
        num_groups=ds.num_groups,
        latent_dim=int(cfg.hybrid_set_latent_dim),
        hidden_dim=int(cfg.hybrid_set_hidden_dim),
        base_output_rank=int(cfg.hybrid_set_output_rank),
        depth=int(cfg.hybrid_set_depth),
        dropout=float(cfg.hybrid_set_dropout),
        num_movies=num_movies,
    )

    logging.info("Moving model to device and initializing optimizer...")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.hybrid_set_lr),
        weight_decay=float(cfg.hybrid_set_weight_decay),
    )

    sample_inputs_cpu, _, sample_idx = next(loader)
    sample_inputs = [x.to(device) for x in sample_inputs_cpu]

    print("\n" + "=" * 50)
    print_model_summary(model, {"field_tensors": sample_inputs, "batch_indices": sample_idx})
    print("=" * 50 + "\n")

    mapping_tensors = {}
    if hasattr(ds, "head_mappings"):
        for name, t in ds.head_mappings.items():
            mapping_tensors[name] = t.to(device)

    head_true_counts = {}
    if hasattr(ds, "heads_padded"):
        for head_name, padded in ds.heads_padded.items():
            mask = padded != -1
            head_true_counts[head_name] = mask.sum(dim=1, keepdim=True)

    scaler = torch.amp.GradScaler("cuda")
    from scripts.simple_set.recon import HybridSetReconLogger
    run_logger = RunLogger(cfg.tensorboard_dir, "hybrid_set", cfg)
    recon_logger = HybridSetReconLogger(ds, interval_steps=int(cfg.hybrid_set_recon_interval))

    num_epochs = int(cfg.hybrid_set_epochs)
    total_steps = batches_per_epoch * num_epochs

    sched = make_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        schedule=cfg.lr_schedule,
        warmup_steps=cfg.lr_warmup_steps,
        warmup_ratio=cfg.lr_warmup_ratio,
        min_factor=cfg.lr_min_factor,
    )

    start_epoch, global_step = 0, 0
    ckpt_path = Path(cfg.model_dir) / "hybrid_set_state.pt"
    if ckpt_path.exists() and not args.no_resume:
        try:
            c = torch.load(ckpt_path, map_location=device)
            state = c.get("model_state_dict", {})
            if state:
                model.load_state_dict(state)
            start_epoch = int(c.get("epoch", 0))
            global_step = int(c.get("global_step", 0))
            logging.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logging.error(f"Resume failed: {e}")
    elif args.no_resume:
        logging.info("Starting new run (ignoring existing checkpoint).")

    if run_logger.run_dir:
        run_logger.step = global_step
    stop_flag = {"stop": False}
    signal.signal(signal.SIGINT, lambda s, f: stop_flag.update({"stop": True}))

    w_bce = float(cfg.hybrid_set_w_bce)
    w_count = float(cfg.hybrid_set_w_count)
    w_recon = 1.0
    w_title = float(cfg.hybrid_set_w_title)

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(range(batches_per_epoch), dynamic_ncols=True, desc=f"Epoch {epoch+1}")

        for _ in pbar:
            iter_start = time.perf_counter()

            inputs_cpu, heads_padded_cpu, indices_cpu = next(loader)
            inputs = [x.to(device, non_blocking=True) for x in inputs_cpu]

            model.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits_dict, counts_dict, recon_table, z_table = model(
                    field_tensors=inputs,
                    batch_indices=indices_cpu,
                    return_embeddings=False,
                )

                z_table_norm = z_table.norm(dim=-1).mean()

                total_set_loss = 0.0
                total_count_loss = 0.0

                recon_loss = 0.0
                for f, p, t in zip(ds.fields, recon_table, inputs):
                    recon_loss = recon_loss + f.compute_loss(p, t) * float(f.weight)

                title_latents = model.encode_titles(inputs)
                z_target = z_table.detach()
                z_target_norm = F.normalize(z_target, p=2, dim=-1)
                z_title_norm = F.normalize(title_latents, p=2, dim=-1)
                title_loss = F.mse_loss(z_title_norm, z_target_norm)

                collect_coords_for_log = (global_step + 1) % recon_logger.every == 0
                coords_dict_for_log = {}
                count_targets_for_log = {}
                head_metrics = {}

                for head_name, logits in logits_dict.items():
                    raw_padded_batch_cpu = heads_padded_cpu.get(head_name)
                    if raw_padded_batch_cpu is None:
                        continue

                    raw_padded_batch = raw_padded_batch_cpu.to(device, non_blocking=True)
                    mask_batch = raw_padded_batch != -1

                    t_cnt_full = head_true_counts.get(head_name)
                    if t_cnt_full is not None:
                        t_cnt_batch = t_cnt_full[indices_cpu].to(device, non_blocking=True).float()
                        c_loss = F.mse_loss(counts_dict[head_name], t_cnt_batch)
                        total_count_loss = total_count_loss + c_loss
                        head_metrics[f"{head_name}_count"] = c_loss.detach()
                        if collect_coords_for_log:
                            count_targets_for_log[head_name] = t_cnt_full
                    else:
                        if collect_coords_for_log:
                            count_targets_for_log[head_name] = None

                    rows_b, cols_b = torch.nonzero(mask_batch, as_tuple=True)
                    targets = torch.zeros_like(logits)

                    coords_for_log = torch.empty(0, 2, dtype=torch.long, device=device)

                    head_map = mapping_tensors.get(head_name)

                    if rows_b.numel() > 0 and head_map is not None:
                        global_person_ids = raw_padded_batch[rows_b, cols_b].long()
                        local_person_ids = head_map[global_person_ids]

                        valid = local_person_ids != -1
                        if valid.any():
                            rows_b = rows_b[valid]
                            local_person_ids = local_person_ids[valid]

                            targets[rows_b, local_person_ids] = 1.0

                            if collect_coords_for_log:
                                idx_batch = indices_cpu.to(device, non_blocking=True)
                                ds_rows = idx_batch[rows_b]
                                coords_for_log = torch.stack(
                                    [ds_rows, local_person_ids],
                                    dim=1,
                                )

                    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
                    total_set_loss = total_set_loss + bce_loss
                    head_metrics[f"{head_name}_bce"] = bce_loss.detach()

                    if collect_coords_for_log and coords_for_log.numel() > 0:
                        coords_dict_for_log[head_name] = coords_for_log.detach().cpu()

                loss = (
                    w_bce * total_set_loss
                    + w_count * total_count_loss
                    + w_recon * recon_loss
                    + w_title * title_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if sched:
                sched.step()
            iter_time = time.perf_counter() - iter_start

            if run_logger:
                run_logger.add_scalar("loss/total", loss.item(), global_step)
                run_logger.add_scalar("loss/set_total", total_set_loss.item(), global_step)
                run_logger.add_scalar("loss/recon", recon_loss.item(), global_step)
                run_logger.add_scalar("loss/title", title_loss.item(), global_step)
                run_logger.add_scalar("time/iter", iter_time, global_step)
                run_logger.add_scalar("debug/z_table_norm", z_table_norm.item(), global_step)

                current_lr = optimizer.param_groups[0]["lr"]
                run_logger.add_scalar("opt/lr", current_lr, global_step)

                for k, v in head_metrics.items():
                    run_logger.add_scalar(f"loss_heads/{k}", v.item(), global_step)

                run_logger.tick()

            if collect_coords_for_log:
                recon_logger.step(
                    global_step,
                    model,
                    inputs,
                    indices_cpu,
                    coords_dict_for_log,
                    count_targets_for_log,
                    run_logger,
                )

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            global_step += 1

            if global_step % int(cfg.hybrid_set_save_interval) == 0:
                save_checkpoint(Path(cfg.model_dir), model, optimizer, sched, epoch, global_step, cfg)

            if stop_flag["stop"]:
                break
        if stop_flag["stop"]:
            break

        save_checkpoint(Path(cfg.model_dir), model, optimizer, sched, epoch + 1, global_step, cfg)

    if run_logger:
        run_logger.close()


if __name__ == "__main__":
    main()
