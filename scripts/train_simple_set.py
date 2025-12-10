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
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import project_config, ensure_dirs
from scripts.autoencoder.run_logger import RunLogger
from scripts.autoencoder.print_model import print_model_summary
from scripts.simple_set.precompute import ensure_hybrid_cache
from scripts.simple_set.model import HybridSetModel
from scripts.simple_set.data import HybridSetDataset, PersonHybridSetDataset
from scripts.simple_set.recon import HybridSetReconLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


MOVIE_COUNT_LOSS_WEIGHT = 0.1
PERSON_COUNT_LOSS_WEIGHT = 0.1


class GracefulStopper:
    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, sig, frame):
        print("\n[GracefulStopper] Signal received! Finishing current step and saving...")
        self.stop = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def make_lr_scheduler(optimizer, total_steps, schedule, warmup_steps, warmup_ratio, min_factor, last_epoch=-1):
    if total_steps is None:
        return None
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
        return max(min_factor, 1.0 - (1.0 - min_factor) * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn, last_epoch=last_epoch)


def save_checkpoint(model_dir, model, optimizer, scheduler, epoch, global_step, config):
    try:
        model_dir.mkdir(parents=True, exist_ok=True)

        config_path = model_dir / "hybrid_set_config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=4)

        tmp_path = model_dir / "hybrid_set_state.tmp"
        final_path = model_dir / "hybrid_set_state.pt"
        json_state_path = model_dir / "hybrid_set_state.json"

        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        state = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": model_state,
        }

        torch.save(state, tmp_path)
        tmp_path.replace(final_path)

        json_state = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "max_epochs": int(config.hybrid_set_epochs),
        }
        with open(json_state_path, "w") as f_json:
            json.dump(json_state, f_json, indent=4)

        logging.info(f"Saved training state to {final_path} and {json_state_path}")
    except Exception as e:
        logging.error(f"Failed to save training state: {e}")


def _count_module_params(module: nn.Module):
    total = 0
    trainable = 0
    for p in module.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def print_param_summary(model: nn.Module):
    total_params, total_trainable = _count_module_params(model)

    movie_field_params, movie_field_trainable = _count_module_params(model.movie_field_decoder)
    person_field_params, person_field_trainable = _count_module_params(model.person_field_decoder)

    movie_head_params, movie_head_trainable = _count_module_params(model.movie_heads)
    person_head_params, person_head_trainable = _count_module_params(model.person_heads)

    movie_emb_params, movie_emb_trainable = _count_module_params(model.movie_embeddings)
    person_emb_params, person_emb_trainable = _count_module_params(model.person_embeddings)

    print("\n" + "=" * 50)
    print("Parameter summary by component")
    print("-" * 50)

    print("Field reconstruction modules:")
    print(f"  movie_field_decoder   total={movie_field_params:10d}  trainable={movie_field_trainable:10d}")
    print(f"  person_field_decoder  total={person_field_params:10d}  trainable={person_field_trainable:10d}")

    print("\nSet relation heads:")
    print(f"  movie_heads           total={movie_head_params:10d}  trainable={movie_head_trainable:10d}")
    print(f"  person_heads          total={person_head_params:10d}  trainable={person_head_trainable:10d}")

    print("\nEmbedding tables:")
    print(f"  movie_embeddings      total={movie_emb_params:10d}  trainable={movie_emb_trainable:10d}")
    print(f"  person_embeddings     total={person_emb_params:10d}  trainable={person_emb_trainable:10d}")

    accounted_total = (
        movie_field_params
        + person_field_params
        + movie_head_params
        + person_head_params
        + movie_emb_params
        + person_emb_params
    )
    accounted_trainable = (
        movie_field_trainable
        + person_field_trainable
        + movie_head_trainable
        + person_head_trainable
        + movie_emb_trainable
        + person_emb_trainable
    )

    print("\n" + "-" * 50)
    print(f"{'Accounted total params:':25s} {accounted_total:10d}")
    print(f"{'Accounted trainable:':25s} {accounted_trainable:10d}")
    print(f"{'Model total params:':25s} {total_params:10d}")
    print(f"{'Model trainable:':25s} {total_trainable:10d}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    args = parser.parse_args()

    cfg = project_config
    ensure_dirs(cfg)
    if args.no_resume:
        cfg.refresh_cache = False

    cache_path = ensure_hybrid_cache(cfg)

    movie_ds = HybridSetDataset(str(cache_path), cfg)
    person_ds = PersonHybridSetDataset(str(cache_path), cfg)

    movie_loader = DataLoader(
        movie_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    person_loader = DataLoader(
        person_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridSetModel(
        movie_fields=movie_ds.fields,
        person_fields=person_ds.fields,
        num_movies=len(movie_ds),
        num_people=movie_ds.num_people,
        heads_config=cfg.hybrid_set_heads,
        movie_head_vocab_sizes=movie_ds.head_vocab_sizes,
        movie_head_local_to_global=movie_ds.head_local_to_global,
        person_head_vocab_sizes=person_ds.head_vocab_sizes,
        person_head_local_to_global=person_ds.person_head_local_to_global,
        movie_dim=cfg.hybrid_set_movie_dim,
        hidden_dim=cfg.hybrid_set_hidden_dim,
        person_dim=cfg.hybrid_set_person_dim,
        dropout=cfg.hybrid_set_dropout,
        hybrid_set_logit_scale=cfg.hybrid_set_logit_scale,
    )
    model.to(device)

    emb_params = list(model.movie_embeddings.parameters()) + list(model.person_embeddings.parameters())
    emb_param_ids = {id(p) for p in emb_params}
    model_params = [p for p in model.parameters() if id(p) not in emb_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model_params,
                "lr": cfg.hybrid_set_model_lr,
                "weight_decay": cfg.hybrid_set_weight_decay,
            },
            {
                "params": emb_params,
                "lr": cfg.hybrid_set_emb_lr,
                "weight_decay": cfg.hybrid_set_weight_decay,
            },
        ]
    )

    sample_movie_idx = next(iter(movie_loader))
    sample_person_idx = next(iter(person_loader))

    sample_inputs = {
        "movie_indices": sample_movie_idx.to(device),
        "person_indices": sample_person_idx.to(device),
    }

    print("\n" + "=" * 50)
    print_model_summary(model, sample_inputs)
    print("=" * 50 + "\n")

    print_param_summary(model)

    movie_mapping_tensors = {}
    if hasattr(movie_ds, "head_mappings"):
        for name, t in movie_ds.head_mappings.items():
            movie_mapping_tensors[name] = t.to(device)

    person_mapping_tensors = {}
    if hasattr(person_ds, "head_mappings"):
        for name, t in person_ds.head_mappings.items():
            person_mapping_tensors[name] = t.to(device)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler("cpu")
    run_logger = RunLogger(cfg.tensorboard_dir, "hybrid_set", cfg)
    recon_logger = HybridSetReconLogger(
        movie_ds,
        person_ds,
        interval_steps=int(cfg.hybrid_set_recon_interval),
    )

    criterion_set = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean").to(device)

    num_epochs = int(cfg.hybrid_set_epochs)
    movie_steps = len(movie_loader)
    person_steps = len(person_loader)
    max_steps_per_epoch = max(movie_steps, person_steps)
    total_steps = max_steps_per_epoch * num_epochs

    sched = make_lr_scheduler(
        optimizer,
        total_steps,
        cfg.lr_schedule,
        cfg.lr_warmup_steps,
        cfg.lr_warmup_ratio,
        cfg.lr_min_factor,
    )

    start_epoch, global_step = 0, 0
    ckpt_path = Path(cfg.model_dir) / "hybrid_set_state.pt"
    json_state_path = Path(cfg.model_dir) / "hybrid_set_state.json"

    if ckpt_path.exists() and not args.no_resume:
        try:
            c = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(c["model_state_dict"])
            start_epoch = c.get("epoch", 0)
            global_step = c.get("global_step", 0)

            if json_state_path.exists():
                try:
                    with open(json_state_path, "r") as f:
                        state_override = json.load(f)
                    start_epoch = int(state_override.get("epoch", start_epoch))
                    global_step = int(state_override.get("global_step", global_step))
                except Exception as e:
                    logging.warning(f"Failed to load JSON state from {json_state_path}: {e}")

            logging.info(f"Resumed from epoch {start_epoch}, global_step {global_step}")
        except Exception as e:
            logging.warning(f"Failed to resume from checkpoint: {e}")

    stopper = GracefulStopper()

    for epoch in range(start_epoch, num_epochs):
        movie_iter = iter(movie_loader)
        person_iter = iter(person_loader)

        pbar = tqdm(range(max_steps_per_epoch), dynamic_ncols=True, desc=f"Epoch {epoch+1}")
        for _ in pbar:
            if stopper.stop:
                pbar.close()
                break

            try:
                batch_movie_idx = next(movie_iter)
            except StopIteration:
                batch_movie_idx = None

            try:
                batch_person_idx = next(person_iter)
            except StopIteration:
                batch_person_idx = None

            if batch_movie_idx is None and batch_person_idx is None:
                continue

            model.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type):
                total_loss = None

                movie_set_loss = torch.tensor(0.0, device=device)
                movie_recon_loss = torch.tensor(0.0, device=device)
                person_set_loss = torch.tensor(0.0, device=device)
                person_recon_loss = torch.tensor(0.0, device=device)

                if batch_movie_idx is not None:
                    idx_m = batch_movie_idx.to(device, non_blocking=True).long()
                    outputs_m = model(movie_indices=idx_m)
                    movie_out = outputs_m.get("movie")
                    if movie_out is not None:
                        logits_dict_m, recon_table_m, _, _ = movie_out
                        movie_inputs = [
                            t[batch_movie_idx].to(device, non_blocking=True)
                            for t in movie_ds.stacked_fields
                        ]

                        for f, p_pred, t_in in zip(movie_ds.fields, recon_table_m, movie_inputs):
                            movie_recon_loss = movie_recon_loss + f.compute_loss(p_pred, t_in) * f.weight

                        for head_name, logits in logits_dict_m.items():
                            mapping = movie_mapping_tensors.get(head_name)
                            padded = movie_ds.heads_padded.get(head_name)
                            if mapping is None or padded is None:
                                continue
                            raw_padded = padded[batch_movie_idx]
                            raw_padded = raw_padded.to(device, non_blocking=True)

                            mask = raw_padded != -1
                            rows, cols = torch.nonzero(mask, as_tuple=True)

                            targets = torch.zeros_like(logits)
                            if rows.numel() > 0:
                                glob = raw_padded[rows, cols].long()
                                loc = mapping[glob]
                                valid = loc != -1
                                if valid.any():
                                    rows_v, loc_v = rows[valid], loc[valid]
                                    targets[rows_v, loc_v] = 1.0

                            head_set_loss = criterion_set(logits, targets)

                            probs = torch.sigmoid(logits)
                            true_counts = targets.sum(dim=-1)
                            soft_counts = probs.sum(dim=-1)
                            count_loss = F.mse_loss(soft_counts, true_counts)

                            movie_set_loss = movie_set_loss + head_set_loss + MOVIE_COUNT_LOSS_WEIGHT * count_loss

                        movie_loss_total = cfg.hybrid_set_w_bce * movie_set_loss + cfg.hybrid_set_w_recon * movie_recon_loss
                        total_loss = movie_loss_total if total_loss is None else total_loss + movie_loss_total

                if batch_person_idx is not None:
                    idx_p = batch_person_idx.to(device, non_blocking=True).long()
                    outputs_p = model(person_indices=idx_p)
                    person_out = outputs_p.get("person")
                    if person_out is not None:
                        logits_dict_p, recon_table_p, _, _ = person_out
                        person_inputs = [
                            t[batch_person_idx].to(device, non_blocking=True)
                            for t in person_ds.stacked_fields
                        ]

                        for f, p_pred, t_in in zip(person_ds.fields, recon_table_p, person_inputs):
                            person_recon_loss = person_recon_loss + f.compute_loss(p_pred, t_in) * f.weight

                        for head_name, logits in logits_dict_p.items():
                            mapping = person_mapping_tensors.get(head_name)
                            padded = person_ds.heads_padded.get(head_name)
                            if mapping is None or padded is None:
                                continue
                            raw_padded = padded[batch_person_idx]
                            raw_padded = raw_padded.to(device, non_blocking=True)

                            mask = raw_padded != -1
                            rows, cols = torch.nonzero(mask, as_tuple=True)

                            targets = torch.zeros_like(logits)
                            if rows.numel() > 0:
                                glob = raw_padded[rows, cols].long()
                                loc = mapping[glob]
                                valid = loc != -1
                                if valid.any():
                                    rows_v, loc_v = rows[valid], loc[valid]
                                    targets[rows_v, loc_v] = 1.0

                            head_set_loss = criterion_set(logits, targets)

                            probs = torch.sigmoid(logits)
                            true_counts = targets.sum(dim=-1)
                            soft_counts = probs.sum(dim=-1)
                            count_loss = F.mse_loss(soft_counts, true_counts)

                            person_set_loss = person_set_loss + head_set_loss + PERSON_COUNT_LOSS_WEIGHT * count_loss

                        person_loss_total = cfg.hybrid_set_w_bce * person_set_loss + cfg.hybrid_set_w_recon * person_recon_loss
                        total_loss = person_loss_total if total_loss is None else total_loss + person_loss_total

                if total_loss is None:
                    continue

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if sched:
                sched.step()

            if run_logger:
                run_logger.add_scalar("loss/total", total_loss.item(), global_step)
                run_logger.add_scalar("loss/movie_set", movie_set_loss.item(), global_step)
                run_logger.add_scalar("loss/movie_recon", movie_recon_loss.item(), global_step)
                run_logger.add_scalar("loss/person_set", person_set_loss.item(), global_step)
                run_logger.add_scalar("loss/person_recon", person_recon_loss.item(), global_step)
                run_logger.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], global_step)
                run_logger.tick()

            recon_logger.step(global_step, model, run_logger)

            pbar.set_postfix(loss=f"{total_loss.item():.4f}")
            global_step += 1

            if global_step % cfg.hybrid_set_save_interval == 0:
                save_checkpoint(Path(cfg.model_dir), model, optimizer, sched, epoch, global_step, cfg)

        if stopper.stop:
            print("[GracefulStopper] Training loop exited. Final save...")
            save_checkpoint(Path(cfg.model_dir), model, optimizer, sched, epoch, global_step, cfg)
            break

        save_checkpoint(Path(cfg.model_dir), model, optimizer, sched, epoch + 1, global_step, cfg)

    save_checkpoint(
        Path(cfg.model_dir),
        model,
        optimizer,
        sched,
        epoch if stopper.stop else num_epochs,
        global_step,
        cfg,
    )
    if run_logger:
        run_logger.close()
    print("Done.")


if __name__ == "__main__":
    main()
