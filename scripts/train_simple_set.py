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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.amp

from config import project_config, ensure_dirs
from scripts.autoencoder.run_logger import RunLogger
from scripts.autoencoder.print_model import print_model_summary
from scripts.simple_set.model import HybridSetModel
from scripts.simple_set.data import HybridSetDataset, FastInfiniteLoader
from scripts.simple_set.precompute import ensure_hybrid_cache
from scripts.simple_set.recon import HybridSetReconLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FieldMasker:
    """
    Applies field dropout to simulate partial information during training.
    """
    def __init__(self, fields, device, probability=0.0, always_drop_names=None):
        self.fields = fields
        self.device = device
        self.prob = probability
        self.num_fields = len(fields)
        self.always_drop_indices = []
        if always_drop_names:
            for i, f in enumerate(fields):
                if f.name in always_drop_names:
                    self.always_drop_indices.append(i)
        self.pad_values = []
        for f in self.fields:
            self.pad_values.append(f.get_base_padding_value().to(device))
        self.last_mask = None

    def __call__(self, inputs):
        """
        inputs: List[Tensor], where inputs[i] corresponds to self.fields[i]
        """
        B = inputs[0].shape[0]
        keep_prob = 1.0 - self.prob
        mask = torch.bernoulli(
            torch.full((B, self.num_fields), keep_prob, device=self.device)
        )

        for idx in self.always_drop_indices:
            mask[:, idx] = 0.0

        valid_indices = [i for i in range(self.num_fields) if i not in self.always_drop_indices]
        if valid_indices:
            all_dropped = (mask.sum(dim=1) == 0)
            if all_dropped.any():
                rows_to_fix = torch.nonzero(all_dropped).squeeze(1)
                rand_select = torch.randint(
                    0,
                    len(valid_indices),
                    (rows_to_fix.size(0),),
                    device=self.device,
                )
                valid_tensor = torch.tensor(valid_indices, device=self.device)
                indices_to_keep = valid_tensor[rand_select]
                mask[rows_to_fix, indices_to_keep] = 1.0

        masked_inputs = []
        for i, (x, pad_val) in enumerate(zip(inputs, self.pad_values)):
            m_col = mask[:, i]
            view_shape = [B] + [1] * (x.dim() - 1)
            m_view = m_col.view(*view_shape)
            m_typed = m_view.to(x.dtype)
            pad_typed = pad_val.unsqueeze(0).to(x.dtype)
            x_masked = x * m_typed + pad_typed * (1 - m_typed)
            masked_inputs.append(x_masked)

        self.last_mask = mask.detach().cpu()
        return masked_inputs


def compute_sampled_asymmetric_loss(
    embedding_batch,
    head_layer,
    positive_coords,
    num_negatives=2048,
    gamma_neg=2.0,
    gamma_pos=0.0,
    eps=1e-8
):
    device = embedding_batch.device
    batch_size = embedding_batch.shape[0]
    num_vocab = head_layer.out_features

    if positive_coords.shape[0] > 0:
        pos_b_idx = positive_coords[:, 0]
        pos_p_idx = positive_coords[:, 1]

        pos_embs = embedding_batch[pos_b_idx]
        pos_w = head_layer.weight[pos_p_idx]
        pos_b = head_layer.bias[pos_p_idx] if head_layer.bias is not None else 0.0

        pos_logits = (pos_embs * pos_w).sum(dim=1) + pos_b

        pos_probs = torch.sigmoid(pos_logits)
        pos_loss_raw = -torch.log(pos_probs.clamp(min=eps))
        if gamma_pos > 0:
            pos_loss_raw = pos_loss_raw * ((1 - pos_probs).pow(gamma_pos))

        loss_pos = pos_loss_raw.sum()
    else:
        loss_pos = torch.tensor(0.0, device=device)

    neg_p_idx = torch.randint(0, num_vocab, (num_negatives,), device=device)
    neg_w = head_layer.weight[neg_p_idx]
    neg_b = head_layer.bias[neg_p_idx] if head_layer.bias is not None else 0.0

    neg_logits = embedding_batch @ neg_w.t() + neg_b

    neg_probs = torch.sigmoid(neg_logits)
    neg_loss_raw = -torch.log((1 - neg_probs).clamp(min=eps))

    if gamma_neg > 0:
        neg_loss_raw = neg_loss_raw * (neg_probs.pow(gamma_neg))

    if positive_coords.shape[0] > 0:
        collisions = (pos_p_idx.unsqueeze(1) == neg_p_idx.unsqueeze(0))
        if collisions.any():
            c_pos_idx, c_neg_idx = torch.nonzero(collisions, as_tuple=True)
            colliding_batch_rows = pos_b_idx[c_pos_idx]
            neg_loss_raw[colliding_batch_rows, c_neg_idx] = 0.0

    loss_neg = neg_loss_raw.sum()

    return (loss_pos + loss_neg) / batch_size


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
        with open(model_dir / "hybrid_set_config.json", "w") as f:
            json.dump(asdict(config), f, indent=4)
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        }
        torch.save(state, model_dir / "hybrid_set_state.pt")
        logging.info(f"Saved training state to {model_dir / 'hybrid_set_state.pt'}")
    except Exception as e:
        logging.error(f"Failed to save training state: {e}")


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
        latent_dim=int(cfg.hybrid_set_latent_dim),
        hidden_dim=int(cfg.hybrid_set_hidden_dim),
        base_output_rank=int(cfg.hybrid_set_output_rank),
        depth=int(cfg.hybrid_set_depth),
        dropout=float(cfg.hybrid_set_dropout),
        num_movies=num_movies
    )

    logging.info("Moving model to device and initializing optimizer...")
    model.to(device)
    model.movie_embeddings.to("cpu")

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

    scaler = torch.amp.GradScaler("cuda")
    run_logger = RunLogger(cfg.tensorboard_dir, "hybrid_set", cfg)
    recon_logger = HybridSetReconLogger(ds, interval_steps=int(cfg.hybrid_set_recon_interval))

    num_epochs = int(cfg.hybrid_set_epochs)
    total_steps = batches_per_epoch * num_epochs

    base_warmup = max(0, int(cfg.lr_warmup_steps))
    ratio = float(cfg.lr_warmup_ratio)
    frac_warmup = int(total_steps * ratio) if ratio > 0.0 else 0
    mass_warmup_steps = max(base_warmup, frac_warmup)
    if total_steps > 1:
        mass_warmup_steps = min(mass_warmup_steps, total_steps - 1)
    else:
        mass_warmup_steps = 0

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
            model.load_state_dict(c["model_state_dict"])
            if "optimizer_state_dict" in c:
                optimizer.load_state_dict(c["optimizer_state_dict"])
            start_epoch = c["epoch"]
            global_step = c["global_step"]
            if c.get("scheduler_state_dict") and sched:
                sched.load_state_dict(c["scheduler_state_dict"])
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
    w_align = float(cfg.hybrid_set_w_align)
    w_mass = float(getattr(cfg, "hybrid_set_w_mass", 0.01))
    w_recon = 1.0
    NUM_NEG_SAMPLES = 2048

    GAMMA_NEG = float(getattr(cfg, "hybrid_set_focal_gamma", 2.0))
    GAMMA_POS = 0.0

    field_masker = FieldMasker(
        fields=ds.fields,
        device=device,
        probability=float(getattr(cfg, "hybrid_set_field_dropout", 0.0)),
        always_drop_names=["tconst"],
    )
    logging.info(f"Field Dropout: {field_masker.prob}. Always drop: {field_masker.always_drop_indices}")
    logging.info(f"Focal Gamma (Neg): {GAMMA_NEG}. Mass Constraint Weight: {w_mass}")

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(range(batches_per_epoch), dynamic_ncols=True, desc=f"Epoch {epoch+1}")

        for _ in pbar:
            iter_start = time.perf_counter()

            inputs_cpu, heads_padded_cpu, indices_cpu = next(loader)
            inputs = [x.to(device, non_blocking=True) for x in inputs_cpu]
            inputs = field_masker(inputs)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits_dict, counts_dict, recon_table, recon_enc, (z_enc, z_table) = model(
                    field_tensors=inputs,
                    batch_indices=indices_cpu,
                    return_embeddings=True,
                )

                z_enc_norm = z_enc.norm(dim=-1).mean()
                z_table_norm = z_table.norm(dim=-1).mean()

                total_set_loss = 0.0
                total_count_loss = 0.0
                total_mass_loss = 0.0

                recon_loss = 0.0
                for f, p, t in zip(ds.fields, recon_table, inputs):
                    recon_loss += f.compute_loss(p, t) * float(f.weight)
                for f, p, t in zip(ds.fields, recon_enc, inputs):
                    recon_loss += f.compute_loss(p, t) * float(f.weight)
                recon_loss = recon_loss * 0.5

                align_loss = F.mse_loss(z_enc, z_table.detach())

                collect_coords_for_log = (global_step + 1) % recon_logger.every == 0
                coords_dict_for_log = {}
                count_targets_for_log = {}
                head_metrics = {}

                for head_name in logits_dict.keys():
                    raw_padded = heads_padded_cpu.get(head_name)
                    if raw_padded is not None:
                        raw_padded = raw_padded.to(device, non_blocking=True)
                        mask = (raw_padded != -1)
                        t_cnt = mask.sum(dim=1, keepdim=True).float()

                        c_loss = F.mse_loss(counts_dict[head_name], t_cnt)
                        total_count_loss += c_loss
                        head_metrics[f"{head_name}_count"] = c_loss.detach()

                        if collect_coords_for_log:
                            count_targets_for_log[head_name] = t_cnt

                        rows, cols = torch.nonzero(mask, as_tuple=True)
                        global_person_ids = raw_padded[rows, cols].long()
                        pos_coords = torch.empty((0, 2), dtype=torch.long, device=device)

                        if global_person_ids.numel() > 0 and head_name in mapping_tensors:
                            local_person_ids = mapping_tensors[head_name][global_person_ids]
                            valid_mask = (local_person_ids != -1)
                            if valid_mask.any():
                                final_rows = rows[valid_mask]
                                final_locals = local_person_ids[valid_mask]
                                pos_coords = torch.stack([final_rows, final_locals], dim=1)

                        head_layer = model.people_expansions[head_name]
                        bottlenecks = logits_dict[head_name]

                        loss_head = compute_sampled_asymmetric_loss(
                            embedding_batch=bottlenecks,
                            head_layer=head_layer,
                            positive_coords=pos_coords,
                            num_negatives=NUM_NEG_SAMPLES,
                            gamma_neg=GAMMA_NEG,
                            gamma_pos=GAMMA_POS,
                        )
                        total_set_loss += loss_head
                        head_metrics[f"{head_name}_bce"] = loss_head.detach()

                        if w_mass > 0.0:
                            full_logits = bottlenecks @ head_layer.weight.t()
                            if head_layer.bias is not None:
                                full_logits += head_layer.bias

                            pred_mass = torch.sigmoid(full_logits).sum(dim=1, keepdim=True)

                            m_loss = F.mse_loss(pred_mass, t_cnt)
                            total_mass_loss += m_loss
                            head_metrics[f"{head_name}_mass"] = m_loss.detach()

                        if collect_coords_for_log:
                            coords_dict_for_log[head_name] = pos_coords.cpu()

                if mass_warmup_steps > 0:
                    mass_scale = float(min(1.0, (global_step + 1) / mass_warmup_steps))
                else:
                    mass_scale = 1.0

                loss = (
                    w_bce * total_set_loss
                    + w_count * total_count_loss
                    + w_mass * mass_scale * total_mass_loss
                    + w_recon * recon_loss
                    + w_align * align_loss
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
                run_logger.add_scalar("loss/mass_total", total_mass_loss.item(), global_step)
                run_logger.add_scalar("loss/align", align_loss.item(), global_step)
                run_logger.add_scalar("loss/recon", recon_loss.item(), global_step)
                run_logger.add_scalar("time/iter", iter_time, global_step)
                run_logger.add_scalar("debug/z_enc_norm", z_enc_norm.item(), global_step)
                run_logger.add_scalar("debug/z_table_norm", z_table_norm.item(), global_step)
                run_logger.add_scalar("loss/mass_scale", mass_scale, global_step)
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
                    field_masker.last_mask,
                    run_logger,
                )

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                align=f"{align_loss.item():.4f}",
                mass=f"{total_mass_loss.item():.4f}",
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
