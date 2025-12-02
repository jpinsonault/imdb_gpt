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
import torch.nn as nn
from tqdm import tqdm

from config import project_config, ensure_dirs, ProjectConfig
from scripts.autoencoder.ae_loader import _load_trainable_autoencoders, _load_frozen_autoencoders
from scripts.autoencoder.run_logger import RunLogger
from scripts.set_decoder.model import SequenceDecoder
from scripts.set_decoder.recon_logger import SetReconstructionLogger
from scripts.precompute_set_cache import ensure_set_decoder_cache
from scripts.set_decoder.data import CachedSequenceDataset, collate_seq_decoder
from scripts.set_decoder.training import compute_sequence_losses

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EndToEndSetModel(nn.Module):
    def __init__(self, mov_ae, per_ae, seq_dec):
        super().__init__()
        self.mov_ae = mov_ae
        self.per_ae = per_ae
        self.seq_dec = seq_dec

    def forward(self, m_inputs, p_inputs):
        """
        Forward pass through Encoders -> Set Decoder.
        """
        # 1. Encode Movie
        # mov_ae.encoder expects List[Tensor]
        z_movie = self.mov_ae.encoder(m_inputs)

        # 2. Encode People Sequence
        # p_inputs is List[Tensor(B, Seq, ...)]
        # Encoder expects List[Tensor(Total, ...)], so we flatten B*Seq
        B, SeqLen = p_inputs[0].shape[:2]
        
        flat_p_inputs = []
        for f_tensor in p_inputs:
            # Flatten B, Seq -> B*Seq
            flat_shape = (-1,) + f_tensor.shape[2:]
            flat_p_inputs.append(f_tensor.reshape(flat_shape))
            
        z_p_flat = self.per_ae.encoder(flat_p_inputs)
        z_p_seq = z_p_flat.view(B, SeqLen, -1)

        # 3. Set Decoder
        z_pred, pres_logits = self.seq_dec(z_movie, z_p_seq)
        
        return z_pred, pres_logits, z_movie, z_p_seq

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
    if total_steps > 1:
        w_steps = min(w_steps, total_steps - 1)
    else:
        w_steps = 0
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

def save_checkpoint(model_dir, model, optimizer, scheduler, epoch, global_step, config, best_loss):
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "seq_decoder_config.json", "w") as f:
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
        torch.save(state, model_dir / "seq_decoder_state.pt")
        logging.info(f"Saved training state to {model_dir / 'seq_decoder_state.pt'}")
    except Exception as e:
        logging.error(f"Failed to save training state: {e}")

def build_seq_decoder_trainer(cfg: ProjectConfig, db_path: str):
    fine_tune = cfg.seq_decoder_fine_tune_ae
    
    if fine_tune:
        logging.info("Building End-to-End Fine-Tuning Trainer (AEs + SetDecoder)")
        mov_ae, per_ae = _load_trainable_autoencoders(cfg)
    else:
        logging.info("Building Frozen AE Trainer")
        mov_ae, per_ae = _load_frozen_autoencoders(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    # CRITICAL FIX: Move the wrappers (encoder/decoder) directly. 
    # Do not use .model.to(device) because .model.enc is the stale base encoder.
    mov_ae.encoder.to(device)
    mov_ae.decoder.to(device)
    per_ae.encoder.to(device)
    per_ae.decoder.to(device)

    # Re-verify gradient state
    if fine_tune:
        mov_ae.encoder.train()
        per_ae.encoder.train()
        for p in mov_ae.encoder.parameters(): p.requires_grad_(True)
        for p in per_ae.encoder.parameters(): p.requires_grad_(True)
    else:
        mov_ae.encoder.eval()
        per_ae.encoder.eval()

    cache_path = ensure_set_decoder_cache(cfg)
    max_len = int(getattr(cfg, "seq_decoder_len", 10))
    ds = CachedSequenceDataset(str(cache_path), max_len=max_len)

    from torch.utils.data import DataLoader
    num_workers = int(getattr(cfg, "num_workers", 0) or 0)
    
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_seq_decoder,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    latent_dim = int(getattr(mov_ae, "latent_dim", cfg.latent_dim))
    
    # Autoregressive Model
    seq_dec = SequenceDecoder(
        latent_dim=latent_dim,
        max_len=max_len,
        hidden_dim=int(getattr(cfg, "seq_decoder_hidden_dim", 256)),
        num_layers=int(getattr(cfg, "seq_decoder_layers", 6)),
        num_heads=int(getattr(cfg, "seq_decoder_heads", 8)),
        dropout=float(getattr(cfg, "seq_decoder_dropout", 0.1)),
    ).to(device)

    # Wrap in EndToEnd model
    model = EndToEndSetModel(mov_ae, per_ae, seq_dec).to(device)

    # Parameters & Optimizer
    # If fine-tuning, we use a separate LR for AEs
    params = []
    
    # Set Decoder Params (Group 1)
    params.append({
        "params": seq_dec.parameters(),
        "lr": float(getattr(cfg, "seq_decoder_lr", 3e-4))
    })

    if fine_tune:
        ae_lr = float(getattr(cfg, "seq_decoder_ae_lr", 1e-5))
        
        # Group 2: All Autoencoder Parameters
        # CRITICAL FIX: We use a set() to collect parameters because the JointAutoencoder
        # architecture shares the 'FiLM' network weights between mov_ae.encoder and per_ae.encoder.
        # Passing them as separate lists triggers "ValueError: some parameters appear in more than one parameter group".
        
        ae_params = set()
        ae_params.update(mov_ae.encoder.parameters())
        ae_params.update(mov_ae.decoder.parameters())
        ae_params.update(per_ae.encoder.parameters())
        ae_params.update(per_ae.decoder.parameters())
        
        params.append({"params": list(ae_params), "lr": ae_lr})

    opt = torch.optim.AdamW(
        params,
        weight_decay=float(getattr(cfg, "seq_decoder_weight_decay", 1e-2)),
    )

    run_logger = RunLogger(cfg.tensorboard_dir, "seq_decoder", cfg)
    
    recon_logger = SetReconstructionLogger(
        model=seq_dec, movie_ae=mov_ae, people_ae=per_ae,
        num_slots=max_len,
        interval_steps=int(getattr(cfg, "seq_decoder_callback_interval", 500)),
        num_samples=int(getattr(cfg, "seq_decoder_recon_samples", 3)),
        table_width=int(getattr(cfg, "seq_decoder_table_width", 60)),
    )

    loss_cfg = {
        "w_latent": float(getattr(cfg, "seq_decoder_w_latent", 1.0)),
        "w_recon": float(getattr(cfg, "seq_decoder_w_recon", 1.0)),
        "w_presence": float(getattr(cfg, "seq_decoder_w_presence", 1.0)),
        "w_ae_anchor": float(getattr(cfg, "seq_decoder_w_ae_recon", 0.5)), # Anchor loss weight
    }

    return model, opt, loader, mov_ae, per_ae, run_logger, recon_logger, loss_cfg

def main():
    parser = argparse.ArgumentParser(description="Train sequence decoder")
    parser.add_argument("--new-run", action="store_true", help="Start fresh")
    args = parser.parse_args()

    cfg = project_config
    ensure_dirs(cfg)
    db_path = cfg.db_path
    
    fine_tune = cfg.seq_decoder_fine_tune_ae
    warmup_epochs = cfg.seq_decoder_ae_warmup_epochs if fine_tune else 0

    model, opt, loader, mov_ae, per_ae, run_logger, recon_logger, loss_cfg = build_seq_decoder_trainer(cfg, db_path=db_path)
    device = next(model.parameters()).device
    
    num_epochs = int(getattr(cfg, "seq_decoder_epochs", 50))
    save_interval = int(getattr(cfg, "seq_decoder_save_interval", 1000))
    
    # Loss Weights
    w_latent, w_recon = loss_cfg["w_latent"], loss_cfg["w_recon"]
    w_presence = loss_cfg["w_presence"]
    w_anchor = loss_cfg["w_ae_anchor"]

    sched = make_lr_scheduler(opt, len(loader) * num_epochs, cfg.lr_schedule, cfg.lr_warmup_steps, cfg.lr_warmup_ratio, cfg.lr_min_factor)

    start_epoch = 0
    global_step = 0
    best_loss = None
    checkpoint_path = Path(cfg.model_dir) / "seq_decoder_state.pt"

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
        pbar = tqdm(loader, dynamic_ncols=True, desc=f"seq-dec epoch {epoch+1}/{num_epochs}")
        
        # --- Handle AE Freezing/Unfreezing based on Epoch ---
        if fine_tune:
            if epoch < warmup_epochs:
                # WARMUP PHASE: Freeze AEs (TARGET .encoder/.decoder explicitly)
                model.mov_ae.encoder.eval()
                model.mov_ae.decoder.eval()
                model.per_ae.encoder.eval()
                model.per_ae.decoder.eval()
                
                for p in model.mov_ae.encoder.parameters(): p.requires_grad_(False)
                for p in model.mov_ae.decoder.parameters(): p.requires_grad_(False)
                for p in model.per_ae.encoder.parameters(): p.requires_grad_(False)
                for p in model.per_ae.decoder.parameters(): p.requires_grad_(False)
                
                pbar.set_description(f"epoch {epoch+1} (AE Frozen/Warmup)")
            else:
                # JOINT PHASE: Unfreeze AEs
                model.mov_ae.encoder.train()
                model.mov_ae.decoder.train()
                model.per_ae.encoder.train()
                model.per_ae.decoder.train()
                
                for p in model.mov_ae.encoder.parameters(): p.requires_grad_(True)
                for p in model.mov_ae.decoder.parameters(): p.requires_grad_(True)
                for p in model.per_ae.encoder.parameters(): p.requires_grad_(True)
                for p in model.per_ae.decoder.parameters(): p.requires_grad_(True)
                
                pbar.set_description(f"epoch {epoch+1} (AE Fine-Tuning)")
        # ----------------------------------------------------

        for batch in pbar:
            iter_start = time.perf_counter()

            # batch is M_raw, P_in_raw, P_tgt_raw, mask
            M_raw, P_in_raw, P_tgt_raw, mask = batch
            M_raw = [x.to(device, non_blocking=True) for x in M_raw]
            P_in_raw = [x.to(device, non_blocking=True) for x in P_in_raw]
            P_tgt_raw = [x.to(device, non_blocking=True) for x in P_tgt_raw]
            mask = mask.to(device, non_blocking=True)

            # We use model.train() for the SeqDecoder, but the AE state is controlled by the logic above.
            model.seq_dec.train() 
            opt.zero_grad()
            
            # 1. Forward Pass (End-to-End)
            # z_movie and z_p_seq are generated dynamically from raw inputs!
            z_pred, pres_logits, z_movie, z_p_seq = model(M_raw, P_in_raw)
            
            # 2. Sequence Losses (Set Prediction)
            # We compare z_pred (from decoder) against z_p_seq (from encoder)
            
            loss_latent, loss_recon = compute_sequence_losses(
                per_ae, z_pred, z_p_seq, P_tgt_raw, mask, w_latent, w_recon
            )
            
            loss_presence = torch.nn.functional.binary_cross_entropy_with_logits(
                pres_logits, mask.float()
            )

            # 3. Anchor Losses (Autoencoder Reconstruction)
            # This is the "Anti-Clobbering" mechanism.
            loss_anchor_m = torch.tensor(0.0, device=device)
            loss_anchor_p = torch.tensor(0.0, device=device)

            if fine_tune and epoch >= warmup_epochs:
                # Movie Recon Loss
                m_rec = model.mov_ae.decoder(z_movie)
                for f, pred, tgt in zip(model.mov_ae.fields, m_rec, M_raw):
                    loss_anchor_m += f.compute_loss(pred, tgt) * f.weight
                
                # Person Recon Loss
                # We have (B, Seq, D). We flatten to (B*Seq, D) and reconstruct P_tgt_raw (flattened)
                B, Seq = z_p_seq.shape[:2]
                z_p_flat = z_p_seq.reshape(B*Seq, -1)
                p_rec_flat = model.per_ae.decoder(z_p_flat)
                
                # Flatten targets
                flat_tgt = [t.reshape(-1, *t.shape[2:]) for t in P_tgt_raw]
                
                for f, pred, tgt in zip(model.per_ae.fields, p_rec_flat, flat_tgt):
                    l = f.compute_loss(pred, tgt)
                    loss_anchor_p += l * f.weight

            total_loss = (
                w_latent * loss_latent +
                w_recon * loss_recon +
                w_presence * loss_presence +
                w_anchor * (loss_anchor_m + loss_anchor_p)
            )
            
            # Metrics
            preds_bin = (torch.sigmoid(pres_logits) > 0.5)
            correct_presence = ((preds_bin == mask).float().sum()) / mask.numel()
            
            log_metrics = {
                'loss_latent': loss_latent.item(),
                'loss_recon': loss_recon.item(),
                'loss_presence': loss_presence.item(),
                'loss_anchor_m': loss_anchor_m.item(),
                'loss_anchor_p': loss_anchor_p.item(),
                'acc_presence': correct_presence.item()
            }
            
            total_loss.backward()
            
            # Clip Gradients to prevent explosion during early fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            opt.step()
            if sched: sched.step()

            iter_time = time.perf_counter() - iter_start

            if run_logger:
                run_logger.add_scalar("loss/total", float(total_loss), global_step)
                run_logger.add_scalar("loss/latent", log_metrics['loss_latent'], global_step)
                run_logger.add_scalar("loss/recon", log_metrics['loss_recon'], global_step)
                run_logger.add_scalar("loss/anchor_m", log_metrics['loss_anchor_m'], global_step)
                run_logger.add_scalar("loss/anchor_p", log_metrics['loss_anchor_p'], global_step)
                run_logger.add_scalar("metric/acc_presence", log_metrics['acc_presence'], global_step)
                run_logger.tick()

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}", 
                lat=f"{log_metrics['loss_latent']:.3f}", 
                anc=f"{(log_metrics['loss_anchor_m']+log_metrics['loss_anchor_p']):.3f}"
            )

            if (global_step+1) % save_interval == 0:
                save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch, global_step, cfg, None)

            if recon_logger:
                if hasattr(loader.dataset, "movies"):
                    sample_tconsts = loader.dataset.movies[: z_movie.size(0)]
                else:
                    sample_tconsts = [""] * z_movie.size(0)

                recon_logger.step(global_step, z_movie.detach().cpu(), mask.detach().cpu(), run_logger, sample_tconsts)

            global_step += 1
            if stop_flag["stop"]: break
        
        save_checkpoint(Path(cfg.model_dir), model, opt, sched, epoch+1, global_step, cfg, None)
        if stop_flag["stop"]: break

    torch.save(model.state_dict(), Path(cfg.model_dir) / "SequenceDecoder_final.pt")
    if run_logger: run_logger.close()

if __name__ == "__main__":
    main()