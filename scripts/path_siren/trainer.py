import time
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import ProjectConfig
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.autoencoder.print_model import print_model_summary
from scripts.path_siren.model import PathSiren
from scripts.path_siren.dataset import TitlePathIterable, collate_batch
from scripts.path_siren.recon_logger import PathSirenReconstructionLogger
from scripts.autoencoder.training_callbacks.training_callbacks import TensorBoardPerBatchLogger


class PathSirenTrainer:
    def __init__(self, config: ProjectConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mov_ae, self.per_ae = _load_frozen_autoencoders(self.cfg)
        self.mov_ae.encoder.to(self.device).eval()
        self.mov_ae.decoder.to(self.device).eval()
        self.per_ae.encoder.to(self.device).eval()
        self.per_ae.decoder.to(self.device).eval()

        self.model = PathSiren(
            latent_dim=self.cfg.latent_dim,
            hidden_mult=float(self.cfg.path_siren_hidden_mult),
            layers=int(self.cfg.path_siren_layers),
            w0_first=float(self.cfg.path_siren_omega0_first),
            w0_hidden=float(self.cfg.path_siren_omega0_hidden),
            time_fourier=int(self.cfg.path_siren_time_fourier),
        ).to(self.device)

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.path_siren_lr),
            weight_decay=float(self.cfg.path_siren_weight_decay),
        )

        self.dataset = TitlePathIterable(
            db_path=self.cfg.db_path,
            principals_table=self.cfg.principals_table,
            movie_ae=self.mov_ae,
            people_ae=self.per_ae,
            num_people=int(self.cfg.path_siren_people_count),
            shuffle=True,
            cache_capacity_movies=int(self.cfg.path_siren_cache_capacity),
            cache_capacity_people=int(self.cfg.path_siren_cache_capacity * 2),
            movie_limit=self.cfg.path_siren_movie_limit,
            seed=int(self.cfg.path_siren_seed),
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=int(self.cfg.batch_size),
            collate_fn=collate_batch,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        Mx, My, Zt, Z_lat_tgts, Yp_tgts, t_grid, time_mask = next(iter(self.loader))
        Zt = Zt.to(self.device)
        t_grid = t_grid.to(self.device)
        print_model_summary(self.model, [Zt, t_grid])

        self.writer = TensorBoardPerBatchLogger(
            log_dir=str(self.cfg.tensorboard_dir),
            run_prefix="path_siren",
        )

        self.recon = PathSirenReconstructionLogger(
            movie_ae=self.mov_ae,
            people_ae=self.per_ae,
            predictor=self.model,
            interval_steps=int(self.cfg.path_siren_callback_interval),
            num_samples=int(self.cfg.path_siren_recon_num_samples),
            table_width=int(self.cfg.path_siren_table_width),
            writer=self.writer.writer,
            n_slots_show=int(self.cfg.path_siren_people_count),
        )

    def _people_loss_all_steps(
        self,
        z_seq: torch.Tensor,
        tgts_per_field: List[torch.Tensor],
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, L, d = z_seq.shape
        if L <= 1:
            return z_seq.new_zeros(())
        z_people = z_seq[:, 1:, :]
        T = z_people.size(1)
        mask_t = time_mask[:, 1:].float().clamp_min(0.0).clamp_max(1.0)

        flat = z_people.reshape(b * T, d)
        preds_per_field: List[torch.Tensor] = []
        for dec in self.per_ae.decoder.decs:
            y = dec(flat)
            y = y.view(b, T, *y.shape[1:])
            preds_per_field.append(y)

        total = z_seq.new_zeros((b,))
        for pred, tgt, field in zip(preds_per_field, tgts_per_field, self.per_ae.fields):
            tgt_t = tgt[:, 1:, ...]
            if pred.dim() == 4 and hasattr(field, "tokenizer"):
                bs, tlen, slen, vocab = pred.shape
                pad_id = int(getattr(field, "pad_token_id", 0)) if hasattr(field, "pad_token_id") else None
                p_flat = pred.reshape(bs * tlen * slen, vocab)
                tgt_ids = tgt_t.reshape(bs * tlen * slen)
                loss_flat = torch.nn.functional.cross_entropy(
                    p_flat,
                    tgt_ids,
                    ignore_index=pad_id if pad_id is not None else -1000,
                    reduction="none",
                )
                loss_tok = loss_flat.view(bs, tlen, slen)
                if pad_id is not None:
                    tok_mask = (tgt_t != pad_id).float()
                else:
                    tok_mask = torch.ones_like(loss_tok)
                tok_denom = tok_mask.sum(dim=2).clamp_min(1.0)
                loss_time = (loss_tok * tok_mask).sum(dim=2) / tok_denom
                time_denom = mask_t.sum(dim=1).clamp_min(1.0)
                total = total + ((loss_time * mask_t).sum(dim=1) / time_denom)
            elif pred.dim() == 4 and hasattr(field, "base"):
                bs, tlen, pos, vocab = pred.shape
                p_flat = pred.reshape(bs * tlen * pos, vocab)
                tgt_flat = tgt_t.reshape(bs * tlen * pos).long()
                loss_flat = torch.nn.functional.cross_entropy(p_flat, tgt_flat, reduction="none")
                loss_pos = loss_flat.view(bs, tlen, pos).mean(dim=2)
                time_denom = mask_t.sum(dim=1).clamp_min(1.0)
                total = total + ((loss_pos * mask_t).sum(dim=1) / time_denom)
            else:
                diff = pred - tgt_t
                loss_feat = diff.pow(2).reshape(b, T, -1).mean(dim=2)
                time_denom = mask_t.sum(dim=1).clamp_min(1.0)
                total = total + ((loss_feat * mask_t).sum(dim=1) / time_denom)

        return total.mean()

    def _latent_path_loss(
        self,
        z_seq: torch.Tensor,
        z_targets: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, L, d = z_seq.shape
        if L <= 1:
            return z_seq.new_zeros(())
        z_pred = z_seq[:, 1:, :]
        z_tgt = z_targets[:, 1:, :]
        mask_t = time_mask[:, 1:].float()
        mse = (z_pred - z_tgt).pow(2).mean(dim=2)
        denom = mask_t.sum(dim=1).clamp_min(1.0)
        per_sample = (mse * mask_t).sum(dim=1) / denom
        return per_sample.mean()

    def _movie_loss_at_t0(
        self,
        preds_per_field: List[torch.Tensor],
        movie_targets: List[torch.Tensor],
    ) -> torch.Tensor:
        total = preds_per_field[0].new_zeros((preds_per_field[0].size(0),))
        for pred, tgt, field in zip(preds_per_field, movie_targets, self.mov_ae.fields):
            b = pred.size(0)
            if pred.dim() == 3 and hasattr(field, "tokenizer"):
                l, v = pred.shape[-2], pred.shape[-1]
                pad_id = int(getattr(field, "pad_token_id", 0))
                loss_flat = torch.nn.functional.cross_entropy(
                    pred.reshape(b * l, v),
                    tgt.reshape(b * l),
                    ignore_index=pad_id,
                    reduction="none",
                )
                total = total + loss_flat.view(b, l).mean(dim=1)
            elif pred.dim() == 3 and hasattr(field, "base"):
                p, v = pred.shape[-2], pred.shape[-1]
                loss_flat = torch.nn.functional.cross_entropy(
                    pred.reshape(b * p, v),
                    tgt.reshape(b * p).long(),
                    reduction="none",
                )
                total = total + loss_flat.view(b, p).mean(dim=1)
            else:
                mse = torch.nn.functional.mse_loss(pred, tgt, reduction="none").reshape(b, -1).mean(dim=1)
                total = total + mse
        return total.mean()

    def _straightness_loss(
        self,
        z_seq: torch.Tensor,
        t_grid: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, L, d = z_seq.shape
        if L <= 1:
            return z_seq.new_zeros(())
        idx_last = time_mask.sum(dim=1).long().clamp(min=1) - 1
        arange_b = torch.arange(b, device=z_seq.device)
        z0 = z_seq[:, 0, :]
        zend = z_seq[arange_b, idx_last, :]
        t_last = t_grid[arange_b, idx_last].clamp_min(1e-6).unsqueeze(1)
        u = (t_grid / t_last).clamp_max(1.0)
        line = z0.unsqueeze(1) + u.unsqueeze(2) * (zend - z0).unsqueeze(1)
        diffsq = (z_seq - line).pow(2).mean(dim=2)
        m0 = torch.zeros_like(time_mask[:, :1], dtype=torch.float32)
        m1 = time_mask[:, 1:].float()
        mask = torch.cat([m0, m1], dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        per_sample = (diffsq * mask).sum(dim=1) / denom
        return per_sample.mean()

    def _curvature_loss(
        self,
        z_seq: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, L, d = z_seq.shape
        if L < 3:
            return z_seq.new_zeros(())
        z_prev = z_seq[:, 0:L-2, :]
        z_cur = z_seq[:, 1:L-1, :]
        z_next = z_seq[:, 2:L, :]
        d2 = z_prev - 2.0 * z_cur + z_next
        curv = d2.pow(2).mean(dim=2)
        mask_mid = (time_mask[:, 0:L-2] * time_mask[:, 1:L-1] * time_mask[:, 2:L]).float()
        denom = mask_mid.sum(dim=1).clamp_min(1.0)
        per_sample = (curv * mask_mid).sum(dim=1) / denom
        return per_sample.mean()

    def train(self):
        epochs = int(self.cfg.path_siren_epochs)
        save_every = int(self.cfg.path_siren_save_interval)
        w_title_latent = float(self.cfg.path_siren_loss_w_title_latent)
        w_title_recon = float(self.cfg.path_siren_loss_w_title_recon)
        w_latent_path = float(self.cfg.path_siren_loss_w_latent_path)
        w_straight = float(self.cfg.path_siren_loss_w_straight)
        w_curv = float(self.cfg.path_siren_loss_w_curvature)
        flush_every = int(self.cfg.flush_interval)

        step = 0
        ckpt_dir = Path(self.cfg.model_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"device: {self.device}")

        for epoch in range(epochs):
            pbar = tqdm(self.loader, unit="batch", dynamic_ncols=True)
            pbar.set_description(f"path_siren epoch {epoch+1}/{epochs}")

            for Mx, My, Zt, Z_lat_tgts, Yp_tgts, t_grid, time_mask in pbar:
                t0 = time.perf_counter()

                Mx = [x.to(self.device) for x in Mx]
                My = [y.to(self.device) for y in My]
                Zt = Zt.to(self.device)
                Z_lat_tgts = Z_lat_tgts.to(self.device)
                Yp_tgts = [y.to(self.device) for y in Yp_tgts]
                t_grid = t_grid.to(self.device)
                time_mask = time_mask.to(self.device)

                z_seq = self.model(Zt, t_grid)

                z0 = z_seq[:, 0, :]
                preds_movie: List[torch.Tensor] = []
                for dec in self.mov_ae.decoder.decs:
                    preds_movie.append(dec(z0))

                loss_movie_t0 = self._movie_loss_at_t0(preds_movie, My)
                loss_latent_t0 = torch.nn.functional.mse_loss(z0, Zt)
                loss_people = self._people_loss_all_steps(z_seq, Yp_tgts, time_mask)
                loss_latent_path = self._latent_path_loss(z_seq, Z_lat_tgts, time_mask)
                loss_straight = self._straightness_loss(z_seq, t_grid, time_mask)
                loss_curv = self._curvature_loss(z_seq, time_mask)

                loss = (
                    loss_people
                    + w_latent_path * loss_latent_path
                    + w_title_latent * loss_latent_t0
                    + w_title_recon * loss_movie_t0
                    + w_straight * loss_straight
                    + w_curv * loss_curv
                )

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

                iter_time = time.perf_counter() - t0
                lr = float(self.optim.param_groups[0].get("lr", 0.0))

                self.writer.log_scalars(
                    step,
                    {
                        "loss/total": float(loss.detach().cpu().item()),
                        "loss/people": float(loss_people.detach().cpu().item()),
                        "loss/latent_path": float(loss_latent_path.detach().cpu().item()),
                        "loss/title_latent_t0": float(loss_latent_t0.detach().cpu().item()),
                        "loss/title_recon_t0": float(loss_movie_t0.detach().cpu().item()),
                        "loss/straight": float(loss_straight.detach().cpu().item()),
                        "loss/curvature": float(loss_curv.detach().cpu().item()),
                        "time/iter_s": float(iter_time),
                        "opt/lr": lr,
                    },
                )

                pbar.set_postfix(
                    loss=f"{float(loss.detach().cpu().item()):.4f}",
                    ppl=f"{float(loss_people.detach().cpu().item()):.4f}",
                    zpath=f"{float(loss_latent_path.detach().cpu().item()):.4f}",
                    z0=f"{float(loss_latent_t0.detach().cpu().item()):.4f}",
                    t0=f"{float(loss_movie_t0.detach().cpu().item()):.4f}",
                    strt=f"{float(loss_straight.detach().cpu().item()):.4f}",
                    curv=f"{float(loss_curv.detach().cpu().item()):.4f}",
                    it_s=f"{iter_time:.3f}",
                )

                self.recon.on_batch_end(
                    global_step=step,
                    Mx=Mx,
                    My=My,
                    Z=Zt,
                    t_grid=t_grid,
                    z_seq=z_seq.detach(),
                    mask=time_mask,
                    Yp_tgts=Yp_tgts,
                )

                step += 1
                if (step % save_every) == 0:
                    out = ckpt_dir / "PathSiren.pt"
                    torch.save(self.model.state_dict(), out)

                if (step % flush_every) == 0:
                    self.writer.flush()

        out = ckpt_dir / "PathSiren_final.pt"
        torch.save(self.model.state_dict(), out)
        self.writer.flush()
        self.writer.close()
