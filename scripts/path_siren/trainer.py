import time
import logging
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
        log = logging.getLogger("path_siren")

        self.mov_ae, self.per_ae = _load_frozen_autoencoders(self.cfg)
        self.mov_ae.encoder.to(self.device).eval()
        self.mov_ae.decoder.to(self.device).eval()
        self.per_ae.encoder.to(self.device).eval()
        self.per_ae.decoder.to(self.device).eval()

        cache_file = Path(self.cfg.data_dir) / "path_siren_cache.pt"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"[path-siren] precomputed cache not found at {cache_file}.\n"
                f"Run `python -m scripts.precompute_path_siren_cache` first."
            )
        log.info(f"[path-siren] using precomputed cache at {cache_file}")

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
            path_siren_cache_path=str(cache_file),
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=int(self.cfg.batch_size),
            collate_fn=collate_batch,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        (
            Mx0,
            My0,
            Zt0,
            Z_lat0,
            Yp0,
            t_grid0,
            time_mask0,
        ) = next(iter(self.loader))
        Zt0 = Zt0.to(self.device)
        t_grid0 = t_grid0.to(self.device)
        print_model_summary(self.model, [Zt0, t_grid0])

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

        self._log_model_and_vocab_info()

    def _count_params(self, module: torch.nn.Module) -> tuple[int, int]:
        total = 0
        trainable = 0
        for p in module.parameters():
            n = int(p.numel())
            total += n
            if p.requires_grad:
                trainable += n
        return total, trainable

    def _log_model_and_vocab_info(self) -> None:
        log = logging.getLogger("path_siren")

        ps_total, ps_train = self._count_params(self.model)
        mov_enc_total, mov_enc_train = self._count_params(self.mov_ae.encoder)
        mov_dec_total, mov_dec_train = self._count_params(self.mov_ae.decoder)
        per_enc_total, per_enc_train = self._count_params(self.per_ae.encoder)
        per_dec_total, per_dec_train = self._count_params(self.per_ae.decoder)

        msg_lines = []

        msg_lines.append("[path-siren] config/latent dims:")
        msg_lines.append(f"  cfg.latent_dim={self.cfg.latent_dim}")
        msg_lines.append(f"  mov_ae.latent_dim={getattr(self.mov_ae, 'latent_dim', None)}")
        msg_lines.append(f"  per_ae.latent_dim={getattr(self.per_ae, 'latent_dim', None)}")

        msg_lines.append("[path-siren] parameter counts:")
        msg_lines.append(f"  PathSiren: total={ps_total} trainable={ps_train}")
        msg_lines.append(f"  MovieAE.encoder: total={mov_enc_total} trainable={mov_enc_train}")
        msg_lines.append(f"  MovieAE.decoder: total={mov_dec_total} trainable={mov_dec_train}")
        msg_lines.append(f"  PeopleAE.encoder: total={per_enc_total} trainable={per_enc_train}")
        msg_lines.append(f"  PeopleAE.decoder: total={per_dec_total} trainable={per_dec_train}")

        msg_lines.append("[path-siren] text field vocab/alphabet sizes:")
        for ae_name, ae in (("movie", self.mov_ae), ("people", self.per_ae)):
            for f in getattr(ae, "fields", []):
                tok = getattr(f, "tokenizer", None)
                if tok is not None and getattr(tok, "trained", False):
                    if hasattr(tok, "get_vocab_size"):
                        vocab_size = tok.get_vocab_size()
                    else:
                        vocab_size = len(getattr(tok, "char_to_index", {}))
                    msg_lines.append(
                        f"  {ae_name}.{f.name}: "
                        f"vocab_size={vocab_size} "
                        f"max_length={getattr(f, 'max_length', None)} "
                        f"dynamic_max_len={getattr(f, 'dynamic_max_len', None)}"
                    )

        same_latent = (
            getattr(self.mov_ae, "latent_dim", None) == self.cfg.latent_dim
            == getattr(self.per_ae, "latent_dim", None)
        )
        msg_lines.append(
            f"[path-siren] latent_dim_match={bool(same_latent)} "
            f"(mov_ae={getattr(self.mov_ae, 'latent_dim', None)}, "
            f"per_ae={getattr(self.per_ae, 'latent_dim', None)}, "
            f"path_siren={self.cfg.latent_dim})"
        )

        text = "\n".join(msg_lines)
        print(text)
        log.info("\n" + text)

        if self.writer and self.writer.writer:
            self.writer.writer.add_text("debug/model_and_vocab", f"```\n{text}\n```", 0)

    def _build_z_seq_from_deltas(
        self,
        Zt: torch.Tensor,
        z_delta: torch.Tensor,
    ) -> torch.Tensor:
        b, L, d = z_delta.shape
        if L == 0:
            return Zt.new_zeros((b, 0, d))
        outs = []
        prev = Zt
        for t in range(L):
            cur = prev + z_delta[:, t, :]
            outs.append(cur.unsqueeze(1))
            prev = cur
        return torch.cat(outs, dim=1)

    def _people_loss_all_steps(
        self,
        z_seq: torch.Tensor,
        tgts_per_field: List[torch.Tensor],
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, L, d = z_seq.shape
        if L <= 0:
            return z_seq.new_zeros(())

        mask_t = time_mask[:, :L].float().clamp(0.0, 1.0)
        if mask_t.sum() <= 0:
            return z_seq.new_zeros(())

        flat = z_seq.reshape(b * L, d)
        preds_per_field = []
        for dec in self.per_ae.decoder.decs:
            y = dec(flat)
            y = y.reshape(b, L, *y.shape[1:])
            preds_per_field.append(y)

        total = z_seq.new_zeros((b,), device=z_seq.device)

        for pred, tgt, field in zip(preds_per_field, tgts_per_field, self.per_ae.fields):
            tgt_t = tgt[:, :L, ...]

            if pred.dim() == 4 and hasattr(field, "tokenizer"):
                bs, tlen, slen, vocab = pred.shape
                pad_id = int(getattr(field, "pad_token_id", 0))

                p_flat = pred.reshape(bs * tlen * slen, vocab)
                tgt_ids = tgt_t.reshape(bs * tlen * slen)

                loss_flat = torch.nn.functional.cross_entropy(
                    p_flat,
                    tgt_ids,
                    ignore_index=pad_id,
                    reduction="none",
                )
                loss_tok = loss_flat.reshape(bs, tlen, slen)
                tok_mask = (tgt_t != pad_id).float()
                tok_denom = tok_mask.sum(dim=2).clamp_min(1.0)
                loss_time = (loss_tok * tok_mask).sum(dim=2) / tok_denom

                time_denom = mask_t.sum(dim=1).clamp_min(1.0)
                total = total + ((loss_time * mask_t).sum(dim=1) / time_denom)

            elif pred.dim() == 4 and hasattr(field, "base"):
                bs, tlen, pos, vocab = pred.shape
                p_flat = pred.reshape(bs * tlen * pos, vocab)
                tgt_flat = tgt_t.reshape(bs * tlen * pos).long()

                loss_flat = torch.nn.functional.cross_entropy(
                    p_flat,
                    tgt_flat,
                    reduction="none",
                )
                loss_pos = loss_flat.reshape(bs, tlen, pos).mean(dim=2)

                time_denom = mask_t.sum(dim=1).clamp_min(1.0)
                total = total + ((loss_pos * mask_t).sum(dim=1) / time_denom)

            else:
                diff = pred - tgt_t
                loss_feat = diff.pow(2).reshape(b, L, -1).mean(dim=2)
                time_denom = mask_t.sum(dim=1).clamp_min(1.0)
                total = total + ((loss_feat * mask_t).sum(dim=1) / time_denom)

        return total.mean()

    def _delta_path_loss(
        self,
        z_delta: torch.Tensor,
        Zt: torch.Tensor,
        Z_lat_tgts: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, L, d = z_delta.shape
        if L <= 0:
            return z_delta.new_zeros(())

        mask = time_mask[:, :L].float().clamp(0.0, 1.0)
        if mask.sum() <= 0:
            return z_delta.new_zeros(())

        if Z_lat_tgts.size(1) < L:
            raise RuntimeError(
                f"Z_lat_tgts too short: {Z_lat_tgts.size()} for L={L}"
            )

        base = Z_lat_tgts.new_zeros((b, L, d))
        base[:, 0, :] = Zt
        if L > 1:
            base[:, 1:, :] = Z_lat_tgts[:, : L - 1, :]

        delta_tgt = Z_lat_tgts[:, :L, :] - base

        mse = (z_delta - delta_tgt).pow(2).mean(dim=2)
        denom = mask.sum(dim=1).clamp_min(1.0)
        per_sample = (mse * mask).sum(dim=1) / denom
        return per_sample.mean()

    def train(self):
        epochs = int(self.cfg.path_siren_epochs)
        save_every = int(self.cfg.path_siren_save_interval)
        w_people = float(self.cfg.path_siren_loss_w_people)
        w_latent_path = float(self.cfg.path_siren_loss_w_latent_path)
        flush_every = int(self.cfg.flush_interval)

        step = 0
        ckpt_dir = Path(self.cfg.model_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"device: {self.device}")

        for epoch in range(epochs):
            pbar = tqdm(self.loader, unit="batch", dynamic_ncols=True)
            pbar.set_description(f"path_siren epoch {epoch + 1}/{epochs}")

            for (
                Mx,
                My,
                Zt,
                Z_lat_tgts,
                Yp_tgts,
                t_grid,
                time_mask,
            ) in pbar:
                t0 = time.perf_counter()

                Mx = [x.to(self.device) for x in Mx]
                My = [y.to(self.device) for y in My]
                Zt = Zt.to(self.device)
                Z_lat_tgts = Z_lat_tgts.to(self.device)
                Yp_tgts = [y.to(self.device) for y in Yp_tgts]
                t_grid = t_grid.to(self.device)
                time_mask = time_mask.to(self.device)

                z_delta = self.model(Zt, t_grid)
                z_seq = self._build_z_seq_from_deltas(Zt, z_delta)

                loss_people = self._people_loss_all_steps(
                    z_seq=z_seq,
                    tgts_per_field=Yp_tgts,
                    time_mask=time_mask,
                )

                loss_delta_path = self._delta_path_loss(
                    z_delta=z_delta,
                    Zt=Zt,
                    Z_lat_tgts=Z_lat_tgts,
                    time_mask=time_mask,
                )

                loss = (
                    w_people * loss_people
                    + w_latent_path * loss_delta_path
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
                        "loss/path_delta": float(loss_delta_path.detach().cpu().item()),
                        "time/iter_s": float(iter_time),
                        "opt/lr": lr,
                    },
                )

                pbar.set_postfix(
                    loss=f"{float(loss.detach().cpu().item()):.4f}",
                    ppl=f"{float(loss_people.detach().cpu().item()):.4f}",
                    path_d=f"{float(loss_delta_path.detach().cpu().item()):.4f}",
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
                    Z_lat_tgts=Z_lat_tgts,
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
