import os
import math
import json
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ProjectConfig, ensure_dirs
from scripts.image_encoding.autoencoder import ConvAutoencoder
from scripts.image_siren.dataset import ImagePairDataset
from scripts.image_siren.model import ImageCondSiren
from scripts.image_siren.recon_logger import ImageSirenReconstructionSaver
from scripts.image_siren.video_export import make_recon_video


class SinLRScheduler:
    def __init__(self, optimizer, base_lr: float, min_lr: float, total_steps: int):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.total_steps = max(1, int(total_steps))
        self.i = 0

    def step(self) -> float:
        t = min(self.i, self.total_steps)
        lr = self.min_lr + (self.base_lr - self.min_lr) * math.sin(math.pi * t / self.total_steps)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.i += 1
        return lr


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_writer(cfg: ProjectConfig, run_dir: str):
    try:
        writer = SummaryWriter(log_dir=run_dir)
    except Exception:
        return None

    try:
        if hasattr(cfg, "__dataclass_fields__"):
            cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()}
        else:
            cfg_dict = dict(vars(cfg))
        text = json.dumps(cfg_dict, indent=2, sort_keys=True)
        writer.add_text("config/json", f"<pre>{text}</pre>", 0)
    except Exception:
        pass

    return writer


def build_coord_grid(image_size: int, device: torch.device) -> torch.Tensor:
    s = int(image_size)
    xs = torch.linspace(-1.0 + 1.0 / s, 1.0 - 1.0 / s, steps=s, device=device)
    ys = torch.linspace(-1.0 + 1.0 / s, 1.0 - 1.0 / s, steps=s, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).view(-1, 2)
    return coords


class ImageSirenTrainer:
    def __init__(self, cfg: ProjectConfig, ae_checkpoint: str):
        self.cfg = cfg
        ensure_dirs(cfg)

        self.device = get_device()

        ae_size: Tuple[int, int] = (
            cfg.image_ae_image_size,
            cfg.image_ae_image_size,
        )
        siren_size: Tuple[int, int] = (
            cfg.image_siren_image_size,
            cfg.image_siren_image_size,
        )

        self.samples_per_image = int(getattr(cfg, "image_siren_samples_per_image", 0) or 0)
        self.density_alpha = float(getattr(cfg, "image_siren_density_alpha", 0.9))
        self.uniform_frac = float(getattr(cfg, "image_siren_uniform_frac", 0.1))
        self.image_repeats = int(getattr(cfg, "image_siren_image_repeats_per_epoch", 1) or 1)
        if self.image_repeats < 1:
            self.image_repeats = 1

        train_root = cfg.image_ae_data_dir
        eval_root = getattr(cfg, "image_siren_eval_data_dir", None)

        self.dataset = ImagePairDataset(
            root=train_root,
            ae_size=ae_size,
            siren_size=siren_size,
            density_alpha=self.density_alpha,
        )

        self.eval_dataset = None
        if isinstance(eval_root, str) and os.path.isdir(eval_root):
            try:
                self.eval_dataset = ImagePairDataset(
                    root=eval_root,
                    ae_size=ae_size,
                    siren_size=siren_size,
                    density_alpha=self.density_alpha,
                )
            except Exception:
                self.eval_dataset = None

        if len(self.dataset) == 0:
            raise RuntimeError("image_ae_data_dir is empty or has no valid images")

        batch_size = max(1, int(cfg.image_siren_batch_size))
        drop_last = len(self.dataset) >= batch_size

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(getattr(cfg, "num_workers", 0) or 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )

        self.eval_loader = None
        if self.eval_dataset is not None and len(self.eval_dataset) > 0:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(getattr(cfg, "num_workers", 0) or 0),
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )

        self.encoder = ConvAutoencoder(
            in_channels=cfg.image_ae_in_channels,
            base_channels=cfg.image_ae_base_channels,
            latent_dim=cfg.image_ae_latent_dim,
            image_size=cfg.image_ae_image_size,
        )

        ckpt = torch.load(ae_checkpoint, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        else:
            state = ckpt
        self.encoder.load_state_dict(state, strict=False)
        self.encoder.to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.siren = ImageCondSiren(
            latent_dim=cfg.image_ae_latent_dim,
            out_channels=cfg.image_ae_in_channels,
            hidden_dim=cfg.image_siren_hidden_dim,
            hidden_layers=cfg.image_siren_hidden_layers,
            w0_first=cfg.image_siren_w0_first,
            w0_hidden=cfg.image_siren_w0_hidden,
        ).to(self.device)

        self.coord_grid = build_coord_grid(cfg.image_siren_image_size, self.device)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(cfg.image_siren_runs_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

        self.writer = build_writer(cfg, self.run_dir)

        sample_bs = min(int(cfg.image_siren_max_recon_samples), len(self.dataset))
        sample_loader = DataLoader(
            self.dataset,
            batch_size=sample_bs,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        sample_ae_t, sample_tgt_t, _ = next(iter(sample_loader))

        samples = {"train": (sample_ae_t, sample_tgt_t)}

        if self.eval_dataset is not None and len(self.eval_dataset) > 0:
            eval_bs = min(
                int(getattr(cfg, "image_siren_eval_max_samples", cfg.image_siren_max_recon_samples)),
                len(self.eval_dataset),
            )
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=eval_bs,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
            sample_ae_e, sample_tgt_e, _ = next(iter(eval_loader))
            samples["eval"] = (sample_ae_e, sample_tgt_e)

        self.recon_saver = ImageSirenReconstructionSaver(
            output_dir=self.run_dir,
            samples=samples,
            every_n_epochs=int(getattr(cfg, "image_siren_eval_recon_every", cfg.image_siren_recon_every)),
            max_samples=cfg.image_siren_max_recon_samples,
            image_size=cfg.image_siren_image_size,
        )

        self.lr_initial = float(cfg.image_siren_lr)
        self.lr_min = self.lr_initial / 10.0

        self.optimizer = torch.optim.Adam(
            self.siren.parameters(),
            lr=self.lr_initial,
        )

        self.w_l1 = float(cfg.image_siren_loss_w_l1)
        self.w_mse = float(cfg.image_siren_loss_w_mse)
        self.use_sigmoid = bool(cfg.image_siren_from_latent_sigmoid)

    def _sample_coords_for_single(
        self,
        dens: Optional[torch.Tensor],
        h: int,
        w: int,
        num_samples: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if num_samples <= 0:
            xs = torch.linspace(
                -1.0 + 1.0 / w,
                1.0 - 1.0 / w,
                steps=w,
                device=self.device,
            )
            ys = torch.linspace(
                -1.0 + 1.0 / h,
                1.0 - 1.0 / h,
                steps=h,
                device=self.device,
            )
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([xx, yy], dim=-1).view(-1, 2)
            return coords, None

        n_pix = h * w
        u = 1.0 / float(n_pix)

        p = None
        if dens is not None and dens.numel() == n_pix:
            p = dens.to(self.device).clamp_min(0.0)
            s = float(p.sum())
            if s > 0.0:
                p = p / s
            else:
                p = None

        uni_frac = min(max(self.uniform_frac, 0.0), 1.0)
        n_uni = int(num_samples * uni_frac)
        n_bias = num_samples - n_uni

        if p is None:
            idx_all = torch.randint(0, n_pix, (num_samples,), device=self.device)
            q = torch.full((num_samples,), u, device=self.device, dtype=torch.float32)
        else:
            idx_parts = []

            if n_bias > 0:
                idx_bias = torch.multinomial(p, n_bias, replacement=True)
                idx_parts.append(idx_bias)

            if n_uni > 0:
                idx_uni = torch.randint(0, n_pix, (n_uni,), device=self.device)
                idx_parts.append(idx_uni)

            if not idx_parts:
                idx_all = torch.randint(0, n_pix, (num_samples,), device=self.device)
                q = torch.full((num_samples,), u, device=self.device, dtype=torch.float32)
            else:
                idx_all = torch.cat(idx_parts, dim=0)
                if idx_all.size(0) > num_samples:
                    idx_all = idx_all[:num_samples]

                w_bias = float(max(n_bias, 0)) / float(num_samples)
                w_uni = float(max(n_uni, 0)) / float(num_samples)

                p_idx = p[idx_all]
                q = w_bias * p_idx + w_uni * u
                q = q.clamp_min(1e-8)

        y = idx_all // w
        x = idx_all % w

        jx = torch.empty_like(x, dtype=torch.float32).uniform_(-0.5, 0.5)
        jy = torch.empty_like(y, dtype=torch.float32).uniform_(-0.5, 0.5)

        xf = (x.to(torch.float32) + 0.5 + jx) / float(w)
        yf = (y.to(torch.float32) + 0.5 + jy) / float(h)

        xf = xf.clamp(0.0, 1.0)
        yf = yf.clamp(0.0, 1.0)

        cx = 2.0 * xf - 1.0
        cy = 2.0 * yf - 1.0

        coords = torch.stack([cx, cy], dim=-1)

        if q is None:
            return coords, None

        w_imp = 1.0 / q
        w_imp = w_imp / w_imp.mean()
        return coords, w_imp

    def _sample_coords_and_targets(
        self,
        imgs_siren: torch.Tensor,
        densities,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        b, c, h, w = imgs_siren.shape
        imgs = imgs_siren.to(self.device, non_blocking=True)

        dens_batch = None
        if isinstance(densities, torch.Tensor) and densities.dim() == 2:
            dens_batch = densities

        coords_all = []
        tgt_all = []
        w_all = []

        for i in range(b):
            dens_i = None
            if dens_batch is not None and dens_batch.size(1) == h * w:
                dens_i = dens_batch[i]

            coords_i, w_i = self._sample_coords_for_single(
                dens=dens_i,
                h=h,
                w=w,
                num_samples=num_samples,
            )
            coords_all.append(coords_i)

            grid = coords_i.view(1, num_samples, 1, 2)

            sample = F.grid_sample(
                imgs[i:i + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            sample = sample.view(c, num_samples).transpose(0, 1)
            tgt_all.append(sample)

            if w_i is not None:
                w_all.append(w_i)
            else:
                w_all = []

        coords = torch.stack(coords_all, dim=0)
        targets = torch.stack(tgt_all, dim=0)

        if w_all and len(w_all) == b:
            weights = torch.stack(w_all, dim=0)
        else:
            weights = None

        return coords, targets, weights

    def _step_loss(
        self,
        imgs_ae: torch.Tensor,
        imgs_siren: torch.Tensor,
        densities,
    ) -> torch.Tensor:
        b, c, h_s, w_s = imgs_siren.shape

        with torch.no_grad():
            z = self.encoder.encode(imgs_ae)

        if self.samples_per_image > 0:
            coords, tgt, w = self._sample_coords_and_targets(
                imgs_siren=imgs_siren,
                densities=densities,
                num_samples=self.samples_per_image,
            )

            logits = self.siren(coords, z)

            if self.use_sigmoid:
                pred = torch.sigmoid(logits)
            else:
                pred = logits

            loss = 0.0

            if self.w_l1 > 0.0:
                if w is not None:
                    diff = (pred - tgt).abs()
                    weighted = diff * w.unsqueeze(-1)
                    denom = float(w.sum()) * float(c)
                    if denom <= 0.0:
                        denom = float(b * self.samples_per_image * c)
                    loss_l1 = weighted.sum() / denom
                else:
                    loss_l1 = F.l1_loss(pred, tgt)
                loss = loss + self.w_l1 * loss_l1

            if self.w_mse > 0.0:
                if w is not None:
                    diff2 = (pred - tgt).pow(2)
                    weighted = diff2 * w.unsqueeze(-1)
                    denom = float(w.sum()) * float(c)
                    if denom <= 0.0:
                        denom = float(b * self.samples_per_image * c)
                    loss_mse = weighted.sum() / denom
                else:
                    loss_mse = F.mse_loss(pred, tgt)
                loss = loss + self.w_mse * loss_mse

            return loss

        coords = self.coord_grid.unsqueeze(0).repeat(b, 1, 1)
        coords = coords.to(self.device, non_blocking=True).contiguous()

        logits = self.siren(coords, z)

        if self.use_sigmoid:
            pred = torch.sigmoid(logits)
        else:
            pred = logits

        pred = pred.view(b, h_s, w_s, c).permute(0, 3, 1, 2).contiguous()
        tgt = imgs_siren.to(self.device, non_blocking=True)

        loss = 0.0

        if self.w_l1 > 0.0:
            loss = loss + self.w_l1 * F.l1_loss(pred, tgt)
        if self.w_mse > 0.0:
            loss = loss + self.w_mse * F.mse_loss(pred, tgt)

        return loss

    @torch.no_grad()
    def _eval_epoch_full(self) -> dict[str, float]:
        if self.eval_loader is None:
            return {}

        self.siren.eval()
        total_l1 = 0.0
        total_mse = 0.0
        total_pix = 0

        for imgs_ae, imgs_siren, _ in self.eval_loader:
            b, c, h, w = imgs_siren.shape
            imgs_ae = imgs_ae.to(self.device, non_blocking=True)
            imgs_siren = imgs_siren.to(self.device, non_blocking=True)

            z = self.encoder.encode(imgs_ae)

            coords = self.coord_grid.unsqueeze(0).repeat(b, 1, 1).to(self.device)
            logits = self.siren(coords, z)
            pred = torch.sigmoid(logits)
            pred = pred.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

            diff = (pred - imgs_siren).abs()
            diff2 = (pred - imgs_siren).pow(2)

            total_l1 += float(diff.sum().detach().cpu())
            total_mse += float(diff2.sum().detach().cpu())
            total_pix += int(b * c * h * w)

        if total_pix <= 0:
            return {}

        avg_l1 = total_l1 / total_pix
        avg_mse = total_mse / total_pix
        return {"eval_l1": avg_l1, "eval_mse": avg_mse}

    def train(self):
        epochs = int(self.cfg.image_siren_epochs)
        global_step = 0

        print(f"device: {self.device}")
        print(f"dataset root: {self.cfg.image_ae_data_dir}")
        print(f"dataset size: {len(self.dataset)} images")
        if self.eval_dataset is not None:
            print(f"eval dataset root: {getattr(self.cfg, 'image_siren_eval_data_dir', None)}")
            print(f"eval dataset size: {len(self.eval_dataset)} images")
        print(f"batch size: {self.cfg.image_siren_batch_size}")
        print(f"siren target size: {self.cfg.image_siren_image_size}")
        print(f"samples per image: {self.samples_per_image}")
        print(f"density alpha (dataset blend): {self.density_alpha}")
        print(f"uniform frac (sampler blend): {self.uniform_frac}")
        print(f"image repeats per epoch: {self.image_repeats}")
        print(f"run dir: {self.run_dir}")
        print("siren model:")
        print(self.siren)

        steps_per_epoch = self.image_repeats * max(1, len(self.loader))
        total_steps = max(1, epochs * steps_per_epoch)
        scheduler = SinLRScheduler(
            optimizer=self.optimizer,
            base_lr=self.lr_initial,
            min_lr=self.lr_min,
            total_steps=total_steps,
        )

        for epoch in range(1, epochs + 1):
            self.siren.train()
            total_loss = 0.0
            total_batches = 0

            for rep in range(self.image_repeats):
                pbar = tqdm(
                    self.loader,
                    desc=f"image_siren epoch {epoch}/{epochs} rep {rep+1}/{self.image_repeats}",
                    unit="batch",
                    dynamic_ncols=True,
                )

                for imgs_ae, imgs_siren, densities in pbar:
                    lr_now = scheduler.step()

                    imgs_ae = imgs_ae.to(self.device, non_blocking=True)
                    imgs_siren = imgs_siren.to(self.device, non_blocking=True)

                    loss = self._step_loss(imgs_ae, imgs_siren, densities)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.siren.parameters(), 1.0)
                    self.optimizer.step()

                    loss_val = float(loss.detach().cpu().item())
                    total_loss += loss_val
                    total_batches += 1
                    global_step += 1

                    pbar.set_postfix(loss=f"{loss_val:.6f}", lr=f"{lr_now:.6e}")

                    if self.writer is not None:
                        self.writer.add_scalar("train/loss_step", loss_val, global_step)
                        self.writer.add_scalar("train/lr_step", lr_now, global_step)

            if total_batches == 0:
                raise RuntimeError("No batches produced. Check dataset and batch size.")

            avg_loss = total_loss / total_batches
            print(f"epoch {epoch} avg_loss {avg_loss:.6f}")

            if self.writer is not None:
                self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)

            self.recon_saver.maybe_save(
                epoch=epoch,
                encoder=self.encoder,
                siren=self.siren,
                device=self.device,
                coord_grid=self.coord_grid,
            )

            if self.eval_loader is not None:
                eval_stats = self._eval_epoch_full()
                if eval_stats and self.writer is not None:
                    for k, v in eval_stats.items():
                        self.writer.add_scalar(f"{k}", v, epoch)

            ckpt_path = os.path.join(self.run_dir, f"image_siren.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "siren_state": self.siren.state_dict(),
                    "config": self.cfg.__dict__,
                },
                ckpt_path,
            )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        video_path = make_recon_video(
            run_dir=self.run_dir,
            fps=4,
            crf=23,
            basename="siren_recon",
        )
        if video_path is not None:
            print(f"reconstruction video: {video_path}")
        else:
            print("no reconstruction video created (no recon images found or export failed)")
