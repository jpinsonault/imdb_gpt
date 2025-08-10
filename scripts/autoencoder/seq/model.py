import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..row_ae.imdb import TitlesAutoencoder, PeopleAutoencoder
from .dataset import MoviesPeopleSequenceDataset, collate_movies_people
from . import db as seq_db
from .sequence_reconstruction_logger import SequenceReconstructionLogger

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

class _SeqResidual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.c1 = nn.Conv1d(width, width, 1)
        self.c2 = nn.Conv1d(width, width, 1)

    def forward(self, x):
        r = x
        x = self.ln(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(self.c1(x))
        x = self.c2(x)
        x = F.gelu(x + r)
        return x

def _reshape_seq(t: torch.Tensor, B: int, T: int) -> torch.Tensor:
    if t.dim() == 2:
        return t.view(B, T, -1)
    if t.dim() == 3:
        C = t.size(1)
        return t.view(B, T, C, -1)
    if t.dim() == 4:
        return t.view(B, T, t.size(1), t.size(2), t.size(3))
    return t

def _merge_bt(x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    s = list(x.shape)
    s[0] = B * T
    s.pop(1)
    return x.contiguous().view(*s)

def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()

def _info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = _cosine_sim(a, b) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    la = F.cross_entropy(logits, labels)
    lb = F.cross_entropy(logits.t(), labels)
    return 0.5 * (la + lb)

class MoviesToPeopleSequenceDecoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.db_path = config["db_path"]
        self.latent_dim = int(config["latent_dim"])
        self.seq_len = int(config["people_sequence_length"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.movie_ae = TitlesAutoencoder(config)
        self.people_ae = PeopleAutoencoder(config)

        self.movie_ae.accumulate_stats()
        self.movie_ae.finalize_stats()
        self.movie_ae.build_autoencoder()
        self.movie_ae.load_model()

        self.people_ae.accumulate_stats()
        self.people_ae.finalize_stats()
        self.people_ae.build_autoencoder()
        self.people_ae.load_model()

        for p in self.movie_ae.encoder.parameters():
            p.requires_grad = False
        for p in self.people_ae.decoder.parameters():
            p.requires_grad = False
        for p in self.people_ae.encoder.parameters():
            p.requires_grad = False

        width = self.latent_dim * 2
        self.project_up = nn.Conv1d(self.latent_dim, width, 1)
        self.blocks = nn.ModuleList([_SeqResidual(width) for _ in range(4)])
        self.project_down = nn.Conv1d(width, self.latent_dim, 1)

        wanted = self.config.get("sequence_active_people_fields")
        if wanted:
            names = [f.name for f in self.people_ae.fields]
            self.active_idx = [names.index(n) for n in wanted if n in names]
        else:
            self.active_idx = list(range(len(self.people_ae.fields)))

        self.to(self.device)
        self.train()

    def forward(self, movie_inputs: List[torch.Tensor]):
        z = self.movie_ae.encoder([x for x in movie_inputs])
        z = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        x = z.transpose(1, 2)
        x = F.gelu(self.project_up(x))
        for b in self.blocks:
            x = b(x)
        x = F.gelu(self.project_down(x))
        z_seq = x.transpose(1, 2)
        B, T, D = z_seq.shape
        flat = z_seq.reshape(B * T, D)
        outs = []
        for i in self.active_idx:
            o = self.people_ae.decoder.decs[i](flat)
            outs.append(_reshape_seq(o, B, T))
        return outs, z_seq

    def _encode_people_targets(self, targets: List[torch.Tensor], B: int, T: int) -> torch.Tensor:
        full = []
        idx_map = {i: k for k, i in enumerate(self.active_idx)}
        for i, f in enumerate(self.people_ae.fields):
            if i in idx_map:
                k = idx_map[i]
                t = targets[k]
                x = _merge_bt(t, B, T)
                full.append(x.to(self.device, non_blocking=True))
            else:
                base = f.get_base_padding_value().to(self.device)
                if base.dim() == 0:
                    base = base.view(1, 1)
                if base.dim() == 1:
                    base = base.unsqueeze(0)
                rep = base.expand(B * T, *base.shape[1:])
                full.append(rep)
        with torch.no_grad():
            z_true = self.people_ae.encoder(full)
        return z_true.view(B, T, -1)

    def compute_loss(self, preds: List[torch.Tensor], targets: List[torch.Tensor], z_seq: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for idx, pred in zip(self.active_idx, preds):
            f = self.people_ae.fields[idx]
            tgt = targets[idx]
            B, T = pred.size(0), pred.size(1)
            p = _merge_bt(pred, B, T)
            y = _merge_bt(tgt, B, T)
            total = total + f.compute_loss(p, y) * float(f.weight)
        B, T, D = z_seq.shape
        z_true = self._encode_people_targets(targets, B, T)
        llw = float(self.config.get("latent_loss_weight", 0.0))
        clw = float(self.config.get("contrastive_loss_weight", 0.0))
        if llw > 0.0:
            total = total + llw * F.mse_loss(z_seq.view(B * T, D), z_true.view(B * T, D))
        if clw > 0.0:
            temp = float(self.config.get("nce_temp", 0.07))
            total = total + clw * _info_nce(z_seq.view(B * T, D), z_true.view(B * T, D), temperature=temp)
        return total

    def _row_gen(self):
        return seq_db.iter_movie_with_people(self.db_path, self.seq_len)

    def _row_generator(self):
        return self._row_gen()

    def make_loader(self) -> DataLoader:
        ds = MoviesPeopleSequenceDataset(
            row_gen_fn=self._row_gen,
            movie_fields=self.movie_ae.fields,
            people_fields=self.people_ae.fields,
            active_idx=self.active_idx,
        )
        bs = int(self.config.get("batch_size", 32))
        num_workers = int(self.config.get("num_workers", 0))
        prefetch = int(self.config.get("prefetch_factor", 2)) if num_workers > 0 else None
        pin = bool(torch.cuda.is_available())
        return DataLoader(
            ds,
            batch_size=bs,
            collate_fn=collate_movies_people,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=pin,
        )

    def fit(self):
        loader = self.make_loader()
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config.get("learning_rate", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        epochs = int(self.config.get("epochs", 10))
        scaler = GradScaler(enabled=(self.device.type == "cuda"))

        writer = None
        if SummaryWriter is not None:
            root = self.config["tensorboard_dir"]
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
            logdir = Path(root) / f"seq_decoder_{ts}"
            writer = SummaryWriter(log_dir=str(logdir))

        recon_logger = SequenceReconstructionLogger(
            seq_model=self,
            interval_steps=int(self.config.get("recon_log_interval", 200)),
            num_samples=int(self.config.get("row_recon_samples", 3)),
            timesteps_to_show=3,
            table_width=38,
            max_movie_scan=5000,
        )

        self.train()
        global_step = 0
        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"SeqDecoder epoch {epoch+1}/{epochs}")
            for M, P in pbar:
                M = [m.to(self.device, non_blocking=True) for m in M]
                P = [p.to(self.device, non_blocking=True) for p in P]
                with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    preds, z_seq = self.forward(M)
                    loss = self.compute_loss(preds, P, z_seq)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_val = float(loss.detach().cpu().item())
                pbar.set_postfix(loss=loss_val)
                if writer is not None:
                    writer.add_scalar("loss/total", loss_val, global_step)
                recon_logger.on_batch_end(global_step)
                global_step += 1

        if writer is not None:
            writer.flush()
            writer.close()
        self.save_model()

    def save_model(self):
        out = Path(self.config["model_dir"])
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "MoviesToPeopleSequenceDecoder.pt")

    def load_model(self):
        p = Path(self.config["model_dir"]) / "MoviesToPeopleSequenceDecoder.pt"
        if p.exists():
            self.load_state_dict(torch.load(p, map_location=self.device))
            self.eval()
        return self
