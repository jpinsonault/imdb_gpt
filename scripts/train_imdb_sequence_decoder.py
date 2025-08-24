# scripts/train_imdb_sequence_decoder.py

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sqlite3
from scripts.autoencoder.print_model import print_model_layers_with_shapes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from config import project_config
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.fields import BaseField
from scripts.autoencoder.fields import TextField, MultiCategoryField, BooleanField, ScalarField, SingleCategoryField, NumericDigitCategoryField
from scripts.autoencoder.training_callbacks import SequenceReconstructionLogger
from scripts.autoencoder.run_logger import build_run_logger


class MoviesPeopleSequenceDataset(IterableDataset):
    def __init__(
        self,
        db_path: str,
        movie_fields: List[BaseField],
        people_fields: List[BaseField],
        seq_len: int,
        movie_limit: int | None = None,
    ):
        super().__init__()
        self.db_path = db_path
        self.movie_fields = movie_fields
        self.people_fields = people_fields
        self.seq_len = seq_len
        self.movie_limit = movie_limit
        self.movie_sql = """
        SELECT
            t.tconst,
            t.primaryTitle,
            t.startYear,
            t.endYear,
            t.runtimeMinutes,
            t.averageRating,
            t.numVotes,
            GROUP_CONCAT(g.genre, ',')
        FROM titles t
        INNER JOIN title_genres g ON g.tconst = t.tconst
        WHERE
            t.startYear IS NOT NULL
            AND t.averageRating IS NOT NULL
            AND t.runtimeMinutes IS NOT NULL
            AND t.runtimeMinutes >= 5
            AND t.startYear >= 1850
            AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries')
            AND t.numVotes >= 10
        GROUP BY t.tconst
        """
        self.people_sql = """
        SELECT
            p.primaryName,
            p.birthYear,
            p.deathYear,
            GROUP_CONCAT(pp.profession, ',')
        FROM people p
        LEFT JOIN people_professions pp ON p.nconst = pp.nconst
        INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
        GROUP BY p.nconst
        HAVING COUNT(pp.profession) > 0
        ORDER BY pr.ordering
        LIMIT ?
        """

    def __iter__(self):
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = con.cursor()
        lim = "" if self.movie_limit is None else f" LIMIT {int(self.movie_limit)}"
        cur.execute(self.movie_sql + lim)
        for tconst, title, startYear, endYear, runtime, rating, votes, genres in cur:
            movie_row = {
                "tconst": tconst,
                "primaryTitle": title,
                "startYear": startYear,
                "endYear": endYear,
                "runtimeMinutes": runtime,
                "averageRating": rating,
                "numVotes": votes,
                "genres": genres.split(",") if genres else [],
            }
            ppl = []
            for r in con.execute(self.people_sql, (tconst, self.seq_len)):
                ppl.append(
                    {
                        "primaryName": r[0],
                        "birthYear": r[1],
                        "deathYear": r[2],
                        "professions": r[3].split(",") if r[3] else None,
                    }
                )
            if not ppl:
                continue
            orig_len = min(len(ppl), self.seq_len)
            if orig_len < self.seq_len:
                ppl = ppl + [{} for _ in range(self.seq_len - orig_len)]
            else:
                ppl = ppl[: self.seq_len]
            x_movie = [f.transform(movie_row.get(f.name)) for f in self.movie_fields]
            y_people = []
            for f in self.people_fields:
                steps = []
                for t in range(self.seq_len):
                    if t < orig_len:
                        steps.append(f.transform_target(ppl[t].get(f.name)))
                    else:
                        steps.append(f.get_base_padding_value())
                y_people.append(torch.stack(steps, dim=0))
            mask = torch.zeros(self.seq_len, dtype=torch.float32)
            mask[:orig_len] = 1.0
            yield x_movie, y_people, mask
        con.close()


def _collate(batch):
    xm_cols = list(zip(*[b[0] for b in batch]))
    yp_cols = list(zip(*[b[1] for b in batch]))
    Xm = [torch.stack(col, dim=0) for col in xm_cols]
    Yp = [torch.stack(col, dim=0) for col in yp_cols]
    M = torch.stack([b[2] for b in batch], dim=0)
    return Xm, Yp, M


class _TransformerTrunk(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.query = nn.Parameter(torch.randn(seq_len, latent_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_mult * latent_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z_m: torch.Tensor) -> torch.Tensor:
        memory = z_m.unsqueeze(0)
        q = self.query.unsqueeze(1).expand(self.seq_len, z_m.size(0), z_m.size(1))
        out = self.decoder(q, memory)
        out = out.transpose(0, 1)
        out = self.norm(out)
        return out


class MovieToPeopleSequencePredictor(nn.Module):
    def __init__(
        self,
        movie_encoder: nn.Module,
        people_decoder: nn.Module,
        latent_dim: int,
        seq_len: int,
        width: int | None = None,
        depth: int = 3,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.movie_encoder = movie_encoder
        self.people_decoder = people_decoder
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.trunk = _TransformerTrunk(
            latent_dim=latent_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )

    def forward(self, movie_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        z_m = self.movie_encoder(movie_inputs)
        z_seq = self.trunk(z_m)
        b = z_seq.size(0)
        flat = z_seq.reshape(b * self.seq_len, self.latent_dim)
        outs = self.people_decoder(flat)
        seq_outs = []
        for y in outs:
            seq_outs.append(y.view(b, self.seq_len, *y.shape[1:]))
        return seq_outs


def _per_sample_field_loss_seq(field: BaseField, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if isinstance(field, ScalarField):
        diff = pred - tgt
        if diff.dim() > 2:
            diff = diff.flatten(2)
            loss = diff.pow(2).mean(dim=2)
        else:
            loss = diff.pow(2)
        return loss
    if isinstance(field, BooleanField):
        if getattr(field, "use_bce_loss", True):
            loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            if loss.dim() > 2:
                loss = loss.flatten(2).mean(dim=2)
            return loss
        diff = torch.tanh(pred) - tgt
        if diff.dim() > 2:
            diff = diff.flatten(2).mean(dim=2)
        else:
            diff = diff
        return diff.pow(2)
    if isinstance(field, MultiCategoryField):
        loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        if loss.dim() > 2:
            loss = loss.flatten(2).mean(dim=2)
        return loss
    if isinstance(field, SingleCategoryField):
        B, T = pred.shape[0], pred.shape[1]
        C = pred.shape[-1]
        t = tgt.long().squeeze(-1)
        loss = F.cross_entropy(pred.view(B * T, C), t.view(B * T), reduction="none")
        return loss.view(B, T)
    if isinstance(field, TextField):
        B, T, L, V = pred.shape
        pad_id = int(field.pad_token_id)
        loss_flat = F.cross_entropy(pred.view(B * T, L, V).reshape(B * T * L, V), tgt.view(B * T, L).reshape(B * T * L), ignore_index=pad_id, reduction="none")
        loss_flat = loss_flat.view(B * T, L)
        mask_tok = (tgt.view(B * T, L) != pad_id).float()
        denom = mask_tok.sum(dim=1).clamp_min(1.0)
        loss_bt = (loss_flat * mask_tok).sum(dim=1) / denom
        return loss_bt.view(B, T)
    if isinstance(field, NumericDigitCategoryField):
        B, T, P, V = pred.shape
        loss_flat = F.cross_entropy(pred.view(B * T * P, V), tgt.view(B * T * P).long(), reduction="none")
        loss_bt = loss_flat.view(B * T, P).mean(dim=1)
        return loss_bt.view(B, T)
    diff = pred - tgt
    if diff.dim() > 2:
        diff = diff.flatten(2).mean(dim=2)
    return diff.pow(2)


def _sequence_loss_and_breakdown(
    fields: List[BaseField],
    preds_seq: List[torch.Tensor],
    targets_seq: List[torch.Tensor],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = 0.0
    field_losses: dict[str, float] = {}
    denom = mask.sum().clamp_min(1.0)
    for f, pred, tgt in zip(fields, preds_seq, targets_seq):
        per_bt = _per_sample_field_loss_seq(f, pred, tgt)
        wsum = (per_bt * mask).sum()
        val = wsum / denom
        total = total + val * float(f.weight)
        field_losses[f.name] = float(val.detach().cpu().item())
    return total, field_losses


def _unit_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _info_nce_pair(z_pred: torch.Tensor, z_tgt: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z_pred @ z_tgt.t()
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def _encode_people_latents_for_timestep(
    people_encoder: nn.Module,
    targets_seq: List[torch.Tensor],
    timestep: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    xs = []
    for y in targets_seq:
        sl = y.index_select(dim=1, index=torch.tensor([timestep], device=y.device))
        sl = sl.squeeze(1)
        sl = sl.index_select(dim=0, index=indices)
        xs.append(sl)
    with torch.no_grad():
        z = people_encoder(xs)
    return z


def _load_frozen_autoencoders(config: Dict[str, Any]) -> Tuple[TitlesAutoencoder, PeopleAutoencoder]:
    mov = TitlesAutoencoder(config)
    per = PeopleAutoencoder(config)
    mov.accumulate_stats()
    mov.finalize_stats()
    per.accumulate_stats()
    per.finalize_stats()
    mov.build_autoencoder()
    per.build_autoencoder()
    model_dir = Path(config["model_dir"])
    mov.encoder.load_state_dict(torch.load(model_dir / "TitlesAutoencoder_encoder.pt", map_location="cpu"))
    per.decoder.load_state_dict(torch.load(model_dir / "PeopleAutoencoder_decoder.pt", map_location="cpu"))
    per.encoder.load_state_dict(torch.load(model_dir / "PeopleAutoencoder_encoder.pt", map_location="cpu"))
    for p in mov.encoder.parameters():
        p.requires_grad = False
    for p in per.decoder.parameters():
        p.requires_grad = False
    for p in per.encoder.parameters():
        p.requires_grad = False
    mov.encoder.eval()
    per.decoder.eval()
    per.encoder.eval()
    return mov, per


def train_sequence_predictor(
    config: Dict[str, Any],
    steps: int,
    save_every: int,
    movie_limit: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mov, per = _load_frozen_autoencoders(config)

    latent_dim = int(config["latent_dim"])
    seq_len = int(config["people_sequence_length"])
    batch_size = int(config["batch_size"])
    lr = float(config.get("learning_rate", 2e-4))
    wd = float(config.get("weight_decay", 1e-4))
    temp = float(config.get("latent_temperature", 0.07))
    w_lat = float(config.get("latent_loss_weight", 1.0))
    w_rec = float(config.get("recon_loss_weight", 0.1))
    log_every = int(config.get("log_interval", 20))

    model = MovieToPeopleSequencePredictor(
        movie_encoder=mov.encoder,
        people_decoder=per.decoder,
        latent_dim=latent_dim,
        seq_len=seq_len,
    ).to(device)

    ds = MoviesPeopleSequenceDataset(
        db_path=str(Path(config["db_path"])),
        movie_fields=mov.fields,
        people_fields=per.fields,
        seq_len=seq_len,
        movie_limit=movie_limit,
    )

    num_workers = int(config.get("num_workers", 2))
    prefetch_factor = int(config.get("prefetch_factor", 2))
    pin = bool(torch.cuda.is_available())

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
    )

    mov.encoder.to(device)
    per.decoder.to(device)
    per.encoder.to(device)

    opt = torch.optim.AdamW(model.trunk.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    run_logger = build_run_logger(config)
    seq_logger = SequenceReconstructionLogger(
        movie_ae=mov,
        people_ae=per,
        predictor=model,
        db_path=str(Path(config["db_path"])),
        seq_len=seq_len,
        interval_steps=int(config.get("callback_interval", 100)),
        num_samples=2,
        table_width=38,
    )

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    it_preview = iter(loader)
    xm0, _, _ = next(it_preview)
    xm0 = [x.to(device) for x in xm0]
    print_model_layers_with_shapes(model, xm0)

    step = 0
    it = iter(loader)
    model.train()

    with tqdm(total=steps, desc="sequence", dynamic_ncols=True, miniters=50) as pbar:
        while step < steps:
            t0 = time.perf_counter()
            try:
                xm, yp, m = next(it)
            except StopIteration:
                it = iter(loader)
                xm, yp, m = next(it)
            t1 = time.perf_counter()
            xm = [x.to(device, non_blocking=True) for x in xm]
            yp = [y.to(device, non_blocking=True) for y in yp]
            m = m.to(device, non_blocking=True)
            _sync()
            t2 = time.perf_counter()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _sync()
                t3 = time.perf_counter()
                z_m = mov.encoder(xm)
                _sync()
                t4 = time.perf_counter()
                z_seq = model.trunk(z_m)
                _sync()
                t5 = time.perf_counter()
                b = z_seq.size(0)
                flat = z_seq.reshape(b * seq_len, latent_dim)
                outs = per.decoder(flat)
                preds = [y.view(b, seq_len, *y.shape[1:]) for y in outs]
                _sync()
                t6 = time.perf_counter()

                rec_loss, field_breakdown = _sequence_loss_and_breakdown(per.fields, preds, yp, m)
                _sync()
                t7 = time.perf_counter()

                z_seq_pred = _unit_normalize(z_seq)
                nce_losses = []
                for t in range(seq_len):
                    idx = torch.nonzero(m[:, t] > 0.5, as_tuple=False).flatten()
                    if idx.numel() < 2:
                        continue
                    zt = _encode_people_latents_for_timestep(per.encoder, yp, t, idx)
                    zt = _unit_normalize(zt)
                    zp = z_seq_pred.index_select(dim=1, index=torch.tensor([t], device=z_seq_pred.device)).squeeze(1)
                    zp = zp.index_select(dim=0, index=idx)
                    nce_losses.append(_info_nce_pair(zp, zt, temp))
                _sync()
                if nce_losses:
                    nce_loss = torch.stack(nce_losses).mean()
                else:
                    nce_loss = torch.tensor(0.0, device=device)

                loss = w_lat * nce_loss + w_rec * rec_loss

            _sync()
            t8 = time.perf_counter()
            opt.zero_grad()
            scaler.scale(loss).backward()
            _sync()
            t9 = time.perf_counter()
            scaler.step(opt)
            scaler.update()
            _sync()
            t10 = time.perf_counter()

            step += 1
            pbar.update(1)

            iter_time = t10 - t0
            tim = {
                "data": t1 - t0,
                "h2d": t2 - t1,
                "mov_enc": t4 - t3,
                "trunk": t5 - t4,
                "ppl_dec": t6 - t5,
                "rec": t7 - t6,
                "tgt_enc": (t8 - t7) - (t8 - t7 - sum([])),
                "nce": 0.0,
                "backward": t9 - t8,
                "opt": t10 - t9,
                "total": iter_time,
            }

            do_log = (step % log_every == 0)
            if do_log:
                run_logger.add_scalars(
                    float(loss.detach().cpu().item()),
                    float(rec_loss.detach().cpu().item()),
                    float(nce_loss.detach().cpu().item()),
                    iter_time,
                    opt,
                )
                run_logger.add_field_losses("loss/sequence_people", field_breakdown)
                run_logger.add_field_losses("time", tim)
                run_logger.tick(
                    float(loss.detach().cpu().item()),
                    float(rec_loss.detach().cpu().item()),
                    float(nce_loss.detach().cpu().item()),
                )

            seq_logger.on_batch_end(step)

            if save_every and step % save_every == 0:
                out = Path(config["model_dir"])
                out.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out / f"MovieToPeopleSequencePredictor_step_{step}.pt")

    out = Path(config["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "MovieToPeopleSequencePredictor_final.pt")
    run_logger.close()


def main():
    parser = argparse.ArgumentParser(description="Train movie-to-people sequence decoder with latent supervision")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--movie-limit", type=int, default=0)
    args = parser.parse_args()
    ml = args.movie_limit if args.movie_limit > 0 else None
    train_sequence_predictor(project_config, steps=args.steps, save_every=args.save_every, movie_limit=ml)


if __name__ == "__main__":
    main()
