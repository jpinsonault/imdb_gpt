import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sqlite3
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from config import project_config
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.fields import BaseField
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
            if len(ppl) < self.seq_len:
                ppl = ppl + [ppl[-1]] * (self.seq_len - len(ppl))
            else:
                ppl = ppl[: self.seq_len]
            x_movie = [f.transform(movie_row.get(f.name)) for f in self.movie_fields]
            y_people = []
            for f in self.people_fields:
                seq = [f.transform_target(pr.get(f.name)) for pr in ppl]
                y_people.append(torch.stack(seq, dim=0))
            yield x_movie, y_people
        con.close()


def _collate(batch):
    xm_cols = list(zip(*[b[0] for b in batch]))
    yp_cols = list(zip(*[b[1] for b in batch]))
    Xm = [torch.stack(col, dim=0) for col in xm_cols]
    Yp = [torch.stack(col, dim=0) for col in yp_cols]
    return Xm, Yp


class MovieToPeopleSequencePredictor(nn.Module):
    def __init__(
        self,
        movie_encoder: nn.Module,
        people_decoder: nn.Module,
        latent_dim: int,
        seq_len: int,
        width: int | None = None,
        depth: int = 3,
    ):
        super().__init__()
        self.movie_encoder = movie_encoder
        self.people_decoder = people_decoder
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        w = width or latent_dim * 2
        layers = []
        layers.append(nn.Linear(latent_dim, w))
        layers.append(nn.GELU())
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(w, w))
            layers.append(nn.GELU())
        layers.append(nn.Linear(w, latent_dim * seq_len))
        self.trunk = nn.Sequential(*layers)

    def forward(self, movie_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        z_m = self.movie_encoder(movie_inputs)
        b = z_m.size(0)
        z_seq = self.trunk(z_m).view(b, self.seq_len, self.latent_dim)
        flat = z_seq.reshape(b * self.seq_len, self.latent_dim)
        outs = self.people_decoder(flat)
        seq_outs = []
        for y in outs:
            seq_outs.append(y.view(b, self.seq_len, *y.shape[1:]))
        return seq_outs


def _sequence_loss_and_breakdown(
    fields: List[BaseField],
    preds_seq: List[torch.Tensor],
    targets_seq: List[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    total = 0.0
    field_losses: dict[str, float] = {}
    for f, pred, tgt in zip(fields, preds_seq, targets_seq):
        b, t = pred.shape[0], pred.shape[1]
        pm = pred.reshape(b * t, *pred.shape[2:])
        tm = tgt.reshape(b * t, *tgt.shape[2:])
        l = f.compute_loss(pm, tm)
        total = total + l * float(f.weight)
        field_losses[f.name] = float(l.detach().cpu().item())
    return total, field_losses


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
    for p in mov.encoder.parameters():
        p.requires_grad = False
    for p in per.decoder.parameters():
        p.requires_grad = False
    mov.encoder.eval()
    per.decoder.eval()
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

    model = MovieToPeopleSequencePredictor(mov.encoder, per.decoder, latent_dim, seq_len).to(device)

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

    step = 0
    it = iter(loader)
    model.train()

    with tqdm(total=steps, desc="sequence", dynamic_ncols=True) as pbar:
        while step < steps:
            t_fetch0 = time.perf_counter()
            try:
                xm, yp = next(it)
            except StopIteration:
                it = iter(loader)
                xm, yp = next(it)
            data_time = time.perf_counter() - t_fetch0

            xm = [x.to(device, non_blocking=True) for x in xm]
            yp = [y.to(device, non_blocking=True) for y in yp]

            t_step0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(xm)
                loss, field_breakdown = _sequence_loss_and_breakdown(per.fields, preds, yp)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            batch_time = time.perf_counter() - t_step0
            iter_time = data_time + batch_time

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

            run_logger.add_scalars(float(loss.detach().cpu().item()), float(loss.detach().cpu().item()), 0.0, iter_time, opt)
            run_logger.add_field_losses("loss/sequence_people", field_breakdown)
            run_logger.tick(float(loss.detach().cpu().item()), float(loss.detach().cpu().item()), 0.0)

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
    parser = argparse.ArgumentParser(description="Train movie-to-people sequence decoder using frozen row autoencoders")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--movie-limit", type=int, default=0)
    args = parser.parse_args()
    ml = args.movie_limit if args.movie_limit > 0 else None
    train_sequence_predictor(project_config, steps=args.steps, save_every=args.save_every, movie_limit=ml)


if __name__ == "__main__":
    main()
