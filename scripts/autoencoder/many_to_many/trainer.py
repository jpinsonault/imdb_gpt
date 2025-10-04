# scripts/autoencoder/many_to_many/trainer.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.autoencoder.ae_loader import load_autoencoders
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.many_to_many.dataset import ManyToManyDataset, collate_many_to_many
from scripts.autoencoder.many_to_many.model import ManyToManyModel
from scripts.autoencoder.sequence_losses import _sequence_loss_and_breakdown
from scripts.autoencoder.prefetch import CudaPrefetcher
from scripts.autoencoder.print_model import print_model_layers_with_shapes
from scripts.autoencoder.run_logger import build_run_logger
from scripts.autoencoder.training_callbacks.many_to_many_reconstruction import ManyToManyReconstructionLogger
from config import ProjectConfig


class _PhaseTimer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        tqdm.write(f"⏳ {self.label}…")
        return self

    def __exit__(self, a, b, c):
        dt = time.perf_counter() - self.t0
        tqdm.write(f"✅ {self.label}  ({dt:.2f}s)")


def _build_models(config: ProjectConfig, warm_start: bool) -> Tuple[TitlesAutoencoder, PeopleAutoencoder]:
    if warm_start:
        mov, per = load_autoencoders(config=config, warm_start=True, freeze_loaded=False)
        return mov, per
    mov = TitlesAutoencoder(config)
    per = PeopleAutoencoder(config)
    mov.accumulate_stats(); mov.finalize_stats(); mov.build_autoencoder()
    per.accumulate_stats(); per.finalize_stats(); per.build_autoencoder()
    return mov, per


def _make_loader(config: ProjectConfig, mov: TitlesAutoencoder, per: PeopleAutoencoder, seq_len_titles: int, seq_len_people: int, force_mode: str):
    ds = ManyToManyDataset(
        db_path=str(Path(config.db_path)),
        movie_fields=mov.fields,
        people_fields=per.fields,
        seq_len_titles=seq_len_titles,
        seq_len_people=seq_len_people,
        movie_limit=None,
        person_limit=None,
        mode_ratio=None,
        force_mode=force_mode,
    )
    num_workers = int(getattr(config, "num_workers", 0) or 0)
    cfg_pf = int(getattr(config, "prefetch_factor", 2) or 0)
    prefetch_factor = None if num_workers == 0 else max(1, cfg_pf)
    pin = bool(torch.cuda.is_available())
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        collate_fn=collate_many_to_many,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
        drop_last=False,
    )
    return loader, ds


def _clamped_ratio(n_movies: int, n_people: int, min_frac: float, max_frac: float) -> float:
    a = float(max(1, n_movies))
    b = float(max(1, n_people))
    r = a / (a + b)
    return max(min_frac, min(max_frac, r))


def _build_schedule(n_movies: int, n_people: int, min_frac: float, max_frac: float, window: int) -> List[int]:
    r = _clamped_ratio(n_movies, n_people, min_frac, max_frac)
    a = int(round(window * r))
    a = max(1, min(window - 1, a))
    b = window - a
    return [0] * a + [1] * b, r


def _masked_seq_loss(fields, preds, tgts, mask):
    return _sequence_loss_and_breakdown(fields, preds, tgts, mask)


def train_many_to_many(
    config: ProjectConfig,
    steps: int | None = None,
    save_every: int = 0,
    warm_start: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with _PhaseTimer("build/load autoencoders"):
        mov, per = _build_models(config, warm_start=warm_start)

    seq_len_titles = int(getattr(config, "titles_sequence_length", 1) or 1)
    seq_len_people = int(getattr(config, "people_sequence_length", 8) or 8)
    latent_dim = int(config.latent_dim)

    with _PhaseTimer("init many-to-many model"):
        model = ManyToManyModel(
            movie_encoder=mov.encoder,
            people_encoder=per.encoder,
            movie_decoder=mov.decoder,
            people_decoder=per.decoder,
            latent_dim=latent_dim,
            seq_len_titles=seq_len_titles,
            seq_len_people=seq_len_people,
        ).to(device)

    with _PhaseTimer("build data loaders"):
        loader_movies, ds_movies = _make_loader(config, mov, per, seq_len_titles, seq_len_people, force_mode="movies")
        loader_people, ds_people = _make_loader(config, mov, per, seq_len_titles, seq_len_people, force_mode="people")

    mov.encoder.to(device); mov.decoder.to(device)
    per.encoder.to(device); per.decoder.to(device)

    with _PhaseTimer("optimizer + logger"):
        opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=(device.type == "cuda"))
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        run_logger = build_run_logger(config)

    with _PhaseTimer("shape preview"):
        it_m = iter(loader_movies)
        xm0, xp0, yt0, yp0, mt0, mp0, mode0 = next(it_m)
        xm0 = [x.to(device) for x in xm0]
        xp0 = [x.to(device) for x in xp0]
        _ = print_model_layers_with_shapes(model, (xm0, xp0))

    with _PhaseTimer("reconstruction logger"):
        rec_logger = ManyToManyReconstructionLogger(
            movie_ae=mov,
            people_ae=per,
            model=model,
            seq_len_titles=seq_len_titles,
            seq_len_people=seq_len_people,
            interval_steps=int(getattr(config, "callback_interval", 200) or 200),
            num_samples=2,
            table_width=44,
            seed=1234,
        )

    with _PhaseTimer("count rows + schedule"):
        n_movies = ds_movies.count_movies()
        n_people = ds_people.count_people()
        min_frac = float(getattr(config, "min_mode_frac", 0.25) or 0.25)
        max_frac = float(getattr(config, "max_mode_frac", 0.75) or 0.75)
        window = 32
        schedule, frac_movies = _build_schedule(n_movies, n_people, min_frac, max_frac, window)
        frac_people = 1.0 - frac_movies
        tqdm.write(f"📊 ratio (titles:people) = {frac_movies:.3f}:{frac_people:.3f}  |  counts {n_movies:,}:{n_people:,}")
        tqdm.write(f"🎛️ schedule window={window} → A={schedule.count(0)} (movie→people), B={schedule.count(1)} (person→titles)")

    with _PhaseTimer("prefetchers"):
        prefetch_m = CudaPrefetcher(loader_movies, device)
        prefetch_p = CudaPrefetcher(loader_people, device)

    def _next_from(mode: int):
        if mode == 0:
            b = prefetch_m.next()
            if b is None:
                pm = CudaPrefetcher(loader_movies, device)
                return pm.next()
            return b
        b = prefetch_p.next()
        if b is None:
            pp = CudaPrefetcher(loader_people, device)
            return pp.next()
        return b

    tqdm.write("🚀 training…")

    model.train()
    step = 0
    infinite = steps is None or steps <= 0
    total_for_bar = None if infinite else steps
    sch_idx = 0

    try:
        with tqdm(total=total_for_bar, desc="train", dynamic_ncols=True, miniters=50) as pbar:
            while infinite or step < steps:
                mode = schedule[sch_idx]
                sch_idx = (sch_idx + 1) % len(schedule)
                tag = "A movie→people" if mode == 0 else "B person→titles"
                pbar.set_description(f"train [{tag}]")

                batch = _next_from(mode)
                if batch is None:
                    continue

                xm, xp, yt, yp, mt, mp, _ = batch

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds_titles_seq, preds_people_seq, z_m2p, z_p2m = model(xm, xp)

                    if mode == 0:
                        rec_people, people_break = _masked_seq_loss(per.fields, preds_people_seq, yp, mp)
                        loss = rec_people
                    else:
                        rec_titles, titles_break = _masked_seq_loss(mov.fields, preds_titles_seq, yt, mt)
                        loss = rec_titles

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=float(loss.detach().cpu().item()))

                if mode == 0:
                    run_logger.add_scalars(float(loss.detach().cpu().item()), float(loss.detach().cpu().item()), 0.0, 0.0, opt)
                    run_logger.add_field_losses("loss/people_seq", people_break)
                else:
                    run_logger.add_scalars(float(loss.detach().cpu().item()), float(loss.detach().cpu().item()), 0.0, 0.0, opt)
                    run_logger.add_field_losses("loss/titles_seq", titles_break)

                run_logger.tick(float(loss.detach().cpu().item()), float(loss.detach().cpu().item()), 0.0)

                rec_logger.on_batch_end(
                    global_step=step,
                    batch=(xm, xp, yt, yp, mt, mp),
                    preds=(preds_titles_seq, preds_people_seq),
                )

                if save_every and step % save_every == 0:
                    out_dir = Path(config.model_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), out_dir / f"ManyToManyModel_step_{step}.pt")
                    tqdm.write(f"💾 saved checkpoint @ step {step}")
    finally:
        out_dir = Path(config.model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "ManyToManyModel_final.pt")
        run_logger.close()
        tqdm.write("🏁 training finished")

    return model
