# scripts/train_joint_autoencoder.py
from __future__ import annotations
import argparse
import logging
import signal
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from config import ProjectConfig, project_config
from scripts.autoencoder.mapping_samplers import LossLedger
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.joint_edge_sampler import make_edge_sampler
from scripts.autoencoder.training_callbacks import JointReconstructionLogger, RowReconstructionLogger
from scripts.autoencoder.run_logger import build_run_logger
from scripts.autoencoder.fields import (
    TextField,
    MultiCategoryField,
    BooleanField,
    ScalarField,
    SingleCategoryField,
    NumericDigitCategoryField,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


def info_nce_components(movie_z: torch.Tensor, person_z: torch.Tensor, temperature: float):
    logits = cosine_similarity(movie_z, person_z) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    row_ce = F.cross_entropy(logits, labels, reduction="none")
    col_ce = F.cross_entropy(logits.t(), labels, reduction="none")
    nce_per = 0.5 * (row_ce + col_ce)
    nce_mean = nce_per.mean()
    return nce_mean, nce_per


class _EdgeIterable(IterableDataset):
    def __init__(self, sampler, mov_ae: TitlesAutoencoder, per_ae: PeopleAutoencoder):
        super().__init__()
        self.sampler = sampler
        self.mov_fields = mov_ae.fields
        self.per_fields = per_ae.fields

    def __iter__(self):
        for m_t, p_t, eid in self.sampler:
            yield m_t, p_t, eid


def _collate_edge(batch):
    ms, ps, eids = zip(*batch)
    m_cols = list(zip(*ms))
    p_cols = list(zip(*ps))
    M = [torch.stack(col, dim=0) for col in m_cols]
    P = [torch.stack(col, dim=0) for col in p_cols]
    e = torch.tensor(eids, dtype=torch.long)
    return M, P, e


class JointAutoencoder(nn.Module):
    def __init__(self, movie_ae: TitlesAutoencoder, person_ae: PeopleAutoencoder):
        super().__init__()
        self.movie_ae = movie_ae
        self.person_ae = person_ae
        self.mov_enc = movie_ae.encoder
        self.per_enc = person_ae.encoder
        self.mov_dec = movie_ae.decoder
        self.per_dec = person_ae.decoder

    def forward(self, movie_in: List[torch.Tensor], person_in: List[torch.Tensor]):
        m_z = self.mov_enc(movie_in)
        p_z = self.per_enc(person_in)
        m_rec = self.mov_dec(m_z)
        p_rec = self.per_dec(p_z)
        return m_z, p_z, m_rec, p_rec


def _reduce_per_sample_mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    diff = pred - tgt
    if diff.dim() > 1:
        return diff.pow(2).reshape(diff.size(0), -1).mean(dim=1)
    return diff.pow(2)


def _per_sample_field_loss(field, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3:
        if tgt.dim() == 3:
            tgt = tgt.argmax(dim=-1)
        if hasattr(field, "tokenizer"):
            B, L, V = pred.shape
            pad_id = int(getattr(field, "pad_token_id", 0) or 0)
            loss_flat = F.cross_entropy(
                pred.reshape(B * L, V),
                tgt.reshape(B * L),
                ignore_index=pad_id,
                reduction="none",
            )
            loss_bt = loss_flat.reshape(B, L)
            mask_tok = (tgt.reshape(B, L) != pad_id).float()
            denom = mask_tok.sum(dim=1).clamp_min(1.0)
            return (loss_bt * mask_tok).sum(dim=1) / denom
        if hasattr(field, "base"):
            B, P, V = pred.shape
            loss_flat = F.cross_entropy(
                pred.reshape(B * P, V),
                tgt.reshape(B * P).long(),
                reduction="none",
            )
            return loss_flat.reshape(B, P).mean(dim=1)

    if isinstance(field, ScalarField):
        return _reduce_per_sample_mse(pred, tgt)

    if isinstance(field, BooleanField):
        if getattr(field, "use_bce_loss", True):
            loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            if loss.dim() > 1:
                loss = loss.reshape(loss.size(0), -1).mean(dim=1)
            return loss
        return _reduce_per_sample_mse(torch.tanh(pred), tgt)

    if isinstance(field, MultiCategoryField):
        loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        if loss.dim() > 1:
            loss = loss.reshape(loss.size(0), -1).mean(dim=1)
        return loss

    if isinstance(field, SingleCategoryField):
        B, C = pred.shape[0], pred.shape[-1]
        t = tgt.long().squeeze(-1)
        return F.cross_entropy(pred.reshape(B, C), t.reshape(B), reduction="none")

    return _reduce_per_sample_mse(pred, tgt)


# scripts/train_joint_autoencoder.py
def build_joint_trainer(
    config: ProjectConfig,
    warm: bool,
    db_path: Path,
):
    movie_ae = TitlesAutoencoder(config)
    people_ae = PeopleAutoencoder(config)

    if warm:
        raise NotImplementedError("Warm start is not implemented yet.")
    else:
        movie_ae.accumulate_stats()
        movie_ae.finalize_stats()
        movie_ae.build_autoencoder()
        people_ae.accumulate_stats()
        people_ae.finalize_stats()
        people_ae.build_autoencoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    movie_ae.model.to(device)
    people_ae.model.to(device)

    loss_logger = LossLedger()

    edge_gen = make_edge_sampler(
        db_path=str(db_path),
        movie_ae=movie_ae,
        person_ae=people_ae,
        batch_size=config.batch_size,
        refresh_batches=config.refresh_batches,
        boost=config.weak_edge_boost,
        loss_logger=loss_logger,
    )

    ds = _EdgeIterable(edge_gen, movie_ae, people_ae)

    num_workers = int(getattr(config, "num_workers", 0) or 0)
    cfg_pf = int(getattr(config, "prefetch_factor", 2) or 0)
    prefetch_factor = None if num_workers == 0 else max(1, cfg_pf)
    pin = bool(torch.cuda.is_available())

    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        collate_fn=_collate_edge,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
    )

    joint = JointAutoencoder(movie_ae, people_ae).to(device)
    total_edges = len(edge_gen.edges)
    logging.info(f"joint trainer ready device={device} edges={total_edges} bs={config.batch_size} workers={num_workers}")
    return joint, loader, loss_logger, movie_ae, people_ae, total_edges


def main(config: ProjectConfig):
    parser = argparse.ArgumentParser(description="Train movie-person joint embedding autoencoder")
    parser.add_argument("--warm", action="store_true")
    args = parser.parse_args()

    data_dir = Path(config.data_dir)
    db_path = data_dir / "imdb.db"

    joint_model, loader, logger, mov_ae, per_ae, total_edges = build_joint_trainer(config, args.warm, db_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = torch.optim.AdamW(
        joint_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    temperature = config.nce_temp
    nce_weight = config.nce_weight
    save_interval = config.save_interval
    flush_interval = config.flush_interval
    batch_size = config.batch_size

    run_logger = build_run_logger(config)
    if run_logger.run_dir:
        logging.info(f"tensorboard logdir: {run_logger.run_dir}")

    jr_interval = config.callback_interval
    rr_interval = config.callback_interval
    rr_samples = config.row_recon_samples

    joint_recon = JointReconstructionLogger(
        mov_ae,
        per_ae,
        str(db_path),
        interval_steps=jr_interval,
        num_samples=4,
        table_width=38,
    )
    row_recon_movies = RowReconstructionLogger(
        interval_steps=rr_interval,
        row_autoencoder=mov_ae,
        db_path=str(db_path),
        num_samples=rr_samples,
        table_width=40,
    )
    row_recon_people = RowReconstructionLogger(
        interval_steps=rr_interval,
        row_autoencoder=per_ae,
        db_path=str(db_path),
        num_samples=rr_samples,
        table_width=40,
    )

    stop_flag = {"stop": False}

    def _handle_sigint(signum, frame):
        stop_flag["stop"] = True
        logging.info("interrupt received; will finish current step and save")

    signal.signal(signal.SIGINT, _handle_sigint)

    if torch.cuda.is_available():
        logging.info(f"cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")

    joint_model.train()
    global_step = 0
    data_iter = iter(loader)
    edges_seen = 0

    with tqdm(unit="batch", dynamic_ncols=True) as pbar:
        while True:
            t_fetch0 = time.perf_counter()
            M, P, eids = next(data_iter)
            data_time = time.perf_counter() - t_fetch0

            M = [m.to(device, non_blocking=True) for m in M]
            P = [p.to(device, non_blocking=True) for p in P]
            eids = eids.to(device)

            t_step0 = time.perf_counter()
            m_z, p_z, m_rec, p_rec = joint_model(M, P)

            rec_loss = 0.0
            field_losses_movie: Dict[str, float] = {}
            field_losses_person: Dict[str, float] = {}
            rec_per_sample = torch.zeros(M[0].size(0), device=device)

            for f, pred, tgt in zip(mov_ae.fields, m_rec, M):
                per_s = _per_sample_field_loss(f, pred, tgt) * float(f.weight)
                rec_per_sample = rec_per_sample + per_s
                field_losses_movie[f.name] = float(per_s.mean().detach().cpu().item())
                rec_loss = rec_loss + per_s.mean()

            for f, pred, tgt in zip(per_ae.fields, p_rec, P):
                per_s = _per_sample_field_loss(f, pred, tgt) * float(f.weight)
                rec_per_sample = rec_per_sample + per_s
                field_losses_person[f.name] = float(per_s.mean().detach().cpu().item())
                rec_loss = rec_loss + per_s.mean()

            nce_mean, nce_per = info_nce_components(m_z, p_z, temperature=temperature)
            total = rec_loss + nce_weight * nce_mean
            total_per_sample = rec_per_sample + nce_weight * nce_per

            opt.zero_grad()
            total.backward()
            opt.step()

            batch_time = time.perf_counter() - t_step0
            iter_time = data_time + batch_time

            total_val = float(total.detach().cpu().item())
            rec_val = float(rec_loss.detach().cpu().item())
            nce_val = float(nce_mean.detach().cpu().item())
            batch_min = float(total_per_sample.min().detach().cpu().item())
            batch_max = float(total_per_sample.max().detach().cpu().item())

            for eid, edgeloss in zip(eids.detach().cpu().tolist(), total_per_sample.detach().cpu().tolist()):
                logger.add(int(eid), float(edgeloss))

            run_logger.add_scalars(total_val, rec_val, nce_val, iter_time, opt)
            run_logger.add_field_losses("loss/movie", field_losses_movie)
            run_logger.add_field_losses("loss/person", field_losses_person)
            run_logger.add_extremes(batch_min, batch_max)
            run_logger.tick(total_val, rec_val, nce_val)

            edges_seen += batch_size
            pbar.update(1)
            pbar.set_description(f"batches {global_step + 1}")
            pbar.set_postfix(
                edges=f"{min(edges_seen, total_edges)}/{total_edges}",
                loss=f"{total_val:.4f}",
                rec=f"{rec_val:.4f}",
                nce=f"{nce_val:.4f}",
                min=f"{batch_min:.4f}",
                max=f"{batch_max:.4f}",
            )

            JointReconstructionLogger.on_batch_end(joint_recon, global_step)
            RowReconstructionLogger.on_batch_end(row_recon_movies, global_step)
            RowReconstructionLogger.on_batch_end(row_recon_people, global_step)

            if (global_step + 1) % flush_interval == 0:
                logger.flush()

            if (global_step + 1) % save_interval == 0:
                mov_ae.save_model()
                per_ae.save_model()
                logging.info(f"checkpoint saved at step {global_step+1}")

            global_step += 1
            if stop_flag["stop"]:
                break

    logger.flush()
    mov_ae.save_model()
    per_ae.save_model()

    out = Path(project_config.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(joint_model.state_dict(), out / "JointMoviePersonAE_final.pt")
    run_logger.close()


if __name__ == "__main__":
    main(project_config)
