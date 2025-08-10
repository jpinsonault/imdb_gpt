from __future__ import annotations
import argparse
import logging
import os
import signal
import time
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from config import project_config
from scripts.autoencoder.edge_loss_logger import EdgeLossLogger
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.joint_edge_sampler import make_edge_sampler
from scripts.autoencoder.training_callbacks import (
    JointReconstructionLogger,
    RowReconstructionLogger,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


def info_nce_loss(movie_z: torch.Tensor, person_z: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = cosine_similarity(movie_z, person_z) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    la = F.cross_entropy(logits, labels)
    lb = F.cross_entropy(logits.t(), labels)
    return 0.5 * (la + lb)


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


def build_joint_trainer(
    config: Dict[str, Any],
    warm: bool,
    db_path: Path,
) -> Tuple[JointAutoencoder, DataLoader, EdgeLossLogger, TitlesAutoencoder, PeopleAutoencoder, int]:

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

    edge_gen = make_edge_sampler(
        db_path=str(db_path),
        movie_ae=movie_ae,
        person_ae=people_ae,
        batch_size=config["batch_size"],
        refresh_batches=config["edge_sampler"]["refresh_batches"],
        boost=config["edge_sampler"]["weak_edge_boost"],
    )

    ds = _EdgeIterable(edge_gen, movie_ae, people_ae)

    num_workers = int(config.get("num_workers", 2))
    prefetch_factor = int(config.get("prefetch_factor", 2))
    pin = bool(torch.cuda.is_available())

    loader = DataLoader(
        ds,
        batch_size=config["batch_size"],
        collate_fn=_collate_edge,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin,
    )

    loss_logger = EdgeLossLogger(str(db_path))

    joint = JointAutoencoder(movie_ae, people_ae).to(device)
    total_edges = len(edge_gen.edges)
    logging.info(f"joint trainer ready device={device} edges={total_edges} bs={config['batch_size']} workers={num_workers}")
    return joint, loader, loss_logger, movie_ae, people_ae, total_edges


def _fmt(x: float) -> str:
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{x/1e3:.2f}k"
    return f"{x:.2f}"


def _unique_log_dir(root: str, base: str) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    name = f"{base}_{ts}"
    path = os.path.join(root, name)
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        cand = f"{path}_{i}"
        if not os.path.exists(cand):
            return cand
        i += 1


def main():
    parser = argparse.ArgumentParser(description="Train movie↔people joint embedding autoencoder")
    parser.add_argument("--warm", action="store_true")
    args = parser.parse_args()

    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / "imdb.db"

    joint_model, loader, logger, mov_ae, per_ae, total_edges = build_joint_trainer(project_config, args.warm, db_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = torch.optim.AdamW(
        joint_model.parameters(),
        lr=float(project_config["learning_rate"]),
        weight_decay=float(project_config["weight_decay"]),
    )
    temperature = float(project_config.get("nce_temp", 0.07))
    log_interval = int(project_config.get("log_interval", 20))
    save_interval = int(project_config.get("save_interval", 10000))
    flush_interval = int(project_config.get("flush_interval", 2000))

    writer = None
    if SummaryWriter is not None:
        tb_root = project_config.get("tensorboard_dir", "runs")
        run_dir = _unique_log_dir(tb_root, "joint_autoencoder")
        writer = SummaryWriter(log_dir=run_dir)
        logging.info(f"tensorboard logdir: {run_dir}")

    jr_interval = int(project_config.get("recon_log_interval", 500))
    rr_interval = int(project_config.get("row_recon_interval", 1000))
    rr_samples = int(project_config.get("row_recon_samples", 3))

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

    last_log_t = time.perf_counter()
    since_last_log = 0

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
        for f, pred, tgt in zip(mov_ae.fields, m_rec, M):
            rec_loss = rec_loss + f.compute_loss(pred, tgt) * float(f.weight)
        for f, pred, tgt in zip(per_ae.fields, p_rec, P):
            rec_loss = rec_loss + f.compute_loss(pred, tgt) * float(f.weight)

        nce = info_nce_loss(m_z, p_z, temperature=temperature)
        total = rec_loss + 0.0 * nce

        opt.zero_grad()
        total.backward()
        opt.step()

        batch_time = time.perf_counter() - t_step0
        iter_time = data_time + batch_time

        total_val = float(total.detach().cpu().item())
        rec_val = float(rec_loss.detach().cpu().item())
        nce_val = float(nce.detach().cpu().item())

        for eid in eids.detach().cpu().tolist():
            logger.add(int(eid), 0, global_step, total_val, {})

        if writer is not None:
            writer.add_scalar("loss/total", total_val, global_step)
            writer.add_scalar("loss/reconstruction", rec_val, global_step)
            writer.add_scalar("loss/nce", nce_val, global_step)
            writer.add_scalar("time/iter_sec", iter_time, global_step)
            for i, g in enumerate(opt.param_groups):
                writer.add_scalar(f"lr/group_{i}", float(g["lr"]), global_step)

        since_last_log += 1
        if since_last_log % log_interval == 0:
            now = time.perf_counter()
            elapsed = max(1e-9, now - last_log_t)
            ips = (log_interval * project_config["batch_size"]) / elapsed
            last_log_t = now
            logging.info(
                f"step {global_step} "
                f"loss {total_val:.4f} rec {rec_val:.4f} nce {nce_val:.4f} "
                f"dt {data_time:.3f}s bt {batch_time:.3f}s ips {_fmt(ips)}"
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

    out = Path(project_config["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(joint_model.state_dict(), out / "JointMoviePersonAE_final.pt")
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
