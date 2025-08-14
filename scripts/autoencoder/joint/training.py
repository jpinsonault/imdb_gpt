from __future__ import annotations
import argparse
import logging
import os
import signal
import time
import datetime
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from scripts.autoencoder.joint.loss_loggers import EdgeLossLogger
from scripts.autoencoder.row_ae.imdb import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.joint.dataset import make_edge_sampler
from scripts.autoencoder.joint.model import JointAutoencoder
from scripts.autoencoder.joint.reconstruction_logger import JointReconstructionLogger
from scripts.autoencoder.row_ae.reconstruction_logger import RowReconstructionLogger

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class _EdgeIterable(IterableDataset):
    def __init__(self, sampler, mov_ae, per_ae):
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


def _ensure_meta_table(db_path: Path):
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS edge_sampler_meta (
            id INTEGER PRIMARY KEY CHECK(id=1),
            version INTEGER NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )
    cur.execute(
        """
        INSERT INTO edge_sampler_meta(id,version,updated_at)
        VALUES(1,0,strftime('%s','now'))
        ON CONFLICT(id) DO NOTHING;
        """
    )
    conn.commit()
    conn.close()


def _bump_sampler_version(db_path: Path):
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE edge_sampler_meta
        SET version = version + 1,
            updated_at = strftime('%s','now')
        WHERE id = 1;
        """
    )
    conn.commit()
    conn.close()


class JointTrainer:
    def __init__(self, config: Dict[str, Any], db_path: Path, warm: bool = False):
        self.config = config
        self.db_path = Path(db_path)
        self.warm = warm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        self.movie_ae: Optional[TitlesAutoencoder] = None
        self.people_ae: Optional[PeopleAutoencoder] = None
        self.joint_model: Optional[JointAutoencoder] = None

        self.edge_sampler = None
        self.loader: Optional[DataLoader] = None
        self.loss_logger: Optional[EdgeLossLogger] = None

        self.joint_recon: Optional[JointReconstructionLogger] = None
        self.row_recon_movies: Optional[RowReconstructionLogger] = None
        self.row_recon_people: Optional[RowReconstructionLogger] = None

        self.writer: Optional[SummaryWriter] = None
        self.run_dir: Optional[str] = None

        self.stop_flag = {"stop": False}
        self.total_edges: int = 0

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return a @ b.t()

    @staticmethod
    def info_nce_loss(movie_z: torch.Tensor, person_z: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = JointTrainer.cosine_similarity(movie_z, person_z) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        la = F.cross_entropy(logits, labels)
        lb = F.cross_entropy(logits.t(), labels)
        return 0.5 * (la + lb)

    @staticmethod
    def compute_joint_field_losses(mov_ae, per_ae, m_pred, p_pred, m_tgt, p_tgt):
        out = {}
        with torch.no_grad():
            for f, pred, tgt in zip(mov_ae.fields, m_pred, m_tgt):
                l = f.compute_loss(pred, tgt) * float(f.weight)
                out[f"movie.{f.name}"] = float(l.detach().cpu().item())
            for f, pred, tgt in zip(per_ae.fields, p_pred, p_tgt):
                l = f.compute_loss(pred, tgt) * float(f.weight)
                out[f"person.{f.name}"] = float(l.detach().cpu().item())
        return out

    @staticmethod
    def _fmt(x: float) -> str:
        if x >= 1e6:
            return f"{x/1e6:.2f}M"
        if x >= 1e3:
            return f"{x/1e3:.2f}k"
        return f"{x:.2f}"

    @staticmethod
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

    def _prepare_writer(self):
        if SummaryWriter is None:
            self.writer, self.run_dir = None, None
            return
        tb_root = self.config.get("tensorboard_dir", "logs")
        self.run_dir = self._unique_log_dir(tb_root, "joint_autoencoder")
        self.writer = SummaryWriter(log_dir=self.run_dir)
        logging.info(f"tensorboard logdir: {self.run_dir}")

    def _build_recon_loggers(self):
        jr_interval = int(self.config.get("recon_log_interval", 500))
        rr_interval = int(self.config.get("row_recon_interval", 1000))
        rr_samples = int(self.config.get("row_recon_samples", 3))

        self.joint_recon = JointReconstructionLogger(
            self.movie_ae,
            self.people_ae,
            str(self.db_path),
            interval_steps=jr_interval,
            num_samples=4,
            table_width=38,
        )
        self.row_recon_movies = RowReconstructionLogger(
            interval_steps=rr_interval,
            row_autoencoder=self.movie_ae,
            db_path=str(self.db_path),
            num_samples=rr_samples,
            table_width=40,
        )
        self.row_recon_people = RowReconstructionLogger(
            interval_steps=rr_interval,
            row_autoencoder=self.people_ae,
            db_path=str(self.db_path),
            num_samples=rr_samples,
            table_width=40,
        )

    def _handle_sigint(self, signum, frame):
        self.stop_flag["stop"] = True
        logging.info("interrupt received; will finish current step and save")

    def build(self):
        t0 = time.perf_counter()
        logging.info("init: building row autoencoders (movies, people)…")

        self.movie_ae = TitlesAutoencoder(self.config)
        self.people_ae = PeopleAutoencoder(self.config)

        if self.warm:
            raise NotImplementedError("Warm start is not implemented yet.")
        else:
            s0 = time.perf_counter()
            self.movie_ae.accumulate_stats()
            self.movie_ae.finalize_stats()
            self.movie_ae.build_autoencoder()
            self.movie_ae.load_model()
            logging.info(f"init: movie AE ready in {time.perf_counter() - s0:.2f}s")

            s1 = time.perf_counter()
            self.people_ae.accumulate_stats()
            self.people_ae.finalize_stats()
            self.people_ae.build_autoencoder()
            self.people_ae.load_model()
            logging.info(f"init: people AE ready in {time.perf_counter() - s1:.2f}s")

        s2 = time.perf_counter()
        logging.info("init: creating edge sampler…")
        self.edge_sampler = make_edge_sampler(
            db_path=str(self.db_path),
            movie_ae=self.movie_ae,
            person_ae=self.people_ae,
            batch_size=int(self.config["batch_size"]),
            refresh_batches=int(self.config["edge_sampler"]["refresh_batches"]),
            boost=float(self.config["edge_sampler"]["weak_edge_boost"]),
        )
        logging.info(f"init: edge sampler ready in {time.perf_counter() - s2:.2f}s")

        ds = _EdgeIterable(self.edge_sampler, self.movie_ae, self.people_ae)

        num_workers = int(self.config.get("num_workers", 2))
        prefetch_factor = int(self.config.get("prefetch_factor", 2))
        pin = bool(torch.cuda.is_available())

        s3 = time.perf_counter()
        logging.info(f"init: constructing DataLoader workers={num_workers} prefetch={prefetch_factor if num_workers>0 else None} pin_memory={pin}")
        self.loader = DataLoader(
            ds,
            batch_size=int(self.config["batch_size"]),
            collate_fn=_collate_edge,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=pin,
        )
        logging.info(f"init: DataLoader ready in {time.perf_counter() - s3:.2f}s")

        self.loss_logger = EdgeLossLogger(str(self.db_path))

        self.joint_model = JointAutoencoder(self.movie_ae, self.people_ae).to(self.device)
        self.total_edges = len(self.edge_sampler.edges)

        _ensure_meta_table(self.db_path)

        logging.info(f"init: joint trainer ready device={self.device} edges={self.total_edges} bs={self.config['batch_size']} workers={num_workers} in {time.perf_counter() - t0:.2f}s")

    def _warm_first_batch(self):
        logging.info("init: warming up DataLoader by prefetching first batch…")
        t0 = time.perf_counter()
        data_iter = iter(self.loader)
        try:
            warm_M, warm_P, warm_eids = next(data_iter)
            dt = time.perf_counter() - t0
            logging.info(f"init: first batch fetched in {dt:.2f}s (batch_size={self.config['batch_size']})")
            return data_iter, (warm_M, warm_P, warm_eids)
        except StopIteration:
            logging.error("init: DataLoader produced no data")
            return iter([]), None

    def _train_one_step(
        self,
        M: List[torch.Tensor],
        P: List[torch.Tensor],
        eids: torch.Tensor,
        opt: torch.optim.Optimizer,
        temperature: float,
        nce_w: float,
    ):
        with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            m_z, p_z, m_rec, p_rec = self.joint_model(M, P)
            rec_loss = 0.0
            for f, pred, tgt in zip(self.movie_ae.fields, m_rec, M):
                rec_loss = rec_loss + f.compute_loss(pred, tgt) * float(f.weight)
            for f, pred, tgt in zip(self.people_ae.fields, p_rec, P):
                rec_loss = rec_loss + f.compute_loss(pred, tgt) * float(f.weight)
            nce = self.info_nce_loss(m_z, p_z, temperature=temperature)
            total = rec_loss + nce_w * nce

        opt.zero_grad(set_to_none=True)
        self.scaler.scale(total).backward()
        self.scaler.step(opt)
        self.scaler.update()

        total_val = float(total.detach().cpu().item())
        rec_val = float(rec_loss.detach().cpu().item())
        nce_val = float(nce.detach().cpu().item())
        field_losses = self.compute_joint_field_losses(self.movie_ae, self.people_ae, m_rec, p_rec, M, P)
        return total_val, rec_val, nce_val, field_losses

    def train(self):
        signal.signal(signal.SIGINT, self._handle_sigint)
        self._prepare_writer()
        self._build_recon_loggers()

        self.joint_model.train()

        opt = torch.optim.AdamW(
            self.joint_model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )

        temperature = float(self.config.get("nce_temp", 0.07))
        nce_w = float(self.config.get("nce_weight", 0.0))
        log_interval = int(self.config.get("log_interval", 20))
        flush_interval = int(self.config.get("flush_interval", 2000))
        save_interval = int(self.config.get("save_interval", 10000))
        max_steps = int(self.config.get("max_steps", max(1, self.total_edges)))

        data_iter, warm = self._warm_first_batch()
        if warm is None:
            return
        pending_batch = warm

        pbar = tqdm(total=max_steps, desc="JointAE training")
        global_step = 0
        last_log_t = time.perf_counter()
        since_last_log = 0

        while global_step < max_steps:
            t_fetch0 = time.perf_counter()
            if pending_batch is not None:
                M, P, eids = pending_batch
                pending_batch = None
                data_time = time.perf_counter() - t_fetch0
            else:
                try:
                    M, P, eids = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.loader)
                    M, P, eids = next(data_iter)
                data_time = time.perf_counter() - t_fetch0

            M = [m.to(self.device, non_blocking=True) for m in M]
            P = [p.to(self.device, non_blocking=True) for p in P]
            eids = eids.to(self.device)

            t_step0 = time.perf_counter()
            total_val, rec_val, nce_val, field_losses = self._train_one_step(
                M=M,
                P=P,
                eids=eids,
                opt=opt,
                temperature=temperature,
                nce_w=nce_w,
            )
            batch_time = time.perf_counter() - t_step0
            iter_time = data_time + batch_time

            for eid in eids.detach().cpu().tolist():
                self.loss_logger.add(int(eid), 0, global_step, total_val, field_losses)

            if self.writer is not None:
                self.writer.add_scalar("loss/total", total_val, global_step)
                self.writer.add_scalar("loss/reconstruction", rec_val, global_step)
                self.writer.add_scalar("loss/nce", nce_val, global_step)
                self.writer.add_scalar("time/iter_sec", iter_time, global_step)
                for k, v in field_losses.items():
                    self.writer.add_scalar(f"loss/fields/{k}", v, global_step)
                for i, g in enumerate(opt.param_groups):
                    self.writer.add_scalar(f"lr/group_{i}", float(g["lr"]), global_step)

            since_last_log += 1
            if since_last_log % log_interval == 0:
                now = time.perf_counter()
                elapsed = max(1e-9, now - last_log_t)
                ips = (log_interval * int(self.config["batch_size"])) / elapsed
                last_log_t = now
                logging.info(
                    f"step {global_step} loss {total_val:.4f} rec {rec_val:.4f} nce {nce_val:.4f} dt {data_time:.3f}s bt {batch_time:.3f}s ips {self._fmt(ips)}"
                )

            JointReconstructionLogger.on_batch_end(self.joint_recon, global_step)
            RowReconstructionLogger.on_batch_end(self.row_recon_movies, global_step)
            RowReconstructionLogger.on_batch_end(self.row_recon_people, global_step)

            if (global_step + 1) % flush_interval == 0:
                self.loss_logger.flush()
                _bump_sampler_version(self.db_path)

            if (global_step + 1) % save_interval == 0:
                self.movie_ae.save_model()
                self.people_ae.save_model()
                out = Path(self.config["model_dir"])
                out.mkdir(parents=True, exist_ok=True)
                torch.save(self.joint_model.state_dict(), out / "JointAutoencoder.pt")
                logging.info(f"checkpoint saved at step {global_step+1}")

            global_step += 1
            pbar.update(1)

            if self.stop_flag["stop"]:
                break

        self.loss_logger.flush()
        _bump_sampler_version(self.db_path)

        self.movie_ae.save_model()
        self.people_ae.save_model()
        out = Path(self.config["model_dir"])
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.joint_model.state_dict(), out / "JointAutoencoder_final.pt")
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def build_joint_trainer(config: Dict[str, Any], warm: bool, db_path: Path):
    t = JointTrainer(config=config, db_path=db_path, warm=warm)
    t.build()
    return t
