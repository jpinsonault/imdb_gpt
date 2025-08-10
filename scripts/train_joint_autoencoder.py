from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from config import project_config
from scripts.autoencoder.edge_loss_logger import EdgeLossLogger
from scripts.autoencoder.imdb_row_autoencoders import TitlesAutoencoder, PeopleAutoencoder
from scripts.autoencoder.joint_edge_sampler import make_edge_sampler

logging.basicConfig(level=logging.INFO)


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
) -> Tuple[JointAutoencoder, DataLoader, EdgeLossLogger, TitlesAutoencoder, PeopleAutoencoder]:

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
    loader = DataLoader(ds, batch_size=config["batch_size"], collate_fn=_collate_edge)

    loss_logger = EdgeLossLogger(str(db_path))

    joint = JointAutoencoder(movie_ae, people_ae).to(device)
    return joint, loader, loss_logger, movie_ae, people_ae


def main():
    parser = argparse.ArgumentParser(description="Train movie↔people joint embedding autoencoder")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.epochs:
        project_config["epochs"] = args.epochs

    data_dir = Path(project_config["data_dir"])
    db_path = data_dir / "imdb.db"

    joint_model, loader, logger, mov_ae, per_ae = build_joint_trainer(project_config, args.warm, db_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = torch.optim.AdamW(
        joint_model.parameters(),
        lr=float(project_config["learning_rate"]),
        weight_decay=float(project_config["weight_decay"]),
    )
    epochs = int(project_config["epochs"])
    temperature = float(project_config.get("nce_temp", 0.07))

    joint_model.train()
    step = 0
    for epoch in range(epochs):
        for M, P, eids in loader:
            M = [m.to(device) for m in M]
            P = [p.to(device) for p in P]
            eids = eids.to(device)

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

            total_val = float(total.detach().cpu().item())

            for eid in eids.detach().cpu().tolist():
                logger.add(int(eid), epoch, step, total_val, {})

            step += 1

        logger.flush()
        mov_ae.save_model()
        per_ae.save_model()

    logger.close()
    out = Path(project_config["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(joint_model.state_dict(), out / "JointMoviePersonAE_final.pt")


if __name__ == "__main__":
    main()
