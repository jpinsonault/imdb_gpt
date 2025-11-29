import datetime
from pathlib import Path
import sqlite3
import random
from itertools import islice
import textwrap
from typing import Dict

import numpy as np
import torch
from prettytable import PrettyTable

from scripts.autoencoder.fields import (
    NumericDigitCategoryField,
    MultiCategoryField,
    SingleCategoryField,
)


class JointReconstructionLogger:
    def __init__(
        self,
        joint_model,
        movie_ae,
        person_ae,
        db_path,
        interval_steps: int = 200,
        num_samples: int = 4,
        table_width: int = 38,
        max_movie_scan: int = 50000,
        neg_pool: int = 256,
    ):
        self.joint = joint_model
        self.m_ae = movie_ae
        self.p_ae = person_ae
        self.every = max(1, int(interval_steps))
        self.w = int(table_width)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        movies = list(islice(self.m_ae.row_generator(), max_movie_scan))
        self.pairs = []
        rng = random.Random(1337)

        candidates = rng.sample(movies, min(num_samples * 3, len(movies)))
        for m in candidates:
            p = self._sample_person(m.get("tconst"))
            if p:
                self.pairs.append((m, p))
            if len(self.pairs) >= num_samples:
                break

        neg_movies = rng.sample(movies, min(neg_pool, len(movies)))
        self.neg_people_latents = []
        for m in neg_movies:
            p = self._sample_person(m.get("tconst"))
            if p:
                z = self._encode_person(p)
                self.neg_people_latents.append(z)

        if self.neg_people_latents:
            self.neg_people_latents = np.stack(self.neg_people_latents)
        else:
            d = int(getattr(self.joint, "latent_dim", 64))
            self.neg_people_latents = np.zeros((1, d), dtype=np.float32)

    def _sample_person(self, tconst: str):
        q = """
        SELECT p.primaryName, p.birthYear, p.deathYear, GROUP_CONCAT(pp.profession, ',')
        FROM people p
        LEFT JOIN people_professions pp ON pp.nconst = p.nconst
        INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
        GROUP BY p.nconst
        HAVING COUNT(pp.profession) > 0
        ORDER BY RANDOM()
        LIMIT 1
        """
        r = self.conn.execute(q, (tconst,)).fetchone()
        if not r:
            return None
        return {
            "primaryName": r[0],
            "birthYear": r[1],
            "deathYear": r[2],
            "professions": r[3].split(",") if r[3] else None,
        }

    @torch.no_grad()
    def _encode_movie(self, row):
        xs = [f.transform(row.get(f.name)) for f in self.m_ae.fields]
        xs = [x.unsqueeze(0).to(self.m_ae.device) for x in xs]
        z = self.m_ae.encoder(xs)
        return z[0].detach().cpu().numpy()

    @torch.no_grad()
    def _encode_person(self, row):
        xs = [f.transform(row.get(f.name)) for f in self.p_ae.fields]
        xs = [x.unsqueeze(0).to(self.p_ae.device) for x in xs]
        z = self.p_ae.encoder(xs)
        return z[0].detach().cpu().numpy()

    @torch.no_grad()
    def _recon_movie(self, z):
        z_t = torch.tensor(z, dtype=torch.float32, device=self.m_ae.device).unsqueeze(0)
        outs = self.joint.mov_dec(z_t)
        #outs = [o[0].detach().cpu().numpy() for o in outs] 
        # Keep as tensor to pass to render_prediction
        rec = {}
        for f, tensor in zip(self.m_ae.fields, outs):
            # Pass the 0-th element (batch size 1)
            rec[f.name] = f.render_prediction(tensor[0])
        return rec

    @torch.no_grad()
    def _recon_person(self, z):
        z_t = torch.tensor(z, dtype=torch.float32, device=self.p_ae.device).unsqueeze(0)
        outs = self.joint.per_dec(z_t)
        rec = {}
        for f, tensor in zip(self.p_ae.fields, outs):
            rec[f.name] = f.render_prediction(tensor[0])
        return rec

    def _roundtrip_string(self, field, raw_value):
        try:
            # Transform raw value -> Tensor
            t = field.transform(raw_value)
            
            # For roundtrip, we treat the transformed tensor as "Ground Truth" 
            # because transform() returns the dataset representation (indices/one-hot),
            # NOT model logits.
            return field.render_ground_truth(t)
        except Exception:
            return "[RT Error]"

    def _rank(self, z_m, z_p):
        sims_neg = np.dot(self.neg_people_latents, z_m) / (
            (np.linalg.norm(self.neg_people_latents, axis=1) + 1e-9)
            * (np.linalg.norm(z_m) + 1e-9)
        )
        pos = float(np.dot(z_m, z_p) / ((np.linalg.norm(z_m) + 1e-9) * (np.linalg.norm(z_p) + 1e-9)))
        sims = np.append(sims_neg, pos)
        return int((-sims).argsort().tolist().index(len(sims) - 1)) + 1

    def _wrap_cell(self, value):
        s = "" if value is None else str(value)
        if not s:
            return ""
        lines = textwrap.wrap(s, self.w)
        if not lines:
            return ""
        return "\n".join(lines)

    def on_batch_end(self, global_step: int):
        if self.every <= 0:
            return
        if (global_step + 1) % self.every != 0:
            return
        if not self.pairs:
            return

        for m_row, p_row in self.pairs:
            z_m = self._encode_movie(m_row)
            z_p = self._encode_person(p_row)

            cos = float(np.dot(z_m, z_p) / ((np.linalg.norm(z_m) + 1e-9) * (np.linalg.norm(z_p) + 1e-9)))
            l2 = float(np.linalg.norm(z_m - z_p))
            ang = float(np.degrees(np.arccos(max(min(cos, 1.0), -1.0))))
            rank = self._rank(z_m, z_p)

            m_rec = self._recon_movie(z_m)
            p_rec = self._recon_person(z_p)

            tm = PrettyTable(["Movie field", "orig", "round", "recon"])
            for f in self.m_ae.fields:
                o = m_row.get(f.name, "")
                rt = self._roundtrip_string(f, o)
                r = m_rec.get(f.name, "")
                tm.add_row(
                    [
                        self._wrap_cell(f.name),
                        self._wrap_cell(o),
                        self._wrap_cell(rt),
                        self._wrap_cell(r),
                    ]
                )

            tp = PrettyTable(["Person field", "orig", "round", "recon"])
            for f in self.p_ae.fields:
                o = p_row.get(f.name, "")
                rt = self._roundtrip_string(f, o)
                r = p_rec.get(f.name, "")
                tp.add_row(
                    [
                        self._wrap_cell(f.name),
                        self._wrap_cell(o),
                        self._wrap_cell(rt),
                        self._wrap_cell(r),
                    ]
                )

            bar_len = 20
            filled = max(0, min(bar_len, int(round(cos * bar_len))))
            bar = "#" * filled + "-" * (bar_len - filled)
            print(
                f"\n--- joint recon @ step {global_step+1} ---\n"
                f"cos={cos:.3f}  angle={ang:5.1f}°  l2={l2:.3f}  rank@neg={rank}\n"
                f"[{bar}]\n"
                f"{tm}\n{tp}"
            )


class TensorBoardPerBatchLogger:
    """Writes basic scalars to a unique TensorBoard run directory."""
    def __init__(self, log_dir: str, run_prefix: str = "run"):
        from torch.utils.tensorboard import SummaryWriter
        root = Path(log_dir)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        base = f"{run_prefix}_{ts}"
        run_dir = root / base
        if run_dir.exists():
            i = 1
            while True:
                cand = root / f"{base}_{i}"
                if not cand.exists():
                    run_dir = cand
                    break
                i += 1
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(run_dir))
        self.run_dir = str(run_dir)

    def log_scalars(self, step: int, scalars: Dict[str, float]):
        for k, v in scalars.items():
            self.writer.add_scalar(k, float(v), step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()

class ModelSaveHook:
    """Calls row_autoencoder.save_model() on epoch end."""
    def __init__(self, row_autoencoder, output_dir: str):
        self.row_autoencoder = row_autoencoder
        self.output_dir = output_dir

    def on_epoch_end(self, epoch: int):
        self.row_autoencoder.save_model()
        print(f"Model saved to {self.output_dir} at the end of epoch {epoch + 1}")