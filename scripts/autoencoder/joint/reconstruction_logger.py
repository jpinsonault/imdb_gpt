from itertools import islice
import sqlite3
import numpy as np
import torch
from prettytable import PrettyTable
from typing import List, Dict

from ..fields import NumericDigitCategoryField

def _norm(x): 
    return np.linalg.norm(x) + 1e-9

def _cos(a, b): 
    return float(np.dot(a, b) / (_norm(a) * _norm(b)))


class JointReconstructionLogger:
    def __init__(
        self,
        movie_ae,
        person_ae,
        db_path,
        interval_steps: int = 200,
        num_samples: int = 4,
        table_width: int = 38,
        max_movie_scan: int = 50000,
        neg_pool: int = 256,
    ):
        self.m_ae = movie_ae
        self.p_ae = person_ae
        self.every = max(1, int(interval_steps))
        self.w = table_width
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        movies = list(islice(movie_ae.row_generator(), max_movie_scan))
        self.pairs = []
        import random
        rng = random.Random(1337)
        candidates = rng.sample(movies, min(num_samples * 3, len(movies)))
        for m in candidates:
            p = self._sample_person(m.get("tconst"))
            if p:
                self.pairs.append((m, p))
            if len(self.pairs) == num_samples:
                break

        neg_movies = rng.sample(movies, min(neg_pool, len(movies)))
        self.neg_people_latents = []
        for m in neg_movies:
            p = self._sample_person(m.get("tconst"))
            if p:
                z = self._encode(self.p_ae, p)
                self.neg_people_latents.append(z)
        if self.neg_people_latents:
            self.neg_people_latents = np.stack(self.neg_people_latents)
        else:
            self.neg_people_latents = np.zeros((1, self.m_ae.latent_dim), dtype=np.float32)

    def _sample_person(self, tconst):
        q = """
        SELECT p.primaryName, p.birthYear, p.deathYear, GROUP_CONCAT(pp.profession, ',')
        FROM people p
        LEFT JOIN people_professions pp ON pp.nconst = p.nconst
        INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND p.birthYear IS NOT NULL
        GROUP BY p.nconst
        HAVING COUNT(pp.profession) > 0
        ORDER BY RANDOM()
        LIMIT 1"""
        r = self.conn.execute(q, (tconst,)).fetchone()
        if not r:
            return None
        return {
            "primaryName":  r[0],
            "birthYear":    r[1],
            "deathYear":    r[2],
            "professions":  r[3].split(',') if r[3] else None,
        }

    @torch.no_grad()
    def _encode(self, ae, row):
        xs = [f.transform(row.get(f.name)) for f in ae.fields]
        xs = [x.unsqueeze(0).to(ae.device) for x in xs]
        z = ae.encoder(xs)
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()[0]
        return z

    def _recon(self, ae, z):
        return ae.reconstruct_row(z)

    def _rank(self, z_m, z_p):
        sims = np.dot(self.neg_people_latents, z_m) / (
            (np.linalg.norm(self.neg_people_latents, axis=1) + 1e-9) * (np.linalg.norm(z_m) + 1e-9)
        )
        sims = np.append(sims, _cos(z_m, z_p))
        return int((-sims).argsort().tolist().index(len(sims) - 1)) + 1

    def _tensor_to_string(self, field, main_tensor, flag_tensor=None):
        try:
            if isinstance(field, NumericDigitCategoryField):
                arr = np.array(main_tensor)
                if arr.ndim == 3:
                    return field.to_string(arr)
                return field.to_string(arr.flatten())
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                return field.to_string(main_tensor, flag_tensor)
            arr = np.array(main_tensor)
            if arr.ndim > 1:
                arr = arr.flatten()
            return field.to_string(arr, flag_tensor)
        except Exception:
            return "[Conversion Error]"

    def _roundtrip_string(self, field, raw_value):
        try:
            t = field.transform(raw_value)
            if isinstance(t, (list, tuple)):
                t = t[0]
            return self._tensor_to_string(field, t)
        except Exception:
            return "[RT Error]"

    def on_batch_end(self, global_step: int):
        if self.every <= 0:
            return
        if (global_step + 1) % self.every != 0:
            return

        for m_row, p_row in self.pairs:
            z_m = self._encode(self.m_ae, m_row)
            z_p = self._encode(self.p_ae, p_row)

            cos = _cos(z_m, z_p)
            l2 = float(np.linalg.norm(z_m - z_p))
            ang = float(np.degrees(np.arccos(max(min(cos, 1.0), -1.0))))
            rank = self._rank(z_m, z_p)

            m_rec = self._recon(self.m_ae, z_m)
            p_rec = self._recon(self.p_ae, z_p)

            tm = PrettyTable(["Movie field", "orig", "round", "recon"])
            for f in self.m_ae.fields:
                o = m_row.get(f.name, "")
                rt = self._roundtrip_string(f, o)
                r = m_rec.get(f.name, "")
                tm.add_row([f.name, str(o)[: self.w], rt[: self.w], str(r)[: self.w]])

            tp = PrettyTable(["Person field", "orig", "round", "recon"])
            for f in self.p_ae.fields:
                o = p_row.get(f.name, "")
                rt = self._roundtrip_string(f, o)
                r = p_rec.get(f.name, "")
                tp.add_row([f.name, str(o)[: self.w], rt[: self.w], str(r)[: self.w]])

            bar_len = 20
            filled = max(0, min(bar_len, int(round(cos * bar_len))))
            bar = "#" * filled + "-" * (bar_len - filled)
            print(
                f"\n--- joint recon @ step {global_step+1} ---\n"
                f"cos={cos:.3f}  angle={ang:5.1f}°  l2={l2:.3f}  rank@neg={rank}\n"
                f"[{bar}]\n"
                f"{tm}\n{tp}"
            )
