from itertools import islice
import datetime
import logging
from pathlib import Path
import random
import sqlite3
import os
import numpy as np
from scripts.sql_filters import people_from_join, people_group_by, people_having, people_where_clause
import torch
from prettytable import PrettyTable
from typing import Any, List, Dict, Optional
from ..fields import NumericDigitCategoryField, TextField

def _sample_random_person(conn, tconst):
    q = f"""
        SELECT p.primaryName,
               p.birthYear,
               p.deathYear,
               GROUP_CONCAT(pp.profession, ',')
        {people_from_join()} INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND {people_where_clause()}
        {people_group_by()}
        {people_having()}
        ORDER BY RANDOM()
        LIMIT 1
    """
    r = conn.execute(q, (tconst,)).fetchone()
    if not r:
        return None
    return {
        "primaryName": r[0],
        "birthYear": r[1],
        "deathYear": r[2],
        "professions": r[3].split(',') if r[3] else None,
    }


def _norm(x): return np.linalg.norm(x) + 1e-9
def _cos(a, b): return float(np.dot(a, b) / (_norm(a) * _norm(b)))

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
        import sqlite3
        import random
        from itertools import islice

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
        outs = [o[0].detach().cpu().numpy() for o in outs]
        rec = {}
        for f, arr in zip(self.m_ae.fields, outs):
            rec[f.name] = self._tensor_to_string(f, arr)
        return rec

    @torch.no_grad()
    def _recon_person(self, z):
        z_t = torch.tensor(z, dtype=torch.float32, device=self.p_ae.device).unsqueeze(0)
        outs = self.joint.per_dec(z_t)
        outs = [o[0].detach().cpu().numpy() for o in outs]
        rec = {}
        for f, arr in zip(self.p_ae.fields, outs):
            rec[f.name] = self._tensor_to_string(f, arr)
        return rec

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
            return self._tensor_to_string(field, t)
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

class RowReconstructionLogger:
    """Prints reconstructions for a single RowAutoencoder at intervals."""
    def __init__(
        self,
        interval_steps: int,
        row_autoencoder,
        db_path: str,
        num_samples: int = 5,
        table_width: int = 40,
    ):
        self.interval_steps = max(1, int(interval_steps))
        self.row_autoencoder = row_autoencoder
        self.db_path = db_path
        self.num_samples = num_samples
        self.w = table_width

        if not self.row_autoencoder.stats_accumulated:
            self.row_autoencoder.accumulate_stats()

        all_rows = list(self.row_autoencoder.row_generator())
        n = min(num_samples, len(all_rows))
        self.samples = random.sample(all_rows, n) if n > 0 else []

    @torch.no_grad()
    def _encode(self, row):
        xs = [f.transform(row.get(f.name)) for f in self.row_autoencoder.fields]
        xs = [x.unsqueeze(0).to(self.row_autoencoder.device) for x in xs]
        z = self.row_autoencoder.encoder(xs)
        return z.detach()

    @torch.no_grad()
    def _decode(self, z_tensor):
        outs = self.row_autoencoder.decoder(z_tensor)
        return [o.detach().cpu().numpy()[0] for o in outs]

    def _tensor_to_string(self, field, main_tensor: np.ndarray) -> str:
        try:
            if isinstance(field, NumericDigitCategoryField):
                arr = np.array(main_tensor)
                return field.to_string(arr if arr.ndim == 2 else arr.flatten())
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                return field.to_string(np.array(main_tensor))
            arr = np.array(main_tensor)
            if arr.ndim > 1:
                arr = arr.flatten()
            return field.to_string(arr)
        except Exception:
            return "[Conversion Error]"

    def on_batch_end(self, global_step: int):
        if not self.samples:
            return
        if (global_step + 1) % self.interval_steps != 0:
            return

        print(f"\nBatch {global_step + 1}: Reconstruction Demo")
        for i, row_dict in enumerate(self.samples):
            print(f"\nSample {i + 1}:")
            table = PrettyTable()
            table.field_names = ["Field", "Original Value", "Reconstructed"]
            table.align = "l"
            for col in ["Original Value", "Reconstructed"]:
                table.max_width[col] = self.w

            z = self._encode(row_dict)
            preds = self._decode(z)

            for field, pred in zip(self.row_autoencoder.fields, preds):
                field_name = field.name
                original_raw = row_dict.get(field_name, "N/A")
                original_str = ", ".join(map(str, original_raw)) if isinstance(original_raw, list) else str(original_raw)
                reconstructed_str = self._tensor_to_string(field, pred)
                table.add_row([field_name, original_str, reconstructed_str])

            print(table)


class SequenceReconstructionLogger:
    def __init__(self, movie_ae, people_ae, predictor, db_path: str, seq_len: int, interval_steps: int = 200, num_samples: int = 2, table_width: int = 38):
        import sqlite3, random
        from itertools import islice
        self.m_ae = movie_ae
        self.p_ae = people_ae
        self.pred = predictor
        self.seq_len = seq_len
        self.every = max(1, int(interval_steps))
        self.w = table_width
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        movies = list(islice(self.m_ae.row_generator(), 50000))
        rng = random.Random()
        rng.shuffle(movies)
        self.samples = []
        for m in movies:
            ppl = self._people_for(m.get("tconst"))
            if ppl:
                if len(ppl) < self.seq_len:
                    ppl = ppl + [ppl[-1]] * (self.seq_len - len(ppl))
                else:
                    ppl = ppl[: self.seq_len]
                self.samples.append((m, ppl))
            if len(self.samples) == num_samples:
                break

    def _people_for(self, tconst: str):
        q = f"""
        SELECT p.primaryName, p.birthYear, p.deathYear, GROUP_CONCAT(pp.profession, ',')
        {people_from_join()} INNER JOIN principals pr ON pr.nconst = p.nconst
        WHERE pr.tconst = ? AND {people_where_clause()}
        {people_group_by()}
        {people_having()}
        ORDER BY pr.ordering
        LIMIT ?
        """
        r = self.conn.execute(q, (tconst, self.seq_len)).fetchall()
        out = []
        for row in r:
            out.append({
                "primaryName": row[0],
                "birthYear": row[1],
                "deathYear": row[2],
                "professions": row[3].split(",") if row[3] else None,
            })
        return out


    def _tensor_to_string(self, field, arr):
        import numpy as np
        try:
            a = np.array(arr)
            from ..fields import NumericDigitCategoryField
            if isinstance(field, NumericDigitCategoryField):
                return field.to_string(a if a.ndim >= 2 else a.flatten())
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                return field.to_string(a)
            return field.to_string(a.flatten() if a.ndim > 1 else a)
        except Exception:
            return "[conv_err]"

    @torch.no_grad()
    def _predict_seq(self, movie_row):
        device = self.m_ae.device
        xs = [f.transform(movie_row.get(f.name)) for f in self.m_ae.fields]
        xs = [x.unsqueeze(0).to(device) for x in xs]
        if self.pred is None:
            raise RuntimeError("SequenceReconstructionLogger requires a non-None predictor")
        outs = self.pred(xs)
        return [o.detach().cpu().numpy()[0] for o in outs]

    @torch.no_grad()
    def _roundtrip_person(self, person_row):
        xs = [f.transform(person_row.get(f.name)) for f in self.p_ae.fields]
        xs = [x.unsqueeze(0).to(self.p_ae.device) for x in xs]
        z = self.p_ae.encoder(xs)
        outs = self.p_ae.decoder(z)
        return [o.detach().cpu().numpy()[0] for o in outs]

    def on_batch_end(self, global_step: int):
        if not self.samples:
            return
        if (global_step + 1) % self.every != 0:
            return
        from prettytable import PrettyTable
        for m_row, ppl in self.samples:
            preds = self._predict_seq(m_row)
            print("\nSequence reconstruction")
            print(f"movie: {m_row.get('primaryTitle', '')}")
            for t in range(self.seq_len):
                tab = PrettyTable(["field", "orig", "round", "recon"])
                tab.align = "l"
                rt = self._roundtrip_person(ppl[t])
                for i, (f, pred) in enumerate(zip(self.p_ae.fields, preds)):
                    y = pred[t]
                    orig_raw = ppl[t].get(f.name, "")
                    if isinstance(orig_raw, list):
                        orig_str = ", ".join(map(str, orig_raw))
                    else:
                        orig_str = str(orig_raw)
                    round_str = self._tensor_to_string(f, rt[i])
                    recon_str = self._tensor_to_string(f, y)
                    tab.add_row([f.name, orig_str[: self.w], round_str[: self.w], recon_str[: self.w]])
                print(f"\ntimestep {t+1}/{self.seq_len}\n{tab}")
