from pathlib import Path
import math
import os
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from tqdm import tqdm
import matplotlib.pyplot as plt


class PerformanceSummary:
    def __init__(
        self,
        movie_ae,
        people_ae,
        max_rows: int = 20,
        batch_size: int = 2048,
        table_width: int = 38,
        similarity_out: str | None = None,
        similarity_max_items: int = 50000,
        plot_dir: str | None = "plots",
    ):
        self.m_ae = movie_ae
        self.p_ae = people_ae
        self.max_rows = int(max_rows)
        self.bs = int(batch_size)
        self.w = int(table_width)
        self.sim_out = Path(similarity_out) if similarity_out else None
        self.sim_max = int(similarity_max_items)
        self.plot_dir = Path(plot_dir) if plot_dir else Path("plots")

    def _as_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, (list, tuple)):
            return np.array(x)
        return x if isinstance(x, np.ndarray) else np.array(x)

    def _per_sample_field_loss(self, field, pred, tgt):
        if pred.dim() == 3 and hasattr(field, "tokenizer"):
            B, L, V = pred.shape
            pad_id = int(getattr(field, "pad_token_id", 0) or 0)
            if tgt.dim() == 3:
                tgt = tgt.argmax(dim=-1)
            loss_flat = F.cross_entropy(
                pred.reshape(B * L, V),
                tgt.reshape(B * L),
                ignore_index=pad_id,
                reduction="none",
            ).reshape(B, L)
            mask = (tgt != pad_id).float()
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (loss_flat * mask).sum(dim=1) / denom
        if pred.dim() == 3 and hasattr(field, "base"):
            B, P, V = pred.shape
            loss_flat = F.cross_entropy(
                pred.reshape(B * P, V),
                tgt.reshape(B * P).long(),
                reduction="none",
            ).reshape(B, P)
            return loss_flat.mean(dim=1)
        diff = pred - tgt
        if diff.dim() > 1:
            return diff.pow(2).reshape(diff.size(0), -1).mean(dim=1)
        return diff.pow(2)

    def _tensor_to_string(self, field, main_tensor):
        arr = self._as_numpy(main_tensor)
        try:
            if hasattr(field, "base"):
                a = np.array(arr)
                return field.to_string(a if a.ndim >= 2 else a.flatten())
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                a = np.array(arr)
                return field.to_string(a)
            a = np.array(arr)
            if a.ndim > 1:
                a = a.flatten()
            return field.to_string(a)
        except Exception:
            return "[conv_err]"

    def _roundtrip_string(self, field, raw_value):
        try:
            t = field.transform(raw_value)
            return self._tensor_to_string(field, t)
        except Exception:
            return "[conv_err]"

    def _eval_extremes(self, ae):
        device = ae.device
        fields = ae.fields
        worst = []
        best = []
        xs_cols = [[] for _ in fields]
        ys_cols = [[] for _ in fields]
        rows = []
        values = []

        @torch.no_grad()
        def _flush():
            nonlocal worst, best, values
            if not rows:
                return
            X = [torch.stack(col, dim=0).to(device) for col in xs_cols]
            Y = [torch.stack(col, dim=0).to(device) for col in ys_cols]
            ae.encoder.eval()
            ae.decoder.eval()
            z = ae.encoder(X)
            outs = ae.decoder(z)
            B = X[0].size(0)
            loss_per = torch.zeros(B, device=device)
            for f, pred, tgt in zip(fields, outs, Y):
                loss_per = loss_per + self._per_sample_field_loss(f, pred, tgt)
            losses = self._as_numpy(loss_per)
            preds_np = [self._as_numpy(o) for o in outs]
            for i in range(B):
                rec = {}
                for f, o in zip(fields, preds_np):
                    rec[f.name] = self._tensor_to_string(f, o[i])
                item = (float(losses[i]), rows[i], rec)

                if len(worst) < self.max_rows:
                    worst.append(item)
                else:
                    min_w = min(worst, key=lambda x: x[0])[0]
                    if item[0] > min_w:
                        idx = min(range(len(worst)), key=lambda k: worst[k][0])
                        worst[idx] = item

                if len(best) < self.max_rows:
                    best.append(item)
                else:
                    max_b = max(best, key=lambda x: x[0])[0]
                    if item[0] < max_b:
                        idx = max(range(len(best)), key=lambda k: best[k][0])
                        best[idx] = item

                row = rows[i]
                txt = ""
                yr = None
                if "primaryTitle" in row:
                    txt = str(row.get("primaryTitle", "") or "")
                    yr = row.get("startYear")
                elif "primaryName" in row:
                    txt = str(row.get("primaryName", "") or "")
                    yr = row.get("birthYear")
                try:
                    yr = int(yr) if yr is not None else None
                except Exception:
                    yr = None
                values.append({"text": txt, "year": yr, "loss": float(losses[i])})

            for j in range(len(xs_cols)):
                xs_cols[j].clear()
                ys_cols[j].clear()
            rows.clear()

        gen = list(ae.row_generator())
        for row in tqdm(gen, desc=f"eval {ae.__class__.__name__}", unit="row", dynamic_ncols=True):
            xs = [f.transform(row.get(f.name)) for f in fields]
            ys = [f.transform_target(row.get(f.name)) for f in fields]
            for j, t in enumerate(xs):
                xs_cols[j].append(t)
            for j, t in enumerate(ys):
                ys_cols[j].append(t)
            rows.append(row)
            if len(rows) >= self.bs:
                _flush()
        _flush()

        worst.sort(key=lambda x: x[0], reverse=True)
        best.sort(key=lambda x: x[0])

        return worst, best, values

    def _print_big_table(self, title, fields, pairs):
        print(f"\n{title}")
        headers = [f.name for f in fields]
        tab = PrettyTable(headers)
        tab.align = "l"
        sep = "=" * max(10, self.w)
        for _, row, rec in pairs:
            orig_vals = []
            recon_vals = []
            for f in fields:
                orig = row.get(f.name, "")
                orig_vals.append(str(self._roundtrip_string(f, orig))[: self.w])
                recon_vals.append(str(rec.get(f.name, ""))[: self.w])
            tab.add_row(orig_vals)
            tab.add_row(recon_vals)
            tab.add_row([sep for _ in headers])
        print(tab)

    def _normalize_base(self, s):
        if not s:
            return ""
        t = unicodedata.normalize("NFKD", s)
        t = t.encode("ascii", "ignore").decode("ascii", errors="ignore")
        t = t.strip().lower()
        t = " ".join(t.split())
        return t

    def _normalize_title(self, s):
        t = self._normalize_base(s)
        for art in ("the ", "a ", "an "):
            if t.startswith(art):
                t = t[len(art):]
                break
        return t

    def _normalize_name(self, s):
        return self._normalize_base(s)

    def _entropy_per_char(self, s):
        if not s:
            return 0.0
        counts = Counter(s)
        n = float(len(s))
        p = [c / n for c in counts.values()]
        h = -sum(q * math.log(q + 1e-12) for q in p)
        return h / n

    def _build_uniqueness_metrics(self, values, is_title: bool):
        n = len(values)
        if n == 0:
            return {}
        norm_fn = self._normalize_title if is_title else self._normalize_name
        keys_raw = []
        keys_pair = []
        lens = []
        ents = []
        for v in values:
            s = v["text"] or ""
            k = norm_fn(s)
            y = v.get("year", None)
            if not is_title and isinstance(y, int):
                yk = (y // 10) * 10
            else:
                yk = y
            keys_raw.append(k)
            keys_pair.append((k, yk))
            lens.append(len(s))
            ents.append(self._entropy_per_char(s))
        c_raw = Counter(keys_raw)
        c_pair = Counter(keys_pair)
        idf_denom = math.log(n + 1.0)
        u_raw = np.array([1.0 / max(1, c_raw[k]) for k in keys_raw], dtype=np.float64)
        u_idf = np.array([math.log((n + 1.0) / (c_raw[k] + 1.0)) / idf_denom for k in keys_raw], dtype=np.float64)
        u_pair = np.array([1.0 / max(1, c_pair[kp]) for kp in keys_pair], dtype=np.float64)
        length = np.array(lens, dtype=np.float64)
        entropy_pc = np.array(ents, dtype=np.float64)
        loss = np.array([v["loss"] for v in values], dtype=np.float64)
        return {
            "loss": loss,
            "u_raw": u_raw,
            "u_idf": u_idf,
            "u_pair": u_pair,
            "length": length,
            "entropy_per_char": entropy_pc,
        }

    def _binned_xy(self, x, y, bins: int = 50):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(x) == 0:
            return None, None
        q = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(x, q))
        if len(edges) < 3:
            return None, None
        idx = np.digitize(x, edges[1:-1], right=True)
        bx = []
        by = []
        for b in range(len(edges) - 1):
            mask = idx == b
            if not np.any(mask):
                continue
            bx.append(np.mean(x[mask]))
            by.append(np.mean(y[mask]))
        if not bx:
            return None, None
        return np.array(bx), np.array(by)

    def _scatter_with_bins(self, series_name: str, metric_name: str, x, y):
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure()
        plt.scatter(x, y, s=6, alpha=0.25)
        bx, by = self._binned_xy(x, y, bins=50)
        if bx is not None and by is not None:
            plt.plot(bx, by, linewidth=2.0)
        plt.xlabel(metric_name)
        plt.ylabel("loss")
        plt.title(f"{series_name}: loss vs {metric_name}")
        out = self.plot_dir / f"{series_name}_{metric_name}.png"
        fig.savefig(str(out), dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _make_plots(self, tag: str, metrics: dict):
        loss = metrics["loss"]
        for key in ("u_raw", "u_idf", "u_pair", "length", "entropy_per_char"):
            x = metrics[key]
            self._scatter_with_bins(tag, key, x, loss)

    def run(self):
        titles_worst, titles_best, titles_vals = self._eval_extremes(self.m_ae)
        people_worst, people_best, people_vals = self._eval_extremes(self.p_ae)

        self._print_big_table(f"worst titles recon (top {self.max_rows})", self.m_ae.fields, titles_worst)
        self._print_big_table(f"best titles recon (top {self.max_rows})", self.m_ae.fields, titles_best)
        self._print_big_table(f"worst people recon (top {self.max_rows})", self.p_ae.fields, people_worst)
        self._print_big_table(f"best people recon (top {self.max_rows})", self.p_ae.fields, people_best)

        t_metrics = self._build_uniqueness_metrics(titles_vals, is_title=True)
        p_metrics = self._build_uniqueness_metrics(people_vals, is_title=False)
        if t_metrics:
            self._make_plots("titles", t_metrics)
        if p_metrics:
            self._make_plots("people", p_metrics)
