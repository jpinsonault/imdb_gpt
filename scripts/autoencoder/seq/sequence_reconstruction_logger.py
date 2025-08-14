from itertools import islice
import numpy as np
import torch
from prettytable import PrettyTable
from ..fields import NumericDigitCategoryField

def _to_str(field, arr):
    a = np.asarray(arr)
    if isinstance(field, NumericDigitCategoryField):
        return field.to_string(a)
    if hasattr(field, "tokenizer") and field.tokenizer is not None:
        return field.to_string(a)
    if a.ndim > 1:
        a = a.flatten()
    return field.to_string(a)

class SequenceReconstructionLogger:
    def __init__(
        self,
        seq_model,
        interval_steps=200,
        num_samples=3,
        timesteps_to_show=None,
        table_width=44,
        max_movie_scan=2000,
    ):
        self.m = seq_model
        self.every = max(1, int(interval_steps))
        self.k = max(1, int(num_samples))
        self.ts = timesteps_to_show
        self.w = table_width
        self.active_idx = getattr(self.m, "active_idx", list(range(len(self.m.people_ae.fields))))
        self.samples = []
        gen = self.m._row_generator()
        for m_row, ppl in islice(gen, max_movie_scan):
            self.samples.append((m_row, ppl))
            if len(self.samples) == self.k:
                break

    @torch.no_grad()
    def _predict_sequence(self, m_row):
        xs = [f.transform(m_row.get(f.name)).unsqueeze(0).to(self.m.device) for f in self.m.movie_ae.fields]
        preds = self.m.forward(xs)
        return [p[0].detach().cpu().numpy() for p in preds]

    def _orig_for_timestep(self, ppl, t):
        fields = [self.m.people_ae.fields[i] for i in self.active_idx]
        row = ppl[t]
        out = []
        for f in fields:
            tgt = f.transform_target(row.get(f.name)).numpy()
            out.append(_to_str(f, tgt))
        return out

    def _recon_for_timestep(self, preds, t):
        fields = [self.m.people_ae.fields[i] for i in self.active_idx]
        out = []
        for f, p in zip(fields, preds):
            out.append(_to_str(f, p[t]))
        return out

    def _movie_header(self, m_row):
        title = str(m_row.get("primaryTitle", "") or "")
        year = str(m_row.get("startYear", "") or "")
        g = m_row.get("genres", [])
        if isinstance(g, list):
            genres = ", ".join(g[:3])
        else:
            genres = str(g or "")
        head = title
        if year:
            head += f" ({year})"
        if genres:
            head += f" • {genres}"
        return head

    def on_batch_end(self, step: int):
        if not self.samples:
            return
        if (step + 1) % self.every != 0:
            return

        for m_row, ppl in self.samples:
            preds = self._predict_sequence(m_row)
            seq_len = preds[0].shape[0]
            show_T = seq_len if self.ts is None else min(self.ts, seq_len)

            print("\n=== sequence reconstruction ===")
            print(self._movie_header(m_row))

            fields = [self.m.people_ae.fields[i] for i in self.active_idx]
            for t in range(show_T):
                tbl = PrettyTable(["field", "original", "reconstructed"])
                tbl.align = "l"
                tbl.max_width["original"] = self.w
                tbl.max_width["reconstructed"] = self.w
                orig_vals = self._orig_for_timestep(ppl, t)
                recon_vals = self._recon_for_timestep(preds, t)
                for f, o, r in zip(fields, orig_vals, recon_vals):
                    tbl.add_row([f.name, str(o), str(r)])
                print(f"\n[timestep {t}]")
                print(tbl)
