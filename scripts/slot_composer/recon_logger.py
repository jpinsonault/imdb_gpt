from typing import List
import numpy as np
import torch
from prettytable import PrettyTable
from tqdm.auto import tqdm

class SlotComposerReconstructionLogger:
    def __init__(
        self,
        movie_ae,
        people_ae,
        interval_steps: int,
        num_samples: int,
        people_slots_to_show: int,
        table_width: int,
    ):
        self.m_ae = movie_ae
        self.p_ae = people_ae
        self.every = max(1, int(interval_steps))
        self.num_samples = max(1, int(num_samples))
        self.show_slots = max(1, int(people_slots_to_show))
        self.w = int(table_width)

    def _to_str(self, field, arr):
        a = np.array(arr)
        if a.ndim > 1 and hasattr(field, "tokenizer") and field.tokenizer is not None:
            return field.to_string(a)
        if a.ndim >= 2 and hasattr(field, "base"):
            return field.to_string(a)
        return field.to_string(a.flatten() if a.ndim > 1 else a)

    @torch.no_grad()
    def _roundtrip_movie(self, Mx_one):
        xs = [x.unsqueeze(0).to(self.m_ae.device) for x in Mx_one]
        z = self.m_ae.encoder(xs)
        outs = self.m_ae.decoder(z)
        return [o.detach().cpu().numpy()[0] for o in outs]

    @torch.no_grad()
    def _roundtrip_person(self, Px_f_one):
        xs = [x.unsqueeze(0).to(self.p_ae.device) for x in Px_f_one]
        z = self.p_ae.encoder(xs)
        outs = self.p_ae.decoder(z)
        return [o.detach().cpu().numpy()[0] for o in outs]

    def _movie_table(self, My, movie_preds, Mx, b_idx):
        tab = PrettyTable(["field", "orig", "round", "recon"])
        tab.align = "l"
        tab.max_width["orig"] = self.w
        tab.max_width["round"] = self.w
        tab.max_width["recon"] = self.w

        rt = self._roundtrip_movie([x[b_idx] for x in Mx])

        for f, y_tgt, y_rt, y_pred in zip(self.m_ae.fields, [y[b_idx] for y in My], rt, [p[b_idx] for p in movie_preds]):
            orig = self._to_str(f, y_tgt.detach().cpu().numpy())
            rts = self._to_str(f, y_rt)
            rec = self._to_str(f, y_pred.detach().cpu().numpy())
            tab.add_row([f.name, orig[: self.w], rts[: self.w], rec[: self.w]])
        return tab

    def _people_table(self, Pys, people_preds_per_field, Pxs, mask, b_idx, slots_to_show):
        tab = PrettyTable(["slot", "field", "orig", "round", "recon"])
        tab.align = "l"
        tab.max_width["orig"] = self.w
        tab.max_width["round"] = self.w
        tab.max_width["recon"] = self.w

        n = mask.size(1)
        valid_slots = [i for i in range(n) if float(mask[b_idx, i].item()) > 0.0]
        valid_slots = valid_slots[:slots_to_show]

        roundtrips = {}
        for i in valid_slots:
            rt = self._roundtrip_person([px[b_idx, i] for px in Pxs])
            roundtrips[i] = rt

        for i in valid_slots:
            for f_idx, f in enumerate(self.p_ae.fields):
                y_tgt = Pys[f_idx][b_idx, i]
                y_pred = people_preds_per_field[f_idx][b_idx, i]
                orig = self._to_str(f, y_tgt.detach().cpu().numpy())
                rts = self._to_str(f, roundtrips[i][f_idx])
                rec = self._to_str(f, y_pred.detach().cpu().numpy())
                tab.add_row([str(i), f.name, orig[: self.w], rts[: self.w], rec[: self.w]])
        return tab

    def on_batch_end(
        self,
        global_step: int,
        Mx: List[torch.Tensor],
        My: List[torch.Tensor],
        Pxs: List[torch.Tensor],
        Pys: List[torch.Tensor],
        mask: torch.Tensor,
        z_movie_hat: torch.Tensor,
        z_slots_hat: torch.Tensor,
        movie_preds: List[torch.Tensor],
        people_preds_per_field: List[torch.Tensor],
    ):
        if (global_step + 1) % self.every != 0:
            return
        b = int(My[0].size(0)) if My else 0
        if b == 0:
            return

        idxs = np.random.choice(b, size=min(self.num_samples, b), replace=False).tolist()
        for j in idxs:
            t_movie = self._movie_table(My, movie_preds, Mx, j)
            tqdm.write("\n[slot-composer movie]")
            tqdm.write(t_movie.get_string())
            t_people = self._people_table(Pys, people_preds_per_field, Pxs, mask, j, self.show_slots)
            tqdm.write("\n[slot-composer people]")
            tqdm.write(t_people.get_string())
