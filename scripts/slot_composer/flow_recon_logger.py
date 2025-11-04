from typing import List
import numpy as np
import torch
from prettytable import PrettyTable
from tqdm.auto import tqdm

class FlowSlotComposerReconstructionLogger:
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

    def _movie_table(self, My, movie_preds, Mx, b_idx, title):
        tab = PrettyTable(["field", "orig", "round", title])
        tab.align = "l"
        tab.max_width["orig"] = self.w
        tab.max_width["round"] = self.w
        tab.max_width[title] = self.w
        rt = self._roundtrip_movie([x[b_idx] for x in Mx])
        for f, y_tgt, y_rt, y_pred in zip(self.m_ae.fields, [y[b_idx] for y in My], rt, [p[b_idx] for p in movie_preds]):
            orig = self._to_str(f, y_tgt.detach().cpu().numpy())
            rts = self._to_str(f, y_rt)
            rec = self._to_str(f, y_pred.detach().cpu().numpy())
            tab.add_row([f.name, orig[: self.w], rts[: self.w], rec[: self.w]])
        return tab

    def _people_table(self, Pys, people_preds_per_field, Pxs, mask, b_idx, slots_to_show, title):
        tab = PrettyTable(["slot", "field", "orig", "round", title])
        tab.align = "l"
        tab.max_width["orig"] = self.w
        tab.max_width["round"] = self.w
        tab.max_width[title] = self.w
        n = mask.size(1)
        valid = [i for i in range(n) if float(mask[b_idx, i].item()) > 0.0]
        valid = valid[:slots_to_show]
        roundtrips = {}
        for i in valid:
            rt = self._roundtrip_person([px[b_idx, i] for px in Pxs])
            roundtrips[i] = rt
        for i in valid:
            for f_idx, f in enumerate(self.p_ae.fields):
                y_tgt = Pys[f_idx][b_idx, i]
                y_pred = people_preds_per_field[f_idx][b_idx, i]
                orig = self._to_str(f, y_tgt.detach().cpu().numpy())
                rts = self._to_str(f, roundtrips[i][f_idx])
                rec = self._to_str(f, y_pred.detach().cpu().numpy())
                tab.add_row([str(i), f.name, orig[: self.w], rts[: self.w], rec[: self.w]])
        return tab

    @torch.no_grad()
    def _decode_people(self, per_decoder, z_slots_hat):
        b, n, d = z_slots_hat.shape
        outs = []
        flat = z_slots_hat.reshape(b * n, d)
        for dec in per_decoder.decs:
            y = dec(flat)
            y = y.view(b, n, *y.shape[1:])
            outs.append(y)
        return outs

    def on_batch_end(
        self,
        global_step: int,
        Mx: List[torch.Tensor],
        My: List[torch.Tensor],
        Pxs: List[torch.Tensor],
        Pys: List[torch.Tensor],
        mask: torch.Tensor,
        model,
        steps_normal: int,
    ):
        if (global_step + 1) % self.every != 0:
            return
        b = int(My[0].size(0)) if My else 0
        if b == 0:
            return
        idxs = np.random.choice(b, size=min(self.num_samples, b), replace=False).tolist()

        with torch.no_grad():
            z_movie = self.m_ae.encoder([x.to(self.m_ae.device) for x in Mx])

        z_m_n, z_s_n = model(z_movie, steps=steps_normal, return_all=False)
        z_m_d, z_s_d = model(z_movie, steps=max(1, steps_normal * 2), return_all=False)

        movie_preds_n = self.m_ae.decoder(z_m_n)
        movie_preds_d = self.m_ae.decoder(z_m_d)

        people_preds_n = self._decode_people(self.p_ae.decoder, z_s_n)
        people_preds_d = self._decode_people(self.p_ae.decoder, z_s_d)

        for j in idxs:
            t_movie_n = self._movie_table(My, movie_preds_n, Mx, j, "recon@normal")
            t_movie_d = self._movie_table(My, movie_preds_d, Mx, j, "recon@double")
            tqdm.write("\n[flow slot-composer movie @ normal]")
            tqdm.write(t_movie_n.get_string())
            tqdm.write("\n[flow slot-composer movie @ double]")
            tqdm.write(t_movie_d.get_string())

            t_people_n = self._people_table(Pys, people_preds_n, Pxs, mask, j, self.show_slots, "recon@normal")
            t_people_d = self._people_table(Pys, people_preds_d, Pxs, mask, j, self.show_slots, "recon@double")
            tqdm.write("\n[flow slot-composer people @ normal]")
            tqdm.write(t_people_n.get_string())
            tqdm.write("\n[flow slot-composer people @ double]")
            tqdm.write(t_people_d.get_string())
