from __future__ import annotations
from typing import List, Tuple, Optional
import random

import torch
from prettytable import PrettyTable

def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu")

def _fmt_field(field, arr_like) -> str:
    try:
        import numpy as _np
        a = arr_like
        if isinstance(a, torch.Tensor):
            a = _to_cpu(a).numpy()
        else:
            a = _np.array(a)
        return field.to_string(a)
    except Exception:
        if isinstance(arr_like, torch.Tensor):
            return repr(_to_cpu(arr_like).view(-1)[:8].tolist())
        try:
            import numpy as _np
            return repr(_np.array(arr_like).reshape(-1)[:8].tolist())
        except Exception:
            return "[unprintable]"

def _summarize_inputs_table(title: str, fields, xs_bt: List[torch.Tensor], b_idx: int, max_width: int) -> str:
    tab = PrettyTable(["inputs (" + title + ")", "value"])
    tab.align = "l"
    tab.max_width["value"] = max_width
    for f, x in zip(fields, xs_bt):
        xi = x[b_idx]
        if xi.ndim >= 2:
            x0 = xi[0]
            val = _fmt_field(f, x0)
        else:
            val = _fmt_field(f, xi)
        tab.add_row([f.name, str(val)[:max_width]])
    return str(tab)

def _pick_indices(batch_size: int, k: int, rng: random.Random) -> List[int]:
    k = max(1, min(batch_size, k))
    idxs = list(range(batch_size))
    rng.shuffle(idxs)
    return idxs[:k]

class ManyToManyReconstructionLogger:
    def __init__(
        self,
        movie_ae,
        people_ae,
        model,
        seq_len_titles: int,
        seq_len_people: int,
        interval_steps: int = 200,
        num_samples: int = 2,
        table_width: int = 44,
        seed: int = 1234,
    ):
        self.m_ae = movie_ae
        self.p_ae = people_ae
        self.model = model
        self.lt = int(seq_len_titles)
        self.lp = int(seq_len_people)
        self.every = max(1, int(interval_steps))
        self.num_samples = max(1, int(num_samples))
        self.w = int(table_width)
        self._rng = random.Random(int(seed))
        self._chosen_m2p: Optional[List[int]] = None
        self._chosen_p2t: Optional[List[int]] = None

    @torch.no_grad()
    def _shortcut_people_seq_from_movie_inputs(self, xm_bt_list: List[torch.Tensor]) -> torch.Tensor:
        device = self.m_ae.device
        xs = [ (x[:, 0] if x.dim() >= 2 else x).to(device) for x in xm_bt_list ]
        z = self.m_ae.encoder(xs)
        z_big = self.model.pool_movies(z)
        z_tiled = z_big.unsqueeze(1).expand(z_big.size(0), self.lp, z_big.size(-1))
        flat = z_tiled.reshape(-1, z_tiled.size(-1))
        outs = self.p_ae.decoder(flat)
        outs_bt = [o.view(z_tiled.size(0), self.lp, *o.shape[1:]) for o in outs]
        return outs_bt

    @torch.no_grad()
    def _shortcut_titles_seq_from_person_inputs(self, xp_bt_list: List[torch.Tensor]) -> torch.Tensor:
        device = self.p_ae.device
        xs = [ (x[:, 0] if x.dim() >= 2 else x).to(device) for x in xp_bt_list ]
        z = self.p_ae.encoder(xs)
        z_big = self.model.pool_people(z)
        z_tiled = z_big.unsqueeze(1).expand(z_big.size(0), self.lt, z_big.size(-1))
        flat = z_tiled.reshape(-1, z_tiled.size(-1))
        outs = self.m_ae.decoder(flat)
        outs_bt = [o.view(z_tiled.size(0), self.lt, *o.shape[1:]) for o in outs]
        return outs_bt

    def _print_seq_tables(
        self,
        header: str,
        fields,
        targets_bt_list: List[torch.Tensor],
        recon_seq_bt_list: List[torch.Tensor],
        recon_short_bt_list: List[torch.Tensor],
        b_idx: int,
        T: int,
        mask_bt: Optional[List[torch.Tensor]] = None,
    ):
        print("\n" + header)
        for t in range(T):
            tab = PrettyTable(["field", "target", "recon_seq", "recon_short"])
            tab.align = "l"
            for col in ["target", "recon_seq", "recon_short"]:
                tab.max_width[col] = self.w
            masked_flag = False
            if mask_bt is not None and len(mask_bt) > 0:
                m0 = mask_bt[0]
                try:
                    masked_flag = (m0[b_idx, t].max() == 0) if m0[b_idx, t].ndim > 0 else (m0[b_idx, t] == 0)
                except Exception:
                    pass
            for f, tgt_bt, rseq_bt, rsho_bt in zip(fields, targets_bt_list, recon_seq_bt_list, recon_short_bt_list):
                tgt = tgt_bt[b_idx, t]
                rseq = rseq_bt[b_idx, t]
                rsho = rsho_bt[b_idx, t]
                tab.add_row([
                    f.name,
                    _fmt_field(f, tgt)[: self.w],
                    _fmt_field(f, rseq)[: self.w],
                    _fmt_field(f, rsho)[: self.w],
                ])
            if masked_flag:
                print(f"\ntimestep {t+1}/{T} (pad)")
            else:
                print(f"\ntimestep {t+1}/{T}")
            print(tab)

    def _maybe_choose_rows(self, B: int):
        if self._chosen_m2p is None:
            self._chosen_m2p = _pick_indices(B, self.num_samples, self._rng)
        if self._chosen_p2t is None:
            self._chosen_p2t = _pick_indices(B, self.num_samples, self._rng)

    @torch.no_grad()
    def on_batch_end(
        self,
        global_step: int,
        batch: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]],
        preds: Tuple[List[torch.Tensor], List[torch.Tensor]],
    ):
        if (global_step + 1) % self.every != 0:
            return

        xm, xp, yt, yp, mt, mp = batch
        preds_titles_seq, preds_people_seq = preds
        B = xm[0].size(0)
        self._maybe_choose_rows(B)

        short_people_bt = self._shortcut_people_seq_from_movie_inputs(xm)
        short_titles_bt = self._shortcut_titles_seq_from_person_inputs(xp)

        for b in self._chosen_m2p:
            print("\n" + "=" * 12 + f"  movie → people  [batch row {b}]  " + "=" * 12)
            print(_summarize_inputs_table("movie", self.m_ae.fields, xm, b_idx=b, max_width=self.w))
            self._print_seq_tables(
                header="people sequence",
                fields=self.p_ae.fields,
                targets_bt_list=yp,
                recon_seq_bt_list=preds_people_seq,
                recon_short_bt_list=short_people_bt,
                b_idx=b,
                T=self.lp,
                mask_bt=mp,
            )

        for b in self._chosen_p2t:
            print("\n" + "=" * 12 + f"  person → titles  [batch row {b}]  " + "=" * 12)
            print(_summarize_inputs_table("person", self.p_ae.fields, xp, b_idx=b, max_width=self.w))
            self._print_seq_tables(
                header="titles sequence",
                fields=self.m_ae.fields,
                targets_bt_list=yt,
                recon_seq_bt_list=preds_titles_seq,
                recon_short_bt_list=short_titles_bt,
                b_idx=b,
                T=self.lt,
                mask_bt=mt,
            )
