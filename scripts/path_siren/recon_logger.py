from typing import List, Optional

import numpy as np
import torch
from prettytable import PrettyTable
from tqdm.auto import tqdm


class PathSirenReconstructionLogger:
    def __init__(
        self,
        movie_ae,
        people_ae,
        predictor,
        interval_steps: int = 200,
        num_samples: int = 2,
        table_width: int = 60,
        writer=None,
        n_slots_show: int = 1,
    ):
        self.m_ae = movie_ae
        self.p_ae = people_ae
        self.predictor = predictor
        self.every = max(1, int(interval_steps))
        self.num_samples = max(1, int(num_samples))
        self.w = int(table_width)
        self.writer = writer
        self.n_slots_show = int(max(1, n_slots_show))

    def _to_str(self, field, arr):
        a = np.array(arr)
        if a.ndim > 1 and hasattr(field, "tokenizer") and field.tokenizer is not None:
            return field.to_string(a)
        if a.ndim >= 2 and hasattr(field, "base"):
            return field.to_string(a)
        flat = a.flatten() if a.ndim > 1 else a
        return field.to_string(flat)

    @torch.no_grad()
    def _decode_movie_latent(self, z_title_batch: torch.Tensor):
        outs = []
        for dec in self.m_ae.decoder.decs:
            y = dec(z_title_batch)
            outs.append(y)
        return [o.detach().cpu().numpy() for o in outs]

    @torch.no_grad()
    def _decode_people_latents(self, z_slots_hat: torch.Tensor):
        b, n, d = z_slots_hat.shape
        if n == 0:
            return []
        flat = z_slots_hat.reshape(b * n, d)
        outs = []
        for dec in self.p_ae.decoder.decs:
            y = dec(flat)
            y = y.view(b, n, *y.shape[1:])
            outs.append(y.detach().cpu().numpy())
        return outs

    def _movie_table(
        self,
        My: List[torch.Tensor],
        movie_recon_fields,
        b_idx: int,
        t0: float,
    ):
        tab = PrettyTable(["t", "field", "orig", "recon@t"])
        tab.align = "l"
        tab.max_width["orig"] = self.w
        tab.max_width["recon@t"] = self.w

        for f, y_tgt_b, y_rec_b in zip(
            self.m_ae.fields,
            [y[b_idx] for y in My],
            [y[b_idx] for y in movie_recon_fields],
        ):
            orig = self._to_str(f, y_tgt_b.detach().cpu().numpy())
            recs = self._to_str(f, y_rec_b)
            tab.add_row(
                [
                    f"{float(t0):.3f}",
                    f.name,
                    orig[: self.w],
                    recs[: self.w],
                ]
            )
        return tab

    def _people_table(
        self,
        tgts_per_field: Optional[List[torch.Tensor]],
        people_recon_fields,
        b_idx: int,
        t_row,
        valid_people: int,
        total_slots: int,
    ):
        tab = PrettyTable(["slot", "t", "field", "orig", "recon"])
        tab.align = "l"
        tab.max_width["orig"] = self.w
        tab.max_width["recon"] = self.w

        if not people_recon_fields:
            return tab

        n_pred = int(people_recon_fields[0].shape[1])
        show = max(0, min(self.n_slots_show, n_pred, max(0, total_slots - 1)))

        for i in range(show):
            t_val = float(t_row[i]) if i < len(t_row) else 0.0
            is_masked = i >= max(0, valid_people)

            for f_idx, f in enumerate(self.p_ae.fields):
                if is_masked:
                    tab.add_row(
                        [
                            str(i + 1),
                            f"{t_val:.3f}",
                            f.name,
                            "",
                            "[masked / no data]",
                        ]
                    )
                    continue

                y_pred = people_recon_fields[f_idx][b_idx, i]
                rec = self._to_str(f, y_pred)

                orig = ""
                if (
                    tgts_per_field is not None
                    and f_idx < len(tgts_per_field)
                    and tgts_per_field[f_idx].dim() >= 2
                    and tgts_per_field[f_idx].size(1) > i + 1
                ):
                    tgt_row = tgts_per_field[f_idx][b_idx, i + 1]
                    orig = self._to_str(f, tgt_row.detach().cpu().numpy())

                tab.add_row(
                    [
                        str(i + 1),
                        f"{t_val:.3f}",
                        f.name,
                        orig[: self.w],
                        rec[: self.w],
                    ]
                )

        return tab

    def _title_string_from_recon(self, movie_recon_fields_b0):
        name = ""
        year = ""
        for f, rec in zip(self.m_ae.fields, movie_recon_fields_b0):
            s = self._to_str(f, rec)
            if f.name == "primaryTitle":
                name = s
            elif f.name == "startYear":
                year = s
        if year:
            return f"{name} ({year})"
        return name

    def on_batch_end(
        self,
        global_step: int,
        Mx: List[torch.Tensor],
        My: List[torch.Tensor],
        Z: torch.Tensor,
        t_grid: torch.Tensor,
        z_seq: torch.Tensor,
        mask: torch.Tensor,
        Yp_tgts: Optional[List[torch.Tensor]] = None,
        Z_lat_tgts: Optional[torch.Tensor] = None,
        Z_spline: Optional[torch.Tensor] = None,
    ):
        if (global_step + 1) % self.every != 0:
            return

        if not My or My[0].size(0) == 0:
            return

        b = int(My[0].size(0))

        with torch.no_grad():
            if z_seq.dim() != 3:
                return

            z_t0 = z_seq[:, 0, :]
            movie_recon_fields = self._decode_movie_latent(z_t0)

            people_recon_fields = None
            t_people = None
            max_valid_people = 0

            if mask is not None and mask.numel() > 0:
                valid_counts = mask.sum(dim=1).to(torch.long).clamp_min(1)
                max_valid_people = int(
                    (valid_counts.max() - 1).clamp_min(0).item()
                )

            total_slots = int(t_grid.size(1)) if t_grid is not None else 0

            if total_slots > 1:
                take = min(self.n_slots_show, total_slots - 1)
                if take > 0:
                    z_people = z_seq[:, 1 : 1 + take, :]
                    t_people = t_grid[:, 1 : 1 + take]
                    people_recon_fields = self._decode_people_latents(z_people)

        if movie_recon_fields is None:
            return

        idxs = np.random.choice(
            b,
            size=min(self.num_samples, b),
            replace=False,
        ).tolist()

        for j in idxs:
            t0 = float(t_grid[j, 0].item()) if t_grid is not None else 0.0

            try:
                title_header = self._title_string_from_recon(
                    [y[j] for y in movie_recon_fields]
                )
            except Exception:
                title_header = ""

            t_movie = self._movie_table(My, movie_recon_fields, j, t0)

            tqdm.write("\n[path-siren title @ t=0]")
            if title_header:
                tqdm.write(f"title: {title_header[: self.w]}")
            tqdm.write(t_movie.get_string())

            if (
                people_recon_fields is not None
                and t_people is not None
                and total_slots > 1
            ):
                valid_people_j = 0
                if mask is not None and mask.size(1) > 0:
                    vc = int(mask[j].sum().item())
                    valid_people_j = max(0, vc - 1)

                t_row = [
                    float(x)
                    for x in t_people[j].detach().cpu().tolist()
                ]
                t_people_tab = self._people_table(
                    tgts_per_field=Yp_tgts,
                    people_recon_fields=people_recon_fields,
                    b_idx=j,
                    t_row=t_row,
                    valid_people=valid_people_j,
                    total_slots=total_slots,
                )

                tqdm.write(
                    f"\n[path-siren people @ fixed Δt=1/{max(1, total_slots - 1)}]"
                )
                tqdm.write(t_people_tab.get_string())

                if self.writer is not None:
                    self.writer.add_text(
                        "recon/path_siren_title",
                        "```\n" + t_movie.get_string() + "\n```",
                        global_step=global_step + 1,
                    )
                    self.writer.add_text(
                        "recon/path_siren_people",
                        "```\n" + t_people_tab.get_string() + "\n```",
                        global_step=global_step + 1,
                    )
            else:
                if self.writer is not None:
                    self.writer.add_text(
                        "recon/path_siren_title",
                        "```\n" + t_movie.get_string() + "\n```",
                        global_step=global_step + 1,
                    )
