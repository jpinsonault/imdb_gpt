import time
from typing import List, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import ProjectConfig
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.slot_composer.model import SlotComposer
from scripts.slot_composer.dataset import TitlePeopleIterable, collate_batch
from scripts.slot_composer.losses import (
    aggregate_movie_loss,
    aggregate_people_loss,
    latent_alignment_loss,
    diversity_penalty,
)

def _ascii_table(title: str, rows: List[Tuple[str, str]]) -> str:
    left_w = max(len(k) for k, _ in rows + [("key", "")])
    right_w = max(len(v) for _, v in rows + [("", "value")])
    line = "+" + "-" * (left_w + 2) + "+" + "-" * (right_w + 2) + "+"
    out = []
    out.append(line)
    hdr = f"| {'RUN':<{left_w}} | {title:<{right_w}} |"
    out.append(hdr)
    out.append(line)
    for k, v in rows:
        out.append(f"| {k:<{left_w}} | {v:<{right_w}} |")
    out.append(line)
    return "\n".join(out)

class SlotComposerTrainer:
    def __init__(self, config: ProjectConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mov_ae, self.per_ae = _load_frozen_autoencoders(self.cfg)
        self.mov_ae.encoder.to(self.device).eval()
        self.mov_ae.decoder.to(self.device).eval()
        self.per_ae.encoder.to(self.device).eval()
        self.per_ae.decoder.to(self.device).eval()

        self.model = SlotComposer(
            latent_dim=self.cfg.latent_dim,
            num_slots=self.cfg.slot_people_count,
            num_layers=self.cfg.slot_layers,
            num_heads=self.cfg.slot_heads,
            ff_mult=self.cfg.slot_ff_mult,
            dropout=self.cfg.slot_dropout,
        ).to(self.device)

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.slot_learning_rate,
            weight_decay=self.cfg.slot_weight_decay,
        )

        num_workers = int(self.cfg.num_workers)
        pf_raw = int(self.cfg.prefetch_factor)
        prefetch_factor = None if num_workers == 0 else max(1, pf_raw)
        pin = bool(torch.cuda.is_available())

        self.loader = DataLoader(
            TitlePeopleIterable(
                db_path=self.cfg.db_path,
                principals_table=self.cfg.principals_table,
                movie_ae=self.mov_ae,
                people_ae=self.per_ae,
                num_slots=self.cfg.slot_people_count,
            ),
            batch_size=self.cfg.batch_size,
            collate_fn=collate_batch,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=pin,
            prefetch_factor=prefetch_factor,
        )

        Path(self.cfg.model_dir).mkdir(parents=True, exist_ok=True)

        rows = [
            ("device", str(self.device)),
            ("latent_dim", str(self.cfg.latent_dim)),
            ("batch_size", str(self.cfg.batch_size)),
            ("slots", str(self.cfg.slot_people_count)),
            ("layers", str(self.cfg.slot_layers)),
            ("heads", str(self.cfg.slot_heads)),
            ("ff_mult", f"{self.cfg.slot_ff_mult:.2f}"),
            ("dropout", f"{self.cfg.slot_dropout:.2f}"),
            ("lr", f"{self.cfg.slot_learning_rate:.6f}"),
            ("weight_decay", f"{self.cfg.slot_weight_decay:.6f}"),
            ("align_weight", f"{self.cfg.slot_latent_align_weight:.3f}"),
            ("div_weight", f"{self.cfg.slot_diversity_weight:.3f}"),
            ("epochs", str(self.cfg.slot_epochs)),
            ("workers", str(num_workers)),
            ("prefetch", str(prefetch_factor)),
            ("pin_memory", str(pin)),
            ("db_path", self.cfg.db_path),
            ("model_dir", self.cfg.model_dir),
        ]
        print(_ascii_table("slot-composer", rows))

    def _encode_movie(self, Mx: List[torch.Tensor]) -> torch.Tensor:
        X = [x.to(self.device, non_blocking=True) for x in Mx]
        with torch.no_grad():
            z = self.mov_ae.encoder(X)
        return z

    def _encode_people_slot(self, Px_f: List[torch.Tensor], slot_index: int) -> torch.Tensor:
        X = [x[:, slot_index, ...].to(self.device, non_blocking=True) for x in Px_f]
        with torch.no_grad():
            z = self.per_ae.encoder(X)
        return z

    def train(self):
        step = 0
        for epoch in range(self.cfg.slot_epochs):
            it = iter(self.loader)
            pbar = tqdm(total=None, dynamic_ncols=True)
            pbar.set_description(f"epoch {epoch + 1}/{self.cfg.slot_epochs}")
            while True:
                stage = "fetch"
                pbar.set_postfix_str(f"stage={stage}")
                t0 = time.perf_counter()
                try:
                    Mx, My, Pxs, Pys, mask = next(it)
                except StopIteration:
                    break
                fetch_t = time.perf_counter() - t0
                bs = int(Mx[0].size(0)) if isinstance(Mx, list) and len(Mx) > 0 else 0

                stage = "encode"
                pbar.set_postfix_str(f"stage={stage} bs={bs}")
                Mx = [x.to(self.device, non_blocking=True) for x in Mx]
                My = [y.to(self.device, non_blocking=True) for y in My]
                Pxs = [px.to(self.device, non_blocking=True) for px in Pxs]
                Pys = [py.to(self.device, non_blocking=True) for py in Pys]
                mask = mask.to(self.device, non_blocking=True)
                with torch.no_grad():
                    z_movie = self.mov_ae.encoder(Mx)

                stage = "forward"
                pbar.set_postfix_str(f"stage={stage} bs={bs}")
                z_movie_hat, z_slots_hat = self.model(z_movie)

                stage = "decode"
                pbar.set_postfix_str(f"stage={stage} bs={bs}")
                movie_preds = self.mov_ae.decoder(z_movie_hat)
                b, n, d = z_slots_hat.shape
                flat = z_slots_hat.reshape(b * n, d)
                people_preds_per_field = []
                for dec in self.per_ae.decoder.decs:
                    y = dec(flat)
                    y = y.view(b, n, *y.shape[1:])
                    people_preds_per_field.append(y)

                stage = "loss"
                pbar.set_postfix_str(f"stage={stage} bs={bs}")
                loss_movie = aggregate_movie_loss(self.mov_ae.fields, movie_preds, My)
                loss_people = aggregate_people_loss(self.per_ae.fields, people_preds_per_field, Pys, mask)

                align = 0.0
                if self.cfg.slot_latent_align_weight != 0.0:
                    z_true_slots = []
                    for i in range(n):
                        zt = self._encode_people_slot(Pxs, i)
                        z_true_slots.append(zt.unsqueeze(1))
                    z_true = torch.cat(z_true_slots, dim=1)
                    align = latent_alignment_loss(z_slots_hat, z_true, mask) * float(self.cfg.slot_latent_align_weight)

                div = 0.0
                if self.cfg.slot_diversity_weight != 0.0:
                    div = diversity_penalty(z_slots_hat) * float(self.cfg.slot_diversity_weight)

                total = loss_movie + loss_people + align + div

                stage = "step"
                pbar.set_postfix_str(f"stage={stage} bs={bs}")
                self.optim.zero_grad(set_to_none=True)
                total.backward()
                self.optim.step()

                step += 1
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "stage": "done",
                        "batch": step,
                        "bs": bs,
                        "total": f"{float(total.detach().cpu().item()):.4f}",
                        "movie": f"{float(loss_movie.detach().cpu().item()):.4f}",
                        "people": f"{float(loss_people.detach().cpu().item()):.4f}",
                        "align": f"{float(align.detach().cpu().item()) if isinstance(align, torch.Tensor) else float(align):.4f}",
                        "fetch_s": f"{fetch_t:.2f}",
                    }
                )

                if (step % self.cfg.slot_save_interval) == 0:
                    out = Path(self.cfg.model_dir) / "SlotComposer.pt"
                    torch.save(self.model.state_dict(), out)

            pbar.close()

        out = Path(self.cfg.model_dir) / "SlotComposer_final.pt"
        torch.save(self.model.state_dict(), out)
