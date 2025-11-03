import logging
from typing import List
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from tqdm.auto import tqdm
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
from scripts.slot_composer.recon_logger import SlotComposerReconstructionLogger

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

        self._print_run_table(pin, prefetch_factor, num_workers)

        self.recon = SlotComposerReconstructionLogger(
            movie_ae=self.mov_ae,
            people_ae=self.per_ae,
            interval_steps=self.cfg.slot_recon_interval,
            num_samples=self.cfg.slot_recon_num_samples,
            people_slots_to_show=self.cfg.slot_recon_show_slots,
            table_width=self.cfg.slot_recon_table_width,
        )

    def _print_run_table(self, pin_memory: bool, prefetch_factor, num_workers: int):
        t = PrettyTable()
        t.field_names = ["item", "value"]
        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        params = sum(p.numel() for p in self.model.parameters())
        t.add_row(["device", str(self.device)])
        t.add_row(["device_name", dev_name])
        t.add_row(["latent_dim", self.cfg.latent_dim])
        t.add_row(["slots", self.cfg.slot_people_count])
        t.add_row(["layers", self.cfg.slot_layers])
        t.add_row(["heads", self.cfg.slot_heads])
        t.add_row(["ff_mult", self.cfg.slot_ff_mult])
        t.add_row(["dropout", self.cfg.slot_dropout])
        t.add_row(["batch_size", self.cfg.batch_size])
        t.add_row(["lr", self.cfg.slot_learning_rate])
        t.add_row(["weight_decay", self.cfg.slot_weight_decay])
        t.add_row(["epochs", self.cfg.slot_epochs])
        t.add_row(["align_weight", self.cfg.slot_latent_align_weight])
        t.add_row(["div_weight", self.cfg.slot_diversity_weight])
        t.add_row(["recon_interval", self.cfg.slot_recon_interval])
        t.add_row(["recon_samples", self.cfg.slot_recon_num_samples])
        t.add_row(["recon_show_slots", self.cfg.slot_recon_show_slots])
        t.add_row(["recon_table_width", self.cfg.slot_recon_table_width])
        t.add_row(["workers", num_workers])
        t.add_row(["prefetch_factor", prefetch_factor if prefetch_factor is not None else "None"])
        t.add_row(["pin_memory", pin_memory])
        t.add_row(["model_params", params])
        t.add_row(["model_dir", self.cfg.model_dir])
        t.add_row(["db_path", self.cfg.db_path])
        logging.info("\n%s", t.get_string())

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
            pbar = tqdm(self.loader, unit="batch", dynamic_ncols=True)
            pbar.set_description(f"epoch {epoch+1}/{self.cfg.slot_epochs}")
            for Mx, My, Pxs, Pys, mask in pbar:
                Mx = [x.to(self.device, non_blocking=True) for x in Mx]
                My = [y.to(self.device, non_blocking=True) for y in My]
                Pxs = [px.to(self.device, non_blocking=True) for px in Pxs]
                Pys = [py.to(self.device, non_blocking=True) for py in Pys]
                mask = mask.to(self.device, non_blocking=True)

                with torch.no_grad():
                    z_movie = self.mov_ae.encoder(Mx)

                z_movie_hat, z_slots_hat = self.model(z_movie)

                movie_preds = self.mov_ae.decoder(z_movie_hat)
                loss_movie = aggregate_movie_loss(self.mov_ae.fields, movie_preds, My)

                b, n, d = z_slots_hat.shape
                people_preds_per_field = []
                flat = z_slots_hat.reshape(b * n, d)
                for dec in self.per_ae.decoder.decs:
                    y = dec(flat)
                    y = y.view(b, n, *y.shape[1:])
                    people_preds_per_field.append(y)

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

                self.optim.zero_grad(set_to_none=True)
                total.backward()
                self.optim.step()

                step += 1
                pbar.set_postfix(
                    loss=float(total.detach().cpu().item()),
                    movie=float(loss_movie.detach().cpu().item()),
                    people=float(loss_people.detach().cpu().item()),
                    align=float(align.detach().cpu().item()) if isinstance(align, torch.Tensor) else float(align),
                    div=float(div.detach().cpu().item()) if isinstance(div, torch.Tensor) else float(div),
                )

                self.recon.on_batch_end(
                    global_step=step - 1,
                    Mx=Mx,
                    My=My,
                    Pxs=Pxs,
                    Pys=Pys,
                    mask=mask,
                    z_movie_hat=z_movie_hat,
                    z_slots_hat=z_slots_hat,
                    movie_preds=movie_preds,
                    people_preds_per_field=people_preds_per_field,
                )

                if (step % self.cfg.slot_save_interval) == 0:
                    out = Path(self.cfg.model_dir) / "SlotComposer.pt"
                    torch.save(self.model.state_dict(), out)

            out = Path(self.cfg.model_dir) / f"SlotComposer_epoch{epoch+1}.pt"
            torch.save(self.model.state_dict(), out)

        out = Path(self.cfg.model_dir) / "SlotComposer_final.pt"
        torch.save(self.model.state_dict(), out)
