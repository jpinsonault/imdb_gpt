import logging
from pathlib import Path
from typing import List, Optional
import re
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from tqdm.auto import tqdm
from datetime import datetime

from config import ProjectConfig
from scripts.autoencoder.ae_loader import _load_frozen_autoencoders
from scripts.slot_composer.dataset import TitlePeopleIterable, collate_batch
from scripts.slot_composer.losses import (
    aggregate_movie_loss,
    latent_alignment_loss,
    diversity_penalty,
)
from scripts.slot_composer.flow_losses import straight_path_loss
from scripts.slot_composer.flow_recon_logger import FlowSlotComposerReconstructionLogger
from scripts.slot_composer.set_flow_model import SetFlowSlotComposer
from scripts.slot_composer.set_losses import hungarian_people_loss


class SetFlowSlotComposerTrainer:
    def __init__(self, config: ProjectConfig, steps: Optional[int] = None, path_weight: float = 1.0, resume: bool = False):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0

        self.mov_ae, self.per_ae = _load_frozen_autoencoders(self.cfg)
        self.mov_ae.encoder.to(self.device).eval()
        self.mov_ae.decoder.to(self.device).eval()
        self.per_ae.encoder.to(self.device).eval()
        self.per_ae.decoder.to(self.device).eval()

        self.steps = int(steps) if steps is not None else int(self.cfg.slot_layers)

        self.model = SetFlowSlotComposer(
            latent_dim=self.cfg.latent_dim,
            num_slots=self.cfg.slot_people_count,
            num_heads=self.cfg.slot_heads,
            ff_mult=self.cfg.slot_ff_mult,
            dropout=self.cfg.slot_dropout,
        ).to(self.device)

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.slot_learning_rate,
            weight_decay=self.cfg.slot_weight_decay,
        )

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
            num_workers=0,
            pin_memory=False,
        )

        self.path_weight = float(path_weight)

        Path(self.cfg.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)

        if resume:
            self._try_resume()

        self._print_run_table()

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_name = f"set_flow_slots_ld{self.cfg.latent_dim}_n{self.cfg.slot_people_count}_h{self.cfg.slot_heads}_ff{self.cfg.slot_ff_mult}_st{self.steps}_{ts}"
        self.tb_dir = str(Path(self.cfg.tensorboard_dir) / run_name)
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        self.recon = FlowSlotComposerReconstructionLogger(
            movie_ae=self.mov_ae,
            people_ae=self.per_ae,
            interval_steps=self.cfg.slot_recon_interval,
            num_samples=self.cfg.slot_recon_num_samples,
            people_slots_to_show=self.cfg.slot_recon_show_slots,
            table_width=self.cfg.slot_recon_table_width,
        )

    def _latest_epoch_ckpt(self) -> Optional[Path]:
        p = Path(self.cfg.model_dir)
        if not p.exists():
            return None
        pat = re.compile(r"^SetFlowSlotComposer_epoch(\d+)\.pt$")
        best_n = -1
        best_path = None
        for f in p.iterdir():
            m = pat.match(f.name)
            if m:
                n = int(m.group(1))
                if n > best_n:
                    best_n = n
                    best_path = f
        return best_path

    def _try_resume(self):
        ckpt = self._latest_epoch_ckpt()
        if ckpt is None:
            return
        state = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        m = re.search(r"epoch(\d+)\.pt$", ckpt.name)
        if m:
            self.start_epoch = int(m.group(1))
        else:
            self.start_epoch = 0

    def _print_run_table(self):
        t = PrettyTable()
        t.field_names = ["item", "value"]
        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        params = sum(p.numel() for p in self.model.parameters())
        t.add_row(["device", str(self.device)])
        t.add_row(["device_name", dev_name])
        t.add_row(["latent_dim", self.cfg.latent_dim])
        t.add_row(["slots", self.cfg.slot_people_count])
        t.add_row(["heads", self.cfg.slot_heads])
        t.add_row(["ff_mult", self.cfg.slot_ff_mult])
        t.add_row(["dropout", self.cfg.slot_dropout])
        t.add_row(["steps", self.steps])
        t.add_row(["batch_size", self.cfg.batch_size])
        t.add_row(["lr", self.cfg.slot_learning_rate])
        t.add_row(["weight_decay", self.cfg.slot_weight_decay])
        t.add_row(["epochs", self.cfg.slot_epochs])
        t.add_row(["start_epoch", self.start_epoch])
        t.add_row(["align_weight", self.cfg.slot_latent_align_weight])
        t.add_row(["div_weight", self.cfg.slot_diversity_weight])
        t.add_row(["path_weight", self.path_weight])
        t.add_row(["recon_interval", self.cfg.slot_recon_interval])
        t.add_row(["recon_samples", self.cfg.slot_recon_num_samples])
        t.add_row(["recon_show_slots", self.cfg.slot_recon_show_slots])
        t.add_row(["recon_table_width", self.cfg.slot_recon_table_width])
        t.add_row(["model_params", params])
        t.add_row(["model_dir", self.cfg.model_dir])
        t.add_row(["db_path", self.cfg.db_path])
        logging.info("\n%s", t.get_string())

    def _encode_people_slot(self, Px_f: List[torch.Tensor], slot_index: int) -> torch.Tensor:
        X = [x[:, slot_index, ...].to(self.device, non_blocking=True) for x in Px_f]
        with torch.no_grad():
            z = self.per_ae.encoder(X)
        return z

    def _log_step(
        self,
        step: int,
        total,
        loss_movie,
        loss_people,
        align,
        div,
        path_loss,
        mask: torch.Tensor,
        z_slots_hat: torch.Tensor,
        iter_time_s: float,
    ):
        lr = float(self.optim.param_groups[0]["lr"])
        self.writer.add_scalar("loss/total", float(total), step)
        self.writer.add_scalar("loss/movie", float(loss_movie), step)
        self.writer.add_scalar("loss/people", float(loss_people), step)
        self.writer.add_scalar("loss/align", float(align), step)
        self.writer.add_scalar("loss/div", float(div), step)
        self.writer.add_scalar("loss/path", float(path_loss), step)
        self.writer.add_scalar("opt/lr", lr, step)
        self.writer.add_scalar("data/slots_valid_mean", float(mask.mean().detach().cpu().item()), step)
        self.writer.add_scalar("time/iter_seconds", float(iter_time_s), step)
        with torch.no_grad():
            self.writer.add_histogram("latents/slots", z_slots_hat.detach().cpu(), step)
        self.writer.flush()

    def train(self):
        step = 0
        for epoch in range(self.start_epoch, self.cfg.slot_epochs):
            pbar = tqdm(self.loader, unit="batch", dynamic_ncols=True)
            pbar.set_description(f"epoch {epoch+1}/{self.cfg.slot_epochs}")
            for Mx, My, Pxs, Pys, mask in pbar:
                t0 = time.perf_counter()

                Mx = [x.to(self.device, non_blocking=True) for x in Mx]
                My = [y.to(self.device, non_blocking=True) for y in My]
                Pxs = [px.to(self.device, non_blocking=True) for px in Pxs]
                Pys = [py.to(self.device, non_blocking=True) for py in Pys]
                mask = mask.to(self.device, non_blocking=True)

                with torch.no_grad():
                    z_movie = self.mov_ae.encoder(Mx)

                z_movie_hat, z_slots_hat, z_seq = self.model(z_movie, steps=self.steps, return_all=True)

                movie_preds = self.mov_ae.decoder(z_movie_hat)
                loss_movie = aggregate_movie_loss(self.mov_ae.fields, movie_preds, My)

                b, n, d = z_slots_hat.shape
                people_preds_per_field = []
                flat = z_slots_hat.reshape(b * n, d)
                for dec in self.per_ae.decoder.decs:
                    y = dec(flat)
                    y = y.view(b, n, *y.shape[1:])
                    people_preds_per_field.append(y)

                loss_people = hungarian_people_loss(self.per_ae.fields, people_preds_per_field, Pys, mask)

                z_true_slots = []
                for i in range(n):
                    zt = self._encode_people_slot(Pxs, i)
                    z_true_slots.append(zt.unsqueeze(1))
                z_true = torch.cat(z_true_slots, dim=1)

                align = 0.0
                if self.cfg.slot_latent_align_weight != 0.0:
                    align = latent_alignment_loss(z_slots_hat, z_true, mask) * float(self.cfg.slot_latent_align_weight)

                div = 0.0
                if self.cfg.slot_diversity_weight != 0.0:
                    div = diversity_penalty(z_slots_hat) * float(self.cfg.slot_diversity_weight)

                s0 = self.model.slots.unsqueeze(0).expand(z_slots_hat.size(0), -1, -1).to(z_slots_hat.device)
                path_loss = straight_path_loss(z_seq, s0, z_true, mask) * self.path_weight

                total = loss_movie + loss_people + align + div + path_loss

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
                    path=float(path_loss.detach().cpu().item()),
                )

                iter_time_s = time.perf_counter() - t0
                self._log_step(
                    step=step - 1,
                    total=float(total.detach().cpu().item()),
                    loss_movie=float(loss_movie.detach().cpu().item()),
                    loss_people=float(loss_people.detach().cpu().item()),
                    align=float(align.detach().cpu().item()) if isinstance(align, torch.Tensor) else float(align),
                    div=float(div.detach().cpu().item()) if isinstance(div, torch.Tensor) else float(div),
                    path_loss=float(path_loss.detach().cpu().item()),
                    mask=mask,
                    z_slots_hat=z_slots_hat,
                    iter_time_s=iter_time_s,
                )

                self.recon.on_batch_end(
                    global_step=step - 1,
                    Mx=Mx,
                    My=My,
                    Pxs=Pxs,
                    Pys=Pys,
                    mask=mask,
                    model=self.model,
                    steps_normal=self.steps,
                )

            out = Path(self.cfg.model_dir) / f"SetFlowSlotComposer_epoch{epoch+1}.pt"
            torch.save(self.model.state_dict(), out)
            self.writer.add_scalar("ckpt/epoch_saved", epoch + 1, step)

        out = Path(self.cfg.model_dir) / "SetFlowSlotComposer_final.pt"
        torch.save(self.model.state_dict(), out)
        self.writer.add_scalar("ckpt/final_saved", 1, step)
        self.writer.flush()
        self.writer.close()
