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
from scripts.slot_composer.vector_field_model import SlotFlowODE
from scripts.slot_composer.ode_recon_logger import ODESlotReconstructionLogger
from scripts.slot_composer.set_losses import (
    _cosine_cost,
    _greedy_assign_indices,
    _gather_true_by_assign,
    latent_alignment_loss_matched,
)
from scripts.slot_composer.losses import aggregate_people_loss, diversity_penalty
from scripts.slot_composer.flow_matching import rectified_flow_loss_matched_multi
from scripts.slot_composer.flow_losses import straight_path_loss

class FlowMatchingSlotComposerTrainer:
    def __init__(self, config: ProjectConfig, resume: bool = False):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0

        self.mov_ae, self.per_ae = _load_frozen_autoencoders(self.cfg)
        self.mov_ae.encoder.to(self.device).eval()
        self.mov_ae.decoder.to(self.device).eval()
        self.per_ae.encoder.to(self.device).eval()
        self.per_ae.decoder.to(self.device).eval()

        self.flow = SlotFlowODE(
            latent_dim=self.cfg.latent_dim,
            num_slots=self.cfg.slot_people_count,
            hidden_mult=self.cfg.slot_flow_hidden_mult,
            layers=self.cfg.slot_flow_layers,
            fourier_dim=self.cfg.slot_flow_fourier_dim,
            steps=self.cfg.slot_flow_steps,
            t0=self.cfg.slot_flow_t0,
            t1=self.cfg.slot_flow_t1,
            noise_scale=self.cfg.slot_flow_noise_scale,
            seed_from_movie=self.cfg.slot_seed_from_movie,
            cond_width=getattr(self.cfg, "slot_cond_width", 2.0),
        ).to(self.device)

        self.optim = torch.optim.AdamW(
            self.flow.parameters(),
            lr=self.cfg.slot_learning_rate,
            weight_decay=self.cfg.slot_weight_decay,
        )

        self.dataset = TitlePeopleIterable(
            db_path=self.cfg.db_path,
            principals_table=self.cfg.principals_table,
            movie_ae=self.mov_ae,
            people_ae=self.per_ae,
            num_slots=self.cfg.slot_people_count,
            shuffle=True,
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_batch,
            num_workers=0,
            pin_memory=False,
        )

        Path(self.cfg.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)

        if resume:
            self._try_resume()

        self._print_run_table()

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_name = f"fm_set_flow_n{self.cfg.slot_people_count}_{ts}"
        self.tb_dir = str(Path(self.cfg.tensorboard_dir) / run_name)
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        self.recon = ODESlotReconstructionLogger(
            movie_ae=self.mov_ae,
            people_ae=self.per_ae,
            interval_steps=self.cfg.slot_recon_interval,
            num_samples=self.cfg.slot_recon_num_samples,
            people_slots_to_show=self.cfg.slot_recon_show_slots,
            table_width=self.cfg.slot_recon_table_width,
            writer=self.writer,
        )

    def _latest_epoch_ckpt(self) -> Optional[Path]:
        p = Path(self.cfg.model_dir)
        if not p.exists():
            return None
        pat = re.compile(r"^SlotFlowFM_epoch(\d+)\.pt$")
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
        self.flow.load_state_dict(state, strict=True)
        m = re.search(r"epoch(\d+)\.pt$", ckpt.name)
        self.start_epoch = int(m.group(1)) if m else 0

    def _print_run_table(self):
        t = PrettyTable()
        t.field_names = ["item", "value"]
        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        params = sum(p.numel() for p in self.flow.parameters())
        t.add_row(["device", str(self.device)])
        t.add_row(["device_name", dev_name])
        t.add_row(["latent_dim", self.cfg.latent_dim])
        t.add_row(["slots", self.cfg.slot_people_count])
        t.add_row(["flow_steps_infer", self.cfg.slot_flow_steps])
        t.add_row(["t0", self.cfg.slot_flow_t0])
        t.add_row(["t1", self.cfg.slot_flow_t1])
        t.add_row(["noise_scale", self.cfg.slot_flow_noise_scale])
        t.add_row(["seed_from_movie", self.cfg.slot_seed_from_movie])
        t.add_row(["conditioning", "film"])
        t.add_row(["cond_width", getattr(self.cfg, "slot_cond_width", 2.0)])
        t.add_row(["batch_size", self.cfg.batch_size])
        t.add_row(["lr", self.cfg.slot_learning_rate])
        t.add_row(["weight_decay", self.cfg.slot_weight_decay])
        t.add_row(["epochs", self.cfg.slot_epochs])
        t.add_row(["recon_interval", self.cfg.slot_recon_interval])
        t.add_row(["recon_samples", self.cfg.slot_recon_num_samples])
        t.add_row(["recon_show_slots", self.cfg.slot_recon_show_slots])
        t.add_row(["recon_table_width", self.cfg.slot_recon_table_width])
        t.add_row(["t_samples", getattr(self.cfg, "slot_flow_t_samples", 1)])
        t.add_row(["w_recon", getattr(self.cfg, "slot_loss_w_recon", 0.0)])
        t.add_row(["w_align", getattr(self.cfg, "slot_loss_w_align", 0.0)])
        t.add_row(["w_div", getattr(self.cfg, "slot_loss_w_div", 0.0)])
        t.add_row(["w_path", getattr(self.cfg, "slot_loss_w_path", 0.0)])
        t.add_row(["model_params", params])
        t.add_row(["model_dir", self.cfg.model_dir])
        t.add_row(["db_path", self.cfg.db_path])
        logging.info("\n%s", t.get_string())

    @torch.no_grad()
    def _encode_people_slots(self, Pxs: List[torch.Tensor]) -> torch.Tensor:
        zs = []
        for i in range(self.cfg.slot_people_count):
            X = [x[:, i, ...].to(self.device, non_blocking=True) for x in Pxs]
            z = self.per_ae.encoder(X)
            zs.append(z.unsqueeze(1))
        return torch.cat(zs, dim=1)

    def _log_step(self, step: int, scalars: dict, mask: torch.Tensor, z_slots_final: torch.Tensor | None, iter_time_s: float):
        for k, v in scalars.items():
            self.writer.add_scalar(k, float(v), step)
        self.writer.add_scalar("data/slots_valid_mean", float(mask.mean().detach().cpu().item()), step)
        self.writer.add_scalar("time/iter_seconds", float(iter_time_s), step)
        if z_slots_final is not None:
            with torch.no_grad():
                self.writer.add_histogram("latents/slots_final", z_slots_final.detach().cpu(), step)
        self.writer.flush()

    def train(self):
        step = 0
        for epoch in range(self.start_epoch, self.cfg.slot_epochs):
            if hasattr(self.loader, "dataset"):
                self.loader.dataset.set_epoch_seed(epoch)
            pbar = tqdm(self.loader, unit="batch", dynamic_ncols=True)
            pbar.set_description(f"epoch {epoch+1}/{self.cfg.slot_epochs}")
            for Mx, My, Pxs, Pys, mask in pbar:
                t0 = time.perf_counter()

                Mx = [x.to(self.device, non_blocking=True) for x in Mx]
                Pxs = [px.to(self.device, non_blocking=True) for px in Pxs]
                Pys = [py.to(self.device, non_blocking=True) for py in Pys]
                mask = mask.to(self.device, non_blocking=True)

                with torch.no_grad():
                    z_movie = self.mov_ae.encoder(Mx)
                    z_true = self._encode_people_slots(Pxs)

                s0 = self.flow._seed(z_movie)

                loss_fm = rectified_flow_loss_matched_multi(
                    vector_field=self.flow.field,
                    z_movie=z_movie,
                    s0=s0,
                    z_true=z_true,
                    mask=mask,
                    t_samples=max(1, int(getattr(self.cfg, "slot_flow_t_samples", 1))),
                )

                w_recon = float(getattr(self.cfg, "slot_loss_w_recon", 0.0))
                w_align = float(getattr(self.cfg, "slot_loss_w_align", 0.0))
                w_div = float(getattr(self.cfg, "slot_loss_w_div", 0.0))
                w_path = float(getattr(self.cfg, "slot_loss_w_path", 0.0))

                sT, traj = self.flow.solver(self.flow.field, s0, z_movie, return_all=(w_path > 0.0))

                with torch.no_grad():
                    b, n, d = sT.shape
                    flat = sT.reshape(b * n, d)
                    people_preds = []
                    for dec in self.per_ae.decoder.decs:
                        y = dec(flat)
                        y = y.view(b, n, *y.shape[1:])
                        people_preds.append(y)

                loss_recon = aggregate_people_loss(self.per_ae.fields, people_preds, Pys, mask) if w_recon > 0.0 else sT.new_zeros(())
                loss_align = latent_alignment_loss_matched(sT, z_true, mask) if w_align > 0.0 else sT.new_zeros(())
                loss_div = diversity_penalty(sT) if w_div > 0.0 else sT.new_zeros(())

                if w_path > 0.0:
                    with torch.no_grad():
                        cost = _cosine_cost(sT, z_true)
                        assign = _greedy_assign_indices(cost, int(z_true.size(1)))
                        z_tgt = _gather_true_by_assign(z_true, assign)
                    loss_path = straight_path_loss(traj, s0, z_tgt, mask)
                else:
                    loss_path = s0.new_zeros(())

                total = loss_fm + w_recon * loss_recon + w_align * loss_align + w_div * loss_div + w_path * loss_path

                self.optim.zero_grad(set_to_none=True)
                total.backward()
                self.optim.step()

                step += 1
                pbar.set_postfix(
                    loss=float(total.detach().cpu().item()),
                    fm=float(loss_fm.detach().cpu().item()),
                    ppl=float(loss_recon.detach().cpu().item()) if w_recon > 0 else 0.0,
                    align=float(loss_align.detach().cpu().item()) if w_align > 0 else 0.0,
                    div=float(loss_div.detach().cpu().item()) if w_div > 0 else 0.0,
                    path=float(loss_path.detach().cpu().item()) if w_path > 0 else 0.0,
                )

                iter_time_s = time.perf_counter() - t0
                self._log_step(
                    step=step - 1,
                    scalars={
                        "loss/total": float(total.detach().cpu().item()),
                        "loss/fm": float(loss_fm.detach().cpu().item()),
                        "loss/people_recon@final": float(loss_recon.detach().cpu().item()) if w_recon > 0 else 0.0,
                        "loss/align_matched@final": float(loss_align.detach().cpu().item()) if w_align > 0 else 0.0,
                        "loss/diversity@final": float(loss_div.detach().cpu().item()) if w_div > 0 else 0.0,
                        "loss/path@traj": float(loss_path.detach().cpu().item()) if w_path > 0 else 0.0,
                        "opt/lr": float(self.optim.param_groups[0]["lr"]),
                    },
                    mask=mask,
                    z_slots_final=sT,
                    iter_time_s=iter_time_s,
                )

                self.recon.on_batch_end(
                    global_step=step - 1,
                    Mx=Mx,
                    My=My,
                    Pxs=Pxs,
                    Pys=Pys,
                    mask=mask,
                    z_movie=z_movie,
                    z_slots_final=sT.detach(),
                    z_slots_traj=None if traj is None else traj.detach(),
                )

                if (step % self.cfg.slot_save_interval) == 0:
                    out = Path(self.cfg.model_dir) / "SlotFlowFM.pt"
                    torch.save(self.flow.state_dict(), out)

            out = Path(self.cfg.model_dir) / f"SlotFlowFM_epoch{epoch+1}.pt"
            torch.save(self.flow.state_dict(), out)

        out = Path(self.cfg.model_dir) / "SlotFlowFM_final.pt"
        torch.save(self.flow.state_dict(), out)
        self.writer.add_scalar("ckpt/final_saved", 1, step)
        self.writer.flush()
        self.writer.close()
