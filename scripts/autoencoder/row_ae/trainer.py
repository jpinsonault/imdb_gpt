import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from .data import _RowDataset, _collate

def fit_row_autoencoder(self) -> None:
    if not self.stats_accumulated:
        self.accumulate_stats()
        self.finalize_stats()
    if self.model is None:
        self.build_autoencoder()

    epochs = int(self.config.get("epochs", 10))
    lr = float(self.config.get("learning_rate", 2e-4))
    wd = float(self.config.get("weight_decay", 1e-4))

    opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    ds = _RowDataset(self.row_generator, self.fields)
    bs = int(self.config.get("batch_size", 32))
    loader = DataLoader(ds, batch_size=bs, collate_fn=_collate)

    writer = None
    if SummaryWriter is not None:
        root = self.config.get("tensorboard_dir", "logs")
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        logdir = Path(root) / f"row_ae_{self.__class__.__name__}_{ts}"
        writer = SummaryWriter(log_dir=str(logdir))

    self.model.train()
    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"{self.__class__.__name__} epoch {epoch+1}/{epochs}")
        for xs, ys in pbar:
            xs = [x.to(self.device) for x in xs]
            ys = [y.to(self.device) for y in ys]
            outs = self.model(xs)
            total = 0.0
            field_losses: Dict[str, float] = {}
            for f, pred, tgt in zip(self.fields, outs, ys):
                l = f.compute_loss(pred, tgt) * float(f.weight)
                total = total + l
                field_losses[f.name] = float(l.detach().cpu().item())
            opt.zero_grad()
            total.backward()
            opt.step()
            total_val = float(total.detach().cpu().item())
            pbar.set_postfix(loss=total_val)
            if writer is not None:
                writer.add_scalar("loss/total", total_val, global_step)
                for k, v in field_losses.items():
                    writer.add_scalar(f"loss/fields/{k}", v, global_step)
            global_step += 1
        self.save_model()
    if writer is not None:
        writer.flush()
        writer.close()
