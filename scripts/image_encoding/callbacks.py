import os
from typing import Optional

import torch
from torchvision.utils import save_image


class ReconstructionSaver:
    def __init__(
        self,
            output_dir: str,
            sample_batch: torch.Tensor,
            every_n_epochs: int = 1,
            max_samples: int = 8,
        ):
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
            self.every_n_epochs = every_n_epochs

            if sample_batch.size(0) > max_samples:
                sample_batch = sample_batch[:max_samples]

            self.sample_batch = sample_batch

        def maybe_save(self, epoch: int, model, device: torch.device):
            if epoch % self.every_n_epochs != 0:
                return

            model.eval()
            with torch.no_grad():
                batch = self.sample_batch.to(device)
                recon = model(batch)

            grid = torch.cat([batch, recon], dim=0)
            path = os.path.join(self.output_dir, f"recon_epoch_{epoch:04d}.png")
            save_image(grid, path, nrow=batch.size(0))
            model.train()
