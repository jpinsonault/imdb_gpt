import os
import torch
from torchvision.utils import save_image


class ImageSirenReconstructionSaver:
    def __init__(
        self,
        output_dir: str,
        samples: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
        every_n_epochs: int,
        max_samples: int,
        image_size: int,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.every_n_epochs = int(every_n_epochs)
        self.image_size = int(image_size)

        self.samples: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        if samples:
            for name, pair in samples.items():
                if not isinstance(pair, tuple) or len(pair) != 2:
                    continue
                ae, tgt = pair
                if not isinstance(ae, torch.Tensor) or not isinstance(tgt, torch.Tensor):
                    continue
                max_samples = max(1, int(max_samples))
                if ae.size(0) > max_samples:
                    ae = ae[:max_samples]
                    tgt = tgt[:max_samples]
                if ae.size(0) == 0:
                    continue
                self.samples[str(name)] = (ae, tgt)

    def _best_nrow(self, num_pairs: int) -> int:
        n = max(2 * int(num_pairs), 2)
        best_nrow = 2
        best_score = None

        for nrow in range(2, n + 1, 2):
            rows = (n + nrow - 1) // nrow
            score = abs(rows - nrow)
            if best_score is None or score < best_score:
                best_score = score
                best_nrow = nrow

        return best_nrow

    @torch.no_grad()
    def maybe_save(
        self,
        epoch: int,
        encoder,
        siren,
        device: torch.device,
        coord_grid: torch.Tensor,
    ):
        if self.every_n_epochs <= 0:
            return
        if epoch % self.every_n_epochs != 0:
            return
        if not self.samples:
            return

        encoder.eval()
        siren.eval()

        for split_name, (sample_ae, sample_target) in self.samples.items():
            imgs_ae = sample_ae.to(device)
            imgs_tgt = sample_target.to(device)

            z = encoder.encode(imgs_ae)

            b = imgs_ae.size(0)
            c = imgs_tgt.size(1)
            h = self.image_size
            w = self.image_size

            coords = coord_grid.to(device)
            coords = coords.unsqueeze(0).repeat(b, 1, 1).contiguous()

            logits = siren(coords, z)
            recon = torch.sigmoid(logits)

            recon = recon.view(b, h, w, c)
            recon = recon.permute(0, 3, 1, 2).contiguous()

            stacked = []
            for i in range(b):
                stacked.append(imgs_tgt[i])
                stacked.append(recon[i])

            grid = torch.stack(stacked, dim=0)
            nrow = self._best_nrow(b)

            path = os.path.join(self.output_dir, f"image_siren_{split_name}_epoch_{epoch:04d}.png")
            save_image(grid, path, nrow=nrow)

        encoder.train()
        siren.train()
