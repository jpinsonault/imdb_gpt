import os
import torch
from torchvision.utils import save_image


class ImageSirenReconstructionSaver:
    def __init__(
        self,
        output_dir: str,
        sample_ae: torch.Tensor,
        sample_target: torch.Tensor,
        every_n_epochs: int,
        max_samples: int,
        image_size: int,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.every_n_epochs = int(every_n_epochs)
        self.image_size = int(image_size)

        max_samples = max(1, int(max_samples))

        if sample_ae.size(0) > max_samples:
            sample_ae = sample_ae[:max_samples]
            sample_target = sample_target[:max_samples]

        self.sample_ae = sample_ae
        self.sample_target = sample_target

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
        if self.sample_ae is None or self.sample_ae.size(0) == 0:
            return

        encoder.eval()
        siren.eval()

        imgs_ae = self.sample_ae.to(device)
        imgs_tgt = self.sample_target.to(device)

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

        path = os.path.join(self.output_dir, f"image_siren_epoch_{epoch:04d}.png")
        save_image(grid, path, nrow=nrow)

        encoder.train()
        siren.train()
