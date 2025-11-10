import os
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class ImagePairDataset(Dataset):
    def __init__(
        self,
        root: str,
        ae_size: Tuple[int, int],
        siren_size: Tuple[int, int],
        extensions: Optional[List[str]] = None,
        density_alpha: float = 0.9,
    ):
        self.root = root
        self.ae_size = ae_size
        self.siren_size = siren_size
        self.density_alpha = float(density_alpha)

        if extensions is None:
            extensions = [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".gif",
                ".webp",
                ".tif",
                ".tiff",
            ]

        self.extensions = set(e.lower() for e in extensions)
        self.paths = self._gather_files()
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

        self._density_cache: List[Optional[torch.Tensor]] = [None] * len(self.paths)

    def _gather_files(self) -> List[str]:
        paths: List[str] = []
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in self.extensions:
                    paths.append(os.path.join(dirpath, name))
        paths.sort()
        return paths

    def _load_image_pair(self, path: str):
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        ae_w, ae_h = self.ae_size
        s_w, s_h = self.siren_size

        if img.size != (ae_w, ae_h):
            img_ae = img.resize((ae_w, ae_h), Image.LANCZOS)
        else:
            img_ae = img

        if img.size != (s_w, s_h):
            img_siren = img.resize((s_w, s_h), Image.LANCZOS)
        else:
            img_siren = img

        t_ae = TF.to_tensor(img_ae)
        t_siren = TF.to_tensor(img_siren)
        return t_ae, t_siren

    def _compute_density(self, t_siren: torch.Tensor) -> torch.Tensor:
        c, h, w = t_siren.shape

        if c >= 3:
            r = t_siren[0]
            g = t_siren[1]
            b = t_siren[2]
            y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        else:
            y = t_siren.mean(dim=0)

        y = y.view(1, 1, h, w)

        dx = y[:, :, :, 1:] - y[:, :, :, :-1]
        dy = y[:, :, 1:, :] - y[:, :, :-1, :]

        grad = y.new_zeros(1, 1, h, w)

        dx_abs = dx.abs()
        dy_abs = dy.abs()

        grad[:, :, :, 1:] += dx_abs
        grad[:, :, :, :-1] += dx_abs
        grad[:, :, 1:, :] += dy_abs
        grad[:, :, :-1, :] += dy_abs

        grad = F.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)

        g = grad.view(-1)

        g_min = float(g.min())
        g_max = float(g.max())
        if g_max > g_min:
            g = (g - g_min) / (g_max - g_min)
        else:
            g = torch.ones_like(g)

        g = g + 1e-6
        g = g / g.sum()

        alpha = max(0.0, min(1.0, self.density_alpha))
        if alpha > 0.0:
            n = g.numel()
            u = g.new_full((n,), 1.0 / float(n))
            g = (1.0 - alpha) * u + alpha * g
            g = g / g.sum()

        return g

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        t_ae, t_siren = self._load_image_pair(path)

        dens = self._density_cache[idx]
        if dens is None:
            with torch.no_grad():
                dens = self._compute_density(t_siren)
            self._density_cache[idx] = dens

        return t_ae, t_siren, dens
