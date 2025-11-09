import os
from typing import Tuple, List

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class ImagePairDataset(Dataset):
    def __init__(
        self,
        root: str,
        ae_size: Tuple[int, int],
        siren_size: Tuple[int, int],
        extensions: List[str] | None = None,
    ):
        self.root = root
        self.ae_size = ae_size
        self.siren_size = siren_size

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

    def _gather_files(self) -> list:
        paths = []
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

        t_ae = F.to_tensor(img_ae)
        t_siren = F.to_tensor(img_siren)
        return t_ae, t_siren

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        return self._load_image_pair(path)
