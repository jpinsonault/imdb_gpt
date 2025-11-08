import os
from typing import Tuple, List

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: Tuple[int, int],
        extensions: List[str] | None = None,
    ):
        self.root = root
        self.image_size = image_size

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

    def _load_image(self, path: str):
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        target_w, target_h = self.image_size
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.LANCZOS)

        tensor = F.to_tensor(img)
        return tensor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        return self._load_image(path)
