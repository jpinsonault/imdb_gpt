import os
from typing import Callable, Optional, Tuple, List

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: Tuple[int, int],
        extensions: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.image_size = image_size

        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]

        self.extensions = set(e.lower() for e in extensions)
        self.paths = self._gather_files(self.root, self.extensions)
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

        if transform is None:
            transform = T.Compose(
                [
                    T.Resize(image_size),
                    T.CenterCrop(image_size),
                    T.ToTensor(),
                ]
            )

        self.transform = transform

    def _gather_files(self, root: str, extensions: set) -> list:
        paths = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in extensions:
                    paths.append(os.path.join(dirpath, name))
        paths.sort()
        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image)
