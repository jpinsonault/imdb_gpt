# scripts/autoencoder/prefetch.py
import torch

class CudaPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.it = iter(loader)
        self.next_batch = None
        self._preload()

    def _move_to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, list):
            return [self._move_to_device(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        return obj

    def _to_device(self, batch):
        return self._move_to_device(batch)

    def _preload(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = self._to_device(batch)
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device(batch)

    def next(self):
        if self.next_batch is None:
            return None
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch
    