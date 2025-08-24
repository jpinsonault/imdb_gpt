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

    def _to_device(self, batch):
        xm, yp, m = batch
        xm = [x.to(self.device, non_blocking=True) for x in xm]
        yp = [y.to(self.device, non_blocking=True) for y in yp]
        m = m.to(self.device, non_blocking=True)
        return xm, yp, m

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
