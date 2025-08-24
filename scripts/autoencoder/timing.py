# scripts/autoencoder/timing.py
import time
import torch

class _GPUEventTimer:
    def __init__(self, print_every: int):
        self.print_every = max(1, int(print_every))
        self.reset_accum()
        self._step = 0

    def reset_accum(self):
        self.accum_ms = {
            "total": 0.0,
            "data": 0.0,
            "h2d": 0.0,
            "mov_enc": 0.0,
            "trunk": 0.0,
            "ppl_dec": 0.0,
            "rec": 0.0,
            "tgt_enc": 0.0,
            "nce": 0.0,
            "backward": 0.0,
            "opt": 0.0,
        }

    def _event_pair(self):
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def start_step(self):
        self._t0 = time.perf_counter()
        self._pairs = {}

    def end_step_and_accumulate(self):
        self.accum_ms["total"] += (time.perf_counter() - self._t0) * 1000.0
        self._step += 1
        if self._step % self.print_every != 0:
            return None
        torch.cuda.synchronize()
        out = {}
        total = self.accum_ms["total"]
        parts = 0.0
        for k, v in self.accum_ms.items():
            if k == "total":
                continue
            parts += v
        residual = max(0.0, total - parts)
        keys = ["total","backward","trunk","mov_enc","tgt_enc","rec","opt","ppl_dec","nce","data","h2d","residual"]
        out["total"] = total
        out["backward"] = self.accum_ms["backward"]
        out["trunk"] = self.accum_ms["trunk"]
        out["mov_enc"] = self.accum_ms["mov_enc"]
        out["tgt_enc"] = self.accum_ms["tgt_enc"]
        out["rec"] = self.accum_ms["rec"]
        out["opt"] = self.accum_ms["opt"]
        out["ppl_dec"] = self.accum_ms["ppl_dec"]
        out["nce"] = self.accum_ms["nce"]
        out["data"] = self.accum_ms["data"]
        out["h2d"] = self.accum_ms["h2d"]
        out["residual"] = residual
        self.reset_accum()
        return keys, out

    def cpu_range(self, name):
        class _R:
            def __init__(self, outer, nm):
                self.o = outer
                self.nm = nm
            def __enter__(self):
                self.t0 = time.perf_counter()
            def __exit__(self, a, b, c):
                self.o.accum_ms[self.nm] += (time.perf_counter() - self.t0) * 1000.0
        return _R(self, name)

    def gpu_range(self, name):
        class _R:
            def __init__(self, outer, nm):
                self.o = outer
                self.nm = nm
            def __enter__(self):
                self.s, self.e = self.o._event_pair()
                self.s.record()
            def __exit__(self, a, b, c):
                self.e.record()
                torch.cuda.synchronize()
                self.o.accum_ms[self.nm] += self.s.elapsed_time(self.e)
        return _R(self, name)

    def print_line(self, keys, vals, step_idx):
        total = vals["total"]
        frags = []
        for k in keys:
            ms = vals[k]
            pct = 0.0 if total <= 0 else (ms / total) * 100.0
            frags.append(f"{k:>8}: {ms:7.2f} ms {pct:6.1f}%")
        print(f"[step {step_idx}] " + " | ".join(frags))
