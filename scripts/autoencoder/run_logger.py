# scripts/autoencoder/run_logger.py
import os
import json
import html
import time
import logging
from typing import Dict, Any

from config import ProjectConfig
import torch

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
except Exception:
    _TBWriter = None


def _unique_log_dir(root: str, base: str) -> str:
    ts = time.strftime("%Y-%m-%d_%H%M")
    path = os.path.join(root, f"{base}_{ts}")
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        cand = f"{path}_{i}"
        if not os.path.exists(cand):
            return cand
        i += 1


class RunLogger:
    def __init__(self, log_root: str, run_prefix: str, config: ProjectConfig):
        self.enabled = _TBWriter is not None
        self.run_dir = _unique_log_dir(log_root, run_prefix) if self.enabled else None
        self.writer = _TBWriter(log_dir=self.run_dir) if self.enabled else None
        self.step = 0
        self.log_every = int(config.log_interval)
        self.batch_size = config.batch_size
        self.last_log_t = time.perf_counter()
        if self.writer:
            self._write_config(config)
            self._write_hardware()

    def _write_config(self, config):
        from dataclasses import is_dataclass, asdict
        if isinstance(config, dict):
            node = config
        elif hasattr(config, "to_dict"):
            node = config.to_dict()
        elif is_dataclass(config):
            node = asdict(config)
        else:
            node = dict(vars(config))
        txt = json.dumps(node, indent=2, sort_keys=True)
        self.writer.add_text("config/json", f"<pre>{html.escape(txt)}</pre>", 0)
        if self.run_dir:
            try:
                with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
                    f.write(txt)
            except Exception:
                pass
        flat = {}
        stack = [("", node)]
        while stack:
            prefix, cur = stack.pop()
            if isinstance(cur, dict):
                for k, v in cur.items():
                    key = f"{prefix}.{k}" if prefix else k
                    stack.append((key, v))
            else:
                flat[prefix] = cur
        nums = {k: float(v) for k, v in flat.items() if isinstance(v, (int, float))}
        try:
            self.writer.add_hparams(nums, {})
        except Exception:
            pass


    def _write_hardware(self):
        lines = [f"torch {torch.__version__}"]
        if torch.cuda.is_available():
            lines.append(f"cuda {torch.version.cuda}")
            try:
                lines.append(f"device {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
        self.writer.add_text("env/hardware", "\n".join(lines), 0)

    def add_scalars(self, total: float, rec: float, nce: float, iter_time: float, opt):
        if not self.writer:
            return
        s = self.step
        self.writer.add_scalar("loss/total", float(total), s)
        self.writer.add_scalar("loss/reconstruction", float(rec), s)
        self.writer.add_scalar("loss/nce", float(nce), s)
        self.writer.add_scalar("time/iter_sec", float(iter_time), s)
        try:
            for i, g in enumerate(opt.param_groups):
                self.writer.add_scalar(f"lr/group_{i}", float(g["lr"]), s)
        except Exception:
            pass

    def add_field_losses(self, namespace: str, losses: Dict[str, float]):
        if not self.writer:
            return
        s = self.step
        for k, v in losses.items():
            self.writer.add_scalar(f"{namespace}/{k}", float(v), s)

    def add_extremes(self, batch_min: float, batch_max: float):
        if not self.writer:
            return
        s = self.step
        self.writer.add_scalar("loss/batch_min", float(batch_min), s)
        self.writer.add_scalar("loss/batch_max", float(batch_max), s)

    def tick(self, total: float, rec: float, nce: float):
        self.step += 1
        if self.step % self.log_every != 0:
            return
        now = time.perf_counter()
        self.last_log_t = now

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()


def build_run_logger(config: ProjectConfig) -> RunLogger:
    log_root = config.tensorboard_dir
    return RunLogger(log_root, "joint_autoencoder", config)
