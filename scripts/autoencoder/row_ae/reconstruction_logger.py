import logging
from itertools import islice
import random
import time
from typing import Dict, List

import numpy as np
import torch
from prettytable import PrettyTable

from ..fields import NumericDigitCategoryField
from . import db as row_db


class RowReconstructionLogger:
    def __init__(
        self,
        interval_steps: int,
        row_autoencoder,
        db_path: str,
        num_samples: int = 5,
        table_width: int = 40,
        max_scan: int = 20000,
        sample_strategy: str = "fast",
    ):
        self.interval_steps = max(1, int(interval_steps))
        self.row_autoencoder = row_autoencoder
        self.db_path = db_path
        self.num_samples = num_samples
        self.w = table_width
        self.samples: List[Dict] = []

        t0 = time.perf_counter()
        fast_ok = False
        if sample_strategy == "fast":
            try:
                names = [f.name for f in self.row_autoencoder.fields]
                if "primaryTitle" in names:
                    cand = row_db.sample_titles_fast(self.db_path, self.num_samples)
                    if cand:
                        self.samples = cand
                        fast_ok = True
                elif "primaryName" in names:
                    cand = row_db.sample_people_fast(self.db_path, self.num_samples)
                    if cand:
                        self.samples = cand
                        fast_ok = True
            except Exception:
                fast_ok = False

        if not fast_ok:
            try:
                it = islice(self.row_autoencoder.row_generator(), max_scan)
                all_rows = list(it)
                n = min(self.num_samples, len(all_rows))
                self.samples = random.sample(all_rows, n) if n > 0 else []
            except Exception:
                self.samples = []

        dt = time.perf_counter() - t0
        logging.info(
            f"RowReconstructionLogger: prepared {len(self.samples)} samples in {dt:.2f}s (strategy={'fast' if fast_ok else 'scan'})"
        )

    @torch.no_grad()
    def _encode(self, row):
        xs = [f.transform(row.get(f.name)) for f in self.row_autoencoder.fields]
        xs = [x.unsqueeze(0).to(self.row_autoencoder.device) for x in xs]
        z = self.row_autoencoder.encoder(xs)
        return z.detach()

    @torch.no_grad()
    def _decode(self, z_tensor):
        outs = self.row_autoencoder.decoder(z_tensor)
        return [o.detach().cpu().numpy()[0] for o in outs]

    def _tensor_to_string(self, field, main_tensor: np.ndarray) -> str:
        try:
            if isinstance(field, NumericDigitCategoryField):
                arr = np.array(main_tensor)
                return field.to_string(arr if arr.ndim == 2 else arr.flatten())
            if hasattr(field, "tokenizer") and field.tokenizer is not None:
                return field.to_string(np.array(main_tensor))
            arr = np.array(main_tensor)
            if arr.ndim > 1:
                arr = arr.flatten()
            return field.to_string(arr)
        except Exception:
            return "[Conversion Error]"

    def on_batch_end(self, global_step: int):
        if not self.samples:
            return
        if (global_step + 1) % self.interval_steps != 0:
            return

        print(f"\nBatch {global_step + 1}: Reconstruction Demo")
        for i, row_dict in enumerate(self.samples):
            print(f"\nSample {i + 1}:")
            table = PrettyTable()
            table.field_names = ["Field", "Original Value", "Reconstructed"]
            table.align = "l"
            for col in ["Original Value", "Reconstructed"]:
                table.max_width[col] = self.w

            z = self._encode(row_dict)
            preds = self._decode(z)

            for field, pred in zip(self.row_autoencoder.fields, preds):
                field_name = field.name
                original_raw = row_dict.get(field_name, "N/A")
                original_str = ", ".join(map(str, original_raw)) if isinstance(original_raw, list) else str(original_raw)
                reconstructed_str = self._tensor_to_string(field, pred)
                table.add_row([field_name, original_str, reconstructed_str])

            print(table)
