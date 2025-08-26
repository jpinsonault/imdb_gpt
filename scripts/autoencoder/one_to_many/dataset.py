# scripts/autoencoder/one_to_many/dataset.py
from __future__ import annotations
from typing import List, Dict
import torch
from torch.utils.data import IterableDataset
from scripts.autoencoder.fields import BaseField
from scripts.autoencoder.one_to_many.provider import OneToManyProvider

class OneToManyDataset(IterableDataset):
    def __init__(
        self,
        provider: OneToManyProvider,
        source_fields: List[BaseField],
        target_fields: List[BaseField],
    ):
        super().__init__()
        self.provider = provider
        self.source_fields = source_fields
        self.target_fields = target_fields
        self.seq_len = int(provider.seq_len)

    def __iter__(self):
        for s_row in self.provider.iter_sources():
            t_rows = self.provider.targets_for(s_row, self.seq_len)
            orig_len = min(len(t_rows), self.seq_len)
            if orig_len < self.seq_len:
                t_rows = t_rows + [{} for _ in range(self.seq_len - orig_len)]
            else:
                t_rows = t_rows[: self.seq_len]
            x_src = [f.transform(s_row.get(f.name)) for f in self.source_fields]
            y_tgt: List[torch.Tensor] = []
            for f in self.target_fields:
                steps = []
                for t in range(self.seq_len):
                    if t < orig_len:
                        steps.append(f.transform_target(t_rows[t].get(f.name)))
                    else:
                        steps.append(f.get_base_padding_value())
                y_tgt.append(torch.stack(steps, dim=0))
            mask = torch.zeros(self.seq_len, dtype=torch.float32)
            mask[:orig_len] = 1.0
            yield x_src, y_tgt, mask

def collate_one_to_many(batch):
    xm_cols = list(zip(*[b[0] for b in batch]))
    yt_cols = list(zip(*[b[1] for b in batch]))
    Xm = [torch.stack(col, dim=0) for col in xm_cols]
    Yt = [torch.stack(col, dim=0) for col in yt_cols]
    M = torch.stack([b[2] for b in batch], dim=0)
    return Xm, Yt, M
