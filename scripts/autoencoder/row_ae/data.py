from typing import List, Tuple, Iterator, Dict
import torch
from torch.utils.data import IterableDataset

from ..fields import BaseField

class _RowDataset(IterableDataset):
    def __init__(self, row_gen_fn, fields: List[BaseField]):
        super().__init__()
        self.row_gen_fn = row_gen_fn
        self.fields = fields

    def __iter__(self):
        for row in self.row_gen_fn():
            xs = [f.transform(row.get(f.name)) for f in self.fields]
            ys = [f.transform_target(row.get(f.name)) for f in self.fields]
            yield xs, ys

def _collate(batch: List[Tuple[List[torch.Tensor], List[torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    x_cols = list(zip(*[b[0] for b in batch]))
    y_cols = list(zip(*[b[1] for b in batch]))
    X = [torch.stack(col, dim=0) for col in x_cols]
    Y = [torch.stack(col, dim=0) for col in y_cols]
    return X, Y
