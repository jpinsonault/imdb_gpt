from typing import Callable, Dict, Iterator, List, Tuple, Optional
import torch
from torch.utils.data import IterableDataset, DataLoader

class MoviesPeopleSequenceDataset(IterableDataset):
    def __init__(
        self,
        row_gen_fn: Callable[[], Iterator[Tuple[Dict, List[Dict]]]],
        movie_fields,
        people_fields,
        active_idx: List[int],
    ):
        super().__init__()
        self.row_gen_fn = row_gen_fn
        self.movie_fields = movie_fields
        self.people_fields = people_fields
        self.active_idx = active_idx

    def __iter__(self):
        for m_row, ppl in self.row_gen_fn():
            X = [f.transform(m_row.get(f.name)) for f in self.movie_fields]
            Y: List[Optional[torch.Tensor]] = []
            for i, f in enumerate(self.people_fields):
                if i not in self.active_idx:
                    Y.append(None)
                    continue
                seq = [f.transform_target(p.get(f.name)) for p in ppl]
                Y.append(torch.stack(seq, dim=0))
            yield X, Y

def collate_movies_people(batch):
    mx = list(zip(*[b[0] for b in batch]))
    my = list(zip(*[b[1] for b in batch]))
    M = [torch.stack(col, dim=0) for col in mx]
    P: List[torch.Tensor] = []
    for col in my:
        first = next((x for x in col if x is not None), None)
        if first is None:
            continue
        P.append(torch.stack(col, dim=0))
    return M, P
