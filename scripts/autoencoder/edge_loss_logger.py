from __future__ import annotations
from typing import List, Tuple, Dict


class EdgeLossLogger:
    """Tracks worst loss per edge in memory."""

    _BULK_SIZE = 10_000

    def __init__(self, db_path: str):
        self._cache: List[Tuple[int, float]] = []
        self._loss_map: Dict[int, Tuple[int, float]] = {}

    def add(
        self,
        edge_id: int,
        epoch: int,
        batch: int,
        total_loss: float,
        field_losses: dict[str, float],
    ):
        self._cache.append((edge_id, total_loss))
        if len(self._cache) >= self._BULK_SIZE:
            self.flush()

    def flush(self):
        if not self._cache:
            return
        for eid, loss in self._cache:
            count, worst = self._loss_map.get(eid, (0, float("-inf")))
            if loss > worst:
                worst = loss
            self._loss_map[eid] = (count + 1, worst)
        self._cache.clear()

    def snapshot(self) -> Dict[int, float]:
        self.flush()
        return {eid: worst for eid, (_, worst) in self._loss_map.items()}

    def close(self):
        self.flush()
        self._loss_map.clear()
