from __future__ import annotations
import sqlite3
from typing import List, Tuple


class EdgeLossLogger:
    """Stores the worst loss seen per edge and how many times that edge has been trained."""

    _BULK_SIZE = 10_000

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._set_pragmas()
        self.cur = self.conn.cursor()
        self.cur.execute("DROP TABLE IF EXISTS edge_losses;")
        self.cur.execute(
            """
            CREATE TABLE edge_losses (
                edgeId      INTEGER PRIMARY KEY,
                num_trained INTEGER NOT NULL,
                total_loss  REAL     NOT NULL
            );
            """
        )
        self.conn.commit()
        self._cache: List[Tuple[int, float]] = []

    def _set_pragmas(self):
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")
        self.conn.execute("PRAGMA cache_size = -200000;")
        self.conn.execute("PRAGMA mmap_size = 268435456;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")

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
        self.cur.executemany(
            """
            INSERT INTO edge_losses (edgeId, num_trained, total_loss)
            VALUES (?, 1, ?)
            ON CONFLICT(edgeId) DO UPDATE SET
                num_trained = edge_losses.num_trained + 1,
                total_loss  = CASE
                                  WHEN excluded.total_loss > edge_losses.total_loss
                                  THEN excluded.total_loss
                                  ELSE edge_losses.total_loss
                              END;
            """,
            self._cache,
        )
        self.conn.commit()
        self._cache.clear()

    def close(self):
        self.flush()
        self.conn.close()
