import sqlite3, json, threading


class EdgeLossLogger:
    """
    Buffered writer for per‑edge training losses.
    Each call to `add` enqueues one row; the buffer is flushed to disk
    automatically every `flush_every` rows or when `flush()` is called.
    """
    def __init__(
        self,
        db_path: str,
        flush_every: int = 2048,
    ):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.flush_every = flush_every
        self.buffer: list[tuple] = []
        self.lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------ api
    def add(
        self,
        edge_id: int,
        epoch: int,
        batch: int,
        total_loss: float,
        field_losses: dict[str, float],
    ):
        rec = (
            edge_id,
            epoch,
            batch,
            float(total_loss),
            json.dumps(field_losses, separators=(",", ":")),
        )
        with self.lock:
            self.buffer.append(rec)
            if len(self.buffer) >= self.flush_every:
                self._flush_locked()

    def flush(self):
        with self.lock:
            self._flush_locked()

    def close(self):
        self.flush()
        self.conn.close()

    # ----------------------------------------------------------- internals
    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous   =OFF;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edge_losses (
                edgeId      INTEGER,
                epoch       INTEGER,
                batch       INTEGER,
                total_loss  REAL,
                field_json  TEXT
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_edge_losses_edge ON edge_losses(edgeId);"
        )
        self.conn.commit()

    def _flush_locked(self):
        if not self.buffer:
            return
        self.conn.executemany(
            """
            INSERT INTO edge_losses (edgeId, epoch, batch, total_loss, field_json)
            VALUES (?,?,?,?,?);
            """,
            self.buffer,
        )
        self.conn.commit()
        self.buffer.clear()
