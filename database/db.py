"""database/db.py — thread-local SQLite manager."""
import io, sqlite3, threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class DB:
    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, 'c', None):
            c = sqlite3.connect(self._path, check_same_thread=False, timeout=15)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA journal_mode = WAL")
            c.execute("PRAGMA synchronous = NORMAL")
            self._local.c = c
        return self._local.c

    @property
    def conn(self): return self._conn()

    @contextmanager
    def tx(self):
        c = self._conn()
        try:
            yield c
            c.commit()
        except Exception as e:
            c.rollback(); raise

    def q(self, sql, p=()):  return self._conn().execute(sql, p)
    def one(self, sql, p=()): r = self.q(sql, p).fetchone(); return dict(r) if r else None
    def all(self, sql, p=()): return [dict(r) for r in self.q(sql, p).fetchall()]

    def run(self, sql, p=()):
        with self.tx() as c: c.execute(sql, p)

    def embed_to_blob(self, arr: np.ndarray) -> bytes:
        buf = io.BytesIO(); np.save(buf, arr.astype(np.float32)); return buf.getvalue()

    def blob_to_embed(self, blob: bytes) -> np.ndarray:
        return np.load(io.BytesIO(blob)).astype(np.float32)
