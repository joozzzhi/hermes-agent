"""Local queue of verbatim turns waiting to reach the shared memory.

A reply must never wait on the network, and the shared store is a database on the other side
of the internet. So a finished turn is written here — a small SQLite file next to the rest of
the agent's state — and shipped later in one batch by the nightly job.

The queue is also the safety net for the case that killed the previous machine's memory: if
the laptop is off, or the database is down for a day, the turns simply wait. Nothing is
dropped for being late, and a row leaves the queue only after the store has confirmed it.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS pending_turns_session_idx ON pending_turns (session_id);
"""


class TurnQueue:
    """Append-only queue of turns; rows are deleted only once they are safely in the store."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False with one lock around every use: turns are enqueued from the
        # agent's thread and drained from the nightly job's, and two connections to one file
        # would only trade this lock for SQLite's "database is locked".
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def enqueue(self, session_id: str, role: str, content: str) -> bool:
        """Remember one side of a turn. Empty content is not a turn and is ignored."""
        content = (content or "").strip()
        if not content:
            return False
        with self._lock:
            self._conn.execute(
                "INSERT INTO pending_turns (session_id, role, content) VALUES (?, ?, ?)",
                (session_id or "session", role, content),
            )
            self._conn.commit()
        return True

    def pending(self, limit: int = 500) -> list[dict[str, Any]]:
        """Oldest first — the shipment keeps the order the conversation happened in."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, role, content, created_at FROM pending_turns ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": row[0], "session_id": row[1], "role": row[2], "content": row[3], "created_at": row[4]}
            for row in rows
        ]

    def release(self, ids: Iterable[int]) -> int:
        """Drop turns the store has confirmed. Called after the insert, never before."""
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        with self._lock:
            placeholders = ",".join("?" for _ in ids)
            cursor = self._conn.execute(f"DELETE FROM pending_turns WHERE id IN ({placeholders})", ids)
            self._conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM pending_turns").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
