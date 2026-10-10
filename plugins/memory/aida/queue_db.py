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
import json
from pathlib import Path
from typing import Any, Iterable

from .write import scrub

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS pending_turns_session_idx ON pending_turns (session_id);

-- Обмены ждут разбора отдельно от реплик. Реплика уезжает в базу сразу, а разбор может
-- не состояться — модель молчит, квота кончилась. Без этой очереди такой обмен не
-- разобрался бы уже никогда: реплик в первой очереди к тому моменту нет.
CREATE TABLE IF NOT EXISTS pending_exchanges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Темы, размеченные внутри одной сессии, ждут разбора по каждой теме целиком (а не по
-- обмену за обменом): одна модельная разметка режет день на смысловые блоки, каждый блок
-- потом разбирается отдельным вызовом. ``turns_json`` хранит сами реплики блока
-- (role+content), а не только границы id — блок остаётся самодостаточным для разбора,
-- даже если сессия к тому моменту уже очищена от исходных pending_turns.
CREATE TABLE IF NOT EXISTS pending_topics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  topic TEXT NOT NULL,
  turns_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
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
        # timeout: the gateway and a CLI session may hold the same file; wait for the other
        # writer instead of failing the turn with "database is locked".
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=15)
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Columns added after the first release; a queue file from before keeps working.

        ``shipped`` — the turn is already in the shared store and waits here only to be
        distilled. ``distill`` — 0 for a turn the agent already saved a fact from on purpose.
        Rows from the old nightly-only scheme get shipped=0, which is exactly what they are.
        """
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(pending_turns)")}
        if "shipped" not in columns:
            self._conn.execute("ALTER TABLE pending_turns ADD COLUMN shipped INTEGER NOT NULL DEFAULT 0")
        if "distill" not in columns:
            self._conn.execute("ALTER TABLE pending_turns ADD COLUMN distill INTEGER NOT NULL DEFAULT 1")

    def enqueue(self, session_id: str, role: str, content: str, *, shipped: bool = False,
                distill: bool = True) -> bool:
        """Remember one side of a turn. Empty content is not a turn and is ignored."""
        content = scrub(content).strip()
        if not content:
            return False
        with self._lock:
            self._conn.execute(
                "INSERT INTO pending_turns (session_id, role, content, shipped, distill) VALUES (?, ?, ?, ?, ?)",
                (session_id or "session", role, content, int(bool(shipped)), int(bool(distill))),
            )
            self._conn.commit()
        return True

    def _turns(self, where: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, role, content, created_at, shipped, distill FROM pending_turns "
                f"WHERE {where} ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": row[0], "session_id": row[1], "role": row[2], "content": row[3],
             "created_at": row[4], "shipped": bool(row[5]), "distill": bool(row[6])}
            for row in rows
        ]

    def pending(self, limit: int = 500) -> list[dict[str, Any]]:
        """Everything waiting, oldest first — the order the conversation happened in."""
        return self._turns("1 = 1", limit)

    def unshipped(self, limit: int = 500) -> list[dict[str, Any]]:
        """Turns the shared store has not confirmed yet."""
        return self._turns("shipped = 0", limit)

    def shipped(self, limit: int = 500) -> list[dict[str, Any]]:
        """Turns already in the store, waiting only to be distilled."""
        return self._turns("shipped = 1", limit)

    def mark_shipped(self, ids: Iterable[int]) -> int:
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        with self._lock:
            placeholders = ",".join("?" for _ in ids)
            cursor = self._conn.execute(f"UPDATE pending_turns SET shipped = 1 WHERE id IN ({placeholders})", ids)
            self._conn.commit()
            return cursor.rowcount

    def newest_age_seconds(self) -> float | None:
        """How long ago the last turn was queued; None when nothing waits."""
        with self._lock:
            row = self._conn.execute(
                "SELECT (julianday('now') - julianday(max(created_at))) * 86400.0 FROM pending_turns"
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

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

    # -- обмены, ждущие разбора ---------------------------------------------------------

    def enqueue_exchange(self, session_id: str, question: str, answer: str) -> bool:
        question, answer = scrub(question).strip(), scrub(answer).strip()
        if not question or not answer:
            return False
        with self._lock:
            self._conn.execute(
                "INSERT INTO pending_exchanges (session_id, question, answer) VALUES (?, ?, ?)",
                (session_id or "session", question, answer),
            )
            self._conn.commit()
        return True

    def pending_exchanges(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, question, answer FROM pending_exchanges ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"id": r[0], "session_id": r[1], "question": r[2], "answer": r[3]} for r in rows]

    def release_exchanges(self, ids: Iterable[int]) -> int:
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        with self._lock:
            placeholders = ",".join("?" for _ in ids)
            cursor = self._conn.execute(
                f"DELETE FROM pending_exchanges WHERE id IN ({placeholders})", ids
            )
            self._conn.commit()
            return cursor.rowcount

    def count_exchanges(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM pending_exchanges").fetchone()[0])

    # -- темы, ждущие разбора -------------------------------------------------------------

    def enqueue_topic(self, session_id: str, topic: str, turns: list[dict[str, str]]) -> bool:
        """One segmented block: a topic label plus the turns (role+content) inside it.

        Empty turns mean the segmenter drew a block with nothing in it — that's a malformed
        answer, not a topic, and is not worth a row.
        """
        topic = scrub(topic).strip()
        if not topic or not turns:
            return False
        turns = [{**turn, "content": scrub(str(turn.get("content", "")))} for turn in turns]
        with self._lock:
            self._conn.execute(
                "INSERT INTO pending_topics (session_id, topic, turns_json) VALUES (?, ?, ?)",
                (session_id or "session", topic, json.dumps(turns, ensure_ascii=False)),
            )
            self._conn.commit()
        return True

    def pending_topics(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, topic, turns_json FROM pending_topics ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            try:
                turns = json.loads(row[3])
            except Exception:
                turns = []
            result.append({"id": row[0], "session_id": row[1], "topic": row[2], "turns": turns})
        return result

    def release_topics(self, ids: Iterable[int]) -> int:
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        with self._lock:
            placeholders = ",".join("?" for _ in ids)
            cursor = self._conn.execute(
                f"DELETE FROM pending_topics WHERE id IN ({placeholders})", ids
            )
            self._conn.commit()
            return cursor.rowcount

    def count_topics(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM pending_topics").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
