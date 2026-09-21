"""Writing into the shared ``memories`` table — the bot's insert, ported.

Three rules this file exists to keep:

* **One dialect.** The bot inserts ``memory_type``, ``content``, ``priority``, ``chat_id``,
  ``role`` and stamps ``last_accessed``; the embedding lands afterwards in its own update.
  Writing anything else into the same table means rows one side can find and the other
  cannot.
* **Never truncate.** The column holds 2000 characters and a turn can run longer. A long
  entry is split into ordered parts, each marked ``[часть i/N]``, because the project's
  standing decision is that the original is never destroyed — a summary that replaces it is
  where invented detail comes from.
* **Do not repeat yourself.** The same sentence saved twice crowds recall with its own echo,
  so an identical entry this agent already wrote is skipped. Entries written by the bot or
  by anyone else are left alone: a duplicate across people is two facts, not one.
"""

from __future__ import annotations

import re
from typing import Sequence

# The database enforces 2000; leave room for the part marker that chunking adds.
MAX_CONTENT = 2000
CHUNK_BUDGET = 1900

# Tag written into chat_id. It records where a row came from and, just as importantly, makes
# every row this plugin ever wrote removable with one condition.
SOURCE_PREFIX = "hermes:"

INSERT_SQL = """
INSERT INTO memories (memory_type, content, priority, chat_id, role, status, last_accessed)
VALUES (%(memory_type)s, %(content)s, %(priority)s, %(chat_id)s, %(role)s, 'active', now())
RETURNING id
"""

# Dedup is scoped to this agent's own rows on purpose — see the module docstring.
DUPLICATE_SQL = """
SELECT id FROM memories
WHERE content = %(content)s AND status = 'active' AND chat_id LIKE %(source)s
LIMIT 1
"""

EMBED_SQL = "UPDATE memories SET embedding = %(vec)s::vector WHERE id = %(id)s"

# Removing a memory archives it instead of deleting: the row stops surfacing in recall while
# staying in the store, so a mistaken removal costs a query, not a fact.
ARCHIVE_SQL = """
UPDATE memories SET status = 'archived'
WHERE content = %(content)s AND status = 'active' AND chat_id LIKE %(source)s
RETURNING id
"""

# What the shared store accepts. The live table has no CHECK constraint left on this column
# (measured 2026-09-21), so the discipline is here or nowhere.
MEMORY_TYPES = (
    "fact", "decision", "preference", "event", "context",
    "lesson", "rule", "project", "contact", "dialogue",
)
PRIORITIES = ("P1", "P2", "P3")
ROLES = ("assistant", "user", "system")

_SPLIT_AT = re.compile(r"(?<=[.!?…])\s+|\n+")


def source_tag(session_id: str = "") -> str:
    """Provenance for ``chat_id``: this agent, and which of its sessions."""
    return SOURCE_PREFIX + (session_id or "session")


def normalise_type(value: str | None, default: str = "fact") -> str:
    return value if value in MEMORY_TYPES else default


def normalise_priority(value: str | None, default: str = "P2") -> str:
    return value if value in PRIORITIES else default


def normalise_role(value: str | None, default: str = "assistant") -> str:
    return value if value in ROLES else default


def chunk_content(content: str, budget: int = CHUNK_BUDGET) -> list[str]:
    """Split a long entry into ordered, marked parts; short entries come back untouched.

    Splitting prefers sentence and line boundaries so a part reads as something, and falls
    back to a hard cut only for an unbroken wall of characters.
    """
    text = (content or "").strip()
    if not text:
        return []
    if len(text) <= MAX_CONTENT:
        return [text]

    pieces: list[str] = []
    current = ""
    for piece in _SPLIT_AT.split(text):
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > budget:  # one sentence longer than the budget: cut it
            pieces.append((current + " " + piece[:budget]).strip() if current else piece[:budget])
            piece, current = piece[budget:].strip(), ""
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= budget:
            current = f"{current} {piece}"
        else:
            pieces.append(current)
            current = piece
    if current:
        pieces.append(current)

    total = len(pieces)
    return [f"[часть {index}/{total}] {piece}" for index, piece in enumerate(pieces, start=1)]


def insert_params(content: str, *, memory_type: str, priority: str, role: str, chat_id: str) -> dict:
    return {
        "content": content,
        "memory_type": normalise_type(memory_type),
        "priority": normalise_priority(priority),
        "role": normalise_role(role),
        "chat_id": chat_id,
    }


def describe_saved(chunks: Sequence[str], memory_type: str) -> str:
    """What the agent is told after a write — short, and honest about splitting."""
    if not chunks:
        return "Пустую запись сохранять нечего."
    if len(chunks) == 1:
        return f"Запомнил ({normalise_type(memory_type)})."
    return f"Запомнил ({normalise_type(memory_type)}), запись длинная — сохранил {len(chunks)} частями."
