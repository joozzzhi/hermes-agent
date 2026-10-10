"""Aida memory plugin — the operator's shared long-term memory, as a MemoryProvider.

One store, several faces. The operator's Telegram bot and his coding sessions already read
and write a single Postgres table (Supabase, table ``memories``). This provider puts Hermes
on that same table, so what was said in Telegram in July is available here without being
retold — and, by the operator's decision, it is presented as Hermes's OWN past, not as a
quotation from another assistant.

Three tiers, the way the engine on the server had them. The CORE (priority P1, written on
purpose) always stands in the system prompt — the context engine puts it there. FACTS and
lines of past DIALOGUE are found by the question of the turn and handed to the host through
``prefetch()``: the host stamps them into that turn's user message for good, so the request
only ever grows at its end and the provider's prompt cache survives from turn to turn.

A finished turn goes into the store at once, verbatim. What needs a model — pulling lasting
facts out of the conversation — waits until the operator has been silent for an hour
(``idle.py``): by then the provider's cache is cold anyway, and nothing is disturbed.

Configuration: ``SUPABASE_DB_URL`` (required, scoped secret) and ``OPENROUTER_API_KEY``
(optional — enables searching by meaning as well as by word).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider, RecallStatus, spawn_context_thread
from agent.secret_scope import get_secret
from tools.registry import tool_error

from . import idle
from .queue_db import TurnQueue
from .search import (
    CORE_SQL,
    DIALOGUE_CAP,
    DISTILLER_SOURCE,
    above_vector_floor,
    combine_rrf,
    format_block,
    format_recall,
    fts_sql,
    pick_facts,
    row_to_memory,
    same_fact,
    to_vector_literal,
    vector_sql,
)
from .write import (
    ARCHIVE_SQL,
    DUPLICATE_SQL,
    EMBED_SQL,
    INSERT_SQL,
    MEMORY_TYPES,
    PRIORITIES,
    SOURCE_PREFIX,
    chunk_content,
    describe_saved,
    insert_params,
    normalise_type,
    scrub,
    source_tag,
)

logger = logging.getLogger(__name__)

# The same embedding model the bot uses. A different model means a different vector space:
# rows written by one side would be unreachable by the other's search while looking present
# in the table — the worst kind of memory loss, the invisible one.
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_TIMEOUT = 12.0
# The question of a turn is embedded while the operator waits for the answer to start. If the
# service is slow, the words of the question still find what they can.
QUERY_EMBEDDING_TIMEOUT = 4.0

# A query must never outlive the turn it was meant for. The manager abandons a stuck
# provider thread rather than killing it, so the ceiling lives here.
CONNECT_TIMEOUT = 10
STATEMENT_TIMEOUT_MS = 8000

# After the store fails, stop dialling it on every single turn.
BACKOFF_SECONDS = 60.0

DEFAULT_LIMIT = 8
MAX_LIMIT = 20

# Shorter than this is "ок", "ага", "поправляй" (the bot's threshold, measured on its own
# chats). Such a reply asks nothing, yet its two letters match half the store.
MIN_QUESTION_LENGTH = 12

# A deliberate lookup may show a stored entry whole (the column caps it at 2000 characters);
# the automatic per-turn block is trimmed harder — see ENTRY_BUDGET in search.py.
TOOL_ENTRY_BUDGET = 2000
TOOL_BLOCK_BUDGET = 12000

REMEMBER_SCHEMA = {
    "name": "aida_remember",
    "description": (
        "Save one thought into your long-term memory, so it outlives this conversation and is "
        "there on every surface you talk to the operator on. One thought per call, phrased so "
        "it still makes sense in a month with no other context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The thought itself, self-contained."},
            "memory_type": {
                "type": "string",
                "enum": list(MEMORY_TYPES[:-1]),  # a deliberate write is never a raw dialogue line
                "description": "What kind of memory this is (default: fact).",
            },
            "priority": {
                "type": "string",
                "enum": list(PRIORITIES),
                "description": "P1 stays, P3 fades (default: P2).",
            },
        },
        "required": ["content"],
    },
}

def _turn_called_remember(messages: list[dict[str, Any]] | None) -> bool:
    """Did this turn's transcript include a call to ``aida_remember``?

    ``messages`` is the turn's full projected transcript (OpenAI tool-call shape), not just
    the final text — a call earlier in the turn is exactly the case this guards against.
    Anything unexpected in the shape is treated as "no call": staying silent here only
    costs a duplicate, never a lost fact, and the nightly distiller is not the place to be
    strict about malformed input.
    """
    if not messages:
        return False
    # The host hands over the conversation SO FAR, not the turn. Reading all of it meant that
    # one ``aida_remember`` anywhere in a long session silenced the keeping of every later
    # turn — in a session that never ends, that is everything after the first week.
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user" and message.get("tool_call_id") is None:
            start = index
            break
    for message in messages[start:]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            name = (call or {}).get("function", {}).get("name") if isinstance(call, dict) else None
            if name == REMEMBER_SCHEMA["name"]:
                return True
    return False


RECALL_SCHEMA = {
    "name": "aida_recall",
    "description": (
        "Search your own long-term memory — everything the operator has told you before, "
        "including conversations held in Telegram and in coding sessions. Use it before "
        "answering anything about past decisions, preferences, plans or events."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in natural language."},
            "top_k": {"type": "integer", "description": f"Max results (default {DEFAULT_LIMIT}, max {MAX_LIMIT})."},
        },
        "required": ["query"],
    },
}


class _Store:
    """A reconnecting connection to the shared memory, with one query shape: recall.

    Deliberately one connection behind a lock rather than a pool: recall runs on a single
    background prefetch thread plus the occasional tool call, and the database sits behind a
    transaction pooler that a client-side pool would only fight with.
    """

    def __init__(self, url: str, api_key: str = ""):
        self._url = url
        self._api_key = api_key
        self._conn: Any = None
        self._lock = threading.Lock()
        self._blocked_until = 0.0

    # -- connection ------------------------------------------------------------------

    def _connect(self) -> Any:
        import psycopg

        # prepare_threshold=None: the connection string points at the transaction pooler,
        # where a server-side prepared statement from an earlier transaction is not there
        # for the next one. Left on, the third identical query starts failing.
        return psycopg.connect(
            self._url,
            connect_timeout=CONNECT_TIMEOUT,
            autocommit=True,
            prepare_threshold=None,
            options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
        )

    def _execute(self, sql: str, params: dict) -> list[tuple]:
        """Run one statement, reconnecting once if the held connection went stale."""
        for attempt in (1, 2):
            try:
                if self._conn is None or self._conn.closed:
                    self._conn = self._connect()
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                    return list(cur.fetchall()) if cur.description else []
            except Exception:
                try:
                    if self._conn is not None:
                        self._conn.close()
                finally:
                    self._conn = None
                if attempt == 2:
                    raise
        return []

    def _fetch(self, sql: str, params: dict) -> list[dict]:
        return [row_to_memory(row) for row in self._execute(sql, params)]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # -- embeddings ------------------------------------------------------------------

    def _embed_all(self, texts: list[str], timeout: float | None = None) -> list[list[float]] | None:
        """Vectors for one or more texts, or None when the service is not answering at all."""
        if not self._api_key or not texts:
            return None
        try:
            import requests

            response = requests.post(
                EMBEDDING_URL,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={"model": EMBEDDING_MODEL, "input": [scrub(text) for text in texts]},
                timeout=timeout or EMBEDDING_TIMEOUT * max(1, len(texts) // 4),
            )
            response.raise_for_status()
            data = response.json().get("data") or []
            vectors = [item.get("embedding") or [] for item in data]
            return vectors if len(vectors) == len(texts) else None
        except Exception as exc:
            logger.debug("Aida memory: embedding unavailable, searching by words only (%s)", exc)
            return None

    def _embed(self, text: str, timeout: float | None = None) -> list[float] | None:
        """Vector for the question, or None — a missing vector costs ranking, not recall."""
        vectors = self._embed_all([text], timeout=timeout)
        return vectors[0] if vectors and vectors[0] else None

    # -- recall ----------------------------------------------------------------------

    def _search(self, kind: str, query: str, vector: list[float] | None, limit: int) -> list[dict]:
        """One tier, both halves, fused. Call with the lock held; raises on a database failure."""
        fts_hits = self._fetch(fts_sql(kind), {"query": query, "limit": limit})
        vector_hits: list[dict] = []
        if vector is not None:
            vector_hits = above_vector_floor(
                self._fetch(vector_sql(kind), {"vec": to_vector_literal(vector), "limit": limit * 2})
            )
        return combine_rrf(vector_hits, fts_hits)

    def recall(self, query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
        """A deliberate lookup over everything. Returns [] when the store is unreachable."""
        query = (query or "").strip()
        if not query:
            return []
        if time.monotonic() < self._blocked_until:
            return []
        vector = self._embed(query)
        with self._lock:
            try:
                found = self._search("all", query, vector, limit)
                self._blocked_until = 0.0
            except Exception as exc:
                self._blocked_until = time.monotonic() + BACKOFF_SECONDS
                logger.warning("Aida memory unreachable, answering without it: %s", exc)
                return []
        return found[:limit]

    def recall_tiers(self, query: str) -> tuple[list[dict], list[dict]] | None:
        """What one question raises: facts and lines of past dialogue, searched apart.

        None means the store could not be asked at all — the caller must be able to tell
        "nothing was found" from "nobody looked".
        """
        query = (query or "").strip()
        if not query:
            return [], []
        if time.monotonic() < self._blocked_until:
            return None
        vector = self._embed(query, timeout=QUERY_EMBEDDING_TIMEOUT)
        with self._lock:
            try:
                facts = self._search("facts", query, vector, 20)
                dialogue = self._search("dialogue", query, vector, DIALOGUE_CAP * 2)
                self._blocked_until = 0.0
            except Exception as exc:
                self._blocked_until = time.monotonic() + BACKOFF_SECONDS
                logger.warning("Aida memory unreachable, answering without it: %s", exc)
                return None
        return pick_facts(facts), dialogue[:DIALOGUE_CAP]

    # -- keeping the store from becoming a warehouse --------------------------------------

    def knows(self, content: str) -> bool:
        """Is this fact already in memory, in these or other words? A database failure answers
        "no": a possible duplicate is cheaper than a fact dropped because the store hiccuped."""
        content = (content or "").strip()
        if not content or time.monotonic() < self._blocked_until:
            return False
        with self._lock:
            try:
                candidates = self._fetch(fts_sql("known"), {"query": content, "limit": 5})
            except Exception as exc:
                logger.debug("Aida memory: could not check for a known fact (%s)", exc)
                return False
        return any(same_fact(content, item.get("content")) for item in candidates)

    def mark_used(self, ids: list[int]) -> None:
        """A memory was put in front of the model. This is what a fact on probation is judged by."""
        if ids:
            self._execute_guarded(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = now() "
                "WHERE id = ANY(%(ids)s)", {"ids": [int(i) for i in ids]},
            )

    def settle_probation(self, days: float) -> tuple[int, int]:
        """Facts the distiller wrote start on probation (P3). One that has been recalled since
        is kept as an ordinary fact; one that nobody needed in ``days`` stops surfacing. The row
        stays in the table either way. Returns (kept, archived)."""
        source = {"source": DISTILLER_SOURCE}
        kept = self._execute_guarded(
            "UPDATE memories SET priority = 'P2' WHERE chat_id = %(source)s AND priority = 'P3' "
            "AND status = 'active' AND memory_type <> 'dialogue' AND access_count > 0 RETURNING id", source)
        archived = self._execute_guarded(
            "UPDATE memories SET status = 'archived' WHERE chat_id = %(source)s AND priority = 'P3' "
            "AND status = 'active' AND memory_type <> 'dialogue' AND access_count = 0 "
            "AND created_at < now() - make_interval(secs => %(seconds)s) RETURNING id",
            {**source, "seconds": float(days) * 86400.0})
        return len(kept), len(archived)

    def core_rows(self) -> list[dict] | None:
        """The core: every P1 entry written on purpose. None when the store did not answer."""
        if time.monotonic() < self._blocked_until:
            return None
        with self._lock:
            try:
                rows = self._execute(CORE_SQL, {})
                self._blocked_until = 0.0
            except Exception as exc:
                self._blocked_until = time.monotonic() + BACKOFF_SECONDS
                logger.warning("Aida memory: the core could not be read: %s", exc)
                return None
        return [{"content": row[0], "memory_type": row[1], "created_at": row[2]} for row in rows]

    # -- writing ---------------------------------------------------------------------

    def save(self, content: str, *, memory_type: str = "fact", priority: str = "P2",
             role: str = "assistant", chat_id: str = "", embed: bool = True,
             created_at: str | None = None, dedupe: bool = True) -> list[int]:
        """Write one memory, split into parts if it is too long. Returns the row ids written.

        Raises nothing upward on a database failure — the caller is a turn in progress, and a
        memory that could not be written is worth a warning, not a broken conversation.
        ``dedupe=False`` is for lines of conversation: "да" said twice was said twice.
        """
        chunks = chunk_content(scrub(content))
        if not chunks:
            return []
        if time.monotonic() < self._blocked_until:
            return []
        written: list[int] = []
        with self._lock:
            try:
                for chunk in chunks:
                    if dedupe and self._execute(
                        DUPLICATE_SQL, {"content": chunk, "source": SOURCE_PREFIX + "%"}
                    ):
                        continue
                    rows = self._execute(
                        INSERT_SQL,
                        insert_params(chunk, memory_type=memory_type, priority=priority, role=role,
                                      chat_id=chat_id or source_tag(), created_at=created_at),
                    )
                    if rows:
                        written.append(int(rows[0][0]))
            except Exception as exc:
                # The back-off is what lets a caller tell "the store refused" from "there was
                # nothing to write" (see alive()); without it a failed turn looked delivered.
                self._blocked_until = time.monotonic() + BACKOFF_SECONDS
                logger.warning("Aida memory: could not write, the thought stays unsaved: %s", exc)
                return []
        if embed and written:
            spawn_context_thread(self._embed_rows, args=(written,), name="aida-embed").start()
        return written

    def _embed_rows(self, ids: list[int]) -> None:
        """Fill in the vector after the row exists. A missing vector costs ranking, not the memory."""
        for row_id in ids:
            content_rows = self._execute_guarded("SELECT content FROM memories WHERE id = %(id)s", {"id": row_id})
            if not content_rows:
                continue
            vector = self._embed(str(content_rows[0][0]))
            if vector is None:
                return  # no embeddings available at all — stop rather than retry per row
            self._execute_guarded(EMBED_SQL, {"vec": to_vector_literal(vector), "id": row_id})

    def _execute_guarded(self, sql: str, params: dict) -> list[tuple]:
        with self._lock:
            try:
                return self._execute(sql, params)
            except Exception as exc:
                logger.debug("Aida memory: background statement failed (%s)", exc)
                return []

    # -- what the nightly job needs ---------------------------------------------------

    def alive(self) -> bool:
        """False while the store is in its back-off after a failure."""
        return time.monotonic() >= self._blocked_until

    # A blank or single-character memory has no meaning to embed, and the service refuses it.
    # Left in the queue it would be retried every night forever and, worse, take the rest of
    # the batch down with it — measured on this store: 68 rows stopped the pass dead.
    _EMBEDDABLE = "embedding IS NULL AND status = 'active' AND char_length(btrim(content)) >= 2"

    def rows_without_embedding(self, limit: int) -> list[tuple]:
        """Oldest memories that cannot be found by meaning yet."""
        return self._execute_guarded(
            f"SELECT id, content FROM memories WHERE {self._EMBEDDABLE} ORDER BY id LIMIT %(limit)s",
            {"limit": limit},
        )

    def count_without_embedding(self) -> int:
        rows = self._execute_guarded(f"SELECT count(*) FROM memories WHERE {self._EMBEDDABLE}", {})
        return int(rows[0][0]) if rows else 0

    def set_embedding(self, row_id: int, vector: list[float]) -> bool:
        return bool(self._execute_guarded(
            EMBED_SQL + " RETURNING id", {"vec": to_vector_literal(vector), "id": row_id}
        ))

    def embed_many(self, texts: list[str]) -> list[list[float]] | None:
        """Vectors for several texts at once, or None when embeddings are unavailable at all.

        The distinction matters to the caller: an empty vector for one text is one memory that
        stays word-searchable, while None means the whole pass should stop rather than spend
        the night retrying a service that is not answering.
        """
        if not texts:
            return []
        batch = self._embed_all(texts)
        if batch is not None:
            return batch
        # The provider may have refused ONE of the texts rather than the work. One at a time
        # is slower but keeps a single unembeddable memory from costing the whole night.
        vectors: list[list[float]] = []
        for text in texts:
            single = self._embed(text)
            vectors.append(single or [])
        return vectors if any(vectors) else None

    def archive(self, content: str) -> int:
        """Stop a memory this agent wrote from surfacing, without deleting it."""
        content = (content or "").strip()
        if not content:
            return 0
        rows = self._execute_guarded(ARCHIVE_SQL, {"content": content, "source": SOURCE_PREFIX + "%"})
        return len(rows)


class AidaMemoryProvider(MemoryProvider):
    """Recall from the operator's shared memory, visibly, on every non-trivial turn."""

    def __init__(self) -> None:
        self._store: _Store | None = None
        self._queue: TurnQueue | None = None
        self._session_id: str = ""
        self._writes_enabled: bool = True
        self._lock = threading.Lock()
        self._last_count: int | None = None
        self._home: Path | None = None

    @property
    def name(self) -> str:
        return "aida"

    def is_available(self) -> bool:
        return bool(get_secret("SUPABASE_DB_URL", ""))

    def unavailable_reason(self) -> str:
        return "SUPABASE_DB_URL is not set — run `hermes memory setup` and paste the shared memory's connection string."

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "db_url",
                "description": "Connection string of the shared memory database (Supabase → Project Settings → Database → URI)",
                "secret": True,
                "required": True,
                "env_var": "SUPABASE_DB_URL",
            },
            {
                "key": "openrouter_key",
                "description": "OpenRouter key — lets memory be searched by meaning, not only by matching words (optional)",
                "secret": True,
                "required": False,
                "env_var": "OPENROUTER_API_KEY",
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._store = _Store(get_secret("SUPABASE_DB_URL", "") or "", get_secret("OPENROUTER_API_KEY", "") or "")
        self._session_id = session_id
        # Only the agent the operator is actually talking to writes. A subagent, a cron run or
        # a flush would otherwise fill the shared memory with work nobody said out loud.
        self._writes_enabled = str(kwargs.get("agent_context", "primary") or "primary") == "primary"
        home = Path(str(kwargs.get("hermes_home") or "."))
        self._home = home
        try:
            self._queue = TurnQueue(home / "aida_queue.db")
        except Exception as exc:
            logger.warning("Aida memory: the local queue is unavailable, turns will not be distilled: %s", exc)
            self._queue = None
        if self._queue is not None and self._writes_enabled:
            # The idle timer lives in the process. A conversation left in the queue by a
            # process that was restarted mid-count is picked up here.
            idle.check_on_start(home, self._queue)

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        """The host renumbers the session without restarting us (compression, /new, /resume)."""
        if new_session_id:
            self._session_id = new_session_id

    def system_prompt_block(self) -> str:
        return (
            "# Long-term memory\n"
            "You have one long-term memory, shared with every surface the operator talks to you on — "
            "this terminal, Telegram, coding sessions. Everything recalled from it is YOUR OWN past "
            "with him, including conversations from before this machine existed: speak of it in the "
            "first person, never as another assistant's transcript.\n"
            "What is relevant to the operator's message arrives with it, inside <memory-context>: facts "
            "whole, lines of old conversation as dated quotes cut to their opening. Use `aida_recall` to "
            "read a quote in full or to look something up on purpose — and always before answering "
            "about past decisions, plans or events when nothing relevant arrived. Memory is quoted, "
            "not guessed — if it is not there, say so.\n"
            "Use `aida_remember` for something that must outlive this conversation — a decision, a "
            "preference, an arrangement, a fact about him. One thought per call, written so it still "
            "makes sense in a month. The conversation itself is kept without you doing anything."
        )

    # -- recall lifecycle --------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Search for THIS question, now. The host stamps what comes back into the turn's user
        message and replays those bytes on every later request, so it is found once per turn
        and never moves again — which is what keeps the provider's prompt cache alive.

        The base class asks for a cached answer prepared after the previous turn. That answers
        the previous question; the lookup here takes a fraction of a second and the host bounds
        it with its own timeout.
        """
        self._last_count = None
        if self._store is None or len((query or "").strip()) < MIN_QUESTION_LENGTH:
            return ""
        tiers = self._store.recall_tiers(query)
        if tiers is None:
            return ""
        block, shown = format_recall(*tiers)
        self._last_count = shown
        if block:
            # Off the reply's path: the count only matters to the next tidy-up.
            used = [item["id"] for item in tiers[0]]
            spawn_context_thread(self._store.mark_used, args=(used,), name="aida-used").start()
        return block

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Nothing to prepare in advance — see :meth:`prefetch`."""

    def recall_status(self) -> RecallStatus | None:
        """What the LAST recall put into the turn, for the indicator above the reply."""
        if not self._last_count:
            return None
        return RecallStatus(provider_label="Память", count=self._last_count)

    # -- writing -----------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: list[dict[str, Any]] | None = None, **kwargs) -> None:
        """Keep the turn: verbatim into the shared store now, and into the local queue to be
        distilled when the operator falls silent.

        The host calls this on its background worker after the reply is out, so the round
        trip to the database costs the operator nothing. Writing at once is what makes a
        conversation findable an hour later instead of the next morning — and what survives
        a restart, a /new or a tail that has moved on. If the store is down the turn waits in
        the queue unshipped and is delivered by the next pass with its own time on it.

        A turn where the agent already called ``aida_remember`` is still kept verbatim, but
        not distilled: the fact is in the store, phrased by the agent, and a second phrasing
        of it would sail past the exact-text dedup and double up in recall.
        """
        if not self._writes_enabled:
            return
        session = session_id or self._session_id
        distill = not _turn_called_remember(messages)
        for role, content in (("user", user_content), ("assistant", assistant_content)):
            shipped = False
            if self._store is not None:
                shipped = bool(self._store.save(
                    content, memory_type="dialogue", priority="P3", role=role,
                    chat_id=source_tag(session), dedupe=False,
                ))
            if self._queue is None:
                continue
            try:
                self._queue.enqueue(session, role, content, shipped=shipped, distill=distill)
            except Exception as exc:
                logger.warning("Aida memory: could not queue the turn for distilling (%s)", exc)
        if self._queue is not None and self._home is not None:
            idle.touch(self._home)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: dict[str, Any] | None = None) -> None:
        """Mirror what the agent writes into its own notes, so both sides hold the same thing.

        A note the agent curated about the operator is a deliberate, lasting statement — it
        goes in at the highest priority. Removing a note stops it surfacing here too, but the
        row stays in the store: a mistaken removal costs a query, never a fact.
        """
        if self._store is None or not self._writes_enabled:
            return
        memory_type = "preference" if target == "user" else "fact"
        old_text = str((metadata or {}).get("old_text") or "")
        if action in ("remove", "replace") and old_text:
            self._store.archive(old_text)
        if action in ("add", "replace"):
            self._store.save(content, memory_type=memory_type, priority="P1",
                             role="system", chat_id=source_tag(self._session_id))
        elif action == "remove" and not old_text:
            self._store.archive(content)

    # -- tools -------------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if self._store is None:
            return tool_error("Shared memory is not configured")
        if tool_name == REMEMBER_SCHEMA["name"]:
            return self._handle_remember(args)
        if tool_name != RECALL_SCHEMA["name"]:
            return tool_error(f"Unknown tool: {tool_name}")
        # `or DEFAULT_LIMIT` would be wrong here: a requested 0 is a nonsense count, not an
        # unstated one, and it belongs at the floor rather than back at the default.
        asked = args.get("top_k")
        try:
            limit = DEFAULT_LIMIT if asked is None else max(1, min(int(asked), MAX_LIMIT))
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        memories = self._store.recall(str(args.get("query", "")), limit)
        if not memories:
            return "Ничего не нашлось."
        # A deliberate lookup is where the full text belongs: the per-turn block is trimmed
        # to leave room for the conversation, and this is how the rest is reached.
        return format_block(memories, entry_budget=TOOL_ENTRY_BUDGET, block_budget=TOOL_BLOCK_BUDGET)

    def _handle_remember(self, args: dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("Nothing to remember — the thought is empty")
        if not self._writes_enabled:
            return tool_error("This agent does not write to the shared memory")
        memory_type = normalise_type(args.get("memory_type"))
        written = self._store.save(
            content,
            memory_type=memory_type,
            priority=str(args.get("priority") or "P2"),
            role="assistant",
            chat_id=source_tag(self._session_id),
        )
        if not written:
            # Either it is already in memory, or the store refused it. Both are worth saying
            # plainly: silence here reads as "saved" and the thought would be lost.
            return "Не записал: либо это уже есть в памяти, либо база сейчас недоступна."
        return describe_saved(chunk_content(content), memory_type)

    def shutdown(self) -> None:
        if self._store is not None:
            self._store.close()
        if self._queue is not None:
            self._queue.close()


def register(ctx) -> None:
    """Register the shared memory as a memory provider plugin."""
    ctx.register_memory_provider(AidaMemoryProvider())
