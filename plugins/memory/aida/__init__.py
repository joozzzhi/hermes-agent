"""Aida memory plugin — the operator's shared long-term memory, as a MemoryProvider.

One store, several faces. The operator's Telegram bot and his coding sessions already read
and write a single Postgres table (Supabase, table ``memories``). This provider puts Hermes
on that same table, so what was said in Telegram in July is available here without being
retold — and, by the operator's decision, it is presented as Hermes's OWN past, not as a
quotation from another assistant.

This is step one of three: it RECALLS and writes nothing. Deliberate writes, mirrored
memory-tool writes and the nightly verbatim shipment come in the later steps, so that a
mistake here can only ever fail to remember something — never damage the store that the
bot depends on.

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

from . import recall_signal
from .queue_db import TurnQueue
from .search import (
    FTS_SQL,
    VECTOR_SQL,
    combine_rrf,
    format_block,
    row_to_memory,
    to_vector_literal,
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
    source_tag,
)

logger = logging.getLogger(__name__)

# The same embedding model the bot uses. A different model means a different vector space:
# rows written by one side would be unreachable by the other's search while looking present
# in the table — the worst kind of memory loss, the invisible one.
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_TIMEOUT = 12.0

# A query must never outlive the turn it was meant for. The manager abandons a stuck
# provider thread rather than killing it, so the ceiling lives here.
CONNECT_TIMEOUT = 10
STATEMENT_TIMEOUT_MS = 8000

# After the store fails, stop dialling it on every single turn.
BACKOFF_SECONDS = 60.0

DEFAULT_LIMIT = 8
MAX_LIMIT = 20

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

    def _embed_all(self, texts: list[str]) -> list[list[float]] | None:
        """Vectors for one or more texts, or None when the service is not answering at all."""
        if not self._api_key or not texts:
            return None
        try:
            import requests

            response = requests.post(
                EMBEDDING_URL,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={"model": EMBEDDING_MODEL, "input": texts},
                timeout=EMBEDDING_TIMEOUT * max(1, len(texts) // 4),
            )
            response.raise_for_status()
            data = response.json().get("data") or []
            vectors = [item.get("embedding") or [] for item in data]
            return vectors if len(vectors) == len(texts) else None
        except Exception as exc:
            logger.debug("Aida memory: embedding unavailable, searching by words only (%s)", exc)
            return None

    def _embed(self, text: str) -> list[float] | None:
        """Vector for the question, or None — a missing vector costs ranking, not recall."""
        vectors = self._embed_all([text])
        return vectors[0] if vectors and vectors[0] else None

    # -- recall ----------------------------------------------------------------------

    def recall(self, query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
        """Hybrid recall. Returns [] when the store is unreachable — never raises upward."""
        query = (query or "").strip()
        if not query:
            return []
        if time.monotonic() < self._blocked_until:
            return []
        vector = self._embed(query)
        with self._lock:
            try:
                fts_hits = self._fetch(FTS_SQL, {"query": query, "limit": limit})
                vector_hits: list[dict] = []
                if vector is not None:
                    vector_hits = self._fetch(
                        VECTOR_SQL, {"vec": to_vector_literal(vector), "limit": limit * 2}
                    )
                self._blocked_until = 0.0
            except Exception as exc:
                self._blocked_until = time.monotonic() + BACKOFF_SECONDS
                logger.warning("Aida memory unreachable, answering without it: %s", exc)
                return []
        return combine_rrf(vector_hits, fts_hits)[:limit]

    # -- writing ---------------------------------------------------------------------

    def save(self, content: str, *, memory_type: str = "fact", priority: str = "P2",
             role: str = "assistant", chat_id: str = "", embed: bool = True) -> list[int]:
        """Write one memory, split into parts if it is too long. Returns the row ids written.

        Raises nothing upward on a database failure — the caller is a turn in progress, and a
        memory that could not be written is worth a warning, not a broken conversation.
        """
        chunks = chunk_content(content)
        if not chunks:
            return []
        written: list[int] = []
        with self._lock:
            try:
                for chunk in chunks:
                    duplicate = self._execute(
                        DUPLICATE_SQL, {"content": chunk, "source": SOURCE_PREFIX + "%"}
                    )
                    if duplicate:
                        continue
                    rows = self._execute(
                        INSERT_SQL,
                        insert_params(chunk, memory_type=memory_type, priority=priority,
                                      role=role, chat_id=chat_id or source_tag()),
                    )
                    if rows:
                        written.append(int(rows[0][0]))
            except Exception as exc:
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
        self._pending: str = ""
        self._pending_count: int = 0
        self._last_count: int | None = None
        self._thread: threading.Thread | None = None

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
        try:
            self._queue = TurnQueue(home / "aida_queue.db")
        except Exception as exc:
            logger.warning("Aida memory: the local queue is unavailable, turns will not be kept: %s", exc)
            self._queue = None

    def system_prompt_block(self) -> str:
        return (
            "# Long-term memory\n"
            "You have one long-term memory, shared with every surface the operator talks to you on — "
            "this terminal, Telegram, coding sessions. Everything recalled from it is YOUR OWN past "
            "with him, including conversations from before this machine existed: speak of it in the "
            "first person, never as another assistant's transcript.\n"
            "Recalled entries arrive automatically before your reply. Use `aida_recall` when you need "
            "to look something up on purpose. Memory is quoted, not guessed — if it is not there, say so.\n"
            "Use `aida_remember` for something that must outlive this conversation — a decision, a "
            "preference, an arrangement, a fact about him. One thought per call, written so it still "
            "makes sense in a month. The conversation itself is kept without you doing anything."
        )

    # -- recall lifecycle --------------------------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Recall in the background after a turn; :meth:`prefetch` hands it over on the next."""
        if self._store is None:
            return
        if recall_signal.engine_is_assembling():
            # The context engine assembles the window from this same store, and it does so
            # for the question being asked rather than the one before it. Searching again
            # here would put a second copy of the same memories into the request.
            return
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.debug("Aida memory: previous recall still running, skipping this one")
                return
        self._thread = spawn_context_thread(self._recall_into_cache, args=(query,), name="aida-recall")
        self._thread.start()

    def _recall_into_cache(self, query: str) -> None:
        memories = self._store.recall(query) if self._store else []
        with self._lock:
            self._pending = format_block(memories)
            self._pending_count = len(memories)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._lock:
            block, count = self._pending, self._pending_count
            self._pending, self._pending_count = "", 0
            self._last_count = count
        return block

    def recall_status(self) -> RecallStatus | None:
        """What the LAST recall injected. The operator asked to always see that memory was
        consulted, including when it came back with nothing — so a completed recall with no
        hits still reports, and only "no recall happened at all" stays silent.

        When the context engine is the one assembling, the count comes from it: the search
        happens once, and the indicator still tells the operator what was raised."""
        count = recall_signal.last_count() if recall_signal.engine_is_assembling() else self._last_count
        if count is None:
            return None
        return RecallStatus(provider_label="Память", count=count)

    # -- writing -----------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", **kwargs) -> None:
        """Keep the turn locally; the nightly job carries it to the shared store.

        Nothing here talks to the database — an answer must not wait on the network, and a
        day's conversation is worth one batch rather than two round trips per reply.
        """
        if self._queue is None or not self._writes_enabled:
            return
        session = session_id or self._session_id
        try:
            self._queue.enqueue(session, "user", user_content)
            self._queue.enqueue(session, "assistant", assistant_content)
        except Exception as exc:
            logger.debug("Aida memory: could not queue the turn (%s)", exc)

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
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._store is not None:
            self._store.close()
        if self._queue is not None:
            self._queue.close()


def register(ctx) -> None:
    """Register the shared memory as a memory provider plugin."""
    ctx.register_memory_provider(AidaMemoryProvider())
