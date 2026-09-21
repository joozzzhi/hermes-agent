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
from typing import Any

from agent.memory_provider import MemoryProvider, RecallStatus, spawn_context_thread
from agent.secret_scope import get_secret
from tools.registry import tool_error

from .search import (
    FTS_SQL,
    VECTOR_SQL,
    combine_rrf,
    format_block,
    row_to_memory,
    to_vector_literal,
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

    def _fetch(self, sql: str, params: dict) -> list[dict]:
        """Run one statement, reconnecting once if the held connection went stale."""
        for attempt in (1, 2):
            try:
                if self._conn is None or self._conn.closed:
                    self._conn = self._connect()
                with self._conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [row_to_memory(row) for row in cur.fetchall()]
            except Exception:
                try:
                    if self._conn is not None:
                        self._conn.close()
                finally:
                    self._conn = None
                if attempt == 2:
                    raise
        return []

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # -- embeddings ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float] | None:
        """Vector for the question, or None — a missing vector costs ranking, not recall."""
        if not self._api_key:
            return None
        try:
            import requests

            response = requests.post(
                EMBEDDING_URL,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={"model": EMBEDDING_MODEL, "input": text},
                timeout=EMBEDDING_TIMEOUT,
            )
            response.raise_for_status()
            vector = response.json()["data"][0]["embedding"]
            return vector if isinstance(vector, list) and vector else None
        except Exception as exc:
            logger.debug("Aida memory: embedding unavailable, searching by words only (%s)", exc)
            return None

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


class AidaMemoryProvider(MemoryProvider):
    """Recall from the operator's shared memory, visibly, on every non-trivial turn."""

    def __init__(self) -> None:
        self._store: _Store | None = None
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

    def system_prompt_block(self) -> str:
        return (
            "# Long-term memory\n"
            "You have one long-term memory, shared with every surface the operator talks to you on — "
            "this terminal, Telegram, coding sessions. Everything recalled from it is YOUR OWN past "
            "with him, including conversations from before this machine existed: speak of it in the "
            "first person, never as another assistant's transcript.\n"
            "Recalled entries arrive automatically before your reply. Use `aida_recall` when you need "
            "to look something up on purpose. Memory is quoted, not guessed — if it is not there, say so."
        )

    # -- recall lifecycle --------------------------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Recall in the background after a turn; :meth:`prefetch` hands it over on the next."""
        if self._store is None:
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
        """What the LAST prefetch injected. The operator asked to always see that memory was
        consulted, including when it came back with nothing — so a completed recall with no
        hits still reports, and only "no recall happened at all" stays silent."""
        if self._last_count is None:
            return None
        return RecallStatus(provider_label="Память", count=self._last_count)

    # -- tools -------------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [RECALL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if tool_name != RECALL_SCHEMA["name"]:
            return tool_error(f"Unknown tool: {tool_name}")
        if self._store is None:
            return tool_error("Shared memory is not configured")
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

    def shutdown(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._store is not None:
            self._store.close()


def register(ctx) -> None:
    """Register the shared memory as a memory provider plugin."""
    ctx.register_memory_provider(AidaMemoryProvider())
