"""Hybrid recall over the shared ``memories`` table — the Telegram bot's search, ported.

The bot (aidabot) and this plugin read ONE table. Two dialects of "what matches this
question" over one store is how a memory becomes unreachable from one side while looking
fine from the other, so the SQL here is a deliberate port of the bot's own
``src/search-query.js`` and ``src/memory.js``, constants included:

* Full text runs over the stored generated column ``search_ru`` (``to_tsvector('russian', …)``),
  so Russian word forms match ("пробежка" finds "на пробежку") and Latin words still work.
* A word of the question is weighted by how RARE it is. Demanding every word made ordinary
  questions find nothing; unweighted ranking put a row matching only "какие" above the row
  that answered the question. Words present in more than ``COMMON_WORD_SHARE`` of rows carry
  no signal and are dropped.
* Only matches within ``MATCH_FLOOR`` of the best match for this question come back. Below
  that the memory merely shares a filler word.
* Vector similarity (pgvector, cosine) runs alongside when an embedding for the question is
  available, and the two ranked lists are fused by Reciprocal Rank Fusion with K=60.

The store holds three tiers, the way the engine on the server had them:

* **core** — priority P1, written on purpose. Always in the system prompt, never searched for.
* **facts** — everything else that is not a line of conversation. Found by the question, and
  an old fact gives way to a fresh one (``DECAY_LAMBDA``).
* **dialogue** — what was actually said. Found by the question and quoted verbatim.

Facts and dialogue are searched SEPARATELY. One ranked list for both let the conversation —
two new rows per turn, forever — push the facts out of their own slot (measured 2026-10-10:
36 of 41 hits on five ordinary questions were dialogue lines).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Sequence

# Every query and the stored column must use the same text-search configuration, or nothing
# matches. Both live in the bot's search-query.js under the same names.
FTS_CONFIG = "russian"
SEARCH_COLUMN = "search_ru"
COMMON_WORD_SHARE = 0.1
MATCH_FLOOR = 0.8
RRF_K = 60

# A vector hit must be about as close as the closest one. Without a floor the nearest-N of a
# question about nothing in particular is still N rows, and they all go into the prompt.
VECTOR_FLOOR = 0.8

# How fast a fact gives way to a newer one, per day of age — the server engine's numbers.
# Dialogue does not decay: an old conversation is exactly what recall is for.
DECAY_LAMBDA = {"P1": 0.05, "P2": 0.05, "P3": 0.15}

# How many facts of each priority one question may raise (the server engine's caps). These are
# ceilings on a list already cut by relevance, not a quota to fill.
FACT_CAPS = {"P2": 10, "P3": 3}
DIALOGUE_CAP = 5

# The rows the live conversation still holds verbatim; recalling them would put a second copy
# of the last half hour into the request (the server engine skipped its 30 newest too).
RECENT_DIALOGUE_SKIP = 30

# How much recalled memory may occupy the turn. A stored entry can be 2000 characters long
# and eight of them would crowd out the conversation actually happening.
ENTRY_BUDGET = 400
BLOCK_BUDGET = 2600
QUOTE_BUDGET = 500
RECALL_BLOCK_BUDGET = 5000

# The distiller used to mark three facts in four as P1, and a core that grows by ten rows a
# night is a feed, not a core. Its rows are treated as ordinary facts wherever tiers matter;
# the core is what was written on purpose.
DISTILLER_SOURCE = "hermes:nightly"
CORE_FILTER = (
    f"m.priority = 'P1' AND m.memory_type <> 'dialogue' "
    f"AND m.chat_id IS DISTINCT FROM '{DISTILLER_SOURCE}'"
)

# 'reasoning' rows are the bot's old thinking-aloud: not facts, not something anyone said.
# They stay reachable through a deliberate ``aida_recall``.
FILTERS = {
    "all": "TRUE",
    # Everything the memory already KNOWS, core included — what a new fact is checked against.
    "known": "m.memory_type NOT IN ('dialogue', 'reasoning')",
    "facts": f"m.memory_type NOT IN ('dialogue', 'reasoning') AND NOT ({CORE_FILTER})",
    "dialogue": (
        "m.memory_type = 'dialogue' AND m.id NOT IN ("
        "SELECT r.id FROM memories r WHERE r.memory_type = 'dialogue' AND r.status = 'active' "
        "AND left(coalesce(r.chat_id, ''), 7) = 'hermes:' "
        f"ORDER BY r.id DESC LIMIT {RECENT_DIALOGUE_SKIP})"
    ),
}

COLUMNS = "m.id, m.content, m.memory_type, m.priority, m.created_at, m.role, m.chat_id"

# quote_literal() around each lexeme: a question can carry quotes, URLs or emoji, and an
# unquoted lexeme would either break the tsquery parse or quietly mean something else.
# Word rarity is measured over the whole store, whatever tier is being searched: a word is
# common or rare in the operator's life, not in one slice of the table.
_FTS_TEMPLATE = f"""
WITH corpus AS (
  SELECT count(*)::numeric AS total FROM memories WHERE status = 'active'
),
question AS (
  SELECT DISTINCT lex
  FROM unnest(tsvector_to_array(to_tsvector('{FTS_CONFIG}', %(query)s))) AS lex
),
weighted AS (
  SELECT question.lex,
         ln((SELECT total FROM corpus) / GREATEST(seen.df, 1)) AS weight
  FROM question
  CROSS JOIN LATERAL (
    SELECT count(*) AS df
    FROM memories m
    WHERE m.status = 'active'
      AND m.{SEARCH_COLUMN} @@ quote_literal(question.lex)::tsquery
  ) AS seen
  WHERE seen.df > 0
    AND seen.df::numeric / (SELECT total FROM corpus) <= {COMMON_WORD_SHARE}
),
scored AS (
  SELECT {COLUMNS}, sum(weighted.weight) AS rank
  FROM memories m
  JOIN weighted ON m.{SEARCH_COLUMN} @@ quote_literal(weighted.lex)::tsquery
  WHERE m.status = 'active' AND (__FILTER__)
  GROUP BY m.id
),
best AS (SELECT max(rank) AS top FROM scored)
SELECT scored.*
FROM scored, best
WHERE scored.rank >= {MATCH_FLOOR} * best.top
ORDER BY scored.rank DESC
LIMIT %(limit)s
"""

_VECTOR_TEMPLATE = f"""
SELECT {COLUMNS}, 1 - (m.embedding <=> %(vec)s::vector) AS rank
FROM memories m
WHERE m.status = 'active' AND m.embedding IS NOT NULL AND (__FILTER__)
ORDER BY m.embedding <=> %(vec)s::vector
LIMIT %(limit)s
"""


def fts_sql(kind: str = "all") -> str:
    return _FTS_TEMPLATE.replace("__FILTER__", FILTERS[kind])


def vector_sql(kind: str = "all") -> str:
    return _VECTOR_TEMPLATE.replace("__FILTER__", FILTERS[kind])


# The deliberate lookup (``aida_recall``) searches everything.
FTS_SQL = fts_sql("all")
VECTOR_SQL = vector_sql("all")

# The core, whole: what was written on purpose comes first, newest first within a type.
CORE_SQL = f"""
SELECT m.content, m.memory_type, m.created_at
FROM memories m
WHERE m.status = 'active' AND {CORE_FILTER}
ORDER BY m.created_at DESC
"""


def to_vector_literal(vec: Sequence[float]) -> str:
    """pgvector reads a vector as the string ``'[0.1,0.2,…]'`` cast to ``::vector``."""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


def row_to_memory(row: Sequence[Any]) -> dict[str, Any]:
    """One result row → the shape the rest of the plugin passes around."""
    return {
        "id": row[0],
        "content": row[1],
        "memory_type": row[2],
        "priority": row[3],
        "created_at": row[4],
        "role": row[5],
        "chat_id": row[6],
        "rank": float(row[7]) if len(row) > 7 and row[7] is not None else 0.0,
    }


def above_vector_floor(hits: list[dict], floor: float = VECTOR_FLOOR) -> list[dict]:
    """Keep the vector hits about as close as the closest one (they arrive best first)."""
    if not hits:
        return []
    best = hits[0].get("rank") or 0.0
    if best <= 0:
        return []
    return [hit for hit in hits if (hit.get("rank") or 0.0) >= floor * best]


def combine_rrf(vector_hits: list[dict], fts_hits: list[dict], k: int = RRF_K) -> list[dict]:
    """Fuse two ranked lists by Reciprocal Rank Fusion, best first.

    RRF is rank-based on purpose: a cosine distance and a summed word weight are not
    comparable numbers, and any attempt to normalise them into one score silently favours
    whichever half happens to have the wider spread that day.
    """
    scored: dict[Any, dict] = {}
    for hits in (vector_hits, fts_hits):
        for position, hit in enumerate(hits, start=1):
            entry = scored.get(hit["id"])
            if entry is None:
                entry = dict(hit)
                entry["score"] = 0.0
                scored[hit["id"]] = entry
            entry["score"] += 1.0 / (k + position)
    return sorted(scored.values(), key=lambda item: item["score"], reverse=True)


def _age_days(created: Any, now: datetime) -> float:
    if not isinstance(created, datetime):
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (now - created).total_seconds() / 86400.0)


def pick_facts(ranked: list[dict], now: datetime | None = None) -> list[dict]:
    """Facts for one question: relevance first, then age, then a ceiling per priority."""
    now = now or datetime.now(timezone.utc)
    for item in ranked:
        decay = DECAY_LAMBDA.get(item.get("priority") or "P2", 0.05)
        item["score"] = item.get("score", 0.0) * math.exp(-decay * _age_days(item.get("created_at"), now))
    picked: list[dict] = []
    counts: dict[str, int] = {}
    for item in sorted(ranked, key=lambda entry: entry["score"], reverse=True):
        # A P1 row that reaches here is the distiller's (the core is filtered out in SQL):
        # an ordinary fact, counted with the ordinary facts.
        priority = "P3" if item.get("priority") == "P3" else "P2"
        if counts.get(priority, 0) >= FACT_CAPS[priority]:
            continue
        counts[priority] = counts.get(priority, 0) + 1
        picked.append(item)
    return picked


# Two wordings of one fact share most of their meaningful words. Words are cut to their first
# five letters: a crude stand-in for Russian word forms that needs no dictionary.
SAME_FACT_OVERLAP = 0.6


def _stems(text: Any) -> set[str]:
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or "")).split()
    return {word[:5] for word in words if len(word) >= 4}


def same_fact(new: Any, known: Any, threshold: float = SAME_FACT_OVERLAP) -> bool:
    """Does ``known`` already say what ``new`` says? Measured against the shorter of the two,
    so a fact restated with one extra clause still counts as the same fact."""
    left, right = _stems(new), _stems(known)
    if not left or not right:
        return False
    return len(left & right) / min(len(left), len(right)) >= threshold


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _clip(text: str, budget: int) -> str:
    return text if len(text) <= budget else text[:budget].rstrip() + " […]"


def _date(item: dict) -> str:
    created = item.get("created_at")
    return created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else ""


def format_block(memories: list[dict], entry_budget: int = ENTRY_BUDGET, block_budget: int = BLOCK_BUDGET) -> str:
    """Render memories found by a deliberate lookup — one line each.

    Everything here is the agent's OWN past: the operator decided that the conversations
    held with the Telegram bot are his memories, not a quoted third party. So the block
    carries no "someone else said" framing, only what was said and when.
    """
    if not memories:
        return ""
    lines: list[str] = []
    spent = 0
    for item in memories:
        tag = " · ".join(part for part in (item.get("memory_type"), _date(item)) if part)
        content = _clip(_flat(item.get("content")), entry_budget)
        line = f"- [{tag}] {content}" if tag else f"- {content}"
        if spent + len(line) > block_budget and lines:
            break
        lines.append(line)
        spent += len(line)
    return "[Из моей памяти — прошлые разговоры с оператором]\n" + "\n".join(lines)


def _fact_line(item: dict) -> str:
    tag = " · ".join(part for part in (item.get("memory_type"), _date(item)) if part)
    return f"- [{tag}] {_flat(item.get('content'))}"


def facts_that_fit(facts: list[dict], block_budget: int = RECALL_BLOCK_BUDGET) -> list[dict]:
    """The facts that actually make it into the block, in order. Only these were put in front
    of the model — and only these may count as used, or a fact nobody saw passes probation."""
    kept: list[dict] = []
    spent = 0
    for item in facts:
        size = len(_fact_line(item))
        if spent + size > block_budget and kept:
            break
        kept.append(item)
        spent += size
    return kept


def format_recall(facts: list[dict], dialogue: list[dict], block_budget: int = RECALL_BLOCK_BUDGET) -> tuple[str, int]:
    """What one question raised, as it goes into the turn. Returns the text and the row count.

    Facts go in whole — a fact cut in half is a different fact. A line of old conversation is
    a quote: shown down to its opening, dated, and attributed to whoever said it, because the
    operator's words handed back as the agent's own are worse than no recall at all.
    """
    sections: list[str] = []
    shown = 0

    lines = [_fact_line(item) for item in facts_that_fit(facts, block_budget)]
    spent = sum(len(line) for line in lines)
    if lines:
        sections.append("# По теме вопроса — что я об этом знаю\n" + "\n".join(lines))
        shown += len(lines)

    lines = []
    for item in dialogue:
        who = "Оператор" if item.get("role") == "user" else "Я"
        when = _date(item)
        quote = _clip(_flat(item.get("content")), QUOTE_BUDGET)
        line = f"[{when}] {who}: {quote}" if when else f"{who}: {quote}"
        if spent + len(line) > block_budget:
            break
        lines.append(line)
        spent += len(line)
    if lines:
        sections.append(
            "# Из прошлых разговоров, по смыслу вопроса (дословно; целиком — через aida_recall)\n"
            + "\n".join(lines)
        )
        shown += len(lines)

    return "\n\n".join(sections), shown
