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

Measured 2026-09-21: only 62 of 525 rows carry an embedding, so full text is today's real
recall and the vector half is a bonus that grows as the backfill fills it in.
"""

from __future__ import annotations

from typing import Any, Sequence

# Every query and the stored column must use the same text-search configuration, or nothing
# matches. Both live in the bot's search-query.js under the same names.
FTS_CONFIG = "russian"
SEARCH_COLUMN = "search_ru"
COMMON_WORD_SHARE = 0.1
MATCH_FLOOR = 0.8
RRF_K = 60

# How much recalled memory may occupy the turn. A stored entry can be 2000 characters long
# and eight of them would crowd out the conversation actually happening.
ENTRY_BUDGET = 400
BLOCK_BUDGET = 2600

COLUMNS = "m.id, m.content, m.memory_type, m.priority, m.created_at"

# quote_literal() around each lexeme: a question can carry quotes, URLs or emoji, and an
# unquoted lexeme would either break the tsquery parse or quietly mean something else.
FTS_SQL = f"""
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
  WHERE m.status = 'active'
  GROUP BY m.id
),
best AS (SELECT max(rank) AS top FROM scored)
SELECT scored.*
FROM scored, best
WHERE scored.rank >= {MATCH_FLOOR} * best.top
ORDER BY scored.rank DESC
LIMIT %(limit)s
"""

VECTOR_SQL = f"""
SELECT {COLUMNS}, 1 - (m.embedding <=> %(vec)s::vector) AS rank
FROM memories m
WHERE m.status = 'active' AND m.embedding IS NOT NULL
ORDER BY m.embedding <=> %(vec)s::vector
LIMIT %(limit)s
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
    }


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


def format_block(memories: list[dict], entry_budget: int = ENTRY_BUDGET, block_budget: int = BLOCK_BUDGET) -> str:
    """Render recalled memories for the prompt — one line each, oldest context first.

    Everything here is the agent's OWN past: the operator decided that the conversations
    held with the Telegram bot are his memories, not a quoted third party. So the block
    carries no "someone else said" framing, only what was said and when.

    What goes into the turn is a display window, not the memory itself. A stored entry runs
    to 2000 characters and eight of them would push a whole conversation out of the way to
    make room for one recalled from months ago, so a long entry is shown down to its opening
    and the rest is fetched on purpose with ``aida_recall``. Nothing is shortened in the
    store — the original is untouched and still findable in full.
    """
    if not memories:
        return ""
    lines: list[str] = []
    spent = 0
    for item in memories:
        created = item.get("created_at")
        when = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else ""
        tag = " · ".join(part for part in (item.get("memory_type"), when) if part)
        content = " ".join(str(item.get("content", "")).split())
        if len(content) > entry_budget:
            content = content[:entry_budget].rstrip() + " […]"
        line = f"- [{tag}] {content}" if tag else f"- {content}"
        if spent + len(line) > block_budget and lines:
            break
        lines.append(line)
        spent += len(line)
    return "[Из моей памяти — прошлые разговоры с оператором]\n" + "\n".join(lines)
