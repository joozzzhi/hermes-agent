from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from plugins.memory.aida.search import (
    BLOCK_BUDGET,
    CORE_SQL,
    FILTERS,
    format_recall,
    pick_facts,
    ENTRY_BUDGET,
    MATCH_FLOOR,
    RRF_K,
    combine_rrf,
    format_block,
    row_to_memory,
    to_vector_literal,
)


def _hit(id_: int, content: str = "x") -> dict:
    return {"id": id_, "content": content, "memory_type": "fact", "priority": "P2", "created_at": None}


def test_rrf_ranks_a_memory_found_by_both_halves_first():
    vector = [_hit(1), _hit(2)]
    fts = [_hit(3), _hit(2)]

    fused = combine_rrf(vector, fts)

    assert [item["id"] for item in fused][0] == 2
    assert fused[0]["score"] == 1 / (RRF_K + 2) + 1 / (RRF_K + 2)
    assert {item["id"] for item in fused} == {1, 2, 3}


def test_rrf_survives_one_empty_half():
    assert [item["id"] for item in combine_rrf([], [_hit(7), _hit(8)])] == [7, 8]
    assert [item["id"] for item in combine_rrf([_hit(7)], [])] == [7]
    assert combine_rrf([], []) == []


def test_rrf_does_not_mutate_the_lists_it_was_given():
    fts = [_hit(1)]
    original = dict(fts[0])

    combine_rrf([], fts)

    assert fts[0] == original


def test_format_block_speaks_in_the_first_person_and_never_quotes_another_assistant():
    block = format_block([
        {"id": 1, "content": "Переезд в декабре", "memory_type": "fact",
         "priority": "P1", "created_at": datetime(2026, 8, 12)},
    ])

    assert block.startswith("[Из моей памяти")
    assert "- [fact · 2026-08-12] Переезд в декабре" in block
    for foreign in ("Аида сказала", "бот", "assistant"):
        assert foreign not in block


def test_format_block_collapses_a_multiline_memory_into_one_line():
    block = format_block([
        {"id": 1, "content": "первая строка\n\n   вторая", "memory_type": None,
         "priority": None, "created_at": None},
    ])

    assert block.splitlines()[1] == "- первая строка вторая"


def test_format_block_is_empty_when_nothing_was_recalled():
    assert format_block([]) == ""


def test_a_long_memory_is_shown_from_its_opening_and_the_rest_is_left_in_the_store():
    long_memory = {"id": 1, "content": "я" * 2000, "memory_type": "dialogue",
                   "priority": "P2", "created_at": None}

    line = format_block([long_memory]).splitlines()[1]

    assert line.endswith("[…]")
    assert len(line) < 500


def test_recalled_memory_never_crowds_out_the_conversation_it_was_recalled_for():
    many = [{"id": i, "content": "помню " * 100, "memory_type": "dialogue",
             "priority": "P2", "created_at": None} for i in range(8)]

    block = format_block(many)

    assert len(block) <= BLOCK_BUDGET + ENTRY_BUDGET
    assert len(block.splitlines()) < 9  # header plus fewer entries than were offered


def test_one_oversized_memory_is_still_shown_rather_than_dropped_for_being_too_big():
    huge = [{"id": 1, "content": "я" * 2000, "memory_type": None, "priority": None, "created_at": None}]

    assert len(format_block(huge, entry_budget=5000, block_budget=10).splitlines()) == 2


def test_vector_literal_is_what_pgvector_parses():
    assert to_vector_literal([0.5, -1.0]) == "[0.5,-1.0]"
    assert to_vector_literal([]) == "[]"


def test_row_to_memory_keeps_the_column_order_the_sql_selects():
    created = datetime(2026, 9, 21)

    assert row_to_memory((11, "текст", "decision", "P1", created, "user", "hermes:s1", 4.2)) == {
        "id": 11,
        "content": "текст",
        "memory_type": "decision",
        "priority": "P1",
        "created_at": created,
        "role": "user",
        "chat_id": "hermes:s1",
        "rank": 4.2,
    }


def test_match_floor_stays_in_step_with_the_bot():
    # The bot cuts weak matches at the same fraction of the best match for the question.
    # Drifting apart here makes one side silent about a memory the other happily returns.
    assert MATCH_FLOOR == 0.8
    assert RRF_K == 60


# -- the three tiers -----------------------------------------------------------------------


def _memories_db(rows):
    """An in-memory stand-in for the shared table, enough to run the tier filters against."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT, memory_type TEXT, "
               "priority TEXT, chat_id TEXT, status TEXT, created_at TEXT)")
    db.executemany(
        "INSERT INTO memories (content, memory_type, priority, chat_id, status, created_at) "
        "VALUES (?, ?, ?, ?, 'active', '2026-10-01')", rows)
    return db


_TIER_ROWS = [
    ("написано руками", "rule", "P1", "hermes:s1"),
    ("заметка бота", "fact", "P1", None),
    ("извлечено разбором", "fact", "P1", "hermes:nightly"),
    ("реплика", "dialogue", "P1", "hermes:s1"),
    ("обычный факт", "fact", "P2", "hermes:s1"),
]


def test_the_core_is_what_was_written_on_purpose_and_not_what_the_distiller_marked_p1():
    db = _memories_db(_TIER_ROWS)

    core = [row[0] for row in db.execute(CORE_SQL)]

    assert sorted(core) == ["заметка бота", "написано руками"]


def test_a_distilled_p1_row_is_searched_for_as_an_ordinary_fact():
    db = _memories_db(_TIER_ROWS)

    facts = [row[0] for row in db.execute(f"SELECT content FROM memories m WHERE {FILTERS['facts']}")]

    assert sorted(facts) == ["извлечено разбором", "обычный факт"]


def _fact(id_, priority="P2", score=1.0, age_days=0.0, now=datetime(2026, 10, 10, tzinfo=timezone.utc)):
    return {"id": id_, "content": f"факт {id_}", "memory_type": "fact", "priority": priority,
            "score": score, "created_at": now - timedelta(days=age_days)}


def test_one_question_raises_at_most_ten_p2_and_three_p3_facts():
    now = datetime(2026, 10, 10, tzinfo=timezone.utc)
    ranked = [_fact(i, "P2") for i in range(20)] + [_fact(100 + i, "P3") for i in range(10)]

    picked = pick_facts(ranked, now=now)

    assert sum(1 for f in picked if f["priority"] == "P2") == 10
    assert sum(1 for f in picked if f["priority"] == "P3") == 3


def test_a_fresh_fact_outranks_an_equally_relevant_old_one():
    now = datetime(2026, 10, 10, tzinfo=timezone.utc)

    picked = pick_facts([_fact(1, age_days=60, now=now), _fact(2, age_days=0, now=now)], now=now)

    assert [f["id"] for f in picked] == [2, 1]


def test_a_fact_on_probation_fades_faster_than_an_ordinary_one():
    now = datetime(2026, 10, 10, tzinfo=timezone.utc)

    picked = pick_facts([_fact(1, "P3", age_days=10, now=now), _fact(2, "P2", age_days=10, now=now)], now=now)

    assert [f["id"] for f in picked] == [2, 1]


def test_old_conversation_is_quoted_with_who_said_it_and_when():
    dialogue = [{"content": "переезжаю в декабре", "role": "user", "created_at": datetime(2026, 8, 12)},
                {"content": "записал", "role": "assistant", "created_at": None}]

    block, shown = format_recall([], dialogue)

    assert "[2026-08-12] Оператор: переезжаю в декабре" in block
    assert "Я: записал" in block
    assert shown == 2


def test_facts_go_in_whole_and_quotes_are_cut_to_their_opening():
    fact = {"content": "ф" * 1500, "memory_type": "fact", "created_at": None}
    quote = {"content": "ц" * 1500, "role": "user", "created_at": None}

    block, _ = format_recall([fact], [quote])

    assert "ф" * 1500 in block
    assert "ц" * 1500 not in block and "[…]" in block


def test_nothing_found_is_an_empty_block():
    assert format_recall([], []) == ("", 0)
