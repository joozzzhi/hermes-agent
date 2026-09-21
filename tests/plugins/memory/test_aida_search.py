from __future__ import annotations

from datetime import datetime

from plugins.memory.aida.search import (
    BLOCK_BUDGET,
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

    assert row_to_memory((11, "текст", "decision", "P1", created, 4.2)) == {
        "id": 11,
        "content": "текст",
        "memory_type": "decision",
        "priority": "P1",
        "created_at": created,
    }


def test_match_floor_stays_in_step_with_the_bot():
    # The bot cuts weak matches at the same fraction of the best match for the question.
    # Drifting apart here makes one side silent about a memory the other happily returns.
    assert MATCH_FLOOR == 0.8
    assert RRF_K == 60
