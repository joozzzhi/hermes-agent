from __future__ import annotations

import threading

import pytest

from plugins.memory.aida.queue_db import TurnQueue
from plugins.memory.aida.write import (
    MAX_CONTENT,
    SOURCE_PREFIX,
    chunk_content,
    describe_saved,
    insert_params,
    normalise_priority,
    normalise_role,
    normalise_type,
    source_tag,
)


# -- splitting a long memory ------------------------------------------------------------


def test_a_short_memory_is_stored_as_it_was_written():
    assert chunk_content("Переезд в декабре") == ["Переезд в декабре"]


def test_a_long_memory_is_split_into_marked_parts_and_nothing_is_thrown_away():
    text = ". ".join(f"Предложение номер {i} про переезд" for i in range(120))

    parts = chunk_content(text)

    assert len(parts) > 1
    assert all(part.startswith(f"[часть {i}/{len(parts)}]") for i, part in enumerate(parts, start=1))
    assert all(len(part) <= MAX_CONTENT for part in parts)
    # every original sentence survives somewhere in the parts
    joined = " ".join(parts)
    assert all(f"Предложение номер {i} про переезд" in joined for i in (0, 60, 119))


def test_an_unbroken_wall_of_characters_is_still_split_rather_than_refused():
    parts = chunk_content("я" * 5000)

    assert len(parts) >= 3
    assert all(len(part) <= MAX_CONTENT for part in parts)
    assert sum(part.count("я") for part in parts) == 5000


def test_an_empty_thought_is_not_a_memory():
    assert chunk_content("") == []
    assert chunk_content("    \n  ") == []


@pytest.mark.parametrize("length", [MAX_CONTENT - 1, MAX_CONTENT])
def test_a_memory_right_at_the_limit_is_left_whole(length):
    assert chunk_content("я" * length) == ["я" * length]


def test_a_memory_one_character_over_the_limit_starts_being_split():
    assert len(chunk_content("я" * (MAX_CONTENT + 1))) > 1


# -- what the store is told --------------------------------------------------------------


def test_the_insert_speaks_the_bot_s_dialect():
    params = insert_params("Переезд", memory_type="decision", priority="P1",
                           role="system", chat_id="hermes:abc")

    assert params == {"content": "Переезд", "memory_type": "decision", "priority": "P1",
                      "role": "system", "chat_id": "hermes:abc"}


@pytest.mark.parametrize("bad", ["сплетня", "", None, "DIALOGUE", 5])
def test_an_unknown_kind_of_memory_falls_back_to_a_plain_fact(bad):
    assert normalise_type(bad) == "fact"


@pytest.mark.parametrize("bad", ["P9", "high", "", None])
def test_an_unknown_priority_falls_back_to_the_middle(bad):
    assert normalise_priority(bad) == "P2"


@pytest.mark.parametrize("bad", ["bot", "", None, "USER"])
def test_an_unknown_speaker_is_recorded_as_the_agent(bad):
    assert normalise_role(bad) == "assistant"


def test_every_row_this_agent_writes_carries_a_tag_that_makes_it_removable():
    assert source_tag("session-7").startswith(SOURCE_PREFIX)
    assert source_tag("") == SOURCE_PREFIX + "session"


def test_the_agent_is_told_plainly_when_a_long_thought_was_split():
    assert describe_saved(["one"], "fact") == "Запомнил (fact)."
    assert "3 частями" in describe_saved(["a", "b", "c"], "decision")
    assert describe_saved([], "fact") == "Пустую запись сохранять нечего."


# -- the local queue of turns -------------------------------------------------------------


def test_a_finished_turn_waits_locally_until_it_is_carried_over(tmp_path):
    queue = TurnQueue(tmp_path / "queue.db")

    queue.enqueue("session-1", "user", "что там с переездом")
    queue.enqueue("session-1", "assistant", "в декабре")

    pending = queue.pending()
    assert [row["role"] for row in pending] == ["user", "assistant"]
    assert queue.count() == 2
    queue.close()


def test_an_empty_side_of_a_turn_is_not_queued(tmp_path):
    queue = TurnQueue(tmp_path / "queue.db")

    assert queue.enqueue("session-1", "assistant", "   ") is False
    assert queue.count() == 0
    queue.close()


def test_a_turn_leaves_the_queue_only_once_it_has_been_carried_over(tmp_path):
    queue = TurnQueue(tmp_path / "queue.db")
    queue.enqueue("session-1", "user", "первое")
    queue.enqueue("session-1", "user", "второе")

    pending = queue.pending()
    released = queue.release([pending[0]["id"]])

    assert released == 1
    assert [row["content"] for row in queue.pending()] == ["второе"]
    queue.close()


def test_releasing_nothing_changes_nothing(tmp_path):
    queue = TurnQueue(tmp_path / "queue.db")
    queue.enqueue("session-1", "user", "первое")

    assert queue.release([]) == 0
    assert queue.count() == 1
    queue.close()


def test_the_queue_survives_the_agent_being_restarted(tmp_path):
    path = tmp_path / "queue.db"
    first = TurnQueue(path)
    first.enqueue("session-1", "user", "переживёт ли это перезапуск")
    first.close()

    second = TurnQueue(path)

    assert second.count() == 1
    second.close()


def test_turns_arriving_from_several_threads_are_all_kept(tmp_path):
    queue = TurnQueue(tmp_path / "queue.db")

    def _write(index: int) -> None:
        for step in range(10):
            queue.enqueue(f"session-{index}", "user", f"реплика {index}-{step}")

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert queue.count() == 60
    assert len({row["content"] for row in queue.pending(limit=100)}) == 60
    queue.close()


def test_the_oldest_turns_are_carried_over_first(tmp_path):
    queue = TurnQueue(tmp_path / "queue.db")
    for step in range(5):
        queue.enqueue("session-1", "user", f"реплика {step}")

    assert [row["content"] for row in queue.pending(limit=3)] == ["реплика 0", "реплика 1", "реплика 2"]
    queue.close()
