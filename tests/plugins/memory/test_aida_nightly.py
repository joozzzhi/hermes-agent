from __future__ import annotations

import pytest

from plugins.memory.aida import nightly_job
from plugins.memory.aida.queue_db import TurnQueue


class _Store:
    """Стоит вместо общей базы: помнит, что в неё писали, и умеет притворяться упавшей."""

    def __init__(self, *, down: bool = False, duplicates: set[str] | None = None,
                 rows_without_vector: list[tuple] | None = None, vectors: list | None = None,
                 unembeddable: set[str] | None = None, known: set[str] | None = None):
        self.down = down
        self.known = known or set()
        self.unembeddable = unembeddable or set()
        self.duplicates = duplicates or set()
        self.saved: list[dict] = []
        self.embedded: list[int] = []
        self._rows = list(rows_without_vector or [])
        self._vectors = vectors
        self.embed_calls: list[list[str]] = []

    def save(self, content, *, memory_type="fact", priority="P2", role="assistant",
             chat_id="", embed=True, created_at=None, dedupe=True):
        if self.down or content in self.duplicates:
            return []
        self.saved.append({"content": content, "memory_type": memory_type,
                           "priority": priority, "role": role, "chat_id": chat_id, "embed": embed,
                           "created_at": created_at, "dedupe": dedupe})
        return [len(self.saved)]

    def knows(self, content):
        return content in self.known

    def alive(self):
        return not self.down

    def rows_without_embedding(self, limit):
        return self._rows[:limit]

    def count_without_embedding(self):
        return max(0, len(self._rows) - len(self.embedded))

    def embed_many(self, texts):
        self.embed_calls.append(list(texts))
        if self._vectors is None:
            return None
        # "" — запись, которую служба не приняла: остальные в пачке должны пройти
        return [[] if text in self.unembeddable else [0.1, 0.2] for text in texts]

    def set_embedding(self, row_id, vector):
        self.embedded.append(row_id)
        return True

    def close(self):
        pass


def _queue(tmp_path, turns):
    queue = TurnQueue(tmp_path / "queue.db")
    for session, role, content in turns:
        queue.enqueue(session, role, content)
    return queue


# -- довоз дословного --------------------------------------------------------------------


def test_late_turns_reach_the_shared_memory_word_for_word_as_dialogue(tmp_path):
    store = _Store()
    queue = _queue(tmp_path, [("s1", "user", "что там с переездом"), ("s1", "assistant", "в декабре")])

    shipped, turns = nightly_job.ship_turns(store, queue)

    assert shipped == 2
    assert [row["content"] for row in store.saved] == ["что там с переездом", "в декабре"]
    assert all(row["memory_type"] == "dialogue" and row["priority"] == "P3" for row in store.saved)
    assert queue.unshipped() == []
    # A delivered turn still waits in the queue — to be distilled.
    assert queue.count() == 2
    queue.close()


def test_a_late_turn_is_dated_by_when_it_was_said_not_when_it_was_delivered(tmp_path):
    store = _Store()
    queue = _queue(tmp_path, [("s1", "user", "что там с переездом")])
    said = queue.pending()[0]["created_at"]

    nightly_job.ship_turns(store, queue)

    assert store.saved[0]["created_at"].startswith(said)
    queue.close()


def test_a_line_said_twice_is_kept_twice(tmp_path):
    store = _Store()
    queue = _queue(tmp_path, [("s1", "user", "да"), ("s1", "user", "да")])

    nightly_job.ship_turns(store, queue)

    assert [row["content"] for row in store.saved] == ["да", "да"]
    assert all(row["dedupe"] is False for row in store.saved)
    queue.close()


def test_who_said_what_survives_the_trip(tmp_path):
    store = _Store()
    queue = _queue(tmp_path, [("s1", "user", "вопрос длиною в жизнь"), ("s1", "assistant", "ответ")])

    nightly_job.ship_turns(store, queue)

    assert [row["role"] for row in store.saved] == ["user", "assistant"]
    assert all(row["chat_id"].startswith("hermes:") for row in store.saved)
    queue.close()


def test_a_pass_with_the_database_down_loses_nothing(tmp_path):
    store = _Store(down=True)
    queue = _queue(tmp_path, [("s1", "user", "важная реплика"), ("s1", "assistant", "ответ")])

    shipped, _ = nightly_job.ship_turns(store, queue)

    assert shipped == 0
    assert len(queue.unshipped()) == 2  # ждут следующего прогона
    queue.close()


def test_a_turn_the_live_store_writes_nothing_for_does_not_block_the_queue_forever(tmp_path):
    # Nothing left after scrubbing, store alive: the turn is let go instead of retried forever.
    store = _Store(duplicates={"это уже записано"})
    queue = _queue(tmp_path, [("s1", "user", "это уже записано"), ("s1", "assistant", "новое")])

    shipped, _ = nightly_job.ship_turns(store, queue)

    assert shipped == 1
    assert queue.unshipped() == []
    queue.close()


def test_an_empty_queue_is_a_quiet_night(tmp_path):
    store = _Store()
    queue = _queue(tmp_path, [])

    assert nightly_job.ship_turns(store, queue) == (0, [])
    assert store.saved == []
    queue.close()


def test_verbatim_turns_are_left_for_the_batch_that_fills_in_meaning_search(tmp_path):
    store = _Store()
    queue = _queue(tmp_path, [("s1", "user", "реплика")])

    nightly_job.ship_turns(store, queue)

    assert store.saved[0]["embed"] is False
    queue.close()


# -- складывание обменов -----------------------------------------------------------------


def test_replies_are_paired_back_with_the_questions_they_answered():
    turns = [
        {"session_id": "s1", "role": "user", "content": "первый вопрос"},
        {"session_id": "s1", "role": "assistant", "content": "первый ответ"},
        {"session_id": "s1", "role": "user", "content": "второй вопрос"},
        {"session_id": "s1", "role": "assistant", "content": "второй ответ"},
    ]

    assert nightly_job.build_exchanges(turns) == [
        ("s1", "первый вопрос", "первый ответ"),
        ("s1", "второй вопрос", "второй ответ"),
    ]


def test_two_conversations_of_the_same_night_do_not_bleed_into_each_other():
    turns = [
        {"session_id": "s1", "role": "user", "content": "про переезд"},
        {"session_id": "s2", "role": "user", "content": "про работу"},
        {"session_id": "s1", "role": "assistant", "content": "ответ про переезд"},
        {"session_id": "s2", "role": "assistant", "content": "ответ про работу"},
    ]

    assert set(nightly_job.build_exchanges(turns)) == {
        ("s1", "про переезд", "ответ про переезд"),
        ("s2", "про работу", "ответ про работу"),
    }


def test_a_question_left_without_an_answer_is_not_invented_into_an_exchange():
    turns = [{"session_id": "s1", "role": "user", "content": "вопрос в пустоту"}]

    assert nightly_job.build_exchanges(turns) == []


# -- разбор ответа модели -----------------------------------------------------------------


def test_facts_are_dug_out_of_an_answer_wrapped_in_prose_and_fences():
    raw = 'Вот что нашлось:\n```json\n[{"content": "Переезд в декабре", "type": "event", "priority": "P1"}]\n```\nВсё.'

    assert nightly_job.parse_facts(raw) == [
        {"content": "Переезд в декабре", "type": "event", "priority": "P3"}
    ]


@pytest.mark.parametrize("raw", ["", "нечего извлекать", "[", "{\"content\": \"не массив\"}", "[не json]"])
def test_an_answer_that_is_not_a_list_of_facts_yields_nothing_rather_than_garbage(raw):
    assert nightly_job.parse_facts(raw) == []


def test_a_distilled_fact_starts_on_probation_whatever_priority_the_model_claims():
    # P1 is the core, and only a deliberate write gets there.
    raw = '[{"content": "Переезд в декабре", "type": "event", "priority": "P1"}]'

    assert [fact["priority"] for fact in nightly_job.parse_facts(raw)] == ["P3"]


def test_an_invented_kind_of_memory_is_recorded_as_a_plain_fact():
    raw = '[{"content": "Любит кофе", "type": "сплетня", "priority": "P9"}]'

    assert nightly_job.parse_facts(raw) == [
        {"content": "Любит кофе", "type": "fact", "priority": "P3"}
    ]


def test_a_flood_of_facts_from_one_exchange_is_capped():
    raw = "[" + ",".join(f'{{"content": "факт номер {i}", "type": "fact"}}' for i in range(20)) + "]"

    assert len(nightly_job.parse_facts(raw)) == nightly_job.MAX_FACTS_PER_EXCHANGE


@pytest.mark.parametrize("content", ["ab", "я" * 2001])
def test_a_fact_too_short_or_too_long_to_be_one_is_dropped(content):
    raw = '[{"content": "%s", "type": "fact"}]' % content

    assert nightly_job.parse_facts(raw) == []


# -- кто отвечает на разбор --------------------------------------------------------------


def test_the_agent_s_own_model_layer_is_asked_first(monkeypatch):
    # У агента уже есть и подписка, и обновление доступа, и запасные пути. Своя копия
    # авторизации рядом была бы вторым местом, где всё это ломается.
    calls = []
    monkeypatch.setattr(nightly_job, "_ask_through_agent", lambda m: calls.append(m) or "ответ")
    monkeypatch.setattr(nightly_job, "_ask_over_http", lambda m: pytest.fail("не должно вызываться"))

    assert nightly_job.ask_model([{"role": "user", "content": "вопрос"}]) == "ответ"
    assert len(calls) == 1


def test_a_key_call_stands_in_when_the_agent_is_not_around(monkeypatch):
    monkeypatch.setattr(nightly_job, "_ask_through_agent", lambda m: None)
    monkeypatch.setattr(nightly_job, "_ask_over_http", lambda m: "запасной ответ")

    assert nightly_job.ask_model([{"role": "user", "content": "вопрос"}]) == "запасной ответ"


def test_silence_from_every_model_is_reported_as_silence(monkeypatch):
    monkeypatch.setattr(nightly_job, "_ask_through_agent", lambda m: None)
    monkeypatch.setattr(nightly_job, "_ask_over_http", lambda m: None)

    assert nightly_job.ask_model([{"role": "user", "content": "вопрос"}]) is None


def test_the_light_model_is_the_one_asked(monkeypatch):
    # Разбор — работа лёгкая: прочитать обмен и назвать, что в нём стоит помнить.
    captured = {}

    class _FakeCaller:
        @staticmethod
        def call_llm(**kwargs):
            captured.update(kwargs)
            return type("R", (), {"choices": [type("C", (), {
                "message": type("M", (), {"content": "[]"})()
            })()]})()

    monkeypatch.setitem(__import__("sys").modules, "agent.auxiliary_client", _FakeCaller)

    nightly_job._ask_through_agent([{"role": "user", "content": "вопрос"}])

    assert captured["model"] == nightly_job.DISTILL_MODEL == "claude-haiku-4-5"
    assert captured["temperature"] == 0


def test_a_model_that_is_down_does_not_take_the_night_with_it(monkeypatch, capsys):
    class _Broken:
        @staticmethod
        def call_llm(**kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setitem(__import__("sys").modules, "agent.auxiliary_client", _Broken)

    assert nightly_job._ask_through_agent([{"role": "user", "content": "вопрос"}]) is None
    assert "недоступна" in capsys.readouterr().out


# -- разбор без ответа модели --------------------------------------------------------------


def test_an_exchange_the_model_could_not_answer_waits_for_the_next_night(tmp_path, monkeypatch, capsys):
    # Реплики к этому моменту уже в базе: не сохрани мы обмен здесь, разобрать его
    # было бы больше нечем и факт пропал бы молча.
    monkeypatch.setattr(nightly_job, "ask_model", lambda messages: None)
    store = _Store()
    queue = _queue(tmp_path, [])
    queue.enqueue_exchange("s1", "длинный вопрос про переезд в декабре", "ответ")

    assert nightly_job.distill_exchanges(store, queue) == 0
    assert queue.count_exchanges() == 1
    assert "модель не ответила" in capsys.readouterr().out
    queue.close()


def test_a_short_exchange_is_not_worth_a_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly_job, "ask_model", lambda messages: pytest.fail("не должно вызываться"))
    store = _Store()
    queue = _queue(tmp_path, [])
    queue.enqueue_exchange("s1", "ок", "ага")

    assert nightly_job.distill_exchanges(store, queue) == 0
    assert queue.count_exchanges() == 0  # короткий обмен отпущен, а не копится вечно
    queue.close()


def test_an_exchange_that_was_distilled_is_not_distilled_again(tmp_path, monkeypatch):
    monkeypatch.setattr(
        nightly_job, "ask_model",
        lambda messages: '[{"content": "Переезд в декабре", "type": "event", "priority": "P1"}]',
    )
    store = _Store()
    queue = _queue(tmp_path, [])
    queue.enqueue_exchange("s1", "длинный вопрос про переезд в декабре", "ответ")

    assert nightly_job.distill_exchanges(store, queue) == 1
    assert store.saved[0]["content"] == "Переезд в декабре"
    assert store.saved[0]["priority"] == "P3"
    assert queue.count_exchanges() == 0
    queue.close()


def test_a_fact_found_at_night_is_left_for_the_batch_that_fills_in_meaning_search(tmp_path, monkeypatch):
    monkeypatch.setattr(
        nightly_job, "ask_model",
        lambda messages: '[{"content": "Переезд в декабре", "type": "event", "priority": "P1"}]',
    )
    store = _Store()
    queue = _queue(tmp_path, [])
    queue.enqueue_exchange("s1", "длинный вопрос про переезд в декабре", "ответ")

    nightly_job.distill_exchanges(store, queue)

    assert store.saved[0]["embed"] is False
    queue.close()


# -- разбор по темам ---------------------------------------------------------------------


def _topic_queue(tmp_path):
    queue = _queue(tmp_path, [])
    queue.enqueue_topic("s1", "переезд", [
        {"role": "user", "content": "решили переезжать в декабре, кота берём с собой"},
        {"role": "assistant", "content": "понял, запомню"},
    ])
    return queue


def test_one_topic_yields_at_most_two_facts(tmp_path, monkeypatch):
    raw = "[" + ",".join(f'{{"content": "устойчивый факт {i}", "type": "fact"}}' for i in range(6)) + "]"
    monkeypatch.setattr(nightly_job, "ask_model", lambda messages: raw)
    store = _Store()
    queue = _topic_queue(tmp_path)

    assert nightly_job.distill_topics(store, queue) == 2
    assert len(store.saved) == 2
    assert all(row["priority"] == "P3" and row["chat_id"] == "hermes:nightly" for row in store.saved)
    assert queue.count_topics() == 0
    queue.close()


def test_a_fact_memory_already_knows_is_not_written_again(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly_job, "ask_model",
                        lambda messages: '[{"content": "Переезд в декабре", "type": "decision"}]')
    store = _Store(known={"Переезд в декабре"})
    queue = _topic_queue(tmp_path)

    assert nightly_job.distill_topics(store, queue) == 0
    assert store.saved == []
    queue.close()


def test_a_topic_the_model_did_not_answer_waits_for_the_next_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly_job, "ask_model", lambda messages: None)
    store = _Store()
    queue = _topic_queue(tmp_path)

    assert nightly_job.distill_topics(store, queue) == 0
    assert queue.count_topics() == 1
    queue.close()


# -- дозаполнение поиска --------------------------------------------------------------------


def test_memories_that_could_only_be_found_by_words_get_their_meaning_back():
    store = _Store(rows_without_vector=[(1, "первая"), (2, "вторая"), (3, "третья")], vectors=True)

    filled, left = nightly_job.backfill_embeddings(store, limit=10)

    assert filled == 3
    assert store.embedded == [1, 2, 3]
    assert left == 0


def test_the_backfill_goes_in_batches_rather_than_one_call_per_memory():
    rows = [(i, f"запись {i}") for i in range(1, 18)]
    store = _Store(rows_without_vector=rows, vectors=True)

    nightly_job.backfill_embeddings(store, limit=17)

    assert all(len(call) <= nightly_job.EMBED_BATCH for call in store.embed_calls)
    assert len(store.embed_calls) == 3


def test_a_night_when_the_embedding_service_is_silent_stops_instead_of_hammering_it():
    store = _Store(rows_without_vector=[(1, "первая"), (2, "вторая")], vectors=None)

    filled, _ = nightly_job.backfill_embeddings(store, limit=10)

    assert filled == 0
    assert len(store.embed_calls) == 1


def test_one_memory_the_service_will_not_take_does_not_cost_the_whole_night():
    # Measured on the live store: a handful of blank and one-character rows stopped the pass
    # dead, and 68 memories stayed word-searchable for it.
    rows = [(1, "первая"), (2, "негодная"), (3, "третья"), (4, "четвёртая")]
    store = _Store(rows_without_vector=rows, vectors=True, unembeddable={"негодная"})

    filled, _ = nightly_job.backfill_embeddings(store, limit=10)

    assert filled == 3
    assert 2 not in store.embedded


def test_nothing_left_without_meaning_is_a_no_op():
    store = _Store(rows_without_vector=[], vectors=True)

    assert nightly_job.backfill_embeddings(store) == (0, 0)
    assert store.embed_calls == []


# -- отчёт о прогоне --------------------------------------------------------------------------


def test_without_a_connection_string_the_night_says_so_and_changes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)

    assert nightly_job.run(tmp_path) == 0
    assert "не настроена" in capsys.readouterr().out
