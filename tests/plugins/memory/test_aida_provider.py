from __future__ import annotations

import threading
import time

import pytest

import plugins.memory.aida as aida
from plugins.memory.aida import AidaMemoryProvider, _Store


class _FakeStore:
    """Stands in for the shared database: records what was asked, returns what it was told to."""

    def __init__(self, memories=None, delay: float = 0.0, dialogue=None, down: bool = False):
        self.memories = memories or []
        self.dialogue = dialogue or []
        self.delay = delay
        self.down = down
        self.queries: list[tuple[str, int]] = []
        self.tier_queries: list[str] = []
        self.used: list[list[int]] = []
        self.closed = False

    def recall_tiers(self, query: str):
        self.tier_queries.append(query)
        if self.down:
            return None
        return list(self.memories), list(self.dialogue)

    def mark_used(self, ids) -> None:
        self.used.append(list(ids))

    def recall(self, query: str, limit: int = 8) -> list[dict]:
        if self.delay:
            time.sleep(self.delay)
        self.queries.append((query, limit))
        return list(self.memories)

    def close(self) -> None:
        self.closed = True


class _RecordingStore(_FakeStore):
    """A store that remembers what it was asked to write, and can be told to fail."""

    def __init__(self, memories=None, fail: bool = False):
        super().__init__(memories)
        self.saved: list[dict] = []
        self.archived: list[str] = []
        self.fail = fail

    def save(self, content, *, memory_type="fact", priority="P2", role="assistant",
             chat_id="", embed=True, created_at=None, dedupe=True) -> list[int]:
        if self.fail:
            return []
        self.saved.append({"content": content, "memory_type": memory_type,
                           "priority": priority, "role": role, "chat_id": chat_id,
                           "dedupe": dedupe})
        return [len(self.saved)]

    def archive(self, content: str) -> int:
        self.archived.append(content)
        return 1


def _memory(id_: int = 1, content: str = "Переезд в декабре") -> dict:
    return {"id": id_, "content": content, "memory_type": "fact", "priority": "P2", "created_at": None}


def _provider(store: _FakeStore) -> AidaMemoryProvider:
    provider = AidaMemoryProvider()
    provider._store = store
    return provider


# -- valid input -----------------------------------------------------------------------


def test_recalled_memory_reaches_the_turn_and_the_indicator_counts_it():
    provider = _provider(_FakeStore([_memory(1), _memory(2, "кот Борис")]))

    block = provider.prefetch("что там с переездом")

    assert "Переезд в декабре" in block and "кот Борис" in block
    assert provider.recall_status().count == 2


def test_lines_of_past_dialogue_are_quoted_with_who_said_them():
    dialogue = [{"id": 5, "content": "переезжаем в декабре", "role": "user", "created_at": None},
                {"id": 6, "content": "записал", "role": "assistant", "created_at": None}]
    provider = _provider(_FakeStore([], dialogue=dialogue))

    block = provider.prefetch("что там с переездом")

    assert "Оператор: переезжаем в декабре" in block
    assert "Я: записал" in block
    assert provider.recall_status().count == 2


def test_nothing_recalled_leaves_no_indicator():
    provider = _provider(_FakeStore([]))

    assert provider.prefetch("чего никогда не было") == ""
    assert provider.recall_status() is None


def test_indicator_stays_silent_until_a_recall_has_actually_happened():
    assert _provider(_FakeStore([_memory()])).recall_status() is None


def test_every_question_is_searched_for_itself_at_the_moment_it_is_asked():
    # No answer prepared after the previous turn: that one answered the previous question.
    store = _FakeStore([_memory()])
    provider = _provider(store)

    provider.prefetch("что там с переездом")
    provider.prefetch("а как зовут кота")

    assert store.tier_queries == ["что там с переездом", "а как зовут кота"]


@pytest.mark.parametrize("query", ["", "ок", "да, давай", "x" * (aida.MIN_QUESTION_LENGTH - 1)])
def test_a_reply_too_short_to_ask_anything_is_not_searched_for(query):
    store = _FakeStore([_memory()])
    provider = _provider(store)

    assert provider.prefetch(query) == ""
    assert store.tier_queries == []
    assert provider.recall_status() is None


def test_an_unreachable_store_leaves_the_turn_without_memory_and_without_an_indicator():
    provider = _provider(_FakeStore([_memory()], down=True))

    assert provider.prefetch("что там с переездом") == ""
    assert provider.recall_status() is None


def test_facts_put_in_front_of_the_model_are_counted_as_used():
    # This is what a distilled fact on probation is judged by.
    store = _FakeStore([_memory(3), _memory(4, "кот Борис")])
    provider = _provider(store)

    provider.prefetch("что там с переездом")

    deadline = time.monotonic() + 5
    while not store.used and time.monotonic() < deadline:
        time.sleep(0.01)
    assert store.used == [[3, 4]]


def test_the_tool_returns_what_was_found():
    provider = _provider(_FakeStore([_memory()]))

    answer = provider.handle_tool_call("aida_recall", {"query": "переезд"})

    assert "Переезд в декабре" in answer


def test_the_tool_says_plainly_when_memory_holds_nothing():
    provider = _provider(_FakeStore([]))

    assert provider.handle_tool_call("aida_recall", {"query": "рыбалка"}) == "Ничего не нашлось."


# -- invalid input ---------------------------------------------------------------------


def test_an_unknown_tool_is_refused_rather_than_guessed_at():
    assert "Unknown tool" in _provider(_FakeStore()).handle_tool_call("aida_forget", {})


def test_the_tool_reports_missing_configuration_instead_of_crashing_the_turn():
    provider = AidaMemoryProvider()

    assert "not configured" in provider.handle_tool_call("aida_recall", {"query": "x"})


@pytest.mark.parametrize("top_k", ["восемь", None, [], {"a": 1}])
def test_a_nonsense_result_count_falls_back_to_the_default(top_k):
    store = _FakeStore([_memory()])

    _provider(store).handle_tool_call("aida_recall", {"query": "переезд", "top_k": top_k})

    assert store.queries[-1][1] == aida.DEFAULT_LIMIT


def test_a_missing_query_is_passed_through_as_empty_rather_than_exploding():
    store = _FakeStore([])

    assert _provider(store).handle_tool_call("aida_recall", {}) == "Ничего не нашлось."
    assert store.queries[-1][0] == ""


# -- boundary values -------------------------------------------------------------------


@pytest.mark.parametrize(
    "asked, expected",
    [(1, 1), (0, 1), (-5, 1), (aida.MAX_LIMIT, aida.MAX_LIMIT), (aida.MAX_LIMIT + 1, aida.MAX_LIMIT), (999, aida.MAX_LIMIT)],
)
def test_the_asked_for_result_count_is_held_inside_its_limits(asked, expected):
    store = _FakeStore([_memory()])

    _provider(store).handle_tool_call("aida_recall", {"query": "переезд", "top_k": asked})

    assert store.queries[-1][1] == expected


def test_an_empty_question_never_reaches_the_database():
    store = _Store("postgresql://nowhere/db")
    store._fetch = lambda *a, **k: pytest.fail("an empty question must not be searched for")

    assert store.recall("   ") == []


# -- service down ----------------------------------------------------------------------


def test_an_unreachable_store_costs_memory_but_not_the_answer():
    store = _Store("postgresql://nowhere/db")

    def _explode(*args, **kwargs):
        raise RuntimeError("connection refused")

    store._fetch = _explode

    assert store.recall("переезд") == []
    assert store._blocked_until > time.monotonic()


def test_a_broken_store_is_left_alone_for_a_while_instead_of_dialled_every_turn():
    store = _Store("postgresql://nowhere/db")
    calls = []

    def _explode(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("connection refused")

    store._fetch = _explode
    store.recall("первый вопрос")
    store.recall("второй вопрос")

    assert len(calls) == 1


def test_a_store_that_recovers_is_used_again():
    store = _Store("postgresql://nowhere/db")
    store._blocked_until = time.monotonic() - 1
    store._fetch = lambda sql, params: [_memory()]
    store._embed = lambda text: None

    assert store.recall("переезд") and store._blocked_until == 0.0


# -- concurrent access -----------------------------------------------------------------


# -- lifecycle -------------------------------------------------------------------------


def test_the_plugin_stays_off_until_the_connection_string_is_given(monkeypatch):
    provider = AidaMemoryProvider()
    monkeypatch.setattr(aida, "get_secret", lambda key, default="": "")
    assert provider.is_available() is False
    assert "hermes memory setup" in provider.unavailable_reason()

    monkeypatch.setattr(aida, "get_secret", lambda key, default="": "postgresql://user@host/db")
    assert provider.is_available() is True


def test_setup_asks_for_the_connection_string_and_treats_it_as_a_secret():
    fields = {field["key"]: field for field in AidaMemoryProvider().get_config_schema()}

    assert fields["db_url"]["secret"] is True and fields["db_url"]["required"] is True
    assert fields["db_url"]["env_var"] == "SUPABASE_DB_URL"
    assert fields["openrouter_key"]["required"] is False


def test_the_agent_can_both_look_things_up_and_put_them_away():
    names = [schema["name"] for schema in AidaMemoryProvider().get_tool_schemas()]

    assert names == ["aida_recall", "aida_remember"]


def test_a_raw_conversation_line_is_not_something_the_agent_saves_by_hand():
    # Verbatim lines are kept for it automatically; offering "dialogue" as a deliberate
    # choice would only invite it to re-save what is already being kept.
    remember = [s for s in AidaMemoryProvider().get_tool_schemas() if s["name"] == "aida_remember"][0]

    assert "dialogue" not in remember["parameters"]["properties"]["memory_type"]["enum"]


def test_the_agent_is_told_the_memory_is_his_own():
    block = AidaMemoryProvider().system_prompt_block()

    assert "first person" in block
    assert "YOUR OWN past" in block


# -- writing: what the operator asks to be remembered --------------------------------------


def test_a_thought_the_operator_asked_to_keep_reaches_the_shared_memory():
    store = _RecordingStore()
    provider = _provider(store)
    provider._session_id = "session-7"

    answer = provider.handle_tool_call(
        "aida_remember", {"content": "Переезд в декабре", "memory_type": "event", "priority": "P1"}
    )

    assert store.saved[0]["content"] == "Переезд в декабре"
    assert store.saved[0]["memory_type"] == "event" and store.saved[0]["priority"] == "P1"
    assert store.saved[0]["chat_id"].startswith("hermes:")
    assert "Запомнил" in answer


def test_a_thought_that_could_not_be_saved_is_reported_rather_than_silently_lost():
    provider = _provider(_RecordingStore(fail=True))

    answer = provider.handle_tool_call("aida_remember", {"content": "Переезд в декабре"})

    assert "Не записал" in answer


def test_an_empty_thought_is_refused():
    store = _RecordingStore()

    assert "empty" in _provider(store).handle_tool_call("aida_remember", {"content": "   "})
    assert store.saved == []


def test_a_long_thought_is_reported_as_split_so_nobody_thinks_it_was_cut():
    store = _RecordingStore()

    answer = _provider(store).handle_tool_call("aida_remember", {"content": "Переезд. " * 400})

    assert "частями" in answer


# -- writing: the agent's own notes --------------------------------------------------------


def test_a_note_the_agent_keeps_about_the_operator_lands_in_the_shared_memory_too():
    store = _RecordingStore()
    provider = _provider(store)

    provider.on_memory_write("add", "user", "Пьёт кофе без сахара", {})

    assert store.saved[0]["memory_type"] == "preference"
    assert store.saved[0]["priority"] == "P1"  # curated by hand, not overheard


def test_a_note_about_the_work_is_mirrored_as_a_plain_fact():
    store = _RecordingStore()

    _provider(store).on_memory_write("add", "memory", "Шлюз перезапускается командой", {})

    assert store.saved[0]["memory_type"] == "fact"


def test_replacing_a_note_puts_the_old_wording_away_and_keeps_the_new_one():
    store = _RecordingStore()

    _provider(store).on_memory_write("replace", "memory", "новое", {"old_text": "старое"})

    assert store.archived == ["старое"]
    assert store.saved[0]["content"] == "новое"


def test_removing_a_note_stops_it_surfacing_without_writing_anything_new():
    store = _RecordingStore()

    _provider(store).on_memory_write("remove", "memory", "уже неправда", {})

    assert store.archived == ["уже неправда"]
    assert store.saved == []


# -- writing: the conversation itself ------------------------------------------------------


def test_a_finished_turn_goes_into_the_shared_store_at_once_word_for_word(tmp_path):
    from plugins.memory.aida.queue_db import TurnQueue

    store = _RecordingStore()
    provider = _provider(store)
    provider._queue = TurnQueue(tmp_path / "queue.db")

    provider.sync_turn("что там с переездом", "в декабре", session_id="session-7")

    assert [(row["role"], row["content"]) for row in store.saved] == [
        ("user", "что там с переездом"), ("assistant", "в декабре")]
    assert all(row["memory_type"] == "dialogue" and row["priority"] == "P3" for row in store.saved)
    assert all(row["chat_id"] == "hermes:session-7" for row in store.saved)
    # "да" said twice was said twice.
    assert all(row["dedupe"] is False for row in store.saved)
    provider._queue.close()


def test_a_written_turn_still_waits_in_the_local_queue_to_be_distilled(tmp_path):
    from plugins.memory.aida.queue_db import TurnQueue

    provider = _provider(_RecordingStore())
    provider._queue = TurnQueue(tmp_path / "queue.db")

    provider.sync_turn("что там с переездом", "в декабре", session_id="session-7")

    rows = provider._queue.pending()
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[0]["session_id"] == "session-7"
    assert all(row["shipped"] for row in rows)
    assert provider._queue.unshipped() == []
    provider._queue.close()


def test_with_the_store_down_the_turn_waits_unshipped_in_the_queue(tmp_path):
    from plugins.memory.aida.queue_db import TurnQueue

    provider = _provider(_RecordingStore(fail=True))
    provider._queue = TurnQueue(tmp_path / "queue.db")

    provider.sync_turn("что там с переездом", "в декабре", session_id="session-7")

    assert [row["content"] for row in provider._queue.unshipped()] == ["что там с переездом", "в декабре"]
    provider._queue.close()


def test_a_turn_where_the_agent_saved_a_fact_itself_is_kept_but_not_distilled(tmp_path):
    from plugins.memory.aida.queue_db import TurnQueue

    store = _RecordingStore()
    provider = _provider(store)
    provider._queue = TurnQueue(tmp_path / "queue.db")
    messages = [{"role": "user", "content": "запомни: переезд в декабре"},
                {"role": "assistant", "tool_calls": [{"function": {"name": "aida_remember"}}]},
                {"role": "tool", "tool_call_id": "1", "content": "ok"},
                {"role": "assistant", "content": "запомнил"}]

    provider.sync_turn("запомни: переезд в декабре", "запомнил", session_id="s", messages=messages)

    assert len(store.saved) == 2
    assert not any(row["distill"] for row in provider._queue.pending())
    provider._queue.close()


def test_a_helper_agent_working_in_the_background_does_not_write_to_the_shared_memory(tmp_path):
    # Only the agent the operator is actually talking to writes; otherwise a subagent's
    # scratch work would end up in the memory as if it had been said out loud.
    store = _RecordingStore()
    provider = AidaMemoryProvider()
    provider._store = store
    provider.initialize("session-7", hermes_home=str(tmp_path), agent_context="subagent")

    provider.sync_turn("вопрос", "ответ", session_id="session-7")
    provider.on_memory_write("add", "memory", "что-то", {})
    answer = provider.handle_tool_call("aida_remember", {"content": "что-то"})

    assert store.saved == []
    assert provider._queue.count() == 0
    assert "does not write" in answer


def test_a_missing_local_queue_costs_the_record_of_the_turn_but_not_the_turn():
    provider = _provider(_RecordingStore())
    provider._queue = None

    provider.sync_turn("вопрос", "ответ")  # must not raise


def test_teardown_releases_the_database():
    store = _FakeStore([_memory()])
    provider = _provider(store)

    provider.shutdown()

    assert store.closed is True


# -- the turn's own remember call --------------------------------------------------------


def test_an_old_remember_in_a_long_chat_does_not_silence_every_later_turn():
    """Regression: sync_turn gets the whole conversation, so a remember call from an earlier
    turn must not mark the current one as already saved (it stopped all nightly shipping)."""
    remember = {"role": "assistant", "tool_calls": [{"function": {"name": "aida_remember"}}]}
    earlier = [{"role": "user", "content": "a"}, remember,
               {"role": "tool", "tool_call_id": "1", "content": "ok"},
               {"role": "assistant", "content": "x"}]
    plain_turn = earlier + [{"role": "user", "content": "b"}, {"role": "assistant", "content": "y"}]
    remember_turn = earlier + [{"role": "user", "content": "b"}, remember]
    assert aida._turn_called_remember(plain_turn) is False
    assert aida._turn_called_remember(remember_turn) is True
