from __future__ import annotations

import threading

import plugins.context_engine.aida as engine_module
from plugins.context_engine.aida import AidaContextEngine


class _Store:
    """Стоит вместо общей базы: отдаёт, что велено, и считает обращения."""

    def __init__(self, *, knowledge=None, found=None, explode: bool = False):
        self.knowledge = knowledge or []
        self.found = found or []
        self.explode = explode
        self.recall_calls = 0
        self.knowledge_calls = 0
        self.closed = False

    def _execute_guarded(self, sql, params):
        if self.explode:
            raise RuntimeError("база недоступна")
        self.knowledge_calls += 1
        return [(row["content"], row["memory_type"]) for row in self.knowledge]

    def recall(self, query, limit=8):
        if self.explode:
            raise RuntimeError("база недоступна")
        self.recall_calls += 1
        return list(self.found)

    def close(self):
        self.closed = True


def _engine(store) -> AidaContextEngine:
    eng = AidaContextEngine()
    eng._store = store
    eng._store_tried = True
    return eng


def _request(question: str = "что там с переездом") -> list[dict]:
    return [
        {"role": "system", "content": "личность"},
        {"role": "user", "content": "раньше"},
        {"role": "assistant", "content": "ответ"},
        {"role": "user", "content": question},
    ]


# -- обычная работа ------------------------------------------------------------------------


def test_the_question_arrives_with_what_memory_knows_about_it():
    store = _Store(
        knowledge=[{"content": "Оператор работает с ноутбука", "memory_type": "fact"}],
        found=[{"content": "переезд в декабре", "memory_type": "event", "created_at": None}],
    )

    selected = _engine(store).select_context(_request(), incoming_message={"role": "user", "content": "переезд"})

    assert any("Оператор работает с ноутбука" in str(m.get("content")) for m in selected)
    assert "переезд в декабре" in str(selected[-1]["content"])
    assert str(selected[-1]["content"]).endswith("что там с переездом")


def test_an_old_line_of_conversation_comes_back_as_itself_not_as_a_summary():
    store = _Store(found=[
        {"content": "я переезжаю в декабре", "memory_type": "dialogue", "role": "user", "created_at": None},
    ])

    selected = _engine(store).select_context(_request())

    assert "Оператор: я переезжаю в декабре" in str(selected[-1]["content"])


def test_the_persisted_conversation_is_never_touched():
    request = _request()
    snapshot = [dict(m) for m in request]

    _engine(_Store(found=[{"content": "деталь", "memory_type": "fact", "created_at": None}])).select_context(request)

    assert request == snapshot


# -- когда что-то не так -------------------------------------------------------------------


def test_without_shared_memory_the_engine_does_not_interfere():
    eng = AidaContextEngine()
    eng._store_tried = True  # память не настроена

    assert eng.select_context(_request()) is None


def test_a_database_that_fails_costs_the_extra_context_not_the_answer():
    assert _engine(_Store(explode=True)).select_context(_request()) is None


def test_an_empty_request_is_left_alone():
    assert _engine(_Store()).select_context([]) is None


def test_a_request_with_no_question_in_it_still_assembles():
    store = _Store(knowledge=[{"content": "знание", "memory_type": "fact"}])
    request = [{"role": "system", "content": "личность"}, {"role": "assistant", "content": "ответ"}]

    selected = _engine(store).select_context(request)

    assert selected is not None
    assert store.recall_calls == 0  # искать нечего — в базу не ходим


# -- цена сборки ---------------------------------------------------------------------------


def test_tool_work_inside_one_turn_does_not_search_the_base_over_and_over():
    store = _Store(found=[{"content": "деталь", "memory_type": "fact", "created_at": None}])
    eng = _engine(store)

    for _ in range(5):
        eng.select_context(_request())

    assert store.recall_calls == 1
    assert store.knowledge_calls == 1


def test_a_new_question_is_searched_for_anew():
    store = _Store(found=[{"content": "деталь", "memory_type": "fact", "created_at": None}])
    eng = _engine(store)

    eng.select_context(_request("первый вопрос"))
    eng.select_context(_request("совсем другой вопрос"))

    assert store.recall_calls == 2


def test_several_threads_assembling_at_once_do_not_trip_over_the_cache():
    store = _Store(found=[{"content": "деталь", "memory_type": "fact", "created_at": None}])
    eng = _engine(store)
    results = []

    def _assemble(index: int) -> None:
        results.append(eng.select_context(_request(f"вопрос {index % 3}")))

    threads = [threading.Thread(target=_assemble, args=(i,)) for i in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(results) == 12
    assert all(r is not None for r in results)


# -- край окна -----------------------------------------------------------------------------


def test_running_out_of_room_lets_old_messages_go_and_never_retells_them():
    eng = _engine(_Store())
    eng.total_budget = 400
    messages = [{"role": "system", "content": "личность"}]
    messages += [{"role": "user", "content": f"реплика {i} " + "я" * 100} for i in range(20)]

    kept = eng.compress(messages)

    assert kept[0]["content"] == "личность"
    assert len(kept) < len(messages)
    assert kept[-1]["content"].startswith("реплика 19")
    assert not any("итог" in str(m.get("content")).lower() for m in kept)  # никакого пересказа
    assert eng.compression_count == 1


def test_a_conversation_that_fits_is_left_exactly_as_it_was():
    eng = _engine(_Store())
    messages = [{"role": "system", "content": "личность"}, {"role": "user", "content": "коротко"}]

    assert eng.compress(messages) == messages
    assert eng.compression_count == 0


def test_the_engine_says_nothing_when_it_lets_old_messages_go():
    # Отпустить старое — это норма работы движка, а не событие, о котором сообщают.
    assert AidaContextEngine.emit_automatic_compaction_status is False


def test_token_use_is_tracked_for_the_host():
    eng = AidaContextEngine()
    eng.update_from_response({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})

    assert eng.get_status()["last_prompt_tokens"] == 100


def test_compaction_only_fires_as_a_safety_net():
    eng = AidaContextEngine()
    eng.threshold_tokens = 1000

    assert eng.should_compress(500) is False
    assert eng.should_compress(1500) is True


def test_ending_a_session_releases_the_database():
    store = _Store()
    eng = _engine(store)

    eng.on_session_end("session-1", [])

    assert store.closed is True


def test_the_engine_registers_itself_the_way_the_host_expects():
    collected = {}

    class _Ctx:
        @staticmethod
        def register_context_engine(engine):
            collected["engine"] = engine

    engine_module.register(_Ctx())

    assert isinstance(collected["engine"], AidaContextEngine)


# -- ход с инструментами и несколько разговоров ---------------------------------------------


def _tool_turn_request(question: str, steps: int) -> list[dict]:
    request = [{"role": "system", "content": "личность"}, {"role": "user", "content": question}]
    for step in range(steps):
        request.append({"role": "assistant", "content": "",
                        "tool_calls": [{"id": f"c{step}", "function": {"name": "bash"}}]})
        request.append({"role": "tool", "tool_call_id": f"c{step}", "content": "готово"})
    return request


def test_what_memory_found_reaches_every_request_of_a_turn_that_uses_tools():
    store = _Store(found=[{"content": "переезд в декабре", "memory_type": "event", "created_at": None}])
    eng = _engine(store)

    requests = [eng.select_context(_tool_turn_request("что с переездом", steps)) for steps in (0, 1, 2)]

    for selected in requests:
        question = next(m for m in selected if m.get("role") == "user")
        assert "переезд в декабре" in question["content"]
    assert requests[0][1] == requests[1][1] == requests[2][1]
    assert store.recall_calls == 1               # искали один раз на ход, не на каждый запрос


def test_two_conversations_through_one_engine_keep_their_own_start_of_tail():
    eng = _engine(_Store())
    eng.total_budget = 40_000

    def conversation(tag: str, turns: int) -> list[dict]:
        request = [{"role": "system", "content": "личность"}]
        for index in range(turns):
            request += [{"role": "user", "content": f"{tag} вопрос {index} " + "в" * 1500},
                        {"role": "assistant", "content": f"{tag} ответ {index} " + "о" * 1500}]
        return request

    a = eng.select_context(conversation("A", 40))
    b = eng.select_context(conversation("B", 6))
    a_again = eng.select_context(conversation("A", 40))

    assert len(a) < 40 * 2                       # у длинного разговора хвост обрезан
    assert len(b) == 1 + 6 * 2                   # у короткого — цел, и чужое начало ему не мешает
    assert a_again == a                          # начало хвоста у A запомнено и не поехало


def _aging_request(turns: int) -> list[dict]:
    request = [{"role": "system", "content": "личность"}]
    for index in range(turns):
        request += [
            {"role": "user", "content": f"вопрос {index}"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{index}", "function": {"name": "read_file"}}]},
            {"role": "tool", "tool_call_id": f"c{index}", "content": "р" * 4_000},
            {"role": "assistant", "content": f"ответ {index}"},
        ]
    return request


def test_the_read_dedup_is_reset_only_when_more_results_have_been_released(monkeypatch):
    resets = []
    monkeypatch.setattr(AidaContextEngine, "_reset_read_dedup", staticmethod(lambda: resets.append(1)))
    eng = _engine(_Store())

    eng.select_context(_aging_request(2))       # ничего не состарено
    assert resets == []
    eng.select_context(_aging_request(4))       # два результата отпущены
    assert len(resets) == 1
    eng.select_context(_aging_request(4))       # тот же запрос ещё раз: новых замен нет
    assert len(resets) == 1
    eng.select_context(_aging_request(5))       # ещё один
    assert len(resets) == 2


def test_a_failing_dedup_reset_never_breaks_the_request(monkeypatch):
    import tools.file_tools_read_tracking as tracking

    def boom(_task_id=None):
        raise RuntimeError("учёт недоступен")

    monkeypatch.setattr(tracking, "reset_file_dedup", boom)

    selected = _engine(_Store()).select_context(_aging_request(4))

    assert selected and selected[-1]["role"] == "assistant" or selected


def test_the_window_of_the_model_sets_the_budget():
    eng = _engine(_Store())

    eng.select_context(_request(), budget_tokens=200_000)

    assert eng.total_budget == 90_000


# -- запрос без разговора ------------------------------------------------------------------


def _long_tool_chain(pairs: int = 40, size: int = 4000) -> list[dict]:
    """Разговор, каким он бывает в середине длинной работы: один вопрос и цепочка инструментов.

    Так выглядела сессия, на которой Гермес 2026-09-22 четырежды получил отказ «в запросе нет
    ни одного сообщения»: последняя реплика человека давно позади, всё после неё — пары
    «обращение — результат», и бюджету хватает только на их конец.
    """
    chain: list[dict] = [
        {"role": "system", "content": "личность"},
        {"role": "user", "content": "разбери, почему падает сборка"},
    ]
    for index in range(pairs):
        chain.append({"role": "assistant", "content": "",
                      "tool_calls": [{"id": f"c{index}", "function": {"name": "read_file"}}]})
        chain.append({"role": "tool", "tool_call_id": f"c{index}", "content": "х" * size})
    return chain


def test_a_long_tool_chain_never_leaves_the_request_without_a_conversation():
    selected = _engine(_Store()).select_context(_long_tool_chain(), budget_tokens=200_000)

    assert selected is not None
    assert any(m.get("role") == "user" and not m.get("tool_call_id") for m in selected)


def test_a_request_that_lost_its_conversation_is_handed_back_untouched(monkeypatch):
    # Если сборка всё же вернёт одну системную часть, отправлять это нельзя: поставщик
    # отвечает отказом, который не повторяют, и ход умирает целиком. Движок отходит в сторону.
    import plugins.context_engine.aida as module

    monkeypatch.setattr(module, "assemble_with_anchor",
                        lambda *args, **kwargs: ([{"role": "system", "content": "только знание"}], None, 0))

    assert _engine(_Store()).select_context(_request()) is None


def test_the_personality_of_the_host_stays_on_the_wire():
    # Знание внутри системной части хозяина, а не вторым системным сообщением: сборщик запроса
    # к Anthropic оставляет только последнее системное сообщение.
    store = _Store(knowledge=[{"content": "Оператор работает с ноутбука", "memory_type": "fact"}])

    selected = _engine(store).select_context(_request())

    systems = [m for m in selected if m.get("role") == "system"]
    assert len(systems) == 1
    assert systems[0]["content"].startswith("личность")
    assert "Оператор работает с ноутбука" in systems[0]["content"]
