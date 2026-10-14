from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest

import plugins.context_engine.aida as engine_module
from plugins.context_engine.aida import AidaContextEngine


class _Store:
    """Стоит вместо общей базы: отдаёт ядро, что велено, и считает обращения."""

    def __init__(self, *, core=None, down: bool = False):
        self.core = core or []
        self.down = down
        self.core_calls = 0
        self.closed = False

    def core_rows(self):
        self.core_calls += 1
        if self.down:
            return None
        return [dict(row) for row in self.core]

    def recall(self, *args, **kwargs):  # pragma: no cover - a call here is the failure
        pytest.fail("the engine must not search memory — the memory provider does")

    def recall_tiers(self, *args, **kwargs):  # pragma: no cover
        pytest.fail("the engine must not search memory — the memory provider does")

    def close(self):
        self.closed = True


def _engine(store) -> AidaContextEngine:
    eng = AidaContextEngine()
    eng._store = store
    return eng


def _engine_without_memory() -> AidaContextEngine:
    eng = AidaContextEngine()
    eng._store = None
    eng._store_retry_at = time.monotonic() + 3600  # base not configured / not reachable
    return eng


def _request(question: str = "что там с переездом") -> list[dict]:
    return [
        {"role": "system", "content": "личность"},
        {"role": "user", "content": "раньше"},
        {"role": "assistant", "content": "ответ"},
        {"role": "user", "content": question},
    ]


def _core(*contents: str) -> list[dict]:
    return [{"content": c, "memory_type": "fact", "created_at": None} for c in contents]


def _system_text(selected: list[dict]) -> str:
    return "\n".join(str(m.get("content")) for m in selected if m.get("role") == "system")


# -- обычная работа ------------------------------------------------------------------------


def test_the_core_stands_in_the_system_part_and_the_conversation_follows_verbatim():
    store = _Store(core=_core("Оператор работает с ноутбука"))

    selected = _engine(store).select_context(_request())

    assert "Оператор работает с ноутбука" in _system_text(selected)
    assert selected[1:] == _request()[1:]


def test_the_engine_does_not_search_memory_for_the_question():
    # Найденное под вопрос приносит плагин памяти, и хозяин закрепляет его при реплике.
    store = _Store(core=_core("ядро"))

    selected = _engine(store).select_context(_request())

    assert selected[-1] == {"role": "user", "content": "что там с переездом"}


def test_what_memory_found_stays_with_its_question_byte_for_byte():
    recalled = "<memory-context>\n- [fact] переезд в декабре\n</memory-context>\n\nчто там с переездом"
    request = _request()[:-1] + [{"role": "user", "content": recalled}]

    selected = _engine(_Store()).select_context(request)

    assert selected[-1]["content"] == recalled


def test_the_persisted_conversation_is_never_touched():
    request = _request()
    snapshot = [dict(m) for m in request]

    _engine(_Store(core=_core("ядро"))).select_context(request)

    assert request == snapshot


def test_the_personality_of_the_host_stays_on_the_wire():
    # Ядро внутри системной части хозяина, а не вторым системным сообщением: сборщик запроса
    # к Anthropic оставляет только последнее системное сообщение.
    store = _Store(core=_core("Оператор работает с ноутбука"))

    selected = _engine(store).select_context(_request())

    systems = [m for m in selected if m.get("role") == "system"]
    assert len(systems) == 1
    assert systems[0]["content"].startswith("личность")
    assert "Оператор работает с ноутбука" in systems[0]["content"]


# -- запрос только дописывается ------------------------------------------------------------


def _growing_conversation(turns: int) -> list[dict]:
    request = [{"role": "system", "content": "личность"}]
    for index in range(turns):
        request += [{"role": "user", "content": f"вопрос {index}"},
                    {"role": "assistant", "content": f"ответ {index}"}]
    return request


def test_consecutive_requests_only_ever_grow_at_their_end():
    # Кеш поставщика жив, пока начало запроса совпадает байт в байт.
    store = _Store(core=_core("ядро один", "ядро два"))
    eng = _engine(store)

    previous = eng.select_context(_growing_conversation(3))
    for turns in range(4, 12):
        store.core = _core("в базе тем временем появилась новая запись P1")
        current = eng.select_context(_growing_conversation(turns))
        assert current[:len(previous)] == previous
        assert len(current) == len(previous) + 2
        previous = current


def test_the_core_is_read_once_and_not_on_every_request():
    store = _Store(core=_core("ядро"))
    eng = _engine(store)

    for turns in range(2, 8):
        eng.select_context(_growing_conversation(turns))

    assert store.core_calls == 1


def test_after_a_pause_longer_than_the_cache_the_core_is_read_again(monkeypatch):
    store = _Store(core=_core("старое ядро"))
    eng = _engine(store)
    clock = [1000.0]
    monkeypatch.setattr(engine_module, "time", type("_Clock", (), {"monotonic": staticmethod(lambda: clock[0])}))

    first = eng.select_context(_growing_conversation(3))
    store.core = _core("новое ядро")
    clock[0] += engine_module.DEFAULT_CACHE_TTL + 1
    second = eng.select_context(_growing_conversation(4))

    assert "старое ядро" in _system_text(first)
    assert "новое ядро" in _system_text(second)
    assert store.core_calls == 2
    # the tail fits, so the pause rebuilds the head without cutting the conversation
    assert second[1:] == _growing_conversation(4)[1:]


def test_a_core_that_could_not_be_read_is_asked_for_again_on_the_next_request():
    store = _Store(core=_core("ядро"), down=True)
    eng = _engine(store)

    first = eng.select_context(_request())
    store.down = False
    second = eng.select_context(_request())

    assert "ядро" not in _system_text(first)
    assert "ядро" in _system_text(second)


# -- когда что-то не так -------------------------------------------------------------------


def test_without_shared_memory_the_engine_still_keeps_the_tail():
    selected = _engine_without_memory().select_context(_request())

    assert selected == _request()


def test_a_database_that_fails_costs_the_core_not_the_answer():
    selected = _engine(_Store(down=True)).select_context(_request())

    assert selected == _request()


def test_an_empty_request_is_left_alone():
    assert _engine(_Store()).select_context([]) is None


def test_a_request_with_only_a_system_part_is_left_alone():
    assert _engine(_Store()).select_context([{"role": "system", "content": "личность"}]) is None


def test_a_request_with_no_question_in_it_still_assembles():
    store = _Store(core=_core("знание"))
    request = [{"role": "system", "content": "личность"}, {"role": "assistant", "content": "ответ"}]

    selected = _engine(store).select_context(request)

    assert selected is not None
    assert selected[-1] == {"role": "assistant", "content": "ответ"}


def test_several_threads_assembling_at_once_do_not_trip_over_each_other():
    eng = _engine(_Store(core=_core("ядро")))
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


def test_running_out_of_room_lets_old_messages_go_and_never_retells_them(monkeypatch):
    def _no_model(*args, **kwargs):  # pragma: no cover
        pytest.fail("compress() must never call a model")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", _no_model, raising=False)
    monkeypatch.setenv("AIDA_CONTEXT_TOKENS", "1000")
    eng = _engine(_Store())
    messages = [{"role": "system", "content": "личность"}]
    for i in range(60):
        messages += [{"role": "user", "content": f"реплика {i} " + "я" * 100},
                     {"role": "assistant", "content": f"ответ {i}"}]

    kept = eng.compress(messages)

    assert kept[0]["content"] == "личность"
    assert len(kept) < len(messages)
    assert kept[1]["role"] == "user"
    assert kept[-1] == messages[-1]
    assert all(m in messages for m in kept)  # ничего не пересказано и не выдумано
    assert eng.compression_count == 1


def test_compress_cuts_inside_a_long_turn_rather_than_giving_up(monkeypatch):
    monkeypatch.setenv("AIDA_CONTEXT_TOKENS", "2000")
    eng = _engine(_Store())
    messages = [{"role": "system", "content": "личность"}, {"role": "user", "content": "почини сборку"}]
    for index in range(200):
        messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{index}", "function": {"name": "bash"}}]},
                     {"role": "tool", "tool_call_id": f"c{index}", "content": "в" * 300}]

    kept = eng.compress(messages)

    assert len(kept) < len(messages)
    assert kept[1] == messages[1]
    assert kept[2]["role"] == "assistant"
    assert kept[-1] == messages[-1]


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


# -- бюджет и несколько разговоров ----------------------------------------------------------


def _long_conversation(tag: str, turns: int, size: int = 1500) -> list[dict]:
    request = [{"role": "system", "content": "личность"}]
    for index in range(turns):
        request += [{"role": "user", "content": f"{tag} вопрос {index} " + "в" * size},
                    {"role": "assistant", "content": f"{tag} ответ {index} " + "о" * size}]
    return request


def test_the_window_of_the_model_sets_the_budget():
    eng = _engine(_Store())
    conversation = _long_conversation("A", 80)  # ~240 000 знаков

    small = eng.select_context(conversation, budget_tokens=200_000)
    large = _engine(_Store()).select_context(conversation, budget_tokens=10_000_000)

    assert len(small) < len(conversation)
    assert large == conversation


def test_two_conversations_through_one_engine_keep_their_own_start_of_tail():
    eng = _engine(_Store())

    a = eng.select_context(_long_conversation("A", 80), budget_tokens=200_000)
    b = eng.select_context(_long_conversation("B", 6), budget_tokens=200_000)
    a_again = eng.select_context(_long_conversation("A", 80), budget_tokens=200_000)

    assert len(a) < 1 + 80 * 2                   # у длинного разговора хвост обрезан
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


def test_the_read_dedup_is_reset_when_results_have_been_released(monkeypatch):
    # Модель, которой понадобился отпущенный результат, перечитает файл — и должна получить
    # содержимое, а не «файл не менялся».
    # Бюджет задан в знаках (≈ 30k): 2 хода по ~4k помещаются, 12 — нет, при любом стартовом курсе.
    monkeypatch.setenv("AIDA_CONTEXT_TOKENS", str(int(30_000 / engine_module.DEFAULT_CHARS_PER_TOKEN)))
    resets = []
    monkeypatch.setattr(AidaContextEngine, "_reset_read_dedup", staticmethod(lambda: resets.append(1)))
    eng = _engine(_Store())

    eng.select_context(_aging_request(2))       # всё помещается, ничего не отпущено
    assert resets == []
    selected = eng.select_context(_aging_request(12))   # не помещается: пересборка с заглушками
    assert any("отпущен из запроса" in str(m.get("content")) for m in selected)
    assert len(resets) == 1
    eng.select_context(_aging_request(12))      # тот же запрос ещё раз: новых замен нет
    assert len(resets) == 1


def test_a_failing_dedup_reset_never_breaks_the_request(monkeypatch):
    import tools.file_tools_read_tracking as tracking

    def boom(_task_id=None):
        raise RuntimeError("учёт недоступен")

    monkeypatch.setattr(tracking, "reset_file_dedup", boom)
    monkeypatch.setenv("AIDA_CONTEXT_TOKENS", "10000")

    selected = _engine(_Store()).select_context(_aging_request(12))

    assert selected is not None and selected[-1]["content"] == "ответ 11"


# -- запрос без разговора ------------------------------------------------------------------


def _long_tool_chain(pairs: int = 40, size: int = 4000) -> list[dict]:
    """Разговор, каким он бывает в середине длинной работы: один вопрос и цепочка инструментов."""
    chain: list[dict] = [
        {"role": "system", "content": "личность"},
        {"role": "user", "content": "разбери, почему падает сборка"},
    ]
    for index in range(pairs):
        chain.append({"role": "assistant", "content": "",
                      "tool_calls": [{"id": f"c{index}", "function": {"name": "read_file"}}]})
        chain.append({"role": "tool", "tool_call_id": f"c{index}", "content": "х" * size})
    return chain


def test_a_long_tool_chain_never_leaves_the_request_without_a_conversation(monkeypatch):
    monkeypatch.setenv("AIDA_CONTEXT_TOKENS", "10000")

    selected = _engine(_Store()).select_context(_long_tool_chain(200, 400))

    assert selected is not None
    assert selected[1] == {"role": "user", "content": "разбери, почему падает сборка"}
    called = {c["id"] for m in selected for c in (m.get("tool_calls") or ())}
    assert all(m.get("tool_call_id") in called for m in selected if m.get("role") == "tool")


def test_a_request_that_lost_its_conversation_is_handed_back_untouched(monkeypatch):
    # Если сборка всё же вернёт одну системную часть, отправлять это нельзя: поставщик
    # отвечает отказом, который не повторяют, и ход умирает целиком. Движок отходит в сторону.
    monkeypatch.setattr(engine_module, "select_tail",
                        lambda messages, budget, state=None, force=False: ([], state, False))

    assert _engine(_Store()).select_context(_request()) is None


def test_an_assembly_that_blows_up_hands_the_request_back_untouched(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("сборка упала")

    monkeypatch.setattr(engine_module, "select_tail", boom)

    assert _engine(_Store()).select_context(_request()) is None


# -- вывод инструментов: потолок, диск, журнал ---------------------------------------------


def _web_session(exchanges: int = 12) -> list[dict]:
    """Проверка готовности: переписка, потом 5 веб-поисков и 2 тяжёлые страницы подряд."""
    request = [{"role": "system", "content": "личность " + "с" * 20_000}]
    for index in range(exchanges):
        request += [{"role": "user", "content": f"karma {index} " + "к" * 400},
                    {"role": "assistant", "content": f"жиши {index} " + "ж" * 1_200}]
    request.append({"role": "user", "content": "найди и сравни"})
    for index in range(7):
        name = "web_search" if index < 5 else "web_extract"
        size = 14_000 if index < 5 else 90_000
        body = f"{name} {index} начало\n" + "\n".join(f"строка {n} " + "в" * 60 for n in range(size // 70)) + "\nконец"
        request += [{"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"w{index}", "type": "function",
                                     "function": {"name": name, "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": f"w{index}", "name": name, "content": body}]
    request.append({"role": "assistant", "content": "сравнил"})
    return request


def test_after_heavy_web_work_the_talk_is_whole_and_every_output_is_capped(monkeypatch, caplog):
    monkeypatch.setenv("AIDA_CONTEXT_TOKENS", "60000")
    request = _web_session()
    eng = _engine(_Store())

    with caplog.at_level("INFO", logger="plugins.context_engine.aida"):
        selected = eng.select_context(request, budget_tokens=1_000_000)
        eng.on_turn_complete(request)

    questions = [m["content"] for m in request if m["role"] == "user"]
    sent = [m["content"] for m in selected if m["role"] == "user"]
    assert sent[-engine_module.PROTECTED_EXCHANGES:] == questions[-engine_module.PROTECTED_EXCHANGES:]
    for original, out in zip([m for m in request if m["role"] == "tool"], [m for m in selected if m["role"] == "tool"]):
        assert len(out["content"]) <= engine_module.TOOL_RESULT_CAP
        path = re.search(r"полный — (\S+?);", out["content"]) or re.search(r"полный вывод: (\S+)", out["content"])
        if len(original["content"]) > engine_module.TOOL_RESULT_CAP:
            assert path, out["content"][:200]
            assert Path(path.group(1)).read_text(encoding="utf-8") == original["content"]
            if out["content"].startswith("[вывод урезан"):      # свежий — начало и конец
                assert out["content"].endswith("конец")
            else:                                             # старый — заглушка со ссылкой
                assert "отпущен из запроса" in out["content"]
    line = next(r.getMessage() for r in caplog.records if "ход — ядро" in r.getMessage())
    assert "переписка" in line and "инструменты" in line and "заглушек" in line


def test_capping_gives_the_same_bytes_on_every_request(monkeypatch):
    monkeypatch.setenv("AIDA_CONTEXT_TOKENS", "60000")
    request = _web_session()
    eng = _engine(_Store())

    first = eng.select_context(request, budget_tokens=1_000_000)
    again = eng.select_context(request + [{"role": "user", "content": "ещё"}], budget_tokens=1_000_000)
    restarted = _engine(_Store()).select_context(request, budget_tokens=1_000_000)

    assert again[:len(first)] == first
    assert restarted == first


def test_an_output_that_could_not_be_saved_is_left_whole(monkeypatch):
    monkeypatch.setattr(AidaContextEngine, "_saved_output", lambda self, state, message, content: None)
    request = [{"role": "system", "content": "личность"}, {"role": "user", "content": "читай"},
               {"role": "assistant", "content": "", "tool_calls": [{"id": "r1", "function": {"name": "read_file"}}]},
               {"role": "tool", "tool_call_id": "r1", "content": "р" * 30_000}]

    selected = _engine(_Store()).select_context(request)

    assert selected[-1]["content"] == "р" * 30_000


def _long_tool_request(size: int) -> list[dict]:
    return [
        {"role": "system", "content": "личность"},
        {"role": "user", "content": "прочитай файл"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "р" * size},
    ]


def test_each_profile_gets_its_own_settings_from_its_own_config(tmp_path, monkeypatch):
    # Один шлюз, одна витрина, два профиля: потолок вывода берётся из config.yaml того профиля,
    # чей это ход, а не из того, кто ходил первым. A → B → A.
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.delenv("AIDA_CONTEXT_TOKENS", raising=False)
    tight, loose = tmp_path / "tight", tmp_path / "loose"
    for home, cap in ((tight, 5_000), (loose, 50_000)):
        home.mkdir()
        (home / "config.yaml").write_text(f"aida:\n  tool_result_cap: {cap}\n", encoding="utf-8")
    eng = _engine(_Store())

    def longest_output(home: Path) -> int:
        token = set_hermes_home_override(home)
        try:
            out = eng.select_context(_long_tool_request(20_000))
        finally:
            reset_hermes_home_override(token)
        return max(len(m["content"]) for m in out if m.get("role") == "tool")

    assert longest_output(tight) <= 5_000
    assert longest_output(loose) == 20_000
    assert longest_output(tight) <= 5_000


def test_without_a_section_the_settings_are_the_defaults(tmp_path, monkeypatch):
    # Нет секции aida: — значения те же, что стояли константами; кривое значение не роняет ход.
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.context_engine.aida.settings import DEFAULTS, load_settings

    for name in ("AIDA_CONTEXT_TOKENS", "AIDA_CONTEXT_SHARE", "AIDA_CACHE_TTL_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    plain, broken = tmp_path / "plain", tmp_path / "broken"
    plain.mkdir(); broken.mkdir()
    (plain / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    (broken / "config.yaml").write_text("aida:\n  protected_exchanges: много\n  conversation_share: 0.3\n",
                                        encoding="utf-8")
    for home, expected in ((plain, DEFAULTS),
                           (broken, DEFAULTS.__class__(conversation_share=0.3))):
        token = set_hermes_home_override(home)
        try:
            assert load_settings() == expected
        finally:
            reset_hermes_home_override(token)
