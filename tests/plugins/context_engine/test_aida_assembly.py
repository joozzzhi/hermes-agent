from __future__ import annotations

from datetime import datetime

import pytest

from plugins.context_engine.aida.assembly import (
    MIN_TAIL_BUDGET,
    assemble,
    build_knowledge_block,
    build_old_history_block,
    build_recalled_block,
    is_clean_boundary,
    safe_tail,
    split_system,
)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _calls_tool(name: str = "bash") -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": name}}]}


def _tool_result(text: str = "готово") -> dict:
    return {"role": "tool", "tool_call_id": "c1", "content": text}


# -- системная часть ----------------------------------------------------------------------


def test_the_host_s_own_system_prompt_is_kept_and_kept_first():
    messages = [{"role": "system", "content": "личность"}, _user("привет")]

    head, rest = split_system(messages)

    assert head == [{"role": "system", "content": "личность"}]
    assert rest == [_user("привет")]


def test_a_conversation_without_a_system_prompt_is_still_understood():
    assert split_system([_user("привет")]) == ([], [_user("привет")])
    assert split_system([]) == ([], [])


# -- граница хвоста: главное опасное место -------------------------------------------------


def test_a_tool_result_is_never_left_without_the_request_that_caused_it():
    # Поставщик отвергает такой запрос целиком — разговор не деградирует, а встаёт.
    messages = [_user("почини"), _calls_tool(), _tool_result(), _assistant("починил"), _user("спасибо")]

    for budget in range(10, 400, 10):
        tail = safe_tail(messages, budget)
        assert not (tail and tail[0].get("role") == "tool"), f"осиротевший результат при бюджете {budget}"


def test_the_tail_starts_where_a_person_spoke():
    messages = [_user("первый"), _assistant("ответ"), _user("второй"), _assistant("ответ два")]

    tail = safe_tail(messages, 60)

    assert tail[0]["role"] == "user"


def test_an_assistant_reply_that_called_nothing_may_open_the_tail():
    assert is_clean_boundary([_assistant("просто ответ"), _user("дальше")], 0) is True


def test_an_assistant_reply_that_called_a_tool_may_not_open_the_tail():
    assert is_clean_boundary([_calls_tool(), _tool_result()], 0) is False


def test_a_whole_conversation_that_fits_is_kept_whole():
    messages = [_user("раз"), _assistant("два"), _user("три")]

    assert safe_tail(messages, 100_000) == messages


def test_the_question_survives_even_when_the_budget_is_absurdly_small():
    # Пустой хвост оставил бы модель без вопроса — это хуже любого превышения бюджета.
    messages = [_user("раз"), _calls_tool(), _tool_result(), _user("что там с переездом")]

    tail = safe_tail(messages, 1)

    assert tail and tail[-1]["content"] == "что там с переездом"


def test_an_empty_conversation_produces_an_empty_tail():
    assert safe_tail([], 1000) == []


def test_a_tail_made_entirely_of_tool_chatter_falls_back_to_the_last_message():
    messages = [_calls_tool(), _tool_result(), _calls_tool(), _tool_result("второй")]

    tail = safe_tail(messages, 10)

    assert len(tail) == 1 and tail[0]["content"] == "второй"


# -- слои ----------------------------------------------------------------------------------


def test_what_is_always_true_is_grouped_so_it_reads_as_a_map():
    rows = [
        {"content": "Оператор работает с ноутбука", "memory_type": "fact"},
        {"content": "Не пьёт кофе после четырёх", "memory_type": "preference"},
        {"content": "Гермес — главный агент", "memory_type": "decision"},
    ]

    block = build_knowledge_block(rows)

    assert "ФАКТЫ ОБ ОПЕРАТОРЕ" in block
    assert "ПРЕДПОЧТЕНИЯ ОПЕРАТОРА" in block
    assert "КЛЮЧЕВЫЕ РЕШЕНИЯ" in block


def test_knowledge_of_an_unknown_kind_still_reaches_the_model():
    assert "ПРОЧЕЕ" in build_knowledge_block([{"content": "нечто", "memory_type": "неизвестно"}])


def test_nothing_known_means_no_block_at_all():
    assert build_knowledge_block([]) == ""


def test_the_knowledge_block_stays_inside_its_budget():
    rows = [{"content": "я" * 500, "memory_type": "fact"} for _ in range(100)]

    assert len(build_knowledge_block(rows, budget=2000)) <= 2010


def test_old_conversation_arrives_as_a_quotation_not_a_retelling():
    rows = [{"content": "переезжаю в декабре", "memory_type": "dialogue",
             "role": "user", "created_at": datetime(2026, 8, 12)}]

    block = build_old_history_block(rows)

    assert "[2026-08-12] Оператор: переезжаю в декабре" in block


def test_who_said_it_is_not_guessed_when_it_was_the_agent():
    rows = [{"content": "записал", "memory_type": "dialogue",
             "role": "assistant", "created_at": datetime(2026, 8, 12)}]

    assert "Я: записал" in build_old_history_block(rows)


def test_recalled_entries_stay_inside_their_budget():
    rows = [{"content": "я" * 400, "memory_type": "fact", "created_at": None} for _ in range(50)]

    assert len(build_recalled_block(rows, budget=1500)) <= 1600


# -- сборка целиком ------------------------------------------------------------------------


def test_the_request_is_assembled_stable_first_changeable_with_the_question():
    system_head = [{"role": "system", "content": "личность"}]
    conversation = [_user("раньше"), _assistant("ответ"), _user("что там с переездом")]

    assembled = assemble(system_head, "# Что я знаю всегда\n- живёт на ноуте",
                         "# Вспомнил\n- переезд в декабре", "", conversation)

    assert assembled[0]["content"] == "личность"
    assert assembled[1]["role"] == "system" and "знаю всегда" in assembled[1]["content"]
    assert "Вспомнил" in assembled[-1]["content"]
    assert assembled[-1]["content"].endswith("что там с переездом")
    assert assembled[-1]["role"] == "user"


def test_no_system_message_is_ever_planted_in_the_middle_of_the_conversation():
    # Модель, на которой работает оператор, отвергает такое сообщение и обрывает ответ.
    conversation = [_user("раньше"), _assistant("ответ"), _user("вопрос")]

    assembled = assemble([{"role": "system", "content": "личность"}], "знание", "найденное", "", conversation)

    first_non_system = next(i for i, m in enumerate(assembled) if m.get("role") != "system")
    assert all(m.get("role") != "system" for m in assembled[first_non_system:])


def test_context_found_for_the_question_is_not_repeated_during_tool_work():
    # Внутри хода вопрос уже прозвучал: подмешивать найденное второй раз — это платить
    # за него снова и сбивать кеш на каждом обращении к инструменту.
    conversation = [_user("вопрос"), _calls_tool(), _tool_result()]

    assembled = assemble([], "", "# Вспомнил\n- деталь", "", conversation)

    assert not any("Вспомнил" in str(m.get("content", "")) for m in assembled)


def test_assembling_does_not_touch_the_conversation_it_was_given():
    conversation = [_user("вопрос")]
    snapshot = [dict(m) for m in conversation]

    assemble([], "знание", "найденное", "давнее", conversation)

    assert conversation == snapshot


def test_a_question_made_of_parts_keeps_its_parts():
    conversation = [{"role": "user", "content": [{"type": "text", "text": "вопрос"}]}]

    assembled = assemble([], "", "найденное", "", conversation)

    content = assembled[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["text"] == "найденное"
    assert content[-1]["text"] == "вопрос"


def test_the_tail_always_gets_room_even_when_the_layers_are_greedy():
    conversation = [_user("вопрос " + "я" * 1000) for _ in range(50)]

    assembled = assemble([], "з" * 100_000, "н" * 100_000, "д" * 100_000, conversation,
                         total_budget=150_000)

    tail = [m for m in assembled if m.get("role") != "system"]
    assert sum(len(str(m["content"])) for m in tail) >= MIN_TAIL_BUDGET * 0.5


@pytest.mark.parametrize("budget", [0, 1, 10, 150_000])
def test_any_budget_produces_a_request_that_is_still_a_conversation(budget):
    conversation = [_user("раз"), _assistant("два"), _user("три")]

    assembled = assemble([{"role": "system", "content": "личность"}], "", "", "", conversation,
                         total_budget=budget)

    assert assembled[-1]["role"] == "user"
    assert assembled[0]["role"] == "system"
