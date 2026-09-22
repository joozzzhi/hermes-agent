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


def test_a_tail_made_entirely_of_tool_chatter_never_ends_up_a_lone_orphan():
    # Прежде здесь отдавалось последнее сообщение — результат инструмента без своего
    # обращения. Чистилка запроса выбрасывает такой результат как сироту, и к модели уходит
    # запрос без единого сообщения: поставщик отвечает отказом, который не повторяют, и ход
    # умирает целиком. Выйти за бюджет — меньшее зло.
    messages = [_calls_tool(), _tool_result(), _calls_tool(), _tool_result("второй")]

    tail = safe_tail(messages, 10)

    assert tail == messages


def test_the_tail_steps_back_to_the_last_question_when_the_chain_has_no_clean_start():
    # Бюджета хватает только на конец длинной цепочки, а чистая граница осталась позади:
    # хвост должен начаться с реплики человека, а не с середины пары.
    messages = [_user("почини сборку"), _calls_tool(), _tool_result("х" * 400),
                _calls_tool(), _tool_result("у" * 400)]

    tail = safe_tail(messages, 50)

    assert tail[0] == _user("почини сборку")
    assert is_clean_boundary(tail, 0)


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

    # Знание едет ВНУТРИ системной части хозяина, а не вторым системным сообщением: сборщик
    # запроса к Anthropic оставляет только последнее системное сообщение, и второе стёрло бы
    # личность с провода — молча, без ошибки.
    assert assembled[0]["role"] == "system"
    assert assembled[0]["content"].startswith("личность")
    assert "знаю всегда" in assembled[0]["content"]
    assert sum(1 for m in assembled if m.get("role") == "system") == 1
    assert "Вспомнил" in assembled[-1]["content"]
    assert assembled[-1]["content"].endswith("что там с переездом")
    assert assembled[-1]["role"] == "user"


def test_no_system_message_is_ever_planted_in_the_middle_of_the_conversation():
    # Модель, на которой работает оператор, отвергает такое сообщение и обрывает ответ.
    conversation = [_user("раньше"), _assistant("ответ"), _user("вопрос")]

    assembled = assemble([{"role": "system", "content": "личность"}], "знание", "найденное", "", conversation)

    first_non_system = next(i for i, m in enumerate(assembled) if m.get("role") != "system")
    assert all(m.get("role") != "system" for m in assembled[first_non_system:])


def test_context_found_for_the_question_stays_with_it_during_tool_work():
    # Ход — это несколько запросов, а хозяин историю после подмены не меняет: подмешанное
    # живёт один запрос. Ответ пишется на последнем запросе хода, уже после инструментов, и не
    # должен остаться без найденного.
    conversation = [_user("вопрос"), _calls_tool(), _tool_result()]

    assembled = assemble([], "", "# Вспомнил\n- деталь", "", conversation)

    question = next(m for m in assembled if m.get("role") == "user")
    assert "Вспомнил" in question["content"] and question["content"].endswith("вопрос")
    assert not any("Вспомнил" in str(m.get("content", "")) for m in assembled if m is not question)


def test_the_question_is_byte_identical_on_every_request_of_the_turn():
    # Одинаковые байты — условие, при котором кеш поставщика не ломается внутри хода.
    turn = [_user("вопрос")]
    first = assemble([], "", "# Вспомнил\n- деталь", "", turn)
    turn += [_calls_tool(), _tool_result()]
    second = assemble([], "", "# Вспомнил\n- деталь", "", turn)
    turn += [_calls_tool(), _tool_result("ещё")]
    third = assemble([], "", "# Вспомнил\n- деталь", "", turn)

    assert first[0] == second[0] == third[0]


def test_only_the_current_question_carries_what_was_found_not_the_older_ones():
    conversation = [_user("старый вопрос"), _assistant("ответ"), _user("новый вопрос"),
                    _calls_tool(), _tool_result()]

    assembled = assemble([], "", "# Вспомнил\n- деталь", "", conversation)

    assert assembled[0]["content"] == "старый вопрос"
    assert "Вспомнил" in assembled[2]["content"]


def test_a_tail_with_no_person_in_it_gets_no_found_block_and_no_crash():
    assembled = assemble([], "", "# Вспомнил\n- деталь", "", [_calls_tool(), _tool_result()],
                         total_budget=1)

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


# -- ступенчатое начало хвоста -------------------------------------------------------------


def _turns(count: int, size: int = 1000) -> list[dict]:
    """count ходов «вопрос — ответ» по size знаков каждая реплика, у каждого вопроса свой текст."""
    out: list[dict] = []
    for index in range(count):
        out.append(_user(f"вопрос {index} " + "в" * size))
        out.append(_assistant(f"ответ {index} " + "о" * size))
    return out


def test_the_start_of_the_tail_stays_put_while_the_tail_fits():
    from plugins.context_engine.aida.assembly import select_tail

    conversation = _turns(10)
    tail, anchor = select_tail(conversation, budget=30_000)
    grown = conversation + _turns(1)[:1]
    tail2, anchor2 = select_tail(grown, budget=30_000, anchor=anchor)

    assert anchor2 == anchor
    assert tail2[0] == tail[0]
    assert len(tail2) == len(tail) + 1


def test_when_the_tail_no_longer_fits_it_steps_forward_in_one_big_move():
    from plugins.context_engine.aida.assembly import STEP_KEEP, message_size, select_tail

    conversation = _turns(20)                    # ~40 000 знаков
    tail, anchor = select_tail(conversation, budget=30_000)
    assert sum(message_size(m) for m in tail) <= 30_000

    # добавляем ходы, пока не потребуется сдвиг, и запоминаем, сколько раз начало двигалось
    starts = [tail[0]["content"]]
    grown = list(conversation)
    for index in range(20, 40):
        grown += [_user(f"вопрос {index} " + "в" * 1000), _assistant(f"ответ {index} " + "о" * 1000)]
        tail, anchor = select_tail(grown, budget=30_000, anchor=anchor)
        if tail[0]["content"] != starts[-1]:
            starts.append(tail[0]["content"])
            # после шага хвост заметно короче бюджета: следующий сдвиг будет не скоро
            assert sum(message_size(m) for m in tail) <= 30_000 * STEP_KEEP + 2_100

    assert 1 < len(starts) <= 4                  # начало сдвинулось несколько раз, а не двадцать


def test_a_stepped_tail_never_splits_a_tool_pair():
    from plugins.context_engine.aida.assembly import select_tail

    conversation = []
    for index in range(30):
        conversation += [_user(f"вопрос {index}"), _calls_tool(), _tool_result("р" * 2_000), _assistant("ок")]
    anchor = None
    for cut in range(8, len(conversation) + 1, 3):
        tail, anchor = select_tail(conversation[:cut], budget=8_000, anchor=anchor)
        assert tail[0].get("role") != "tool"
        called = {c["id"] for m in tail for c in (m.get("tool_calls") or ())}
        assert all(m.get("tool_call_id") in called for m in tail if m.get("role") == "tool")


def test_a_lost_anchor_falls_back_to_a_fresh_cut_instead_of_failing():
    from plugins.context_engine.aida.assembly import select_tail

    conversation = _turns(10)
    tail, anchor = select_tail(conversation, budget=30_000, anchor=12345)

    assert tail and anchor != 12345


def test_an_old_line_of_conversation_is_not_cut_at_four_hundred_signs():
    long_line = "я думаю о переезде " * 60          # ~1 100 знаков
    rows = [{"content": long_line, "memory_type": "dialogue", "role": "user",
             "created_at": datetime(2026, 8, 12)}]

    block = build_old_history_block(rows)

    assert long_line.strip() in block and "[…]" not in block


# -- старение выхлопа инструментов и бюджет от окна ----------------------------------------


def _work_turn(index: int, result_size: int = 5_000) -> list[dict]:
    call_id = f"c{index}"
    return [
        _user(f"вопрос {index}"),
        {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": call_id, "name": "read_file", "content": "р" * result_size},
        _assistant(f"ответ {index}"),
    ]


def test_old_long_tool_results_are_replaced_and_recent_ones_are_left_whole():
    from plugins.context_engine.aida.assembly import age_tool_results

    conversation = [m for i in range(5) for m in _work_turn(i)]

    aged, count = age_tool_results(conversation)

    results = [m for m in aged if m.get("role") == "tool"]
    assert count == 3                                   # ходы 0, 1, 2 состарены; 3 и 4 целые
    assert all("отпущен из запроса" in m["content"] for m in results[:3])
    assert all(m["content"] == "р" * 5_000 for m in results[3:])
    assert "5000 знаков" in results[0]["content"] and "read_file" in results[0]["content"]


def test_aging_keeps_every_tool_pair_intact_and_leaves_the_input_alone():
    from plugins.context_engine.aida.assembly import age_tool_results

    conversation = [m for i in range(5) for m in _work_turn(i)]
    snapshot = [dict(m) for m in conversation]

    aged, _ = age_tool_results(conversation)

    assert conversation == snapshot
    assert [m.get("tool_call_id") for m in aged] == [m.get("tool_call_id") for m in conversation]
    assert [m.get("role") for m in aged] == [m.get("role") for m in conversation]
    assert [m.get("tool_calls") for m in aged] == [m.get("tool_calls") for m in conversation]


def test_short_results_and_results_of_the_current_turn_are_never_aged():
    from plugins.context_engine.aida.assembly import age_tool_results

    short = [m for i in range(5) for m in _work_turn(i, result_size=400)]
    assert age_tool_results(short) == (short, 0)

    current = [_user("вопрос"), _calls_tool(), _tool_result("р" * 15_000)]
    assert age_tool_results(current) == (current, 0)


def test_results_that_are_not_text_are_left_alone():
    from plugins.context_engine.aida.assembly import age_tool_results

    conversation = [m for i in range(5) for m in _work_turn(i)]
    conversation[2] = {**conversation[2], "content": [{"type": "image", "source": "..."}]}

    aged, count = age_tool_results(conversation)

    assert aged[2]["content"] == conversation[2]["content"] and count == 2


def test_with_aging_the_same_budget_holds_many_more_turns():
    conversation = [m for i in range(60) for m in _work_turn(i)]

    def person_turns(age: bool) -> int:
        request = assemble([], "", "", "", conversation, total_budget=60_000, age_tools=age)
        return sum(1 for m in request if m.get("role") == "user")

    assert person_turns(True) >= 3 * person_turns(False)


def test_aged_request_still_never_starts_with_an_orphaned_result():
    conversation = [m for i in range(30) for m in _work_turn(i)]

    for budget in (25_000, 40_000, 60_000):
        request = assemble([], "", "", "", conversation, total_budget=budget)
        called = {c["id"] for m in request for c in (m.get("tool_calls") or ())}
        assert all(m.get("tool_call_id") in called for m in request if m.get("role") == "tool")


def test_the_budget_follows_the_window_of_the_model():
    from plugins.context_engine.aida.assembly import TOTAL_BUDGET, total_budget_for

    assert total_budget_for(1_000_000) == 150_000 == TOTAL_BUDGET   # для окна в миллион — прежнее число
    assert total_budget_for(2_000_000) == 300_000
    assert total_budget_for(200_000) == 90_000                       # не ниже пола
    assert total_budget_for(0) == TOTAL_BUDGET and total_budget_for(None) == TOTAL_BUDGET
