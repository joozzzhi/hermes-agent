from __future__ import annotations

import pytest

import json

from plugins.context_engine.aida.assembly import (
    CONVERSATION_SHARE,
    PROTECTED_EXCHANGES,
    STEP_KEEP,
    breakdown,

    age_before,
    apply_state,
    build_knowledge_block,
    is_conversation_message,
    merge_into_system,
    message_size,
    rebase,
    select_tail,
    split_system,
    tail_size,
)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _calls_tool(call_id: str = "c1", name: str = "bash") -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "function": {"name": name}}]}


def _tool_result(text: str = "готово", call_id: str = "c1", name: str | None = None) -> dict:
    message = {"role": "tool", "tool_call_id": call_id, "content": text}
    if name:
        message["name"] = name
    return message


def _turns(count: int, size: int = 1000, start: int = 0) -> list[dict]:
    """count ходов «вопрос — ответ», у каждой реплики свой текст длиной около size."""
    out: list[dict] = []
    for index in range(start, start + count):
        out.append(_user(f"вопрос {index} " + "в" * size))
        out.append(_assistant(f"ответ {index} " + "о" * size))
    return out


def _work_turn(index: int, result_size: int = 5_000) -> list[dict]:
    call_id = f"c{index}"
    return [
        _user(f"вопрос {index}"),
        _calls_tool(call_id, "read_file"),
        _tool_result("р" * result_size, call_id, "read_file"),
        _assistant(f"ответ {index}"),
    ]


def _pairs_intact(tail: list[dict]) -> bool:
    called = {c["id"] for m in tail for c in (m.get("tool_calls") or ())}
    return all(m.get("tool_call_id") in called for m in tail if m.get("role") == "tool")


# -- системная часть ----------------------------------------------------------------------


def test_the_host_s_own_system_prompt_is_kept_and_kept_first():
    messages = [{"role": "system", "content": "личность"}, _user("привет")]

    head, rest = split_system(messages)

    assert head == [{"role": "system", "content": "личность"}]
    assert rest == [_user("привет")]


def test_a_conversation_without_a_system_prompt_is_still_understood():
    assert split_system([_user("привет")]) == ([], [_user("привет")])
    assert split_system([]) == ([], [])


def test_the_core_rides_inside_the_host_s_system_message_not_as_a_second_one():
    # Сборщик запроса к Anthropic оставляет только последнее системное сообщение: второе
    # стёрло бы личность хозяина с провода — молча, без ошибки.
    merged = merge_into_system([{"role": "system", "content": "личность"}], "# Что я знаю всегда")

    assert len(merged) == 1
    assert merged[0]["content"].startswith("личность")
    assert merged[0]["content"].endswith("# Что я знаю всегда")


def test_a_system_message_made_of_parts_keeps_its_parts():
    head = [{"role": "system", "content": [{"type": "text", "text": "личность"}]}]

    merged = merge_into_system(head, "ядро")

    assert merged[0]["content"][0] == {"type": "text", "text": "личность"}
    assert merged[0]["content"][-1] == {"type": "text", "text": "ядро"}
    assert head == [{"role": "system", "content": [{"type": "text", "text": "личность"}]}]


def test_no_core_leaves_the_host_s_system_part_untouched():
    head = [{"role": "system", "content": "личность"}]

    assert merge_into_system(head, "") == head
    assert merge_into_system([], "ядро") == [{"role": "system", "content": "ядро"}]


def test_only_real_messages_count_as_a_conversation():
    assert is_conversation_message(_user("привет"))
    assert is_conversation_message(_assistant("ответ"))
    assert not is_conversation_message({"role": "system", "content": "личность"})
    assert not is_conversation_message(_tool_result())


# -- ядро ----------------------------------------------------------------------------------


def test_what_is_always_true_is_grouped_so_it_reads_as_a_map():
    rows = [
        {"content": "Оператор работает с ноутбука", "memory_type": "fact"},
        {"content": "Не пьёт кофе после четырёх", "memory_type": "preference"},
        {"content": "Гермес — главный агент", "memory_type": "decision"},
    ]

    block = build_knowledge_block(rows, 10_000)

    assert block.startswith("# Что я знаю всегда")
    for label in ("ФАКТЫ ОБ ОПЕРАТОРЕ", "ПРЕДПОЧТЕНИЯ ОПЕРАТОРА", "КЛЮЧЕВЫЕ РЕШЕНИЯ"):
        assert label in block


def test_the_order_of_groups_does_not_depend_on_which_entry_is_newer():
    # Блок стоит в начале запроса: перетасовка групп сломала бы кеш поставщика.
    a = {"content": "правило", "memory_type": "rule"}
    b = {"content": "факт", "memory_type": "fact"}

    assert build_knowledge_block([a, b], 10_000) == build_knowledge_block([b, a], 10_000)


def test_knowledge_of_an_unknown_kind_still_reaches_the_model():
    assert "ПРОЧЕЕ" in build_knowledge_block([{"content": "нечто", "memory_type": "неизвестно"}], 1000)


def test_nothing_known_means_no_block_at_all():
    assert build_knowledge_block([], 1000) == ""
    assert build_knowledge_block([{"content": "   ", "memory_type": "fact"}], 1000) == ""


def test_an_entry_that_does_not_fit_is_skipped_whole_never_cut():
    rows = [{"content": "я" * 500, "memory_type": "fact"},
            {"content": "длинное правило " * 100, "memory_type": "rule"},
            {"content": "короткое", "memory_type": "fact"}]

    block = build_knowledge_block(rows, 600)

    assert "я" * 500 in block and "короткое" in block
    assert "длинное правило" not in block


def test_the_core_stays_inside_its_budget():
    rows = [{"content": "я" * 500, "memory_type": "fact"} for _ in range(100)]

    block = build_knowledge_block(rows, 2000)

    assert 0 < block.count("я" * 500) <= 2000 // 500


# -- граница хвоста: пара не рвётся --------------------------------------------------------


def test_a_tool_result_is_never_left_without_the_request_that_caused_it():
    messages = [_user("почини"), _calls_tool(), _tool_result(), _assistant("починил"), _user("спасибо")]

    for budget in range(10, 400, 10):
        tail, _, _ = select_tail(messages, budget)
        assert tail and tail[0].get("role") != "tool", f"осиротевший результат при бюджете {budget}"
        assert _pairs_intact(tail)


def test_the_tail_starts_where_a_person_spoke():
    tail, _, _ = select_tail(_turns(6), 2_500)

    assert tail[0]["role"] == "user"
    assert tail_size(tail) <= 2_500


def test_a_whole_conversation_that_fits_is_kept_whole():
    messages = [_user("раз"), _assistant("два"), _user("три")]

    tail, _, _ = select_tail(messages, 100_000)

    assert tail == messages


def test_the_question_survives_even_when_the_budget_is_absurdly_small():
    # Пустой хвост оставил бы модель без вопроса — это хуже любого превышения бюджета.
    messages = [_user("раз"), _calls_tool(), _tool_result(), _user("что там с переездом")]

    tail, _, _ = select_tail(messages, 1)

    assert tail and tail[-1]["content"] == "что там с переездом"


def test_an_empty_conversation_produces_an_empty_tail():
    tail, _, _ = select_tail([], 1000)

    assert tail == []


def test_a_conversation_with_no_person_in_it_is_handed_over_as_is():
    messages = [_calls_tool(), _tool_result(), _calls_tool("c2"), _tool_result("второй", "c2")]

    tail, _, _ = select_tail(messages, 10)

    assert tail == messages


@pytest.mark.parametrize("budget", [0, 1, 10, 150_000])
def test_any_budget_produces_a_tail_that_ends_with_the_question(budget):
    conversation = [_user("раз"), _assistant("два"), _user("три")]

    tail, _, _ = select_tail(conversation, budget)

    assert tail[-1] == _user("три")
    assert is_conversation_message(tail[0])


def test_selecting_a_tail_does_not_touch_the_conversation_it_was_given():
    conversation = [m for i in range(6) for m in _work_turn(i)]
    snapshot = [dict(m) for m in conversation]

    select_tail(conversation, 8_000)

    assert conversation == snapshot


# -- запрос только дописывается --------------------------------------------------------------


def test_while_the_tail_fits_each_request_is_the_previous_one_plus_what_was_said():
    conversation = _turns(4)
    tail, state, _ = select_tail(conversation, 30_000)

    for step in range(4, 10):
        conversation = conversation + _turns(1, start=step)
        new_tail, state, rebased = select_tail(conversation, 30_000, state)
        assert not rebased
        assert new_tail[:len(tail)] == tail
        assert len(new_tail) == len(tail) + 2
        tail = new_tail


def test_the_same_decisions_give_the_same_bytes():
    conversation = [m for i in range(10) for m in _work_turn(i)]
    state = rebase(conversation, 12_000)

    assert apply_state(conversation, state) == apply_state(list(conversation), state)


def test_when_the_tail_no_longer_fits_it_steps_forward_in_one_big_move():
    budget = 30_000
    conversation = _turns(10)
    tail, state, _ = select_tail(conversation, budget)
    rebases = 0

    for step in range(10, 40):
        conversation = conversation + _turns(1, start=step)
        tail, state, rebased = select_tail(conversation, budget, state)
        assert tail_size(tail) <= budget
        if rebased:
            rebases += 1
            # после пересборки хвост заметно короче бюджета: следующая будет не скоро
            assert tail_size(tail) <= budget * STEP_KEEP

    assert 1 <= rebases <= 4  # начало сдвинулось несколько раз, а не на каждом ходу


def test_a_pause_longer_than_the_cache_rebuilds_but_does_not_cut_a_tail_that_fits():
    conversation = _turns(5)
    tail, state, _ = select_tail(conversation, 30_000)

    again, _, rebased = select_tail(conversation, 30_000, state, force=True)

    assert rebased
    assert again == tail


def test_a_conversation_that_is_not_the_one_remembered_gets_a_fresh_cut():
    _, state, _ = select_tail(_turns(10), 30_000)
    other = [_user("совсем другой разговор"), _assistant("да")]

    tail, fresh, rebased = select_tail(other, 30_000, state)

    assert rebased and tail == other
    assert fresh.fits(other)


def test_a_remembered_place_recognises_its_conversation_only():
    conversation = _turns(3)
    state = rebase(conversation, 30_000)

    assert state.fits(conversation)
    assert state.fits(conversation + _turns(1, start=3))
    assert not state.fits([_user("другое")])
    assert not state.fits([])


def test_a_stepped_tail_never_splits_a_tool_pair():
    conversation = []
    for index in range(30):
        conversation += [_user(f"вопрос {index}"), _calls_tool(f"c{index}"),
                         _tool_result("р" * 2_000, f"c{index}"), _assistant("ок")]
    state = None
    for cut in range(8, len(conversation) + 1, 3):
        tail, state, _ = select_tail(conversation[:cut], 8_000, state)
        assert tail[0].get("role") != "tool"
        assert _pairs_intact(tail)


# -- длинный ход -------------------------------------------------------------------------------


def _long_turn(pairs: int = 60, size: int = 3_000) -> list[dict]:
    turn = [_user("разбери, почему падает сборка")]
    for index in range(pairs):
        turn += [_calls_tool(f"c{index}", "read_file"), _tool_result("х" * size, f"c{index}", "read_file")]
    return turn


def test_a_turn_longer_than_the_budget_keeps_its_question_and_its_freshest_work():
    # Results too short to be stubbed out: only cutting the middle of the turn can help.
    turn = _long_turn(pairs=200, size=400)

    tail, _, _ = select_tail(turn, 30_000)

    assert tail[0] == turn[0]
    assert tail[-1] == turn[-1]
    assert len(tail) < len(turn)
    assert tail[1]["role"] == "assistant"  # середина выброшена целыми связками
    assert _pairs_intact(tail)
    assert tail_size(tail) <= 30_000


def test_a_long_turn_that_keeps_growing_is_still_cut_cleanly():
    turn = _long_turn(10)
    state = None
    for index in range(10, 80):
        turn = turn + [_calls_tool(f"c{index}", "read_file"),
                       _tool_result("х" * 3_000, f"c{index}", "read_file")]
        tail, state, _ = select_tail(turn, 30_000, state)
        assert tail[0] == turn[0]
        assert _pairs_intact(tail)
        assert tail_size(tail) <= 30_000


# -- старение выхлопа инструментов --------------------------------------------------------------


def test_old_long_tool_results_are_replaced_and_recent_ones_are_left_whole():
    conversation = [m for i in range(5) for m in _work_turn(i)]

    aged = age_before(conversation, 8)  # первые два хода

    results = [m for m in aged if m.get("role") == "tool"]
    assert all("отпущен из запроса" in m["content"] for m in results[:2])
    assert all(m["content"] == "р" * 5_000 for m in results[2:])
    assert "5000 знаков" in results[0]["content"] and "read_file" in results[0]["content"]


def test_aging_keeps_every_tool_pair_intact_and_leaves_the_input_alone():
    conversation = [m for i in range(5) for m in _work_turn(i)]
    snapshot = [dict(m) for m in conversation]

    aged = age_before(conversation, len(conversation))

    assert conversation == snapshot
    assert [m.get("tool_call_id") for m in aged] == [m.get("tool_call_id") for m in conversation]
    assert [m.get("role") for m in aged] == [m.get("role") for m in conversation]
    assert [m.get("tool_calls") for m in aged] == [m.get("tool_calls") for m in conversation]


def test_short_results_are_never_aged():
    short = [m for i in range(5) for m in _work_turn(i, result_size=400)]

    assert age_before(short, len(short)) == short


def test_results_that_are_not_text_are_left_alone():
    conversation = [m for i in range(5) for m in _work_turn(i)]
    conversation[2] = {**conversation[2], "content": [{"type": "image", "source": "..."}]}

    aged = age_before(conversation, len(conversation))

    assert aged[2]["content"] == conversation[2]["content"]


def test_what_memory_found_for_an_old_question_is_let_go_and_the_question_stays():
    recalled = "<memory-context>\n" + "найденное " * 100 + "\n</memory-context>\n\n"
    conversation = [_user(recalled + "старый вопрос"), _assistant("ответ"), _user("новый вопрос")]

    aged = age_before(conversation, 1)

    assert aged[0]["content"] == "старый вопрос"
    assert conversation[0]["content"].startswith("<memory-context>")


def test_with_aging_the_same_budget_holds_many_more_turns():
    conversation = [m for i in range(60) for m in _work_turn(i)]

    aged_tail = apply_state(conversation, rebase(conversation, 60_000))
    raw_tail = apply_state(conversation, rebase(conversation, 60_000, age=False))

    def person_turns(tail):
        return sum(1 for m in tail if m.get("role") == "user")

    assert person_turns(aged_tail) >= 3 * person_turns(raw_tail)


def test_the_freshest_result_is_always_left_whole():
    # Модель его ещё не читала.
    conversation = [_user("вопрос"), _calls_tool(), _tool_result("р" * 50_000)]

    tail = apply_state(conversation, rebase(conversation, 10_000))

    assert tail[-1]["content"] == "р" * 50_000


def test_an_aged_tail_still_never_starts_with_an_orphaned_result():
    conversation = [m for i in range(30) for m in _work_turn(i)]

    for budget in (25_000, 40_000, 60_000):
        tail, _, _ = select_tail(conversation, budget)
        assert tail[0]["role"] == "user"
        assert _pairs_intact(tail)
        assert sum(message_size(m) for m in tail) <= budget


# -- переписка под защитой ----------------------------------------------------------------------


def _talk(index: int, size: int = 1_500) -> list[dict]:
    return [_user(f"karma {index} " + "к" * size), _assistant(f"жиши {index} " + "ж" * size)]


def _tool_turn(index: int, calls: int = 6, size: int = 12_000) -> list[dict]:
    turn = [_user(f"karma задача {index}")]
    for c in range(calls):
        call_id = f"t{index}_{c}"
        turn += [_calls_tool(call_id, "terminal"), _tool_result("т" * size, call_id, "terminal")]
    turn.append(_assistant(f"жиши итог {index}"))
    return turn


def _exchanges(tail: list[dict]) -> list[str]:
    return [m["content"] for m in tail if m.get("role") == "user" and not m.get("tool_call_id")]


def test_the_last_exchanges_survive_any_amount_of_tool_work():
    # Вчерашний случай: несколько ходов с терминалом вытеснили переписку до трёх сообщений.
    budget = 145_000
    conversation = [m for i in range(20) for m in _talk(i)]
    state = None
    for index in range(12):
        conversation = conversation + _tool_turn(index)
        tail, state, _ = select_tail(conversation, budget, state)
        questions = [m["content"] for m in conversation if m.get("role") == "user" and not m.get("tool_call_id")]
        assert _exchanges(tail)[-PROTECTED_EXCHANGES:] == questions[-PROTECTED_EXCHANGES:]
        assert tail_size(tail) <= budget
        assert _pairs_intact(tail)


def test_protected_answers_stay_verbatim_and_only_silent_tool_pairs_go():
    budget = 60_000
    conversation = [m for i in range(6) for m in _talk(i)] + [m for i in range(6) for m in _tool_turn(i)] + _talk(99)

    tail, state, _ = select_tail(conversation, budget)

    words = [m for m in conversation if m["role"] in ("user", "assistant") and not m.get("tool_calls")]
    kept = [m for m in tail if m["role"] in ("user", "assistant") and not m.get("tool_calls")]
    assert kept[-len(words[-2 * PROTECTED_EXCHANGES:]):] == words[-2 * PROTECTED_EXCHANGES:]
    assert all(m in conversation for m in kept)                # ни слова не пересказано
    assert _pairs_intact(tail) and tail[0]["role"] == "user"
    roles = [m["role"] for m in tail if m["role"] != "tool"]
    assert all(not (a == b == "user") for a, b in zip(roles, roles[1:]))


def test_a_long_chain_now_keeps_the_conversation_before_it():
    budget = 60_000
    conversation = [m for i in range(10) for m in _talk(i, 800)] + _long_turn(pairs=120, size=3_000)

    tail, _, _ = select_tail(conversation, budget)

    assert _exchanges(tail)[-1] == "разбери, почему падает сборка"
    assert len(_exchanges(tail)) > 1                            # переписка до цепочки осталась
    assert tail[-1] == conversation[-1]
    assert _pairs_intact(tail)
    assert breakdown(tail)[0] <= budget * CONVERSATION_SHARE + 2_000


def test_folded_turns_keep_the_request_append_only_until_the_next_rebuild():
    budget = 60_000
    conversation = [m for i in range(6) for m in _tool_turn(i)]
    tail, state, _ = select_tail(conversation, budget)
    for step in range(3):
        grown = conversation + _talk(100 + step, 200)
        new_tail, state, rebased = select_tail(grown, budget, state)
        if not rebased:
            assert new_tail[:len(tail)] == tail
        conversation, tail = grown, new_tail


def test_old_call_arguments_are_released_but_stay_a_valid_call():
    big = json.dumps({"path": "a.py", "content": "x" * 5_000})
    call = {"role": "assistant", "content": "",
            "tool_calls": [{"id": "w1", "type": "function", "function": {"name": "write_file", "arguments": big}}]}
    conversation = [_user("запиши"), call, _tool_result("ok", "w1"), _assistant("записал"), _user("дальше")]

    aged = age_before(conversation, 2)

    released = aged[1]["tool_calls"][0]
    assert released["id"] == "w1" and released["function"]["name"] == "write_file"
    assert "5" in json.loads(released["function"]["arguments"])["aida_released"]
    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == big


def test_a_released_result_points_to_its_full_copy_on_disk():
    conversation = [m for i in range(3) for m in _work_turn(i)]

    aged = age_before(conversation, 4, {"c0": "C:/h/cache/aida-tool-output/read_file_c0.txt"})

    assert "C:/h/cache/aida-tool-output/read_file_c0.txt" in aged[2]["content"]


def test_the_breakdown_splits_words_from_tool_output():
    tail = [_user("вопрос"), _calls_tool(), _tool_result("р" * 1_000), _assistant("ответ")]

    talk, tools, stubs = breakdown(tail)

    assert talk + tools == tail_size(tail)
    assert tools >= 1_000 and stubs == 0
