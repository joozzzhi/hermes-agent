"""Сборка витрины: что из разговора уходит в модель и в каком виде.

Здесь нет ни базы, ни сети — только превращение разговора в запрос. Два правила, ради которых
этот файл существует:

**Пара не рвётся.** Обращение к инструменту и его результат — пара, и запрос, разрезанный
между ними, поставщик отвергает целиком. Поэтому хвост начинается только с реплики человека,
а внутри длинного хода выбрасываются только целые связки «обращение — результаты».

**Середина запроса не меняется.** Кеш поставщика жив, пока начало запроса совпадает байт в
байт. Поэтому всё, что решено про хвост, запоминается (`TailState`) и применяется одинаково
на каждом запросе: начало хвоста, место разреза внутри хода, граница, до которой результаты
инструментов заменены заглушками. Запрос при этом только дописывается в конец. Решения
пересматриваются разом (`rebase`) и только в двух случаях: хвост перестал помещаться, или
оператор молчал дольше, чем живёт кеш, и ломать уже нечего.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .settings import DEFAULTS

# Доли, а не размеры: сам бюджет приходит снаружи и считается от окна модели. Здесь — значения
# по умолчанию; профиль переопределяет их в секции `aida:` своего конфига (settings.py).
#
# У переписки и у инструментов отдельные бюджеты. Замер по двум разговорам с оператором
# (state.db, 107 обменов): обмен «реплика — ответ» без инструментов в среднем 1,6 тыс. знаков,
# десять подряд — не больше 24,7 тыс.; выхлоп инструментов — 85% объёма хода. При бюджете
# хвоста около 145 тыс. знаков (60 тыс. токенов минус системная часть и ядро) переписке
# отведена четверть: последние десять обменов влезают с запасом в полтора раза.
CONVERSATION_SHARE = DEFAULTS.conversation_share
# Какую долю бюджета после пересборки могут занимать выводы инструментов (результаты и
# аргументы обращений); всё, что старше, заменяется заглушкой.
KEEP_RESULTS_SHARE = DEFAULTS.keep_results_share
# Сколько хвоста остаётся после пересборки — оба бюджета вместе. Пересборка ломает кеш,
# поэтому должна быть редкой и крупной: до следующей хвост может вырасти на 40% бюджета.
STEP_KEEP = CONVERSATION_SHARE + KEEP_RESULTS_SHARE
# Сколько последних обменов с человеком остаются в окне дословно, сколько бы инструментов ни
# гонялось. Текущий, ещё не отвеченный вопрос — один из них.
PROTECTED_EXCHANGES = DEFAULTS.protected_exchanges
# Короче этого результат не трогается: заглушка сама занимает около ста знаков.
AGE_MIN_CHARS = 500

# Заголовки групп — из движка, работавшего на сервере. Порядок фиксирован: блок знания стоит
# в начале запроса и не должен перетасовываться от того, какая запись оказалась новее.
GROUP_LABELS = {
    "rule": "ПРАВИЛА РАБОТЫ",
    "preference": "ПРЕДПОЧТЕНИЯ ОПЕРАТОРА",
    "decision": "КЛЮЧЕВЫЕ РЕШЕНИЯ",
    "fact": "ФАКТЫ ОБ ОПЕРАТОРЕ",
    "project": "ПРОЕКТЫ",
    "contact": "КОНТАКТЫ",
    "context": "КОНТЕКСТ",
    "lesson": "УРОКИ",
    "event": "СОБЫТИЯ",
}
OTHER_LABEL = "ПРОЧЕЕ"


def is_system(message: dict) -> bool:
    return message.get("role") == "system"


def split_system(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Отделить системную часть хозяина от разговора.

    Она идёт первой и не трогается: там личность Гермеса и его инструкции, и она же —
    устойчивое начало, на котором держится кеш поставщика.
    """
    head: list[dict] = []
    for index, message in enumerate(messages):
        if is_system(message):
            head.append(message)
            continue
        return head, list(messages[index:])
    return head, []


def _is_tool_result(message: dict) -> bool:
    return message.get("role") == "tool" or message.get("tool_call_id") is not None


def is_real_user(message: dict) -> bool:
    return message.get("role") == "user" and not _is_tool_result(message)


def tool_part_size(message: dict) -> int:
    """Сколько в сообщении приходится на инструменты: результат целиком, у ответа помощника —
    аргументы обращений, у реплики человека — найденное под неё в памяти. Остальное — переписка."""
    if _is_tool_result(message):
        return message_size(message)
    if message.get("role") == "assistant":
        calls = message.get("tool_calls") or ()
        if calls and not _has_text(message):
            return message_size(message)
        return sum(len(str(call)) for call in calls)
    return _recalled_size(message)


def message_size(message: dict) -> int:
    """Грубый размер сообщения в символах — вместе с его инструментальной частью."""
    content = message.get("content")
    if isinstance(content, str):
        size = len(content)
    elif isinstance(content, list):
        size = sum(len(str(part)) for part in content)
    else:
        size = len(str(content or ""))
    for call in message.get("tool_calls") or ():
        size += len(str(call))
    return size + 16


def message_key(message: dict) -> int:
    """Отпечаток сообщения по содержимому — им сверяется, что запомненное место всё ещё то же."""
    return hash((message.get("role"), str(message.get("content")),
                 message.get("tool_call_id"), str(message.get("tool_calls"))))


def _stub_for(message: dict, size: int, paths: dict | None = None) -> str:
    name = message.get("name") or message.get("tool_name")
    label = f"«{name}» " if name else ""
    path = (paths or {}).get(message.get("tool_call_id"))
    if path:
        return f"[{label}отпущен из запроса: {size} знаков; полный вывод: {path} — читай read_file кусками]"
    return (f"[{label}отпущен из запроса: {size} знаков; оригинал в сессии — повтори вызов или "
            "найди через session_search]")


def _call_arguments(call: Any) -> str:
    function = call.get("function") if isinstance(call, dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    return arguments if isinstance(arguments, str) else ""


def _aged_call(call: dict) -> dict:
    """Аргументы старого обращения (запись файла, правка) отпускаются, как и результаты: имя и
    id остаются, так что пара цела, а аргументы — всё ещё объект, как требует поставщик."""
    arguments = _call_arguments(call)
    if len(arguments) <= AGE_MIN_CHARS:
        return call
    stub = json.dumps({"aida_released": f"аргументы отпущены из запроса: {len(arguments)} знаков; "
                                         "оригинал в сессии"}, ensure_ascii=False)
    return {**call, "function": {**call["function"], "arguments": stub}}


# Найденное в памяти хозяин приклеивает к реплике, под которую оно найдено, в такой обёртке.
# Пока реплика свежая, найденное нужно; через двадцать ходов это балласт, который нашёлся бы
# заново, если бы понадобился. Стареет он тем же способом и в тот же момент, что результаты.
_RECALLED = re.compile(r"\s*<memory-context>.*?</memory-context>", re.DOTALL)


def _recalled_size(message: dict) -> int:
    content = message.get("content")
    if not is_real_user(message) or not isinstance(content, str) or "<memory-context>" not in content:
        return 0
    return sum(len(match.group(0)) for match in _RECALLED.finditer(content))


def _ageable(message: dict) -> bool:
    content = message.get("content")
    if message.get("role") == "tool":
        return isinstance(content, str) and len(content) > AGE_MIN_CHARS
    if message.get("role") == "assistant":
        return any(len(_call_arguments(call)) > AGE_MIN_CHARS for call in message.get("tool_calls") or ())
    return _recalled_size(message) > AGE_MIN_CHARS


def _aged(message: dict, paths: dict | None = None) -> dict:
    if message.get("role") == "tool":
        return {**message, "content": _stub_for(message, len(message["content"]), paths)}
    if message.get("role") == "assistant":
        return {**message, "tool_calls": [_aged_call(call) for call in message["tool_calls"]]}
    return {**message, "content": _RECALLED.sub("", message["content"]).strip() or message["content"]}


def age_before(messages: list[dict], boundary: int, paths: dict | None = None) -> list[dict]:
    """Состарить то, что стоит до границы: длинные результаты и аргументы обращений заменить
    заглушкой, а с давних вопросов снять найденное под них в памяти. Вход не меняется.

    У результата меняется только текст: роль и `tool_call_id` остаются, поэтому пара цела.
    """
    if boundary <= 0:
        return messages
    out = list(messages)
    for index in range(min(boundary, len(messages))):
        if _ageable(messages[index]):
            out[index] = _aged(messages[index], paths)
    return out


def count_aged(messages: list[dict], boundary: int) -> int:
    return sum(1 for index in range(min(max(boundary, 0), len(messages))) if _ageable(messages[index]))


def _aged_size(message: dict, aged: bool) -> int:
    return message_size(_aged(message)) if aged and _ageable(message) else message_size(message)


def _has_text(message: dict) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(isinstance(part, dict) and str(part.get("text") or "").strip() for part in content)
    return bool(content)


def folded_indices(messages: list[dict], boundary: int) -> set[int]:
    """Что убирается из законченных ходов до границы: молчаливые связки «обращение —
    результаты» (у обращения нет ни слова текста). Вопрос человека и все слова помощника
    остаются, так что переписка цела; пара уходит только целиком.

    Ход сворачивается, только если в нём останется ответ помощника — иначе две реплики
    человека встали бы подряд.
    """
    if boundary <= 0:
        return set()
    users = [index for index, message in enumerate(messages) if is_real_user(message)]
    dropped: set[int] = set()
    for start, end in zip(users, users[1:] + [len(messages)]):
        if end > boundary:
            break
        turn: set[int] = set()
        for index in range(start + 1, end):
            message = messages[index]
            calls = message.get("tool_calls") or ()
            if message.get("role") == "assistant" and calls and not _has_text(message):
                ids = {call.get("id") for call in calls if isinstance(call, dict)}
                turn.add(index)
                turn.update(j for j in range(index + 1, end)
                            if _is_tool_result(messages[j]) and messages[j].get("tool_call_id") in ids)
        if any(messages[i].get("role") == "assistant" for i in range(start + 1, end) if i not in turn):
            dropped |= turn
    return dropped


@dataclass
class TailState:
    """Что решено про хвост этого разговора. Место хранится номером и отпечатком сообщения:
    история у хозяина только дописывается, так что номер верен, пока отпечаток совпадает."""

    head: int = 0            # с какого сообщения начинается хвост (реплика человека)
    head_key: int = 0
    resume: int = 0          # разрез внутри хода: после head сразу идёт это сообщение; 0 — разреза нет
    resume_key: int = 0
    aged: int = 0            # результаты инструментов до этого номера заменены заглушками
    aged_key: int = 0
    folded: int = 0          # в законченных ходах до этого номера убраны молчаливые связки
    folded_key: int = 0

    def fits(self, messages: list[dict]) -> bool:
        """Запомненные места на месте — разговор тот же и только вырос."""
        def same(index: int, key: int) -> bool:
            return 0 <= index < len(messages) and message_key(messages[index]) == key

        if not same(self.head, self.head_key):
            return False
        if self.resume and not same(self.resume, self.resume_key):
            return False
        # Граница заглушек стоит ПОСЛЕ последнего состаренного сообщения; сверяется оно.
        if self.aged and not same(self.aged - 1, self.aged_key):
            return False
        return not self.folded or same(self.folded - 1, self.folded_key)


def apply_state(messages: list[dict], state: TailState, paths: dict | None = None) -> list[dict]:
    """Хвост по запомненным решениям. Одни и те же решения дают одни и те же байты."""
    aged = age_before(messages, state.aged, paths)
    picked: Iterable[int] = range(state.head, len(messages))
    if state.resume > state.head:
        # Разрез внутри последнего хода: его вопрос остаётся, середина до resume выбрасывается.
        question = max(i for i in range(state.head, state.resume) if i == state.head or is_real_user(messages[i]))
        picked = list(range(state.head, question + 1)) + list(range(state.resume, len(messages)))
    dropped = folded_indices(messages, state.folded)
    return [aged[index] for index in picked if index not in dropped]


def tail_size(tail: Iterable[dict]) -> int:
    return sum(message_size(message) for message in tail)


def _aging_boundary(messages: list[dict], budget: int, keep_share: float = KEEP_RESULTS_SHARE) -> int:
    """До какого места результаты инструментов и найденное в памяти уходят из запроса.

    Идём с конца и оставляем целым свежее, пока оно занимает не больше своей доли бюджета.
    Самое свежее остаётся целым всегда — модель его ещё не читала.
    """
    limit = budget * keep_share
    spent = 0
    seen = False
    for index in range(len(messages) - 1, -1, -1):
        if not _ageable(messages[index]):
            continue
        spent += tool_part_size(messages[index])
        if spent > limit and seen:
            return index + 1
        seen = True
    return 0


def _key_at(messages: list[dict], index: int) -> int:
    return message_key(messages[index]) if 0 <= index < len(messages) else 0


def protected_start(messages: list[dict], exchanges: int = PROTECTED_EXCHANGES) -> int:
    """С какого сообщения начинаются последние `exchanges` обменов с человеком."""
    users = [index for index, message in enumerate(messages) if is_real_user(message)]
    if not users:
        return 0
    return users[-min(max(exchanges, 1), len(users))]


def rebase(messages: list[dict], budget: int, previous: TailState | None = None, *, age: bool = True,
           overflow: bool = False, protect: int = PROTECTED_EXCHANGES,
           talk_share: float = CONVERSATION_SHARE, keep_share: float = KEEP_RESULTS_SHARE) -> TailState:
    """Пересмотреть всё разом: границу заглушек, начало хвоста и разрез внутри хода.

    Последние `protect` обменов с человеком остаются всегда: хвост не начинается позже них.
    Прежнее начало хвоста сохраняется, если хвост от него помещается: срезать разговор, который
    влезает, незачем. Не помещается — режем сразу до STEP_KEEP бюджета, по реплике человека.
    Если инструменты не дают уместить даже защищённые обмены, из законченных ходов убираются
    молчаливые связки «обращение — результаты» — слова обеих сторон остаются. Не помещается и
    тогда (последний ход — длинная цепочка инструментов), от последнего хода остаются вопрос и
    свежая часть цепочки, а середина выбрасывается целыми связками.
    """
    if not messages:
        return TailState()
    aged = _aging_boundary(messages, budget, keep_share) if age else 0
    users = [index for index, message in enumerate(messages) if is_real_user(message)]

    def plan(folded: int) -> tuple[list[int], list[int]]:
        dropped = folded_indices(messages, folded)
        sizes = [0 if index in dropped else _aged_size(message, index < aged)
                 for index, message in enumerate(messages)]
        suffix = [0] * (len(messages) + 1)
        for index in range(len(messages) - 1, -1, -1):
            suffix[index] = suffix[index + 1] + sizes[index]
        return sizes, suffix

    def state(head: int, resume: int = 0, folded: int = 0) -> TailState:
        return TailState(head=head, head_key=_key_at(messages, head),
                         resume=resume, resume_key=_key_at(messages, resume) if resume else 0,
                         aged=aged, aged_key=_key_at(messages, aged - 1) if aged else 0,
                         folded=folded, folded_key=_key_at(messages, folded - 1) if folded else 0)

    if not users:
        return state(0)  # разговор без реплик человека резать не по чему — отдаём как есть

    sizes, suffix = plan(0)
    target = budget * (talk_share + keep_share)
    # После паузы достаточно, чтобы хвост помещался. А если пересборку вызвало переполнение,
    # одних новых заглушек должно хватить с запасом — иначе через ход она случится снова, а
    # каждая пересборка стоит кеша.
    enough = target if overflow else budget
    guard = protected_start(messages, protect)
    heads = [index for index in users if index <= guard]
    if previous is not None and previous.fits(messages) and not previous.resume and not previous.folded \
            and previous.head in heads and suffix[previous.head] <= enough:
        return state(previous.head)
    if suffix[users[0]] <= budget and (previous is None or not previous.fits(messages)):
        return state(users[0])
    for head in heads:
        if suffix[head] <= target:
            return state(head)

    if not age:
        # compress() режет историю хозяина, а не запрос: там связки не сворачиваются.
        return _long_turn_cut(messages, users, sizes, suffix, users[-1], target, state)

    # Защищённые обмены не помещаются из-за инструментов: сворачиваем законченные ходы.
    folded = users[-1]
    sizes, suffix = plan(folded)
    for head in users:
        if suffix[head] <= target:
            return state(head, folded=folded)

    # Не помещается даже так: последний ход — длинная цепочка. Оставляем переписку в пределах
    # её бюджета (сколько из защищённых обменов влезает), вопрос последнего хода и столько
    # свежих связок, сколько влезает в остаток.
    last = users[-1]
    talk = budget * talk_share
    head = next((index for index in users if index >= guard and suffix[index] - suffix[last + 1] <= talk), last)
    return _long_turn_cut(messages, users, sizes, suffix, head, target,
                          lambda h, r=0: state(h, r, folded))


def _long_turn_cut(messages, users, sizes, suffix, head, target, state) -> TailState:
    """Хвост от `head`, у последнего хода — вопрос и свежая часть цепочки; связка начинается с
    ответа помощника, и резать можно только по её началу."""
    last = users[-1]
    room = target - (suffix[head] - suffix[last + 1])
    starts = [index for index in range(last + 1, len(messages)) if messages[index].get("role") == "assistant"]
    if not starts:
        return state(head)
    for resume in starts:
        if suffix[resume] <= room:
            return state(head, resume if resume > last + 1 else 0)
    return state(head, starts[-1] if starts[-1] > last + 1 else 0)


def select_tail(messages: list[dict], budget: int, state: TailState | None = None, *,
                force: bool = False, paths: dict | None = None,
                protect: int = PROTECTED_EXCHANGES, talk_share: float = CONVERSATION_SHARE,
                keep_share: float = KEEP_RESULTS_SHARE) -> tuple[list[dict], TailState, bool]:
    """Хвост разговора для этого запроса. Возвращает хвост, решения и признак пересборки.

    Пока запомненные решения годятся и хвост помещается, он тот же, что в прошлый раз, плюс то,
    что дописалось. `force` — оператор молчал дольше жизни кеша: самое время пересобрать.
    """
    overflow = False
    if state is not None and state.fits(messages):
        tail = apply_state(messages, state, paths)
        overflow = tail_size(tail) > budget
        if not overflow and not force:
            return tail, state, False
    fresh = rebase(messages, budget, state, overflow=overflow, protect=protect,
                   talk_share=talk_share, keep_share=keep_share)
    return apply_state(messages, fresh, paths), fresh, True


def breakdown(tail: Iterable[dict]) -> tuple[int, int, int]:
    """Сколько знаков хвоста ушло на переписку, сколько на инструменты и сколько заглушек."""
    conversation = tools = stubs = 0
    for message in tail:
        size, tool = message_size(message), tool_part_size(message)
        conversation += size - tool
        tools += tool
        content = message.get("content")
        if _is_tool_result(message) and isinstance(content, str) and "отпущен из запроса" in content:
            stubs += 1
    return conversation, tools, stubs


def build_knowledge_block(rows: Iterable[dict], budget: int) -> str:
    """Ядро: то, что верно всегда. Записи приходят в порядке важности и берутся, пока влезают.

    Запись не обрезается: половина правила — это другое правило. Не влезла — её найдёт поиск.
    """
    grouped: dict[str, list[str]] = {}
    spent = 0
    for row in rows:
        content = " ".join(str(row.get("content", "")).split())
        if not content:
            continue
        cost = len(content) + 3
        if spent + cost > budget:
            continue
        spent += cost
        label = GROUP_LABELS.get(row.get("memory_type") or "", OTHER_LABEL)
        grouped.setdefault(label, []).append(content)
    if not grouped:
        return ""
    lines = ["# Что я знаю всегда"]
    for label in list(GROUP_LABELS.values()) + [OTHER_LABEL]:
        items = grouped.get(label)
        if not items:
            continue
        lines.append("")
        lines.append(f"## {label}")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def merge_into_system(system_head: list[dict], block: str) -> list[dict]:
    """Влить блок в системную часть хозяина, а НЕ добавить вторым системным сообщением.

    Сборщик запроса к Anthropic кладёт в поле системной части ПОСЛЕДНЕЕ системное сообщение, а
    не все: второе системное сообщение вытесняет с провода личность хозяина (SOUL.md) — молча,
    без ошибки и без предупреждения, так что заметно это только по тому, что агент отвечает
    суше, чем он есть. Поэтому знание едет внутри той же системной части, следом за её текстом:
    устойчивое начало сохраняется целиком, а кеш поставщика по-прежнему режется хозяином на
    границе его собственного неизменного текста.
    """
    if not block:
        return list(system_head)
    if not system_head:
        return [{"role": "system", "content": block}]
    merged = list(system_head)
    last = dict(merged[-1])
    content = last.get("content")
    if isinstance(content, list):
        last["content"] = list(content) + [{"type": "text", "text": block}]
    else:
        text = content if isinstance(content, str) else str(content or "")
        last["content"] = f"{text}\n\n{block}" if text else block
    merged[-1] = last
    return merged


def is_conversation_message(message: dict) -> bool:
    """Переживёт ли сообщение дорогу до модели — то есть будет ли в запросе разговор.

    Системные сообщения уезжают в отдельное поле, а результат инструмента без своего обращения
    хозяин выбрасывает как сироту. Если ничего, кроме них, в собранном запросе нет, до модели
    доедет запрос без единого сообщения.
    """
    return not is_system(message) and not _is_tool_result(message)
