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

import re
from dataclasses import dataclass
from typing import Any, Iterable

# Доли, а не размеры: сам бюджет приходит снаружи и считается от окна модели.
# Сколько хвоста остаётся после пересборки. Она ломает кеш, поэтому должна быть редкой и
# крупной: срезать по чуть-чуть значило бы пересобирать на каждом ходу.
STEP_KEEP = 0.6
# Какую долю бюджета могут занимать результаты инструментов целиком; всё, что старше,
# заменяется заглушкой. Выхлоп инструментов — три четверти объёма разговора (замер оператора).
KEEP_RESULTS_SHARE = 0.35
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


def _stub_for(message: dict, size: int) -> str:
    name = message.get("name") or message.get("tool_name")
    label = f"«{name}» " if name else ""
    return (f"[{label}отпущен из запроса: {size} знаков; оригинал в сессии — повтори вызов или "
            "найди через session_search]")


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
    return _recalled_size(message) > AGE_MIN_CHARS


def _aged(message: dict) -> dict:
    if message.get("role") == "tool":
        return {**message, "content": _stub_for(message, len(message["content"]))}
    return {**message, "content": _RECALLED.sub("", message["content"]).strip() or message["content"]}


def age_before(messages: list[dict], boundary: int) -> list[dict]:
    """Состарить то, что стоит до границы: длинные результаты инструментов заменить заглушкой,
    а с давних вопросов снять найденное под них в памяти. Вход не меняется.

    У результата меняется только текст: роль и `tool_call_id` остаются, поэтому пара цела.
    """
    if boundary <= 0:
        return messages
    out = list(messages)
    for index in range(min(boundary, len(messages))):
        if _ageable(messages[index]):
            out[index] = _aged(messages[index])
    return out


def count_aged(messages: list[dict], boundary: int) -> int:
    return sum(1 for index in range(min(max(boundary, 0), len(messages))) if _ageable(messages[index]))


def _aged_size(message: dict, aged: bool) -> int:
    return message_size(_aged(message)) if aged and _ageable(message) else message_size(message)


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

    def fits(self, messages: list[dict]) -> bool:
        """Запомненные места на месте — разговор тот же и только вырос."""
        def same(index: int, key: int) -> bool:
            return 0 <= index < len(messages) and message_key(messages[index]) == key

        if not same(self.head, self.head_key):
            return False
        if self.resume and not same(self.resume, self.resume_key):
            return False
        # Граница заглушек стоит ПОСЛЕ последнего состаренного сообщения; сверяется оно.
        return not self.aged or same(self.aged - 1, self.aged_key)


def apply_state(messages: list[dict], state: TailState) -> list[dict]:
    """Хвост по запомненным решениям. Одни и те же решения дают одни и те же байты."""
    aged = age_before(messages, state.aged)
    if state.resume > state.head:
        return [aged[state.head]] + aged[state.resume:]
    return aged[state.head:]


def tail_size(tail: Iterable[dict]) -> int:
    return sum(message_size(message) for message in tail)


def _aging_boundary(messages: list[dict], budget: int) -> int:
    """До какого места результаты инструментов и найденное в памяти уходят из запроса.

    Идём с конца и оставляем целым свежее, пока оно занимает не больше своей доли бюджета.
    Самое свежее остаётся целым всегда — модель его ещё не читала.
    """
    limit = budget * KEEP_RESULTS_SHARE
    spent = 0
    seen = False
    for index in range(len(messages) - 1, -1, -1):
        if not _ageable(messages[index]):
            continue
        spent += message_size(messages[index]) if messages[index].get("role") == "tool" \
            else _recalled_size(messages[index])
        if spent > limit and seen:
            return index + 1
        seen = True
    return 0


def _key_at(messages: list[dict], index: int) -> int:
    return message_key(messages[index]) if 0 <= index < len(messages) else 0


def rebase(messages: list[dict], budget: int, previous: TailState | None = None, *, age: bool = True,
           overflow: bool = False) -> TailState:
    """Пересмотреть всё разом: границу заглушек, начало хвоста и разрез внутри хода.

    Прежнее начало хвоста сохраняется, если хвост от него помещается: срезать разговор, который
    влезает, незачем. Не помещается — режем сразу до STEP_KEEP бюджета, по реплике человека. Если
    не помещается даже последний ход (длинная цепочка инструментов после одной реплики), от хода
    остаются его вопрос и свежая часть цепочки, а середина выбрасывается целыми связками.
    """
    if not messages:
        return TailState()
    aged = _aging_boundary(messages, budget) if age else 0
    sizes = [_aged_size(message, index < aged) for index, message in enumerate(messages)]
    suffix = [0] * (len(messages) + 1)
    for index in range(len(messages) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + sizes[index]

    def state(head: int, resume: int = 0) -> TailState:
        return TailState(head=head, head_key=_key_at(messages, head),
                         resume=resume, resume_key=_key_at(messages, resume) if resume else 0,
                         aged=aged, aged_key=_key_at(messages, aged - 1) if aged else 0)

    users = [index for index, message in enumerate(messages) if is_real_user(message)]
    if not users:
        return state(0)  # разговор без реплик человека резать не по чему — отдаём как есть

    target = budget * STEP_KEEP
    # После паузы достаточно, чтобы хвост помещался. А если пересборку вызвало переполнение,
    # одних новых заглушек должно хватить с запасом — иначе через ход она случится снова, а
    # каждая пересборка стоит кеша.
    enough = target if overflow else budget
    if previous is not None and previous.fits(messages) and not previous.resume \
            and previous.head in users and suffix[previous.head] <= enough:
        return state(previous.head)
    if suffix[users[0]] <= budget and (previous is None or not previous.fits(messages)):
        return state(users[0])

    for head in users:
        if suffix[head] <= target:
            return state(head)

    # Не помещается даже последний ход. Оставляем его вопрос и столько свежих связок, сколько
    # влезает; связка начинается с ответа помощника, и резать можно только по её началу.
    head = users[-1]
    room = target - sizes[head]
    starts = [index for index in range(head + 1, len(messages)) if messages[index].get("role") == "assistant"]
    if not starts:
        return state(head)
    for resume in starts:
        if suffix[resume] <= room:
            return state(head, resume if resume > head + 1 else 0)
    return state(head, starts[-1] if starts[-1] > head + 1 else 0)


def select_tail(messages: list[dict], budget: int, state: TailState | None = None, *,
                force: bool = False) -> tuple[list[dict], TailState, bool]:
    """Хвост разговора для этого запроса. Возвращает хвост, решения и признак пересборки.

    Пока запомненные решения годятся и хвост помещается, он тот же, что в прошлый раз, плюс то,
    что дописалось. `force` — оператор молчал дольше жизни кеша: самое время пересобрать.
    """
    overflow = False
    if state is not None and state.fits(messages):
        tail = apply_state(messages, state)
        overflow = tail_size(tail) > budget
        if not overflow and not force:
            return tail, state, False
    fresh = rebase(messages, budget, state, overflow=overflow)
    return apply_state(messages, fresh), fresh, True


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
