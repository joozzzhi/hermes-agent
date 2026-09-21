"""Сборка витрины: слоты, бюджеты и безопасная граница хвоста.

Здесь нет ни базы, ни сети — только превращение «что нашлось» в «что уходит в модель».
Вынесено отдельно, потому что самое опасное место движка проверяется именно тут: обращение
к инструменту и его результат — пара, и хвост, разрезанный между ними, отвергается
поставщиком целиком. Разговор при этом не деградирует, а встаёт.
"""

from __future__ import annotations

from typing import Any, Iterable

# Бюджеты в символах — как в том движке, что работал на сервере. Символы, а не токены,
# потому что считать их нечем и незачем: это витрина, а не бухгалтерия.
TOTAL_BUDGET = 150_000
KNOWLEDGE_BUDGET = 12_000
RECALLED_BUDGET = 15_000
OLD_HISTORY_BUDGET = 6_000
MIN_TAIL_BUDGET = 20_000

# Заголовки групп — из того же движка. Модель читает их как оглавление, человек — как
# карту того, что о нём знают.
GROUP_LABELS = {
    "preference": "ПРЕДПОЧТЕНИЯ ОПЕРАТОРА",
    "fact": "ФАКТЫ ОБ ОПЕРАТОРЕ",
    "decision": "КЛЮЧЕВЫЕ РЕШЕНИЯ",
    "rule": "ПРАВИЛА РАБОТЫ",
    "context": "КОНТЕКСТ",
    "project": "ПРОЕКТЫ",
    "contact": "КОНТАКТЫ",
    "event": "СОБЫТИЯ",
    "lesson": "УРОКИ",
}


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


def _has_tool_call(message: dict) -> bool:
    return bool(message.get("tool_calls") or message.get("function_call"))


def _is_tool_result(message: dict) -> bool:
    return message.get("role") == "tool" or message.get("tool_call_id") is not None


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


def is_clean_boundary(messages: list[dict], index: int) -> bool:
    """Можно ли начать хвост с этого сообщения, не оборвав пару «обращение — результат».

    Чистая граница — это реплика человека. Ответ помощника с обращением к инструменту
    тянет за собой результат, а результат без своего обращения поставщик не принимает.
    """
    if index >= len(messages):
        return False
    message = messages[index]
    if _is_tool_result(message):
        return False
    if message.get("role") == "user":
        return True
    # Ответ помощника без инструментов — тоже допустимое начало, но только если следом
    # не идёт осиротевший результат.
    if message.get("role") == "assistant" and not _has_tool_call(message):
        return not (index + 1 < len(messages) and _is_tool_result(messages[index + 1]))
    return False


def safe_tail(messages: list[dict], budget: int) -> list[dict]:
    """Свежий хвост разговора, целиком влезающий в бюджет и не рвущий ни одной пары.

    Отсчитывается с конца; найденная по бюджету граница сдвигается вперёд до ближайшей
    чистой. Если чистой границы нет вовсе (весь хвост — одна длинная инструментальная
    цепочка), берётся последнее сообщение: пустой хвост оставил бы модель без вопроса.
    """
    if not messages:
        return []
    spent = 0
    start = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        spent += message_size(messages[index])
        if spent > budget and index != len(messages) - 1:
            break
        start = index
    while start < len(messages) and not is_clean_boundary(messages, start):
        start += 1
    if start >= len(messages):
        return messages[-1:]
    return messages[start:]


def _clip(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[:budget].rstrip() + " […]"


def build_knowledge_block(rows: Iterable[dict], budget: int = KNOWLEDGE_BUDGET) -> str:
    """То, что верно всегда: знание с приоритетом P1, сгруппированное по типам."""
    grouped: dict[str, list[str]] = {}
    for row in rows:
        label = GROUP_LABELS.get(row.get("memory_type") or "", "ПРОЧЕЕ")
        grouped.setdefault(label, []).append(" ".join(str(row.get("content", "")).split()))
    if not grouped:
        return ""
    lines = ["# Что я знаю всегда"]
    for label, items in grouped.items():
        lines.append("")
        lines.append(f"## {label}")
        lines.extend(f"- {item}" for item in items)
    return _clip("\n".join(lines), budget)


def build_recalled_block(rows: Iterable[dict], budget: int = RECALLED_BUDGET) -> str:
    """Найденное по смыслу этого вопроса."""
    lines = []
    spent = 0
    for row in rows:
        content = " ".join(str(row.get("content", "")).split())
        created = row.get("created_at")
        when = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else ""
        tag = " · ".join(part for part in (row.get("memory_type"), when) if part)
        line = f"- [{tag}] {content}" if tag else f"- {content}"
        if spent + len(line) > budget:
            break
        lines.append(line)
        spent += len(line)
    if not lines:
        return ""
    return "# Вспомнил под этот вопрос\n" + "\n".join(lines)


def build_old_history_block(rows: Iterable[dict], budget: int = OLD_HISTORY_BUDGET) -> str:
    """Куски давнего разговора, которые всплыли по смыслу вопроса.

    Дословно, с датой и с указанием, кто говорил: это не пересказ, а цитата из того, что
    было сказано на самом деле.
    """
    lines = []
    spent = 0
    for row in rows:
        content = " ".join(str(row.get("content", "")).split())
        created = row.get("created_at")
        when = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else ""
        who = "Оператор" if row.get("role") == "user" else "Я"
        line = f"[{when}] {who}: {_clip(content, 400)}" if when else f"{who}: {_clip(content, 400)}"
        if spent + len(line) > budget:
            break
        lines.append(line)
        spent += len(line)
    if not lines:
        return ""
    return "# Из давнего разговора, по смыслу вопроса\n" + "\n".join(lines)


def _prepend_to_question(message: dict, block: str) -> dict:
    """Подмешать динамический слой к самому вопросу, не создавая нового сообщения.

    Системное сообщение в середине разговора принимают не все модели — на той, где
    работает оператор, оно возвращает ошибку и обрывает ответ. Поэтому изменчивая часть
    едет вместе с вопросом, ровно как в том движке, что работал на сервере.
    """
    content = message.get("content")
    if isinstance(content, str):
        merged: Any = f"{block}\n\n---\n\n{content}"
    elif isinstance(content, list):
        merged = [{"type": "text", "text": block}] + list(content)
    else:
        merged = block
    return {**message, "content": merged}


def assemble(
    system_head: list[dict],
    knowledge: str,
    recalled: str,
    old_history: str,
    conversation: list[dict],
    *,
    total_budget: int = TOTAL_BUDGET,
) -> list[dict]:
    """Собрать запрос заново: устойчивое впереди, изменчивое при вопросе, хвост дословно.

    Порядок не случайный. Системная часть и знание меняются редко и стоят первыми — на них
    держится кеш поставщика. Изменчивое подмешивается к последнему вопросу, чтобы не
    вставлять системные сообщения в середину разговора и не сбивать кеш каждым ходом.

    Если последнее сообщение — не вопрос человека (идёт обмен с инструментами внутри
    хода), изменчивый слой не добавляется: он уже приехал вместе с самим вопросом.
    """
    spent = len(knowledge) + len(recalled) + len(old_history)
    spent += sum(message_size(m) for m in system_head)
    tail_budget = max(MIN_TAIL_BUDGET, total_budget - spent)

    assembled = list(system_head)
    if knowledge:
        assembled.append({"role": "system", "content": knowledge})

    tail = safe_tail(conversation, tail_budget)
    dynamic = "\n\n".join(block for block in (recalled, old_history) if block)
    if dynamic and tail and tail[-1].get("role") == "user":
        tail = tail[:-1] + [_prepend_to_question(tail[-1], dynamic)]
    assembled.extend(tail)
    return assembled
