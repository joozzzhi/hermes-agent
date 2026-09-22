"""Один поиск на ход — и один, кто о нём рассказывает.

Память и контекстный движок читают одну базу и ищут одно и то же. Если искать дважды, в
запрос уедут две копии найденного, а оператор заплатит за это и временем, и чужой квотой.

Поэтому: когда движок-витрина включён, ищет он, а показывает найденное по-прежнему память —
над ответом стоит отметка с числом поднятых записей, как и договаривались. Этот модуль —
единственная ниточка между ними: движок кладёт сюда число, память его забирает.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_engine_active = False
_last_count: int | None = None


def announce_engine() -> None:
    """Движок собирает контекст сам — памяти больше не нужно искать под каждый ход."""
    global _engine_active
    with _lock:
        _engine_active = True


def engine_is_assembling() -> bool:
    with _lock:
        return _engine_active


def publish(count: int) -> None:
    """Сколько записей движок поднял под этот вопрос."""
    global _last_count
    with _lock:
        _last_count = int(count)


def last_count() -> int | None:
    """Что показать оператору; None — движок ещё ничего не поднимал."""
    with _lock:
        return _last_count


def reset() -> None:
    """Для тестов: вернуть состояние к исходному."""
    global _engine_active, _last_count
    with _lock:
        _engine_active = False
        _last_count = None
