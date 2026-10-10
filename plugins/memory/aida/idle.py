"""Разбор по паузе: оператор замолчал — память приводит себя в порядок.

Расписание здесь не подходит по двум причинам. Ночной прогон разбирает разговор через много
часов после того, как он закончился, и всё это время факты из него не находятся. А главное,
разбор меняет то, что стоит в начале запроса, и делать это посреди живого разговора значит
сбрасывать кеш поставщика. Пауза длиннее жизни кеша — единственный момент, когда это бесплатно.

Устроено просто: каждый законченный ход переводит таймер; дошёл до нуля — запускается разбор.
Таймер живёт в процессе, поэтому при запуске агента очередь проверяется отдельно: если в ней
лежит разговор старше порога, он разбирается сразу (процесс перезапускали, пока таймер шёл).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_IDLE_MINUTES = 60.0
# Замок старше этого считается брошенным: процесс, который его взял, умер посреди разбора.
STALE_LOCK_SECONDS = 45 * 60
LOCK_NAME = "aida_consolidate.lock"

_guard = threading.Lock()
_timers: dict[str, threading.Timer] = {}


def idle_seconds() -> float:
    """Порог тишины. По умолчанию час — столько живёт кеш поставщика в длинном режиме."""
    try:
        minutes = float(os.environ.get("AIDA_IDLE_MINUTES", "") or DEFAULT_IDLE_MINUTES)
    except ValueError:
        minutes = DEFAULT_IDLE_MINUTES
    return max(60.0, minutes * 60.0)


def _bound(fn):
    """Таймер стартует в пустом окружении; профиль и секреты живут в окружении вызвавшего."""
    try:
        from agent.memory_provider import ctx_bound

        return ctx_bound(fn)
    except Exception:
        return fn


def touch(home: Path | str, delay: float | None = None) -> None:
    """Ход закончен — отсчёт тишины начинается заново."""
    key = str(home)
    timer = threading.Timer(idle_seconds() if delay is None else max(1.0, delay), _bound(_fire), args=(key,))
    timer.daemon = True
    timer.name = "aida-idle"
    with _guard:
        previous = _timers.pop(key, None)
        if previous is not None:
            previous.cancel()
        _timers[key] = timer
    timer.start()


def cancel(home: Path | str) -> None:
    with _guard:
        timer = _timers.pop(str(home), None)
    if timer is not None:
        timer.cancel()


def check_on_start(home: Path | str, queue) -> None:
    """При запуске агента: в очереди есть разговор — разобрать сейчас или дождаться порога."""
    try:
        age = queue.newest_age_seconds()
        waiting = age is not None or queue.count_topics() > 0
    except Exception:
        logger.debug("Память: очередь при запуске прочитать не удалось", exc_info=True)
        return
    if not waiting:
        return
    remaining = idle_seconds() - (age if age is not None else idle_seconds())
    with _guard:
        already = str(home) in _timers
    if not already:
        touch(home, delay=max(5.0, remaining))


def _fire(key: str) -> None:
    with _guard:
        _timers.pop(key, None)
    run_now(Path(key))


def run_now(home: Path) -> bool:
    """Разобрать очередь. False — разбор уже идёт в другом процессе или не удался."""
    lock = home / LOCK_NAME
    if not _acquire(lock):
        logger.info("Память: разбор уже идёт в другом процессе, этот пропущен")
        return False
    started = time.monotonic()
    try:
        from .nightly_job import consolidate

        report = consolidate(home)
        logger.info("Память: разбор по паузе за %.0f с — %s", time.monotonic() - started,
                    "; ".join(line.strip() for line in report))
        return True
    except Exception:
        logger.warning("Память: разбор по паузе не удался, очередь цела и подождёт", exc_info=True)
        return False
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _acquire(lock: Path) -> bool:
    """Один разбор на профиль, даже если шлюз и консоль открыты одновременно."""
    for _ in (1, 2):
        try:
            handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, str(os.getpid()).encode())
            os.close(handle)
            return True
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime < STALE_LOCK_SECONDS:
                    return False
                lock.unlink()
            except OSError:
                return False
        except OSError:
            return False
    return False
