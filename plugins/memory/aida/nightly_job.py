"""Ночная работа общей памяти: довезти разговор, вытащить суть, дозаполнить поиск.

Днём разговор копится на ноуте: ответ не должен ждать базу на другом конце интернета.
Ночью этот скрипт делает три вещи, в этом порядке:

1. **Довозит дословное.** Каждая реплика из локальной очереди ложится в общую базу как есть.
   Строка уходит из очереди только после того, как база подтвердила запись, поэтому упавший
   прогон стоит задержки, а не разговора.
2. **Вытаскивает суть.** Из привезённых обменов дешёвая модель достаёт то, что стоит помнить
   надолго. Разбор ошибётся — оригинал на месте и разбирается заново; в этом весь смысл того,
   чтобы не решать «что важно» в момент разговора.
3. **Дозаполняет поиск.** У записи без вектора нет смыслового поиска: её находит только
   совпадение слов. На 21 сентября 2026 таких было 463 из 525 — это и есть давняя жалоба
   «память есть, а поиск не работает».

Запускается без модели-агента: `hermes cron` зовёт его как обычный скрипт, и всё, что он
печатает, попадает в отчёт целиком.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

try:  # обычный путь: модуль плагина
    from .queue_db import TurnQueue
    from .write import normalise_type, source_tag
except ImportError:  # запуск файлом, как его зовёт расписание
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plugins.memory.aida.queue_db import TurnQueue
    from plugins.memory.aida.write import normalise_type, source_tag

# Сколько работы за один прогон. Ограничения стоят не ради скорости, а ради чужой квоты:
# ключ к моделям общий с ботом, и его ответы человеку не должны её терять.
MAX_TURNS_PER_RUN = 600
MAX_EXCHANGES_PER_RUN = 80
MAX_EMBEDDINGS_PER_RUN = 200
EMBED_BATCH = 8

# Короче этого — «ок», «ага», «поправляй». Порог взят у бота: он измерен на его переписке.
MIN_EXCHANGE_LENGTH = 12
MAX_FACTS_PER_EXCHANGE = 5

# Ночь не торопится, а модель с рассуждением отвечает дольше, чем кажется: на двадцати
# секундах разбор молча возвращал ноль, хотя модель работала (замер 21.09.2026).
DISTILL_TIMEOUT = 60.0

# Разбор ведёт Haiku — работа лёгкая (прочитать обмен и назвать, что в нём стоит помнить),
# и она ей по силам: в замере 21.09.2026 она достала из обмена три факта с верными типами
# за 4 секунды, там где прежняя модель доставала один и думала двадцать.
#
# Зовём её через собственный слой моделей Гермеса: он уже умеет и подписку, и обновление
# доступа, и запасные пути. Своя копия авторизации рядом означала бы второе место, где всё
# это ломается.
DISTILL_MODEL = os.environ.get("AIDA_DISTILL_MODEL", "claude-haiku-4-5")

# Запасной путь — прямой вызов по ключу, для случая, когда скрипт запускают отдельно от
# агента и его слой моделей недоступен.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = os.environ.get("AIDA_DISTILL_GEMINI_MODEL", "gemini-3.6-flash")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Запасных моделей по умолчанию НЕТ, и это осознанно. Бесплатные слаги, на которые
# рассчитывал бот, закрыты: OpenRouter отвечает 404 и предлагает платную версию
# (проверено 21.09.2026 на всех трёх). Подставить платную молча значит начать тратить
# чужие деньги без спроса. Кому нужен запасной вариант — перечисляет модели сам:
#   AIDA_DISTILL_MODELS=deepseek/deepseek-chat-v3.1,mistralai/mistral-small-3.2-24b-instruct
OPENROUTER_MODELS = tuple(
    model.strip() for model in os.environ.get("AIDA_DISTILL_MODELS", "").split(",") if model.strip()
)

# Дословно тот же промпт, что у бота. Два разных определения «что стоит помнить» в одной
# базе — это две разные памяти, между которыми человеку придётся выбирать.
SYSTEM_PROMPT = """Ты — экстрактор фактов из диалога. Твоя задача — вытащить только то, что стоит помнить надолго.

Верни ТОЛЬКО JSON-массив, без markdown, без пояснений, без обёрток.
Формат каждого элемента:
{"content": "факт одним предложением от третьего лица", "type": "...", "priority": "P1"|"P2"}

type — одно из: fact, decision, preference, event, context, rule, project, contact

priority:
- P1 — устойчивое и важное: личные данные пользователя, принятые решения, правила и принципы работы, ключевые факты о проекте, стабильные предпочтения.
- P2 — полезное, но временное: детали текущей задачи, разовые события, контекст этой недели.

НЕ извлекай:
- вопросы, размышления вслух, гипотезы
- то, что сказал ассистент о себе
- пересказ содержания диалога («обсуждали X»)
- то, что верно только в этой реплике

Пиши content самодостаточно: он будет прочитан без диалога вокруг.
Если извлекать нечего — верни []."""


def load_profile_env(hermes_home: Path) -> None:
    """Ключи лежат в `.env` профиля. Скрипт может быть запущен и в обход агента."""
    env_file = hermes_home / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ─── 1. Довезти дословное ─────────────────────────────────────────────────────


def ship_turns(store, queue, limit: int = MAX_TURNS_PER_RUN) -> tuple[int, list[dict]]:
    """Перенести реплики из очереди в общую базу. Возвращает счёт и то, что доехало."""
    pending = queue.pending(limit=limit)
    shipped: list[dict] = []
    for turn in pending:
        ids = store.save(
            turn["content"],
            memory_type="dialogue",
            priority="P2",
            role=turn["role"],
            chat_id=source_tag(turn["session_id"]),
            embed=False,  # вектора ставит третий проход, пачкой
        )
        if not ids:
            # База отказала или это уже есть. Первое — повод остановиться и подождать
            # следующей ночи; второе оставило бы строку в очереди навсегда, поэтому
            # отличаем: пустой ответ при живой базе значит дубликат, его отпускаем.
            if not store.alive():
                break
            queue.release([turn["id"]])
            continue
        queue.release([turn["id"]])
        shipped.append(turn)
    return len(shipped), shipped


# ─── 2. Вытащить суть ─────────────────────────────────────────────────────────


def build_exchanges(turns: Iterable[dict]) -> list[tuple[str, str, str]]:
    """Сложить реплики обратно в обмены «человек → ответ», по каждому разговору отдельно."""
    by_session: dict[str, list[dict]] = {}
    for turn in turns:
        by_session.setdefault(turn["session_id"], []).append(turn)

    exchanges: list[tuple[str, str, str]] = []
    for session_id, session_turns in by_session.items():
        question = None
        for turn in session_turns:
            if turn["role"] == "user":
                question = turn["content"]
            elif turn["role"] == "assistant" and question:
                exchanges.append((session_id, question, turn["content"]))
                question = None
    return exchanges


def parse_facts(raw: str) -> list[dict]:
    """Достать массив из ответа модели. Дешёвые модели заворачивают JSON в прозу и в забор."""
    if not raw:
        return []
    text = re.sub(r"```json|```", "", str(raw)).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    facts = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not 3 <= len(content) <= 2000:
            continue
        facts.append({
            "content": content,
            "type": normalise_type(item.get("type")),
            "priority": "P1" if item.get("priority") == "P1" else "P2",
        })
        if len(facts) >= MAX_FACTS_PER_EXCHANGE:
            break
    return facts


def ask_model(messages: list[dict]) -> str | None:
    """Спросить модель один раз: сначала через слой Гермеса, потом по прямому ключу.

    Возвращает текст ответа или None, если не ответил никто. Молчание здесь обязано быть
    отличимо от «ничего важного не нашлось» — иначе потерянный факт выглядит как работа.
    """
    content = _ask_through_agent(messages)
    if content:
        return content
    return _ask_over_http(messages)


def _ask_through_agent(messages: list[dict]) -> str | None:
    """Вызов через собственный слой моделей агента — основной путь."""
    try:
        from agent.auxiliary_client import call_llm
    except Exception:
        return None  # скрипт запущен отдельно от агента
    try:
        response = call_llm(
            task="memory_extraction",
            model=DISTILL_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=800,
            timeout=DISTILL_TIMEOUT,
        )
        return response.choices[0].message.content or None
    except Exception as exc:
        print(f"  разбор: модель агента недоступна ({type(exc).__name__}), пробую по ключу")
        return None


def _ask_over_http(messages: list[dict]) -> str | None:
    """Запасной путь: прямой вызов по ключу из окружения."""
    targets = _model_targets()
    if not targets:
        return None
    import requests

    for target in targets:
        try:
            response = requests.post(
                f"{target['url']}/chat/completions",
                headers={"Authorization": f"Bearer {target['key']}", "Content-Type": "application/json"},
                json={"model": target["model"], "messages": messages,
                      "temperature": 0, "max_tokens": 800, "stream": False},
                timeout=DISTILL_TIMEOUT,
            )
            if response.status_code >= 400:
                continue
            content = response.json()["choices"][0]["message"]["content"]
            if content:
                return content
        except Exception:
            continue
    return None


def _model_targets() -> list[dict]:
    """Цепочка попыток. У Gemini своя квота, поэтому он первый, если ключ есть."""
    targets = []
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        model = os.environ.get("AIDA_DISTILL_GEMINI_MODEL", GEMINI_MODEL)
        targets.append({"label": "gemini", "url": GEMINI_BASE_URL, "key": gemini_key, "model": model})
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    fallbacks = tuple(
        model.strip() for model in os.environ.get("AIDA_DISTILL_MODELS", "").split(",") if model.strip()
    ) or OPENROUTER_MODELS
    if openrouter_key:
        targets += [
            {"label": "openrouter", "url": OPENROUTER_BASE_URL, "key": openrouter_key, "model": model}
            for model in fallbacks
        ]
    return targets


def distill(store, queue, limit: int = MAX_EXCHANGES_PER_RUN) -> int:
    """Разобрать ждущие обмены и положить найденное в общую базу.

    Обмен покидает очередь, только когда модель на него ответила. Не ответила — он ждёт
    следующей ночи: реплики к тому времени уже в базе, и разобрать их иначе было бы нечем.
    """
    written = 0
    unanswered = 0
    for exchange in queue.pending_exchanges(limit=limit):
        question, answer = exchange["question"], exchange["answer"]
        if len(question.strip()) < MIN_EXCHANGE_LENGTH:
            queue.release_exchanges([exchange["id"]])  # слишком коротко, чтобы нести факт
            continue
        content = ask_model([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Человек: {question}\n\nАссистент: {answer}"},
        ])
        if not content:
            unanswered += 1
            continue
        for fact in parse_facts(content):
            if store.save(fact["content"], memory_type=fact["type"], priority=fact["priority"],
                          role="system", chat_id=source_tag("nightly"), embed=False):
                written += 1
        queue.release_exchanges([exchange["id"]])
    if unanswered:
        # Молчаливый ноль здесь читался бы как «в разговоре не было ничего важного»,
        # а это совсем другая новость.
        print(f"  разбор: модель не ответила на {unanswered} обменов — они разберутся в следующий раз")
    return written


# ─── 3. Дозаполнить поиск ─────────────────────────────────────────────────────


def backfill_embeddings(store, limit: int = MAX_EMBEDDINGS_PER_RUN) -> tuple[int, int]:
    """Проставить векторы записям, у которых их нет. Возвращает «сделано» и «осталось»."""
    rows = store.rows_without_embedding(limit)
    filled = 0
    for start in range(0, len(rows), EMBED_BATCH):
        batch = rows[start:start + EMBED_BATCH]
        vectors = store.embed_many([str(row[1]) for row in batch])
        if vectors is None:
            break  # служба молчит целиком — не тратим ночь на повтор
        for (row_id, _), vector in zip(batch, vectors):
            # Пустой вектор — это одна запись, которую служба не приняла. Она остаётся
            # находимой по словам, и следующая пачка идёт как ни в чём не бывало.
            if vector and store.set_embedding(int(row_id), vector):
                filled += 1
    return filled, store.count_without_embedding()


# ─── Прогон ───────────────────────────────────────────────────────────────────


def run(hermes_home: Path) -> int:
    load_profile_env(hermes_home)
    db_url = os.environ.get("SUPABASE_DB_URL", "")
    if not db_url:
        print("Общая память не настроена: нет строки подключения. Ночная работа пропущена.")
        return 0

    # Импорт здесь, а не наверху: модуль пакета тянет за собой драйвер базы, а этот файл
    # должен оставаться читаемым и запускаемым даже там, где драйвера нет.
    from plugins.memory.aida import _Store

    store = _Store(db_url, os.environ.get("OPENROUTER_API_KEY", ""))
    queue = TurnQueue(hermes_home / "aida_queue.db")
    try:
        waiting = queue.count()
        shipped, turns = ship_turns(store, queue)
        for session_id, question, answer in build_exchanges(turns):
            queue.enqueue_exchange(session_id, question, answer)
        facts = distill(store, queue)
        filled, left = backfill_embeddings(store)

        print("Ночная работа общей памяти")
        print(f"  разговор: довезено {shipped} реплик из {waiting}, ждут ещё {queue.count()}")
        print(f"  суть: записано фактов — {facts}, ждут разбора {queue.count_exchanges()} обменов")
        print(f"  поиск по смыслу: дозаполнено {filled}, осталось без него {left}")
        if not store.alive():
            print("  база в эту ночь была недоступна — всё, что не доехало, ждёт в очереди")
    finally:
        queue.close()
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if argv:
        home = Path(argv[0])
    else:
        try:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        except Exception:
            home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    return run(Path(home))


if __name__ == "__main__":
    raise SystemExit(main())
