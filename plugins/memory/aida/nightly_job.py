"""Разбор общей памяти: довезти опоздавшее, вытащить суть, дозаполнить поиск.

Имя файла осталось от времён, когда это делалось раз в сутки. Теперь разбор запускается,
когда оператор замолчал на час (idle.py), а реплики попадают в базу сразу после хода; сюда
доезжает только то, что не записалось из-за недоступной базы. Три вещи, в этом порядке:

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
    from .write import normalise_type, scrub, source_tag
except ImportError:  # запуск файлом, как его зовёт расписание
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plugins.memory.aida.queue_db import TurnQueue
    from plugins.memory.aida.write import normalise_type, scrub, source_tag

# Сколько работы за один прогон. Ограничения стоят не ради скорости, а ради чужой квоты:
# ключ к моделям общий с ботом, и его ответы человеку не должны её терять.
MAX_TURNS_PER_RUN = 600
MAX_EXCHANGES_PER_RUN = 80  # доразбор старой очереди, оставшейся с однопроходной схемы
MAX_TOPICS_PER_RUN = 60
MAX_EMBEDDINGS_PER_RUN = 200
EMBED_BATCH = 8

# Короче этого — «ок», «ага», «поправляй». Порог взят у бота: он измерен на его переписке.
MIN_EXCHANGE_LENGTH = 12
MAX_FACTS_PER_EXCHANGE = 5
# Разговор и так лежит в базе дословно и находится поиском, поэтому факт — исключение, а не
# итог каждой темы. При восьми на тему разбор за 18 дней записал 260 «фактов» из сотни обменов.
MAX_FACTS_PER_TOPIC = 2
# Сколько дней факт из разбора живёт на испытательном сроке: не пригодился ни разу — уходит в архив.
PROBATION_DAYS = 14.0

# Сессия длиннее этого на разметку не идёт целиком — один вызов размётки должен читать
# кусок, который влезает в голову модели за раз, а не весь день разговора.
MAX_TURNS_PER_SEGMENT_CALL = 120

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

# Промпт извлечения строже, чем у бота: оператор решил, что память не должна превращаться в
# склад, а сам разговор теперь хранится дословно и ничего не теряется, если факт не извлечён.
SEGMENT_SYSTEM_PROMPT = """Ты размечаешь кусок разговора на смысловые темы — блоки подряд идущих реплик,
которые держатся вокруг одного предмета (одной задачи, решения, события).

Верни ТОЛЬКО JSON-массив, без markdown, без пояснений, без обёрток.
Формат каждого элемента:
{"topic": "короткое название темы (3-8 слов)", "from_index": 0, "to_index": 4}

from_index и to_index — это индексы реплик из списка ниже (0-based, включительно), которые входят в блок.
Блоки идут по порядку и не пропускают реплики — каждая реплика входит ровно в один блок.
Разговор ни о чём (приветствие, «ок», смена темы без сути) — тоже блок, просто мелкий.
Если реплик мало и они об одном — верни один блок на все."""

FACTS_SYSTEM_PROMPT = """Ты решаешь, есть ли в куске диалога то, что нужно помнить через месяц. Обычно — нет.

Сам диалог уже сохранён дословно и находится поиском. Твоя задача — не пересказать его, а заметить
редкое: то, что изменит, как ассистент будет действовать в будущих разговорах.

Верни ТОЛЬКО JSON-массив, без markdown, без пояснений, без обёрток. Не больше двух элементов.
Формат каждого элемента:
{"content": "факт одним предложением от третьего лица", "type": "..."}

type — одно из: decision, rule, preference, fact, project, contact

Извлекай ТОЛЬКО если это:
- decision — оператор принял решение, которое действует дальше («переезжаем на X», «Y больше не используем»)
- rule / preference — оператор сказал, как надо или не надо делать впредь
- fact / project / contact — устойчивое сведение об операторе, его проекте или людях, которое не изменится через неделю

НЕ извлекай:
- ход текущей задачи, что сделано, что сломалось, что починили
- события дня и планы на ближайшие дни
- вопросы, размышления вслух, гипотезы, варианты, которые обсуждали, но не выбрали
- то, что сказал или предложил ассистент, если оператор этого прямо не подтвердил
- пересказ содержания диалога («обсуждали X»)
- то, что уже очевидно из самого предыдущего опыта работы

Сомневаешься — не извлекай. Пустой ответ — нормальный и самый частый.
Пиши content самодостаточно: он будет прочитан без диалога вокруг.
Если извлекать нечего — верни []."""

# Старое имя оставлено как алиас — на случай, если что-то ещё импортирует прежнее имя.
SYSTEM_PROMPT = FACTS_SYSTEM_PROMPT


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


def said_at(turn: dict) -> str | None:
    """Время реплики из очереди (SQLite хранит его в UTC без пояса) — для колонки created_at."""
    stamp = str(turn.get("created_at") or "").strip()
    return f"{stamp}+00" if stamp else None


def ship_turns(store, queue, limit: int = MAX_TURNS_PER_RUN) -> tuple[int, list[dict]]:
    """Довезти реплики, которые не попали в базу сразу после хода (база была недоступна).

    Реплика остаётся в очереди и после доставки — она ещё ждёт разбора; здесь она только
    помечается доставленной. Дата в базе — время, когда это было сказано, а не довезено.
    """
    shipped: list[dict] = []
    for turn in queue.unshipped(limit=limit):
        ids = store.save(
            turn["content"],
            memory_type="dialogue",
            priority="P3",
            role=turn["role"],
            chat_id=source_tag(turn["session_id"]),
            created_at=said_at(turn),
            dedupe=False,  # «да» и «продолжай» повторяются, и каждое было сказано
            embed=False,  # вектора ставит последний проход, пачкой
        )
        if not ids and not store.alive():
            break  # база отказала — остальное подождёт следующего раза
        # Пустой ответ при живой базе значит, что после вычистки секретов от реплики ничего
        # не осталось: держать её в очереди незачем.
        queue.mark_shipped([turn["id"]])
        if ids:
            shipped.append(turn)
    return len(shipped), shipped


# ─── 2. Вытащить суть — двухпроходная схема ────────────────────────────────────
#
# Проход А (разметка): дешёвый разговор режется не по парам вопрос-ответ, а по смыслу —
# несколько реплик подряд об одном деле превращаются в один блок с названием темы.
# Один блок про решение "берём Гермеса", а не пять отдельных обменов внутри него.
#
# Проход Б (извлечение): каждый блок читается моделью целиком, единожды — вместо того
# чтобы гонять модель на каждую пару реплик и терять связь между ними (кто на что отвечал).


def build_sessions(turns: Iterable[dict]) -> dict[str, list[dict]]:
    """Сложить реплики обратно по разговорам, сохранив порядок внутри каждого."""
    by_session: dict[str, list[dict]] = {}
    for turn in turns:
        by_session.setdefault(turn["session_id"], []).append(turn)
    return by_session


def build_exchanges(turns: Iterable[dict]) -> list[tuple[str, str, str]]:
    """Оставлено для доразбора старой очереди (pending_exchanges), см. distill_exchanges."""
    by_session = build_sessions(turns)
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


def _format_turns_for_segmenting(turns: list[dict]) -> str:
    lines = []
    for i, turn in enumerate(turns):
        who = "Человек" if turn["role"] == "user" else "Ассистент"
        lines.append(f"[{i}] {who}: {turn['content']}")
    return "\n\n".join(lines)


def parse_segments(raw: str, n_turns: int) -> list[tuple[str, int, int]]:
    """Достать разметку из ответа модели: список (тема, from_idx, to_idx) внутри [0, n_turns)."""
    if not raw or n_turns <= 0:
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

    segments: list[tuple[str, int, int]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "").strip()
        try:
            from_idx = int(item.get("from_index"))
            to_idx = int(item.get("to_index"))
        except (TypeError, ValueError):
            continue
        if not topic or from_idx < 0 or to_idx < from_idx or to_idx >= n_turns:
            continue
        segments.append((topic, from_idx, to_idx))
    # Модель иногда путает границы соседних блоков — сортировка по началу восстанавливает
    # порядок разговора, даже если сама разметка чуть промахнулась.
    segments.sort(key=lambda s: s[1])
    return segments


def segment_session(session_id: str, turns: list[dict]) -> list[tuple[str, list[dict]]]:
    """Проход А для одной сессии: разбить её реплики на смысловые блоки.

    Длинная сессия режется на куски по MAX_TURNS_PER_SEGMENT_CALL реплик заранее — один
    вызов модели должен видеть разговор целиком, а не половину дня разом.
    """
    blocks: list[tuple[str, list[dict]]] = []
    for chunk_start in range(0, len(turns), MAX_TURNS_PER_SEGMENT_CALL):
        chunk = turns[chunk_start:chunk_start + MAX_TURNS_PER_SEGMENT_CALL]
        if not chunk:
            continue
        content = ask_model([
            {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
            {"role": "user", "content": _format_turns_for_segmenting(chunk)},
        ])
        segments = parse_segments(content, len(chunk)) if content else []
        if not segments:
            # Модель не разметила — весь кусок идёт одним блоком без названия темы,
            # чтобы реплики не потерялись молча.
            segments = [("разговор", 0, len(chunk) - 1)]
        for topic, from_idx, to_idx in segments:
            block_turns = chunk[from_idx:to_idx + 1]
            if block_turns:
                blocks.append((topic, block_turns))
    return blocks


def segment(queue, sessions: dict[str, list[dict]], limit: int = MAX_TOPICS_PER_RUN) -> tuple[int, list[int]]:
    """Разметить сессии на темы и сложить блоки в очередь на извлечение.

    Возвращает число блоков и номера реплик, которые в блоки попали: только их можно убирать
    из очереди. Сессия, до которой в этот прогон не дошла очередь, ждёт следующего целиком.
    """
    enqueued = 0
    handled: list[int] = []
    for session_id, turns in sessions.items():
        if enqueued >= limit:
            break
        for topic, block_turns in segment_session(session_id, turns):
            payload = [{"role": t["role"], "content": t["content"]} for t in block_turns]
            if queue.enqueue_topic(session_id, topic, payload):
                enqueued += 1
        handled.extend(int(t["id"]) for t in turns)
    return enqueued, handled


def _format_topic_for_extraction(topic: str, turns: list[dict]) -> str:
    lines = [f"Тема: {topic}", ""]
    for turn in turns:
        who = "Человек" if turn.get("role") == "user" else "Ассистент"
        lines.append(f"{who}: {turn.get('content', '')}")
    return "\n\n".join(lines)


def distill_topics(store, queue, limit: int = MAX_TOPICS_PER_RUN) -> int:
    """Проход Б: разобрать размеченные темой блоки и записать найденные факты.

    Блок целиком идёт в модель одним вызовом — так факт вида «решили X после того как
    обсудили Y и Z» не теряется, как терялся бы при разборе по одной паре реплик.
    """
    written = 0
    unanswered = 0
    for topic_row in queue.pending_topics(limit=limit):
        turns = topic_row["turns"]
        transcript = _format_topic_for_extraction(topic_row["topic"], turns)
        if len(transcript.strip()) < MIN_EXCHANGE_LENGTH:
            queue.release_topics([topic_row["id"]])
            continue
        content = ask_model([
            {"role": "system", "content": FACTS_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ])
        if not content:
            unanswered += 1
            continue
        for fact in parse_facts(content, limit=MAX_FACTS_PER_TOPIC):
            if store.knows(fact["content"]):
                continue  # это уже в памяти, пусть и другими словами
            if store.save(fact["content"], memory_type=fact["type"], priority=fact["priority"],
                          role="system", chat_id=source_tag("nightly"), embed=False):
                written += 1
        queue.release_topics([topic_row["id"]])
    if unanswered:
        print(f"  разбор: модель не ответила на {unanswered} тем — они разберутся в следующий раз")
    return written


def parse_facts(raw: str, limit: int = MAX_FACTS_PER_EXCHANGE) -> list[dict]:
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
            # Что бы модель ни поставила — P3, испытательный срок: факт из разбора должен
            # сначала хоть раз пригодиться. P1 это ядро, и попадает туда только осознанная
            # запись: разбор, который сам назначал P1 трём фактам из четырёх, за две недели
            # вытеснил из ядра всё написанное руками.
            "priority": "P3",
        })
        if len(facts) >= limit:
            break
    return facts


def ask_model(messages: list[dict]) -> str | None:
    """Спросить модель один раз: сначала через слой Гермеса, потом по прямому ключу.

    Возвращает текст ответа или None, если не ответил никто. Молчание здесь обязано быть
    отличимо от «ничего важного не нашлось» — иначе потерянный факт выглядит как работа.
    """
    # Turns queued before redaction existed may still hold a credential in clear.
    messages = [{**m, "content": scrub(str(m.get("content", "")))} for m in messages]
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


def distill_exchanges(store, queue, limit: int = MAX_EXCHANGES_PER_RUN) -> int:
    """Доразобрать хвост старой однопроходной очереди (pending_exchanges).

    Новые разговоры туда больше не попадают — их с этой ночи размечает segment().
    Функция остаётся, пока в очереди могут лежать обмены, поставленные до перехода на
    двухпроходную схему; когда pending_exchanges опустеет насовсем, её можно будет убрать.

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


def distilling_enabled() -> bool:
    """AIDA_DISTILL=off — жить только на осознанных записях и поиске по разговору."""
    return os.environ.get("AIDA_DISTILL", "on").strip().lower() not in ("off", "0", "false", "no")


def probation_days() -> float:
    try:
        return max(1.0, float(os.environ.get("AIDA_PROBATION_DAYS", "") or PROBATION_DAYS))
    except ValueError:
        return PROBATION_DAYS


def consolidate(hermes_home: Path) -> list[str]:
    """Один прогон разбора. Возвращает отчёт построчно — его печатает скрипт и пишет в журнал триггер.

    Зовётся, когда оператор замолчал (см. idle.py), и вручную — скриптом. Реплики к этому
    моменту уже в базе: их кладёт туда сам ход. Здесь остаётся то, что требует модели и
    времени, и чему самое место в паузе: кеш поставщика к этому часу всё равно остыл.
    """
    load_profile_env(hermes_home)
    db_url = os.environ.get("SUPABASE_DB_URL", "")
    if not db_url:
        return ["Общая память не настроена: нет строки подключения. Разбор пропущен."]

    # Импорт здесь, а не наверху: модуль пакета тянет за собой драйвер базы, а этот файл
    # должен оставаться читаемым и запускаемым даже там, где драйвера нет.
    from plugins.memory.aida import _Store

    store = _Store(db_url, os.environ.get("OPENROUTER_API_KEY", ""))
    queue = TurnQueue(hermes_home / "aida_queue.db")
    report: list[str] = []
    try:
        late = len(queue.unshipped(limit=MAX_TURNS_PER_RUN))
        shipped, _ = ship_turns(store, queue)

        ready = queue.shipped(limit=MAX_TURNS_PER_RUN)
        distilling = distilling_enabled()
        topics_enqueued, facts, old_exchanges_left = 0, 0, 0
        if distilling:
            # Проход А: разметить доставленный разговор на темы и сложить блоки в очередь.
            sessions = build_sessions(turn for turn in ready if turn["distill"])
            topics_enqueued, handled = segment(queue, sessions)
            # Ход, из которого агент сам записал факт, разбирать незачем — он уходит без разметки.
            queue.release(handled + [int(turn["id"]) for turn in ready if not turn["distill"]])

            # Проход Б: разобрать размеченные темой блоки на факты.
            facts = distill_topics(store, queue)

            # Хвост старой очереди — обмены, поставленные до перехода на двухпроходную схему.
            old_exchanges_left = queue.count_exchanges()
            if old_exchanges_left:
                facts += distill_exchanges(store, queue)
        else:
            # Разбор выключен: разговор уже в базе, очередь просто освобождается.
            queue.release(int(turn["id"]) for turn in ready)

        kept, archived = store.settle_probation(probation_days())
        filled, left = backfill_embeddings(store)

        report.append("Разбор общей памяти")
        if late:
            report.append(f"  разговор: довезено {shipped} реплик из {late} опоздавших")
        if not distilling:
            report.append("  разбор на факты выключен (AIDA_DISTILL=off)")
        else:
            report.append(f"  темы: размечено {topics_enqueued} блоков, ждут разбора {queue.count_topics()}")
        if old_exchanges_left:
            report.append(f"  старая очередь: доразобрано {old_exchanges_left}, осталось {queue.count_exchanges()}")
        report.append(f"  суть: записано фактов — {facts}")
        report.append(f"  испытательный срок: пригодились и оставлены — {kept}, не пригодились и убраны в архив — {archived}")
        report.append(f"  поиск по смыслу: дозаполнено {filled}, осталось без него {left}")
        if not store.alive():
            report.append("  база была недоступна — всё, что не доехало, ждёт в очереди")
    finally:
        queue.close()
        store.close()
    return report


def run(hermes_home: Path) -> int:
    from .idle import consolidate_locked

    report = consolidate_locked(hermes_home)
    if report is None:
        print("Разбор уже идёт (замок aida_consolidate.lock занят) — этот запуск пропущен.")
        return 1
    for line in report:
        print(line)
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
