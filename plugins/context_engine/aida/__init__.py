"""Контекстный движок «витрина»: разговор не пересказывается никогда.

Встроенный движок копит разговор в окне, а когда оно заполняется — пересказывает старую
часть и продолжает с пересказа. Дальше модель опирается на пересказ и достраивает утраченные
детали; это прямой источник выдуманных подробностей, и именно поэтому так делать запрещено.

Здесь вместо этого повторено устройство движка, который работал на сервере, — три уровня:

* **Ядро** (P1 — личность оператора и то, что верно всегда) стоит в системной части. Его
  собирает этот движок.
* **Найденное под вопрос** (факты и куски давнего разговора) приходит вместе с вопросом. Его
  ищет плагин памяти `aida`, а хозяин навсегда прикрепляет к реплике, под которую оно найдено.
* **Свежий хвост разговора** остаётся дословно, настоящими сообщениями. Его ведёт этот движок.

Движок на сервере собирал запрос с нуля на каждое сообщение: там каждый ответ был отдельным
запуском. Гермес ведёт один длинный разговор, и повторять это здесь буквально значит на каждом
ходу выбрасывать кеш поставщика. Поэтому правило другое: запрос только ДОПИСЫВАЕТСЯ в конец, а
всё, что меняет его середину (ядро, начало хвоста, заглушки на месте старых результатов
инструментов), решается разом и редко — когда хвост перестал помещаться или когда оператор
молчал дольше, чем живёт кеш, и терять уже нечего.

Ничего не пересказывается: всё, что ушло из хвоста, лежит в общей памяти дословно и находится
поиском. Включается строкой `context.engine: aida`; без плагина памяти и даже без базы движок
работает — ведёт хвост, просто без ядра.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from agent.context_engine import ContextEngine

from .settings import DEFAULTS, AidaSettings, load_settings
from .assembly import (
    AGE_MIN_CHARS,
    PROTECTED_EXCHANGES,
    TailState,
    apply_state,
    breakdown,
    build_knowledge_block,
    count_aged,
    is_conversation_message,
    merge_into_system,
    message_key,
    message_size,
    rebase,
    select_tail,
    split_system,
)

logger = logging.getLogger(__name__)

# Бюджет запроса — доля окна модели, а не число: у окна в миллион токенов и у окна в двести
# тысяч «много» значит разное. Нижняя граница нужна маленьким окнам, чтобы хвост не схлопнулся.
# Значения по умолчанию; профиль переопределяет их в секции `aida:` своего конфига (settings.py).
CONTEXT_SHARE = DEFAULTS.context_share
MIN_CONTEXT_TOKENS = DEFAULTS.min_context_tokens
UNKNOWN_WINDOW_TOKENS = DEFAULTS.unknown_window_tokens
# Ядро занимает не больше этой доли бюджета; что не влезло, находится поиском.
KNOWLEDGE_SHARE = DEFAULTS.knowledge_share
# Хвосту остаётся не меньше этой доли, какой бы большой ни была системная часть хозяина.
MIN_TAIL_SHARE = DEFAULTS.min_tail_share

# Бюджет считается в знаках, потому что знаки можно посчитать, не трогая модель. Сколько знаков
# в токене, заранее неизвестно (у русского текста около двух с половиной, у кода и JSON больше),
# поэтому число уточняется по каждому ответу: поставщик сообщает, во сколько токенов обошёлся
# запрос, а сколько в нём было знаков, движок знает сам.
DEFAULT_CHARS_PER_TOKEN = DEFAULTS.chars_per_token
CHARS_PER_TOKEN_RANGE = (2.0, 4.5)
CALIBRATE_FROM_CHARS = 60_000

# Сколько живёт кеш поставщика. Тишина длиннее — кеш остыл, и запрос можно пересобрать даром.
# Берётся из того же места, откуда его берёт хозяин (`prompt_caching.cache_ttl`): движок,
# считающий кеш часовым при пятиминутном кеше, пропускает дешёвую пересборку и платит за
# полную запись кеша после каждой паузы длиннее пяти минут.
CACHE_TTL_SECONDS = {"5m": 300.0, "1h": 3600.0}
DEFAULT_CACHE_TTL = CACHE_TTL_SECONDS["5m"]

MAX_TRACKED_CONVERSATIONS = 64

# Потолок одного вывода инструмента в окне. Замер по state.db обоих профилей (2208 результатов):
# 95% результатов короче 12,3 тыс. знаков; длиннее — 4%, но это 21% всего выхлопа (read_file,
# web_extract, длинный терминал). Остальное уходит на диск, в окне — начало, конец и путь.
TOOL_RESULT_CAP = DEFAULTS.tool_result_cap
CAP_TAIL_CHARS = DEFAULTS.cap_tail_chars
# Куда кладутся полные выводы. Не cache/spillover хозяина: тот чистится через сутки, а
# заглушка в окне обещает, что оригинал можно достать и через неделю.
TOOL_OUTPUT_SUBDIR = ("cache", "aida-tool-output")
TOOL_OUTPUT_MAX_AGE_DAYS = DEFAULTS.tool_output_max_age_days
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def cache_ttl_seconds(settings: AidaSettings | None = None) -> float:
    """Жизнь кеша поставщика: явная настройка `aida.cache_ttl_seconds`, иначе — то, что настроено
    у хозяина в `prompt_caching.cache_ttl` (по умолчанию у него пять минут)."""
    explicit = (settings or load_settings()).cache_ttl_seconds
    if explicit > 0:
        return explicit
    try:
        from agent.agent_runtime_helpers import configured_cache_ttl

        return CACHE_TTL_SECONDS.get(configured_cache_ttl() or "5m", DEFAULT_CACHE_TTL)
    except Exception:
        return DEFAULT_CACHE_TTL


@dataclass
class _Conversation:
    """Что решено про один разговор. Шлюз ведёт их несколько, у каждого своё."""

    tail: TailState | None = None
    knowledge: str | None = None  # None — ядро ещё не удалось прочитать
    last_seen: float = 0.0
    aged: int = 0
    paths: dict = field(default_factory=dict)   # tool_call_id → полный вывод на диске
    lock: threading.Lock = field(default_factory=threading.Lock)


class AidaContextEngine(ContextEngine):
    """Ведёт хвост разговора и ядро в системной части; никогда не пересказывает."""

    # Пересказа нет, поэтому и сообщать не о чем: «отпустил старое» — это норма работы,
    # а не событие. Предупреждения и ошибки хозяин показывает сам.
    emit_automatic_compaction_status = False

    @property
    def name(self) -> str:
        return "aida"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conversations: OrderedDict[int, _Conversation] = OrderedDict()
        self._store: Any = None
        self._store_retry_at = 0.0
        self._announced = False
        self._announced_no_url = False
        # Стартовый курс берётся из настроек профиля при первом подсчёте бюджета.
        self._chars_per_token: float | None = None
        self._last_sent_chars = 0
        self._report: dict | None = None
        self._turn_requests = 0
        self._turn_rebuilds = 0
        self._pruned_homes: set[str] = set()

    # -- хозяйство -----------------------------------------------------------------

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or 0)
        # Уточняем, сколько знаков в токене, по настоящему счёту поставщика. В токены запроса
        # входят ещё описания инструментов, которых движок не видит, поэтому число выходит
        # чуть заниженным — то есть бюджет чуть строже, а не мягче.
        sent = self._last_sent_chars
        if sent >= CALIBRATE_FROM_CHARS and self.last_prompt_tokens > 0 and self._chars_per_token is not None:
            low, high = CHARS_PER_TOKEN_RANGE
            measured = min(high, max(low, sent / self.last_prompt_tokens))
            self._chars_per_token = 0.7 * self._chars_per_token + 0.3 * measured

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Почти никогда: запрос уже собран под бюджет. Остаётся как страховка хозяина."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return bool(self.threshold_tokens and tokens and tokens > self.threshold_tokens)

    def compress(self, messages: list[dict], current_tokens: int = None,
                 focus_topic: str = None, force: bool = False, memory_context: str = "") -> list[dict]:
        """Отпустить старое, а не пересказать его.

        Модель не зовём вообще: всё, что уходит из истории, лежит в общей памяти дословно и
        находится поиском. Результаты инструментов здесь не трогаем — это история хозяина, а
        не запрос; режем только по тем же границам, что и хвост, включая разрез внутри
        длинного хода (иначе ход из одной реплики и сотен обращений не сжимался бы вовсе).
        """
        system_head, conversation = split_system(messages)
        if not conversation:
            return list(messages)
        head_size = sum(message_size(m) for m in system_head)
        settings = load_settings()
        budget = max(self._budget_chars(self.context_length, settings) - head_size, 1)
        kept = apply_state(conversation, rebase(conversation, budget, age=False,
                                                protect=settings.protected_exchanges,
                                                talk_share=settings.conversation_share,
                                                keep_share=settings.keep_results_share))
        if len(kept) < len(conversation):
            self.compression_count += 1
            logger.info("Витрина: отпущено %d старых сообщений, они остаются в памяти",
                        len(conversation) - len(kept))
        return system_head + kept

    def on_turn_complete(self, messages: list[dict], usage: dict[str, Any] = None, **kwargs: Any) -> None:
        """Строка в журнал на каждый ход: куда ушёл бюджет последнего запроса этого хода."""
        report, self._report = self._report, None
        requests, self._turn_requests = self._turn_requests, 0
        rebuilds, self._turn_rebuilds = self._turn_rebuilds, 0
        if report is None:
            return
        logger.info("Витрина: ход — ядро %d (система %d + знание %d), переписка %d, инструменты %d, "
                    "заглушек %d, урезано по потолку %d; сообщений %d из %d, бюджет хвоста %d; "
                    "запросов %d, пересборок %d",
                    report["head"] + report["knowledge"], report["head"], report["knowledge"],
                    report["conversation"], report["tools"], report["stubs"], report["capped"],
                    report["sent"], report["total"], report["tail_budget"], requests, rebuilds)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        with self._lock:
            self._conversations.clear()

    def on_session_end(self, session_id: str, messages: list[dict]) -> None:
        with self._lock:
            store, self._store = self._store, None
            self._store_retry_at = 0.0
        if store is not None:
            store.close()

    # -- бюджет --------------------------------------------------------------------

    def _budget_chars(self, window_tokens: int, settings: AidaSettings | None = None) -> int:
        """Бюджет запроса в знаках: доля окна модели, пересчитанная по настоящему курсу токена."""
        settings = settings or load_settings()
        if self._chars_per_token is None:
            self._chars_per_token = settings.chars_per_token or DEFAULT_CHARS_PER_TOKEN
        if settings.context_tokens > 0:
            tokens = settings.context_tokens
        elif window_tokens and window_tokens > 0:
            tokens = max(settings.min_context_tokens, window_tokens * settings.context_share)
        else:
            tokens = settings.unknown_window_tokens
        return int(tokens * self._chars_per_token)

    # -- витрина -------------------------------------------------------------------

    def select_context(self, request_messages: list[dict], *, conversation_messages: list[dict] = None,
                       incoming_message: dict = None, budget_tokens: int = 0) -> list[dict] | None:
        """Собрать запрос: системная часть с ядром и хвост разговора. None — оставить как есть."""
        if not request_messages:
            return None
        try:
            system_head, conversation = split_system(request_messages)
            if not conversation:
                return None
            settings = load_settings()
            if not self._announced:
                # Один раз за запуск, чтобы «движок работает» было видно в журнале, а не
                # выводилось из того, что ничего не сломалось.
                self._announced = True
                logger.info("Витрина: движок ведёт контекст этого разговора (бюджет — %.0f%% окна)",
                            100 * settings.context_share)
            state = self._conversation(message_key(conversation[0]))
            with state.lock:
                assembled = self._assemble(state, system_head, conversation, budget_tokens, settings)
            if not any(is_conversation_message(m) for m in assembled):
                # До модели доехал бы запрос без единого сообщения, а на такой поставщик
                # отвечает отказом, который не повторяют: ход умер бы целиком. Разговор у нас
                # есть — значит, ошиблась сборка, и правильный ответ здесь один: отойти в
                # сторону и дать хозяину отправить запрос так, как он отправил бы без движка.
                logger.warning("Витрина: в собранном запросе не осталось разговора — "
                               "запрос уходит как есть (сообщений было %d)", len(conversation))
                return None
            self._last_sent_chars = sum(message_size(m) for m in assembled)
            return assembled
        except Exception:
            logger.warning("Витрина: собрать не удалось, запрос уходит как есть", exc_info=True)
            return None

    def _assemble(self, state: _Conversation, system_head: list[dict], conversation: list[dict],
                  budget_tokens: int, settings: AidaSettings) -> list[dict]:
        now = time.monotonic()
        # Оператор молчал дольше жизни кеша: начало запроса всё равно будет прочитано заново,
        # так что это единственный момент, когда его можно менять даром.
        cold = bool(state.last_seen) and now - state.last_seen > cache_ttl_seconds(settings)
        state.last_seen = now
        budget = self._budget_chars(budget_tokens, settings)

        if state.knowledge is None or cold:
            fresh = self._load_knowledge(int(budget * settings.knowledge_share))
            if fresh is not None:
                state.knowledge = fresh
        knowledge = state.knowledge or ""

        system_size = sum(message_size(m) for m in system_head)
        head_size = system_size + len(knowledge)
        tail_budget = max(int(budget * settings.min_tail_share), budget - head_size)
        conversation, capped = self._cap_tool_output(state, conversation, settings)
        tail, state.tail, rebased = select_tail(conversation, tail_budget, state.tail, force=cold,
                                                paths=state.paths, protect=settings.protected_exchanges,
                                                talk_share=settings.conversation_share,
                                                keep_share=settings.keep_results_share)
        talk, tools, stubs = breakdown(tail)
        self._turn_requests += 1
        self._turn_rebuilds += int(rebased)
        self._report = {"head": system_size, "knowledge": len(knowledge), "conversation": talk,
                        "tools": tools, "stubs": stubs, "capped": capped, "sent": len(tail),
                        "total": len(conversation), "tail_budget": tail_budget}
        if rebased:
            aged = count_aged(conversation, state.tail.aged)
            if aged > state.aged:
                # Модель, которой понадобился отпущенный результат, перечитает файл. Без
                # сброса хозяин ответил бы «файл не менялся» — в расчёте на содержимое,
                # которое мы только что убрали из запроса.
                self._reset_read_dedup()
            state.aged = aged
            logger.info("Витрина: хвост пересобран (%s) — %d сообщений из %d, заглушек %d; "
                        "переписка %d, инструменты %d знаков",
                        "после паузы" if cold else "перестал помещаться", len(tail), len(conversation), aged,
                        talk, tools)
        return merge_into_system(system_head, knowledge) + tail

    # -- вывод инструментов ----------------------------------------------------------

    def _cap_tool_output(self, state: _Conversation, conversation: list[dict],
                         settings: AidaSettings = DEFAULTS) -> tuple[list[dict], int]:
        """Каждый длинный вывод инструмента — на диск (один раз), а в окно — не больше потолка.

        Решение зависит только от самого сообщения, поэтому одинаковое на каждом запросе: байты
        запроса не плывут и кеш не страдает. Не записалось на диск — вывод остаётся целым: потолок
        не стоит того, чтобы потерять оригинал.
        """
        out = conversation
        capped = 0
        for index, message in enumerate(conversation):
            content = message.get("content")
            if message.get("role") != "tool" or not isinstance(content, str) or len(content) <= AGE_MIN_CHARS:
                continue
            path = self._saved_output(state, message, content)
            if path is None or len(content) <= settings.tool_result_cap:
                continue
            if out is conversation:
                out = list(conversation)
            out[index] = {**message, "content": capped_preview(content, path, settings.tool_result_cap,
                                                               settings.cap_tail_chars)}
            capped += 1
        return out, capped

    def _output_dir(self):
        from hermes_constants import get_hermes_home

        directory = get_hermes_home().joinpath(*TOOL_OUTPUT_SUBDIR)
        if str(directory) not in self._pruned_homes:
            self._pruned_homes.add(str(directory))
            cutoff = time.time() - load_settings().tool_output_max_age_days * 86400
            try:
                for item in directory.iterdir():
                    if item.is_file() and item.stat().st_mtime < cutoff:
                        item.unlink()
            except OSError:
                pass
        return directory

    def _saved_output(self, state: _Conversation, message: dict, content: str) -> str | None:
        call_id = str(message.get("tool_call_id") or "")
        known = state.paths.get(call_id)
        if known:
            return known
        digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:12]
        stem = _UNSAFE_NAME.sub("_", call_id)[:80].strip("._-") or "result"
        name = message.get("name") or message.get("tool_name") or "tool"
        target = self._output_dir() / f"{_UNSAFE_NAME.sub('_', str(name))[:40]}_{stem}_{digest}.txt"
        try:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".part")
                tmp.write_text(content, encoding="utf-8", errors="replace")
                tmp.replace(target)
        except OSError:
            logger.warning("Витрина: не удалось сохранить вывод инструмента на диск", exc_info=True)
            return None
        path = str(target).replace("\\", "/")
        if call_id:
            state.paths[call_id] = path
        return path

    def _conversation(self, key: int) -> _Conversation:
        with self._lock:
            state = self._conversations.get(key)
            if state is None:
                state = self._conversations[key] = _Conversation()
            self._conversations.move_to_end(key)
            while len(self._conversations) > MAX_TRACKED_CONVERSATIONS:
                self._conversations.popitem(last=False)
            return state

    @staticmethod
    def _reset_read_dedup() -> None:
        """Штатный сброс учёта «этот файл уже читали и он не менялся» (для всех задач сразу)."""
        try:
            from tools.file_tools_read_tracking import reset_file_dedup

            reset_file_dedup(None)
        except Exception:
            logger.debug("Витрина: сбросить учёт чтения файлов не удалось", exc_info=True)
        try:
            from tools.skills_tool_dedup import reset_skill_view_dedup

            reset_skill_view_dedup(None)
        except Exception:
            logger.debug("Витрина: сбросить учёт просмотра навыков не удалось", exc_info=True)

    def _load_knowledge(self, budget: int) -> str | None:
        """Ядро из общей памяти. None — прочитать не удалось, попробуем на следующем запросе."""
        store = self._get_store()
        if store is None:
            return None
        rows = store.core_rows()
        if rows is None:
            return None
        return build_knowledge_block(rows, budget)

    def _get_store(self) -> Any:
        """Та же база, что у памяти Гермеса. Не получилось — пробуем снова через минуту, а не
        никогда: одна неудача при запуске не должна оставлять разговор без ядра до перезапуска."""
        with self._lock:
            if self._store is not None:
                return self._store
            if time.monotonic() < self._store_retry_at:
                return None
            self._store_retry_at = time.monotonic() + 60.0
        try:
            from agent.secret_scope import get_secret
            from plugins.memory.aida import _Store

            url = get_secret("SUPABASE_DB_URL", "") or ""
            if not url:
                (logger.warning if not self._announced_no_url else logger.debug)("Витрина: SUPABASE_DB_URL не задан — движок ведёт хвост, но без ядра из памяти")
                self._announced_no_url = True
                return None
            store = _Store(url, get_secret("OPENROUTER_API_KEY", "") or "")
        except Exception:
            logger.warning("Витрина: подключиться к общей памяти не удалось, повтор через минуту", exc_info=True)
            return None
        with self._lock:
            self._store = store
        return store


def capped_preview(content: str, path: str, cap: int = TOOL_RESULT_CAP, tail_chars: int = CAP_TAIL_CHARS) -> str:
    """Начало и конец вывода и путь к полному файлу — всё вместе не длиннее `cap`."""
    lines = content.count("\n") + 1
    header = (f"[вывод урезан витриной: {len(content)} знаков, {lines} строк; полный — {path}; "
              "дочитывай read_file с offset/limit, кусками по 100–150 строк]\n")
    marker = "\n[… середина опущена — она в файле выше …]\n"
    room = max(cap - len(header) - len(marker), 0)
    tail_room = min(tail_chars, room // 3)
    return header + content[:room - tail_room] + marker + (content[-tail_room:] if tail_room else "")


def register(ctx) -> None:
    """Зарегистрировать витрину как контекстный движок."""
    ctx.register_context_engine(AidaContextEngine())
