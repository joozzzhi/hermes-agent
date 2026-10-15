"""Настройки витрины из секции `aida:` конфига профиля.

Один шлюз обслуживает несколько профилей, поэтому настройки не живут в константах модуля и не
берутся из окружения процесса: на каждом запросе читается config.yaml того профиля, чей это ход
(`get_hermes_home()` хозяин привязывает к ходу). Чтение дешёвое — хозяин кеширует разбор файла
по его подписи.

Порядок: значение в `aida:` → старая переменная окружения `AIDA_*` (для тех трёх, что были) →
значение по умолчанию. Значения по умолчанию — ровно те, что раньше стояли константами.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AidaSettings:
    # Бюджет запроса — доля окна модели; нижняя граница для маленьких окон; бюджет, если окно
    # неизвестно; точный бюджет в токенах (0 — не задан).
    context_share: float = 0.05
    min_context_tokens: int = 60_000
    unknown_window_tokens: int = 50_000
    context_tokens: int = 0
    # Ядро занимает не больше этой доли бюджета; хвосту — не меньше этой.
    knowledge_share: float = 0.2
    min_tail_share: float = 0.4
    # Стартовый курс знаков на токен (дальше уточняется по счёту поставщика). Около двух — к
    # этому счёт сходится на живом разговоре: в токены входят и описания инструментов.
    chars_per_token: float = 2.0
    # Хвост: доля переписки, доля выводов инструментов, сколько обменов защищено.
    conversation_share: float = 0.25
    keep_results_share: float = 0.35
    protected_exchanges: int = 10
    # Потолок одного вывода в окне, сколько конца оставить, сколько дней хранить полный вывод.
    tool_result_cap: int = 12_000
    cap_tail_chars: int = 2_500
    tool_output_max_age_days: int = 30
    # Жизнь кеша поставщика; 0 — как у хозяина (`prompt_caching.cache_ttl`).
    cache_ttl_seconds: float = 0.0
    # Сколько последних выводов инструментов не становятся заглушкой, какой бы ни была доля
    # (0 — граница только по доле). Гарантия, что последний read_file или вывод тестов цел.
    keep_last_tool_results: int = 0
    # Выводы этих инструментов не становятся заглушкой никогда (например, todo — план задачи).
    exclude_tools: tuple = ()
    # Какие типы записей ядра видит профиль (rule, decision, project, lesson…); пусто — все.
    core_types: tuple = ()

    @property
    def step_keep(self) -> float:
        return self.conversation_share + self.keep_results_share


DEFAULTS = AidaSettings()

# Переменные окружения, которые движок понимал до секции в конфиге.
_LEGACY_ENV = {
    "context_share": "AIDA_CONTEXT_SHARE",
    "context_tokens": "AIDA_CONTEXT_TOKENS",
    "cache_ttl_seconds": "AIDA_CACHE_TTL_SECONDS",
}


def _section() -> dict:
    try:
        from hermes_cli.config import read_raw_config_readonly

        section = read_raw_config_readonly().get("aida")
    except Exception:
        logger.debug("Витрина: конфиг профиля не прочитан, беру значения по умолчанию", exc_info=True)
        return {}
    return section if isinstance(section, dict) else {}


def _names(raw) -> tuple | None:
    """Список имён из конфига: YAML-список или строка через запятую. Порядок не важен — сортируем,
    чтобы одинаковый смысл давал одинаковые настройки."""
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        items = raw
    else:
        return None
    if not all(isinstance(item, str) for item in items):
        return None
    return tuple(sorted({item.strip() for item in items if item.strip()}))


def load_settings() -> AidaSettings:
    """Настройки для текущего хода. Кривое значение — предупреждение и значение по умолчанию."""
    section = _section()
    values = {}
    for item in fields(AidaSettings):
        default = getattr(DEFAULTS, item.name)
        raw = section.get(item.name)
        if raw is None and item.name in _LEGACY_ENV:
            raw = os.environ.get(_LEGACY_ENV[item.name]) or None
        if raw is None:
            continue
        if isinstance(default, tuple):
            names = _names(raw)
            if names is None:
                logger.warning("Витрина: aida.%s = %r — нужен список имён, беру %s", item.name, raw, default)
                continue
            values[item.name] = names
            continue
        try:
            value = type(default)(float(raw)) if isinstance(default, int) else float(raw)
        except (TypeError, ValueError):
            logger.warning("Витрина: aida.%s = %r — не число, беру %s", item.name, raw, default)
            continue
        if value < 0:
            logger.warning("Витрина: aida.%s = %r — меньше нуля, беру %s", item.name, raw, default)
            continue
        values[item.name] = value
    unknown = sorted(set(section) - {item.name for item in fields(AidaSettings)})
    if unknown:
        logger.debug("Витрина: в секции aida незнакомые ключи %s", unknown)
    return AidaSettings(**values) if values else DEFAULTS
