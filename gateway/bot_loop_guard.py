"""Sliding-window budget for bot-authored inbound messages.

``{PLATFORM}_ALLOW_BOTS`` only decides admission, so two Hermes profiles replying to each other never stop.
The guard counts admitted bot messages per conversation and drops further ones for ``cooldown_seconds``
once ``max_events`` land inside ``window_seconds``. Settings: config.yaml ``gateway.bot_loop_guard``.

``max_chain`` (0 = off) caps consecutive bot-to-bot hand-offs in one chat across ALL profiles of the process:
distinct bot messages since the last human message (or since ``chain_reset_seconds`` of bot silence).
The message past the cap is dropped, so a human has to speak before bots continue.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Hashable, Tuple

__all__ = ["BotLoopGuard", "BotLoopGuardSettings", "load_settings", "settings_from_config"]

_TRUTHY = frozenset({"true", "1", "yes", "on"})
_FALSY = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class BotLoopGuardSettings:
    enabled: bool = True
    max_events: int = 20
    window_seconds: float = 300.0
    cooldown_seconds: float = 600.0
    max_chain: int = 0
    chain_reset_seconds: float = 900.0


def _as_bool(raw, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower() if raw is not None else ""
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default


def _as_positive(raw, default: float) -> float:
    if isinstance(raw, bool):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _as_positive_int(raw, default: int) -> int:
    value = _as_positive(raw, 0.0)
    return int(value) if value >= 1 and value == int(value) else default


def _as_non_negative_int(raw, default: int) -> int:
    if raw is None or isinstance(raw, bool):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return int(value) if value >= 0 and value == int(value) else default


def settings_from_config(cfg) -> BotLoopGuardSettings:
    """Read ``gateway.bot_loop_guard`` from a loaded config dict. Unusable values keep the default."""
    from hermes_cli.config import cfg_get

    block = cfg_get(cfg, "gateway", "bot_loop_guard", default=None)
    if not isinstance(block, dict):
        return BotLoopGuardSettings()
    defaults = BotLoopGuardSettings()
    return BotLoopGuardSettings(
        enabled=_as_bool(block.get("enabled"), defaults.enabled),
        max_events=_as_positive_int(block.get("max_events"), defaults.max_events),
        window_seconds=_as_positive(block.get("window_seconds"), defaults.window_seconds),
        cooldown_seconds=_as_positive(block.get("cooldown_seconds"), defaults.cooldown_seconds),
        max_chain=_as_non_negative_int(block.get("max_chain"), defaults.max_chain),
        chain_reset_seconds=_as_positive(block.get("chain_reset_seconds"), defaults.chain_reset_seconds),
    )


def load_settings() -> BotLoopGuardSettings:
    """Settings from the live config.yaml. Defaults when the config cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly

        return settings_from_config(load_config_readonly())
    except Exception:
        return BotLoopGuardSettings()


class BotLoopGuard:
    """Per-conversation sliding window with a cooldown once the budget trips. Thread-safe.
    Settings are re-read on every call so a config.yaml edit takes effect without a restart."""

    def __init__(
        self,
        settings: Callable[[], BotLoopGuardSettings] = load_settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._events: Dict[Hashable, Deque[float]] = {}
        self._cooldown_until: Dict[Hashable, float] = {}
        # chat -> (message ids of the current bot chain, time of the last bot message)
        self._chains: Dict[Hashable, Tuple[set, float]] = {}
        self._last_sweep = 0.0

    @property
    def tracked_conversations(self) -> int:
        with self._lock:
            return len(self._events)

    def blocked(self, conversation: Hashable) -> bool:
        """True while ``conversation`` is cooling down. Reads only, so callers may ask as often as they like."""
        if not self._settings().enabled:
            return False
        with self._lock:
            return self._cooldown_until.get(conversation, 0.0) > self._clock()

    def admit(self, conversation: Hashable) -> Tuple[bool, str]:
        """Count one admitted bot-authored message for ``conversation``.
        Returns ``(allowed, state)``; state is ``disabled``, ``ok``, ``tripped`` (this message started the cooldown) or ``cooldown``."""
        settings = self._settings()
        if not settings.enabled:
            return True, "disabled"
        now = self._clock()
        with self._lock:
            self._sweep(now, settings)
            if self._cooldown_until.get(conversation, 0.0) > now:
                return False, "cooldown"
            events = self._events.setdefault(conversation, deque())
            cutoff = now - settings.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= settings.max_events:
                self._cooldown_until[conversation] = now + settings.cooldown_seconds
                events.clear()
                return False, "tripped"
            events.append(now)
            return True, "ok"

    def chain(self, chat: Hashable, message_id, is_bot: bool) -> Tuple[bool, int]:
        """Track the bot-to-bot hand-off chain of ``chat``. A human message resets it.
        Returns ``(allowed, length)``; a bot message past ``max_chain`` is refused. Re-delivery of
        the same message id (several profiles, busy path + drain) counts once."""
        settings = self._settings()
        if not settings.enabled or settings.max_chain <= 0:
            return True, 0
        now = self._clock()
        with self._lock:
            if not is_bot:
                self._chains.pop(chat, None)
                return True, 0
            ids, last = self._chains.get(chat, (set(), now))
            if now - last > settings.chain_reset_seconds:
                ids = set()
            key = str(message_id) if message_id not in (None, "") else f"anon:{now}:{len(ids)}"
            if key in ids:
                self._chains[chat] = (ids, now)
                return True, len(ids)
            if len(ids) >= settings.max_chain:
                return False, len(ids) + 1
            ids.add(key)
            self._chains[chat] = (ids, now)
            return True, len(ids)

    def _sweep(self, now: float, settings: BotLoopGuardSettings) -> None:
        """Drop idle conversations at most once per window so memory stays bounded."""
        if now - self._last_sweep < settings.window_seconds:
            return
        self._last_sweep = now
        idle_cutoff = now - settings.window_seconds
        for key in [k for k, dq in self._events.items() if not dq or dq[-1] <= idle_cutoff]:
            del self._events[key]
        for key in [k for k, until in self._cooldown_until.items() if until <= now]:
            del self._cooldown_until[key]
        for key in [k for k, (_ids, last) in self._chains.items() if now - last > settings.chain_reset_seconds]:
            del self._chains[key]
