"""idle-reset: start a fresh session after a long silence (вариант «б», решение Karma 10.10).

Why: Hermes no longer resets sessions on idle (session_reset / idle_minutes are ignored), so a
group session lives for days, piles up and gets compressed. This hook runs on every addressed
inbound message (pre_gateway_dispatch, under the serving profile's scope). If the current
session's last real activity — the newest non-observed row in this profile's state.db, i.e. the
last addressed message or my last reply/tool step — is older than the threshold, it rotates the
routing key to a new session exactly like the gateway's compression-exhausted reset does
(reset_session + evict cached agent + clear conversation scope) and prefixes the message with a
short note pointing at the previous session id for session_search.

Never resets: internal events, slash commands, unauthorized senders, a session with a running
turn or live async delegations, a session with no recorded activity.

Config (env, profile .env): IDLE_RESET_MINUTES (default 60; 0 disables),
IDLE_RESET_CHAT_TYPES (default "group,dm").
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MINUTES = 60


def _minutes() -> float:
    try:
        return float(os.environ.get("IDLE_RESET_MINUTES", DEFAULT_MINUTES))
    except ValueError:
        return DEFAULT_MINUTES


def _chat_types() -> set[str]:
    raw = os.environ.get("IDLE_RESET_CHAT_TYPES", "group,dm")
    return {p.strip() for p in raw.split(",") if p.strip()}


def _state_db(home: Optional[Path] = None) -> Path:
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    return Path(home) / "state.db"


def last_activity(session_id: str, db_path: Path) -> Optional[float]:
    """Epoch of the newest non-observed message of the session, None if unknown."""
    if not session_id or not db_path.exists():
        return None
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=2)
    try:
        row = con.execute(
            "SELECT MAX(timestamp) FROM messages WHERE session_id = ? AND COALESCE(observed, 0) = 0",
            (session_id,),
        ).fetchone()
    finally:
        con.close()
    return float(row[0]) if row and row[0] is not None else None


def _note(old_sid: str, idle_min: int) -> str:
    return (
        f"[Системная заметка: прошлая сессия ({old_sid}) закрыта после {idle_min} мин тишины, "
        "это новая сессия. Если нужно что-то из прошлой работы — session_search / "
        "chat_history_lookup по этому id.]"
    )


def _busy(gateway: Any, session_key: str) -> bool:
    try:
        if gateway._is_session_running(session_key):
            return True
    except Exception:
        pass
    try:
        from tools.async_delegation import has_live_for_session
        if has_live_for_session(session_key=session_key):
            return True
    except Exception:
        pass
    return False


def maybe_reset(event: Any, gateway: Any, session_store: Any, *, now: Optional[float] = None,
                db_path: Optional[Path] = None) -> Optional[dict]:
    threshold = _minutes()
    if threshold <= 0 or gateway is None or session_store is None:
        return None
    if getattr(event, "internal", False):
        return None
    source = getattr(event, "source", None)
    if source is None or getattr(source, "chat_type", "") not in _chat_types():
        return None
    text = getattr(event, "text", "") or ""
    if text.lstrip().startswith("/"):
        return None  # /new, /resume, /stop … own their session semantics
    try:
        if not gateway._is_user_authorized_for_source(source):
            return None
    except Exception:
        return None

    session_key = gateway._session_key_for_source(source)
    old_sid = session_store.peek_session_id(session_key)
    if not old_sid:
        return None
    last = last_activity(old_sid, db_path or _state_db())
    if last is None:
        return None
    idle = (now if now is not None else time.time()) - last
    if idle < threshold * 60:
        return None
    if _busy(gateway, session_key):
        return None

    new_entry = session_store.reset_session(session_key)
    if new_entry is None:
        return None
    for fn, kw in (("_evict_cached_agent", {}), ("_clear_conversation_scope", {"reason": "idle_reset"})):
        try:
            getattr(gateway, fn)(session_key, **kw)
        except Exception:
            logger.debug("idle-reset: %s failed", fn, exc_info=True)
    try:
        from hermes_cli.lifecycle import invoke_hook
        invoke_hook("on_session_reset", session_id=new_entry.session_id, reason="idle_reset",
                    platform=source.platform.value if source.platform else "",
                    old_session_id=old_sid, new_session_id=new_entry.session_id)
    except Exception:
        pass
    idle_min = int(idle // 60)
    logger.info("idle-reset: %s idle %d min -> new session %s (was %s)",
                session_key, idle_min, new_entry.session_id, old_sid)
    return {"action": "rewrite", "text": f"{_note(old_sid, idle_min)}\n\n{text}"}


def _on_pre_gateway_dispatch(event=None, gateway=None, session_store=None, **_kw):
    try:
        return maybe_reset(event, gateway, session_store)
    except Exception:
        logger.warning("idle-reset failed", exc_info=True)
        return None


def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
