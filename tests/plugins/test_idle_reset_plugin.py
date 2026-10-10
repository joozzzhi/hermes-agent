"""idle-reset plugin: a long silence rotates the session, a short one never does."""

from __future__ import annotations

import importlib.util
import sqlite3
import types
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "idle-reset" / "__init__.py"
_spec = importlib.util.spec_from_file_location("idle_reset_plugin", _PLUGIN)
ir = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ir)

NOW = 1_800_000_000.0
KEY = "agent:coder:telegram:group:-100"


def _db(tmp_path, rows):
    p = tmp_path / "state.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE messages (session_id TEXT, timestamp REAL, observed INTEGER)")
    con.executemany("INSERT INTO messages VALUES (?,?,?)", rows)
    con.commit()
    con.close()
    return p


class Store:
    def __init__(self, sid="old"):
        self.sid = sid
        self.resets = 0

    def peek_session_id(self, key):
        return self.sid

    def reset_session(self, key):
        self.resets += 1
        self.sid = "new"
        return types.SimpleNamespace(session_id="new")


class Gateway:
    def __init__(self, running=False, authorized=True):
        self.running, self.authorized = running, authorized
        self.evicted, self.cleared = [], []

    def _is_user_authorized_for_source(self, source):
        return self.authorized

    def _session_key_for_source(self, source):
        return KEY

    def _is_session_running(self, key):
        return self.running

    def _evict_cached_agent(self, key):
        self.evicted.append(key)

    def _clear_conversation_scope(self, key, *, reason):
        self.cleared.append(reason)


def _event(text="привет", chat_type="group", internal=False):
    src = types.SimpleNamespace(chat_type=chat_type, platform=types.SimpleNamespace(value="telegram"))
    return types.SimpleNamespace(text=text, source=src, internal=internal)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("IDLE_RESET_MINUTES", raising=False)
    monkeypatch.delenv("IDLE_RESET_CHAT_TYPES", raising=False)
    monkeypatch.setattr(ir, "_busy", lambda gw, key: gw.running)


def test_an_hour_of_silence_starts_a_new_session_and_says_where_the_old_one_is(tmp_path):
    db = _db(tmp_path, [("old", NOW - 61 * 60, 0)])
    store, gw = Store(), Gateway()
    out = ir.maybe_reset(_event("сделай X"), gw, store, now=NOW, db_path=db)
    assert store.resets == 1
    assert gw.evicted == [KEY] and gw.cleared == ["idle_reset"]
    assert out["action"] == "rewrite"
    assert "old" in out["text"] and out["text"].endswith("сделай X")


def test_a_short_pause_keeps_the_session(tmp_path):
    db = _db(tmp_path, [("old", NOW - 59 * 60, 0)])
    store = Store()
    assert ir.maybe_reset(_event(), Gateway(), store, now=NOW, db_path=db) is None
    assert store.resets == 0


def test_group_chatter_the_bot_only_observed_does_not_count_as_activity(tmp_path):
    db = _db(tmp_path, [("old", NOW - 3 * 3600, 0), ("old", NOW - 60, 1)])
    store = Store()
    assert ir.maybe_reset(_event(), Gateway(), store, now=NOW, db_path=db) is not None
    assert store.resets == 1


def test_a_running_turn_is_never_cut(tmp_path):
    db = _db(tmp_path, [("old", NOW - 3 * 3600, 0)])
    store = Store()
    assert ir.maybe_reset(_event(), Gateway(running=True), store, now=NOW, db_path=db) is None
    assert store.resets == 0


@pytest.mark.parametrize("event", [_event("/new"), _event(internal=True), _event(chat_type="channel")])
def test_commands_internal_events_and_other_chat_types_are_left_alone(tmp_path, event):
    db = _db(tmp_path, [("old", NOW - 3 * 3600, 0)])
    store = Store()
    assert ir.maybe_reset(event, Gateway(), store, now=NOW, db_path=db) is None
    assert store.resets == 0


def test_an_unauthorized_sender_cannot_rotate_the_session(tmp_path):
    db = _db(tmp_path, [("old", NOW - 3 * 3600, 0)])
    store = Store()
    assert ir.maybe_reset(_event(), Gateway(authorized=False), store, now=NOW, db_path=db) is None
    assert store.resets == 0


def test_no_recorded_activity_means_no_reset(tmp_path):
    db = _db(tmp_path, [("other", NOW - 3 * 3600, 0)])
    store = Store()
    assert ir.maybe_reset(_event(), Gateway(), store, now=NOW, db_path=db) is None


def test_threshold_comes_from_env_and_zero_disables(tmp_path, monkeypatch):
    db = _db(tmp_path, [("old", NOW - 20 * 60, 0)])
    monkeypatch.setenv("IDLE_RESET_MINUTES", "15")
    assert ir.maybe_reset(_event(), Gateway(), Store(), now=NOW, db_path=db) is not None
    monkeypatch.setenv("IDLE_RESET_MINUTES", "0")
    assert ir.maybe_reset(_event(), Gateway(), Store(), now=NOW, db_path=db) is None


def test_hook_swallows_errors():
    class Boom(Gateway):
        def _session_key_for_source(self, source):
            raise RuntimeError("x")
    assert ir._on_pre_gateway_dispatch(event=_event(), gateway=Boom(), session_store=Store()) is None
