"""group_log: collecting the Telegram group journal from every profile's state.db."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from plugins.memory.aida import group_log as gl

G = "-100500"
NOW = 1_800_000_000.0


def _db(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        """CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, chat_id TEXT, chat_type TEXT);
           CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
               content TEXT, platform_message_id TEXT, timestamp REAL);"""
    )
    con.execute("INSERT INTO sessions VALUES ('s1','telegram',?,'group')", (G,))
    con.execute("INSERT INTO sessions VALUES ('d1','telegram','42','dm')")
    con.executemany(
        "INSERT INTO messages (session_id, role, content, platform_message_id, timestamp) VALUES (?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes"
    (home / "profiles" / "coder").mkdir(parents=True)
    (home / ".env").write_text("TELEGRAM_BOT_TOKEN=111:" + "a" * 35 + "\n", encoding="utf-8")
    (home / "profiles" / "coder" / ".env").write_text("TELEGRAM_BOT_TOKEN=222:" + "b" * 35 + "\n",
                                                       encoding="utf-8")
    # default (жиши, bot 111) saw the operator's message and replied; coder observed both.
    _db(home / "state.db", [
        ("s1", "user", "[Karma|900]\nрешили: крон раз в 3 часа", "10", NOW - 3600),
        ("s1", "assistant", "💭 Reasoning:\nдумаю\n\n\nОк, **крон** раз в 3 часа", None, NOW - 3500),
        ("d1", "user", "[Karma|900]\nличное", "5", NOW - 3000),
    ])
    _db(home / "profiles" / "coder" / "state.db", [
        ("s1", "user", "[Karma|900]\nрешили: крон раз в 3 часа", "10", NOW - 3599),
        ("s1", "user", "[Observed Telegram group context - context only, not requests]\n"
                       "[Wed 2026-09-23 13:49:02 MSK] [Joe|700]\nболтовня\n\n[Joe|700]\nещё\n\n"
                       '[Replying to: "q"]\n\n[Karma|900]\nстарый формат', "9", NOW - 5000),
        ("s1", "user", "[jishii|111]\nОк, крон раз в 3 часа", "11", NOW - 3499),
        ("s1", "user", '[Replying to: "старое"]\n\n[Системная заметка: x]\n\n[Karma|900]\nробжи, сделай', "12", NOW - 900),
        ("s1", "user", "[jishii|111]\n🧠 Память — recalled 7 memories", "13", NOW - 800),
        ("s1", "assistant", "Сделал, ключ sk-" + "Z" * 40, None, NOW - 700),
        ("s1", "assistant", "свежий ответ", None, NOW - 60),
    ])
    return home


def test_collect_dedupes_merges_and_skips(tmp_path):
    home = _home(tmp_path)
    entries = gl.collect(home, now=NOW)
    by_id = {e.msg_id: e for e in entries}

    # one row per Telegram message, seen_by merged across profiles
    assert by_id["10"].seen_by == {"default", "coder"}
    assert by_id["10"].author_name == "Karma" and not by_id["10"].is_bot
    # жиши's own reply is matched to coder's observation (id 11) despite markdown/reasoning
    assert by_id["11"].seen_by == {"default", "coder"} and by_id["11"].is_bot
    # reply quote and system note are stripped, quote kept as reply_to
    assert by_id["12"].content == "робжи, сделай" and by_id["12"].reply_to == "старое"
    # old-format row with observed chatter baked into content keeps only the addressed message
    assert by_id["9"].content == "старый формат" and by_id["9"].reply_to == "q"
    assert by_id["9"].author_id == "900"
    # gateway noise and DMs never land in the log
    assert "13" not in by_id and "5" not in by_id
    # unobserved own reply older than the grace period -> self row, secret scrubbed
    selfs = [e for e in entries if e.msg_id.startswith("self:coder:")]
    assert len(selfs) == 1 and "Z" * 40 not in selfs[0].content
    # reply younger than the grace period waits for the next sync
    assert not any("свежий" in e.content for e in entries)


def test_collect_since_window(tmp_path):
    home = _home(tmp_path)
    ids = {e.msg_id for e in gl.collect(home, since=NOW - 1000, now=NOW)}
    assert "10" not in ids and "12" in ids


def test_search_local_finds_decision(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.setattr(gl.time, "time", lambda: NOW)
    rows = gl.search_local("что решили про крон", home, limit=3)
    assert rows and "крон раз в 3 часа" in rows[0][2]


def test_main_dry_run(tmp_path, monkeypatch, capsys):
    home = _home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert gl.main(["sync", "--days", "0", "--dry-run"]) == 0
    assert '"entries"' in capsys.readouterr().out


def test_hermes_home_is_root_for_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes" / "profiles" / "coder"))
    assert gl.hermes_home() == tmp_path / "hermes"


def test_plugin_tool_falls_back_to_local(tmp_path, monkeypatch):
    import importlib
    import json

    home = _home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "coder"))
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    monkeypatch.setattr(gl.time, "time", lambda: NOW)
    plugin = importlib.import_module("plugins.group-log")
    out = json.loads(plugin._handle({"query": "решили крон"}))
    assert out["source"] == "local" and out["count"] >= 1
    assert "крон раз в 3 часа" in out["messages"]
