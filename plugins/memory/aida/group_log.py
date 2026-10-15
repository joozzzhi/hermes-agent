"""group_log: the Telegram group as a shared journal in Supabase (backlog item 18).

Why: only жиши's turns reach Supabase (aida); what робжи and сёрч said, and what they were told,
lives only in their own state.db. Once sessions start rotating (idle-reset, session-per-task) that
history becomes unreachable. This module collects every group message any bot has seen from all
profiles' state.db (read-only) and upserts it into one table ``group_log``, so any bot can answer
"what did we decide about X" after its session was reset.

Source of truth for messages is each profile's state.db, so the gateway is not touched at all and
a Supabase outage loses nothing: the next sync re-reads the window and catches up.

- Incoming rows (role=user): addressed and observed messages carry ``platform_message_id`` and a
  ``[name|user_id]`` header -> (chat_id, msg_id) is the key, ``seen_by`` merges which bots saw it.
- Own replies (role=assistant) have no Telegram id. Most of them are also observed by another bot
  (with the id); those are skipped. The rest get ``msg_id = "self:<profile>:<hash>"`` once they are
  older than ``SELF_GRACE_S`` (so a late observation by another bot wins first).
- Text is scrubbed (agent.redact) before it leaves the machine.

CLI (run with the repo venv):
  python -m plugins.memory.aida.group_log sync [--days N] [--dry-run]   (N=0: full history)
  python -m plugins.memory.aida.group_log search "что решили про крон" [--limit 10] [--local]

The connection string is read from SUPABASE_DB_URL (env, else HERMES_HOME/.env).
Off switch: disable the Windows task Hermes_GroupLog (schtasks /change /tn Hermes_GroupLog /disable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

SELF_GRACE_S = 600
MSK = timezone(timedelta(hours=3))  # the operator's clock; the host runs UTC+4
ROLE_HEADER = re.compile(r"^\[([^\]\n|]+)\|(\d+)\]\n", re.S)
REPLY_QUOTE = re.compile(r'^\[Replying to: "(.*?)"\]\s*', re.S)
SYSTEM_NOTE = re.compile(r"^\[Системная заметка:[^\]]*\]\s*", re.S)
OBSERVED_HEADER = "[Observed Telegram group context - context only, not requests]"
CURRENT_HEADER = "[Current addressed message - answer only this unless it explicitly asks you to use the observed context]"
BLOCK_SPLIT = re.compile(r'\n\n(?=\[Replying to: "|\[[^\]\n|]+\|\d+\]\n)')
REASONING = re.compile(r"💭 Reasoning:\n.*?\n\n\n", re.S)
# Gateway status noise — not part of the conversation.
NOISE = re.compile(r"^(⏳ Working|💾 Self-improvement review|📬 No home channel|🧠 Память — recalled)")

DDL = """
CREATE TABLE IF NOT EXISTS group_log (
    chat_id     text        NOT NULL,
    msg_id      text        NOT NULL,
    author_id   text,
    author_name text,
    is_bot      boolean     NOT NULL DEFAULT false,
    reply_to    text,
    content     text        NOT NULL,
    ts          timestamptz NOT NULL,
    seen_by     text[]      NOT NULL DEFAULT '{}',
    search_ru   tsvector GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED,
    PRIMARY KEY (chat_id, msg_id)
);
CREATE INDEX IF NOT EXISTS group_log_ts_idx     ON group_log (chat_id, ts);
CREATE INDEX IF NOT EXISTS group_log_search_idx ON group_log USING gin (search_ru);
ALTER TABLE group_log ENABLE ROW LEVEL SECURITY;
"""

UPSERT = """
INSERT INTO group_log (chat_id, msg_id, author_id, author_name, is_bot, reply_to, content, ts, seen_by)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chat_id, msg_id) DO UPDATE SET
    seen_by = ARRAY(SELECT DISTINCT unnest(group_log.seen_by || EXCLUDED.seen_by) ORDER BY 1),
    content = EXCLUDED.content,
    reply_to = EXCLUDED.reply_to
"""


@dataclass
class Entry:
    chat_id: str
    msg_id: str
    author_id: str | None
    author_name: str | None
    is_bot: bool
    reply_to: str | None
    content: str
    ts: float
    seen_by: set[str] = field(default_factory=set)

    def row(self) -> tuple:
        return (self.chat_id, self.msg_id, self.author_id, self.author_name, self.is_bot,
                self.reply_to, self.content, datetime.fromtimestamp(self.ts, timezone.utc),
                sorted(self.seen_by))


# -- parsing -----------------------------------------------------------------------------------


def hermes_home() -> Path:
    """The ROOT Hermes dir (not a profile dir): the journal spans every profile."""
    env = os.environ.get("HERMES_HOME")
    if env:
        p = Path(env)
        return p.parent.parent if p.parent.name == "profiles" else p
    try:
        from hermes_constants import get_default_hermes_root

        return get_default_hermes_root()
    except Exception:
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hermes"


def profile_dbs(home: Path) -> dict[str, Path]:
    dbs = {"default": home / "state.db"}
    prof = home / "profiles"
    if prof.is_dir():
        for p in sorted(prof.iterdir()):
            if (p / "state.db").is_file():
                dbs[p.name] = p / "state.db"
    return {k: v for k, v in dbs.items() if v.is_file()}


def bot_ids(home: Path) -> dict[str, str]:
    """profile -> Telegram bot user id (the numeric prefix of TELEGRAM_BOT_TOKEN)."""
    out: dict[str, str] = {}
    for name, env in [("default", home / ".env")] + [
        (p.name, p / ".env") for p in sorted((home / "profiles").glob("*")) if p.is_dir()
    ]:
        tok = read_env(env).get("TELEGRAM_BOT_TOKEN", "")
        if ":" in tok and tok.split(":", 1)[0].isdigit():
            out[name] = tok.split(":", 1)[0]
    return out


def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def norm_key(text: str) -> str:
    """Formatting-insensitive key: the gateway re-renders markdown, so compare bare characters."""
    return re.sub(r"[\W_]+", "", text or "")[:200]


def strip_observed(text: str) -> str:
    """Older gateway builds stored the observed-chatter prefix inside ``content``; those
    observed messages are logged on their own rows, so keep only the addressed message."""
    if not text.startswith(OBSERVED_HEADER):
        return text
    if CURRENT_HEADER in text:
        return text.split(CURRENT_HEADER, 1)[1].lstrip("\n")
    parts = BLOCK_SPLIT.split(text)
    last = parts[-1]
    if len(parts) > 1 and parts[-2].startswith('[Replying to: "') and parts[-2].rstrip().endswith('"]'):
        last = parts[-2] + "\n\n" + last
    return last if ROLE_HEADER.match(REPLY_QUOTE.sub("", last, count=1)) else text


def parse_incoming(content: str) -> tuple[str | None, str | None, str | None, str]:
    """-> (author_name, author_id, reply_to, text) of an inbound group row."""
    text = strip_observed(content or "")
    reply_to = None
    m = REPLY_QUOTE.match(text)
    if m:
        reply_to, text = m.group(1), text[m.end():]
    text = SYSTEM_NOTE.sub("", text, count=1)
    m = ROLE_HEADER.match(text)
    if not m:
        return None, None, reply_to, text.strip()
    return m.group(1).strip(), m.group(2), reply_to, text[m.end():].strip()


def clean_reply(content: str) -> str:
    return REASONING.sub("", content or "").strip()


def _scrub(text: str) -> str:
    try:
        from agent.redact import redact_sensitive_text
    except Exception:  # standalone use outside the repo: never ship unscrubbed — drop secrets crudely
        return re.sub(r"(?i)(postgres(ql)?://\S+|sk-[\w-]{16,}|\d{6,}:[\w-]{30,})", "[REDACTED]", text)
    return redact_sensitive_text(text, force=True, redact_url_credentials=True)


def collect(home: Path, *, since: float = 0.0, now: float | None = None,
            chat_types: Iterable[str] = ("group",)) -> list[Entry]:
    now = time.time() if now is None else now
    bots = bot_ids(home)
    entries: dict[tuple[str, str], Entry] = {}
    by_text: dict[tuple[str, str, str], Entry] = {}  # (chat, author_id, norm text) -> entry
    replies: list[tuple[str, str, str, float]] = []
    types = tuple(chat_types)
    q_marks = ",".join("?" * len(types))
    for profile, db in profile_dbs(home).items():
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
        try:
            rows = con.execute(
                f"""SELECT s.chat_id, m.role, m.content, m.platform_message_id, m.timestamp
                    FROM messages m JOIN sessions s ON s.id = m.session_id
                    WHERE s.source = 'telegram' AND s.chat_type IN ({q_marks})
                      AND m.role IN ('user', 'assistant') AND COALESCE(m.content, '') <> ''
                      AND m.timestamp >= ?
                    ORDER BY m.timestamp""",
                (*types, since),
            ).fetchall()
        finally:
            con.close()
        for chat_id, role, content, pmid, ts in rows:
            chat_id = str(chat_id)
            if role == "assistant":
                replies.append((profile, chat_id, content, ts))
                continue
            if not pmid:
                continue
            name, uid, reply_to, text = parse_incoming(content)
            if not text or NOISE.match(text):
                continue
            key = (chat_id, str(pmid))
            e = entries.get(key)
            if e is None:
                e = entries[key] = Entry(chat_id, str(pmid), uid, name, uid in bots.values(),
                                         reply_to, text, ts)
            elif len(text) > len(e.content):
                e.content = text
            e.ts = min(e.ts, ts)
            e.seen_by.add(profile)
            if uid:
                by_text[(chat_id, uid, norm_key(text))] = e
    for profile, chat_id, content, ts in replies:
        text = clean_reply(content)
        if not text or NOISE.match(text):
            continue
        uid = bots.get(profile)
        known = by_text.get((chat_id, uid or "", norm_key(text)))
        if known is not None:
            known.seen_by.add(profile)
            continue
        if now - ts < SELF_GRACE_S:
            continue  # another bot may still observe it with the real Telegram id
        mid = f"self:{profile}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"
        key = (chat_id, mid)
        if key not in entries:
            entries[key] = Entry(chat_id, mid, uid, profile, True, None, text, ts, {profile})
    out = sorted(entries.values(), key=lambda e: e.ts)
    names = {e.author_id: e.author_name for e in out if e.author_id and not e.msg_id.startswith("self:")}
    for e in out:
        if e.msg_id.startswith("self:") and e.author_id in names:
            e.author_name = names[e.author_id]  # Telegram display name instead of profile name
    for e in out:
        e.content = _scrub(e.content)[:20000]
        if e.reply_to:
            e.reply_to = _scrub(e.reply_to)[:500]
    return out


# -- Supabase ----------------------------------------------------------------------------------


def db_url(home: Path) -> str:
    return os.environ.get("SUPABASE_DB_URL") or read_env(home / ".env").get("SUPABASE_DB_URL", "")


def connect(url: str):
    import psycopg

    return psycopg.connect(url, connect_timeout=10, autocommit=True, prepare_threshold=None,
                           options="-c statement_timeout=30000")


def push(entries: list[Entry], url: str) -> int:
    with connect(url) as con:
        con.execute(DDL)
        with con.cursor() as cur:
            cur.executemany(UPSERT, [e.row() for e in entries])
    return len(entries)


def search_remote(query: str, url: str, limit: int = 10) -> list[tuple]:
    sql = """
    SELECT ts, author_name, content,
           ts_rank(search_ru, websearch_to_tsquery('russian', %(q)s)) AS r
    FROM group_log
    WHERE search_ru @@ websearch_to_tsquery('russian', %(q)s)
       OR search_ru @@ to_tsquery('russian', %(any)s)
    ORDER BY (search_ru @@ websearch_to_tsquery('russian', %(q)s)) DESC, r DESC, ts DESC
    LIMIT %(n)s
    """
    words = [w for w in re.findall(r"\w{3,}", query.lower())]
    anyq = " | ".join(words) or "x"
    with connect(url) as con:
        return [(r[0].timestamp(), r[1], r[2]) for r in
                con.execute(sql, {"q": query, "any": anyq, "n": limit}).fetchall()]


def search_local(query: str, home: Path, limit: int = 10) -> list[tuple]:
    words = [w.lower() for w in re.findall(r"\w{3,}", query)]
    stems = [w[:max(3, len(w) - 2)] for w in words]
    scored = []
    for e in collect(home, now=time.time() + SELF_GRACE_S):
        low = e.content.lower()
        hits = sum(1 for s in stems if s in low)
        if hits:
            scored.append((hits, e.ts, e))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [(e.ts, e.author_name, e.content) for _, _, e in scored[:limit]]


def _fmt(rows: list[tuple], width: int) -> str:
    out = []
    for ts, who, text in rows:
        when = datetime.fromtimestamp(ts, MSK).strftime("%d.%m %H:%M")
        body = text if len(text) <= width else text[:width] + "…"
        out.append(f"[{when}] {who}: {body}")
    return "\n\n".join(out) or "ничего не найдено"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="group_log")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--days", type=float, default=3.0, help="window to re-read; 0 = full history")
    s.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("search")
    f.add_argument("query")
    f.add_argument("--limit", type=int, default=10)
    f.add_argument("--width", type=int, default=600)
    f.add_argument("--local", action="store_true", help="search state.db directly (no Supabase)")
    a = ap.parse_args(argv)
    home = hermes_home()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "sync":
        since = 0.0 if a.days <= 0 else time.time() - a.days * 86400
        entries = collect(home, since=since)
        if a.dry_run:
            print(json.dumps({"entries": len(entries),
                              "self": sum(e.msg_id.startswith("self:") for e in entries)}))
            return 0
        url = db_url(home)
        if not url:
            print("SUPABASE_DB_URL not set", file=sys.stderr)
            return 2
        n = push(entries, url)
        print(f"group_log: upserted {n}")
        return 0
    url = "" if a.local else db_url(home)
    rows = None
    if url:
        try:
            rows = search_remote(a.query, url, a.limit)
        except Exception as exc:  # Supabase down -> local scan, same answer shape
            print(f"(Supabase недоступен: {type(exc).__name__}; ищу локально)", file=sys.stderr)
    if rows is None:
        rows = search_local(a.query, home, a.limit)
    print(_fmt(rows, a.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
