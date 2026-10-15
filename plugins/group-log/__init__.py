"""group-log: search tool over the shared group journal (backlog item 18).

The journal itself is written by plugins/memory/aida/group_log.py (Windows task Hermes_GroupLog,
every 10 min). This plugin only reads it: Supabase full-text search when SUPABASE_DB_URL is in the
serving profile's secret scope, otherwise the same search over all profiles' state.db (read-only),
so a profile without the Supabase secret — or a Supabase outage — still gets an answer.

Off: remove ``group-log`` from ``plugins.enabled`` in the profile's config.yaml, restart.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SCHEMA = {
    "name": "group_log_search",
    "description": (
        "Search the shared journal of the Telegram group «херомесы» (every message of the operator "
        "and all bots, incl. ones from closed sessions). Use it when asked what was decided/said/"
        "agreed about something earlier, or when you need context from before this session. "
        "Returns newest-relevant messages with MSK time and author."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words to look for, e.g. 'крон бэклога часа'."},
            "limit": {"type": "integer", "description": "Max messages (default 8, max 30)."},
        },
        "required": ["query"],
    },
}


def _handle(args: dict, **_kw) -> str:
    from tools.registry import tool_error, tool_result
    from plugins.memory.aida import group_log as gl

    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query is required")
    limit = max(1, min(int(args.get("limit") or 8), 30))
    source, rows = "local", None
    try:
        from agent.secret_scope import get_secret

        url = get_secret("SUPABASE_DB_URL", "") or ""
    except Exception:
        url = ""
    if url:
        try:
            rows, source = gl.search_remote(query, url, limit), "supabase"
        except Exception as exc:
            logger.warning("group_log_search: Supabase failed (%s), falling back to local", type(exc).__name__)
    if rows is None:
        rows = gl.search_local(query, gl.hermes_home(), limit)
    return tool_result({"source": source, "count": len(rows), "messages": gl._fmt(rows, 1500)})


def register(ctx) -> None:
    ctx.register_tool(name="group_log_search", toolset="group_log", schema=SCHEMA, handler=_handle,
                      emoji="📜")
