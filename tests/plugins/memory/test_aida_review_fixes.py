"""Правки по разбору патча 10.10: кеш, бюджет, ключи профиля, один эмбеддинг на ход, учёт показанного."""

from __future__ import annotations

import os
import threading

import pytest

import plugins.context_engine.aida as engine_module
from plugins.memory.aida import AidaMemoryProvider, _Store, nightly_job
from plugins.memory.aida.search import facts_that_fit, format_recall


# -- 1. жизнь кеша берётся у хозяина ---------------------------------------------------------

@pytest.mark.parametrize("configured, seconds", [("1h", 3600.0), ("5m", 300.0), (None, 300.0)])
def test_the_engine_believes_the_cache_lives_as_long_as_the_host_configured(monkeypatch, configured, seconds):
    import agent.agent_runtime_helpers as helpers

    monkeypatch.delenv("AIDA_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.setattr(helpers, "configured_cache_ttl", lambda: configured)
    assert engine_module.cache_ttl_seconds() == seconds


def test_an_explicit_cache_lifetime_wins_over_the_host(monkeypatch):
    import agent.agent_runtime_helpers as helpers

    monkeypatch.setattr(helpers, "configured_cache_ttl", lambda: "1h")
    monkeypatch.setenv("AIDA_CACHE_TTL_SECONDS", "120")
    assert engine_module.cache_ttl_seconds() == 120.0


# -- 2. нижняя граница бюджета ---------------------------------------------------------------

def test_a_small_window_still_gets_the_minimum_budget(monkeypatch):
    monkeypatch.delenv("AIDA_CONTEXT_TOKENS", raising=False)
    monkeypatch.delenv("AIDA_CONTEXT_SHARE", raising=False)
    engine = AidaEngine()
    small = engine._budget_chars(200_000)  # 5% от 200k = 10k — меньше минимума
    assert small == int(engine_module.MIN_CONTEXT_TOKENS * engine._chars_per_token)
    assert engine_module.MIN_CONTEXT_TOKENS >= 60_000


def AidaEngine():
    return engine_module.AidaContextEngine()


# -- 3. ключи разбора — из .env своего профиля -------------------------------------------------

def _home(tmp_path, name, lines):
    home = tmp_path / name
    home.mkdir()
    (home / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return home


def test_each_profile_s_run_sees_its_own_keys_and_the_process_env_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "launch-profile-key")
    a = _home(tmp_path, "a", ["SUPABASE_DB_URL=postgres://a", "OPENROUTER_API_KEY=key-a"])
    b = _home(tmp_path, "b", ["SUPABASE_DB_URL=postgres://b", 'OPENROUTER_API_KEY="key-b"'])

    seen = []

    def fake(home):
        seen.append((nightly_job._secret("SUPABASE_DB_URL"), nightly_job._secret("OPENROUTER_API_KEY")))
        return []

    monkeypatch.setattr(nightly_job, "_consolidate", fake)
    nightly_job.consolidate(a)
    nightly_job.consolidate(b)
    nightly_job.consolidate(a)

    assert seen == [("postgres://a", "key-a"), ("postgres://b", "key-b"), ("postgres://a", "key-a")]
    assert os.environ["OPENROUTER_API_KEY"] == "launch-profile-key"


def test_under_multiplex_a_key_missing_from_the_profile_is_not_borrowed_from_the_process(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "someone-elses-key")
    monkeypatch.setattr(nightly_job, "_multiplex_active", lambda: True)
    home = _home(tmp_path, "p", ["SUPABASE_DB_URL=postgres://p"])
    assert "OPENROUTER_API_KEY" not in nightly_job.profile_secrets(home)


def test_a_single_profile_process_may_still_take_keys_from_its_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-service-env")
    monkeypatch.setattr(nightly_job, "_multiplex_active", lambda: False)
    home = _home(tmp_path, "p", ["SUPABASE_DB_URL=postgres://p"])
    assert nightly_job.profile_secrets(home)["OPENROUTER_API_KEY"] == "from-service-env"


# -- 4. один эмбеддинг на ход -----------------------------------------------------------------

class _CountingStore(_Store):
    """Настоящий _Store без базы: считает обращения к сервису эмбеддингов."""

    def __init__(self):
        super().__init__("postgres://unused", "key")
        self.embed_requests: list[list[str]] = []
        self.vectors_written: dict[int, list[float]] = {}
        self._next = 0

    def _embed_all(self, texts, timeout=None):
        self.embed_requests.append(list(texts))
        return [[float(len(t))] for t in texts]

    def _execute(self, sql, params):
        if "INSERT" in sql.upper():
            self._next += 1
            return [(self._next,)]
        if "embedding" in sql and "vec" in params:
            self.vectors_written[params["id"]] = params["vec"]
        return []

    def _search(self, kind, query, vector, limit):
        return []


def _run_embed_threads():
    for thread in threading.enumerate():
        if thread.name == "aida-embed":
            thread.join(5)


def test_the_question_is_embedded_once_and_its_vector_reused_for_the_stored_line():
    store = _CountingStore()
    question = "какие машины везём из Китая в декабре"
    store.recall_tiers(question)
    store.save(question, memory_type="dialogue", priority="P3", role="user", dedupe=False)
    store.save("Две: Zeekr и Li Auto.", memory_type="dialogue", priority="P3", role="assistant", dedupe=False)
    _run_embed_threads()

    asked = [text for request in store.embed_requests for text in request]
    assert asked.count(question) == 1
    assert len(store.vectors_written) == 2


def test_a_line_that_was_not_the_question_is_still_embedded():
    store = _CountingStore()
    store.recall_tiers("вопрос про растаможку")
    store.save("совсем другая реплика", memory_type="dialogue", priority="P3", role="user", dedupe=False)
    _run_embed_threads()
    assert ["совсем другая реплика"] in store.embed_requests


# -- 5. использованными считаются только показанные факты ---------------------------------------

def _fact(i, size=300):
    return {"id": i, "content": f"факт {i} " + "x" * size, "memory_type": "fact", "created_at": None}


def test_only_facts_that_made_it_into_the_block_count_as_used():
    facts = [_fact(i, 2000) for i in range(1, 6)]
    shown = facts_that_fit(facts)
    block, _ = format_recall(facts, [])
    assert 0 < len(shown) < len(facts)
    for item in facts:
        assert (item["content"] in block) == (item in shown)


class _TierStore:
    def __init__(self, facts):
        self.facts = facts
        self.used: list[list[int]] = []

    def recall_tiers(self, query):
        return list(self.facts), []

    def mark_used(self, ids):
        self.used.append(list(ids))


def test_prefetch_marks_used_exactly_what_it_showed():
    facts = [_fact(i, 2000) for i in range(1, 6)]
    store = _TierStore(facts)
    provider = AidaMemoryProvider()
    provider._store = store  # type: ignore[assignment]
    block = provider.prefetch("что мы решили по поездке в Китай")
    for thread in threading.enumerate():
        if thread.name == "aida-used":
            thread.join(5)
    used = store.used[0]
    assert used and all(f"факт {i} " in block for i in used)
    assert len(used) < len(facts)
