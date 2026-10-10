from __future__ import annotations

import os
import time

import pytest

from plugins.memory.aida import idle


def test_only_one_consolidation_holds_the_lock_at_a_time(tmp_path):
    lock = tmp_path / idle.LOCK_NAME

    assert idle._acquire(lock) is True
    assert idle._acquire(lock) is False


def test_a_lock_left_by_a_process_that_died_mid_run_is_reclaimed(tmp_path):
    lock = tmp_path / idle.LOCK_NAME
    lock.write_text("12345")
    old = time.time() - idle.STALE_LOCK_SECONDS - 60
    os.utime(lock, (old, old))

    assert idle._acquire(lock) is True
    assert lock.read_text() == str(os.getpid())


def test_a_fresh_lock_held_by_someone_else_is_respected(tmp_path):
    lock = tmp_path / idle.LOCK_NAME
    lock.write_text("12345")
    recent = time.time() - idle.STALE_LOCK_SECONDS + 60
    os.utime(lock, (recent, recent))

    assert idle._acquire(lock) is False
    assert lock.read_text() == "12345"


def test_a_run_skipped_for_the_lock_never_consolidates(tmp_path, monkeypatch):
    from plugins.memory.aida import nightly_job

    called = []
    monkeypatch.setattr(nightly_job, "consolidate", lambda home: called.append(home) or [])
    (tmp_path / idle.LOCK_NAME).write_text("12345")

    assert idle.run_now(tmp_path) is False
    assert called == []


def test_a_finished_run_lets_go_of_the_lock(tmp_path, monkeypatch):
    from plugins.memory.aida import nightly_job

    monkeypatch.setattr(nightly_job, "consolidate", lambda home: ["ok"])

    assert idle.run_now(tmp_path) is True
    assert not (tmp_path / idle.LOCK_NAME).exists()


def test_consolidation_is_skipped_while_another_run_holds_the_lock(tmp_path, monkeypatch):
    from plugins.memory.aida import nightly_job

    called = []
    monkeypatch.setattr(nightly_job, "consolidate", lambda home: called.append(home) or ["ok"])
    (tmp_path / idle.LOCK_NAME).write_text("12345")

    assert idle.consolidate_locked(tmp_path) is None
    assert called == []
    assert (tmp_path / idle.LOCK_NAME).read_text() == "12345"  # someone else's lock is left alone


def test_consolidation_returns_the_report_and_releases_the_lock(tmp_path, monkeypatch):
    from plugins.memory.aida import nightly_job

    monkeypatch.setattr(nightly_job, "consolidate", lambda home: ["Разбор общей памяти"])

    assert idle.consolidate_locked(tmp_path) == ["Разбор общей памяти"]
    assert not (tmp_path / idle.LOCK_NAME).exists()
    assert idle.consolidate_locked(tmp_path) == ["Разбор общей памяти"]  # the next run is not blocked


def test_a_run_that_fails_midway_still_releases_the_lock(tmp_path, monkeypatch):
    from plugins.memory.aida import nightly_job

    def boom(home):
        raise RuntimeError("база упала посреди разбора")

    monkeypatch.setattr(nightly_job, "consolidate", boom)

    assert idle.run_now(tmp_path) is False
    assert not (tmp_path / idle.LOCK_NAME).exists()


def test_a_manual_run_says_so_and_fails_while_a_run_is_in_progress(tmp_path, monkeypatch, capsys):
    from plugins.memory.aida import nightly_job

    monkeypatch.setattr(nightly_job, "consolidate", lambda home: pytest.fail("must not run twice"))
    (tmp_path / idle.LOCK_NAME).write_text("12345")

    assert nightly_job.run(tmp_path) == 1
    assert "пропущен" in capsys.readouterr().out


def test_the_silence_threshold_follows_the_setting_but_never_drops_below_a_minute(monkeypatch):
    monkeypatch.setenv("AIDA_IDLE_MINUTES", "30")
    assert idle.idle_seconds() == 1800.0

    monkeypatch.setenv("AIDA_IDLE_MINUTES", "0.1")
    assert idle.idle_seconds() == 60.0

    monkeypatch.setenv("AIDA_IDLE_MINUTES", "soon")
    assert idle.idle_seconds() == idle.DEFAULT_IDLE_MINUTES * 60.0
