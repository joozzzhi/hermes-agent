"""coerce_epoch accepts ISO-8601 strings (gateway observer stamps) without a WARNING."""

import logging
from datetime import datetime, timezone

from hermes_cli.timefmt import coerce_epoch


def test_iso_utc_string_is_not_corrupt(caplog):
    caplog.set_level(logging.WARNING, logger="hermes_cli.timefmt")
    stamp = "2026-10-15T18:56:57.320323+00:00"  # live value from errors.log
    expected = datetime(2026, 10, 15, 18, 56, 57, 320323, tzinfo=timezone.utc).timestamp()
    assert coerce_epoch(stamp, field="message timestamp") == expected
    assert "corrupt" not in caplog.text


def test_iso_z_suffix_and_naive_local():
    assert coerce_epoch("2026-10-15T18:56:57Z") == datetime(2026, 10, 15, 18, 56, 57, tzinfo=timezone.utc).timestamp()
    naive = datetime(2026, 10, 15, 22, 0, 0)
    assert coerce_epoch(naive.isoformat()) == naive.timestamp()


def test_garbage_still_corrupt(caplog):
    caplog.set_level(logging.WARNING, logger="hermes_cli.timefmt")
    assert coerce_epoch("not-a-timestamp") is None
    assert coerce_epoch("9999-12-31T23:59:59+00:00") is None  # outside the trusted window
    assert caplog.text.count("Ignoring corrupt") == 2


def test_numeric_unchanged():
    assert coerce_epoch(1_760_000_000) == 1_760_000_000.0
    assert coerce_epoch("1760000000.5") == 1_760_000_000.5
    assert coerce_epoch(None) is None and coerce_epoch("") is None
