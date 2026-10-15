"""``gateway.bot_loop_guard.max_chain``: at most N bot-to-bot hand-offs in a row per chat; a human resets the chain."""

from types import SimpleNamespace

import pytest

from gateway.bot_loop_guard import BotLoopGuard, BotLoopGuardSettings, settings_from_config
from gateway.session import Platform, SessionSource

GROUP = "-1003944328658"
OTHER = "-1001111111111"
JISHII, ROBZHI, SERCH, KARMA = "1001", "1002", "1003", "8615216994"


@pytest.fixture
def clock():
    state = {"t": 1_000_000.0}
    return SimpleNamespace(now=lambda: state["t"], advance=lambda s: state.__setitem__("t", state["t"] + s))


@pytest.fixture
def settings():
    return {"value": BotLoopGuardSettings(max_chain=2, chain_reset_seconds=900)}


@pytest.fixture
def runner(clock, settings):
    from gateway.run import GatewayRunner

    r = object.__new__(GatewayRunner)
    r._bot_loop_guard = BotLoopGuard(settings=lambda: settings["value"], clock=clock.now)
    r._adapter_profile_for_source = lambda source: source.profile or ""
    return r


_seq = iter(range(1, 10_000))


def _msg(user: str, *, bot: bool, profile: str, mid: "str | None" = None, chat: str = GROUP) -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat, chat_type="group", user_id=user, is_bot=bot,
                         message_id=mid or str(next(_seq)), profile=profile)


def test_third_hand_off_in_a_row_is_dropped(runner):
    assert runner._admit_bot_message(_msg(KARMA, bot=False, profile="default"))
    # жиши answers Karma -> робжи sees hand-off 1; робжи answers -> жиши sees hand-off 2; жиши answers -> dropped.
    assert runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder", mid="11"))
    assert runner._admit_bot_message(_msg(ROBZHI, bot=True, profile="default", mid="12"))
    assert not runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder", mid="13"))
    assert not runner._admit_bot_message(_msg(SERCH, bot=True, profile="default", mid="14"))


def test_same_message_seen_by_several_profiles_counts_once(runner):
    for profile in ("coder", "serch"):
        assert runner._admit_bot_message(_msg(JISHII, bot=True, profile=profile, mid="21"))
    for profile in ("default", "serch"):
        assert runner._admit_bot_message(_msg(ROBZHI, bot=True, profile=profile, mid="22"))
    assert not runner._admit_bot_message(_msg(SERCH, bot=True, profile="coder", mid="23"))


def test_human_message_resets_the_chain(runner):
    assert runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder"))
    assert runner._admit_bot_message(_msg(ROBZHI, bot=True, profile="default"))
    assert not runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder"))
    assert runner._admit_bot_message(_msg(KARMA, bot=False, profile="coder"))
    assert runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder"))


def test_bot_silence_resets_the_chain(runner, clock):
    assert runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder"))
    assert runner._admit_bot_message(_msg(ROBZHI, bot=True, profile="default"))
    clock.advance(901)
    assert runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder"))


def test_chain_is_per_chat(runner):
    assert runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder"))
    assert runner._admit_bot_message(_msg(ROBZHI, bot=True, profile="default"))
    assert runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder", chat=OTHER))


def test_off_by_default_and_when_zero(runner, settings):
    settings["value"] = settings_from_config({"gateway": {"bot_loop_guard": {"max_events": 100}}})
    assert settings["value"].max_chain == 0
    assert all(runner._admit_bot_message(_msg(JISHII, bot=True, profile="coder")) for _ in range(10))


def test_settings_parse_max_chain():
    s = settings_from_config({"gateway": {"bot_loop_guard": {"max_chain": 2, "chain_reset_seconds": 600}}})
    assert (s.max_chain, s.chain_reset_seconds) == (2, 600)
    assert settings_from_config({"gateway": {"bot_loop_guard": {"max_chain": "x"}}}).max_chain == 0
    assert settings_from_config({"gateway": {"bot_loop_guard": {"max_chain": -1}}}).max_chain == 0
