"""The detached /restart watcher must relaunch the LAUNCH gateway, not the turn's routed profile."""
from __future__ import annotations

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_restart_watcher_env_pins_process_home_under_profile_override(tmp_path, monkeypatch):
    from gateway.run_shutdown import GatewayShutdownMixin

    launch = tmp_path / "hermes"
    served = launch / "profiles" / "coder"
    served.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))

    token = set_hermes_home_override(served)
    try:
        env = GatewayShutdownMixin._restart_watcher_env()
    finally:
        reset_hermes_home_override(token)

    assert env["HERMES_HOME"] == str(launch)
    assert "_HERMES_GATEWAY" not in env


def test_restart_watcher_env_without_override_keeps_launch_home(tmp_path, monkeypatch):
    from gateway.run_shutdown import GatewayShutdownMixin

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    env = GatewayShutdownMixin._restart_watcher_env()
    assert env["HERMES_HOME"] == str(tmp_path)
    assert "_HERMES_GATEWAY" not in env
