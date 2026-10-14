"""Every test inherits protection from network and live provider access."""

from __future__ import annotations

import _socket
import asyncio
import socket
import subprocess
from functools import partial

import pytest


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_unmarked_tests_cannot_start_live_provider_executables(provider):
    with pytest.raises(AssertionError, match="must not start live"):
        subprocess.run([provider, "--version"], check=True)


def test_unmarked_tests_cannot_connect_to_the_network():
    with pytest.raises(AssertionError, match="must not connect to the network"):
        socket.create_connection(("127.0.0.1", 1))
    with socket.socket() as connection, pytest.raises(AssertionError, match="must not connect"):
        connection.connect(("127.0.0.1", 1))
    with socket.socket() as connection, pytest.raises(AssertionError, match="must not connect"):
        connection.connect_ex(("127.0.0.1", 1))


def test_asyncio_can_use_the_windows_socketpair_fallback(monkeypatch):
    # Exercise the stdlib fallback even on systems with a native socketpair.
    monkeypatch.setattr(_socket, "socketpair", vars(socket)["_fallback_socketpair"], raising=False)
    monkeypatch.setattr(socket, "socketpair", partial(socket.socketpair, family=socket.AF_INET))

    async def ready():
        await asyncio.sleep(0)
        return "ready"

    assert asyncio.run(ready()) == "ready"
    with socket.socket() as connection, pytest.raises(AssertionError, match="must not connect"):
        connection.connect(("127.0.0.1", 1))


def test_socketpair_failure_does_not_enable_network_connections():
    with pytest.raises((ValueError, OSError)):
        socket.socketpair(family=-1)
    with socket.socket() as connection, pytest.raises(AssertionError, match="must not connect"):
        connection.connect(("127.0.0.1", 1))
