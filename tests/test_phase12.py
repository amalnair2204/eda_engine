"""Tests for Phase 12 — deployment helpers (env-based host/port)."""

from __future__ import annotations

import app as app_module
from app import _host_port


def test_host_port_defaults(monkeypatch) -> None:
    """With no env set, _host_port returns the container defaults."""
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert _host_port() == ("0.0.0.0", 8000)


def test_host_port_reads_env(monkeypatch) -> None:
    """HOST and PORT are read from the environment when present."""
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9001")
    assert _host_port() == ("127.0.0.1", 9001)


def test_host_port_bad_port_falls_back(monkeypatch) -> None:
    """A non-integer PORT falls back to 8000 instead of crashing."""
    monkeypatch.setenv("PORT", "not-a-number")
    host, port = _host_port()
    assert port == 8000
