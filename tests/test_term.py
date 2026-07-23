"""Terminal styling: everything must degrade to plain text off-TTY.

Under pytest stdout is not a TTY, so these tests exercise the
no-color/no-animation paths — the ones piped output relies on.
"""

from __future__ import annotations

import vaillant_rag.term as term
from vaillant_rag.term import PigeonFlight, accent, dim, error, ok


def _force_no_color(monkeypatch):
    monkeypatch.setattr(term, "_color_enabled", None)
    monkeypatch.setenv("NO_COLOR", "1")


def test_styles_are_plain_without_tty(monkeypatch):
    _force_no_color(monkeypatch)
    assert accent("x") == "x"
    assert ok("x") == "x"
    assert dim("x") == "x"
    assert error("x") == "x"


def test_pigeon_flight_is_noop_without_tty(monkeypatch, capsys):
    _force_no_color(monkeypatch)
    flight = PigeonFlight()
    flight.start()
    flight.stop()  # must be safe even though start() was a no-op
    assert capsys.readouterr().out == ""


def test_supports_color_caches_result(monkeypatch):
    _force_no_color(monkeypatch)
    assert term.supports_color() is False
    # Cached: flipping the environment afterwards must not change the answer.
    monkeypatch.delenv("NO_COLOR")
    assert term.supports_color() is False
