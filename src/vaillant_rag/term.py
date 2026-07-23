"""Terminal styling and the pigeon-flight loading animation.

The palette mirrors the project demo (``assets/demo.svg``) and logo:
amber accent, green for answers, slate for secondary text. Styling
degrades to plain text when stdout is not a TTY or ``NO_COLOR`` is set,
and the animation is skipped entirely, so piped output stays clean.
"""

from __future__ import annotations

import os
import sys
import threading

# Palette from assets/demo.svg / assets/logo.svg.
_ACCENT = (232, 135, 30)  # #e8871e — beak-orange accent
_OK = (63, 158, 110)  # #3f9e6e — neck-sheen green
_DIM = (107, 118, 137)  # #6b7689 — slate secondary text
_ERROR = (224, 96, 79)  # #e0604f — terminal red light

_RESET = "\x1b[0m"

_color_enabled: bool | None = None


def _enable_windows_vt() -> None:  # pragma: no cover - depends on the host console
    """Best-effort: turn on ANSI escape processing in the Windows console."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            # 0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def supports_color() -> bool:
    """True when stdout is an ANSI-capable TTY and NO_COLOR is unset."""
    global _color_enabled
    if _color_enabled is None:
        enabled = not os.environ.get("NO_COLOR") and sys.stdout.isatty()
        if enabled:
            _enable_windows_vt()
        _color_enabled = enabled
    return _color_enabled


def _fg(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"\x1b[38;2;{red};{green};{blue}m"


def _sgr(text: str, rgb: tuple[int, int, int]) -> str:
    if not supports_color():
        return text
    return f"{_fg(rgb)}{text}{_RESET}"


def accent(text: str) -> str:
    """Amber accent — highlights, the pigeon."""
    return _sgr(text, _ACCENT)


def ok(text: str) -> str:
    """Green — answer headers, success."""
    return _sgr(text, _OK)


def dim(text: str) -> str:
    """Slate — secondary information (stats, timings)."""
    return _sgr(text, _DIM)


def error(text: str) -> str:
    """Red — errors."""
    return _sgr(text, _ERROR)


def prompt_input(prompt: str) -> str:
    """``input()`` with the user's typed text echoed in the accent color."""
    if not supports_color():
        return input(prompt)
    sys.stdout.write(prompt + _fg(_ACCENT))
    sys.stdout.flush()
    try:
        return input()
    finally:
        sys.stdout.write(_RESET)
        sys.stdout.flush()


# Two wing positions, alternated every tick: tail, wing, body, beak.
_PIGEON_FRAMES = ("-\\(o>", "-/(o>")


class PigeonFlight:
    """The logo pigeon flies across the line while the answer loads.

    Runs in a daemon thread and redraws the current line; ``stop()``
    clears it. A no-op when stdout is not an ANSI TTY, so redirected
    output is never polluted. One instance per flight::

        flight = PigeonFlight()
        flight.start()
        try:
            ...  # retrieval + first token
        finally:
            flight.stop()
    """

    def __init__(self, track_width: int = 34, interval_seconds: float = 0.09) -> None:
        self._track_width = track_width
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not supports_color() or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._fly, daemon=True)
        self._thread.start()

    def _fly(self) -> None:
        tick = 0
        while not self._stop_event.wait(self._interval):
            position = tick % (self._track_width + 1)
            sprite = _PIGEON_FRAMES[tick % len(_PIGEON_FRAMES)]
            # \x1b[K erases the remainder of the previous, longer line.
            sys.stdout.write("\r" + " " * position + accent(sprite) + "\x1b[K")
            sys.stdout.flush()
            tick += 1

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None
        sys.stdout.write("\r\x1b[K")
        sys.stdout.flush()
