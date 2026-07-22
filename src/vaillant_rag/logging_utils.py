"""Logging configuration: file + console handlers."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Settings

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(settings: Settings) -> None:
    """Configure root logging to both a rotating file and the console.

    File gets the configured level; console shows warnings and errors only
    (normal output goes through ``print`` in the CLI).
    """
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    file_handler = logging.FileHandler(log_dir / settings.log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logging.basicConfig(level=level, handlers=[file_handler, console_handler], force=True)
