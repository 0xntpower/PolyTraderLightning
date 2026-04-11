"""Shared logging factory for all PolySignalLab Python components.

Provides colored console output and rotating file logs.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
WHITE = "\033[0m"

LEVEL_STYLES = {
    logging.DEBUG: (GRAY, "DBG"),
    logging.INFO: (WHITE, "INF"),
    logging.WARNING: (YELLOW, "WRN"),
    logging.ERROR: (RED + BOLD, "ERR"),
    logging.CRITICAL: (RED + BOLD, "CRT"),
}


class _ColorConsoleFormatter(logging.Formatter):
    """Compact colored console format: timestamp level component message."""

    def format(self, record: logging.LogRecord) -> str:
        color, tag = LEVEL_STYLES.get(record.levelno, (WHITE, "???"))
        ts = self.formatTime(record, self.datefmt)
        name = record.name.split(".")[-1] if "." in record.name else record.name
        msg = record.getMessage()
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        exc = f"\n{record.exc_text}" if record.exc_text else ""
        return f"{DIM}{ts}{RESET} {color}{tag}{RESET} {BOLD}{name}{RESET}  {color}{msg}{RESET}{exc}"


class _PlainFileFormatter(logging.Formatter):
    """Plain text format for log files — no ANSI codes."""

    pass


def _enable_windows_ansi() -> None:
    if sys.platform == "win32":
        os.system("")  # noqa: S605, S607  # trusted log rotation command


def setup_logging(
    component_name: str,
    *,
    level: str = "INFO",
    log_dir: str | Path = "logs",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure root logger with colored console and rotating file handlers.

    Args:
        component_name: Used in the log filename (e.g. "orchestrator").
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of rotated log files to keep.

    Returns:
        A logger instance for the calling component.
    """
    _enable_windows_ansi()

    log_level = getattr(logging, level.upper(), logging.INFO)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    datefmt = "%H:%M:%S"
    file_fmt = "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s: %(message)s"
    file_datefmt = "%Y-%m-%d %H:%M:%S"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_ColorConsoleFormatter(datefmt=datefmt))

    log_file = log_path / f"{component_name}.log"
    if sys.platform == "linux":
        from logging.handlers import WatchedFileHandler

        # WatchedFileHandler detects when the file is moved/deleted and
        # reopens automatically — enables external log rotation via mv.
        # The bot seamlessly starts writing to a new file on next log emit.
        file_handler = WatchedFileHandler(
            log_file,
            encoding="utf-8",
        )
    else:
        # Windows: keep size-based rotation (no external rotation needed)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    file_handler.setFormatter(_PlainFileFormatter(file_fmt, datefmt=file_datefmt))

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    logger = logging.getLogger(component_name)
    logger.info("logging initialized — level=%s file=%s", level, log_file)
    return logger
