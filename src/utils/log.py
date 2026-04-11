"""Structured logging with ANSI color output (console) and plain file output."""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_RED = "\033[91m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_MAGENTA = "\033[95m"

LEVEL_COLORS = {
    logging.DEBUG: WHITE,
    logging.INFO: "",
    logging.WARNING: BRIGHT_YELLOW,
    logging.ERROR: BRIGHT_RED + BOLD,
    logging.CRITICAL: BRIGHT_RED + BOLD,
}

# Keywords that trigger highlight colors in log messages.
# Checked in order — first match wins, so put more specific patterns first.
HIGHLIGHTS: list[tuple[str, str]] = [
    # ── Outcomes ──
    ("[WIN]", BRIGHT_GREEN + BOLD),
    ("[LOSS]", BRIGHT_RED + BOLD),
    ("[SKIP]", WHITE),
    ("[FLAT]", BRIGHT_YELLOW),
    # ── Per-window decision summary ──
    ("WINDOW_DECISION", BRIGHT_GREEN + BOLD),
    # ── Trades ──
    ("rules_strategy rule #", BRIGHT_CYAN + BOLD),
    ("paper rule#", BRIGHT_CYAN),
    ("maker order placed:", BRIGHT_CYAN),
    ("taker order placed:", BRIGHT_MAGENTA),
    ("paper rule", BRIGHT_CYAN),
    ("maker fill:", BRIGHT_GREEN + BOLD),
    ("FIRED", BRIGHT_GREEN + BOLD),
    # ── Mechanisms (cyan family) ──
    ("KELLY", BRIGHT_CYAN),
    ("KELLY_NO_EDGE", BRIGHT_RED),
    ("REGIME", BRIGHT_CYAN),
    ("vol_stddev=", CYAN),
    ("chop_flips=", CYAN),
    ("WARMUP", BRIGHT_YELLOW),
    # ── SPRT / decay ──
    ("DECAY DETECTED", BRIGHT_RED + BOLD),
    ("SPRT", BRIGHT_MAGENTA),
    # ── Regime shifts ──
    ("regime_shift", BRIGHT_MAGENTA + BOLD),
    # ── Status heartbeat (dim) ──
    ("STATUS", DIM),
    # ── Window lifecycle ──
    ("WINDOW_SUMMARY", BRIGHT_YELLOW + BOLD),
    ("new window:", BRIGHT_CYAN + BOLD),
    ("window open price captured:", BRIGHT_CYAN),
    # ── Risk ──
    ("KILL SWITCH:", BRIGHT_RED + BOLD),
    ("blocked by risk", BRIGHT_RED),
    ("insufficient balance", BRIGHT_RED),
    # ── Results ──
    ("paper window_ts=", BRIGHT_YELLOW + BOLD),
    # ── Startup ──
    ("all feeds live:", BRIGHT_GREEN + BOLD),
    ("PREFLIGHT", BRIGHT_GREEN + BOLD),
    ("rule accepted:", BRIGHT_GREEN + BOLD),
    ("rule rejected", BRIGHT_RED + BOLD),
    ("starting polymarket bot", BRIGHT_GREEN + BOLD),
    ("paper trading mode", BRIGHT_GREEN),
    ("LIVE trading mode", BRIGHT_RED + BOLD),
]


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)

        # Keyword highlights override level color (first match wins)
        raw = record.getMessage()
        raw_lower = raw.lower()
        for keyword, color in HIGHLIGHTS:
            if keyword.lower() in raw_lower:
                return color + msg + RESET

        level_color = LEVEL_COLORS.get(record.levelno, "")
        if level_color:
            return level_color + msg + RESET

        return msg


# Custom log level for trade results
TRADE = 25
logging.addLevelName(TRADE, "TRADE")


def _enable_windows_ansi() -> None:
    if sys.platform == "win32":
        os.system("")  # noqa: S605, S607  # empty command enables Windows VT100 ANSI escape codes


def _purge_old_logs(log_dir: Path, retention_days: int) -> None:
    """Delete log files older than retention_days based on the timestamp in the filename."""
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = 0
    for f in log_dir.glob("*.log"):
        # Expected filename format: YYYY-MM-DD_HH-MM-SS.log
        try:
            ts = datetime.strptime(f.stem, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=UTC)
        except ValueError:
            continue  # skip files not matching our naming scheme
        if ts < cutoff:
            try:
                f.unlink()
                deleted += 1
            except OSError as exc:
                logging.getLogger(__name__).debug("failed to purge log %s: %s", f.name, exc)
    if deleted:
        logging.getLogger(__name__).info(
            "purged %d log file(s) older than %d days", deleted, retention_days
        )


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    log_retention_days: int = 7,
) -> None:
    """Configure root logger.

    Console handler: colored output to stdout.
    File handler:    plain text (no ANSI) in logs/YYYY-MM-DD_HH-MM-SS.log.
                     A new file is created on every run.
                     Files older than log_retention_days are deleted at startup.
    """
    _enable_windows_ansi()

    fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_level = getattr(logging, level.upper(), logging.INFO)

    # --- console handler (colored) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter(fmt, datefmt=datefmt))

    # --- file handler (plain text, new file each run) ---
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_path / f"{run_ts}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    # --- root logger ---
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Quiet noisy libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    # Purge old files after handlers are set so the purge message lands in the new log
    _purge_old_logs(log_path, log_retention_days)

    logging.getLogger(__name__).info("logging to file: %s", log_file)
