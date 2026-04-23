"""Structured logging with ANSI color output (console) and plain file output."""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Keep the last N sessions' logs in logs/log_archive/ as a local safety net;
# older entries get hard-deleted on the next startup.
_ARCHIVE_SUBDIR = "log_archive"
_ARCHIVE_CAP = 30

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


def _archive_previous_logs(log_dir: Path) -> tuple[int, int]:
    """Move prior session logs into log_dir/log_archive/ and prune the
    archive down to the most recent _ARCHIVE_CAP files.

    Matches the YYYY-MM-DD_HH-MM-SS.log filename scheme so we don't sweep
    unrelated .log files that may live in the dir. Called before handlers
    are installed — errors go to stderr. Returns (moved, pruned) so the
    caller can report counts into the new session log.
    """
    archive_dir = log_dir / _ARCHIVE_SUBDIR
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for src in log_dir.glob("*.log"):
        if not src.is_file():
            continue
        try:
            datetime.strptime(src.stem, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        dest = archive_dir / src.name
        counter = 1
        while dest.exists():
            dest = archive_dir / f"{src.stem}_{counter}.log"
            counter += 1
        try:
            src.rename(dest)
            moved += 1
        except OSError as exc:
            print(f"warning: failed to archive {src.name}: {exc}", file=sys.stderr)

    archives = sorted(archive_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    pruned = 0
    excess = len(archives) - _ARCHIVE_CAP
    for old in archives[: max(0, excess)]:
        try:
            old.unlink()
            pruned += 1
        except OSError as exc:
            print(f"warning: failed to prune archive {old.name}: {exc}", file=sys.stderr)

    return moved, pruned


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
) -> None:
    """Configure root logger.

    Console handler: colored output to stdout.
    File handler:    plain text (no ANSI) in logs/YYYY-MM-DD_HH-MM-SS.log.
                     A new file is created on every run; the previous
                     session's log is moved into logs/log_archive/ (capped
                     at 30 files, oldest pruned on overflow).
    """
    _enable_windows_ansi()

    fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    log_level = getattr(logging, level.upper(), logging.INFO)

    # --- console handler (colored) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter(fmt, datefmt=datefmt))

    # --- prepare log dir and archive previous session before opening new file ---
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    moved, pruned = _archive_previous_logs(log_path)

    # --- file handler (plain text, new file each run) ---
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

    logging.getLogger(__name__).info("logging to file: %s", log_file)
    if moved or pruned:
        logging.getLogger(__name__).info(
            "log archive: moved %d previous log(s), pruned %d (cap=%d)",
            moved,
            pruned,
            _ARCHIVE_CAP,
        )
