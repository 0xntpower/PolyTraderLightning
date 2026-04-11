"""Data quality primitives shared across PolySignalLab.

Used by the standalone verify_data CLI tool, the SignalOrchestrator, and the
collector's rolling pool management. All Parquet filename parsing and basic
pool-level checks live here so the logic stays consistent.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Engine pipeline parameters — mirrored from Config.hpp so Python tools
# stay in sync with what the C++ engine actually expects.
WALK_FORWARD_FOLDS = 8
MIN_TRAIN_FOLDS = 2
TEST_FOLDS = WALK_FORWARD_FOLDS - MIN_TRAIN_FOLDS  # 6
SIGNAL_FIRE_RATE = 0.25
MIN_OOS_MATCHING_WINDOWS = 20
WINDOW_DURATION_S = 300

# Dataset size thresholds (derived from engine requirements)
MIN_PER_FOLD = int(MIN_OOS_MATCHING_WINDOWS / SIGNAL_FIRE_RATE)  # 80
MINIMUM_WINDOWS = MIN_PER_FOLD * WALK_FORWARD_FOLDS  # 400
GOOD_WINDOWS = 500
STRONG_WINDOWS = 750
IDEAL_WINDOWS = 1000

COLLECTION_RATE_PER_HOUR = 12  # ~1 window every 5 minutes

_FILENAME_RE = re.compile(r"market_(\d+)_resolved_(UP|DOWN)(?:_PARTIAL)?\.parquet")


def parse_parquet_filename(filename: str) -> tuple[int | None, str | None]:
    """Extract (window_timestamp, resolution) from a Parquet filename.

    Returns (None, None) if the filename doesn't match the expected pattern.
    """
    m = _FILENAME_RE.match(filename)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def parse_window_ts(filename: str) -> int | None:
    """Extract just the window timestamp from a Parquet filename."""
    ts, _ = parse_parquet_filename(filename)
    return ts


def list_data_pool(data_dir: Path) -> list[tuple[Path, int, str]]:
    """List all valid Parquet files in data_dir with parsed metadata.

    Returns list of (path, window_ts, resolution) sorted by timestamp.
    """
    results = []
    for f in data_dir.glob("market_*_resolved_*.parquet"):
        ts, resolution = parse_parquet_filename(f.name)
        if ts is not None and resolution is not None:
            results.append((f, ts, resolution))
    results.sort(key=lambda x: x[1])
    return results


def check_pool_size(
    data_dir: Path,
    min_windows: int,
) -> tuple[bool, int, str]:
    """Check whether the data pool has enough windows.

    Returns (passes, window_count, reason).
    """
    pool = list_data_pool(data_dir)
    n = len(pool)
    if n < min_windows:
        deficit = min_windows - n
        hours = deficit / COLLECTION_RATE_PER_HOUR
        return False, n, f"need {min_windows} windows, have {n} ({hours:.1f}h to collect)"
    return True, n, "ok"


def check_staleness(
    data_dir: Path,
    max_stale_hours: float,
) -> tuple[bool, float, str]:
    """Check whether the newest data is too old.

    Returns (passes, age_hours, reason).
    """
    pool = list_data_pool(data_dir)
    if not pool:
        return False, float("inf"), "no data files found"

    newest_ts = pool[-1][1]  # already sorted by timestamp
    age_hours = (time.time() - newest_ts) / 3600

    if age_hours > max_stale_hours:
        return False, age_hours, f"newest data is {age_hours:.1f}h old (max {max_stale_hours}h)"
    return True, age_hours, "ok"


def estimate_wait_seconds(data_dir: Path, min_windows: int) -> float:
    """Estimate seconds until data pool reaches min_windows based on collection rate."""
    pool = list_data_pool(data_dir)
    deficit = max(0, min_windows - len(pool))
    return (deficit / COLLECTION_RATE_PER_HOUR) * 3600


def estimate_fold_quality(n_windows: int) -> tuple[int, int, bool]:
    """Estimate what the engine's walk-forward folds would look like.

    Returns (per_fold_size, estimated_oos_matches, can_qualify).
    """
    per_fold = n_windows // WALK_FORWARD_FOLDS
    matches_per_test_fold = int(per_fold * SIGNAL_FIRE_RATE)
    can_qualify = matches_per_test_fold >= MIN_OOS_MATCHING_WINDOWS
    return per_fold, matches_per_test_fold, can_qualify
