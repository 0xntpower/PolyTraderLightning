"""Intra-window chop detector for adaptive bet sizing.

Tracks direction flips and delta range within each 5-minute window.
When recent windows show elevated chop (frequent reversals, wide delta
swings), the detector signals that momentum patterns are less reliable
and bet size should be reduced.

The key observation: in choppy regimes, BTC reverses direction multiple
times within a single window, causing momentum signals to fire on a move
that doesn't hold through window close.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WindowChopStats:
    """Summary of choppiness within a single 5-minute window."""

    direction_flips: int = 0
    delta_range_pct: float = 0.0  # max_delta - min_delta within the window
    n_ticks: int = 0


class ChopDetector:
    """Tracks intra-window price chop across rolling windows.

    Feed direction ticks during each window. At window boundaries, call
    ``finalize_window()`` to record the stats and ``bet_scale()`` to get
    the current scaling factor.
    """

    def __init__(
        self,
        lookback_windows: int = 6,
        baseline_flips: float = 2.0,
        elevated_flips: float = 5.0,
        min_samples: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        lookback_windows : int
            Number of recent windows to average over (~30 min at 6).
        baseline_flips : float
            Avg direction flips per window at or below which no scaling occurs.
        elevated_flips : float
            Avg flips at which scaling hits the floor.
        min_samples : int
            Minimum completed windows before chop scaling is active.
        """
        self._lookback = lookback_windows
        self._baseline = baseline_flips
        self._elevated = elevated_flips
        self._min_samples = min_samples
        self._history: deque[WindowChopStats] = deque(maxlen=lookback_windows)

        # Current window tracking
        self._last_direction: str = ""
        self._flips: int = 0
        self._min_delta: float = float("inf")
        self._max_delta: float = float("-inf")
        self._n_ticks: int = 0

    def tick(self, direction: str, delta_pct: float) -> None:
        """Feed one tick from the strategy loop.

        Parameters
        ----------
        direction : str
            "up", "down", or "none"
        delta_pct : float
            Current chainlink delta from open in percent.
        """
        if direction == "none":
            return

        self._n_ticks += 1

        # Track delta range
        self._min_delta = min(self._min_delta, delta_pct)
        self._max_delta = max(self._max_delta, delta_pct)

        # Track direction flips
        if self._last_direction and direction != self._last_direction:
            self._flips += 1
        self._last_direction = direction

    def finalize_window(self) -> WindowChopStats:
        """Record stats for the completed window and reset for the next one."""
        delta_range = (self._max_delta - self._min_delta) if self._n_ticks > 0 else 0.0
        stats = WindowChopStats(
            direction_flips=self._flips,
            delta_range_pct=delta_range,
            n_ticks=self._n_ticks,
        )
        self._history.append(stats)

        # Reset for next window
        self._last_direction = ""
        self._flips = 0
        self._min_delta = float("inf")
        self._max_delta = float("-inf")
        self._n_ticks = 0

        return stats

    @property
    def avg_flips(self) -> float:
        """Average direction flips per window over the lookback."""
        if not self._history:
            return 0.0
        return sum(w.direction_flips for w in self._history) / len(self._history)

    @property
    def avg_delta_range(self) -> float:
        """Average delta range per window over the lookback."""
        if not self._history:
            return 0.0
        return sum(w.delta_range_pct for w in self._history) / len(self._history)

    @property
    def n_windows(self) -> int:
        """Number of completed windows in the lookback."""
        return len(self._history)

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def save_cache(self, path: Path) -> None:
        """Persist window history to disk for fast restart."""
        data = {
            "history": [
                {"flips": w.direction_flips, "range": w.delta_range_pct, "ticks": w.n_ticks}
                for w in self._history
            ],
            "saved_at": time.time(),
        }
        try:
            path.write_text(json.dumps(data))
        except OSError as exc:
            log.warning("failed to save chop cache: %s", exc)

    def load_cache(self, path: Path, staleness_seconds: float) -> tuple[int, float]:
        """Restore window history from disk if recent enough.

        Returns (n_windows_loaded, cache_age_seconds).
        """
        if not path.exists():
            return 0, 0.0
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
            log.warning("failed to read chop cache: %s", exc)
            return 0, 0.0

        saved_at = data.get("saved_at")
        if saved_at is None:
            log.warning("chop cache missing 'saved_at' timestamp — discarding")
            try:
                path.unlink()
            except OSError as exc:
                log.warning("failed to unlink chop cache: %s", exc)
            return 0, 0.0

        age = time.time() - saved_at
        if age > staleness_seconds:
            log.info(
                "chop cache stale (%.1f min old, limit=%.1f min) — discarding",
                age / 60,
                staleness_seconds / 60,
            )
            try:
                path.unlink()
            except OSError as exc:
                log.warning("failed to unlink chop cache: %s", exc)
            return 0, age

        history = data.get("history", [])
        for w in history:
            self._history.append(
                WindowChopStats(
                    direction_flips=w.get("flips", 0),
                    delta_range_pct=w.get("range", 0.0),
                    n_ticks=w.get("ticks", 0),
                )
            )
        log.info(
            "chop cache loaded: windows=%d avg_flips=%.1f age=%.1f min",
            len(self._history),
            self.avg_flips,
            age / 60,
        )
        return len(history), age
