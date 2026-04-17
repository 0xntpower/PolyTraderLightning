"""Recent window outcome tracker for directional bias detection.

Tracks the resolved direction (up/down) of recent windows to detect
short-term directional regimes that contradict the active signal.

When the market has been consistently resolving against the signal's
direction (e.g., 5 of the last 6 windows resolved DOWN while the
signal predicts UP), the tracker produces a scaling factor that
gradually reduces bet size — following the same gradual-taper
philosophy as volatility and chop scaling.

The tracker only considers resolved outcomes, not whether the bot
traded them. Every 5-minute window produces one outcome regardless
of whether a signal fired.

v3.2 §5.9: magnitude-weighted mode weights each window by its
``|close_delta_pct|`` so tiny-move noise windows don't dominate the
indicator.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class OutcomeRecord(NamedTuple):
    """Per-window direction + magnitude used for agreement weighting."""

    direction: str  # "up" or "down"
    magnitude_pct: float  # |close_delta_pct|, non-negative


class OutcomeTracker:
    """Rolling window of recent market outcomes for directional bias detection.

    Call ``record_outcome()`` at each window boundary with the resolved
    direction and (optionally) the ``|close_delta_pct|`` magnitude. Before
    firing, query ``direction_agreement()`` to get the fraction of recent
    windows that resolved in the signal's direction.
    """

    def __init__(
        self,
        lookback_windows: int = 6,
        *,
        magnitude_weighted: bool = False,
        min_magnitude_pct: float = 0.0,
    ) -> None:
        self._lookback = lookback_windows
        self._history: deque[OutcomeRecord] = deque(maxlen=lookback_windows)
        self._magnitude_weighted = magnitude_weighted
        self._min_magnitude = max(0.0, min_magnitude_pct)

    def record_outcome(self, direction: str, magnitude_pct: float = 0.0) -> None:
        """Record a resolved window outcome ("up" or "down") with magnitude."""
        if direction not in ("up", "down"):
            return
        self._history.append(OutcomeRecord(direction, abs(magnitude_pct)))

    def direction_agreement(self, signal_side: str) -> float:
        """Fraction of recent outcomes matching the signal's side.

        Returns 1.0 when all recent windows agree with the signal, 0.0
        when none do. Returns 1.0 (no penalty) when insufficient data is
        available. With ``magnitude_weighted=True``, each window's
        contribution is scaled by ``max(|delta|, min_magnitude)`` so
        small-move windows are down-weighted.
        """
        if len(self._history) < 3:
            return 1.0
        if not self._magnitude_weighted:
            matching = sum(1 for r in self._history if r.direction == signal_side)
            return matching / len(self._history)

        total_weight = 0.0
        matching_weight = 0.0
        for r in self._history:
            w = max(r.magnitude_pct, self._min_magnitude)
            total_weight += w
            if r.direction == signal_side:
                matching_weight += w
        if total_weight <= 0.0:
            # Fall back to count-based if all weights are zero (no magnitude data).
            matching = sum(1 for r in self._history if r.direction == signal_side)
            return matching / len(self._history)
        return matching_weight / total_weight

    @property
    def n_windows(self) -> int:
        return len(self._history)

    def summary(self) -> str:
        """Human-readable summary of recent outcomes."""
        if not self._history:
            return "no data"
        ups = sum(1 for r in self._history if r.direction == "up")
        downs = len(self._history) - ups
        return f"{ups}U/{downs}D over {len(self._history)}w"

    # ------------------------------------------------------------------
    # Cache persistence (same pattern as ChopDetector)
    # ------------------------------------------------------------------

    def save_cache(self, path: Path) -> None:
        data = {
            "history": [
                {"direction": r.direction, "magnitude_pct": r.magnitude_pct} for r in self._history
            ],
            "saved_at": time.time(),
        }
        try:
            path.write_text(json.dumps(data))
        except OSError as exc:
            log.warning("failed to save outcome cache: %s", exc)

    def load_cache(self, path: Path, staleness_seconds: float) -> tuple[int, float]:
        """Restore from disk if recent enough. Returns (n_loaded, age_seconds)."""
        if not path.exists():
            return 0, 0.0
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
            log.warning("failed to read outcome cache: %s", exc)
            return 0, 0.0

        saved_at = data.get("saved_at")
        if saved_at is None:
            log.warning("outcome cache missing 'saved_at' — discarding")
            try:
                path.unlink()
            except OSError as exc:
                log.warning("failed to unlink outcome cache: %s", exc)
            return 0, 0.0

        age = time.time() - saved_at
        if age > staleness_seconds:
            log.info(
                "outcome cache stale (%.1f min old, limit=%.1f min) — discarding",
                age / 60,
                staleness_seconds / 60,
            )
            try:
                path.unlink()
            except OSError as exc:
                log.warning("failed to unlink outcome cache: %s", exc)
            return 0, age

        history = data.get("history", [])
        for entry in history:
            # Accept both legacy string format and the v3.2 dict format.
            if isinstance(entry, str):
                if entry in ("up", "down"):
                    self._history.append(OutcomeRecord(entry, 0.0))
            elif isinstance(entry, dict):
                d = entry.get("direction")
                m = entry.get("magnitude_pct", 0.0)
                if d in ("up", "down") and isinstance(m, int | float):
                    self._history.append(OutcomeRecord(d, abs(float(m))))
        log.info(
            "outcome cache loaded: windows=%d recent=%s age=%.1f min",
            len(self._history),
            self.summary(),
            age / 60,
        )
        return len(history), age
