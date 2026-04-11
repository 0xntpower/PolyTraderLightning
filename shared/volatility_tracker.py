"""Rolling BTC volatility tracker for adaptive bet sizing.

Tracks the standard deviation of 5-minute window returns (close-to-close)
using a rolling window. When volatility is elevated above a baseline, the
tracker provides a scaling factor to reduce bet size proportionally.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class VolatilityTracker:
    """Tracks rolling volatility of BTC 5-minute returns.

    Feed close prices at the end of each window. After enough samples
    accumulate, ``bet_scale()`` returns a [floor, 1.0] multiplier that
    reduces bets when recent volatility exceeds the baseline threshold.
    """

    def __init__(
        self,
        lookback_windows: int = 24,
        baseline_stddev_pct: float = 0.10,
        elevated_stddev_pct: float = 0.20,
        min_samples: int = 6,
    ) -> None:
        """
        Parameters
        ----------
        lookback_windows : int
            Number of recent 5-min returns to track (~2 hours at 24).
        baseline_stddev_pct : float
            Stddev of returns (in %) at or below which no scaling occurs.
        elevated_stddev_pct : float
            Stddev of returns (in %) at which scaling hits the floor.
        min_samples : int
            Minimum number of returns before volatility is considered valid.
        """
        self._lookback = lookback_windows
        self._baseline = baseline_stddev_pct
        self._elevated = elevated_stddev_pct
        self._min_samples = min_samples
        self._prices: deque[float] = deque(maxlen=lookback_windows + 1)
        self._last_stddev: float = 0.0

    def record_close(self, close_price: float) -> None:
        """Record the close price of a completed 5-minute window."""
        if close_price > 0:
            self._prices.append(close_price)

    @property
    def current_stddev_pct(self) -> float:
        """Current rolling stddev of returns in percent."""
        return self._last_stddev

    @property
    def n_returns(self) -> int:
        """Number of returns currently in the rolling window."""
        return max(0, len(self._prices) - 1)

    def update_stddev(self) -> float:
        """Recompute and cache the rolling stddev. Returns current value."""
        returns = self._compute_returns()
        if len(returns) >= self._min_samples:
            self._last_stddev = self._stddev(returns)
        return self._last_stddev

    def _compute_returns(self) -> list[float]:
        """Compute percent returns from consecutive close prices."""
        prices = list(self._prices)
        if len(prices) < 2:
            return []
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                ret = (prices[i] - prices[i - 1]) / prices[i - 1] * 100.0
                returns.append(ret)
        return returns

    @staticmethod
    def _stddev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def save_cache(self, path: Path) -> None:
        """Persist prices to disk for fast restart."""
        data = {
            "prices": list(self._prices),
            "last_stddev": self._last_stddev,
            "saved_at": time.time(),
        }
        try:
            path.write_text(json.dumps(data))
        except OSError as exc:
            log.warning("failed to save vol cache: %s", exc)

    def load_cache(self, path: Path, staleness_seconds: float) -> tuple[int, float]:
        """Restore prices from disk if recent enough.

        Returns (n_prices_loaded, cache_age_seconds).
        """
        if not path.exists():
            return 0, 0.0
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
            log.warning("failed to read vol cache: %s", exc)
            return 0, 0.0

        saved_at = data.get("saved_at")
        if saved_at is None:
            log.warning("vol cache missing 'saved_at' timestamp — discarding")
            try:
                path.unlink()
            except OSError as exc:
                log.warning("failed to unlink vol cache: %s", exc)
            return 0, 0.0

        age = time.time() - saved_at
        if age > staleness_seconds:
            log.info(
                "vol cache stale (%.1f min old, limit=%.1f min) — discarding",
                age / 60,
                staleness_seconds / 60,
            )
            try:
                path.unlink()
            except OSError as exc:
                log.warning("failed to unlink vol cache: %s", exc)
            return 0, age

        prices = data.get("prices", [])
        for p in prices:
            self._prices.append(p)
        self._last_stddev = data.get("last_stddev", 0.0)
        self.update_stddev()
        log.info(
            "vol cache loaded: prices=%d stddev=%.3f%% age=%.1f min",
            len(self._prices),
            self._last_stddev,
            age / 60,
        )
        return len(prices), age
