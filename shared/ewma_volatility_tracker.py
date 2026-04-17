"""Short-horizon EWMA volatility tracker.

The RiskMetrics-style exponentially weighted variance estimator reacts to
recent returns roughly 10-20x faster than the flat-weighted 2-hour
:class:`shared.volatility_tracker.VolatilityTracker`.

The motivation is concrete: v3.1 T4 fired during a squeeze that materialised
after window-open, so the slow 5-min close-to-close tracker — only sampled
at window boundaries — carried a stale reading straight through the fire
decision. A tick-fed EWMA catches that regime shift inside the same window.

Math
----
Let ``r_i`` be the log return between two consecutive samples (one sample
taken every ``sample_interval_s`` seconds). The EWMA variance update is

    σ²_t = λ · σ²_{t-1} + (1 - λ) · r_t²

with λ = 0.94 (RiskMetrics default). For a 10-second sampling cadence this
gives an effective half-life of roughly two minutes — short enough to see a
mid-window squeeze, long enough not to alarm on single-tick noise.

The exposed ``current_stddev_pct`` is ``sqrt(σ²_t) * 100`` — the one-sample
standard deviation of returns expressed in percent, directly comparable in
magnitude to the existing 5-minute close-to-close stddev when the sample
interval is chosen so the per-sample horizon is similar.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class EwmaVolatilityTracker:
    """Tick-fed EWMA variance estimator for BTC price returns.

    Feed ``record_sample(price, ts)`` from the strategy tick loop. Internal
    sampling throttles to ``sample_interval_s`` so noise from the 4 Hz tick
    cadence does not dominate.
    """

    def __init__(
        self,
        *,
        sample_interval_s: float = 10.0,
        decay_lambda: float = 0.94,
        min_samples: int = 6,
        warmup_variance: float = 0.0,
    ) -> None:
        if not (0.0 < decay_lambda < 1.0):
            msg = f"decay_lambda must be in (0, 1), got {decay_lambda}"
            raise ValueError(msg)
        if sample_interval_s <= 0.0:
            msg = f"sample_interval_s must be > 0, got {sample_interval_s}"
            raise ValueError(msg)

        self._interval = sample_interval_s
        self._lambda = decay_lambda
        self._min_samples = min_samples
        self._ewma_var: float = warmup_variance
        self._n_updates: int = 0
        self._last_price: float = 0.0
        self._last_sample_ts: float = 0.0

    def record_sample(self, price: float, ts: float) -> None:
        """Record a price observation.

        Only the first observation after ``sample_interval_s`` has elapsed
        since the last kept sample is used to update the variance. Calls
        before that window simply return.
        """
        if price <= 0.0:
            return
        # Throttle once a first sample has been seated — the sentinel is the
        # presence of a last_price rather than ts>0.0, since ts=0.0 is a
        # legitimate first observation in unit tests and monotonic clocks.
        if self._last_price > 0.0 and (ts - self._last_sample_ts) < self._interval:
            return

        if self._last_price > 0.0:
            r = math.log(price / self._last_price)
            self._ewma_var = self._lambda * self._ewma_var + (1.0 - self._lambda) * r * r
            self._n_updates += 1

        self._last_price = price
        self._last_sample_ts = ts

    @property
    def current_stddev_pct(self) -> float:
        """One-sample stddev of log returns, in percent."""
        if self._n_updates < self._min_samples:
            return 0.0
        return math.sqrt(self._ewma_var) * 100.0

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def ready(self) -> bool:
        return self._n_updates >= self._min_samples

    def reset(self) -> None:
        self._ewma_var = 0.0
        self._n_updates = 0
        self._last_price = 0.0
        self._last_sample_ts = 0.0

    # ------------------------------------------------------------------
    # Cache persistence — lets the fast estimator survive a bot restart
    # without waiting min_samples * sample_interval for re-warmup.
    # ------------------------------------------------------------------

    def save_cache(self, path: Path) -> None:
        data = {
            "ewma_var": self._ewma_var,
            "n_updates": self._n_updates,
            "last_price": self._last_price,
            "last_sample_ts": self._last_sample_ts,
            "saved_at": time.time(),
        }
        try:
            path.write_text(json.dumps(data))
        except OSError as exc:
            log.warning("failed to save fast-vol cache: %s", exc)

    def load_cache(self, path: Path, staleness_seconds: float) -> tuple[int, float]:
        """Restore state from disk if recent enough.

        Returns ``(n_updates_loaded, cache_age_seconds)``.
        """
        if not path.exists():
            return 0, 0.0
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
            log.warning("failed to read fast-vol cache: %s", exc)
            return 0, 0.0

        saved_at = data.get("saved_at")
        if saved_at is None:
            log.warning("fast-vol cache missing 'saved_at' — discarding")
            try:
                path.unlink()
            except OSError as exc:
                log.warning("failed to unlink fast-vol cache: %s", exc)
            return 0, 0.0

        age = time.time() - saved_at
        if age > staleness_seconds:
            log.info(
                "fast-vol cache stale (%.1f min old, limit=%.1f min) — discarding",
                age / 60,
                staleness_seconds / 60,
            )
            try:
                path.unlink()
            except OSError as exc:
                log.warning("failed to unlink fast-vol cache: %s", exc)
            return 0, age

        self._ewma_var = float(data.get("ewma_var", 0.0))
        self._n_updates = int(data.get("n_updates", 0))
        self._last_price = float(data.get("last_price", 0.0))
        self._last_sample_ts = float(data.get("last_sample_ts", 0.0))
        log.info(
            "fast-vol cache loaded: n_updates=%d stddev=%.3f%% age=%.1f min",
            self._n_updates,
            self.current_stddev_pct,
            age / 60,
        )
        return self._n_updates, age
