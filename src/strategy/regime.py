"""RegimeManager — owns vol/chop/outcome trackers and warmup credit.

Consolidates tracker creation, cache lifecycle, and warmup credit
calculation that were previously scattered across ``run()`` and
``_strategy_loop()`` in ``main.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import DataPaths, RegimeConfig
    from shared.chop_detector import ChopDetector
    from shared.ewma_volatility_tracker import EwmaVolatilityTracker
    from shared.outcome_tracker import OutcomeTracker
    from shared.volatility_tracker import VolatilityTracker

log = logging.getLogger(__name__)


class RegimeManager:
    """Owns vol/chop/outcome trackers and computes warmup credit.

    Lifecycle:
      1. ``create()`` — class method that builds trackers + loads caches
      2. ``compute_warmup_credit()`` — backdate bot_start_time
      3. Tracker accessors for WindowEventHandler / strategy loop
      4. ``save_caches()`` — persist tracker state
    """

    def __init__(
        self,
        vol_tracker: VolatilityTracker,
        chop_detector: ChopDetector,
        outcome_tracker: OutcomeTracker,
        fast_vol_tracker: EwmaVolatilityTracker | None = None,
    ) -> None:
        self.vol_tracker = vol_tracker
        self.chop_detector = chop_detector
        self.outcome_tracker = outcome_tracker
        self.fast_vol_tracker = fast_vol_tracker

    @classmethod
    def create(cls, cfg: RegimeConfig, paths: DataPaths) -> RegimeManager:
        """Create all trackers and load caches in one call."""
        from shared.chop_detector import ChopDetector
        from shared.ewma_volatility_tracker import EwmaVolatilityTracker
        from shared.outcome_tracker import OutcomeTracker
        from shared.volatility_tracker import VolatilityTracker

        vol_tracker = VolatilityTracker(
            lookback_windows=cfg.vol_lookback_windows,
            baseline_stddev_pct=cfg.vol_normal_pct,
            elevated_stddev_pct=cfg.vol_high_pct,
            min_samples=cfg.vol_min_samples,
        )
        chop_detector = ChopDetector(
            lookback_windows=cfg.chop_lookback_windows,
            baseline_flips=cfg.chop_normal_flips,
            elevated_flips=cfg.chop_high_flips,
            min_samples=cfg.chop_min_samples,
        )
        outcome_tracker = OutcomeTracker(
            lookback_windows=cfg.outcome_lookback_windows,
        )
        fast_vol_tracker: EwmaVolatilityTracker | None = None
        if cfg.vol_fast_enabled:
            fast_vol_tracker = EwmaVolatilityTracker(
                sample_interval_s=cfg.vol_fast_sample_interval_s,
                decay_lambda=cfg.vol_fast_decay_lambda,
                min_samples=cfg.vol_fast_min_samples,
            )

        staleness_sec = cfg.cache_staleness_minutes * 60
        if staleness_sec > 0:
            vol_tracker.load_cache(paths.vol_cache, staleness_sec)
            chop_detector.load_cache(paths.chop_cache, staleness_sec)
            outcome_tracker.load_cache(paths.outcome_cache, staleness_sec)
            if fast_vol_tracker is not None:
                fast_vol_tracker.load_cache(paths.fast_vol_cache, staleness_sec)

        return cls(vol_tracker, chop_detector, outcome_tracker, fast_vol_tracker)

    def compute_warmup_credit(self, warmup_minutes: float) -> float:
        """Compute warmup credit from pre-existing tracker data.

        Returns the number of seconds to backdate bot_start_time.
        Logs the credit applied.
        """
        tracker_windows = min(
            self.vol_tracker.n_returns,
            self.chop_detector.n_windows,
        )
        if tracker_windows <= 0:
            return 0.0

        credit_minutes = tracker_windows * 5.0
        remaining = max(0.0, warmup_minutes - credit_minutes)
        log.info(
            "warmup credit: vol=%d_prices chop=%d_windows — "
            "crediting %.0f min toward warmup (%.0f min remaining)",
            self.vol_tracker.n_returns,
            self.chop_detector.n_windows,
            credit_minutes,
            remaining,
        )
        return credit_minutes * 60  # seconds

    def save_caches(self, paths: DataPaths) -> None:
        """Persist all tracker caches."""
        self.vol_tracker.save_cache(paths.vol_cache)
        self.chop_detector.save_cache(paths.chop_cache)
        self.outcome_tracker.save_cache(paths.outcome_cache)
        if self.fast_vol_tracker is not None:
            self.fast_vol_tracker.save_cache(paths.fast_vol_cache)
