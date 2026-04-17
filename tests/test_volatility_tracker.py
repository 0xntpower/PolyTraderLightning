"""Tests for VolatilityTracker — focused on v3.2 §5.2 signed_return_t_stat."""

from __future__ import annotations

import math

import pytest

from shared.volatility_tracker import VolatilityTracker


def _feed(tracker: VolatilityTracker, prices: list[float]) -> None:
    for p in prices:
        tracker.record_close(p)


class TestSignedReturnTStat:
    def test_zero_when_insufficient_samples(self):
        t = VolatilityTracker(lookback_windows=24, min_samples=6)
        _feed(t, [100.0, 100.5, 101.0])  # only 2 returns
        assert t.signed_return_t_stat(min_samples=4) == 0.0

    def test_positive_t_for_consistent_uptrend(self):
        """Monotonically increasing prices → strong positive t-stat."""
        t = VolatilityTracker(lookback_windows=24, min_samples=6)
        prices = [100.0 * (1.002**i) for i in range(12)]  # ~0.2% per step
        _feed(t, prices)
        t_stat = t.signed_return_t_stat(min_samples=4)
        assert t_stat > 5.0, f"expected strong +t for uptrend, got {t_stat}"

    def test_negative_t_for_consistent_downtrend(self):
        t = VolatilityTracker(lookback_windows=24, min_samples=6)
        prices = [100.0 * (0.998**i) for i in range(12)]
        _feed(t, prices)
        t_stat = t.signed_return_t_stat(min_samples=4)
        assert t_stat < -5.0, f"expected strong -t for downtrend, got {t_stat}"

    def test_near_zero_for_oscillating_returns(self):
        """Alternating up/down moves should produce a small-magnitude t-stat."""
        t = VolatilityTracker(lookback_windows=24, min_samples=6)
        price = 100.0
        prices = [price]
        for i in range(11):
            factor = 1.005 if i % 2 == 0 else 1.0 / 1.005
            price *= factor
            prices.append(price)
        _feed(t, prices)
        t_stat = t.signed_return_t_stat(min_samples=4)
        # Oscillating → mean ≈ 0, t stays small in magnitude.
        assert abs(t_stat) < 1.0

    def test_zero_when_stddev_is_zero(self):
        """Constant prices → no returns variation → t-stat defaults to 0."""
        t = VolatilityTracker(lookback_windows=24, min_samples=6)
        _feed(t, [100.0] * 12)
        assert t.signed_return_t_stat(min_samples=4) == 0.0

    def test_matches_manual_t_statistic(self):
        """Direct comparison against manually computed t = mean / (sd / sqrt(n))."""
        t = VolatilityTracker(lookback_windows=24, min_samples=6)
        # 5 prices → 4 returns of roughly: +1%, -0.5%, +0.75%, -0.25%
        prices = [100.0, 101.0, 100.4949, 101.2511, 100.9979]
        _feed(t, prices)
        # compute expected returns in percent
        rets = [(prices[i] - prices[i - 1]) / prices[i - 1] * 100.0 for i in range(1, len(prices))]
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        expected = mean / (math.sqrt(var) / math.sqrt(len(rets)))
        assert t.signed_return_t_stat(min_samples=4) == pytest.approx(expected, rel=1e-9)
