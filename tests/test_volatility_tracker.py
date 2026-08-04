"""Tests for VolatilityTracker — signed_return_t_stat, stddev, rolling window, cache."""

from __future__ import annotations

import json
import math
import time

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


# ======================================================================
# Standard deviation correctness (relocated from PolySignalLab root tests/)
# ======================================================================


class TestStddev:
    """Verify the sample standard deviation uses N-1 (Bessel's correction)."""

    def test_sample_variance_two_values(self):
        """With 2 values, sample variance divides by 1."""
        vt = VolatilityTracker(lookback_windows=10, min_samples=2)
        # Two prices: 100 -> 102 gives one return = 2%
        # One return can't produce a meaningful stddev, so we need at least 2 returns.
        vt.record_close(100.0)
        vt.record_close(102.0)
        vt.record_close(100.0)
        # Returns: [+2.0%, -1.9608%]
        stddev = vt.update_stddev()
        # Manual: mean ~ 0.0196, var = ((2.0 - 0.0196)^2 + (-1.9608 - 0.0196)^2) / 1
        # ~ (3.922 + 3.922) / 1 = 7.844, stddev ~ 2.80
        assert stddev > 0
        assert stddev == pytest.approx(2.80, abs=0.05)

    def test_sample_vs_population_variance(self):
        """Confirm we use N-1, not N. With small N the difference is large."""
        vt = VolatilityTracker(lookback_windows=50, min_samples=2)
        # Feed prices that produce known returns
        prices = [100.0, 101.0, 99.5, 100.5, 98.0, 101.5]
        for p in prices:
            vt.record_close(p)
        stddev = vt.update_stddev()

        # Manually compute with N-1
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1] * 100.0 for i in range(1, len(prices))
        ]
        mean = sum(returns) / len(returns)
        sample_var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        pop_var = sum((r - mean) ** 2 for r in returns) / len(returns)
        expected_sample = math.sqrt(sample_var)
        expected_pop = math.sqrt(pop_var)

        assert stddev == pytest.approx(expected_sample, abs=0.001)
        assert stddev != pytest.approx(expected_pop, abs=0.001)

    def test_constant_prices_zero_stddev(self):
        vt = VolatilityTracker(lookback_windows=10, min_samples=2)
        for _ in range(5):
            vt.record_close(100.0)
        assert vt.update_stddev() == pytest.approx(0.0)


# ======================================================================
# Rolling window
# ======================================================================


class TestRollingWindow:
    def test_n_returns_correct(self):
        vt = VolatilityTracker(lookback_windows=10)
        assert vt.n_returns == 0
        vt.record_close(100.0)
        assert vt.n_returns == 0  # need 2 prices for 1 return
        vt.record_close(101.0)
        assert vt.n_returns == 1

    def test_lookback_window_bounded(self):
        """Prices beyond lookback_windows + 1 are dropped."""
        vt = VolatilityTracker(lookback_windows=5)
        for i in range(20):
            vt.record_close(100.0 + i)
        assert vt.n_returns == 5  # lookback_windows

    def test_min_samples_enforced(self):
        vt = VolatilityTracker(lookback_windows=10, min_samples=6)
        for i in range(4):  # only 3 returns
            vt.record_close(100.0 + i)
        stddev = vt.update_stddev()
        assert stddev == 0.0  # not enough samples

    def test_zero_price_ignored(self):
        vt = VolatilityTracker(lookback_windows=10, min_samples=2)
        vt.record_close(100.0)
        vt.record_close(0.0)  # should be ignored
        vt.record_close(101.0)
        assert vt.n_returns == 1  # 0.0 not recorded


# ======================================================================
# Cache persistence
# ======================================================================


class TestCache:
    def test_save_and_load(self, tmp_path):
        vt1 = VolatilityTracker(lookback_windows=10, min_samples=2)
        for p in [100.0, 101.0, 99.0, 102.0]:
            vt1.record_close(p)
        vt1.update_stddev()

        cache_file = tmp_path / "vol_cache.json"
        vt1.save_cache(cache_file)

        vt2 = VolatilityTracker(lookback_windows=10, min_samples=2)
        n_loaded, age = vt2.load_cache(cache_file, staleness_seconds=60)

        assert n_loaded == 4
        assert age < 5  # just saved
        assert vt2.n_returns == 3
        assert vt2.current_stddev_pct == pytest.approx(vt1.current_stddev_pct, abs=0.001)

    def test_stale_cache_discarded(self, tmp_path):
        vt = VolatilityTracker(lookback_windows=10, min_samples=2)
        vt.record_close(100.0)
        vt.record_close(101.0)

        cache_file = tmp_path / "vol_cache.json"
        vt.save_cache(cache_file)

        # Tamper with saved_at to make it old
        data = json.loads(cache_file.read_text())
        data["saved_at"] = time.time() - 3600  # 1 hour old
        cache_file.write_text(json.dumps(data))

        vt2 = VolatilityTracker(lookback_windows=10, min_samples=2)
        n_loaded, age = vt2.load_cache(cache_file, staleness_seconds=300)
        assert n_loaded == 0  # discarded

    def test_missing_cache_returns_zero(self, tmp_path):
        vt = VolatilityTracker()
        n_loaded, age = vt.load_cache(tmp_path / "nonexistent.json", staleness_seconds=60)
        assert n_loaded == 0
        assert age == 0.0
