"""Tests for EwmaVolatilityTracker."""

from __future__ import annotations

import math
import time

import pytest

from shared.ewma_volatility_tracker import EwmaVolatilityTracker


class TestRecordSample:
    def test_interval_throttles_updates(self):
        t = EwmaVolatilityTracker(sample_interval_s=10.0, decay_lambda=0.94, min_samples=1)
        t.record_sample(100.0, ts=0.0)
        t.record_sample(101.0, ts=5.0)  # within interval — ignored
        assert t.n_updates == 0

    def test_first_kept_sample_seeds_no_update(self):
        """A single sample cannot form a return — no variance update yet."""
        t = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        t.record_sample(100.0, ts=0.0)
        assert t.n_updates == 0
        assert t.current_stddev_pct == 0.0

    def test_two_samples_produce_one_update(self):
        t = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        t.record_sample(100.0, ts=0.0)
        t.record_sample(101.0, ts=2.0)
        assert t.n_updates == 1

    def test_not_ready_below_min_samples(self):
        t = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=5)
        t.record_sample(100.0, ts=0.0)
        t.record_sample(101.0, ts=2.0)
        assert not t.ready
        assert t.current_stddev_pct == 0.0

    def test_ready_at_min_samples(self):
        t = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=3)
        for i, p in enumerate([100.0, 100.5, 99.8, 100.2]):
            t.record_sample(p, ts=float(i * 2))
        assert t.ready
        assert t.current_stddev_pct > 0.0

    def test_zero_price_is_ignored(self):
        t = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        t.record_sample(0.0, ts=0.0)
        t.record_sample(100.0, ts=2.0)
        t.record_sample(101.0, ts=4.0)
        # zero was rejected; only two valid samples → one update
        assert t.n_updates == 1


class TestEwmaFormula:
    def test_variance_matches_risk_metrics_recursion(self):
        """EWMA σ² = λ·σ²_prev + (1-λ)·r²"""
        decay = 0.9
        t = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=decay, min_samples=1)
        # Three samples: r1 between s0/s1, r2 between s1/s2
        prices = [100.0, 101.0, 100.5]
        for i, p in enumerate(prices):
            t.record_sample(p, ts=float(i * 2))

        r1 = math.log(101.0 / 100.0)
        r2 = math.log(100.5 / 101.0)
        expected_var = decay * (decay * 0.0 + (1.0 - decay) * r1 * r1) + (1.0 - decay) * r2 * r2
        expected_stddev_pct = math.sqrt(expected_var) * 100.0
        assert t.current_stddev_pct == pytest.approx(expected_stddev_pct, rel=1e-9)

    def test_responds_faster_than_flat_average(self):
        """EWMA with λ=0.94 should react visibly to a recent burst even
        after a long calm history."""
        t = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        # 40 calm samples (~0 return)
        price = 100.0
        for i in range(40):
            price *= 1.00005  # 0.005% drift
            t.record_sample(price, ts=float(i * 2))
        calm_stddev = t.current_stddev_pct

        # One large burst
        price *= 1.01  # 1% jump
        t.record_sample(price, ts=float(80))
        burst_stddev = t.current_stddev_pct

        # The burst must push EWMA stddev meaningfully higher than calm baseline.
        assert burst_stddev > calm_stddev * 5.0


class TestReset:
    def test_reset_zeros_all_state(self):
        t = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        t.record_sample(100.0, ts=0.0)
        t.record_sample(101.0, ts=2.0)
        assert t.n_updates == 1
        t.reset()
        assert t.n_updates == 0
        assert t.current_stddev_pct == 0.0


class TestCachePersistence:
    def test_save_then_load_round_trip(self, tmp_path):
        path = tmp_path / "fast_vol.json"
        t1 = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        t1.record_sample(100.0, ts=0.0)
        t1.record_sample(101.0, ts=2.0)
        t1.record_sample(100.8, ts=4.0)
        saved_stddev = t1.current_stddev_pct
        t1.save_cache(path)

        t2 = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        n, _ = t2.load_cache(path, staleness_seconds=3600.0)
        assert n == t1.n_updates
        assert t2.current_stddev_pct == pytest.approx(saved_stddev, rel=1e-9)

    def test_load_rejects_stale_cache(self, tmp_path):
        path = tmp_path / "fast_vol.json"
        t1 = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        t1.record_sample(100.0, ts=0.0)
        t1.record_sample(101.0, ts=2.0)
        t1.save_cache(path)

        # Make it look ancient
        import json

        data = json.loads(path.read_text())
        data["saved_at"] = time.time() - 7200.0  # 2h old
        path.write_text(json.dumps(data))

        t2 = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        n, _ = t2.load_cache(path, staleness_seconds=1800.0)  # 30 min limit
        assert n == 0
        assert not path.exists()

    def test_missing_file_is_noop(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        t = EwmaVolatilityTracker(sample_interval_s=1.0, decay_lambda=0.94, min_samples=1)
        n, age = t.load_cache(path, staleness_seconds=3600.0)
        assert n == 0
        assert age == 0.0


class TestValidation:
    def test_rejects_bad_decay(self):
        with pytest.raises(ValueError, match="decay_lambda"):
            EwmaVolatilityTracker(decay_lambda=0.0)
        with pytest.raises(ValueError, match="decay_lambda"):
            EwmaVolatilityTracker(decay_lambda=1.0)

    def test_rejects_bad_interval(self):
        with pytest.raises(ValueError, match="sample_interval_s"):
            EwmaVolatilityTracker(sample_interval_s=0.0)
