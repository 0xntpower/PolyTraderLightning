"""Tests for RegimeManager — tracker creation, cache lifecycle, warmup credit."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from strategy.regime import RegimeManager

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeVolTracker:
    def __init__(self, n_returns=0):
        self.n_returns = n_returns
        self._cache_loaded = False
        self._cache_saved = False

    def load_cache(self, path, staleness):
        self._cache_loaded = True

    def save_cache(self, path):
        self._cache_saved = True


class FakeChopDetector:
    def __init__(self, n_windows=0):
        self.n_windows = n_windows
        self._cache_loaded = False
        self._cache_saved = False

    def load_cache(self, path, staleness):
        self._cache_loaded = True

    def save_cache(self, path):
        self._cache_saved = True


class FakeOutcomeTracker:
    def __init__(self):
        self._cache_loaded = False
        self._cache_saved = False

    def load_cache(self, path, staleness):
        self._cache_loaded = True

    def save_cache(self, path):
        self._cache_saved = True


class FakePaths:
    vol_cache = "vol.json"
    chop_cache = "chop.json"
    outcome_cache = "outcome.json"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegimeManagerConstruction:
    def test_holds_trackers(self):
        vol = FakeVolTracker()
        chop = FakeChopDetector()
        out = FakeOutcomeTracker()
        mgr = RegimeManager(vol, chop, out)

        assert mgr.vol_tracker is vol
        assert mgr.chop_detector is chop
        assert mgr.outcome_tracker is out

    @patch("shared.volatility_tracker.VolatilityTracker")
    @patch("shared.chop_detector.ChopDetector")
    @patch("shared.outcome_tracker.OutcomeTracker")
    def test_create_builds_and_loads_caches(self, MockOutcome, MockChop, MockVol):
        MockVol.return_value = FakeVolTracker()
        MockChop.return_value = FakeChopDetector()
        MockOutcome.return_value = FakeOutcomeTracker()

        from fakes import make_regime_config

        cfg = make_regime_config(cache_staleness_minutes=30)
        paths = FakePaths()

        mgr = RegimeManager.create(cfg, paths)

        assert mgr.vol_tracker._cache_loaded
        assert mgr.chop_detector._cache_loaded
        assert mgr.outcome_tracker._cache_loaded

    @patch("shared.volatility_tracker.VolatilityTracker")
    @patch("shared.chop_detector.ChopDetector")
    @patch("shared.outcome_tracker.OutcomeTracker")
    def test_create_skips_cache_when_staleness_zero(self, MockOutcome, MockChop, MockVol):
        MockVol.return_value = FakeVolTracker()
        MockChop.return_value = FakeChopDetector()
        MockOutcome.return_value = FakeOutcomeTracker()

        from fakes import make_regime_config

        cfg = make_regime_config(cache_staleness_minutes=0)
        paths = FakePaths()

        mgr = RegimeManager.create(cfg, paths)

        assert not mgr.vol_tracker._cache_loaded
        assert not mgr.chop_detector._cache_loaded


class TestWarmupCredit:
    def test_no_credit_when_empty_trackers(self):
        mgr = RegimeManager(
            FakeVolTracker(n_returns=0),
            FakeChopDetector(n_windows=0),
            FakeOutcomeTracker(),
        )
        credit_s = mgr.compute_warmup_credit(warmup_minutes=30.0)
        assert credit_s == 0.0

    def test_credit_from_vol_and_chop(self):
        mgr = RegimeManager(
            FakeVolTracker(n_returns=10),
            FakeChopDetector(n_windows=6),
            FakeOutcomeTracker(),
        )
        # min(10, 6) = 6 windows × 5 min = 30 min = 1800s
        credit_s = mgr.compute_warmup_credit(warmup_minutes=60.0)
        assert credit_s == pytest.approx(1800.0)

    def test_credit_uses_min_of_trackers(self):
        mgr = RegimeManager(
            FakeVolTracker(n_returns=100),
            FakeChopDetector(n_windows=3),
            FakeOutcomeTracker(),
        )
        # min(100, 3) = 3 windows × 5 min = 15 min = 900s
        credit_s = mgr.compute_warmup_credit(warmup_minutes=30.0)
        assert credit_s == pytest.approx(900.0)


class TestSaveCaches:
    def test_saves_all_caches(self):
        vol = FakeVolTracker()
        chop = FakeChopDetector()
        out = FakeOutcomeTracker()
        mgr = RegimeManager(vol, chop, out)

        mgr.save_caches(FakePaths())

        assert vol._cache_saved
        assert chop._cache_saved
        assert out._cache_saved
