"""Tests for OutcomeTracker — v3.2 §5.9 magnitude-weighted agreement."""

from __future__ import annotations

import json

from shared.outcome_tracker import OutcomeTracker


class TestRecordAndBasics:
    def test_invalid_direction_ignored(self):
        t = OutcomeTracker(lookback_windows=6)
        t.record_outcome("sideways", 0.5)
        t.record_outcome("", 1.0)
        assert t.n_windows == 0

    def test_record_preserves_magnitude(self):
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=True)
        t.record_outcome("up", -0.25)  # sign stripped by abs
        t.record_outcome("down", 0.75)
        assert t.n_windows == 2
        assert t.summary() == "1U/1D over 2w"

    def test_lookback_truncation(self):
        t = OutcomeTracker(lookback_windows=3)
        for _ in range(5):
            t.record_outcome("up", 0.1)
        assert t.n_windows == 3

    def test_insufficient_samples_returns_one(self):
        t = OutcomeTracker(lookback_windows=6)
        t.record_outcome("up", 0.5)
        t.record_outcome("up", 0.5)
        assert t.direction_agreement("up") == 1.0


class TestLegacyCountBased:
    def test_full_agreement(self):
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=False)
        for _ in range(6):
            t.record_outcome("up", 0.5)
        assert t.direction_agreement("up") == 1.0
        assert t.direction_agreement("down") == 0.0

    def test_half_agreement(self):
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=False)
        for _ in range(3):
            t.record_outcome("up", 0.5)
        for _ in range(3):
            t.record_outcome("down", 2.0)
        # Count-based ignores magnitude → 3/6 = 0.5
        assert t.direction_agreement("up") == 0.5

    def test_magnitude_ignored_when_disabled(self):
        """Big DOWN moves should not reduce up-agreement when count-based."""
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=False)
        for _ in range(4):
            t.record_outcome("up", 0.05)  # tiny ups
        for _ in range(2):
            t.record_outcome("down", 5.0)  # huge downs
        # 4 of 6 match UP → 4/6
        assert abs(t.direction_agreement("up") - 4 / 6) < 1e-9


class TestMagnitudeWeighted:
    def test_big_move_dominates(self):
        """5 tiny UP windows + 1 huge DOWN window → down bias."""
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=True, min_magnitude_pct=0.0)
        for _ in range(5):
            t.record_outcome("up", 0.01)  # 0.01% each → total weight 0.05
        t.record_outcome("down", 5.0)  # weight 5.0
        # up_weight = 0.05, total = 5.05 → up agreement ≈ 0.0099
        agreement = t.direction_agreement("up")
        assert agreement < 0.02, f"expected ≈0.01 down-dominated, got {agreement}"

    def test_equal_magnitude_matches_count_based(self):
        """If all windows have the same magnitude, weighted == count-based."""
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=True, min_magnitude_pct=0.0)
        for _ in range(3):
            t.record_outcome("up", 1.0)
        for _ in range(3):
            t.record_outcome("down", 1.0)
        assert t.direction_agreement("up") == 0.5

    def test_min_magnitude_floor_prevents_drop_out(self):
        """Zero-magnitude windows still count, floored to min_magnitude."""
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=True, min_magnitude_pct=0.01)
        for _ in range(3):
            t.record_outcome("up", 0.0)  # floored to 0.01
        for _ in range(3):
            t.record_outcome("down", 0.0)
        assert abs(t.direction_agreement("up") - 0.5) < 1e-9

    def test_zero_total_weight_falls_back_to_count(self):
        """If all weights somehow become zero, fall back gracefully."""
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=True, min_magnitude_pct=0.0)
        for _ in range(3):
            t.record_outcome("up", 0.0)
        for _ in range(3):
            t.record_outcome("down", 0.0)
        # All weights are zero → fall back to count-based 3/6 = 0.5
        assert t.direction_agreement("up") == 0.5

    def test_partial_magnitude_weighting(self):
        """4 weak ups (0.1%) + 2 strong downs (1.0%) → down-leaning agreement."""
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=True, min_magnitude_pct=0.0)
        for _ in range(4):
            t.record_outcome("up", 0.1)  # up_weight = 0.4
        for _ in range(2):
            t.record_outcome("down", 1.0)  # down_weight = 2.0
        # total = 2.4, up share = 0.4/2.4 ≈ 0.1667
        agreement = t.direction_agreement("up")
        assert abs(agreement - (0.4 / 2.4)) < 1e-9


class TestCachePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "outcome.json"
        t1 = OutcomeTracker(lookback_windows=6, magnitude_weighted=True)
        t1.record_outcome("up", 0.5)
        t1.record_outcome("down", 1.5)
        t1.record_outcome("up", 0.1)
        t1.save_cache(path)

        t2 = OutcomeTracker(lookback_windows=6, magnitude_weighted=True)
        n_loaded, _ = t2.load_cache(path, staleness_seconds=3600)
        assert n_loaded == 3
        assert t2.n_windows == 3
        # Magnitudes preserved → weighted agreement should match what t1 produced.
        assert t2.direction_agreement("up") == t1.direction_agreement("up")

    def test_load_legacy_string_history(self, tmp_path):
        """Old caches stored plain direction strings — load them at zero magnitude."""
        path = tmp_path / "outcome_legacy.json"
        import time

        path.write_text(json.dumps({"history": ["up", "down", "up"], "saved_at": time.time()}))
        t = OutcomeTracker(lookback_windows=6, magnitude_weighted=True, min_magnitude_pct=0.01)
        n_loaded, _ = t.load_cache(path, staleness_seconds=3600)
        assert n_loaded == 3
        assert t.n_windows == 3
        assert t.summary() == "2U/1D over 3w"

    def test_stale_cache_discarded(self, tmp_path):
        path = tmp_path / "outcome.json"
        # Write a fake 'old' cache
        path.write_text(
            json.dumps({"history": [{"direction": "up", "magnitude_pct": 0.5}], "saved_at": 0.0})
        )
        t = OutcomeTracker(lookback_windows=6)
        n_loaded, age = t.load_cache(path, staleness_seconds=60)
        assert n_loaded == 0
        assert age > 60
