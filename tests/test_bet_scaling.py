"""Tests for adaptive bet scaling (SPRT confidence + age taper).

Validates the pure functions that map SPRT state and signal age to a
bet size multiplier.
"""

from __future__ import annotations

import pytest

from shared.risk import age_taper, compute_bet_scale, llr_confidence

# ======================================================================
# llr_confidence
# ======================================================================


class TestLLRConfidence:
    def test_at_alive_boundary(self):
        """LLR at alive boundary → full confidence (1.0)."""
        assert llr_confidence(llr=-2.25, boundary_alive=-2.25, boundary_dead=2.89) == 1.0

    def test_below_alive_boundary(self):
        """LLR below alive boundary → still 1.0."""
        assert llr_confidence(llr=-5.0, boundary_alive=-2.25, boundary_dead=2.89) == 1.0

    def test_at_dead_boundary(self):
        """LLR at dead boundary → no confidence (0.0)."""
        assert llr_confidence(llr=2.89, boundary_alive=-2.25, boundary_dead=2.89) == 0.0

    def test_above_dead_boundary(self):
        """LLR above dead boundary → 0.0."""
        assert llr_confidence(llr=5.0, boundary_alive=-2.25, boundary_dead=2.89) == 0.0

    def test_midpoint(self):
        """LLR at midpoint → ~0.5."""
        mid = (-2.25 + 2.89) / 2  # ≈ 0.32
        conf = llr_confidence(llr=mid, boundary_alive=-2.25, boundary_dead=2.89)
        assert conf == pytest.approx(0.5, abs=0.01)

    def test_zero_llr(self):
        """LLR=0 (fresh start) → between 0 and 1."""
        conf = llr_confidence(llr=0.0, boundary_alive=-2.25, boundary_dead=2.89)
        assert 0.0 < conf < 1.0

    def test_equal_boundaries_returns_one(self):
        assert llr_confidence(llr=0.0, boundary_alive=1.0, boundary_dead=1.0) == 1.0


# ======================================================================
# age_taper
# ======================================================================


class TestAgeTaper:
    def test_before_taper_start(self):
        assert age_taper(50, taper_start=200, taper_end=400, floor=0.5) == 1.0

    def test_at_taper_start(self):
        assert age_taper(200, taper_start=200, taper_end=400, floor=0.5) == 1.0

    def test_midpoint(self):
        result = age_taper(300, taper_start=200, taper_end=400, floor=0.5)
        assert result == pytest.approx(0.75, abs=0.01)

    def test_at_taper_end(self):
        assert age_taper(400, taper_start=200, taper_end=400, floor=0.5) == 0.5

    def test_beyond_taper_end(self):
        assert age_taper(1000, taper_start=200, taper_end=400, floor=0.5) == 0.5

    def test_degenerate_start_eq_end(self):
        assert age_taper(100, taper_start=200, taper_end=200, floor=0.5) == 1.0
        assert age_taper(300, taper_start=200, taper_end=200, floor=0.5) == 0.5


# ======================================================================
# compute_bet_scale, combined
# ======================================================================


class TestComputeBetScale:
    def test_dormant_sprt_when_not_stale(self):
        """When signal is NOT stale, SPRT is dormant → scale depends only on age."""
        scale = compute_bet_scale(
            llr=5.0,  # would be DEAD if active
            boundary_alive=-2.25,
            boundary_dead=2.89,
            signal_age_windows=50,
            signal_is_stale=False,
            taper_start=200,
            taper_end=400,
        )
        # SPRT dormant (1.0), age < taper_start (1.0) → 1.0
        assert scale == 1.0

    def test_active_sprt_when_stale(self):
        """When signal IS stale, SPRT activates and can reduce scale."""
        scale = compute_bet_scale(
            llr=2.89,  # at DEAD boundary → confidence=0.0
            boundary_alive=-2.25,
            boundary_dead=2.89,
            signal_age_windows=50,
            signal_is_stale=True,
        )
        # SPRT conf=0.0, age=1.0 → scale=0.0, clamped to min_total_scale
        assert scale == pytest.approx(0.10)

    def test_combined_sprt_and_age(self):
        """Both SPRT and age taper apply multiplicatively."""
        scale = compute_bet_scale(
            llr=0.0,  # midway → conf ≈ 0.56
            boundary_alive=-2.25,
            boundary_dead=2.89,
            signal_age_windows=300,  # midway → age ≈ 0.75
            signal_is_stale=True,
            taper_start=200,
            taper_end=400,
            age_floor=0.5,
        )
        expected_sprt = (2.89 - 0.0) / (2.89 - (-2.25))
        expected_age = 0.75
        expected = max(0.10, expected_sprt * expected_age)
        assert scale == pytest.approx(expected, abs=0.01)

    def test_min_total_scale_floor(self):
        """Scale never drops below min_total_scale."""
        scale = compute_bet_scale(
            llr=100.0,
            boundary_alive=-2.25,
            boundary_dead=2.89,
            signal_age_windows=10000,
            signal_is_stale=True,
            min_total_scale=0.10,
        )
        assert scale == pytest.approx(0.10)

    def test_fresh_signal_full_scale(self):
        """Fresh signal, not stale, young → 1.0."""
        scale = compute_bet_scale(
            llr=0.0,
            boundary_alive=-2.25,
            boundary_dead=2.89,
            signal_age_windows=0,
            signal_is_stale=False,
        )
        assert scale == 1.0
