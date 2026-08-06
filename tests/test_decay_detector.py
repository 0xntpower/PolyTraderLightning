"""Tests for SPRT-based signal decay detection.

Validates LLR accumulation, verdict boundaries, reset behavior, and
rolling win rate calculation.
"""

from __future__ import annotations

import math

import pytest

from shared.decay_detector import DecayDetector

# ======================================================================
# Construction and boundary math
# ======================================================================


class TestBoundaries:
    def test_boundaries_computed_correctly(self):
        """Verify Wald boundaries: dead = ln((1-beta)/alpha), alive = ln(beta/(1-alpha))."""
        d = DecayDetector("sig1", p_alive=0.65, p_dead=0.50, alpha=0.05, beta=0.10)
        expected_dead = math.log(0.90 / 0.05)  # ln(18) ≈ 2.89
        expected_alive = math.log(0.10 / 0.95)  # ln(0.1053) ≈ -2.25
        assert d.state.boundary_dead == pytest.approx(expected_dead, abs=0.01)
        assert d.state.boundary_alive == pytest.approx(expected_alive, abs=0.01)

    def test_p_dead_clamped_below_p_alive(self):
        """p_dead >= p_alive should be corrected to p_alive - 0.05."""
        d = DecayDetector("sig1", p_alive=0.60, p_dead=0.70)
        assert d.state.p_dead < d.state.p_alive
        assert d.state.p_dead == pytest.approx(0.55, abs=0.01)

    def test_p_dead_floor(self):
        """p_dead can't go below 0.01."""
        d = DecayDetector("sig1", p_alive=0.04, p_dead=0.03)
        # p_alive - 0.05 = -0.01, clamped to 0.01
        assert d.state.p_dead >= 0.01


# ======================================================================
# LLR accumulation and verdicts
# ======================================================================


class TestLLRAccumulation:
    def test_initial_state_inconclusive(self):
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        assert d.state.verdict == "INCONCLUSIVE"
        assert d.state.llr == 0.0
        assert d.state.n_trades == 0

    def test_wins_push_llr_negative(self):
        """Wins are evidence toward ALIVE (negative LLR)."""
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        for _ in range(5):
            d.update(won=True)
        assert d.state.llr < 0
        assert d.state.n_wins == 5

    def test_losses_push_llr_positive(self):
        """Losses are evidence toward DEAD (positive LLR)."""
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        for _ in range(5):
            d.update(won=False)
        assert d.state.llr > 0

    def test_enough_losses_trigger_dead(self):
        """Sustained losses should eventually cross the DEAD boundary."""
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50, alpha=0.05, beta=0.10)
        for _ in range(100):
            state = d.update(won=False)
            if state.verdict == "DEAD":
                break
        assert state.verdict == "DEAD"
        assert state.llr >= state.boundary_dead

    def test_enough_wins_trigger_alive(self):
        """Sustained wins should eventually cross the ALIVE boundary."""
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50, alpha=0.05, beta=0.10)
        for _ in range(100):
            state = d.update(won=True)
            if state.verdict == "ALIVE":
                break
        assert state.verdict == "ALIVE"
        assert state.llr <= state.boundary_alive

    def test_mixed_results_stay_inconclusive(self):
        """Alternating W/L near midpoint between p_dead and p_alive stays INCONCLUSIVE."""
        d = DecayDetector("sig1", p_alive=0.65, p_dead=0.50, alpha=0.05, beta=0.10)
        # Feed a ~55% WR sequence (between p_dead=0.50 and p_alive=0.65)
        # This is ambiguous evidence — shouldn't reach a conclusion quickly
        pattern = [True, True, True, True, True, True, False, False, False, False]  # 60%
        for won in pattern:
            d.update(won=won)
        assert d.state.verdict == "INCONCLUSIVE"


# ======================================================================
# Rolling win rate
# ======================================================================


class TestRollingWinRate:
    def test_rolling_wr_all_wins(self):
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        for _ in range(10):
            d.update(won=True)
        assert d.state.rolling_win_rate == pytest.approx(1.0)

    def test_rolling_wr_all_losses(self):
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        for _ in range(10):
            d.update(won=False)
        assert d.state.rolling_win_rate == pytest.approx(0.0)

    def test_rolling_wr_mixed(self):
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        for _ in range(7):
            d.update(won=True)
        for _ in range(3):
            d.update(won=False)
        assert d.state.rolling_win_rate == pytest.approx(0.7, abs=0.01)

    def test_rolling_window_bounded(self):
        """Only last 20 trades are in the rolling window."""
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        # Feed 20 wins, then 20 losses — rolling should be 0%
        for _ in range(20):
            d.update(won=True)
        for _ in range(20):
            d.update(won=False)
        assert d.state.rolling_win_rate == pytest.approx(0.0)


# ======================================================================
# Reset
# ======================================================================


class TestReset:
    def test_reset_clears_llr_and_counts(self):
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        for _ in range(10):
            d.update(won=True)
        d.reset()
        s = d.state
        assert s.llr == 0.0
        assert s.n_trades == 0
        assert s.n_wins == 0
        assert s.verdict == "INCONCLUSIVE"
        assert s.rolling_win_rate == 0.0

    def test_reset_with_new_params(self):
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        d.reset(p_alive=0.80, p_dead=0.60)
        assert d.state.p_alive == 0.80
        assert d.state.p_dead == 0.60

    def test_reset_preserves_old_params_if_none(self):
        d = DecayDetector("sig1", p_alive=0.70, p_dead=0.50)
        d.reset()
        assert d.state.p_alive == 0.70
        assert d.state.p_dead == 0.50
