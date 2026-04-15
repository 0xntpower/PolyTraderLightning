"""Tests for Kelly Criterion sizing and regime-adjusted win rate estimation."""

from __future__ import annotations

from collections import deque

import pytest

from strategy.kelly import (
    KELLY_OUTCOME_WINDOW_SIZE,
    KELLY_WIN_RATE_FLOOR,
    AdjustedWinRateResult,
    BankrollTracker,
    conservative_win_rate,
    estimate_adjusted_win_rate,
    kelly_size,
    wilson_lower_bound,
)

# ---------------------------------------------------------------------------
# Wilson lower bound
# ---------------------------------------------------------------------------


class TestWilsonLowerBound:
    def test_perfect_small_sample_shrinks(self):
        """16/16 with z=1.5 → ~87.7%, well below 100%."""
        p = wilson_lower_bound(16, 16)
        assert 0.85 < p < 0.95

    def test_large_sample_converges(self):
        """87/100 → close to observed 87%."""
        p = wilson_lower_bound(87, 100)
        assert 0.80 < p < 0.87

    def test_zero_samples_returns_half(self):
        assert wilson_lower_bound(0, 0) == 0.5

    def test_all_losses(self):
        p = wilson_lower_bound(0, 20)
        assert p < 0.10

    def test_monotonically_increases_with_sample_size(self):
        """Same observed rate, larger n → higher Wilson bound."""
        p_small = wilson_lower_bound(9, 10)
        p_large = wilson_lower_bound(90, 100)
        assert p_large > p_small


class TestConservativeWinRate:
    def test_returns_fraction(self):
        """Result should be a fraction (0-1), not a percentage."""
        p = conservative_win_rate(90.0, 30)
        assert 0.0 < p < 1.0

    def test_max_shrink_cap(self):
        """Wilson correction should be capped by max_shrink_pct."""
        p = conservative_win_rate(90.0, 10, max_shrink_pct=1.0)
        # Should not shrink more than 1 percentage point from 0.90
        assert p >= 0.89


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------


class TestKellySize:
    def test_has_edge_at_fair_price(self):
        """p=0.90, entry=0.85 → clear edge."""
        kr = kelly_size(
            p=0.90,
            entry_price=0.85,
            bankroll=1000.0,
            kelly_fraction=0.25,
            min_bet=1.0,
            max_bet=50.0,
        )
        assert kr.has_edge
        assert kr.raw_kelly > 0
        assert kr.bet_size > 0

    def test_no_edge_at_expensive_price(self):
        """p=0.85, entry=0.90 → no edge (entry > p)."""
        kr = kelly_size(
            p=0.85,
            entry_price=0.90,
            bankroll=1000.0,
            kelly_fraction=0.25,
            min_bet=1.0,
            max_bet=50.0,
        )
        assert not kr.has_edge
        assert kr.raw_kelly < 0
        assert kr.bet_size == 0.0

    def test_fractional_kelly_reduces_bet(self):
        """Quarter Kelly should be ~25% of full Kelly."""
        full = kelly_size(
            p=0.90,
            entry_price=0.85,
            bankroll=1000.0,
            kelly_fraction=1.0,
            min_bet=0.0,
            max_bet=10000.0,
        )
        quarter = kelly_size(
            p=0.90,
            entry_price=0.85,
            bankroll=1000.0,
            kelly_fraction=0.25,
            min_bet=0.0,
            max_bet=10000.0,
        )
        assert quarter.bet_size == pytest.approx(full.bet_size * 0.25, rel=0.01)

    def test_max_bet_cap(self):
        kr = kelly_size(
            p=0.95,
            entry_price=0.50,
            bankroll=100000.0,
            kelly_fraction=1.0,
            min_bet=1.0,
            max_bet=50.0,
        )
        assert kr.bet_size == 50.0

    def test_below_min_bet_returns_zero(self):
        kr = kelly_size(
            p=0.90,
            entry_price=0.85,
            bankroll=10.0,
            kelly_fraction=0.25,
            min_bet=100.0,
            max_bet=1000.0,
        )
        assert kr.bet_size == 0.0
        assert kr.has_edge  # edge exists, just too small

    def test_zero_bankroll(self):
        kr = kelly_size(
            p=0.90, entry_price=0.85, bankroll=0.0, kelly_fraction=0.25, min_bet=1.0, max_bet=50.0
        )
        assert not kr.has_edge
        assert kr.bet_size == 0.0

    def test_entry_price_at_boundary(self):
        """entry_price=1.0 → division by zero guard."""
        kr = kelly_size(
            p=0.90, entry_price=1.0, bankroll=1000.0, kelly_fraction=0.25, min_bet=1.0, max_bet=50.0
        )
        assert not kr.has_edge

    def test_entry_price_zero(self):
        kr = kelly_size(
            p=0.90, entry_price=0.0, bankroll=1000.0, kelly_fraction=0.25, min_bet=1.0, max_bet=50.0
        )
        assert not kr.has_edge

    def test_implied_ev_positive_with_edge(self):
        kr = kelly_size(
            p=0.90,
            entry_price=0.85,
            bankroll=1000.0,
            kelly_fraction=0.25,
            min_bet=1.0,
            max_bet=50.0,
        )
        assert kr.implied_ev > 0

    def test_implied_ev_negative_without_edge(self):
        kr = kelly_size(
            p=0.80,
            entry_price=0.90,
            bankroll=1000.0,
            kelly_fraction=0.25,
            min_bet=1.0,
            max_bet=50.0,
        )
        assert kr.implied_ev < 0

    def test_kelly_formula_correctness(self):
        """Verify: f* = (p - entry) / (1 - entry)."""
        p, entry = 0.90, 0.85
        expected_raw = (p - entry) / (1.0 - entry)
        kr = kelly_size(
            p=p,
            entry_price=entry,
            bankroll=1000.0,
            kelly_fraction=1.0,
            min_bet=0.0,
            max_bet=10000.0,
        )
        assert kr.raw_kelly == pytest.approx(expected_raw, rel=1e-6)


# ---------------------------------------------------------------------------
# Regime-adjusted win rate
# ---------------------------------------------------------------------------


class TestEstimateAdjustedWinRate:
    def _call(self, **overrides) -> AdjustedWinRateResult:
        defaults = {
            "base_win_rate": 0.90,
            "vol_stddev": 0.10,
            "chop_avg_flips": 3.0,
            "outcome_agreement": 0.50,
            "vol_baseline": 0.10,
            "vol_elevated": 0.30,
            "chop_baseline": 3.0,
            "chop_elevated": 10.0,
            "outcome_baseline": 0.50,
            "outcome_elevated": 0.15,
            "max_discount": 0.15,
            "vol_weight": 1.0,
            "chop_weight": 1.0,
            "outcome_weight": 0.8,
            "regime_ready": True,
            "recent_outcomes": deque(maxlen=KELLY_OUTCOME_WINDOW_SIZE),
            "min_outcomes_for_feedback": 10,
        }
        defaults.update(overrides)
        return estimate_adjusted_win_rate(**defaults)

    def test_calm_regime_no_discount(self):
        """At baseline values, no discount should be applied."""
        result = self._call()
        assert result.total_discount == 0.0
        assert result.adjusted_p == pytest.approx(0.90)

    def test_high_vol_discounts(self):
        result = self._call(vol_stddev=0.30)  # at elevated
        assert result.vol_discount > 0
        assert result.total_discount > 0
        assert result.adjusted_p < 0.90

    def test_high_chop_discounts(self):
        result = self._call(chop_avg_flips=10.0)  # at elevated
        assert result.chop_discount > 0
        assert result.adjusted_p < 0.90

    def test_low_outcome_agreement_discounts(self):
        result = self._call(outcome_agreement=0.15)  # at elevated (inverted)
        assert result.outcome_discount > 0
        assert result.adjusted_p < 0.90

    def test_max_weighted_not_additive(self):
        """With both vol and chop elevated, discount should be max(contrib), not sum."""
        both = self._call(vol_stddev=0.30, chop_avg_flips=10.0)
        vol_only = self._call(vol_stddev=0.30)
        chop_only = self._call(chop_avg_flips=10.0)
        # Max-weighted: total should be the larger of the two, not sum
        assert both.total_discount == pytest.approx(
            max(vol_only.total_discount, chop_only.total_discount),
        )

    def test_regime_not_ready_returns_base(self):
        result = self._call(regime_ready=False)
        assert result.adjusted_p == 0.90
        assert result.total_discount == 0.0
        assert not result.regime_ready

    def test_floor_at_coin_flip(self):
        """adjusted_p should never go below 0.50."""
        result = self._call(
            base_win_rate=0.55,
            vol_stddev=0.30,
            chop_avg_flips=10.0,
            max_discount=0.50,  # aggressive discount
        )
        assert result.adjusted_p >= KELLY_WIN_RATE_FLOOR

    def test_feedback_positive_when_winning(self):
        """Recent wins should nudge adjusted_p up."""
        outcomes = deque([1] * 15, maxlen=KELLY_OUTCOME_WINDOW_SIZE)
        result = self._call(recent_outcomes=outcomes, min_outcomes_for_feedback=10)
        assert result.feedback_adjustment > 0

    def test_feedback_negative_when_losing(self):
        """Recent losses should nudge adjusted_p down."""
        outcomes = deque([0] * 15, maxlen=KELLY_OUTCOME_WINDOW_SIZE)
        result = self._call(recent_outcomes=outcomes, min_outcomes_for_feedback=10)
        assert result.feedback_adjustment < 0

    def test_feedback_ignored_below_min_outcomes(self):
        """Feedback should be zero when not enough outcomes."""
        outcomes = deque([0] * 5, maxlen=KELLY_OUTCOME_WINDOW_SIZE)
        result = self._call(recent_outcomes=outcomes, min_outcomes_for_feedback=10)
        assert result.feedback_adjustment == 0.0

    def test_severity_values_between_0_and_1(self):
        result = self._call(vol_stddev=0.20, chop_avg_flips=6.0, outcome_agreement=0.30)
        assert 0.0 <= result.vol_severity <= 1.0
        assert 0.0 <= result.chop_severity <= 1.0
        assert 0.0 <= result.outcome_severity <= 1.0


# ---------------------------------------------------------------------------
# Bankroll tracker
# ---------------------------------------------------------------------------


class TestBankrollTracker:
    def test_win_increases_bankroll(self, tmp_path):
        path = tmp_path / "bankroll.json"
        bt = BankrollTracker(initial_bankroll=1000.0, path=path)
        new = bt.update_win(size=10.0, entry_price=0.85, fee=0.01)
        # shares = 10/0.85 ≈ 11.76, profit = 11.76 * 0.15 - 0.01 ≈ 1.75
        assert new > 1000.0
        assert bt.bankroll == new

    def test_loss_decreases_bankroll(self, tmp_path):
        path = tmp_path / "bankroll.json"
        bt = BankrollTracker(initial_bankroll=1000.0, path=path)
        new = bt.update_loss(size=10.0, entry_price=0.85, fee=0.01)
        # shares = 10/0.85 ≈ 11.76, loss = 11.76 * 0.85 + 0.01 = 10.01
        assert new < 1000.0

    def test_persistence(self, tmp_path):
        path = tmp_path / "bankroll.json"
        bt1 = BankrollTracker(initial_bankroll=1000.0, path=path)
        bt1.update_win(size=10.0, entry_price=0.85)
        saved = bt1.bankroll

        bt2 = BankrollTracker(initial_bankroll=500.0, path=path)  # different initial
        assert bt2.bankroll == pytest.approx(saved, abs=0.01)

    def test_sync_from_api(self, tmp_path):
        path = tmp_path / "bankroll.json"
        bt = BankrollTracker(initial_bankroll=1000.0, path=path)
        drift = bt.sync_from_api(1050.0)
        assert drift == pytest.approx(50.0)
        assert bt.bankroll == 1050.0

    def test_sync_no_drift(self, tmp_path):
        path = tmp_path / "bankroll.json"
        bt = BankrollTracker(initial_bankroll=1000.0, path=path)
        drift = bt.sync_from_api(1000.0)
        assert abs(drift) <= 0.01

    # ------------------------------------------------------------------
    # Corruption handling (Item 4)
    # ------------------------------------------------------------------

    def test_fresh_tracker_not_corrupted(self, tmp_path):
        bt = BankrollTracker(initial_bankroll=1000.0, path=tmp_path / "b.json")
        assert not bt.is_corrupted
        assert bt.corruption_reason == ""

    def test_loss_exceeding_bankroll_clamps_at_zero_and_marks_corrupted(self, tmp_path, caplog):
        bt = BankrollTracker(initial_bankroll=5.0, path=tmp_path / "b.json")
        with caplog.at_level("CRITICAL", logger="strategy.kelly"):
            result = bt.update_loss(size=100.0, entry_price=0.85, fee=0.50)
        assert result == 0.0
        assert bt.bankroll == 0.0
        assert bt.is_corrupted
        assert "update_loss" in bt.corruption_reason
        assert any("BANKROLL CORRUPTED" in rec.message for rec in caplog.records)

    def test_corruption_log_is_idempotent(self, tmp_path, caplog):
        """Second corrupting event must not emit another CRITICAL line."""
        bt = BankrollTracker(initial_bankroll=5.0, path=tmp_path / "b.json")
        with caplog.at_level("CRITICAL", logger="strategy.kelly"):
            bt.update_loss(size=100.0, entry_price=0.85, fee=0.0)
            first_count = sum(1 for r in caplog.records if "BANKROLL CORRUPTED" in r.message)
            bt.update_loss(size=50.0, entry_price=0.85, fee=0.0)
            second_count = sum(1 for r in caplog.records if "BANKROLL CORRUPTED" in r.message)
        assert first_count == 1
        assert second_count == 1

    def test_normal_loss_does_not_mark_corrupted(self, tmp_path):
        bt = BankrollTracker(initial_bankroll=1000.0, path=tmp_path / "b.json")
        bt.update_loss(size=10.0, entry_price=0.85, fee=0.01)
        assert not bt.is_corrupted
        assert bt.bankroll > 980.0

    def test_sync_rejects_negative_balance(self, tmp_path, caplog):
        bt = BankrollTracker(initial_bankroll=1000.0, path=tmp_path / "b.json")
        with caplog.at_level("CRITICAL", logger="strategy.kelly"):
            drift = bt.sync_from_api(-50.0)
        assert drift == 0.0
        assert bt.bankroll == 1000.0  # unchanged
        assert bt.is_corrupted
        assert "negative" in bt.corruption_reason.lower()

    def test_sync_rejects_10x_jump(self, tmp_path, caplog):
        """10x+ jump suggests API glitch — reject, flag corrupted."""
        bt = BankrollTracker(initial_bankroll=1000.0, path=tmp_path / "b.json")
        with caplog.at_level("CRITICAL", logger="strategy.kelly"):
            drift = bt.sync_from_api(20000.0)
        assert drift == 0.0
        assert bt.bankroll == 1000.0
        assert bt.is_corrupted

    def test_sync_accepts_small_deposit_within_ratio(self, tmp_path):
        """Reasonable deposit (e.g. 50% top-up) must still apply."""
        bt = BankrollTracker(initial_bankroll=1000.0, path=tmp_path / "b.json")
        bt.sync_from_api(1500.0)
        assert bt.bankroll == 1500.0
        assert not bt.is_corrupted

    def test_sync_does_not_apply_ratio_guard_when_bankroll_tiny(self, tmp_path):
        """When bankroll is $1 or less, initial top-up to any value is fine."""
        bt = BankrollTracker(initial_bankroll=0.50, path=tmp_path / "b.json")
        bt.sync_from_api(1000.0)
        assert bt.bankroll == 1000.0
        assert not bt.is_corrupted

    def test_reset_clears_corruption(self, tmp_path):
        bt = BankrollTracker(initial_bankroll=5.0, path=tmp_path / "b.json")
        bt.update_loss(size=100.0, entry_price=0.85, fee=0.0)
        assert bt.is_corrupted

        bt.reset(new_bankroll=1000.0)
        assert not bt.is_corrupted
        assert bt.corruption_reason == ""
        assert bt.bankroll == 1000.0
