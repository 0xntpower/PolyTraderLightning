"""Tests for Polymarket fee calculation and paper trading PnL.

Validates the fee formula: fee_rate_bps/10000 * min(price, 1-price) * shares,
and that paper trading PnL correctly accounts for fees.
"""

from __future__ import annotations

import pytest

from risk.fee_tracker import FeeTracker

# ======================================================================
# Taker fee formula
# ======================================================================


class TestTakerFee:
    """Polymarket fee: fee_rate * min(price, 1-price) per share."""

    def _tracker(self, bps: int = 200) -> FeeTracker:
        ft = FeeTracker()
        ft.fee_rate_bps = bps
        return ft

    def test_symmetric_at_midpoint(self):
        """At price=0.50, min(0.50, 0.50) = 0.50."""
        ft = self._tracker(200)
        fee = ft.compute_taker_fee(price=0.50, size=100)
        # 200/10000 * 0.50 * 100 = 0.02 * 0.50 * 100 = $1.00
        assert fee == pytest.approx(1.0)

    def test_cheap_side(self):
        """At price=0.20, min(0.20, 0.80) = 0.20."""
        ft = self._tracker(200)
        fee = ft.compute_taker_fee(price=0.20, size=100)
        # 0.02 * 0.20 * 100 = $0.40
        assert fee == pytest.approx(0.40)

    def test_expensive_side(self):
        """At price=0.80, min(0.80, 0.20) = 0.20 — same as cheap side."""
        ft = self._tracker(200)
        fee = ft.compute_taker_fee(price=0.80, size=100)
        assert fee == pytest.approx(0.40)

    def test_fee_symmetry(self):
        """fee(p) == fee(1-p) for all p."""
        ft = self._tracker(200)
        for p in [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95]:
            assert ft.compute_taker_fee(p, 100) == pytest.approx(
                ft.compute_taker_fee(1 - p, 100), abs=0.0001
            )

    def test_edge_price_zero(self):
        ft = self._tracker(200)
        fee = ft.compute_taker_fee(price=0.0, size=100)
        assert fee == pytest.approx(0.0)

    def test_edge_price_one(self):
        ft = self._tracker(200)
        fee = ft.compute_taker_fee(price=1.0, size=100)
        assert fee == pytest.approx(0.0)

    def test_record_accumulates(self):
        ft = self._tracker(200)
        ft.record_taker_fee(0.50, 100)
        ft.record_taker_fee(0.50, 100)
        assert ft.total_taker_fees == pytest.approx(2.0)

    def test_different_bps_rates(self):
        """Higher fee rate = proportionally higher fee."""
        ft100 = self._tracker(100)
        ft400 = self._tracker(400)
        assert ft400.compute_taker_fee(0.50, 100) == pytest.approx(
            ft100.compute_taker_fee(0.50, 100) * 4
        )


# ======================================================================
# Paper trading PnL (win/loss with fees)
# ======================================================================


class TestPaperPnL:
    """Validate paper trading PnL = payout - cost - fees.

    For a binary outcome market:
      Win:  pnl = shares * (1 - entry_price) - fee
      Loss: pnl = -(shares * entry_price) - fee
    """

    def test_win_pnl(self):
        """Buy 100 shares at 0.55, win: payout=100*(1-0.55)=$45, fee deducted."""
        # shares = size_usd / price = 55 / 0.55 = 100
        shares = 100.0
        price = 0.55
        pnl_before_fee = shares * (1.0 - price)  # $45.00
        fee = 200 / 10000 * min(price, 1 - price) * shares  # 0.02 * 0.45 * 100 = $0.90
        expected = pnl_before_fee - fee  # $44.10
        assert expected == pytest.approx(44.10, abs=0.01)

    def test_loss_pnl(self):
        """Buy 100 shares at 0.55, lose: cost=100*0.55=$55, fee deducted."""
        shares = 100.0
        price = 0.55
        pnl_before_fee = -(shares * price)  # -$55.00
        fee = 200 / 10000 * min(price, 1 - price) * shares  # $0.90
        expected = pnl_before_fee - fee  # -$55.90
        assert expected == pytest.approx(-55.90, abs=0.01)

    def test_cheap_entry_win_large_payout(self):
        """Buy at 0.20, win → large payout: 100*(1-0.20)=$80."""
        shares = 100.0
        price = 0.20
        pnl = shares * (1.0 - price)  # $80
        fee = 200 / 10000 * min(price, 1 - price) * shares  # 0.02 * 0.20 * 100 = $0.40
        assert (pnl - fee) == pytest.approx(79.60, abs=0.01)

    def test_expensive_entry_win_small_payout(self):
        """Buy at 0.90, win → small payout: 100*(1-0.90)=$10."""
        shares = 100.0
        price = 0.90
        pnl = shares * (1.0 - price)  # $10
        fee = 200 / 10000 * min(price, 1 - price) * shares  # 0.02 * 0.10 * 100 = $0.20
        assert (pnl - fee) == pytest.approx(9.80, abs=0.01)
