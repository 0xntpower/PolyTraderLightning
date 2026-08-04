"""Tests for CLOB user WebSocket fill tracking.

Validates that status transitions (MATCHED→MINED→CONFIRMED) for the same
trade do NOT double-count fill size, and that fill matching works correctly.
"""

from __future__ import annotations

import pytest

from market_data.clob_user_ws import _handle_trade
from market_data.state import MarketState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    up_token: str = "tok_up",
    down_token: str = "tok_down",
    active_orders: list[str] | None = None,
) -> MarketState:
    s = MarketState()
    s.up_token_id = up_token
    s.down_token_id = down_token
    if active_orders:
        s.active_order_ids = list(active_orders)
    return s


def _trade_event(
    *,
    token_id: str = "tok_up",
    size: float = 10.0,
    price: float = 0.55,
    side: str = "BUY",
    status: str = "MATCHED",
    maker_order_id: str = "order_abc",
) -> dict:
    return {
        "asset_id": token_id,
        "size": str(size),
        "price": str(price),
        "side": side,
        "status": status,
        "maker_order_id": maker_order_id,
    }


# ======================================================================
# Status transition tests (the critical double-count bug fix)
# ======================================================================


class TestStatusTransitions:
    """Ensure MATCHED→MINED→CONFIRMED doesn't inflate fill size."""

    def test_single_matched_creates_fill(self):
        state = _make_state(active_orders=["order_abc"])
        _handle_trade(_trade_event(status="MATCHED"), state)

        assert "order_abc" in state.live_fills
        fill = state.live_fills["order_abc"]
        assert fill.size == pytest.approx(10.0)
        assert fill.size_usd == pytest.approx(5.5)  # 0.55 * 10
        assert fill.price == pytest.approx(0.55)
        assert fill.confirmed is False

    def test_mined_does_not_accumulate(self):
        state = _make_state(active_orders=["order_abc"])
        _handle_trade(_trade_event(status="MATCHED"), state)
        _handle_trade(_trade_event(status="MINED"), state)

        fill = state.live_fills["order_abc"]
        assert fill.size == pytest.approx(10.0), "MINED should not add size"
        assert fill.size_usd == pytest.approx(5.5), "MINED should not add USD"
        assert fill.confirmed is False

    def test_confirmed_does_not_accumulate(self):
        state = _make_state(active_orders=["order_abc"])
        _handle_trade(_trade_event(status="MATCHED"), state)
        _handle_trade(_trade_event(status="CONFIRMED"), state)

        fill = state.live_fills["order_abc"]
        assert fill.size == pytest.approx(10.0), "CONFIRMED should not add size"
        assert fill.size_usd == pytest.approx(5.5), "CONFIRMED should not add USD"
        assert fill.confirmed is True

    def test_full_lifecycle_no_inflation(self):
        """MATCHED → MINED → CONFIRMED = exactly 1x fill, not 3x."""
        state = _make_state(active_orders=["order_abc"])

        _handle_trade(_trade_event(status="MATCHED"), state)
        _handle_trade(_trade_event(status="MINED"), state)
        _handle_trade(_trade_event(status="CONFIRMED"), state)

        fill = state.live_fills["order_abc"]
        assert fill.size == pytest.approx(10.0)
        assert fill.size_usd == pytest.approx(5.5)
        assert fill.price == pytest.approx(0.55)
        assert fill.confirmed is True

    def test_duplicate_matched_after_confirmed_ignored(self):
        state = _make_state(active_orders=["order_abc"])
        _handle_trade(_trade_event(status="MATCHED"), state)
        _handle_trade(_trade_event(status="CONFIRMED"), state)
        _handle_trade(_trade_event(status="MATCHED"), state)  # duplicate

        fill = state.live_fills["order_abc"]
        assert fill.size == pytest.approx(10.0)


# ======================================================================
# Position tracking
# ======================================================================


class TestPositionTracking:
    """Position accumulation for BUY/SELL events."""

    def test_buy_up_token_increments_position(self):
        state = _make_state(active_orders=["order_abc"])
        _handle_trade(_trade_event(token_id="tok_up", side="BUY", status="MATCHED"), state)
        assert state.position_up == pytest.approx(10.0)
        assert state.position_down == pytest.approx(0.0)

    def test_buy_down_token_increments_position(self):
        state = _make_state(active_orders=["order_abc"])
        _handle_trade(
            _trade_event(token_id="tok_down", side="BUY", status="MATCHED"),
            state,
        )
        assert state.position_down == pytest.approx(10.0)
        assert state.position_up == pytest.approx(0.0)

    def test_sell_decrements_position(self):
        state = _make_state(active_orders=["order_abc"])
        state.position_up = 20.0
        _handle_trade(
            _trade_event(token_id="tok_up", side="SELL", status="MATCHED"),
            state,
        )
        assert state.position_up == pytest.approx(10.0)

    def test_unmatched_fill_does_not_track(self):
        """Fill for unknown order is ignored (no active orders)."""
        state = _make_state(active_orders=[])
        _handle_trade(_trade_event(status="MATCHED"), state)
        assert len(state.live_fills) == 0


# ======================================================================
# Fill matching
# ======================================================================


class TestFillMatching:
    """Method 1 (order_id) and Method 2 (token+side fallback)."""

    def test_method1_direct_order_id(self):
        state = _make_state(active_orders=["order_abc"])
        _handle_trade(
            _trade_event(maker_order_id="order_abc", status="MATCHED"),
            state,
        )
        assert "order_abc" in state.live_fills

    def test_method2_fallback_buy_no_order_id(self):
        state = _make_state(active_orders=["order_xyz"])
        _handle_trade(
            _trade_event(maker_order_id="", side="BUY", status="MATCHED"),
            state,
        )
        # Should match via Method 2
        assert "order_xyz" in state.live_fills

    def test_method2_no_match_for_unknown_token(self):
        state = _make_state(active_orders=["order_xyz"])
        _handle_trade(
            _trade_event(
                token_id="unknown_token",
                maker_order_id="",
                side="BUY",
                status="MATCHED",
            ),
            state,
        )
        assert len(state.live_fills) == 0

    def test_unexpected_status_ignored(self):
        state = _make_state(active_orders=["order_abc"])
        _handle_trade(
            _trade_event(status="CANCELLED"),
            state,
        )
        assert len(state.live_fills) == 0
