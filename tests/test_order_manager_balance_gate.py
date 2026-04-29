"""Tests for the on-chain balance pre-trade gate in OrderManager.

This gate is what kept the bot from blind-firing during the 2026-04-28
Polymarket V2 / pUSD migration cutover: ``refresh_balance`` returned
``$0.00`` for the entire post-cutover session and the gate correctly
blocked all three fires (post-mortem ``current_tmp_session/post_mortem.md``
section 2 mechanism table). Before this file the gate had zero unit-test
coverage; the V2 migration is the right time to lock its behaviour in
so a future refactor cannot quietly regress it.

Three properties:

1. ``_cached_balance_usd is None`` (refresh never succeeded) -> both
   ``place_taker_order`` and ``place_maker_order`` return ``None`` with
   the "balance unknown" warning, and exposure is **not** booked.
2. ``_cached_balance_usd < size_usd`` -> both return ``None`` with the
   "insufficient balance" warning, and exposure is **not** booked.
3. After a successful place, ``_cached_balance_usd`` is debited by
   ``size_usd``; failed places do NOT debit.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from execution.order_manager import OrderManager
from market_data.state import MarketState
from risk.fee_tracker import FeeTracker
from risk.position_tracker import PositionTracker
from risk.registry import RiskRegistry

# Same non-sensitive fixtures as test_order_manager_taker.py. Argument is
# named ``token_id`` but the value is a market asset identifier, not a secret.
_TEST_TOKEN_ID = "token-up-1"
_TEST_TIER = "momentum1"


@pytest.fixture
def tracker() -> PositionTracker:
    return PositionTracker()


@pytest.fixture
def risk(tracker: PositionTracker) -> RiskRegistry:
    registry = RiskRegistry()
    registry.tracker = tracker
    return registry


@pytest.fixture
def manager(risk: RiskRegistry) -> OrderManager:
    cfg = MagicMock()
    clob = MagicMock()
    state = MarketState()
    fee_tracker = FeeTracker()
    return OrderManager(cfg=cfg, state=state, clob=clob, risk=risk, fee_tracker=fee_tracker)


def test_taker_blocked_when_balance_is_none(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """The bot must not place taker orders before refresh_balance succeeds.

    On a clean startup ``_cached_balance_usd`` is ``None`` until the first
    ``refresh_balance`` round-trip completes. A taker fire in that window
    would commit on-chain against an unknown balance.
    """
    assert manager._cached_balance_usd is None
    order_id = asyncio.run(
        manager.place_taker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.75,
            size_usd=10.0,
            tier=_TEST_TIER,
        ),
    )
    assert order_id is None
    assert tracker.window_exposure_usd == pytest.approx(0.0)
    manager.clob.create_market_order.assert_not_called()
    manager.clob.post_order.assert_not_called()


def test_maker_blocked_when_balance_is_none(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """Same gate must apply to the maker entry path."""
    assert manager._cached_balance_usd is None
    manager.state.best_ask_up = 0.0
    order_id = asyncio.run(
        manager.place_maker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.74,
            size_usd=10.0,
            tier=_TEST_TIER,
        ),
    )
    assert order_id is None
    assert tracker.window_exposure_usd == pytest.approx(0.0)
    manager.clob.create_and_post_order.assert_not_called()


def test_taker_blocked_when_balance_below_order_size(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """Reproduces the v3.7 live-session character: cached balance $0.00,
    every fire blocked. Lock in that the gate fires *before* exposure is
    booked, so v3.6.2's ``add_exposure`` leak class cannot recur here.
    """
    manager._cached_balance_usd = 0.0
    order_id = asyncio.run(
        manager.place_taker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.75,
            size_usd=1.40,
            tier=_TEST_TIER,
        ),
    )
    assert order_id is None
    assert tracker.window_exposure_usd == pytest.approx(0.0)
    manager.clob.create_market_order.assert_not_called()


def test_taker_blocked_when_balance_just_under_size(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """Edge: cached balance = size - epsilon. Strict-greater (>), not >=."""
    manager._cached_balance_usd = 9.99
    order_id = asyncio.run(
        manager.place_taker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.75,
            size_usd=10.00,
            tier=_TEST_TIER,
        ),
    )
    assert order_id is None
    assert tracker.window_exposure_usd == pytest.approx(0.0)


def test_taker_allowed_when_balance_exactly_size(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """Boundary case: balance == size. Strict-greater means equality
    still allows the trade. Pin so a refactor to >= is caught.
    """
    manager._cached_balance_usd = 10.00
    manager.clob.create_market_order.return_value = {"signed": True}
    manager.clob.post_order.return_value = {"orderID": "0xboundary"}
    order_id = asyncio.run(
        manager.place_taker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.75,
            size_usd=10.00,
            tier=_TEST_TIER,
        ),
    )
    assert order_id == "0xboundary"
    assert tracker.window_exposure_usd == pytest.approx(10.00)


def test_maker_blocked_when_balance_below_order_size(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    manager._cached_balance_usd = 0.0
    manager.state.best_ask_up = 0.0
    order_id = asyncio.run(
        manager.place_maker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.74,
            size_usd=1.40,
            tier=_TEST_TIER,
        ),
    )
    assert order_id is None
    assert tracker.window_exposure_usd == pytest.approx(0.0)
    manager.clob.create_and_post_order.assert_not_called()


def test_taker_place_debits_cached_balance_by_size(manager: OrderManager) -> None:
    """A successful taker place must debit ``_cached_balance_usd`` by
    exactly ``size_usd`` so the next pre-trade check sees the reduced
    headroom (without waiting for the next ``refresh_balance`` round).
    """
    manager._cached_balance_usd = 100.0
    manager.clob.create_market_order.return_value = {"signed": True}
    manager.clob.post_order.return_value = {"orderID": "0xdebit-test"}
    asyncio.run(
        manager.place_taker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.75,
            size_usd=10.0,
            tier=_TEST_TIER,
        ),
    )
    assert manager._cached_balance_usd == pytest.approx(90.0)


def test_maker_place_debits_cached_balance_by_size(manager: OrderManager) -> None:
    manager._cached_balance_usd = 100.0
    manager.state.best_ask_up = 0.0
    manager.clob.create_and_post_order.return_value = {"orderID": "0xmaker-debit"}
    asyncio.run(
        manager.place_maker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.74,
            size_usd=10.0,
            tier=_TEST_TIER,
        ),
    )
    assert manager._cached_balance_usd == pytest.approx(90.0)


def test_failed_place_does_not_debit_cached_balance(manager: OrderManager) -> None:
    """If the SDK call fails (returned no orderID), the cached balance
    must not be debited. Otherwise repeated failed attempts could drive
    the cache negative and lock the bot out of subsequent fires until
    the next ``refresh_balance``.
    """
    manager._cached_balance_usd = 100.0
    manager.clob.create_market_order.return_value = {"signed": True}
    manager.clob.post_order.return_value = {"orderID": ""}  # rejected
    order_id = asyncio.run(
        manager.place_taker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.75,
            size_usd=10.0,
            tier=_TEST_TIER,
        ),
    )
    assert order_id is None
    assert manager._cached_balance_usd == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Maker minimum-size pre-check (closes v3.6.2 §5.2)
# ---------------------------------------------------------------------------


def test_maker_skipped_when_size_below_clob_minimum(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """Maker orders below 5 shares must be pre-empted before the CLOB
    round-trip. Polymarket rejects them with HTTP 400 "Size (X) lower than
    the minimum: 5", which costs a circuit-breaker increment and a wasted
    API call. v3.6.2 lived with this for two sessions before the fix
    landed (post-mortem 2026-04-23 §5.2).

    At a $1.40 bet with price 0.75: size_shares = 1.87 < 5 -> skip.
    The caller (momentum_signal._fire_with_retry) then drops through to
    the taker path.
    """
    manager._cached_balance_usd = 100.0
    manager.state.best_ask_up = 0.0  # let post-only check pass

    order_id = asyncio.run(
        manager.place_maker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.75,
            size_usd=1.40,
            tier=_TEST_TIER,
        ),
    )

    assert order_id is None
    # Critical: the pre-check fires BEFORE add_exposure so no risk-tracker
    # cleanup is needed and the circuit breaker is NOT incremented.
    assert tracker.window_exposure_usd == pytest.approx(0.0)
    manager.clob.create_and_post_order.assert_not_called()
    # Cached balance untouched — pre-check returns before debit.
    assert manager._cached_balance_usd == pytest.approx(100.0)


def test_maker_allowed_when_size_meets_clob_minimum(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """Boundary: size_shares == 5 IS allowed (strict <). Bet $4 at price
    0.80 -> 5 shares exactly. The pre-check uses ``<``, so 5 sh passes.
    Pin so a refactor to ``<=`` is caught.
    """
    manager._cached_balance_usd = 100.0
    manager.state.best_ask_up = 0.0
    manager.clob.create_and_post_order.return_value = {"orderID": "0xmin-boundary"}

    order_id = asyncio.run(
        manager.place_maker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.80,
            size_usd=4.00,
            tier=_TEST_TIER,
        ),
    )

    assert order_id == "0xmin-boundary"
    assert tracker.window_exposure_usd == pytest.approx(4.00)
    manager.clob.create_and_post_order.assert_called_once()
