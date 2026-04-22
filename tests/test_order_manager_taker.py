"""Tests for OrderManager.place_taker_order — the LIVE taker entry path.

Covers the two fixes from the 2026-04-22 live-session post-mortem:

- Fix A (§5.1): ``MarketOrderArgs`` must be constructed with ``side=BUY``;
  omitting it raises ``TypeError`` at py-clob-client construction time and
  crashes every SKIP_MAKER fire before an order is placed.
- Fix B (§5.3): any exit from the post path that does not commit must roll
  back ``risk.tracker.add_exposure`` — including unanticipated exceptions
  that escape the narrow ``except`` blocks.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from py_clob_client.exceptions import (  # type: ignore[import-untyped]
    PolyApiException,
)
from py_clob_client.order_builder.constants import BUY  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from py_clob_client.clob_types import (  # type: ignore[import-untyped]
        MarketOrderArgs,
    )

from execution.order_manager import OrderManager
from market_data.state import MarketState
from risk.fee_tracker import FeeTracker
from risk.position_tracker import PositionTracker
from risk.registry import RiskRegistry

# Non-sensitive test fixtures. Extracted to silence ruff S106 false positives —
# py-clob-client's argument is named ``token_id`` but the value is a market
# asset identifier, not a secret.
_TEST_TOKEN_ID = "token-up-1"
_TEST_TIER = "momentum1"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> PositionTracker:
    return PositionTracker()


@pytest.fixture
def risk(tracker: PositionTracker) -> RiskRegistry:
    registry = RiskRegistry()
    registry.tracker = tracker
    # No checks registered — can_trade() returns True. The exposure-tracking
    # math on the tracker itself is what this test cares about.
    return registry


@pytest.fixture
def state() -> MarketState:
    return MarketState()


@pytest.fixture
def manager(
    risk: RiskRegistry,
    state: MarketState,
) -> OrderManager:
    cfg = MagicMock()
    clob = MagicMock()
    fee_tracker = FeeTracker()
    mgr = OrderManager(cfg=cfg, state=state, clob=clob, risk=risk, fee_tracker=fee_tracker)
    # Skip on-chain balance gate: pretend a refresh already succeeded.
    mgr._cached_balance_usd = 1_000.0
    return mgr


# ---------------------------------------------------------------------------
# Fix A — side=BUY is passed to MarketOrderArgs
# ---------------------------------------------------------------------------


def test_place_taker_order_constructs_market_order_args_with_side_buy(
    manager: OrderManager,
) -> None:
    """Regression guard: taker path must pass ``side=BUY`` to MarketOrderArgs.

    Without this, py-clob-client raises ``TypeError: MarketOrderArgs.__init__()
    missing 1 required positional argument: 'side'`` (post-mortem §5.1).
    """
    captured_args: list[MarketOrderArgs] = []

    def capture_create(args: MarketOrderArgs) -> dict[str, Any]:
        captured_args.append(args)
        return {"signed": True}

    manager.clob.create_market_order.side_effect = capture_create
    manager.clob.post_order.return_value = {"orderID": "0xtest-order-1"}

    order_id = asyncio.run(
        manager.place_taker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.75,
            size_usd=10.0,
            tier=_TEST_TIER,
        ),
    )

    assert order_id == "0xtest-order-1"
    assert len(captured_args) == 1
    args = captured_args[0]
    assert args.side == BUY
    assert args.token_id == _TEST_TOKEN_ID
    assert args.amount == 10.0
    assert args.price == 0.75


def test_place_taker_order_succeeds_end_to_end_when_book_has_liquidity(
    manager: OrderManager,
    tracker: PositionTracker,
    state: MarketState,
) -> None:
    """Full happy path: exposure committed, order id returned, state updated."""
    manager.clob.create_market_order.return_value = {"signed": True}
    manager.clob.post_order.return_value = {"orderID": "0xtest-happy-path"}

    order_id = asyncio.run(
        manager.place_taker_order(
            token_id=_TEST_TOKEN_ID,
            price=0.75,
            size_usd=10.0,
            tier=_TEST_TIER,
        ),
    )

    assert order_id == "0xtest-happy-path"
    assert tracker.window_exposure_usd == pytest.approx(10.0)
    assert "0xtest-happy-path" in state.active_order_ids
    assert manager._cached_balance_usd == pytest.approx(990.0)


# ---------------------------------------------------------------------------
# Fix B — exposure rollback on every non-commit exit path
# ---------------------------------------------------------------------------


def test_exposure_rolled_back_on_unanticipated_typeerror(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """A ``TypeError`` escaping the narrow excepts must still roll back exposure.

    This is the exact shape of the failure observed in the 2026-04-22 session:
    a py-clob-client constructor mismatch raised TypeError *inside* the try
    block but was not caught by ``(OSError, ValueError, KeyError,
    PolyApiException)``. Without the commit-flag finally, the
    ``add_exposure($size_usd)`` call at the top of the method leaked.
    """

    def raise_type_error(_args: MarketOrderArgs) -> dict[str, Any]:
        raise TypeError("simulated py-clob-client arg mismatch")

    manager.clob.create_market_order.side_effect = raise_type_error

    with pytest.raises(TypeError):
        asyncio.run(
            manager.place_taker_order(
                token_id=_TEST_TOKEN_ID,
                price=0.75,
                size_usd=10.0,
                tier=_TEST_TIER,
            ),
        )

    assert tracker.window_exposure_usd == pytest.approx(0.0)


def test_exposure_rolled_back_on_timeout(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """TimeoutError is one of the named excepts; exposure must still roll back."""

    def raise_timeout(_args: MarketOrderArgs) -> dict[str, Any]:
        raise TimeoutError

    manager.clob.create_market_order.side_effect = raise_timeout

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


def test_exposure_rolled_back_on_poly_api_exception(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """Named exception from py-clob-client — exposure rolled back via finally."""

    def raise_poly(_args: MarketOrderArgs) -> dict[str, Any]:
        raise PolyApiException(error_msg="rejected")

    manager.clob.create_market_order.side_effect = raise_poly

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


def test_exposure_rolled_back_when_post_order_returns_no_order_id(
    manager: OrderManager,
    tracker: PositionTracker,
) -> None:
    """Rejection with empty ``orderID`` is not an exception — still a non-commit."""
    manager.clob.create_market_order.return_value = {"signed": True}
    manager.clob.post_order.return_value = {"orderID": ""}

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
