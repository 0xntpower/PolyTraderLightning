"""Tests for the CLOB user-WS trade handler.

Focus: the trade-id dedupe behaviour added to close the latent
multi-trade aggregation gap (post-mortem 2026-04-22 §5.2 follow-up).
A single trade arrives as MATCHED → MINED → CONFIRMED status events
carrying the same ``id``; a resting maker nibbled by two distinct
counterparty takers emits events with two different ``id`` values
that both reference the same maker ``order_id``. The former must
only promote the ``confirmed`` flag; the latter must accumulate
size, size_usd and the aggregate position counters.
"""

from __future__ import annotations

import pytest

from market_data.clob_user_ws import _handle_trade
from market_data.state import MarketState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ORDER_ID = "0xorder-abc"
_TRADE_A = "trade-AAA"
_TRADE_B = "trade-BBB"
_UP_TOKEN = "token-up"
_DOWN_TOKEN = "token-down"


@pytest.fixture
def state() -> MarketState:
    s = MarketState()
    s.up_token_id = _UP_TOKEN
    s.down_token_id = _DOWN_TOKEN
    s.active_order_ids = [_ORDER_ID]
    s.maker_order_ids = {_ORDER_ID}
    return s


def _event(
    *,
    trade_id: str = _TRADE_A,
    status: str = "MATCHED",
    size: float = 0.22,
    price: float = 0.77,
    side: str = "BUY",
    token_id: str = _DOWN_TOKEN,
    order_id: str = _ORDER_ID,
) -> dict:
    return {
        "id": trade_id,
        "asset_id": token_id,
        "size": size,
        "price": price,
        "side": side,
        "status": status,
        "maker_order_id": order_id,
    }


# ---------------------------------------------------------------------------
# Single-trade lifecycle (MATCHED → MINED → CONFIRMED)
# ---------------------------------------------------------------------------


def test_first_matched_event_creates_live_fill(state: MarketState) -> None:
    _handle_trade(_event(status="MATCHED", size=0.22), state)

    assert _ORDER_ID in state.live_fills
    fill = state.live_fills[_ORDER_ID]
    assert fill.size == pytest.approx(0.22)
    assert fill.size_usd == pytest.approx(0.22 * 0.77)
    assert fill.price == pytest.approx(0.77)
    assert fill.confirmed is False
    assert fill.is_maker is True
    assert fill.seen_trade_ids == {_TRADE_A}
    # Position counter moved once.
    assert state.position_down == pytest.approx(0.22)


def test_status_transition_does_not_duplicate_size(state: MarketState) -> None:
    """MATCHED → MINED → CONFIRMED for a single trade must keep size unchanged."""
    _handle_trade(_event(status="MATCHED"), state)
    _handle_trade(_event(status="MINED"), state)
    _handle_trade(_event(status="CONFIRMED"), state)

    fill = state.live_fills[_ORDER_ID]
    # Exactly one accumulation — not three.
    assert fill.size == pytest.approx(0.22)
    assert fill.size_usd == pytest.approx(0.22 * 0.77)
    # Confirmed flag promoted on the terminal event.
    assert fill.confirmed is True
    assert fill.seen_trade_ids == {_TRADE_A}
    # Position counter moved exactly once — not three times.
    assert state.position_down == pytest.approx(0.22)


# ---------------------------------------------------------------------------
# Multi-trade: resting maker nibbled by two distinct counterparties
# ---------------------------------------------------------------------------


def test_second_distinct_trade_accumulates_size(state: MarketState) -> None:
    """Two different trade_ids against the same order accumulate size + USD."""
    _handle_trade(_event(trade_id=_TRADE_A, size=0.22), state)
    _handle_trade(_event(trade_id=_TRADE_B, size=0.50), state)

    fill = state.live_fills[_ORDER_ID]
    assert fill.size == pytest.approx(0.72)
    assert fill.size_usd == pytest.approx(0.72 * 0.77, abs=0.0001)
    assert fill.seen_trade_ids == {_TRADE_A, _TRADE_B}
    # Position counter reflects both trades.
    assert state.position_down == pytest.approx(0.72)


def test_multi_trade_with_status_transitions_interleaved(state: MarketState) -> None:
    """Realistic ordering: A MATCHED, A CONFIRMED, B MATCHED, B CONFIRMED."""
    _handle_trade(_event(trade_id=_TRADE_A, status="MATCHED", size=0.22), state)
    _handle_trade(_event(trade_id=_TRADE_A, status="CONFIRMED", size=0.22), state)
    _handle_trade(_event(trade_id=_TRADE_B, status="MATCHED", size=0.10), state)
    _handle_trade(_event(trade_id=_TRADE_B, status="CONFIRMED", size=0.10), state)

    fill = state.live_fills[_ORDER_ID]
    assert fill.size == pytest.approx(0.32)
    assert fill.size_usd == pytest.approx(0.32 * 0.77, abs=0.0001)
    assert fill.confirmed is True
    assert fill.seen_trade_ids == {_TRADE_A, _TRADE_B}
    assert state.position_down == pytest.approx(0.32)


def test_multi_trade_vwap_when_prices_differ(state: MarketState) -> None:
    """If two distinct trades on the same order land at different prices
    (edge case; normally a maker at P fills only at P), the stored price
    becomes the volume-weighted average across trades.
    """
    _handle_trade(_event(trade_id=_TRADE_A, size=1.0, price=0.80), state)
    _handle_trade(_event(trade_id=_TRADE_B, size=3.0, price=0.78), state)

    fill = state.live_fills[_ORDER_ID]
    expected_vwap = (1.0 * 0.80 + 3.0 * 0.78) / 4.0
    assert fill.price == pytest.approx(expected_vwap, abs=0.0001)
    assert fill.size == pytest.approx(4.0)
    assert fill.size_usd == pytest.approx(1.0 * 0.80 + 3.0 * 0.78, abs=0.0001)


# ---------------------------------------------------------------------------
# Fallback: event with no trade_id falls back to order-level dedupe
# ---------------------------------------------------------------------------


def test_event_without_trade_id_falls_back_to_order_dedupe(state: MarketState) -> None:
    """If the event lacks an ``id`` we can't dedupe at the trade level.

    Safer to treat any subsequent event on the same order as a status
    transition (no size accumulation) than to risk triple-counting a
    MATCHED → MINED → CONFIRMED sequence as three separate fills.
    """
    evt_first = _event(size=0.22)
    evt_first.pop("id")
    _handle_trade(evt_first, state)

    evt_second = _event(size=0.22, status="MINED")
    evt_second.pop("id")
    _handle_trade(evt_second, state)

    fill = state.live_fills[_ORDER_ID]
    # Size not doubled.
    assert fill.size == pytest.approx(0.22)
    assert state.position_down == pytest.approx(0.22)


# ---------------------------------------------------------------------------
# Unknown status / unmatched order
# ---------------------------------------------------------------------------


def test_unknown_status_is_skipped(state: MarketState) -> None:
    _handle_trade(_event(status="CANCELLED"), state)
    assert _ORDER_ID not in state.live_fills
    assert state.position_down == 0.0


def test_trade_for_unknown_order_is_ignored(state: MarketState) -> None:
    """If maker_order_id isn't in active_order_ids and token doesn't match
    either of our tokens, drop the event.
    """
    unrelated_token = "other-market-token"
    evt = _event(order_id="0xsomeone-else", token_id=unrelated_token)
    _handle_trade(evt, state)
    assert _ORDER_ID not in state.live_fills


# ---------------------------------------------------------------------------
# SELL side: position counter decrements
# ---------------------------------------------------------------------------


def test_sell_side_decrements_position_counter(state: MarketState) -> None:
    # Give ourselves a long position first.
    state.position_down = 5.0
    # Add a SELL order to active set so the matcher accepts it.
    state.active_order_ids = ["0xsell-order"]
    state.maker_order_ids = set()

    evt = _event(
        order_id="0xsell-order",
        side="SELL",
        size=1.5,
        price=0.80,
    )
    _handle_trade(evt, state)

    # Position reduced.
    assert state.position_down == pytest.approx(5.0 - 1.5)
    assert state.live_fills["0xsell-order"].side == "SELL"


# ---------------------------------------------------------------------------
# V2 structured fields — maker_orders[] and taker_order_id
# ---------------------------------------------------------------------------
#
# Polymarket CLOB V2 (live since 2026-04-28) emits trade events with
# ``maker_orders[]`` (array of resting-side counterparties) and
# ``taker_order_id`` (aggressor side) instead of (or in addition to) the
# flat ``maker_order_id`` / ``order_id`` fields. The handler's order-ID
# resolver must look at both layers so we can directly match V2 fills
# instead of relying on the token+side fuzzy fallback.


def test_v2_maker_orders_array_resolves_our_order_id(state: MarketState) -> None:
    """Bot's maker order id appears inside V2 ``maker_orders[]``. Resolver
    must find it and treat the trade as ours.
    """
    evt: dict = {
        "id": _TRADE_A,
        "asset_id": _DOWN_TOKEN,
        "size": 0.5,
        "price": 0.80,
        "side": "BUY",
        "status": "MATCHED",
        # V2 schema: no flat maker_order_id — it's nested in maker_orders[].
        "maker_orders": [
            {"order_id": "0xother-counterparty", "matched_amount": "0.10", "price": "0.80"},
            {"order_id": _ORDER_ID, "matched_amount": "0.50", "price": "0.80"},
        ],
        "taker_order_id": "0xtaker-aggressor",
    }
    _handle_trade(evt, state)

    assert _ORDER_ID in state.live_fills
    fill = state.live_fills[_ORDER_ID]
    assert fill.size == pytest.approx(0.5)
    assert fill.is_maker is True
    assert fill.seen_trade_ids == {_TRADE_A}


def test_v2_taker_order_id_resolves_our_fok(state: MarketState) -> None:
    """When the bot places a FOK/FAK and it fills, our order_id appears
    as ``taker_order_id`` on the trade event. Resolver must match.
    """
    # Reset to a taker-style active order set: order is in active_order_ids
    # but NOT in maker_order_ids.
    state.active_order_ids = ["0xtaker-fok"]
    state.maker_order_ids = set()

    evt: dict = {
        "id": _TRADE_A,
        "asset_id": _DOWN_TOKEN,
        "size": 1.0,
        "price": 0.80,
        "side": "BUY",
        "status": "MATCHED",
        "maker_orders": [
            {"order_id": "0xresting-maker", "matched_amount": "1.00", "price": "0.80"},
        ],
        "taker_order_id": "0xtaker-fok",
    }
    _handle_trade(evt, state)

    assert "0xtaker-fok" in state.live_fills
    fill = state.live_fills["0xtaker-fok"]
    assert fill.size == pytest.approx(1.0)
    # is_maker reflects whether the order is in maker_order_ids — should
    # be False here since this was a taker placement.
    assert fill.is_maker is False


def test_v2_unrelated_maker_orders_does_not_match(state: MarketState) -> None:
    """A trade where neither ``maker_orders[]`` nor ``taker_order_id``
    references one of our active orders, AND the token is unrelated, must
    be dropped — not fuzzy-matched against an unrelated active order.
    """
    state.active_order_ids = ["0xour-order"]
    evt: dict = {
        "id": _TRADE_A,
        "asset_id": "0xunrelated-market-token",
        "size": 1.0,
        "price": 0.80,
        "side": "BUY",
        "status": "MATCHED",
        "maker_orders": [
            {"order_id": "0xother-1", "matched_amount": "0.5", "price": "0.80"},
            {"order_id": "0xother-2", "matched_amount": "0.5", "price": "0.80"},
        ],
        "taker_order_id": "0xother-taker",
    }
    _handle_trade(evt, state)

    assert "0xour-order" not in state.live_fills


def test_v2_flat_maker_order_id_still_takes_precedence(state: MarketState) -> None:
    """Some V2 deployments (and older ones) may still emit flat
    ``maker_order_id``. Resolver picks the flat field FIRST so legacy
    behaviour is preserved when both are present.
    """
    evt: dict = {
        "id": _TRADE_A,
        "asset_id": _DOWN_TOKEN,
        "size": 0.5,
        "price": 0.77,
        "side": "BUY",
        "status": "MATCHED",
        "maker_order_id": _ORDER_ID,  # flat — wins
        "maker_orders": [
            {"order_id": "0xshould-not-be-picked", "matched_amount": "0.5", "price": "0.77"},
        ],
    }
    _handle_trade(evt, state)

    assert _ORDER_ID in state.live_fills
    assert "0xshould-not-be-picked" not in state.live_fills
