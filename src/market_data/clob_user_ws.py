"""WS4: CLOB user channel — our order fills and status updates."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, TypedDict

import orjson

from market_data.state import LiveFill, MarketState

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    from market_data.latency_tracker import LatencyTracker


class _OrderEventMsg(TypedDict, total=False):
    id: str
    order_id: str
    type: str  # PLACEMENT, UPDATE, CANCELLATION
    event_type: str


class _TradeEventMsg(TypedDict, total=False):
    id: str  # unique trade ID; distinguishes status transitions from new partials
    asset_id: str
    size: float | str
    price: float | str
    side: str
    status: str
    maker_order_id: str
    makerOrderId: str
    order_id: str
    orderId: str
    event_type: str


log = logging.getLogger(__name__)


async def handle_clob_user(
    ws: ClientConnection,
    state: MarketState,
    api_creds: dict[str, str],
    latency: LatencyTracker | None = None,
) -> None:
    auth_msg = orjson.dumps(
        {
            "type": "user",
            "auth": api_creds,
        }
    ).decode()
    await ws.send(auth_msg)
    log.info("CLOB user channel subscribed")

    _last_msg_ts = 0.0

    async def _ping_loop() -> None:
        """CLOB WS requires literal 'PING' text every 10s."""
        while True:
            await asyncio.sleep(10)
            await ws.send("PING")

    ping_task = asyncio.create_task(_ping_loop())
    try:
        async for raw in ws:
            if raw == "PONG":
                continue
            try:
                msg = orjson.loads(raw)
            except orjson.JSONDecodeError:
                continue

            if not isinstance(msg, dict):
                continue

            # Record inter-arrival gap
            now = time.time()
            if latency is not None and _last_msg_ts > 0:
                latency.record_ws("clob_user", (now - _last_msg_ts) * 1000)
            _last_msg_ts = now

            event_type = msg.get("event_type", "")

            # Order events: event_type="order", sub-type in "type" field
            if event_type == "order":
                _handle_order_event(msg, state)  # type: ignore[arg-type]  # JSON boundary, validated in handler
            elif event_type == "trade":
                _handle_trade(msg, state)  # type: ignore[arg-type]  # JSON boundary, validated in handler
    finally:
        ping_task.cancel()


def _handle_order_event(data: _OrderEventMsg, state: MarketState) -> None:
    order_id = data.get("id", data.get("order_id", ""))
    order_type = data.get("type", "")  # PLACEMENT, UPDATE, CANCELLATION
    log.info("order %s type=%s", order_id[:12] if order_id else "?", order_type)

    if order_type == "CANCELLATION" and order_id in state.active_order_ids:
        state.active_order_ids.remove(order_id)


def _handle_trade(data: _TradeEventMsg, state: MarketState) -> None:
    token_id = data.get("asset_id", "")
    size = float(data.get("size", 0))
    price = float(data.get("price", 0))
    side = data.get("side", "").upper()
    status = data.get("status", "")
    trade_id = data.get("id", "")

    # Try to extract maker_order_id (field name varies by CLOB version)
    maker_order_id = (
        data.get("maker_order_id")
        or data.get("makerOrderId")
        or data.get("order_id")
        or data.get("orderId")
        or ""
    )

    log.info(
        "fill token=%s side=%s size=%.2f price=%.4f status=%s order=%s",
        token_id[:12],
        side,
        size,
        price,
        status,
        maker_order_id[:12] if maker_order_id else "?",
    )

    # Log full message at DEBUG for diagnostics on field names
    log.info("raw trade event: %s", data)

    # Only process confirmed fills
    if status not in ("MATCHED", "MINED", "CONFIRMED"):
        log.warning("unexpected trade status: %r, skipping", status)
        return

    # --- Fill tracking: match to our active orders ---
    matched_id = ""

    # Method 1: direct order_id match
    if maker_order_id and maker_order_id in state.active_order_ids:
        matched_id = maker_order_id

    # Method 2: if no order_id in event, match by token_id + side
    # (the bot only places one order per window per token, so this is safe)
    if not matched_id and side == "BUY" and state.active_order_ids:
        for oid in state.active_order_ids:
            # Any BUY fill on a token we're actively trading is ours
            if token_id in (state.up_token_id, state.down_token_id):
                matched_id = oid
                break

    if not matched_id:
        return

    now = time.time()
    fill_usd = price * size
    existing = state.live_fills.get(matched_id)

    # Dedupe by trade_id. A status transition (MATCHED → MINED → CONFIRMED)
    # for a trade we've already counted should only promote the confirmed
    # flag. A new trade_id against the same resting order is a distinct
    # counterparty match and must accumulate size (post-mortem 2026-04-22
    # §5.2 follow-up). If the event carries no trade_id we can't dedupe
    # safely, so we fall back to the old order-level dedupe — losing a
    # size on the unlikely multi-trade case is strictly safer than
    # triple-counting a confirmed flag promotion as three distinct fills.
    if existing is not None:
        if not trade_id or trade_id in existing.seen_trade_ids:
            # Known trade or unknown trade_id: treat as a status transition.
            if status == "CONFIRMED":
                existing.confirmed = True
            return
        # Distinct new trade against the same order — accumulate.
        existing.seen_trade_ids.add(trade_id)
        new_size = existing.size + size
        # Volume-weighted average price. Fills on a single maker-at-P order
        # should all land at P, but VWAP handles any edge-case drift safely.
        existing.price = (existing.price * existing.size + price * size) / new_size
        existing.size = new_size
        existing.size_usd += fill_usd
        if status == "CONFIRMED":
            existing.confirmed = True
        # Mirror the position counter updates that the first trade would
        # have triggered if we had logged it separately.
        _update_position(state, side, token_id, size)
        log.info(
            "LIVE FILL ACCUMULATED: order=%s token=%s trades=%d "
            "price_vwap=%.4f size=%.2f usd=$%.2f confirmed=%s",
            matched_id[:12],
            token_id[:12],
            len(existing.seen_trade_ids),
            existing.price,
            existing.size,
            existing.size_usd,
            existing.confirmed,
        )
        return

    # First trade for this order — create the LiveFill.
    _update_position(state, side, token_id, size)
    seen_ids: set[str] = {trade_id} if trade_id else set()
    state.live_fills[matched_id] = LiveFill(
        order_id=matched_id,
        token_id=token_id,
        side=side,
        price=price,
        size=size,
        size_usd=fill_usd,
        fill_time=now,
        confirmed=(status == "CONFIRMED"),
        is_maker=matched_id in state.maker_order_ids,
        seen_trade_ids=seen_ids,
    )

    fill = state.live_fills[matched_id]
    log.info(
        "LIVE FILL DETECTED: order=%s token=%s price=%.4f size=%.2f usd=$%.2f confirmed=%s",
        matched_id[:12],
        token_id[:12],
        fill.price,
        fill.size,
        fill.size_usd,
        fill.confirmed,
    )


def _update_position(state: MarketState, side: str, token_id: str, size: float) -> None:
    """Apply a trade's size to the aggregate position counters.

    Called once per distinct trade — NOT per status transition — so a
    MATCHED → MINED → CONFIRMED sequence for one trade only moves the
    counter once.
    """
    if side == "BUY":
        if token_id == state.up_token_id:
            state.position_up += size
        elif token_id == state.down_token_id:
            state.position_down += size
    elif side == "SELL":
        if token_id == state.up_token_id:
            state.position_up -= size
        elif token_id == state.down_token_id:
            state.position_down -= size
