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

    # Update aggregate position
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

    if matched_id in state.live_fills:
        # Status transition (MATCHED→MINED→CONFIRMED) for an already-tracked
        # fill.  Only promote the confirmed flag — do NOT accumulate size,
        # because each status event carries the same trade size, not an
        # incremental partial.
        existing = state.live_fills[matched_id]
        if status == "CONFIRMED":
            existing.confirmed = True
        return
    state.live_fills[matched_id] = LiveFill(
        order_id=matched_id,
        token_id=token_id,
        side=side,
        price=price,
        size=size,
        size_usd=fill_usd,
        fill_time=now,
        confirmed=(status == "CONFIRMED"),
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
