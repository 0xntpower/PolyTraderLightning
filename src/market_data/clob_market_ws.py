"""WS3: CLOB market channel — orderbook and price changes for target market."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, TypedDict

import orjson

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    from market_data.latency_tracker import LatencyTracker
    from market_data.state import MarketState


class _BookLevel(TypedDict):
    price: str


class _PriceChangeMsg(TypedDict, total=False):
    asset_id: str
    best_bid: str
    best_ask: str


class _OrderbookMsg(TypedDict, total=False):
    asset_id: str
    bids: list[_BookLevel]
    asks: list[_BookLevel]
    event_type: str
    price_changes: list[_PriceChangeMsg]


log = logging.getLogger(__name__)


def _clear_orderbook(state: MarketState) -> None:
    """Zero out bid/ask prices to avoid stale data after reconnect."""
    state.best_bid_up = 0.0
    state.best_ask_up = 0.0
    state.best_bid_down = 0.0
    state.best_ask_down = 0.0


async def subscribe_market(
    ws: ClientConnection,
    state: MarketState,
    prev_ids: tuple[str, str] | None = None,
) -> None:
    """Subscribe to market channel. Unsubscribes from old tokens if provided."""
    if not state.up_token_id or not state.down_token_id:
        log.warning("cannot subscribe to market channel — no token IDs set")
        return

    # Clear stale orderbook data before subscribing
    _clear_orderbook(state)

    if prev_ids:
        # Unsubscribe old tokens, then dynamically subscribe new ones
        unsub = orjson.dumps(
            {
                "operation": "unsubscribe",
                "assets_ids": list(prev_ids),
            }
        ).decode()
        await ws.send(unsub)

        sub = orjson.dumps(
            {
                "operation": "subscribe",
                "assets_ids": [state.up_token_id, state.down_token_id],
            }
        ).decode()
        await ws.send(sub)
    else:
        # Initial subscription on fresh connection
        msg = orjson.dumps(
            {
                "type": "market",
                "assets_ids": [state.up_token_id, state.down_token_id],
            }
        ).decode()
        await ws.send(msg)

    log.info(
        "CLOB market subscribed for tokens up=%s down=%s",
        state.up_token_id[:12],
        state.down_token_id[:12],
    )


async def handle_clob_market(
    ws: ClientConnection,
    state: MarketState,
    latency: LatencyTracker | None = None,
) -> None:
    last_ids: tuple[str, str] | None = None
    _last_msg_ts = 0.0

    async def _subscription_loop() -> None:
        nonlocal last_ids
        while True:
            if state.up_token_id and state.down_token_id:
                current_ids = (state.up_token_id, state.down_token_id)
                if current_ids != last_ids:
                    await subscribe_market(ws, state, prev_ids=last_ids)
                    last_ids = current_ids
            await asyncio.sleep(1.0)

    async def _ping_loop() -> None:
        """CLOB WS requires literal 'PING' text every 10s."""
        while True:
            await asyncio.sleep(10)
            await ws.send("PING")

    sub_task = asyncio.create_task(_subscription_loop())
    ping_task = asyncio.create_task(_ping_loop())
    try:
        async for raw in ws:
            if raw == "PONG":
                continue
            try:
                msg = orjson.loads(raw)
            except orjson.JSONDecodeError:
                continue

            # Record inter-arrival gap for CLOB market messages
            now = time.time()
            if latency is not None and _last_msg_ts > 0:
                latency.record_ws("clob_market", (now - _last_msg_ts) * 1000)
            _last_msg_ts = now
            state.last_clob_market_msg_ts = now

            if isinstance(msg, dict):
                event_type = msg.get("event_type")

                if event_type == "book":
                    _handle_orderbook(msg, state)  # type: ignore[arg-type]  # JSON boundary
                elif event_type == "price_change":
                    for pc in msg.get("price_changes", []):
                        _handle_price_change(pc, state)
                elif event_type == "best_bid_ask":
                    _handle_price_change(msg, state)  # type: ignore[arg-type]  # JSON boundary
                elif event_type == "last_trade_price":
                    pass
                elif event_type is None:
                    # Messages without event_type (e.g. initial dump)
                    if "price_changes" in msg:
                        for pc in msg["price_changes"]:
                            _handle_price_change(pc, state)
                    elif "bids" in msg or "asks" in msg:
                        _handle_orderbook(msg, state)  # type: ignore[arg-type]  # JSON boundary

            elif isinstance(msg, list):
                for item in msg:
                    if isinstance(item, dict):
                        _handle_orderbook(item, state)  # type: ignore[arg-type]  # JSON boundary
    finally:
        sub_task.cancel()
        ping_task.cancel()


def _handle_price_change(data: _PriceChangeMsg, state: MarketState) -> None:
    token_id = data.get("asset_id", "")
    best_bid = float(data.get("best_bid") or 0)
    best_ask = float(data.get("best_ask") or 0)

    if token_id == state.up_token_id:
        if best_bid > 0:
            state.best_bid_up = best_bid
        if best_ask > 0:
            state.best_ask_up = best_ask
        if best_bid > 0 or best_ask > 0:
            state.has_fresh_book_data = True

    elif token_id == state.down_token_id:
        if best_bid > 0:
            state.best_bid_down = best_bid
        if best_ask > 0:
            state.best_ask_down = best_ask
        if best_bid > 0 or best_ask > 0:
            state.has_fresh_book_data = True


def _handle_orderbook(data: _OrderbookMsg, state: MarketState) -> None:
    token_id = data.get("asset_id", "")
    bids = data.get("bids", [])
    asks = data.get("asks", [])

    try:
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 0.0
    except (KeyError, TypeError, IndexError, ValueError):
        return

    if token_id == state.up_token_id:
        if best_bid > 0:
            state.best_bid_up = best_bid
        if best_ask > 0:
            state.best_ask_up = best_ask
        if best_bid > 0 or best_ask > 0:
            state.has_fresh_book_data = True
    elif token_id == state.down_token_id:
        if best_bid > 0:
            state.best_bid_down = best_bid
        if best_ask > 0:
            state.best_ask_down = best_ask
        if best_bid > 0 or best_ask > 0:
            state.has_fresh_book_data = True
