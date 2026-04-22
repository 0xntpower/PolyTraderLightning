"""WS2: RTDS — Chainlink BTC/USD oracle price (resolution source)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    from market_data.latency_tracker import LatencyTracker
    from market_data.state import MarketState

log = logging.getLogger(__name__)

# Filters must be strings, not objects — RTDS quirk
SUB_CHAINLINK = orjson.dumps(
    {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": '{"symbol":"btc/usd"}',
            }
        ],
    }
).decode()


async def _ping_loop(ws: ClientConnection, interval: int) -> None:
    """RTDS requires a literal "PING" text message, not a WebSocket ping frame."""
    while True:
        await asyncio.sleep(interval)
        await ws.send("PING")


async def handle_rtds(
    ws: ClientConnection,
    state: MarketState,
    ping_interval: int = 5,
    latency: LatencyTracker | None = None,
) -> None:
    await ws.send(SUB_CHAINLINK)
    log.info("RTDS subscribed to chainlink")

    ping_task = asyncio.create_task(_ping_loop(ws, ping_interval))
    try:
        async for raw in ws:
            if raw == "PONG":
                continue
            try:
                msg = orjson.loads(raw)
            except orjson.JSONDecodeError:
                continue

            topic = msg.get("topic")
            payload = msg.get("payload")
            if not payload:
                continue

            if topic == "crypto_prices_chainlink":
                now = time.time()
                price = float(payload.get("value", 0))
                oracle_ts_ms = payload.get("timestamp")
                if price > 0:
                    state.btc_chainlink = price
                    state.btc_chainlink_ts = now
                    state.last_chainlink_msg_ts = now
                    # Buffer tick with oracle's own timestamp for boundary-aligned open price
                    if isinstance(oracle_ts_ms, int) and oracle_ts_ms > 0:
                        from market_data.state import ChainlinkTick

                        state.chainlink_tick_buffer.append(
                            ChainlinkTick(oracle_ts_ms=oracle_ts_ms, price=price)
                        )
                # True transport lag: oracle stamps the message at emit time,
                # ``now * 1000 - oracle_ts_ms`` is the wire + processing delay.
                # The old inter-arrival gap reported Chainlink's ~1 s emit
                # cadence as "latency", which is the update rate not a delay.
                if latency is not None and isinstance(oracle_ts_ms, int) and oracle_ts_ms > 0:
                    latency.record_ws("rtds", max(0.0, now * 1000.0 - oracle_ts_ms))
    finally:
        ping_task.cancel()
