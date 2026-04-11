"""WS1: Binance combined stream — btcusdt@trade (price) + btcusdt@depth5@100ms (OBI)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, TypedDict

import orjson

if TYPE_CHECKING:
    from collections.abc import Sequence

    from websockets.asyncio.client import ClientConnection

    from market_data.latency_tracker import LatencyTracker
    from market_data.state import MarketState

log = logging.getLogger(__name__)


class _BinanceTradeMsg(TypedDict, total=False):
    p: str  # price
    T: int  # exchange timestamp ms


class _BinanceDepthMsg(TypedDict, total=False):
    bids: list[list[str]]  # [[price, qty], ...]
    asks: list[list[str]]


class _BinanceCombinedMsg(TypedDict, total=False):
    stream: str
    data: _BinanceTradeMsg | _BinanceDepthMsg


def _compute_obi(bids: Sequence[Sequence[str]], asks: Sequence[Sequence[str]]) -> float:
    """Order Book Imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)."""
    try:
        bid_qty = sum(float(b[1]) for b in bids)
        ask_qty = sum(float(a[1]) for a in asks)
        total = bid_qty + ask_qty
        return (bid_qty - ask_qty) / total if total > 0 else 0.0
    except (IndexError, ValueError, TypeError):
        return 0.0


async def handle_binance(
    ws: ClientConnection,
    state: MarketState,
    latency: LatencyTracker | None = None,
) -> None:
    async for raw in ws:
        try:
            msg = orjson.loads(raw)
        except orjson.JSONDecodeError:
            continue

        # Combined stream wraps each message: {"stream": "...", "data": {...}}
        if "stream" in msg and "data" in msg:
            stream = msg["stream"]
            data = msg["data"]
            if "trade" in stream:
                _handle_trade(data, state, latency)
            elif "depth" in stream:
                _handle_depth(data, state)
        else:
            # Single stream fallback (trade-only URL)
            _handle_trade(msg, state, latency)


def _handle_trade(
    data: _BinanceTradeMsg,
    state: MarketState,
    latency: LatencyTracker | None = None,
) -> None:
    try:
        state.btc_binance = float(data["p"])
        now = time.time()
        state.btc_binance_ts = now
        state.last_binance_msg_ts = now
        # Binance trade messages include exchange timestamp "T" (epoch ms).
        # Difference = wire + processing latency.
        if latency is not None:
            exchange_ms = data.get("T")
            if exchange_ms is not None:
                latency.record_ws("binance", now * 1000 - exchange_ms)
    except (KeyError, ValueError):
        log.warning("binance trade: skipping malformed tick")


def _handle_depth(data: _BinanceDepthMsg, state: MarketState) -> None:
    try:
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        state.binance_obi = _compute_obi(bids, asks)
        state.binance_obi_ts = time.time()
    except (KeyError, ValueError, TypeError, IndexError):
        pass  # best-effort OBI update, malformed depth tick is non-critical
