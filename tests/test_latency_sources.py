"""Tests for the latency instrumentation on event-driven WS channels.

Prior to v3.6.2 the ``rtds`` and ``clob_user`` channels recorded
inter-arrival gaps as a latency proxy. That produced numbers that
looked like "latency" but were actually:

- ``rtds`` ≈ Chainlink's ~1 s oracle emit cadence (not a delay)
- ``clob_user`` ≈ Polygon's ~8 s MINED→CONFIRMED confirmation wait
  between our own status-transition events (not a transport delay)

Both channels now report true wire latency: ``now - server_emit_ts``.
These tests lock that in.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import orjson
import pytest

from market_data.clob_user_ws import handle_clob_user
from market_data.latency_tracker import LatencyTracker
from market_data.rtds_ws import handle_rtds
from market_data.state import MarketState

# ---------------------------------------------------------------------------
# Async WS fakes — yield a fixed sequence of raw messages, then exit
# ---------------------------------------------------------------------------


class _FakeWS:
    """Minimal async-iterable substitute for a websockets ClientConnection.

    Iterating yields the queued raw messages in order. ``send`` swallows
    PING subscriptions without touching the queue.
    """

    def __init__(self, messages: list[str]) -> None:
        self._messages = list(messages)
        self.sent: list[str] = []

    async def send(self, msg: str) -> None:
        self.sent.append(msg)

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        # Yield control so the PING loop can run at least once without
        # starving the iterator.
        await asyncio.sleep(0)
        return self._messages.pop(0)


def _as_ws_message(payload: dict[str, Any]) -> str:
    return orjson.dumps(payload).decode()


# ---------------------------------------------------------------------------
# rtds_ws — oracle timestamp feeds true transport latency
# ---------------------------------------------------------------------------


def test_rtds_records_oracle_timestamp_latency() -> None:
    """Chainlink tick with ``timestamp`` field produces now-oracle_ts_ms lag."""
    oracle_ts_ms = 1_776_000_000_000  # known instant (ms)
    fixed_now = 1_776_000_000.250  # 250 ms later (s)

    msg = _as_ws_message(
        {
            "topic": "crypto_prices_chainlink",
            "payload": {"value": 78500.5, "timestamp": oracle_ts_ms},
        }
    )
    ws = _FakeWS([msg])
    state = MarketState()
    tracker = LatencyTracker()

    with patch("market_data.rtds_ws.time.time", return_value=fixed_now):
        asyncio.run(handle_rtds(ws, state, ping_interval=1_000, latency=tracker))

    stats = {s.name: s for s in tracker.all_stats()}
    assert "rtds" in stats
    assert stats["rtds"].samples == 1
    assert stats["rtds"].median_ms == pytest.approx(250.0)


def test_rtds_no_oracle_timestamp_skips_recording() -> None:
    """If Chainlink drops the ``timestamp`` field, we don't invent a number."""
    msg = _as_ws_message(
        {
            "topic": "crypto_prices_chainlink",
            "payload": {"value": 78500.5},
        }
    )
    ws = _FakeWS([msg])
    state = MarketState()
    tracker = LatencyTracker()

    asyncio.run(handle_rtds(ws, state, ping_interval=1_000, latency=tracker))

    stats = {s.name: s for s in tracker.all_stats()}
    # Price update still lands on state, but no latency sample is recorded.
    assert state.btc_chainlink == pytest.approx(78500.5)
    assert "rtds" not in stats or stats["rtds"].samples == 0


def test_rtds_two_ticks_same_emit_time_report_low_latency() -> None:
    """Back-to-back ticks at matching emit times should report near-zero
    latency, not ~0 s from inter-arrival (old behaviour would have
    reported the gap between arrivals).
    """
    oracle_ts_ms = 1_776_000_000_000
    fixed_now = 1_776_000_000.010  # 10 ms after emit

    messages = [
        _as_ws_message(
            {
                "topic": "crypto_prices_chainlink",
                "payload": {"value": 78500.0, "timestamp": oracle_ts_ms},
            }
        ),
        _as_ws_message(
            {
                "topic": "crypto_prices_chainlink",
                "payload": {"value": 78501.0, "timestamp": oracle_ts_ms},
            }
        ),
    ]
    ws = _FakeWS(messages)
    tracker = LatencyTracker()

    with patch("market_data.rtds_ws.time.time", return_value=fixed_now):
        asyncio.run(handle_rtds(ws, MarketState(), ping_interval=1_000, latency=tracker))

    stats = {s.name: s for s in tracker.all_stats()}
    assert stats["rtds"].samples == 2
    assert stats["rtds"].median_ms == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# clob_user_ws — server timestamp feeds true transport latency
# ---------------------------------------------------------------------------


def test_clob_user_records_server_timestamp_latency() -> None:
    """Trade event with ``timestamp`` field produces now-server_ts_ms lag."""
    server_ts_ms = 1_776_000_000_000
    fixed_now = 1_776_000_000.035  # 35 ms later

    trade_evt = {
        "type": "TRADE",
        "event_type": "trade",
        "id": "test-trade-1",
        "asset_id": "asset-down",
        "size": "1.77",
        "price": "0.88",
        "side": "BUY",
        "status": "MATCHED",
        "timestamp": str(server_ts_ms),
    }
    ws = _FakeWS([_as_ws_message(trade_evt)])
    tracker = LatencyTracker()

    with patch("market_data.clob_user_ws.time.time", return_value=fixed_now):
        asyncio.run(handle_clob_user(ws, MarketState(), api_creds={}, latency=tracker))

    stats = {s.name: s for s in tracker.all_stats()}
    assert "clob_user" in stats
    assert stats["clob_user"].samples == 1
    assert stats["clob_user"].median_ms == pytest.approx(35.0)


def test_clob_user_three_status_transitions_report_transport_not_blockchain() -> None:
    """MATCHED → MINED → CONFIRMED each carry their own emit timestamp.

    The OLD inter-arrival metric reported the ~8 s MINED→CONFIRMED blockchain
    wait as "latency". The NEW metric reports wire delay for each event
    separately — much smaller and semantically meaningful.
    """
    ws_fast_emit = 1_776_000_000_000  # emit time in ms

    # Each event stamped at the same emit instant (simulating server-side
    # prompt dispatch); local arrival 20 ms later uniformly.
    fixed_now = 1_776_000_000.020

    def _evt(status: str, trade_id: str = "t-1") -> str:
        return _as_ws_message(
            {
                "type": "TRADE",
                "event_type": "trade",
                "id": trade_id,
                "asset_id": "asset-down",
                "size": "1.77",
                "price": "0.88",
                "side": "BUY",
                "status": status,
                "timestamp": str(ws_fast_emit),
            }
        )

    ws = _FakeWS([_evt("MATCHED"), _evt("MINED"), _evt("CONFIRMED")])
    tracker = LatencyTracker()

    with patch("market_data.clob_user_ws.time.time", return_value=fixed_now):
        asyncio.run(handle_clob_user(ws, MarketState(), api_creds={}, latency=tracker))

    stats = {s.name: s for s in tracker.all_stats()}
    assert stats["clob_user"].samples == 3
    # All samples should reflect the 20 ms wire delay, not the multi-second
    # blockchain confirmation wait.
    assert stats["clob_user"].median_ms == pytest.approx(20.0)
    assert stats["clob_user"].max_ms == pytest.approx(20.0)


def test_clob_user_no_timestamp_skips_recording() -> None:
    """Messages without a ``timestamp`` field — don't invent a latency."""
    evt = {
        "type": "TRADE",
        "event_type": "trade",
        "id": "test-no-ts",
        "asset_id": "asset-down",
        "size": "1.0",
        "price": "0.50",
        "side": "BUY",
        "status": "MATCHED",
    }
    ws = _FakeWS([_as_ws_message(evt)])
    tracker = LatencyTracker()

    asyncio.run(handle_clob_user(ws, MarketState(), api_creds={}, latency=tracker))

    stats = {s.name: s for s in tracker.all_stats()}
    assert "clob_user" not in stats or stats["clob_user"].samples == 0


def test_clob_user_negative_latency_clamped_to_zero() -> None:
    """Clock skew can produce a future-stamped emit time. Clamp to 0 rather
    than record a nonsensical negative latency."""
    server_ts_ms = 1_776_000_000_500  # 500 ms in the future
    fixed_now = 1_776_000_000.000

    evt = {
        "type": "TRADE",
        "event_type": "trade",
        "id": "test-skew",
        "asset_id": "asset-down",
        "size": "1.0",
        "price": "0.50",
        "side": "BUY",
        "status": "MATCHED",
        "timestamp": str(server_ts_ms),
    }
    ws = _FakeWS([_as_ws_message(evt)])
    tracker = LatencyTracker()

    with patch("market_data.clob_user_ws.time.time", return_value=fixed_now):
        asyncio.run(handle_clob_user(ws, MarketState(), api_creds={}, latency=tracker))

    stats = {s.name: s for s in tracker.all_stats()}
    assert stats["clob_user"].samples == 1
    assert stats["clob_user"].median_ms == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Discord STALE threshold — clob_user no longer flagged on routine silence
# ---------------------------------------------------------------------------


def test_clob_user_stale_threshold_allows_long_quiet_periods() -> None:
    """Sanity guard for shared.discord._STALE_THRESHOLDS: clob_user must be
    large enough that a session with no fires for 30-60 min does NOT
    render a ⚠️ STALE. The real threshold lives inline in
    ``send_latency_report``; this test re-extracts it to pin the intent.
    """
    import inspect

    from shared import discord

    src = inspect.getsource(discord.send_latency_report)
    # The dict literal is defined inside the function; grep the source for
    # the clob_user entry. This is a soft check — if someone renames it,
    # the test will flag it as intentional surgery rather than a silent
    # drop-back to the old 60 s default.
    assert '"clob_user": 3600' in src or "'clob_user': 3600" in src
