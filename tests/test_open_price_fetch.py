"""Tests for the non-blocking Gamma priceToBeat fetch system.

Covers:
  - _fetch_price_to_beat: shared HTTP helper (success, errors, malformed data)
  - try_fetch_open_price: background task lifecycle (launch, harvest, rate limit)
  - prefetch_next_open_price: pre-fetch in last seconds (timing, caching, harvest)
  - on_new_window: prefetch cache consumption, fallback to background fetch
  - Integration: full flow from pre-fetch through window transition
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from config import Config, ConnectionsConfig
from market_data.state import MarketState
from strategy.window_tracker import WINDOW_DURATION, WindowTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_events_response(price_to_beat: float | str | None) -> list[dict]:
    """Build a Gamma /events?metadata=true response with priceToBeat."""
    metadata: dict = {}
    if price_to_beat is not None:
        metadata["priceToBeat"] = price_to_beat
    return [{"eventMetadata": metadata}]


def _make_markets_response() -> list[dict]:
    """Minimal /markets response for token discovery."""
    return [
        {
            "outcomes": json.dumps(["Up", "Down"]),
            "clobTokenIds": json.dumps(["token_up", "token_down"]),
        }
    ]


class _FakeResponse:
    """Fake aiohttp response for session.get()."""

    def __init__(self, status: int, body: object) -> None:
        self.status = status
        self._body = body

    async def json(self) -> object:
        return self._body

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_tracker(
    session: MagicMock | None = None,
    state: MarketState | None = None,
) -> WindowTracker:
    cfg = Config(connections=ConnectionsConfig(gamma_rest="https://gamma-api.polymarket.com"))
    return WindowTracker(
        cfg,
        state or MarketState(),
        session or MagicMock(),
    )


# ---------------------------------------------------------------------------
# _fetch_price_to_beat — shared HTTP helper
# ---------------------------------------------------------------------------


class TestFetchPriceToBeat:
    @pytest.mark.asyncio
    async def test_returns_price_on_success(self) -> None:
        session = MagicMock()
        body = _make_events_response(72224.95)
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker._fetch_price_to_beat("btc-updown-5m-1000")

        assert result == pytest.approx(72224.95)

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self) -> None:
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(500, []))

        tracker = _make_tracker(session)
        assert await tracker._fetch_price_to_beat("slug") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_response(self) -> None:
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, []))

        tracker = _make_tracker(session)
        assert await tracker._fetch_price_to_beat("slug") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_event_metadata(self) -> None:
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, [{"id": "123"}]))

        tracker = _make_tracker(session)
        assert await tracker._fetch_price_to_beat("slug") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_price_to_beat_missing(self) -> None:
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, _make_events_response(None)))

        tracker = _make_tracker(session)
        assert await tracker._fetch_price_to_beat("slug") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_price_to_beat_zero(self) -> None:
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, _make_events_response(0.0)))

        tracker = _make_tracker(session)
        assert await tracker._fetch_price_to_beat("slug") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_price_to_beat_not_numeric(self) -> None:
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, _make_events_response("invalid")))

        tracker = _make_tracker(session)
        assert await tracker._fetch_price_to_beat("slug") is None

    @pytest.mark.asyncio
    async def test_parses_string_price(self) -> None:
        """priceToBeat may come as a string from the API."""
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, _make_events_response("72224.95")))

        tracker = _make_tracker(session)
        assert await tracker._fetch_price_to_beat("slug") == pytest.approx(72224.95)

    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(self) -> None:
        session = MagicMock()
        session.get = MagicMock(side_effect=TimeoutError("connect timeout"))

        tracker = _make_tracker(session)
        assert await tracker._fetch_price_to_beat("slug") is None


# ---------------------------------------------------------------------------
# try_fetch_open_price — background task lifecycle
# ---------------------------------------------------------------------------


class TestTryFetchOpenPrice:
    def test_returns_none_when_no_slug(self) -> None:
        tracker = _make_tracker()
        tracker._open_price_slug = None
        assert tracker.try_fetch_open_price() is None

    @pytest.mark.asyncio
    async def test_first_call_launches_task_returns_none(self) -> None:
        """First call should launch a background task and return None (non-blocking)."""
        tracker = _make_tracker()
        tracker._open_price_slug = "btc-updown-5m-1000"

        result = tracker.try_fetch_open_price()

        assert result is None
        assert tracker._open_price_task is not None

    @pytest.mark.asyncio
    async def test_harvests_successful_task(self) -> None:
        """When the background task completes with a price, next call returns it."""
        session = MagicMock()
        body = _make_events_response(72224.95)
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        tracker._open_price_slug = "btc-updown-5m-1000"

        # Launch
        assert tracker.try_fetch_open_price() is None
        # Let the task complete
        await asyncio.sleep(0)

        # Harvest
        result = tracker.try_fetch_open_price()
        assert result == pytest.approx(72224.95)

    @pytest.mark.asyncio
    async def test_returns_none_while_task_in_flight(self) -> None:
        """While a task is running, try_fetch should return None without launching another."""
        tracker = _make_tracker()
        tracker._open_price_slug = "btc-updown-5m-1000"

        # Create a task that won't complete immediately
        never_done: asyncio.Future[float | None] = asyncio.get_event_loop().create_future()
        tracker._open_price_task = asyncio.ensure_future(never_done)

        result = tracker.try_fetch_open_price()
        assert result is None
        # Should NOT have launched a second task — still the same one
        assert tracker._open_price_task is not None

        # Cleanup
        never_done.set_result(None)

    @pytest.mark.asyncio
    async def test_retries_after_failed_task(self) -> None:
        """When a task returns None, the next call (after rate limit) launches a new one."""
        session = MagicMock()
        # First call returns empty (no priceToBeat)
        session.get = MagicMock(return_value=_FakeResponse(200, []))

        tracker = _make_tracker(session)
        tracker._open_price_slug = "btc-updown-5m-1000"

        # Launch first task
        tracker.try_fetch_open_price()
        await asyncio.sleep(0)

        # Harvest failure — should clear the task
        assert tracker.try_fetch_open_price() is None
        assert tracker._open_price_task is None

        # Bypass rate limit
        tracker._open_price_last_launch = 0.0

        # Should launch a new task
        tracker.try_fetch_open_price()
        assert tracker._open_price_task is not None

    @pytest.mark.asyncio
    async def test_rate_limits_launches(self) -> None:
        """Calls within GAMMA_OPEN_PRICE_RETRY_DELAY should not launch new tasks."""
        tracker = _make_tracker()
        tracker._open_price_slug = "btc-updown-5m-1000"

        # First call launches
        tracker.try_fetch_open_price()
        first_task = tracker._open_price_task
        assert first_task is not None

        # Simulate the task completing with None
        first_task.cancel()
        tracker._open_price_task = None
        # Don't reset _open_price_last_launch — rate limit should block

        result = tracker.try_fetch_open_price()
        assert result is None
        assert tracker._open_price_task is None  # blocked by rate limit

    @pytest.mark.asyncio
    async def test_handles_task_exception(self) -> None:
        """If the background task raises, it should be caught and treated as None."""
        tracker = _make_tracker()
        tracker._open_price_slug = "btc-updown-5m-1000"

        # Create a task that raises
        async def _raise() -> float | None:
            raise RuntimeError("boom")

        tracker._open_price_task = asyncio.create_task(_raise())
        await asyncio.sleep(0)

        # Should not propagate the exception — harvests error, then launches new task
        result = tracker.try_fetch_open_price()
        assert result is None
        # After catching the error, it falls through and launches a new task
        # (because _open_price_last_launch is 0.0)
        assert tracker._open_price_task is not None


# ---------------------------------------------------------------------------
# prefetch_next_open_price — pre-fetch in last seconds of window
# ---------------------------------------------------------------------------


class TestPrefetchNextOpenPrice:
    def test_no_op_when_too_early(self) -> None:
        tracker = _make_tracker()
        tracker._current_window_ts = 1000

        tracker.prefetch_next_open_price(time_remaining=10.0)
        assert tracker._prefetch_task is None

    def test_no_op_when_too_late(self) -> None:
        tracker = _make_tracker()
        tracker._current_window_ts = 1000

        tracker.prefetch_next_open_price(time_remaining=0.3)
        assert tracker._prefetch_task is None

    def test_no_op_when_no_current_window(self) -> None:
        tracker = _make_tracker()
        tracker._current_window_ts = 0

        tracker.prefetch_next_open_price(time_remaining=3.0)
        assert tracker._prefetch_task is None

    @pytest.mark.asyncio
    async def test_launches_task_in_window(self) -> None:
        """Should launch a background task when 0.5 < time_remaining <= 5.0."""
        tracker = _make_tracker()
        tracker._current_window_ts = 1000

        tracker.prefetch_next_open_price(time_remaining=3.0)

        assert tracker._prefetch_task is not None
        assert tracker._prefetch_next_ts == 1000 + WINDOW_DURATION

    def test_skips_when_already_cached(self) -> None:
        tracker = _make_tracker()
        tracker._current_window_ts = 1000
        next_ts = 1000 + WINDOW_DURATION
        tracker._prefetch_next_ts = next_ts
        tracker._prefetch_price = 72224.95

        tracker.prefetch_next_open_price(time_remaining=3.0)
        assert tracker._prefetch_task is None  # already cached, no task needed

    def test_skips_when_task_in_flight(self) -> None:
        tracker = _make_tracker()
        tracker._current_window_ts = 1000

        # Simulate an in-flight task
        loop = asyncio.new_event_loop()
        fut: asyncio.Future[float | None] = loop.create_future()
        tracker._prefetch_task = asyncio.ensure_future(fut, loop=loop)  # type: ignore[arg-type]

        # Mock done() to return False
        tracker._prefetch_task = MagicMock()
        tracker._prefetch_task.done = MagicMock(return_value=False)

        tracker.prefetch_next_open_price(time_remaining=3.0)
        # Should not launch another task
        assert tracker._prefetch_task.done.called
        loop.close()

    @pytest.mark.asyncio
    async def test_harvests_completed_task(self) -> None:
        """When a prefetch task completes with a price, it should be cached."""
        session = MagicMock()
        body = _make_events_response(72224.95)
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        tracker._current_window_ts = 1000

        # Launch
        tracker.prefetch_next_open_price(time_remaining=3.0)
        assert tracker._prefetch_task is not None

        # Let task complete
        await asyncio.sleep(0)

        # Harvest on next call
        tracker.prefetch_next_open_price(time_remaining=2.5)
        assert tracker._prefetch_price == pytest.approx(72224.95)
        assert tracker._prefetch_task is None  # consumed

    @pytest.mark.asyncio
    async def test_retries_after_failed_prefetch(self) -> None:
        """When prefetch returns None, should try again after rate limit."""
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, []))

        tracker = _make_tracker(session)
        tracker._current_window_ts = 1000

        # Launch and let fail
        tracker.prefetch_next_open_price(time_remaining=4.0)
        await asyncio.sleep(0)

        # Harvest failure
        tracker._prefetch_last_launch = 0.0  # bypass rate limit
        tracker.prefetch_next_open_price(time_remaining=3.5)
        # Should have harvested None and launched a new task
        assert tracker._prefetch_price is None

    @pytest.mark.asyncio
    async def test_rate_limits_launches(self) -> None:
        tracker = _make_tracker()
        tracker._current_window_ts = 1000

        # First launch
        tracker.prefetch_next_open_price(time_remaining=3.0)
        task1 = tracker._prefetch_task
        assert task1 is not None

        # Simulate task completed with None
        tracker._prefetch_task = None

        # Second call immediately — should be rate limited
        tracker.prefetch_next_open_price(time_remaining=2.8)
        assert tracker._prefetch_task is None  # rate limited


# ---------------------------------------------------------------------------
# on_new_window — prefetch cache consumption
# ---------------------------------------------------------------------------


class TestOnNewWindowOpenPrice:
    @pytest.mark.asyncio
    async def test_uses_cached_prefetch(self) -> None:
        """If prefetch has a cached price, on_new_window should use it immediately."""
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, _make_markets_response()))

        state = MarketState()
        tracker = _make_tracker(session, state)

        # Simulate a pre-fetched price for the next window
        wts = tracker.current_window_ts()
        next_wts = wts + WINDOW_DURATION
        tracker._current_window_ts = wts  # pretend we're in the current window
        tracker._prefetch_next_ts = next_wts
        tracker._prefetch_price = 72224.95

        # Force the tracker to think a new window happened
        tracker._current_window_ts = wts  # will detect next_wts as new

        # Monkeypatch current_window_ts to return next_wts
        tracker.current_window_ts = lambda: next_wts  # type: ignore[assignment]

        info = await tracker.on_new_window()

        assert info is not None
        assert info.open_price == pytest.approx(72224.95)
        assert state.window_open_price == pytest.approx(72224.95)
        assert state.open_price_captured is True
        assert state.open_price_tier == 0
        assert tracker.open_price_fetched is True

    @pytest.mark.asyncio
    async def test_uses_completed_prefetch_task(self) -> None:
        """If prefetch task completed (but result not cached yet), harvest it."""
        session = MagicMock()
        body_markets = _make_markets_response()
        session.get = MagicMock(return_value=_FakeResponse(200, body_markets))

        state = MarketState()
        tracker = _make_tracker(session, state)

        wts = tracker.current_window_ts()
        next_wts = wts + WINDOW_DURATION
        tracker._current_window_ts = wts

        # Simulate a completed but not-yet-harvested prefetch task
        tracker._prefetch_next_ts = next_wts
        tracker._prefetch_price = None  # not cached yet

        async def _return_price() -> float | None:
            return 72252.90

        tracker._prefetch_task = asyncio.create_task(_return_price())
        await asyncio.sleep(0)  # let it complete

        tracker.current_window_ts = lambda: next_wts  # type: ignore[assignment]
        info = await tracker.on_new_window()

        assert info is not None
        assert info.open_price == pytest.approx(72252.90)
        assert state.window_open_price == pytest.approx(72252.90)
        assert tracker.open_price_fetched is True

    @pytest.mark.asyncio
    async def test_no_prefetch_launches_background_fetch(self) -> None:
        """If no prefetch available, should launch a background task and not block."""
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, _make_markets_response()))

        state = MarketState()
        tracker = _make_tracker(session, state)

        wts = tracker.current_window_ts()
        next_wts = wts + WINDOW_DURATION
        tracker._current_window_ts = wts

        tracker.current_window_ts = lambda: next_wts  # type: ignore[assignment]
        info = await tracker.on_new_window()

        assert info is not None
        assert info.open_price is None  # not available yet
        assert state.open_price_captured is False
        assert tracker.open_price_fetched is False
        # Background task should have been launched
        assert tracker._open_price_task is not None
        assert tracker._open_price_slug is not None

    @pytest.mark.asyncio
    async def test_clears_prefetch_state_after_consumption(self) -> None:
        """Prefetch state should be fully cleared after on_new_window, even on success."""
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, _make_markets_response()))

        state = MarketState()
        tracker = _make_tracker(session, state)

        wts = tracker.current_window_ts()
        next_wts = wts + WINDOW_DURATION
        tracker._current_window_ts = wts
        tracker._prefetch_next_ts = next_wts
        tracker._prefetch_price = 72224.95

        tracker.current_window_ts = lambda: next_wts  # type: ignore[assignment]
        await tracker.on_new_window()

        assert tracker._prefetch_next_ts == 0
        assert tracker._prefetch_price is None
        assert tracker._prefetch_task is None
        assert tracker._prefetch_last_launch == 0.0

    @pytest.mark.asyncio
    async def test_ignores_stale_prefetch_for_wrong_window(self) -> None:
        """If prefetch was for a different window_ts, it should be ignored."""
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, _make_markets_response()))

        state = MarketState()
        tracker = _make_tracker(session, state)

        wts = tracker.current_window_ts()
        next_wts = wts + WINDOW_DURATION
        tracker._current_window_ts = wts

        # Prefetch is for a completely different window
        tracker._prefetch_next_ts = 99999
        tracker._prefetch_price = 55555.55

        tracker.current_window_ts = lambda: next_wts  # type: ignore[assignment]
        info = await tracker.on_new_window()

        assert info is not None
        assert info.open_price is None  # should NOT use the stale prefetch
        assert state.window_open_price == 0.0


# ---------------------------------------------------------------------------
# Integration: full flow from prefetch through window transition and upgrade
# ---------------------------------------------------------------------------


class TestOpenPriceFullFlow:
    @pytest.mark.asyncio
    async def test_prefetch_hit_skips_all_fallbacks(self) -> None:
        """Happy path: prefetch succeeds → price set at tier 0 from first tick."""
        session = MagicMock()
        body_events = _make_events_response(72224.95)
        body_markets = _make_markets_response()

        def _route_get(url: str, **_kwargs: object) -> _FakeResponse:
            if "/markets?" in url:
                return _FakeResponse(200, body_markets)
            return _FakeResponse(200, body_events)

        session.get = _route_get

        state = MarketState()
        tracker = _make_tracker(session, state)

        # Phase 1: prefetch in last seconds of current window
        wts = tracker.current_window_ts()
        next_wts = wts + WINDOW_DURATION
        tracker._current_window_ts = wts

        tracker.prefetch_next_open_price(time_remaining=3.0)
        assert tracker._prefetch_task is not None
        await asyncio.sleep(0)  # let task complete

        # Harvest the prefetch
        tracker.prefetch_next_open_price(time_remaining=2.5)
        assert tracker._prefetch_price == pytest.approx(72224.95)

        # Phase 2: window transition
        tracker.current_window_ts = lambda: next_wts  # type: ignore[assignment]
        info = await tracker.on_new_window()

        assert info is not None
        assert info.open_price == pytest.approx(72224.95)
        assert state.window_open_price == pytest.approx(72224.95)
        assert state.open_price_tier == 0
        assert tracker.open_price_fetched is True

    @pytest.mark.asyncio
    async def test_prefetch_miss_then_background_upgrade(self) -> None:
        """Prefetch fails → RTDS fallback used → background task upgrades price."""
        session = MagicMock()
        body_markets = _make_markets_response()

        responses: list[_FakeResponse] = []

        def _route_get(url: str, **_kwargs: object) -> _FakeResponse:
            if "/markets?" in url:
                return _FakeResponse(200, body_markets)
            if responses:
                return responses.pop(0)
            return _FakeResponse(200, [])  # no price yet

        session.get = _route_get

        state = MarketState()
        state.btc_chainlink = 72220.0  # RTDS has a price
        tracker = _make_tracker(session, state)

        wts = tracker.current_window_ts()
        next_wts = wts + WINDOW_DURATION
        tracker._current_window_ts = wts

        # No prefetch available
        tracker.current_window_ts = lambda: next_wts  # type: ignore[assignment]
        info = await tracker.on_new_window()

        assert info is not None
        assert info.open_price is None
        assert tracker.open_price_fetched is False

        # Let the background task complete (returns empty)
        await asyncio.sleep(0)

        # Simulate main loop: try_fetch_open_price returns None
        result = tracker.try_fetch_open_price()
        assert result is None  # task returned empty

        # Now Gamma has the price
        responses.append(_FakeResponse(200, _make_events_response(72224.95)))
        tracker._open_price_last_launch = 0.0  # bypass rate limit

        # Launch new task
        tracker.try_fetch_open_price()
        await asyncio.sleep(0)

        # Harvest success
        result = tracker.try_fetch_open_price()
        assert result == pytest.approx(72224.95)

    @pytest.mark.asyncio
    async def test_prefetch_task_error_falls_through(self) -> None:
        """If the prefetch task raised an exception, on_new_window handles gracefully."""
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, _make_markets_response()))

        state = MarketState()
        tracker = _make_tracker(session, state)

        wts = tracker.current_window_ts()
        next_wts = wts + WINDOW_DURATION
        tracker._current_window_ts = wts
        tracker._prefetch_next_ts = next_wts

        # Create a task that raises
        async def _raise() -> float | None:
            raise RuntimeError("network error")

        tracker._prefetch_task = asyncio.create_task(_raise())
        await asyncio.sleep(0)

        tracker.current_window_ts = lambda: next_wts  # type: ignore[assignment]
        info = await tracker.on_new_window()

        # Should not crash, should fall through to background fetch
        assert info is not None
        assert info.open_price is None
        assert tracker.open_price_fetched is False

    @pytest.mark.asyncio
    async def test_multiple_windows_reset_state(self) -> None:
        """Each new window should fully reset open price fetch state."""
        session = MagicMock()
        body_markets = _make_markets_response()
        body_events = _make_events_response(72224.95)

        def _route_get(url: str, **_kwargs: object) -> _FakeResponse:
            if "/markets?" in url:
                return _FakeResponse(200, body_markets)
            return _FakeResponse(200, body_events)

        session.get = _route_get

        state = MarketState()
        tracker = _make_tracker(session, state)

        # Window 1
        wts1 = tracker.current_window_ts()
        next_wts1 = wts1 + WINDOW_DURATION
        tracker._current_window_ts = wts1
        tracker._prefetch_next_ts = next_wts1
        tracker._prefetch_price = 72224.95

        tracker.current_window_ts = lambda: next_wts1  # type: ignore[assignment]
        info1 = await tracker.on_new_window()
        assert info1 is not None
        assert tracker.open_price_fetched is True

        # Window 2 — state should be clean
        next_wts2 = next_wts1 + WINDOW_DURATION
        tracker.current_window_ts = lambda: next_wts2  # type: ignore[assignment]
        info2 = await tracker.on_new_window()

        # No prefetch for window 2 — should fall through
        assert info2 is not None
        # open_price_fetched should have been reset for the new window
        # (it was set to False in on_new_window before checking prefetch)
        # Since no prefetch is cached for next_wts2, it should launch background
        assert tracker._open_price_slug is not None
