"""Tests for WindowTracker — resolution polling uses /events endpoint."""

import json
from unittest.mock import MagicMock

import pytest

from config import Config, ConnectionsConfig
from strategy.window_tracker import WindowTracker


def _make_events_response(uma_status: str, outcomes: list, prices: list) -> list:
    """Build a Gamma /events API response with nested markets."""
    return [
        {
            "id": "353809",
            "slug": "btc-updown-5m-1775650800",
            "markets": [
                {
                    "umaResolutionStatus": uma_status,
                    "outcomes": json.dumps(outcomes),
                    "outcomePrices": json.dumps(prices),
                    "clobTokenIds": json.dumps(["token_up", "token_down"]),
                }
            ],
        }
    ]


def _make_markets_response(outcomes: list, token_ids: list) -> list:
    """Build a Gamma /markets API response (flat, for token discovery)."""
    return [
        {
            "outcomes": json.dumps(outcomes),
            "clobTokenIds": json.dumps(token_ids),
        }
    ]


class _FakeResponse:
    def __init__(self, status: int, body):
        self.status = status
        self._body = body

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _make_tracker(session: MagicMock) -> WindowTracker:
    cfg = Config(connections=ConnectionsConfig(gamma_rest="https://gamma-api.polymarket.com"))
    state = MagicMock()
    return WindowTracker(cfg, state, session)


class TestFetchMarketResolution:
    """Verify resolution uses /events and extracts nested market data."""

    @pytest.mark.asyncio
    async def test_resolves_down(self):
        session = MagicMock()
        body = _make_events_response("resolved", ["Up", "Down"], ["0", "1"])
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is not None
        assert result.outcome == "down"
        assert result.price_to_beat is None
        assert result.final_price is None
        call_args = session.get.call_args
        assert "/events?slug=" in call_args[0][0]
        assert "/markets?slug=" not in call_args[0][0]

    @pytest.mark.asyncio
    async def test_resolves_up(self):
        session = MagicMock()
        body = _make_events_response("resolved", ["Up", "Down"], ["1", "0"])
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is not None
        assert result.outcome == "up"

    @pytest.mark.asyncio
    async def test_not_resolved_returns_none(self):
        session = MagicMock()
        body = _make_events_response("proposed", ["Up", "Down"], ["0.5", "0.5"])
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_markets_returns_none(self):
        session = MagicMock()
        body = [{"id": "353809", "markets": []}]
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        session = MagicMock()
        session.get = MagicMock(return_value=_FakeResponse(200, []))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is None

    # -- price-threshold boundary and malformed-price handling (relocated from
    # -- PolySignalLab root tests/test_resolution.py, rewritten against the
    # -- real fetch_market_resolution instead of a hand-copied parsing helper)

    @pytest.mark.asyncio
    async def test_resolves_at_exact_threshold(self):
        """0.99 is the >= boundary — must still resolve."""
        session = MagicMock()
        body = _make_events_response("resolved", ["Up", "Down"], ["0.99", "0.01"])
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is not None
        assert result.outcome == "up"

    @pytest.mark.asyncio
    async def test_below_threshold_resolved_status_returns_none(self):
        """0.98 is below the 0.99 threshold — resolved status alone isn't enough."""
        session = MagicMock()
        body = _make_events_response("resolved", ["Up", "Down"], ["0.98", "0.02"])
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_price_value_skipped(self):
        """A non-numeric price is skipped, not a crash — next price still resolves."""
        session = MagicMock()
        body = _make_events_response("resolved", ["Up", "Down"], ["abc", "1"])
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is not None
        assert result.outcome == "down"

    @pytest.mark.asyncio
    async def test_none_price_value_skipped(self):
        session = MagicMock()
        body = _make_events_response("resolved", ["Up", "Down"], [None, "1"])
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is not None
        assert result.outcome == "down"

    @pytest.mark.asyncio
    async def test_insufficient_outcomes_returns_none(self):
        """A single outcome/price, even with status=resolved, can't determine a winner."""
        session = MagicMock()
        body = _make_events_response("resolved", ["Up"], ["1"])
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_market_resolution("btc-updown-5m-1775650800")

        assert result is None


class TestFetchTokenIds:
    """Token discovery still uses /markets (which works for clobTokenIds)."""

    @pytest.mark.asyncio
    async def test_fetches_tokens(self):
        session = MagicMock()
        body = _make_markets_response(["Up", "Down"], ["token_up", "token_down"])
        session.get = MagicMock(return_value=_FakeResponse(200, body))

        tracker = _make_tracker(session)
        result = await tracker.fetch_token_ids("btc-updown-5m-1775650800")

        assert result == ("token_up", "token_down")
        call_args = session.get.call_args
        assert "/markets?slug=" in call_args[0][0]
