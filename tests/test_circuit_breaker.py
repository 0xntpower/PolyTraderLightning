"""Tests for the circuit breaker utility."""

from __future__ import annotations

import time

import pytest

from utils.circuit_breaker import CircuitBreaker


class TestCircuitBreakerStates:
    def test_starts_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == "CLOSED"
        assert cb.can_attempt() is True

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"
        assert cb.can_attempt() is True

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_sec=60.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_attempt() is False

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "CLOSED"
        # Should need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"

    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown_sec=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"

        time.sleep(0.15)
        assert cb.state == "HALF_OPEN"
        assert cb.can_attempt() is True

    def test_half_open_success_closes(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown_sec=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "HALF_OPEN"

        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.can_attempt() is True

    def test_half_open_failure_reopens_with_longer_cooldown(self):
        cb = CircuitBreaker("test", failure_threshold=2, cooldown_sec=0.1, max_cooldown_sec=10.0)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "HALF_OPEN"

        cb.record_failure()
        assert cb.state == "OPEN"
        # Cooldown should have doubled: 0.1 -> 0.2
        assert cb._current_cooldown == pytest.approx(0.2)

    def test_cooldown_capped_at_max(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_sec=100.0, max_cooldown_sec=300.0)
        cb.record_failure()  # OPEN with cooldown=100
        cb._opened_at = time.time() - 200  # force expired
        assert cb.state == "HALF_OPEN"
        cb.record_failure()  # OPEN with cooldown=200
        cb._opened_at = time.time() - 300
        assert cb.state == "HALF_OPEN"
        cb.record_failure()  # OPEN with cooldown=min(400, 300)=300
        assert cb._current_cooldown == 300.0


class TestFeedStaleness:
    def test_fresh_feeds_not_stale(self):
        from market_data.state import MarketState

        state = MarketState()
        state.last_binance_msg_ts = time.time()
        state.last_chainlink_msg_ts = time.time()
        state.last_clob_market_msg_ts = time.time()
        assert state.is_feed_stale() is None

    def test_binance_stale(self):
        from market_data.state import MarketState

        state = MarketState()
        state.last_binance_msg_ts = time.time() - 20
        state.last_chainlink_msg_ts = time.time()
        state.last_clob_market_msg_ts = time.time()
        assert state.is_feed_stale(binance_threshold=15.0) == "binance"

    def test_chainlink_stale(self):
        from market_data.state import MarketState

        state = MarketState()
        state.last_binance_msg_ts = time.time()
        state.last_chainlink_msg_ts = time.time() - 40
        state.last_clob_market_msg_ts = time.time()
        assert state.is_feed_stale(chainlink_threshold=30.0) == "chainlink"

    def test_clob_book_stale(self):
        from market_data.state import MarketState

        state = MarketState()
        state.last_binance_msg_ts = time.time()
        state.last_chainlink_msg_ts = time.time()
        state.last_clob_market_msg_ts = time.time() - 70
        assert state.is_feed_stale(clob_book_threshold=60.0) == "clob_book"

    def test_no_messages_yet_not_stale(self):
        """Feeds that haven't received any messages yet should NOT be flagged."""
        from market_data.state import MarketState

        state = MarketState()
        # All timestamps at 0.0 (default)
        assert state.is_feed_stale() is None

    def test_priority_order_binance_first(self):
        """Binance is checked first, so it should be reported even if others are also stale."""
        from market_data.state import MarketState

        state = MarketState()
        state.last_binance_msg_ts = time.time() - 20
        state.last_chainlink_msg_ts = time.time() - 40
        state.last_clob_market_msg_ts = time.time() - 70
        assert state.is_feed_stale() == "binance"
