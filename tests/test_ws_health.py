"""Tests for WebSocket health monitoring."""

from __future__ import annotations

import time

from utils.reconnect import FeedHealthMonitor, WsHealthState


class TestWsHealthState:
    def test_connected_not_disconnected(self):
        hs = WsHealthState(name="test", is_connected=True)
        assert hs.seconds_disconnected() == 0.0

    def test_disconnected_duration(self):
        hs = WsHealthState(
            name="test",
            is_connected=False,
            last_disconnected_at=time.time() - 10.0,
        )
        assert hs.seconds_disconnected() >= 9.0

    def test_never_connected_zero(self):
        hs = WsHealthState(name="test")
        assert hs.seconds_disconnected() == 0.0


class TestFeedHealthMonitor:
    def test_register_and_lookup(self):
        monitor = FeedHealthMonitor()
        hs = monitor.register("binance")
        assert hs.name == "binance"
        assert "binance" in monitor.feeds

    def test_all_connected_no_alert(self):
        monitor = FeedHealthMonitor()
        for name in ("binance", "rtds", "clob-market"):
            hs = monitor.register(name)
            hs.is_connected = True
        assert monitor.critical_feed_down() is None

    def test_binance_down_detected(self):
        monitor = FeedHealthMonitor()
        for name in ("binance", "rtds", "clob-market"):
            hs = monitor.register(name)
            hs.is_connected = True
        # Disconnect binance
        monitor.feeds["binance"].is_connected = False
        monitor.feeds["binance"].last_disconnected_at = time.time() - 60.0
        assert monitor.critical_feed_down(threshold_sec=30.0) == "binance"

    def test_short_disconnect_ok(self):
        monitor = FeedHealthMonitor()
        for name in ("binance", "rtds", "clob-market"):
            hs = monitor.register(name)
            hs.is_connected = True
        # Brief disconnect
        monitor.feeds["rtds"].is_connected = False
        monitor.feeds["rtds"].last_disconnected_at = time.time() - 5.0
        assert monitor.critical_feed_down(threshold_sec=30.0) is None

    def test_non_critical_feed_ignored(self):
        monitor = FeedHealthMonitor()
        monitor.register("binance").is_connected = True
        monitor.register("rtds").is_connected = True
        monitor.register("clob-market").is_connected = True
        # clob-user is not critical
        hs = monitor.register("clob-user")
        hs.is_connected = False
        hs.last_disconnected_at = time.time() - 120.0
        assert monitor.critical_feed_down(threshold_sec=30.0) is None
