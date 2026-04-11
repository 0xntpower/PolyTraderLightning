"""Tests for SignalLifecycle — state machine for signal age, fire-rate, and decay.

Covers:
- Signal age and fire-rate tracking
- Fire-stall detection → IDLE transition
- SPRT decay → IDLE transition
- Shadow tracking (accumulation, window finalization, expiry)
- Signal swap transitions (same signal refresh vs different signal reset)
- Running entry average tracking
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fakes import make_rules_config, make_signal_config
from strategy.signal import Direction, Signal
from strategy.signal_lifecycle import (
    SignalLifecycle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMarketState:
    """Minimal fake for market state needed by MomentumSignalStrategy construction."""

    btc_binance = 100_000.0
    btc_chainlink = 100_000.0
    window_open_price = 100_000.0
    binance_window_open_price = 100_000.0
    open_price_captured = True
    binance_open_price_captured = True
    best_ask_up = 0.85
    best_ask_down = 0.15
    best_bid_up = 0.84
    best_bid_down = 0.14
    up_token_id = "token_up"
    down_token_id = "token_down"
    time_remaining = 200.0
    has_fresh_book_data = True


class FakeJournal:
    """Records trade entries for assertion."""

    def __init__(self):
        self.trades: list = []

    def record_trade(self, record) -> bool:
        self.trades.append(record)
        return True

    @staticmethod
    def now_iso() -> str:
        return "2026-04-08T00:00:00+00:00"


def _make_signal(bn_direction_from_open_pct: float = 0.001) -> Signal:
    return Signal(
        delta_pct=0.10,
        direction=Direction.UP,
        feeds_agree=True,
        bn_direction_from_open_pct=bn_direction_from_open_pct,
        cl_direction_from_open_pct=bn_direction_from_open_pct,
        poly_spread_up=0.0,
        poly_spread_down=0.0,
        binance_obi=0.0,
        time_remaining=200.0,
    )


# ---------------------------------------------------------------------------
# Initial state tests
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_starts_active(self):
        lc = SignalLifecycle()
        assert not lc.is_idle
        assert lc.idle_reason is None
        assert lc.signal_age_windows == 0
        assert lc.windows_since_last_fire == 0

    def test_no_shadow_initially(self):
        lc = SignalLifecycle()
        assert lc.shadow is None


# ---------------------------------------------------------------------------
# Signal age and fire-rate tracking
# ---------------------------------------------------------------------------


class TestFireRateTracking:
    def test_age_increments_each_window(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        lc.on_window_complete(
            fired=True, fire_stall_windows=50, signal_cfg=sc, rules_cfg=rc, state=state
        )
        assert lc.signal_age_windows == 1

        lc.on_window_complete(
            fired=False, fire_stall_windows=50, signal_cfg=sc, rules_cfg=rc, state=state
        )
        assert lc.signal_age_windows == 2

    def test_fire_resets_windows_since_last_fire(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        # 3 windows without firing
        for _ in range(3):
            lc.on_window_complete(
                fired=False, fire_stall_windows=50, signal_cfg=sc, rules_cfg=rc, state=state
            )
        assert lc.windows_since_last_fire == 3

        # Fire resets counter
        lc.on_window_complete(
            fired=True, fire_stall_windows=50, signal_cfg=sc, rules_cfg=rc, state=state
        )
        assert lc.windows_since_last_fire == 0

    def test_no_stall_below_threshold(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        for _ in range(49):
            event = lc.on_window_complete(
                fired=False, fire_stall_windows=50, signal_cfg=sc, rules_cfg=rc, state=state
            )
            assert event is None
        assert not lc.is_idle


# ---------------------------------------------------------------------------
# Fire-stall detection
# ---------------------------------------------------------------------------


class TestFireStall:
    def test_fire_stall_triggers_at_threshold(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        event = None
        for _ in range(50):
            event = lc.on_window_complete(
                fired=False, fire_stall_windows=50, signal_cfg=sc, rules_cfg=rc, state=state
            )
        assert event is not None
        assert event.kind == "fire_stall"
        assert lc.is_idle
        assert lc.idle_reason == "fire_stall"

    def test_fire_stall_starts_shadow_tracking(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        for _ in range(50):
            lc.on_window_complete(
                fired=False, fire_stall_windows=50, signal_cfg=sc, rules_cfg=rc, state=state
            )

        assert lc.shadow is not None
        assert lc.shadow.age_at_idle == 50
        assert lc.shadow.windows_tracked == 0

    def test_fire_stall_only_triggers_once(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        events = []
        for _ in range(100):
            event = lc.on_window_complete(
                fired=False, fire_stall_windows=50, signal_cfg=sc, rules_cfg=rc, state=state
            )
            if event:
                events.append(event)

        # Only one fire_stall event (at window 50)
        assert len(events) == 1
        assert events[0].kind == "fire_stall"


# ---------------------------------------------------------------------------
# SPRT decay handling
# ---------------------------------------------------------------------------


class TestSPRTDecay:
    def test_decay_sets_idle(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        event = lc.on_sprt_verdict("DEAD", sc, rc, state)
        assert event is not None
        assert event.kind == "decay"
        assert lc.is_idle
        assert lc.idle_reason == "decay"

    def test_decay_starts_shadow(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        lc.signal_age_windows = 25
        lc.on_sprt_verdict("DEAD", sc, rc, state)

        assert lc.shadow is not None
        assert lc.shadow.age_at_idle == 25

    def test_alive_verdict_no_transition(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        event = lc.on_sprt_verdict("ALIVE", sc, rc, state)
        assert event is None
        assert not lc.is_idle

    def test_inconclusive_verdict_no_transition(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        event = lc.on_sprt_verdict("INCONCLUSIVE", sc, rc, state)
        assert event is None
        assert not lc.is_idle

    def test_decay_ignored_if_already_idle(self):
        """Second DEAD verdict while already idle should not emit event."""
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        lc.on_sprt_verdict("DEAD", sc, rc, state)
        event2 = lc.on_sprt_verdict("DEAD", sc, rc, state)
        assert event2 is None  # already idle


# ---------------------------------------------------------------------------
# Running entry average
# ---------------------------------------------------------------------------


class TestRunningEntryAvg:
    def test_single_entry(self):
        lc = SignalLifecycle()
        avg = lc.record_entry_price(0.85)
        assert avg == pytest.approx(0.85)
        assert lc.avg_entry_price == pytest.approx(0.85)

    def test_multiple_entries(self):
        lc = SignalLifecycle()
        lc.record_entry_price(0.80)
        lc.record_entry_price(0.90)
        assert lc.avg_entry_price == pytest.approx(0.85)

    def test_empty_returns_zero(self):
        lc = SignalLifecycle()
        assert lc.avg_entry_price == 0.0


# ---------------------------------------------------------------------------
# Shadow tracking
# ---------------------------------------------------------------------------


class TestShadowTracking:
    def _make_idle_lifecycle(self):
        """Create a lifecycle in IDLE:decay state with shadow active."""
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()
        lc.signal_age_windows = 10
        lc.on_sprt_verdict("DEAD", sc, rc, state)
        return lc

    def test_tick_shadow_accumulates(self):
        lc = self._make_idle_lifecycle()
        sig = _make_signal(bn_direction_from_open_pct=0.001)

        # Inside observation window (200 is between 180 and 240)
        lc.tick_shadow(sig, 200.0)
        assert lc.shadow.signal._n == 1

    def test_tick_shadow_noop_when_no_shadow(self):
        lc = SignalLifecycle()
        sig = _make_signal()
        # Should not raise
        lc.tick_shadow(sig, 200.0)

    @patch("shared.discord.send_shadow_tracking_result")
    def test_finalize_increments_windows(self, mock_discord):
        lc = self._make_idle_lifecycle()
        journal = FakeJournal()

        lc.finalize_shadow_window(
            outcome_dir=None,
            journal=journal,
            window_ts=1000,
            shadow_tracking_windows=500,
            mode="test",
        )
        assert lc.shadow.windows_tracked == 1
        assert len(journal.trades) == 1
        assert journal.trades[0].source == "shadow"

    @patch("shared.discord.send_shadow_tracking_result")
    def test_finalize_records_shadow_win(self, mock_discord):
        lc = self._make_idle_lifecycle()
        journal = FakeJournal()

        # Manually set shadow to have "fired" with conditions met
        sh = lc.shadow
        sh.signal._fired = True
        # Force _conditions_met to return True by accumulating enough data
        for i in range(10):
            sh.signal._accumulate(0.10, 230.0 - i, 0.0)

        lc.finalize_shadow_window(
            outcome_dir="up",
            journal=journal,
            window_ts=1000,
            shadow_tracking_windows=500,
            mode="test",
        )
        assert sh.fills == 1
        assert sh.wins == 1  # UP signal + outcome=up → win
        assert journal.trades[0].won is True

    @patch("shared.discord.send_shadow_tracking_result")
    def test_finalize_records_shadow_loss(self, mock_discord):
        lc = self._make_idle_lifecycle()
        journal = FakeJournal()

        sh = lc.shadow
        sh.signal._fired = True
        for i in range(10):
            sh.signal._accumulate(0.10, 230.0 - i, 0.0)

        lc.finalize_shadow_window(
            outcome_dir="down",
            journal=journal,
            window_ts=1000,
            shadow_tracking_windows=500,
            mode="test",
        )
        assert sh.fills == 1
        assert sh.wins == 0  # UP signal + outcome=down → loss

    @patch("shared.discord.send_shadow_tracking_result")
    def test_shadow_expiry(self, mock_discord):
        lc = self._make_idle_lifecycle()
        journal = FakeJournal()

        # Simulate reaching the tracking window limit
        lc.shadow.windows_tracked = 499  # will become 500 on finalize

        event = lc.finalize_shadow_window(
            outcome_dir=None,
            journal=journal,
            window_ts=1000,
            shadow_tracking_windows=500,
            mode="test",
        )
        assert event is not None
        assert event.kind == "shadow_complete"
        assert lc.shadow is None  # cleared after expiry
        mock_discord.assert_called_once()

    @patch("shared.discord.send_shadow_tracking_result")
    def test_shadow_resets_strategy_each_window(self, mock_discord):
        lc = self._make_idle_lifecycle()
        journal = FakeJournal()

        # Accumulate some data in shadow
        for i in range(5):
            lc.shadow.signal._accumulate(0.10, 230.0 - i, 0.0)
        assert lc.shadow.signal._n > 0

        lc.finalize_shadow_window(
            outcome_dir=None,
            journal=journal,
            window_ts=1000,
            shadow_tracking_windows=500,
            mode="test",
        )
        # Shadow strategy should be reset for next window
        assert lc.shadow.signal._n == 0

    def test_finalize_noop_when_no_shadow(self):
        lc = SignalLifecycle()
        journal = FakeJournal()
        event = lc.finalize_shadow_window(
            outcome_dir=None,
            journal=journal,
            window_ts=1000,
            shadow_tracking_windows=500,
            mode="test",
        )
        assert event is None


# ---------------------------------------------------------------------------
# Signal swap transitions
# ---------------------------------------------------------------------------


class TestSignalSwap:
    def test_same_signal_refresh_preserves_decay(self):
        lc = SignalLifecycle()
        lc.idle_reason = "decay"
        lc.windows_since_last_fire = 10

        lc.on_same_signal_refresh()

        assert lc.idle_reason == "decay"  # preserved
        assert lc.windows_since_last_fire == 0  # reset

    def test_same_signal_refresh_clears_fire_stall(self):
        lc = SignalLifecycle()
        lc.idle_reason = "fire_stall"
        lc.windows_since_last_fire = 50

        lc.on_same_signal_refresh()

        assert lc.idle_reason is None  # cleared
        assert lc.windows_since_last_fire == 0

    def test_reset_for_new_signal_clears_everything(self):
        lc = SignalLifecycle()
        lc.signal_age_windows = 100
        lc.windows_since_last_fire = 50
        lc.idle_reason = "decay"
        lc.running_entry_sum = 8.5
        lc.running_entry_count = 10

        lc.reset_for_new_signal()

        assert lc.signal_age_windows == 0
        assert lc.windows_since_last_fire == 0
        assert lc.idle_reason is None
        assert lc.running_entry_sum == 0.0
        assert lc.running_entry_count == 0

    def test_reset_preserves_shadow(self):
        """Shadow tracking continues for the OLD signal even after swap."""
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        lc.signal_age_windows = 10
        lc.on_sprt_verdict("DEAD", sc, rc, state)
        assert lc.shadow is not None

        lc.reset_for_new_signal()
        # Shadow should still be tracking the old signal
        assert lc.shadow is not None

    def test_clear_shadow(self):
        lc = SignalLifecycle()
        sc = make_signal_config()
        rc = make_rules_config()
        state = FakeMarketState()

        lc.on_sprt_verdict("DEAD", sc, rc, state)
        assert lc.shadow is not None

        lc.clear_shadow()
        assert lc.shadow is None
