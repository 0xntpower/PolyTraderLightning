"""Tests for WindowEventHandler — window boundary processing.

Covers:
- _window_decision_tag helper
- Paper window finalization flow
- Trade outcome extraction (paper and live paths)
- Skip streak tracking
- Shadow tracking delegation
- Signal swap transitions (same signal vs different signal)
- SPRT update + WR checkpoint
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from strategy.signal import Direction, Signal
from strategy.signal_lifecycle import SignalLifecycle
from strategy.window_handler import WindowEventHandler, _window_decision_tag

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeWindowSnapshot:
    window_ts: int = 1000
    chainlink_price: float = 100_050.0
    binance_price: float = 100_050.0


class FakeMarketState:
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
    binance_obi = 0.0

    def __init__(self):
        self.end_snapshot = None
        self.live_fills = {}


class FakeWindowTracker:
    def __init__(self):
        self._slug = "test-slug"

    def make_slug(self, ts):
        return self._slug

    async def on_new_window(self):
        return FakeWindowInfo()


@dataclass
class FakeWindowInfo:
    window_ts: int = 2000


class FakeResolutionManager:
    def __init__(self):
        self._is_pending = False
        self._create_result = None
        self._force_resolve_calls = []
        self._pending = None
        self._skip_writes = []

    @property
    def is_pending(self):
        return self._is_pending

    @property
    def pending(self):
        return self._pending

    def create_pending(self, **kwargs):
        return self._create_result

    def force_resolve(self, **kwargs):
        self._force_resolve_calls.append(kwargs)

    def write_skipped_window_record(self, **kwargs):
        self._skip_writes.append(kwargs)


class FakePositionTracker:
    def __init__(self):
        self.windows_recorded = []
        self.consecutive_losses = 0
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.windows_won = 0
        self.windows_traded = 0
        self._window_exposure_reset = False

    def record_window(self, window_ts, pnl, traded, mode, balance):
        self.windows_recorded.append(
            {
                "window_ts": window_ts,
                "pnl": pnl,
                "traded": traded,
                "mode": mode,
                "balance": balance,
            }
        )

    def save_state(self, path, date_str, signal_id=None):
        pass

    def reset_window_exposure(self):
        self._window_exposure_reset = True


class FakeFeeTracker:
    fee_rate = 0.002

    def compute_taker_fee(self, entry, shares):
        return round(shares * entry * self.fee_rate, 6)

    async def fetch_fee_rate(self, session, url):
        pass


class FakeBankrollTracker:
    bankroll = 100.0

    def update_win(self, size, entry, fee=0.0):
        pass

    def update_loss(self, size, entry, fee=0.0):
        pass

    def sync_from_api(self, balance):
        pass


class FakeSessionStats:
    signals_received = 0
    net_pnl = 0.0

    def record_trade(self, won, pnl):
        pass

    def record_window(self, bet_scale, vol, chop):
        pass


class FakeSkipTracker:
    def __init__(self):
        self.fills = 0
        self.skips = []

    def record_fill(self):
        self.fills += 1

    def record_skip(self, reason):
        self.skips.append(reason)

    def check_alert(self, **kwargs):
        pass


class FakeLossTracker:
    def __init__(self):
        self.alerts = []
        self.wins = 0

    def check_and_alert(self, **kwargs):
        self.alerts.append(kwargs)

    def record_win(self):
        self.wins += 1


class FakeSqueezeTracker:
    def update(self, **kwargs):
        pass


class FakeJournal:
    def __init__(self):
        self.trades = []

    def record_trade(self, record):
        self.trades.append(record)
        return True

    @staticmethod
    def now_iso():
        return "2026-04-08T00:00:00+00:00"


class FakeVolTracker:
    current_stddev_pct = 0.05
    n_returns = 100

    def record_close(self, price):
        pass

    def save_cache(self, path):
        pass

    def update_stddev(self):
        pass

    def signed_return_t_stat(self, min_samples=4):
        return 0.0


@dataclass
class FakeChopStats:
    n_ticks: int = 10
    direction_flips: int = 1
    delta_range_pct: float = 0.05


class FakeChopDetector:
    avg_flips = 2.0
    n_windows = 50

    def finalize_window(self):
        return FakeChopStats()

    def save_cache(self, path):
        pass


class FakeOutcomeTracker:
    def record_outcome(self, direction):
        pass

    def save_cache(self, path):
        pass

    def direction_agreement(self, side):
        return 0.5

    def summary(self):
        return "balanced"


class FakePaperRecord:
    def __init__(self, *, pnl=0.05, entry_price=0.85, filled=True):
        self.pnl_total = pnl
        self.rule_entry_price = entry_price
        self.rule_simulated_fill = filled


class FakeOrderManager:
    def __init__(self, mode="paper"):
        self.mode = mode
        self.balance = 100.0
        self._current_record = None
        self._reset_calls = []

    def finalize_window(self, delta_pct, direction, outcome, snapshot=None):
        return self._current_record or FakePaperRecord(filled=False, pnl=0.0, entry_price=0.0)

    def reset_window(self, window_ts):
        self._reset_calls.append(window_ts)

    async def refresh_balance(self):
        return self.balance


class FakeStrategy:
    def __init__(self, signal_cfg=None):
        from fakes import make_signal_config

        self.signal_cfg = signal_cfg or make_signal_config()
        self._fired = False
        self._order_placed = False
        self.last_entry_price = 0.0
        self.last_size_usd = 0.0
        self.bet_scale = 1.0
        self.sprt_factor = 1.0
        self.kelly_wr_result = None
        self.bankroll = 100.0
        self.sizing_cfg = None
        self.erosion_cfg = None
        self.warmup_active = False
        self._reset_called = False

    def reset(self):
        self._reset_called = True


class FakeDecayDetector:
    def __init__(self):
        self.state = FakeDecayState()
        self._updates = []

    def update(self, won):
        self._updates.append(won)
        return self.state

    def reset(self, p_dead=None):
        pass


@dataclass
class FakeDecayState:
    n_trades: int = 5
    n_wins: int = 4
    llr: float = 1.5
    rolling_win_rate: float = 0.80
    p_alive: float = 0.85
    p_dead: float = 0.55
    verdict: str = "INCONCLUSIVE"
    boundary_alive: float = 2.94
    boundary_dead: float = -2.94


class FakePath:
    """Fake pathlib.Path that avoids real filesystem access."""

    def exists(self):
        return False

    def read_text(self):
        return "{}"

    def write_text(self, text):
        pass


class FakeDataPaths:
    state = FakePath()
    vol_cache = FakePath()
    chop_cache = FakePath()
    outcome_cache = FakePath()


class FakeConfig:
    def __init__(self):
        from fakes import (
            make_erosion_config,
            make_lifecycle_config,
            make_regime_config,
            make_rules_config,
            make_sizing_config,
        )

        self.rules_strategy = make_rules_config()
        self.signal_lifecycle = make_lifecycle_config()
        self.sizing = make_sizing_config()
        self.regime = make_regime_config()
        self.erosion = make_erosion_config()
        self.risk = FakeRiskConfig()
        self.connections = FakeConnectionsConfig()


class FakeRiskConfig:
    max_consecutive_losses = 5
    cancel_unfilled_at_sec = 10


class FakeConnectionsConfig:
    clob_rest = "http://test"
    binance_stale_sec = 30.0
    chainlink_stale_sec = 30.0
    clob_book_stale_sec = 30.0


# ---------------------------------------------------------------------------
# Builder for WindowEventHandler with all fakes
# ---------------------------------------------------------------------------


def _make_handler(**overrides):
    """Create a WindowEventHandler with all-fake dependencies."""
    defaults = {
        "cfg": FakeConfig(),
        "state": FakeMarketState(),
        "window_tracker": FakeWindowTracker(),
        "resolution_mgr": FakeResolutionManager(),
        "lifecycle": SignalLifecycle(),
        "position_tracker": FakePositionTracker(),
        "fee_tracker": FakeFeeTracker(),
        "bankroll_tracker": FakeBankrollTracker(),
        "recent_outcomes": deque(maxlen=50),
        "optimistic_outcomes": deque(maxlen=50),
        "session_stats": FakeSessionStats(),
        "skip_tracker": FakeSkipTracker(),
        "loss_tracker": FakeLossTracker(),
        "squeeze_tracker": FakeSqueezeTracker(),
        "journal": FakeJournal(),
        "vol_tracker": FakeVolTracker(),
        "chop_detector": FakeChopDetector(),
        "outcome_tracker": FakeOutcomeTracker(),
        "paths": FakeDataPaths(),
        "session": MagicMock(),
        "pending_signal_mgr": None,
        "bot_start_time": 0.0,
        "build_strategy_fn": lambda cfg, state, data: None,
        "signal_cfg_to_dict_fn": lambda sc: {},
    }
    defaults.update(overrides)
    return WindowEventHandler(**defaults)


# ---------------------------------------------------------------------------
# _window_decision_tag tests
# ---------------------------------------------------------------------------


class TestWindowDecisionTag:
    def test_win(self):
        tag, why = _window_decision_tag(True, True, True, True, None)
        assert tag == "[WIN]"
        assert why == ""

    def test_loss(self):
        tag, why = _window_decision_tag(True, False, True, True, None)
        assert tag == "[LOSS]"
        assert why == ""

    def test_flat(self):
        tag, why = _window_decision_tag(True, None, True, True, None)
        assert tag == "[FLAT]"

    def test_skip_idle(self):
        tag, why = _window_decision_tag(False, None, False, False, "decay")
        assert tag == "[SKIP]"
        assert "idle:decay" in why

    def test_skip_no_fire(self):
        tag, why = _window_decision_tag(False, None, False, False, None)
        assert tag == "[SKIP]"
        assert "no fire" in why

    def test_skip_conditions_not_met(self):
        tag, why = _window_decision_tag(False, None, True, False, None)
        assert "conditions not met" in why

    def test_skip_order_not_filled(self):
        tag, why = _window_decision_tag(False, None, True, True, None)
        assert "order not filled" in why


# ---------------------------------------------------------------------------
# Trade outcome: paper mode
# ---------------------------------------------------------------------------


class TestPaperTradeOutcome:
    @pytest.mark.asyncio
    async def test_paper_win_records_journal_and_session(self):
        journal = FakeJournal()
        session_stats = FakeSessionStats()
        skip_tracker = FakeSkipTracker()
        loss_tracker = FakeLossTracker()
        recent_outcomes = deque(maxlen=50)

        handler = _make_handler(
            journal=journal,
            session_stats=session_stats,
            skip_tracker=skip_tracker,
            loss_tracker=loss_tracker,
            recent_outcomes=recent_outcomes,
        )

        order_mgr = FakeOrderManager(mode="paper")
        order_mgr._current_record = FakePaperRecord(pnl=0.05, entry_price=0.85, filled=True)

        strategy = FakeStrategy()
        strategy._fired = True
        strategy._order_placed = True

        strategy, _ = handler._process_trade_outcome(
            strategy,
            order_mgr,
            FakeDecayDetector(),
            last_window_ts=1000,
        )

        # Journal recorded
        assert len(journal.trades) == 1
        rec = journal.trades[0]
        assert rec.fired is True
        assert rec.won is True
        assert rec.pnl == 0.05

        # Skip tracker: fill recorded
        assert skip_tracker.fills == 1

        # Loss tracker: win recorded
        assert loss_tracker.wins == 1

    @pytest.mark.asyncio
    async def test_paper_loss_triggers_loss_alert(self):
        loss_tracker = FakeLossTracker()
        position_tracker = FakePositionTracker()
        position_tracker.consecutive_losses = 3
        recent_outcomes = deque(maxlen=50)

        handler = _make_handler(
            loss_tracker=loss_tracker,
            position_tracker=position_tracker,
            recent_outcomes=recent_outcomes,
        )

        order_mgr = FakeOrderManager(mode="paper")
        order_mgr._current_record = FakePaperRecord(pnl=-0.10, entry_price=0.85, filled=True)

        strategy = FakeStrategy()
        strategy._fired = True

        handler._process_trade_outcome(
            strategy,
            order_mgr,
            FakeDecayDetector(),
            last_window_ts=1000,
        )

        assert len(loss_tracker.alerts) == 1
        assert loss_tracker.alerts[0]["streak"] == 3

    @pytest.mark.asyncio
    async def test_paper_skip_records_skip_reason(self):
        skip_tracker = FakeSkipTracker()
        handler = _make_handler(skip_tracker=skip_tracker)

        order_mgr = FakeOrderManager(mode="paper")
        # No fill, not fired
        strategy = FakeStrategy()

        handler._process_trade_outcome(
            strategy,
            order_mgr,
            FakeDecayDetector(),
            last_window_ts=1000,
        )

        assert skip_tracker.fills == 0
        assert len(skip_tracker.skips) == 1
        assert "no_signal_fire" in skip_tracker.skips[0]


# ---------------------------------------------------------------------------
# v2.9 paper-mode Kelly/bankroll reconcile
# ---------------------------------------------------------------------------


from execution.paper_trading import PaperOrderManager as _RealPaperOrderManager


class FakePaperOrderManager(_RealPaperOrderManager):
    """Subclass of the real PaperOrderManager that skips heavy init.

    Exists solely so the `isinstance(order_mgr, PaperOrderManager)` check in
    window_handler._process_trade_outcome passes for unit tests, without
    having to build a full Config / MarketState / RiskRegistry / FeeTracker
    fixture. Satisfies every attribute _process_trade_outcome touches on
    the order manager:
    - ``mode`` (class attribute on the parent, defaults to "paper")
    - ``_current_record`` (the WindowRecord for the finalised window)
    - ``balance`` (inherited @property reads ``_balance``)
    """

    def __init__(self, *, balance: float = 900.0) -> None:
        # Deliberately skip super().__init__ — we don't need the order book,
        # risk registry, fee tracker, or results directory for these tests.
        self._balance = balance
        self._current_record: object | None = None


class TestPaperBankrollReconcile:
    """v2.9 per-settle paper-mode Kelly/bankroll reconcile.

    v2.8 drifted: paper balance ended at $872 while Kelly bankroll ended at
    $515 because update_win/update_loss ignores CUSUM early-exit sell prices.
    After v2.9, every settled paper trade calls sync_from_api(order_mgr.balance)
    so Kelly sizes off the authoritative paper balance.
    """

    @pytest.mark.asyncio
    async def test_paper_win_reconciles_bankroll_from_order_mgr(self):
        bankroll = FakeBankrollTracker()
        bankroll.sync_from_api = MagicMock()
        handler = _make_handler(bankroll_tracker=bankroll)

        order_mgr = FakePaperOrderManager(balance=925.50)
        order_mgr._current_record = FakePaperRecord(pnl=0.05, entry_price=0.85, filled=True)

        strategy = FakeStrategy()
        strategy._fired = True
        strategy._order_placed = True
        strategy.last_size_usd = 10.0

        handler._process_trade_outcome(
            strategy,
            order_mgr,
            FakeDecayDetector(),
            last_window_ts=1000,
        )

        bankroll.sync_from_api.assert_called_once_with(925.50)

    @pytest.mark.asyncio
    async def test_paper_loss_still_reconciles(self):
        """Losing trades must reconcile too — CUSUM early-exits land here."""
        bankroll = FakeBankrollTracker()
        bankroll.sync_from_api = MagicMock()
        handler = _make_handler(bankroll_tracker=bankroll)

        order_mgr = FakePaperOrderManager(balance=872.10)
        order_mgr._current_record = FakePaperRecord(pnl=-0.10, entry_price=0.85, filled=True)

        strategy = FakeStrategy()
        strategy._fired = True
        strategy._order_placed = True
        strategy.last_size_usd = 10.0

        handler._process_trade_outcome(
            strategy,
            order_mgr,
            FakeDecayDetector(),
            last_window_ts=1000,
        )

        bankroll.sync_from_api.assert_called_once_with(872.10)

    @pytest.mark.asyncio
    async def test_paper_unfilled_does_not_reconcile(self):
        """No fill → no settled trade → no reconcile."""
        bankroll = FakeBankrollTracker()
        bankroll.sync_from_api = MagicMock()
        handler = _make_handler(bankroll_tracker=bankroll)

        order_mgr = FakePaperOrderManager(balance=1000.0)
        order_mgr._current_record = FakePaperRecord(pnl=0.0, entry_price=0.0, filled=False)

        strategy = FakeStrategy()
        strategy._fired = True

        handler._process_trade_outcome(
            strategy,
            order_mgr,
            FakeDecayDetector(),
            last_window_ts=1000,
        )

        bankroll.sync_from_api.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_mode_does_not_trigger_paper_reconcile(self):
        """The per-settle reconcile is paper-only — live keeps the separate
        throttled sync path.
        """
        bankroll = FakeBankrollTracker()
        bankroll.sync_from_api = MagicMock()
        state = FakeMarketState()
        state.live_fills = {}  # no fill → _has_resolved_outcome is False anyway
        handler = _make_handler(bankroll_tracker=bankroll, state=state)

        order_mgr = FakeOrderManager(mode="live")

        strategy = FakeStrategy()
        strategy._fired = True
        strategy._order_placed = True

        handler._process_trade_outcome(
            strategy,
            order_mgr,
            FakeDecayDetector(),
            last_window_ts=1000,
        )

        bankroll.sync_from_api.assert_not_called()


# ---------------------------------------------------------------------------
# Trade outcome: live mode
# ---------------------------------------------------------------------------


@dataclass
class FakeLiveFill:
    order_id: str = "order123456789"
    price: float = 0.84
    size: float = 10.0
    size_usd: float = 8.40
    is_maker: bool = False


class TestLiveTradeOutcome:
    @pytest.mark.asyncio
    async def test_live_fill_creates_pending_resolution(self):
        resolution_mgr = FakeResolutionManager()
        state = FakeMarketState()
        state.live_fills = {"order123456789": FakeLiveFill()}

        handler = _make_handler(
            state=state,
            resolution_mgr=resolution_mgr,
        )

        order_mgr = FakeOrderManager(mode="live")
        strategy = FakeStrategy()
        strategy._fired = True
        strategy._order_placed = True

        # Patch compute_signal_from_snapshot to avoid needing real snapshot
        with patch("strategy.window_handler.compute_signal_from_snapshot"):
            handler._process_trade_outcome(
                strategy,
                order_mgr,
                FakeDecayDetector(),
                last_window_ts=1000,
            )

        # Skip tracker should show a fill
        assert handler._skip_tracker.fills == 1

    @pytest.mark.asyncio
    async def test_live_no_fill_records_no_trade(self):
        position_tracker = FakePositionTracker()
        state = FakeMarketState()
        state.live_fills = {}  # No fills

        handler = _make_handler(
            state=state,
            position_tracker=position_tracker,
        )

        order_mgr = FakeOrderManager(mode="live")
        strategy = FakeStrategy()

        handler._process_trade_outcome(
            strategy,
            order_mgr,
            FakeDecayDetector(),
            last_window_ts=1000,
        )

        assert len(position_tracker.windows_recorded) == 1
        assert position_tracker.windows_recorded[0]["traded"] is False


# ---------------------------------------------------------------------------
# SPRT update
# ---------------------------------------------------------------------------


class TestSPRTUpdate:
    def test_sprt_dead_triggers_idle(self):
        lifecycle = SignalLifecycle()
        handler = _make_handler(lifecycle=lifecycle)

        decay_detector = FakeDecayDetector()
        decay_detector.state.verdict = "DEAD"
        decay_detector.state.rolling_win_rate = 0.50

        from fakes import make_signal_config

        sc = make_signal_config()

        with (
            patch("shared.discord.send_sprt_decay_alert"),
            patch("shared.discord.send_live_wr_checkpoint"),
        ):
            handler._update_sprt(decay_detector, True, 0.85, sc, "paper")

        assert lifecycle.is_idle
        assert lifecycle.idle_reason == "decay"

    def test_sprt_alive_resets_detector(self):
        lifecycle = SignalLifecycle()
        handler = _make_handler(lifecycle=lifecycle)

        decay_detector = FakeDecayDetector()
        decay_detector.state.verdict = "ALIVE"

        from fakes import make_signal_config

        sc = make_signal_config()

        with patch("shared.discord.send_live_wr_checkpoint"):
            handler._update_sprt(decay_detector, True, 0.85, sc, "paper")

        assert not lifecycle.is_idle

    @patch("shared.discord.send_live_wr_checkpoint")
    def test_wr_checkpoint_sent_at_milestone(self, mock_wr):
        lifecycle = SignalLifecycle()
        handler = _make_handler(lifecycle=lifecycle)

        decay_detector = FakeDecayDetector()
        decay_detector.state.n_trades = 25  # milestone
        decay_detector.state.verdict = "INCONCLUSIVE"

        from fakes import make_signal_config

        sc = make_signal_config()

        handler._update_sprt(decay_detector, True, 0.85, sc, "paper")
        mock_wr.assert_called_once()


# ---------------------------------------------------------------------------
# Shadow tracking delegation
# ---------------------------------------------------------------------------


class TestShadowTracking:
    @patch("shared.discord.send_shadow_tracking_result")
    def test_finalize_shadow_delegates_to_lifecycle(self, mock_discord):
        lifecycle = SignalLifecycle()
        journal = FakeJournal()

        # Put lifecycle in idle with shadow
        from fakes import make_rules_config, make_signal_config

        sc = make_signal_config()
        rc = make_rules_config()
        lifecycle.signal_age_windows = 10
        lifecycle.on_sprt_verdict("DEAD", sc, rc, FakeMarketState())
        assert lifecycle.shadow is not None

        handler = _make_handler(lifecycle=lifecycle, journal=journal)
        order_mgr = FakeOrderManager(mode="paper")

        handler._process_shadow_tracking(last_window_ts=1000, order_mgr=order_mgr)

        # Shadow window finalized — journal should have a shadow record
        assert len(journal.trades) == 1
        assert journal.trades[0].source == "shadow"


# ---------------------------------------------------------------------------
# Signal swap
# ---------------------------------------------------------------------------


class TestSignalSwap:
    def test_no_pending_signal_returns_same(self):
        handler = _make_handler(pending_signal_mgr=None)
        strategy = FakeStrategy()
        dd = FakeDecayDetector()

        result_s, result_dd = handler._handle_signal_swap(strategy, FakeOrderManager(), dd)
        assert result_s is strategy
        assert result_dd is dd

    @patch("shared.discord.send_signal_updated")
    def test_same_signal_preserves_sprt(self, mock_discord):
        from fakes import make_signal_config

        sc = make_signal_config()
        old_strategy = FakeStrategy(signal_cfg=sc)
        new_strategy = FakeStrategy(signal_cfg=sc)

        pending_mgr = MagicMock()
        pending_mgr.take_pending.return_value = ({"mock": "data"}, "summary.json")

        lifecycle = SignalLifecycle()
        lifecycle.signal_age_windows = 50

        handler = _make_handler(
            lifecycle=lifecycle,
            pending_signal_mgr=pending_mgr,
            build_strategy_fn=lambda cfg, state, data: new_strategy,
        )

        result_s, result_dd = handler._handle_signal_swap(
            old_strategy,
            FakeOrderManager(),
            FakeDecayDetector(),
        )

        assert result_s is new_strategy
        # Lifecycle preserved — same signal refresh
        assert lifecycle.signal_age_windows == 50
        assert lifecycle.windows_since_last_fire == 0

    @patch("shared.discord.send_signal_updated")
    def test_different_signal_resets_lifecycle(self, mock_discord):
        from fakes import make_signal_config

        old_sc = make_signal_config(side=Direction.UP)
        new_sc = make_signal_config(side=Direction.DOWN)  # Different signal_id
        old_strategy = FakeStrategy(signal_cfg=old_sc)
        new_strategy = FakeStrategy(signal_cfg=new_sc)

        pending_mgr = MagicMock()
        pending_mgr.take_pending.return_value = ({"mock": "data"}, "summary.json")

        lifecycle = SignalLifecycle()
        lifecycle.signal_age_windows = 50
        recent_outcomes = deque([1, 0, 1], maxlen=50)

        handler = _make_handler(
            lifecycle=lifecycle,
            recent_outcomes=recent_outcomes,
            pending_signal_mgr=pending_mgr,
            build_strategy_fn=lambda cfg, state, data: new_strategy,
        )

        with patch("shared.decay_detector.DecayDetector") as MockDD:
            MockDD.return_value = FakeDecayDetector()
            result_s, result_dd = handler._handle_signal_swap(
                old_strategy,
                FakeOrderManager(),
                FakeDecayDetector(),
            )

        assert result_s is new_strategy
        assert lifecycle.signal_age_windows == 0
        assert len(recent_outcomes) == 0

    @patch("shared.discord.send_signal_updated")
    def test_different_signal_force_resolves_pending(self, mock_discord):
        from fakes import make_signal_config

        old_sc = make_signal_config(side=Direction.UP)
        new_sc = make_signal_config(side=Direction.DOWN)
        old_strategy = FakeStrategy(signal_cfg=old_sc)
        new_strategy = FakeStrategy(signal_cfg=new_sc)

        pending_mgr = MagicMock()
        pending_mgr.take_pending.return_value = ({"mock": "data"}, "summary.json")

        resolution_mgr = FakeResolutionManager()
        resolution_mgr._is_pending = True

        handler = _make_handler(
            resolution_mgr=resolution_mgr,
            pending_signal_mgr=pending_mgr,
            build_strategy_fn=lambda cfg, state, data: new_strategy,
        )

        with patch("shared.decay_detector.DecayDetector") as MockDD:
            MockDD.return_value = FakeDecayDetector()
            handler._handle_signal_swap(
                old_strategy,
                FakeOrderManager(),
                FakeDecayDetector(),
            )

        assert len(resolution_mgr._force_resolve_calls) == 1


# ---------------------------------------------------------------------------
# Full on_window_transition integration
# ---------------------------------------------------------------------------


class TestOnWindowTransition:
    @pytest.mark.asyncio
    @patch("strategy.window_handler.compute_signal_from_snapshot")
    @patch("strategy.window_handler.compute_signal")
    async def test_first_window_skips_finalization(self, mock_signal, mock_snap_signal):
        """When last_window_ts=0, phases 1-4 are skipped."""
        handler = _make_handler()
        strategy = FakeStrategy()
        order_mgr = FakeOrderManager(mode="paper")

        result = await handler.on_window_transition(
            last_window_ts=0,
            strategy=strategy,
            order_mgr=order_mgr,
            decay_detector=FakeDecayDetector(),
        )

        assert result.last_window_ts == 2000  # new window from FakeWindowTracker
        assert result.snapshot_taken is False
        assert len(handler._journal.trades) == 0  # no finalization

    @pytest.mark.asyncio
    @patch("strategy.window_handler.compute_signal_from_snapshot")
    @patch("strategy.window_handler.compute_signal")
    async def test_normal_transition_finalizes_and_resets(self, mock_signal, mock_snap_signal):
        """Normal window transition: finalizes previous, sets up new."""
        mock_signal.return_value = Signal(
            delta_pct=0.05,
            direction=Direction.UP,
            feeds_agree=True,
            bn_direction_from_open_pct=0.001,
            cl_direction_from_open_pct=0.001,
            poly_spread_up=0.0,
            poly_spread_down=0.0,
            binance_obi=0.0,
            time_remaining=200.0,
        )
        mock_snap_signal.return_value = mock_signal.return_value

        handler = _make_handler()
        strategy = FakeStrategy()
        order_mgr = FakeOrderManager(mode="paper")
        order_mgr._current_record = FakePaperRecord(filled=False, pnl=0.0, entry_price=0.0)

        result = await handler.on_window_transition(
            last_window_ts=1000,
            strategy=strategy,
            order_mgr=order_mgr,
            decay_detector=FakeDecayDetector(),
        )

        assert result.last_window_ts == 2000
        assert result.strategy._reset_called


# ---------------------------------------------------------------------------
# Bankroll sync timing
# ---------------------------------------------------------------------------


class TestBankrollSyncTiming:
    @pytest.mark.asyncio
    @patch("strategy.window_handler.compute_signal_from_snapshot")
    @patch("strategy.window_handler.compute_signal")
    async def test_recent_sync_prevents_redundant_call(self, mock_signal, mock_snap_signal):
        """If _last_bankroll_sync is recent, sync_from_api should NOT be called."""
        import time as _time

        mock_signal.return_value = Signal(
            delta_pct=0.05,
            direction=Direction.UP,
            feeds_agree=True,
            bn_direction_from_open_pct=0.001,
            cl_direction_from_open_pct=0.001,
            poly_spread_up=0.0,
            poly_spread_down=0.0,
            binance_obi=0.0,
            time_remaining=200.0,
        )
        mock_snap_signal.return_value = mock_signal.return_value

        bankroll = FakeBankrollTracker()
        bankroll.sync_from_api = MagicMock()

        handler = _make_handler(bankroll_tracker=bankroll)
        # Simulate a recent sync (within 1800s)
        handler._last_bankroll_sync = _time.time()

        order_mgr = FakeOrderManager(mode="live")

        await handler.on_window_transition(
            last_window_ts=0,
            strategy=FakeStrategy(),
            order_mgr=order_mgr,
            decay_detector=FakeDecayDetector(),
        )

        bankroll.sync_from_api.assert_not_called()

    @pytest.mark.asyncio
    @patch("strategy.window_handler.compute_signal_from_snapshot")
    @patch("strategy.window_handler.compute_signal")
    async def test_old_sync_allows_refresh(self, mock_signal, mock_snap_signal):
        """If _last_bankroll_sync is old (>1800s), sync_from_api should be called."""
        import time as _time

        mock_signal.return_value = Signal(
            delta_pct=0.05,
            direction=Direction.UP,
            feeds_agree=True,
            bn_direction_from_open_pct=0.001,
            cl_direction_from_open_pct=0.001,
            poly_spread_up=0.0,
            poly_spread_down=0.0,
            binance_obi=0.0,
            time_remaining=200.0,
        )
        mock_snap_signal.return_value = mock_signal.return_value

        bankroll = FakeBankrollTracker()
        bankroll.sync_from_api = MagicMock()

        handler = _make_handler(bankroll_tracker=bankroll)
        # Simulate an old sync (>1800s ago)
        handler._last_bankroll_sync = _time.time() - 2000.0

        order_mgr = FakeOrderManager(mode="live")
        order_mgr.balance = 500.0

        await handler.on_window_transition(
            last_window_ts=0,
            strategy=FakeStrategy(),
            order_mgr=order_mgr,
            decay_detector=FakeDecayDetector(),
        )

        bankroll.sync_from_api.assert_called_once()


# ---------------------------------------------------------------------------
# _compute_kelly_context sets strategy fields
# ---------------------------------------------------------------------------


class TestComputeKellyContext:
    @patch("shared.discord.send_risk_level")
    def test_kelly_context_populates_strategy_fields(self, mock_risk):
        from fakes import make_lifecycle_config

        cfg = FakeConfig()
        cfg.signal_lifecycle = make_lifecycle_config(bet_scaling_enabled=True)

        handler = _make_handler(cfg=cfg)
        strategy = FakeStrategy()
        dd = FakeDecayDetector()
        order_mgr = FakeOrderManager(mode="paper")

        handler._compute_kelly_context(strategy, order_mgr, dd)

        assert strategy.sprt_factor is not None
        assert strategy.kelly_wr_result is not None
        assert strategy.bankroll == handler._bankroll_tracker.bankroll
        assert strategy.sizing_cfg is cfg.sizing
        assert strategy.erosion_cfg is cfg.erosion
