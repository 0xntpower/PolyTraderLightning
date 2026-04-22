"""Tests for ResolutionManager — the single resolution pipeline.

Covers:
- Confirmed resolution (win + loss) with P&L verification
- Timeout fallback (with and without snapshot)
- Force-resolve (snapshot available vs conservative loss)
- SPRT decay trigger from resolution
- Bankroll update accuracy
- Pending lifecycle (create, clear, replace)
- Win-rate checkpoint dispatch
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fakes import make_signal_config
from market_data.state import MarketState
from strategy.resolution import (
    RESOLUTION_POLL_DELAY,
    RESOLUTION_TIMEOUT,
    ResolutionManager,
    ResolutionResult,
)
from strategy.signal import Direction
from strategy.window_tracker import ResolutionData

# ---------------------------------------------------------------------------
# Lightweight fakes for ResolutionManager dependencies
# ---------------------------------------------------------------------------


class FakeWindowTracker:
    """Fake that returns configurable resolution results."""

    def __init__(self, outcomes: dict[str, str | None] | None = None) -> None:
        # slug -> outcome ("up"/"down"/None)
        self._outcomes = outcomes or {}
        # slug -> (price_to_beat, final_price)
        self._oracle_prices: dict[str, tuple[float | None, float | None]] = {}

    async def fetch_market_resolution(self, slug: str) -> ResolutionData | None:
        outcome = self._outcomes.get(slug)
        if outcome is None:
            return None
        ptb, fp = self._oracle_prices.get(slug, (None, None))
        return ResolutionData(outcome=outcome, price_to_beat=ptb, final_price=fp)  # type: ignore[arg-type]  # test fake uses str values


class FakeBankrollTracker:
    def __init__(self, bankroll: float = 1000.0) -> None:
        self.bankroll = bankroll
        self.win_calls: list[tuple] = []
        self.loss_calls: list[tuple] = []

    def update_win(self, size: float, entry_price: float, fee: float = 0.0) -> float:
        profit = (size / entry_price) * (1.0 - entry_price) - fee
        self.bankroll += profit
        self.win_calls.append((size, entry_price, fee))
        return self.bankroll

    def update_loss(self, size: float, entry_price: float, fee: float = 0.0) -> float:
        loss = (size / entry_price) * entry_price + fee
        self.bankroll -= loss
        self.loss_calls.append((size, entry_price, fee))
        return self.bankroll


class FakePositionTracker:
    def __init__(self) -> None:
        self.records: list[tuple] = []
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self._saved = False

    def record_window(self, window_ts, pnl, traded=True, mode="live", balance=None):
        self.records.append((window_ts, pnl, traded, mode, balance))
        self.total_pnl += pnl

    def save_state(self, path, date_str, signal_id=""):
        self._saved = True


class FakeJournal:
    def __init__(self) -> None:
        self.trades: list = []

    def record_trade(self, record) -> bool:
        self.trades.append(record)
        return True

    @staticmethod
    def now_iso() -> str:
        return "2026-04-08T00:00:00+00:00"


class FakeDecayDetector:
    """Returns configurable SPRT verdicts."""

    def __init__(self, verdict: str = "INCONCLUSIVE", n_trades: int = 5) -> None:
        self._verdict = verdict
        self._n_trades = n_trades
        self._n_wins = 0

    def update(self, won: bool):
        self._n_trades += 1
        if won:
            self._n_wins += 1
        return MagicMock(
            verdict=self._verdict,
            n_trades=self._n_trades,
            n_wins=self._n_wins,
            llr=0.0,
        )


class FakeFeeTracker:
    def __init__(self, fee: float = 0.01) -> None:
        self._fee = fee

    def record_taker_fee(self, price: float, size: float) -> float:
        return self._fee


class FakeSessionStats:
    def __init__(self) -> None:
        self.trades: list[tuple] = []

    def record_trade(self, won: bool, pnl: float) -> None:
        self.trades.append((won, pnl))


class FakeLossTracker:
    def __init__(self) -> None:
        self.win_count = 0
        self.alert_calls: list = []

    def record_win(self):
        self.win_count += 1

    def check_and_alert(self, **kwargs):
        self.alert_calls.append(kwargs)


@dataclass
class FakeConfig:
    @dataclass
    class FakeRisk:
        max_consecutive_losses: int = 10

    risk: FakeRisk | None = None

    def __post_init__(self):
        if self.risk is None:
            self.risk = self.FakeRisk()


@dataclass
class FakeDataPaths:
    state: Path = Path("/tmp/test_state.json")


# ---------------------------------------------------------------------------
# Fixture: build a ResolutionManager with fake dependencies
# ---------------------------------------------------------------------------


@pytest.fixture
def resolution_env(tmp_path):
    """Create a ResolutionManager with all-fake dependencies."""
    window_tracker = FakeWindowTracker()
    fee_tracker = FakeFeeTracker(fee=0.01)
    bankroll_tracker = FakeBankrollTracker(bankroll=1000.0)
    position_tracker = FakePositionTracker()
    journal = FakeJournal()
    decay_detector = FakeDecayDetector()
    recent_outcomes = deque(maxlen=20)
    optimistic_outcomes = deque(maxlen=20)
    results_dir = tmp_path / "results"
    session_stats = FakeSessionStats()
    loss_tracker = FakeLossTracker()
    cfg = FakeConfig()
    paths = FakeDataPaths(state=tmp_path / "state.json")

    state = MarketState()
    from strategy.post_loss_cooldown import PostLossCooldown

    post_loss_cooldown = PostLossCooldown(
        enabled=False,
        loss_pct_threshold=2.0,
        cooldown_windows=1,
    )
    mgr = ResolutionManager(
        window_tracker=window_tracker,
        state=state,
        fee_tracker=fee_tracker,
        bankroll_tracker=bankroll_tracker,
        position_tracker=position_tracker,
        journal=journal,
        decay_detector=decay_detector,
        recent_outcomes=recent_outcomes,
        results_dir=results_dir,
        session_stats=session_stats,
        loss_tracker=loss_tracker,
        cfg=cfg,
        paths=paths,
        optimistic_outcomes=optimistic_outcomes,
        post_loss_cooldown=post_loss_cooldown,
    )

    return {
        "mgr": mgr,
        "window_tracker": window_tracker,
        "fee_tracker": fee_tracker,
        "bankroll_tracker": bankroll_tracker,
        "position_tracker": position_tracker,
        "journal": journal,
        "decay_detector": decay_detector,
        "recent_outcomes": recent_outcomes,
        "results_dir": results_dir,
        "session_stats": session_stats,
        "loss_tracker": loss_tracker,
        "cfg": cfg,
        "paths": paths,
    }


def _create_pending(
    mgr,
    side=Direction.UP,
    entry_price=0.85,
    size_usd=10.0,
    snapshot_outcome=None,
    slug="btc-up",
    entry_taker_fee=0.01,
    maker_usd=0.0,
    taker_usd=0.0,
):
    """Helper to create a pending resolution.

    ``entry_taker_fee`` defaults to 0.01 to match the FakeFeeTracker's fixed
    fee — post-PR-C the fee is pre-computed at aggregation time by
    ``window_handler`` rather than inside ``_resolve``, so the test helper
    mirrors that shape.
    """
    sc = make_signal_config(side=side)
    return mgr.create_pending(
        window_ts=1000,
        slug=slug,
        signal_cfg=sc,
        entry_price=entry_price,
        size_usd=size_usd,
        signal_age_windows=5,
        snapshot_outcome=snapshot_outcome,
        entry_taker_fee=entry_taker_fee,
        maker_usd=maker_usd,
        taker_usd=taker_usd,
    )


# ---------------------------------------------------------------------------
# Pending lifecycle tests
# ---------------------------------------------------------------------------


class TestPendingLifecycle:
    def test_initially_not_pending(self, resolution_env):
        mgr = resolution_env["mgr"]
        assert not mgr.is_pending
        assert mgr.pending is None

    def test_create_pending_sets_state(self, resolution_env):
        mgr = resolution_env["mgr"]
        _create_pending(mgr)
        assert mgr.is_pending
        assert mgr.pending is not None
        assert mgr.pending.slug == "btc-up"
        assert mgr.pending.entry_price == 0.85

    def test_clear_removes_pending(self, resolution_env):
        mgr = resolution_env["mgr"]
        _create_pending(mgr)
        mgr.clear()
        assert not mgr.is_pending

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_create_pending_force_resolves_existing(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]
        _create_pending(mgr, snapshot_outcome="up")
        result = _create_pending(mgr, slug="btc-down", side=Direction.DOWN, snapshot_outcome="down")
        # First pending was force-resolved
        assert result is not None
        assert isinstance(result, ResolutionResult)
        # New pending is the second one
        assert mgr.pending.slug == "btc-down"


# ---------------------------------------------------------------------------
# Confirmed resolution tests (win and loss)
# ---------------------------------------------------------------------------


class TestConfirmedResolution:
    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_win_resolution(self, mock_bet, mock_wr, resolution_env):
        """UP signal, outcome=up → WIN. Verify P&L is positive."""
        mgr = resolution_env["mgr"]
        wt = resolution_env["window_tracker"]
        wt._outcomes["btc-up"] = "up"

        _create_pending(mgr, side=Direction.UP, entry_price=0.85, size_usd=10.0)

        # Advance past poll delay
        mgr._pending.created_at = time.time() - RESOLUTION_POLL_DELAY - 1
        result = await mgr.tick(time.time())

        assert result is not None
        assert result.won is True
        assert result.pnl > 0
        assert not mgr.is_pending

    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_loss_resolution(self, mock_bet, mock_wr, resolution_env):
        """UP signal, outcome=down → LOSS. Verify P&L is negative."""
        mgr = resolution_env["mgr"]
        wt = resolution_env["window_tracker"]
        wt._outcomes["btc-up"] = "down"

        _create_pending(mgr, side=Direction.UP, entry_price=0.85, size_usd=10.0)
        mgr._pending.created_at = time.time() - RESOLUTION_POLL_DELAY - 1

        result = await mgr.tick(time.time())

        assert result is not None
        assert result.won is False
        assert result.pnl < 0

    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_pnl_formula_win(self, mock_bet, mock_wr, resolution_env):
        """Verify exact P&L formula: shares * (1 - entry) - taker_fee."""
        mgr = resolution_env["mgr"]
        wt = resolution_env["window_tracker"]
        wt._outcomes["btc-up"] = "up"

        entry = 0.85
        size = 10.0
        fee = 0.01
        shares = size / entry

        _create_pending(mgr, entry_price=entry, size_usd=size)
        mgr._pending.created_at = time.time() - RESOLUTION_POLL_DELAY - 1

        result = await mgr.tick(time.time())
        expected_pnl = round(shares * (1.0 - entry) - fee, 4)
        assert result.pnl == pytest.approx(expected_pnl, abs=0.001)

    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_pnl_formula_loss(self, mock_bet, mock_wr, resolution_env):
        """Verify exact P&L formula for loss: -(shares * entry) - taker_fee."""
        mgr = resolution_env["mgr"]
        wt = resolution_env["window_tracker"]
        wt._outcomes["btc-up"] = "down"

        entry = 0.85
        size = 10.0
        fee = 0.01
        shares = size / entry

        _create_pending(mgr, entry_price=entry, size_usd=size)
        mgr._pending.created_at = time.time() - RESOLUTION_POLL_DELAY - 1

        result = await mgr.tick(time.time())
        expected_pnl = round(-(shares * entry) - fee, 4)
        assert result.pnl == pytest.approx(expected_pnl, abs=0.001)

    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_combined_entry_passes_split_to_discord(self, mock_bet, mock_wr, resolution_env):
        """A maker-partial + taker-remainder combined entry must forward the
        USD split to send_bet_result so the WIN/LOSS embed can render the
        percent breakdown.
        """
        mgr = resolution_env["mgr"]
        wt = resolution_env["window_tracker"]
        wt._outcomes["btc-up"] = "up"

        _create_pending(
            mgr,
            entry_price=0.78,
            size_usd=3.16,
            entry_taker_fee=0.006,
            maker_usd=0.17,
            taker_usd=2.99,
        )
        mgr._pending.created_at = time.time() - RESOLUTION_POLL_DELAY - 1

        result = await mgr.tick(time.time())
        assert result.won is True

        assert mock_bet.call_count == 1
        kwargs = mock_bet.call_args.kwargs
        assert kwargs["maker_usd"] == pytest.approx(0.17, abs=0.001)
        assert kwargs["taker_usd"] == pytest.approx(2.99, abs=0.001)
        assert kwargs["outcome"] == "WIN"

    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_pure_maker_entry_passes_zero_split(self, mock_bet, mock_wr, resolution_env):
        """Pure-maker entries pass maker_usd>0, taker_usd=0 — embed renders
        the existing single-label form, not a split.
        """
        mgr = resolution_env["mgr"]
        wt = resolution_env["window_tracker"]
        wt._outcomes["btc-up"] = "up"

        _create_pending(
            mgr,
            entry_price=0.77,
            size_usd=3.12,
            entry_taker_fee=0.0,
            maker_usd=3.12,
            taker_usd=0.0,
        )
        mgr._pending.created_at = time.time() - RESOLUTION_POLL_DELAY - 1
        await mgr.tick(time.time())

        kwargs = mock_bet.call_args.kwargs
        assert kwargs["maker_usd"] == pytest.approx(3.12)
        assert kwargs["taker_usd"] == 0.0


# ---------------------------------------------------------------------------
# Timeout fallback tests
# ---------------------------------------------------------------------------


class TestTimeoutFallback:
    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_timeout_with_snapshot(self, mock_bet, mock_wr, resolution_env):
        """Timeout with snapshot_outcome='up' → uses snapshot as fallback."""
        mgr = resolution_env["mgr"]
        # No Gamma resolution will be returned
        _create_pending(mgr, side=Direction.UP, snapshot_outcome="up")
        mgr._pending.created_at = time.time() - RESOLUTION_TIMEOUT - 1

        result = await mgr.tick(time.time())
        assert result is not None
        assert result.won is True  # UP signal, outcome=up (from snapshot)

    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_timeout_without_snapshot_conservative_loss(
        self,
        mock_bet,
        mock_wr,
        resolution_env,
    ):
        """Timeout with no snapshot → conservative loss (opposite of side)."""
        mgr = resolution_env["mgr"]
        _create_pending(mgr, side=Direction.UP, snapshot_outcome=None)
        mgr._pending.created_at = time.time() - RESOLUTION_TIMEOUT - 1

        result = await mgr.tick(time.time())
        assert result is not None
        assert result.won is False  # Conservative: UP signal → outcome=down

    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_timeout_down_signal_no_snapshot(self, mock_bet, mock_wr, resolution_env):
        """DOWN signal, no snapshot → conservative loss means outcome='up'."""
        mgr = resolution_env["mgr"]
        _create_pending(mgr, side=Direction.DOWN, snapshot_outcome=None)
        mgr._pending.created_at = time.time() - RESOLUTION_TIMEOUT - 1

        result = await mgr.tick(time.time())
        assert result.won is False  # DOWN signal, outcome=up → loss


# ---------------------------------------------------------------------------
# Force-resolve tests
# ---------------------------------------------------------------------------


class TestForceResolve:
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_force_with_snapshot(self, mock_bet, mock_wr, resolution_env):
        """Force-resolve uses snapshot when available."""
        mgr = resolution_env["mgr"]
        _create_pending(mgr, side=Direction.UP, snapshot_outcome="up")

        result = mgr.force_resolve(context="test", signal_id="test_sig")
        assert result.won is True
        assert not mgr.is_pending

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_force_without_snapshot_conservative(self, mock_bet, mock_wr, resolution_env):
        """Force-resolve with no snapshot → conservative loss."""
        mgr = resolution_env["mgr"]
        _create_pending(mgr, side=Direction.UP, snapshot_outcome=None)

        result = mgr.force_resolve(context="test", signal_id="test_sig")
        assert result.won is False  # UP signal → outcome=down

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_force_down_signal_without_snapshot(self, mock_bet, mock_wr, resolution_env):
        """DOWN signal + no snapshot → outcome=up → loss."""
        mgr = resolution_env["mgr"]
        _create_pending(mgr, side=Direction.DOWN, snapshot_outcome=None)

        result = mgr.force_resolve(context="test", signal_id="test_sig")
        assert result.won is False  # DOWN + outcome=up → loss

    def test_force_resolve_no_pending_asserts(self, resolution_env):
        """force_resolve with no pending raises AssertionError."""
        mgr = resolution_env["mgr"]
        with pytest.raises(AssertionError):
            mgr.force_resolve(context="test", signal_id="test")


# ---------------------------------------------------------------------------
# Bookkeeping integration tests
# ---------------------------------------------------------------------------


class TestBookkeeping:
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_bankroll_updates_on_win(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]
        bt = resolution_env["bankroll_tracker"]
        initial = bt.bankroll

        _create_pending(mgr, side=Direction.UP, snapshot_outcome="up")
        mgr.force_resolve(context="test", signal_id="test")

        assert bt.bankroll > initial
        assert len(bt.win_calls) == 1

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_bankroll_updates_on_loss(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]
        bt = resolution_env["bankroll_tracker"]
        initial = bt.bankroll

        _create_pending(mgr, side=Direction.UP, snapshot_outcome="down")
        mgr.force_resolve(context="test", signal_id="test")

        assert bt.bankroll < initial
        assert len(bt.loss_calls) == 1

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_recent_outcomes_appended(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]
        outcomes = resolution_env["recent_outcomes"]

        _create_pending(mgr, side=Direction.UP, snapshot_outcome="up")
        mgr.force_resolve(context="test", signal_id="test")
        assert list(outcomes) == [1]

        _create_pending(mgr, side=Direction.UP, snapshot_outcome="down")
        mgr.force_resolve(context="test", signal_id="test")
        assert list(outcomes) == [1, 0]

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_position_tracker_records_window(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]
        pt = resolution_env["position_tracker"]

        _create_pending(mgr, snapshot_outcome="up")
        mgr.force_resolve(context="test", signal_id="test")

        assert len(pt.records) == 1
        assert pt.records[0][2] is True  # traded=True

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_journal_records_trade(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]
        journal = resolution_env["journal"]

        _create_pending(mgr, snapshot_outcome="up")
        mgr.force_resolve(context="test", signal_id="test")

        assert len(journal.trades) == 1
        record = journal.trades[0]
        assert record.fired is True
        assert record.filled is True
        assert record.won is True

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_session_stats_updated(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]
        stats = resolution_env["session_stats"]

        _create_pending(mgr, snapshot_outcome="up")
        mgr.force_resolve(context="test", signal_id="test")

        assert len(stats.trades) == 1
        assert stats.trades[0][0] is True  # won

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_loss_tracker_check_and_alert_on_loss(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]
        lt = resolution_env["loss_tracker"]

        _create_pending(mgr, snapshot_outcome="down")
        mgr.force_resolve(context="test", signal_id="test")

        assert len(lt.alert_calls) == 1  # check_and_alert was called

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_loss_tracker_record_win(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]
        lt = resolution_env["loss_tracker"]

        _create_pending(mgr, snapshot_outcome="up")
        mgr.force_resolve(context="test", signal_id="test")

        assert lt.win_count == 1

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_discord_notification_sent(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]

        _create_pending(mgr, snapshot_outcome="up")
        mgr.force_resolve(context="test", signal_id="test")

        mock_bet.assert_called_once()
        call_kwargs = mock_bet.call_args
        assert call_kwargs.kwargs["outcome"] == "WIN"


# ---------------------------------------------------------------------------
# SPRT decay trigger tests
# ---------------------------------------------------------------------------


class TestSPRTDecay:
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_dead_verdict_returned(self, mock_bet, mock_wr, resolution_env):
        """SPRT DEAD verdict is returned in ResolutionResult."""
        resolution_env["decay_detector"]._verdict = "DEAD"
        mgr = resolution_env["mgr"]

        _create_pending(mgr, snapshot_outcome="down")
        result = mgr.force_resolve(context="test", signal_id="test")

        assert result.verdict == "DEAD"

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_alive_verdict_returned(self, mock_bet, mock_wr, resolution_env):
        resolution_env["decay_detector"]._verdict = "ALIVE"
        mgr = resolution_env["mgr"]

        _create_pending(mgr, snapshot_outcome="up")
        result = mgr.force_resolve(context="test", signal_id="test")

        assert result.verdict == "ALIVE"

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_inconclusive_verdict_default(self, mock_bet, mock_wr, resolution_env):
        mgr = resolution_env["mgr"]

        _create_pending(mgr, snapshot_outcome="up")
        result = mgr.force_resolve(context="test", signal_id="test")

        assert result.verdict == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Polling timing tests
# ---------------------------------------------------------------------------


class TestPollingTiming:
    @pytest.mark.asyncio
    async def test_no_poll_before_delay(self, resolution_env):
        """Tick returns None if RESOLUTION_POLL_DELAY hasn't elapsed."""
        mgr = resolution_env["mgr"]
        _create_pending(mgr)
        # created_at is ~now, so delay not reached
        result = await mgr.tick(time.time())
        assert result is None
        assert mgr.is_pending  # still pending

    @pytest.mark.asyncio
    async def test_no_poll_before_interval(self, resolution_env):
        """Tick returns None if poll interval not met since last poll."""
        mgr = resolution_env["mgr"]
        _create_pending(mgr)
        now = time.time()
        mgr._pending.created_at = now - RESOLUTION_POLL_DELAY - 1
        mgr._pending.last_poll_at = now - 1  # polled 1s ago, interval is 5s

        result = await mgr.tick(now)
        assert result is None

    @pytest.mark.asyncio
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    async def test_polls_after_delay(self, mock_bet, mock_wr, resolution_env):
        """Tick polls Gamma after delay and resolves if outcome available."""
        mgr = resolution_env["mgr"]
        wt = resolution_env["window_tracker"]
        wt._outcomes["btc-up"] = "up"

        _create_pending(mgr)
        mgr._pending.created_at = time.time() - RESOLUTION_POLL_DELAY - 1

        result = await mgr.tick(time.time())
        assert result is not None
        assert result.won is True

    @pytest.mark.asyncio
    async def test_no_pending_returns_none(self, resolution_env):
        """Tick returns None when nothing is pending."""
        mgr = resolution_env["mgr"]
        result = await mgr.tick(time.time())
        assert result is None


# ---------------------------------------------------------------------------
# Oracle open price upgrade (Tier 2) tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JSONL record writing tests
# ---------------------------------------------------------------------------


class TestJSONLRecords:
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_writes_jsonl_file(self, mock_bet, mock_wr, resolution_env):
        """Resolution writes a JSONL record to the results directory."""
        mgr = resolution_env["mgr"]
        results_dir = resolution_env["results_dir"]

        _create_pending(mgr, snapshot_outcome="up")
        mgr.force_resolve(context="test", signal_id="test")

        # Should have created a .jsonl file
        jsonl_files = list(results_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        import orjson

        content = jsonl_files[0].read_bytes().strip()
        record = orjson.loads(content)
        assert record["window_ts"] == 1000
        # Canonical paper-schema fields: rule_simulated_fill replaces "fired"+"filled",
        # pnl_total/pnl_rules replace "pnl", balance_usd replaces "bankroll".
        assert record["rule_simulated_fill"] is True
        assert "pnl_total" in record
        assert "pnl_rules" in record
        assert "balance_usd" in record


# ---------------------------------------------------------------------------
# Down-signal resolution tests (direction symmetry)
# ---------------------------------------------------------------------------


class TestDirectionSymmetry:
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_down_signal_win(self, mock_bet, mock_wr, resolution_env):
        """DOWN signal + outcome=down → WIN."""
        mgr = resolution_env["mgr"]
        _create_pending(mgr, side=Direction.DOWN, snapshot_outcome="down")
        result = mgr.force_resolve(context="test", signal_id="test")
        assert result.won is True
        assert result.pnl > 0

    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_down_signal_loss(self, mock_bet, mock_wr, resolution_env):
        """DOWN signal + outcome=up → LOSS."""
        mgr = resolution_env["mgr"]
        _create_pending(mgr, side=Direction.DOWN, snapshot_outcome="up")
        result = mgr.force_resolve(context="test", signal_id="test")
        assert result.won is False
        assert result.pnl < 0


# ---------------------------------------------------------------------------
# decay_detector propagation after signal swap
# ---------------------------------------------------------------------------


class TestDecayDetectorPropagation:
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_new_detector_receives_updates_after_swap(self, mock_bet, mock_wr, resolution_env):
        """After setting a new decay_detector, resolutions update the new one."""
        mgr = resolution_env["mgr"]
        old_detector = resolution_env["decay_detector"]

        # Create a new detector and swap it in
        new_detector = FakeDecayDetector(verdict="INCONCLUSIVE", n_trades=0)
        mgr.decay_detector = new_detector

        # Resolve a trade — should update the NEW detector, not the old one
        _create_pending(mgr, snapshot_outcome="up")
        mgr.force_resolve(context="test", signal_id="test")

        # New detector should have received the update (n_trades incremented from 0)
        assert new_detector._n_trades == 1
        # Old detector should NOT have been updated (stayed at 5)
        assert old_detector._n_trades == 5


# ---------------------------------------------------------------------------
# JSONL record contains fee-adjusted PnL
# ---------------------------------------------------------------------------


class TestJSONLPnLIncludesFee:
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_jsonl_pnl_includes_taker_fee(self, mock_bet, mock_wr, resolution_env):
        """The JSONL record's pnl field must match the fee-adjusted pnl from _resolve()."""
        import json

        mgr = resolution_env["mgr"]
        results_dir = resolution_env["results_dir"]

        # entry=0.85, size=10.0 → shares=10/0.85≈11.7647
        # fee = 0.01 (from FakeFeeTracker)
        # win pnl = shares * (1 - 0.85) - 0.01 = 11.7647 * 0.15 - 0.01 ≈ 1.7547 - 0.01 = 1.7447
        _create_pending(mgr, entry_price=0.85, size_usd=10.0, snapshot_outcome="up")
        result = mgr.force_resolve(context="test", signal_id="test")

        # Read the JSONL file
        jsonl_files = list(results_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        record = json.loads(jsonl_files[0].read_text().strip())

        # JSONL pnl_total must equal the result pnl (which includes fee)
        assert record["pnl_total"] == result.pnl
        assert record["pnl_rules"] == result.pnl
        # And it should NOT equal the raw calculation without fee
        shares = 10.0 / 0.85
        raw_pnl_no_fee = round(shares * (1.0 - 0.85), 4)
        assert record["pnl_total"] != raw_pnl_no_fee


# ---------------------------------------------------------------------------
# create_pending force-resolves existing pending
# ---------------------------------------------------------------------------


class TestCreatePendingForceResolves:
    @patch("shared.discord.send_live_wr_checkpoint")
    @patch("shared.discord.send_bet_result")
    def test_create_pending_returns_result_for_old_pending(self, mock_bet, mock_wr, resolution_env):
        """Creating a new pending while one exists force-resolves the old one."""
        mgr = resolution_env["mgr"]
        sc = make_signal_config(side=Direction.UP)

        # Create first pending
        mgr.create_pending(
            window_ts=1000,
            slug="first",
            signal_cfg=sc,
            entry_price=0.85,
            size_usd=10.0,
            signal_age_windows=5,
            snapshot_outcome="up",
        )
        assert mgr.is_pending

        # Create second pending — should force-resolve the first
        result = mgr.create_pending(
            window_ts=2000,
            slug="second",
            signal_cfg=sc,
            entry_price=0.86,
            size_usd=12.0,
            signal_age_windows=6,
            snapshot_outcome=None,
        )

        # Should have returned the resolution of the first pending
        assert result is not None
        assert isinstance(result, ResolutionResult)
        assert result.won is True  # UP signal + snapshot "up" → win

        # New pending should now be the second one
        assert mgr.is_pending
        assert mgr.pending.window_ts == 2000
