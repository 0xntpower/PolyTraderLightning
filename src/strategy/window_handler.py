"""WindowEventHandler — processes everything that happens at a window boundary.

Consolidates the ~570-line ``if wts != last_window_ts:`` block from
``_strategy_loop()`` into a testable, well-structured class.

The handler is organized into sequential phases:
  1. Finalize previous window (paper P&L, vol/chop/outcome recording)
  2. Process trade outcome (paper immediate, live deferred to resolution)
  3. Update signal lifecycle (age, fire-rate, SPRT)
  4. Handle shadow tracking
  5. Handle signal swap (IPC)
  6. Setup new window (reset strategy, Kelly context, regime logging)
"""

from __future__ import annotations

import json as _json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import (
    TYPE_CHECKING,
    Any,  # IPC signal dicts with externally-defined schema
    Protocol,
)

from strategy.signal import Direction, compute_signal, compute_signal_from_snapshot

if TYPE_CHECKING:
    from collections import deque
    from collections.abc import Callable
    from pathlib import Path

    import aiohttp

    from config import Config, DataPaths
    from execution.base import KellyTelemetrySnapshot
    from execution.order_manager import OrderManager
    from execution.paper_trading import PaperOrderManager
    from market_data.state import MarketState, WindowSnapshot
    from risk.fee_tracker import FeeTracker
    from risk.position_tracker import PositionTracker
    from shared.chop_detector import ChopDetector
    from shared.decay_detector import DecayDetector
    from shared.ewma_volatility_tracker import EwmaVolatilityTracker
    from shared.outcome_tracker import OutcomeTracker
    from shared.trade_journal import TradeJournal
    from shared.volatility_tracker import VolatilityTracker
    from strategy.kelly import BankrollTracker
    from strategy.momentum_signal import MomentumSignalConfig, MomentumSignalStrategy
    from strategy.monitors import (
        BetScaleSqueezeTracker,
        ConsecutiveLossTracker,
        SessionAccumulator,
        SkipStreakTracker,
    )
    from strategy.post_loss_cooldown import PostLossCooldown
    from strategy.resolution import ResolutionManager
    from strategy.signal_lifecycle import SignalLifecycle
    from strategy.window_tracker import WindowTracker


class _PendingSignalMgr(Protocol):
    """Structural interface for PendingSignalManager (defined in main.py)."""

    @property
    def last_signal_time(self) -> float: ...
    def take_pending(self) -> tuple[dict[str, Any], str] | None: ...


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass — communicates state changes back to the caller
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowTransitionResult:
    """Returned by on_window_transition to communicate state changes."""

    strategy: MomentumSignalStrategy
    decay_detector: DecayDetector
    last_window_ts: int
    snapshot_taken: bool


# ---------------------------------------------------------------------------
# Helper: compute snapshot outcome direction
# ---------------------------------------------------------------------------


def _compute_snapshot_outcome(snap: WindowSnapshot | None) -> str | None:
    """Determine outcome direction from a window snapshot."""
    if snap is None:
        return None
    sig = compute_signal_from_snapshot(snap)
    if sig.direction == Direction.NONE:
        return None
    result: str = sig.direction.value
    return result


# ---------------------------------------------------------------------------
# WindowEventHandler
# ---------------------------------------------------------------------------


class WindowEventHandler:
    """Processes everything that happens at a window boundary.

    Replaces the monolithic ``if wts != last_window_ts:`` block that was
    previously inline in ``_strategy_loop()``.
    """

    def __init__(
        self,
        cfg: Config,
        state: MarketState,
        window_tracker: WindowTracker,
        resolution_mgr: ResolutionManager,
        lifecycle: SignalLifecycle,
        position_tracker: PositionTracker,
        fee_tracker: FeeTracker,
        bankroll_tracker: BankrollTracker,
        recent_outcomes: deque[int],
        optimistic_outcomes: deque[int],
        session_stats: SessionAccumulator,
        skip_tracker: SkipStreakTracker,
        loss_tracker: ConsecutiveLossTracker,
        squeeze_tracker: BetScaleSqueezeTracker,
        journal: TradeJournal,
        vol_tracker: VolatilityTracker,
        chop_detector: ChopDetector,
        outcome_tracker: OutcomeTracker,
        paths: DataPaths,
        session: aiohttp.ClientSession,
        pending_signal_mgr: _PendingSignalMgr | None,
        bot_start_time: float,
        build_strategy_fn: Callable[
            [Config, MarketState, dict[str, object]],
            MomentumSignalStrategy | None,
        ],
        signal_cfg_to_dict_fn: Callable[[MomentumSignalConfig], dict[str, object]],
        post_loss_cooldown: PostLossCooldown,
        fast_vol_tracker: EwmaVolatilityTracker | None = None,
    ) -> None:
        self._cfg = cfg
        self._state = state
        self._window_tracker = window_tracker
        self._resolution_mgr = resolution_mgr
        self._lifecycle = lifecycle
        self._position_tracker = position_tracker
        self._fee_tracker = fee_tracker
        self._bankroll_tracker = bankroll_tracker
        self._recent_outcomes = recent_outcomes
        self._optimistic_outcomes = optimistic_outcomes
        self._session_stats = session_stats
        self._skip_tracker = skip_tracker
        self._loss_tracker = loss_tracker
        self._squeeze_tracker = squeeze_tracker
        self._journal = journal
        self._vol_tracker = vol_tracker
        self._chop_detector = chop_detector
        self._outcome_tracker = outcome_tracker
        self._fast_vol_tracker = fast_vol_tracker
        self._paths = paths
        self._session = session
        self._pending_signal_mgr = pending_signal_mgr
        self._bot_start_time = bot_start_time
        self._build_strategy_fn = build_strategy_fn
        self._signal_cfg_to_dict_fn = signal_cfg_to_dict_fn
        self._post_loss_cooldown = post_loss_cooldown

        # Mutable state managed by the handler
        self._warmup_alert_sent = False
        self._last_bankroll_sync = 0.0

    @property
    def is_frozen_by_cooldown(self) -> bool:
        """True while a post-loss cooldown window is active (v3.2 §5.8)."""
        return self._post_loss_cooldown.is_frozen

    async def on_window_transition(
        self,
        last_window_ts: int,
        strategy: MomentumSignalStrategy,
        order_mgr: OrderManager | PaperOrderManager,
        decay_detector: DecayDetector,
    ) -> WindowTransitionResult:
        """Handle everything at a window boundary.

        Phases:
          1. Finalize previous window
          2. Process trade outcome
          3. Update lifecycle + SPRT
          4. Shadow tracking
          5. Signal swap (IPC)
          6. New window setup

        Returns WindowTransitionResult with potentially updated strategy/decay_detector.
        """
        # v3.2 §5.8: decrement the post-loss cooldown at the TOP of the
        # transition, before outcome processing can re-arm it. A loss
        # resolved this window arms the counter for the NEXT window.
        self._post_loss_cooldown.on_window_boundary()

        # Phase 1: Finalize previous window (paper mode)
        if last_window_ts > 0:
            self._finalize_previous_window(order_mgr, last_window_ts, strategy)

        # Phase 2-3: Process trade outcome + lifecycle updates
        if last_window_ts > 0:
            strategy, decay_detector = self._process_trade_outcome(
                strategy,
                order_mgr,
                decay_detector,
                last_window_ts,
            )

        # Phase 4: Shadow tracking
        if self._lifecycle.shadow is not None and last_window_ts > 0:
            self._process_shadow_tracking(last_window_ts, order_mgr)

        # Phase 5: Signal swap (IPC)
        strategy, decay_detector = self._handle_signal_swap(
            strategy,
            order_mgr,
            decay_detector,
        )

        # Phase 6: New window setup
        new_window_ts = await self._setup_new_window(
            strategy,
            order_mgr,
            decay_detector,
            last_window_ts,
        )

        return WindowTransitionResult(
            strategy=strategy,
            decay_detector=decay_detector,
            last_window_ts=new_window_ts or last_window_ts,
            snapshot_taken=False,
        )

    # ------------------------------------------------------------------
    # Phase 1: Finalize previous window
    # ------------------------------------------------------------------

    def _finalize_previous_window(
        self,
        order_mgr: OrderManager | PaperOrderManager,
        last_window_ts: int,
        strategy: MomentumSignalStrategy,
    ) -> None:
        """Paper finalization + vol/chop/outcome recording."""
        state = self._state

        # Paper mode finalization
        if order_mgr.mode == "paper":
            self._finalize_paper_window(order_mgr, last_window_ts, strategy)

        # Record close price for volatility tracking
        close_price = (
            state.end_snapshot.chainlink_price if state.end_snapshot else state.btc_chainlink
        )
        if close_price > 0:
            self._vol_tracker.record_close(close_price)
        self._vol_tracker.save_cache(self._paths.vol_cache)

        # Finalize chop stats
        chop_stats = self._chop_detector.finalize_window()
        self._chop_detector.save_cache(self._paths.chop_cache)
        if chop_stats.n_ticks > 0 and chop_stats.direction_flips >= 3:
            log.info(
                "REGIME chop_detected flips=%d delta_range=%.4f%% avg_flips=%.1f over=%d_windows",
                chop_stats.direction_flips,
                chop_stats.delta_range_pct,
                self._chop_detector.avg_flips,
                self._chop_detector.n_windows,
            )

        # Record outcome direction for bias tracking
        snap = state.end_snapshot
        if snap and snap.window_ts == last_window_ts:
            outcome_sig = compute_signal_from_snapshot(snap)
        else:
            outcome_sig = compute_signal(state)
        outcome_dir = (
            outcome_sig.direction.value if outcome_sig.direction != Direction.NONE else None
        )
        if outcome_dir:
            self._outcome_tracker.record_outcome(outcome_dir)
            self._outcome_tracker.save_cache(self._paths.outcome_cache)

    def _finalize_paper_window(
        self,
        order_mgr: OrderManager | PaperOrderManager,
        window_ts: int,
        strategy: MomentumSignalStrategy,
    ) -> None:
        """Finalize paper trading for the previous window."""
        state = self._state
        snap = state.end_snapshot
        if snap and snap.window_ts == window_ts:
            sig = compute_signal_from_snapshot(snap)
        else:
            sig = compute_signal(state)
            log.warning("no snapshot for window %d — using live state for finalization", window_ts)
            snap = None

        outcome = sig.direction.value if sig.direction != Direction.NONE else None
        # Only PaperOrderManager has finalize_window/balance — guarded by mode check in caller
        rec = order_mgr.finalize_window(sig.delta_pct, sig.direction.value, outcome, snapshot=snap)  # type: ignore[union-attr]

        traded = rec.rule_simulated_fill
        self._position_tracker.record_window(
            window_ts,
            rec.pnl_total,
            traded=traded,
            mode="paper",
            balance=order_mgr.balance,  # type: ignore[union-attr]
        )

        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        self._position_tracker.save_state(
            self._paths.state, today_str, signal_id=strategy.signal_cfg.signal_id
        )
        self._save_paper_balance(self._paths.state, order_mgr.balance)  # type: ignore[union-attr]

    @staticmethod
    def _save_paper_balance(state_file: Path, balance: float) -> None:
        try:
            import json

            data = json.loads(state_file.read_text()) if state_file.exists() else {}
            data["balance_usd"] = round(balance, 4)
            state_file.write_text(json.dumps(data, indent=2))
        except OSError as exc:
            log.warning("failed to write paper balance: %s", exc)

    # ------------------------------------------------------------------
    # Phase 2-3: Trade outcome + lifecycle + SPRT
    # ------------------------------------------------------------------

    def _process_trade_outcome(
        self,
        strategy: MomentumSignalStrategy,
        order_mgr: OrderManager | PaperOrderManager,
        decay_detector: DecayDetector,
        last_window_ts: int,
    ) -> tuple[MomentumSignalStrategy, DecayDetector]:
        """Extract trade outcome, update lifecycle, SPRT, bookkeeping."""
        cfg = self._cfg
        state = self._state
        sc = strategy.signal_cfg
        fired = strategy._fired
        is_paper = order_mgr.mode == "paper"
        source = "paper" if is_paper else "live"

        rec = order_mgr._current_record if is_paper else None  # type: ignore[union-attr]  # mode=="paper" guarantees PaperOrderManager
        _live_fill = next(iter(state.live_fills.values()), None) if not is_paper else None
        _live_filled = _live_fill is not None
        filled = rec.rule_simulated_fill if rec else _live_filled
        won: bool | None = None
        pnl = 0.0
        entry_price = 0.0

        # Paper mode: immediate outcome
        if is_paper and rec and rec.rule_simulated_fill:
            pnl = rec.pnl_total
            entry_price = rec.rule_entry_price
            if pnl > 0:
                won = True
            elif pnl < 0:
                won = False

        # Live mode: actual fill from CLOB user WebSocket
        elif not is_paper and _live_filled:
            assert _live_fill is not None  # noqa: S101  # guaranteed by _live_filled
            entry_price = _live_fill.price
            size_usd = _live_fill.size_usd
            log.info(
                "live_fill window=%d order=%s "
                "fill_price=%.4f fill_size=%.2f fill_usd=$%.2f "
                "order_price=%.2f order_size=$%.2f",
                last_window_ts,
                _live_fill.order_id[:12],
                entry_price,
                _live_fill.size,
                size_usd,
                strategy.last_entry_price,
                strategy.last_size_usd,
            )
            if entry_price > 0 and size_usd > 0:
                snap_outcome = _compute_snapshot_outcome(state.end_snapshot)
                # Snapshot Kelly telemetry + closing window delta from the live
                # OrderManager so the resolution-time JSONL record mirrors
                # paper's WindowRecord schema. This call also clears the
                # capture so the next window starts fresh.
                from execution.order_manager import OrderManager as _LiveMgr

                _telemetry: KellyTelemetrySnapshot | None = None
                _early_exit: tuple[float, float, float, float] | None = None
                if isinstance(order_mgr, _LiveMgr):
                    _telemetry = order_mgr.take_kelly_telemetry()
                    _early_exit = order_mgr.take_early_exit()
                _close_snap = state.end_snapshot
                _close_delta = (
                    compute_signal_from_snapshot(_close_snap).delta_pct
                    if _close_snap and _close_snap.window_ts == last_window_ts
                    else compute_signal(state).delta_pct
                )
                _force_result = self._resolution_mgr.create_pending(
                    window_ts=last_window_ts,
                    slug=self._window_tracker.make_slug(last_window_ts),
                    signal_cfg=sc,
                    entry_price=entry_price,
                    size_usd=size_usd,
                    signal_age_windows=self._lifecycle.signal_age_windows,
                    snapshot_outcome=snap_outcome,
                    mode=order_mgr.mode,
                    signal_id=strategy.signal_cfg.signal_id,
                    kelly_telemetry=_telemetry,
                    window_delta_pct=_close_delta,
                    early_exit_sell_price=_early_exit[0] if _early_exit is not None else None,
                    early_exit_pnl=_early_exit[1] if _early_exit is not None else None,
                    early_exit_residual_shares=_early_exit[2] if _early_exit is not None else 0.0,
                    early_exit_residual_entry=_early_exit[3] if _early_exit is not None else 0.0,
                    is_maker_entry=_live_fill.is_maker,
                )
                # Crash recovery between fill and resolution is covered by
                # TradeJournal.record_trade below; the JSONL is written as a
                # single definitive record at resolution time so live matches
                # paper's one-line-per-window schema.
                if (
                    _force_result is not None
                    and _force_result.verdict == "DEAD"
                    and self._lifecycle.idle_reason is None
                ):
                    self._lifecycle.idle_reason = "decay"
                    ds = decay_detector.state
                    from shared.discord import send_sprt_decay_alert

                    send_sprt_decay_alert(
                        mode=order_mgr.mode,
                        signal_id=sc.signal_id,
                        n_trades=ds.n_trades,
                        llr=ds.llr,
                        rolling_win_rate_pct=ds.rolling_win_rate * 100,
                        signal_age_windows=self._lifecycle.signal_age_windows,
                        p_alive=ds.p_alive,
                        p_dead=ds.p_dead,
                    )
                filled = True

        # Live mode: no fill this window
        if not is_paper and not _live_filled:
            if strategy._order_placed and not _live_filled:
                log.info(
                    "order was placed but NOT filled in window %d (cancelled before fill)",
                    last_window_ts,
                )
            self._position_tracker.record_window(
                last_window_ts,
                0.0,
                traded=False,
                mode=order_mgr.mode,
                balance=self._bankroll_tracker.bankroll,
            )
            # Write a skip record so the live JSONL has one line per window,
            # matching paper. Pull whatever Kelly telemetry was captured this
            # window (empty dict for warmup / pre-sizing skips) and the close
            # delta for downstream analysis.
            from execution.order_manager import OrderManager as _LiveMgr

            _skip_telemetry: KellyTelemetrySnapshot = {}
            if isinstance(order_mgr, _LiveMgr):
                _skip_telemetry = order_mgr.take_kelly_telemetry()
            _skip_snap = state.end_snapshot
            _skip_delta = (
                compute_signal_from_snapshot(_skip_snap).delta_pct
                if _skip_snap and _skip_snap.window_ts == last_window_ts
                else compute_signal(state).delta_pct
            )
            _skip_outcome = _compute_snapshot_outcome(_skip_snap)
            self._resolution_mgr.write_skipped_window_record(
                window_ts=last_window_ts,
                window_delta_pct=_skip_delta,
                direction=sc.side.value,
                kelly_telemetry=_skip_telemetry,
                actual_outcome=_skip_outcome,
            )

        # Journal — paper records full outcome immediately; live records the
        # fire event here (won/pnl None) so skipped/unfilled/fired rows are
        # captured even if the bot dies before resolution. Resolution will
        # append its own finalized entry via resolution.py's journal write.
        from shared.trade_journal import TradeRecord

        self._journal.record_trade(
            TradeRecord(
                timestamp=self._journal.now_iso(),
                signal_id=sc.signal_id,
                signal_side=sc.side.value,
                window_ts=last_window_ts,
                fired=fired,
                filled=filled,
                won=won if is_paper else None,
                entry_price=entry_price,
                pnl=pnl if is_paper else 0.0,
                source=source,
                signal_age_windows=self._lifecycle.signal_age_windows,
            )
        )

        # Window decision log
        _order_placed = strategy._order_placed
        _tag, _skip_why = _window_decision_tag(
            filled,
            won,
            fired,
            _order_placed,
            self._lifecycle.idle_reason,
        )
        log.info(
            "WINDOW_DECISION %s rank=%d side=%s pnl=$%.4f entry=%.2f size=$%.2f%s",
            _tag,
            sc.rank,
            sc.side.value,
            pnl,
            entry_price,
            strategy.last_size_usd,
            _skip_why,
        )

        # Skip streak tracking
        if filled:
            self._skip_tracker.record_fill()
        else:
            if self._lifecycle.idle_reason:
                reason = f"idle:{self._lifecycle.idle_reason}"
            elif not fired:
                reason = "no_signal_fire"
            else:
                reason = "fired_not_filled"
            self._skip_tracker.record_skip(reason)
            self._skip_tracker.check_alert(
                mode=source,
                position_tracker=self._position_tracker,
                signal_cfg=strategy.signal_cfg,
                bot_start_time=self._bot_start_time,
            )

        # Outcome-dependent processing
        _has_resolved_outcome = (won is not None) and not self._resolution_mgr.is_pending

        if filled and won is False and _has_resolved_outcome:
            self._loss_tracker.check_and_alert(
                streak=self._position_tracker.consecutive_losses,
                mode=source,
                max_allowed=cfg.risk.max_consecutive_losses,
                entry_price=entry_price,
                pnl=pnl,
                session_pnl=self._position_tracker.total_pnl,
                daily_pnl=self._position_tracker.daily_pnl,
            )
        elif filled and won is True and _has_resolved_outcome:
            self._loss_tracker.record_win()

        # Session stats
        if filled and _has_resolved_outcome and won is not None:
            self._session_stats.record_trade(won, pnl)
        self._session_stats.record_window(
            strategy.bet_scale,
            self._vol_tracker.current_stddev_pct,
            self._chop_detector.avg_flips,
        )

        # Fire-rate tracking + fire-stall detection
        self._lifecycle.on_window_complete(
            fired=fired,
            fire_stall_windows=cfg.signal_lifecycle.fire_stall_windows,
            signal_cfg=sc,
            rules_cfg=cfg.rules_strategy,
            state=self._state,
        )

        # Kelly bankroll + outcome tracking (paper mode — live is deferred to resolution)
        if filled and _has_resolved_outcome:
            _kelly_entry = entry_price if entry_price > 0 else 0.0
            _kelly_size = strategy.last_size_usd
            _kelly_shares = _kelly_size / _kelly_entry if _kelly_entry > 0 else 0.0
            _kelly_fee = self._fee_tracker.compute_taker_fee(_kelly_entry, _kelly_shares)
            # Snapshot bankroll BEFORE update_loss so the cooldown sees the
            # pre-loss denominator (v3.2 §5.8).
            _bankroll_before = self._bankroll_tracker.bankroll
            if won:
                self._recent_outcomes.append(1)
                if _kelly_entry > 0 and _kelly_size > 0:
                    self._bankroll_tracker.update_win(_kelly_size, _kelly_entry, fee=_kelly_fee)
            else:
                self._recent_outcomes.append(0)
                if _kelly_entry > 0 and _kelly_size > 0:
                    self._bankroll_tracker.update_loss(_kelly_size, _kelly_entry, fee=_kelly_fee)
                if pnl < 0.0:
                    self._post_loss_cooldown.register_loss(-pnl, _bankroll_before)

            # v2.9 Kelly/paper bankroll reconcile. In paper mode the
            # authoritative balance is PaperOrderManager._balance, which is
            # updated by the simulator using actual (sell_price - entry_price)
            # proceeds including CUSUM early-exit PnL. The Kelly
            # update_win/update_loss path only knows about size/entry and
            # ignores early-exit sell prices, so the two drift — v2.8 paper
            # ended at $872 while Kelly bankroll ended at $515, causing
            # Kelly to size off a stale, pessimistic base. Reconcile after
            # every settled paper trade so Kelly sizes from the real balance.
            from execution.paper_trading import PaperOrderManager

            if isinstance(order_mgr, PaperOrderManager):
                self._bankroll_tracker.sync_from_api(order_mgr.balance)

        # SPRT decay detection
        if (
            filled
            and _has_resolved_outcome
            and self._lifecycle.idle_reason is None
            and won is not None
        ):
            decay_detector = self._update_sprt(
                decay_detector,
                won,
                entry_price,
                sc,
                source,
            )

        return strategy, decay_detector

    def _update_sprt(
        self,
        decay_detector: DecayDetector,
        won: bool,
        entry_price: float,
        sc: MomentumSignalConfig,
        source: str,
    ) -> DecayDetector:
        """Run SPRT decay detection and WR checkpoint."""
        avg_entry = self._lifecycle.record_entry_price(entry_price)
        new_p_dead = avg_entry + 0.02
        if new_p_dead >= decay_detector.state.p_alive:
            new_p_dead = decay_detector.state.p_alive - 0.05

        ds = decay_detector.update(won)

        # WR checkpoint
        fills = ds.n_trades
        send_wr = fills <= 20 or fills in (25, 30, 40, 50, 75, 100) or fills % 50 == 0
        if send_wr:
            wins = ds.n_wins
            losses = fills - wins
            live_wr = (wins / fills * 100) if fills > 0 else 0.0
            be_wr = avg_entry * 100
            is_paper = source == "paper"
            from shared.discord import send_live_wr_checkpoint

            send_live_wr_checkpoint(
                mode=source,
                signal_id=sc.signal_id,
                fills=fills,
                wins=wins,
                losses=losses,
                live_wr_pct=live_wr,
                expected_wr_pct=sc.oos_win_rate_pct,
                entry_avg=avg_entry,
                breakeven_wr_pct=be_wr,
                net_pnl=(
                    self._position_tracker.total_pnl if is_paper else self._session_stats.net_pnl
                ),
                llr=ds.llr,
                signal_age_windows=self._lifecycle.signal_age_windows,
            )

        self._lifecycle.on_sprt_verdict(
            ds.verdict,
            sc,
            self._cfg.rules_strategy,
            self._state,
        )
        if ds.verdict == "DEAD":
            log.warning(
                "DECAY DETECTED signal=%s trades=%d llr=%.2f rolling_wr=%.0f%% — entering IDLE",
                sc.signal_id,
                ds.n_trades,
                ds.llr,
                ds.rolling_win_rate * 100,
            )
            from shared.discord import send_sprt_decay_alert

            send_sprt_decay_alert(
                mode=source,
                signal_id=sc.signal_id,
                n_trades=ds.n_trades,
                llr=ds.llr,
                rolling_win_rate_pct=ds.rolling_win_rate * 100,
                signal_age_windows=self._lifecycle.signal_age_windows,
                p_alive=ds.p_alive,
                p_dead=ds.p_dead,
            )
        elif ds.verdict == "ALIVE":
            log.info(
                "SPRT reset signal=%s trades=%d wr=%.0f%% — confirmed ALIVE",
                sc.signal_id,
                ds.n_trades,
                ds.rolling_win_rate * 100,
            )
            decay_detector.reset(p_dead=new_p_dead)

        return decay_detector

    # ------------------------------------------------------------------
    # Phase 4: Shadow tracking
    # ------------------------------------------------------------------

    def _process_shadow_tracking(
        self,
        last_window_ts: int,
        order_mgr: OrderManager | PaperOrderManager,
    ) -> None:
        """Finalize shadow tracking for the completed window."""
        sh = self._lifecycle.shadow
        assert sh is not None  # noqa: S101  # only called when shadow tracking is active
        outcome_dir: str | None = None
        if sh.signal._fired and sh.signal._conditions_met():
            snap = self._state.end_snapshot
            sig_eval = (
                compute_signal_from_snapshot(snap)
                if snap and snap.window_ts == last_window_ts
                else compute_signal(self._state)
            )
            outcome_dir = sig_eval.direction.value if sig_eval.direction != Direction.NONE else None

        self._lifecycle.finalize_shadow_window(
            outcome_dir=outcome_dir,
            journal=self._journal,
            window_ts=last_window_ts,
            shadow_tracking_windows=self._cfg.signal_lifecycle.shadow_tracking_windows,
            mode=order_mgr.mode,
        )

    # ------------------------------------------------------------------
    # Phase 5: Signal swap (IPC)
    # ------------------------------------------------------------------

    def _handle_signal_swap(
        self,
        strategy: MomentumSignalStrategy,
        order_mgr: OrderManager | PaperOrderManager,
        decay_detector: DecayDetector,
    ) -> tuple[MomentumSignalStrategy, DecayDetector]:
        """Handle IPC signal transition if a new signal is pending."""
        if not self._pending_signal_mgr:
            return strategy, decay_detector

        pending = self._pending_signal_mgr.take_pending()
        if not pending:
            return strategy, decay_detector

        self._session_stats.signals_received += 1
        new_data, _summary_file = pending
        old_sc = strategy.signal_cfg
        old_label = f"#{old_sc.rank} {old_sc.side.value}"

        new_strategy = self._build_strategy_fn(self._cfg, self._state, new_data)
        if not new_strategy:
            return strategy, decay_detector

        new_sc = new_strategy.signal_cfg
        same_signal = old_sc.signal_id == new_sc.signal_id

        if same_signal:
            log.info(
                "Signal refreshed (same pattern): %s — rank %d->%d, "
                "keeping SPRT state (age=%dw, trades=%d, LLR=%.2f)",
                new_sc.signal_id,
                old_sc.rank,
                new_sc.rank,
                self._lifecycle.signal_age_windows,
                decay_detector.state.n_trades,
                decay_detector.state.llr,
            )
        else:
            log.info(
                "Signal changed: %s -> %s (age was %d windows)",
                old_sc.signal_id,
                new_sc.signal_id,
                self._lifecycle.signal_age_windows,
            )

        # Force-resolve any pending resolution before replacing decay_detector
        if self._resolution_mgr.is_pending and not same_signal:
            self._resolution_mgr.force_resolve(
                context="signal change while resolution pending",
                signal_id=old_sc.signal_id,
                mode=order_mgr.mode,
            )

        strategy = new_strategy
        new_label = f"#{new_sc.rank} {new_sc.side.value}"
        log.info(
            "SIGNAL_SWAP_ACTIVE: %s",
            _json.dumps(self._signal_cfg_to_dict_fn(new_sc), separators=(",", ":")),
        )
        log.info("switched to new signal: %s (previous: %s)", new_label, old_label)

        from shared.discord import send_signal_updated

        send_signal_updated(
            new_rank=new_sc.rank,
            new_side=new_sc.side.value,
            old_rank=old_sc.rank,
            old_side=old_sc.side.value,
            score=new_sc.smart_score if new_sc.smart_score > 0 else None,
            ev=new_sc.ev_per_trade,
            avg_entry=new_sc.avg_entry_price or None,
            conservative_wr_pct=new_sc.conservative_win_rate_pct,
            folds_appeared=new_sc.wf_folds_appeared,
            total_folds=new_sc.wf_total_test_folds,
        )

        if same_signal:
            self._lifecycle.on_same_signal_refresh()
        else:
            if self._lifecycle.shadow is not None:
                log.info(
                    "New signal activated while shadow tracking %s. "
                    "Shadow tracking continues in background.",
                    self._lifecycle.shadow.signal_cfg.signal_id,
                )
            self._lifecycle.reset_for_new_signal()
            self._recent_outcomes.clear()
            self._optimistic_outcomes.clear()

            from shared.decay_detector import DecayDetector

            new_p_alive = new_sc.conservative_p(self._cfg.sizing.wilson_max_shrink_pct)
            new_p_dead_init = (new_sc.avg_entry_price or 0.85) + 0.02
            if new_p_dead_init >= new_p_alive:
                new_p_dead_init = new_p_alive - 0.05
            decay_detector = DecayDetector(
                signal_id=new_sc.signal_id,
                p_alive=new_p_alive,
                p_dead=new_p_dead_init,
            )
            # Propagate new detector to resolution manager so pending
            # resolutions update the correct signal's SPRT state.
            self._resolution_mgr.decay_detector = decay_detector

        return strategy, decay_detector

    # ------------------------------------------------------------------
    # Phase 6: New window setup
    # ------------------------------------------------------------------

    async def _setup_new_window(
        self,
        strategy: MomentumSignalStrategy,
        order_mgr: OrderManager | PaperOrderManager,
        decay_detector: DecayDetector,
        _last_window_ts: int,
    ) -> int | None:
        """Fetch new window info, reset strategy, compute Kelly context.

        Returns new window_ts, or None if window_tracker didn't provide info.
        """
        cfg = self._cfg

        window_info = await self._window_tracker.on_new_window()
        if not window_info:
            return None

        strategy.reset()
        self._position_tracker.reset_window_exposure()

        if order_mgr.mode == "paper":
            order_mgr.reset_window(window_info.window_ts)  # type: ignore[union-attr]  # mode=="paper" guarantees PaperOrderManager

        new_window_ts = window_info.window_ts
        await self._fee_tracker.fetch_fee_rate(self._session, cfg.connections.clob_rest)

        # Refresh on-chain balance for live mode
        if order_mgr.mode == "live":
            _api_bal = await order_mgr.refresh_balance()  # type: ignore[union-attr]  # mode=="live" guarantees OrderManager
            _BANKROLL_SYNC_INTERVAL = 1800.0  # noqa: N806  # constant defined in function scope
            if (
                _api_bal is not None
                and _api_bal > 0
                and time.time() - self._last_bankroll_sync >= _BANKROLL_SYNC_INTERVAL
            ):
                self._bankroll_tracker.sync_from_api(_api_bal)
                self._last_bankroll_sync = time.time()

        # Adaptive bet sizing (Kelly + SPRT)
        if cfg.signal_lifecycle.bet_scaling_enabled:  # master switch for all bet scaling
            self._compute_kelly_context(strategy, order_mgr, decay_detector)

        return new_window_ts

    def _effective_vol_stddev(self) -> tuple[float, float]:
        """Return (effective_stddev_pct, fast_equivalent_pct).

        The effective stddev fed to the severity scoring is the max of the
        slow 2-hour close-to-close tracker and the short-horizon EWMA
        tracker scaled up to an equivalent 5-minute horizon. That way a
        mid-window squeeze lifts the severity immediately instead of
        waiting for the next window boundary, while a calm slow reading
        still dominates when the fast tracker is quiet.
        """
        import math as _math

        slow = self._vol_tracker.current_stddev_pct
        fast_equivalent = 0.0
        if self._fast_vol_tracker is not None and self._fast_vol_tracker.ready:
            fcfg = self._cfg.regime
            if fcfg.vol_fast_sample_interval_s > 0.0:
                scale = _math.sqrt(fcfg.vol_fast_horizon_s / fcfg.vol_fast_sample_interval_s)
                fast_equivalent = self._fast_vol_tracker.current_stddev_pct * scale
        return max(slow, fast_equivalent), fast_equivalent

    def _build_kelly_wr_result(
        self,
        strategy: MomentumSignalStrategy,
    ) -> tuple[object, float, float, float]:
        """Run estimate_adjusted_win_rate against current tracker readings.

        Returns the AdjustedWinRateResult plus the raw vol_stddev, fast
        equivalent, and chop flips used, so callers can log them.
        """
        from strategy.kelly import estimate_adjusted_win_rate

        cfg = self._cfg
        self._vol_tracker.update_stddev()
        effective_vol, fast_equivalent = self._effective_vol_stddev()
        chop_flips = self._chop_detector.avg_flips
        regime_ready = (
            self._vol_tracker.n_returns >= cfg.regime.vol_min_samples
            or self._chop_detector.n_windows >= cfg.regime.chop_min_samples
            or (self._fast_vol_tracker is not None and self._fast_vol_tracker.ready)
        )
        outcome_agreement = self._outcome_tracker.direction_agreement(
            strategy.signal_cfg.side.value,
        )

        kelly_outcomes: deque[int] = self._recent_outcomes
        if self._optimistic_outcomes:
            from collections import deque as _deque

            kelly_outcomes = _deque(
                list(self._recent_outcomes) + list(self._optimistic_outcomes),
                maxlen=self._recent_outcomes.maxlen,
            )

        wr = estimate_adjusted_win_rate(
            base_win_rate=strategy.signal_cfg.conservative_p(cfg.sizing.wilson_max_shrink_pct),
            vol_stddev=effective_vol,
            chop_avg_flips=chop_flips,
            outcome_agreement=outcome_agreement,
            vol_baseline=cfg.regime.vol_normal_pct,
            vol_elevated=cfg.regime.vol_high_pct,
            chop_baseline=cfg.regime.chop_normal_flips,
            chop_elevated=cfg.regime.chop_high_flips,
            outcome_baseline=cfg.regime.outcome_normal_agreement,
            outcome_elevated=cfg.regime.outcome_high_agreement,
            max_discount=cfg.sizing.kelly_regime_cap,
            vol_weight=cfg.sizing.vol_weight,
            chop_weight=cfg.sizing.chop_weight,
            outcome_weight=cfg.sizing.outcome_weight,
            regime_ready=regime_ready,
            recent_outcomes=kelly_outcomes,
            min_outcomes_for_feedback=cfg.sizing.feedback_min_trades,
            soft_or_combine=cfg.sizing.kelly_soft_or_combine,
            max_discount_2_axes=cfg.sizing.kelly_regime_cap_2_axes,
            max_discount_3_axes=cfg.sizing.kelly_regime_cap_3_axes,
            hot_axis_threshold=cfg.sizing.kelly_hot_axis_threshold,
        )
        return wr, effective_vol, fast_equivalent, chop_flips

    def refresh_regime_context(self, strategy: MomentumSignalStrategy) -> None:
        """Recompute kelly_wr_result from current tracker readings.

        Safe to call at strategy-tick cadence. Only touches fields that
        depend on intra-window regime drift (kelly_wr_result). Bankroll,
        sizing_cfg, sprt_factor, warmup_active are set once at window
        boundary by _compute_kelly_context and never touched here.

        Intended call site: strategy tick loop when
        ``time_remaining <= cfg.regime.intra_window_refresh_s``. Before
        v3.2 the Kelly context was frozen at window-open; v3.1 T4 fired on
        a 2-hour squeeze that materialised mid-window, invisible to the
        hostile-regime gate — this method closes that gap.
        """
        cfg = self._cfg
        if not cfg.signal_lifecycle.bet_scaling_enabled:
            return
        if strategy.sizing_cfg is None:
            return

        from strategy.kelly import AdjustedWinRateResult

        wr, _, _, _ = self._build_kelly_wr_result(strategy)
        assert isinstance(wr, AdjustedWinRateResult)  # noqa: S101  # narrow for mypy --strict
        strategy.kelly_wr_result = wr

    def _compute_kelly_context(
        self,
        strategy: MomentumSignalStrategy,
        order_mgr: OrderManager | PaperOrderManager,
        decay_detector: DecayDetector,
    ) -> None:
        """Compute Kelly win rate adjustment and bet sizing context."""
        cfg = self._cfg
        ds = decay_detector.state

        from shared.risk import age_taper, compute_bet_scale, llr_confidence
        from strategy.kelly import AdjustedWinRateResult

        # SPRT staleness check
        _stale_threshold_s = cfg.signal_lifecycle.sprt_activation_minutes * 60.0
        _signal_is_stale = (
            self._pending_signal_mgr is not None
            and self._pending_signal_mgr.last_signal_time > 0
            and (time.time() - self._pending_signal_mgr.last_signal_time) > _stale_threshold_s
        )

        sprt_factor = compute_bet_scale(
            llr=ds.llr,
            boundary_alive=ds.boundary_alive,
            boundary_dead=ds.boundary_dead,
            signal_age_windows=self._lifecycle.signal_age_windows,
            signal_is_stale=_signal_is_stale,
            taper_start=cfg.signal_lifecycle.age_taper_start_windows,
            taper_end=cfg.signal_lifecycle.age_taper_end_windows,
            age_floor=cfg.signal_lifecycle.age_floor,
            min_total_scale=cfg.signal_lifecycle.min_bet_scale,
        )

        wr, _vol_stddev, _fast_equivalent, _chop_flips = self._build_kelly_wr_result(strategy)
        assert isinstance(wr, AdjustedWinRateResult)  # noqa: S101  # narrow for mypy --strict
        _kelly_wr_result = wr

        # Pass Kelly context to strategy
        strategy.sprt_factor = sprt_factor
        strategy.kelly_wr_result = _kelly_wr_result
        # v3.2 §5.2: directional regime t-stat — t-stat of rolling signed
        # returns; the fire-time gate in momentum_signal vetoes when the
        # drift opposes the signal side with magnitude >= threshold.
        strategy.directional_t = self._vol_tracker.signed_return_t_stat(
            min_samples=cfg.rules_strategy.directional_min_samples,
        )

        # v3.0: paper/live startup drift check moved to main.py, before the
        # strategy loop starts, so it runs before any fire decisions.

        strategy.bankroll = self._bankroll_tracker.bankroll
        strategy.sizing_cfg = cfg.sizing
        strategy.erosion_cfg = cfg.erosion

        # Warmup clamp
        _warmup_secs = cfg.sizing.warmup_minutes * 60
        _was_warming_up = strategy.warmup_active
        strategy.warmup_active = (
            _warmup_secs > 0 and (time.time() - self._bot_start_time) < _warmup_secs
        )
        if _was_warming_up and not strategy.warmup_active and not self._warmup_alert_sent:
            self._warmup_alert_sent = True
            _wu_mode = order_mgr.mode
            _wu_wr = (
                (self._position_tracker.windows_won / self._position_tracker.windows_traded * 100)
                if self._position_tracker.windows_traded > 0
                else 0.0
            )
            log.info(
                "WARMUP complete after %.0f min — full Kelly sizing active",
                cfg.sizing.warmup_minutes,
            )
            from shared.discord import send_warmup_complete

            send_warmup_complete(
                mode=_wu_mode,
                warmup_minutes=cfg.sizing.warmup_minutes,
                signal_id=strategy.signal_cfg.signal_id,
                session_pnl=self._position_tracker.total_pnl,
                windows_traded=self._position_tracker.windows_traded,
                win_rate_pct=_wu_wr,
                wins=self._position_tracker.windows_won,
                total=self._position_tracker.windows_traded,
                bankroll=self._bankroll_tracker.bankroll,
            )

        llr_conf = (
            llr_confidence(ds.llr, ds.boundary_alive, ds.boundary_dead) if ds.n_trades > 0 else 1.0
        )
        age_tap = age_taper(
            self._lifecycle.signal_age_windows,
            cfg.signal_lifecycle.age_taper_start_windows,
            cfg.signal_lifecycle.age_taper_end_windows,
            cfg.signal_lifecycle.age_floor,
        )

        # Regime log. vol_stddev is the effective (max of slow + scaled
        # fast EWMA) value; vol_fast is the horizon-scaled fast equivalent
        # for visibility into which tracker drove the reading.
        log.info(
            "REGIME vol_stddev=%.3f%% vol_fast=%.3f%% chop_flips=%.1f "
            "outcome=%s dir_t=%+.2f "
            "sprt=%s llr=%.2f age=%dw bankroll=$%.2f",
            _vol_stddev,
            _fast_equivalent,
            _chop_flips,
            self._outcome_tracker.summary(),
            strategy.directional_t,
            f"active_{llr_conf * 100:.0f}pct" if _signal_is_stale else "dormant",
            ds.llr,
            self._lifecycle.signal_age_windows,
            self._bankroll_tracker.bankroll,
        )

        if _kelly_wr_result.total_discount > 0:
            log.info(
                "KELLY wr_adj=%.1f%% base=%.1f%% "
                "total_disc=%.1f%% vol=%.1f%% chop=%.1f%% "
                "outcome=%.1f%% feedback=%+.1f%%",
                _kelly_wr_result.adjusted_p * 100,
                strategy.signal_cfg.oos_win_rate_pct,
                _kelly_wr_result.total_discount * 100,
                _kelly_wr_result.vol_discount * 100,
                _kelly_wr_result.chop_discount * 100,
                _kelly_wr_result.outcome_discount * 100,
                _kelly_wr_result.feedback_adjustment * 100,
            )

        # Bet scale squeeze alert
        self._squeeze_tracker.update(
            sprt_factor=sprt_factor,
            mode=order_mgr.mode,
            llr_conf_pct=llr_conf * 100 if _signal_is_stale else 100.0,
            age_tap_pct=age_tap * 100,
            vol_stddev_pct=_vol_stddev,
            chop_avg_flips=_chop_flips,
            llr=ds.llr,
            signal_age_windows=self._lifecycle.signal_age_windows,
        )

        # Risk level notification
        if _kelly_wr_result.total_discount > 0:
            _side = strategy.signal_cfg.side
            _entry = self._state.best_ask_up if _side.value == "up" else self._state.best_ask_down
            from shared.discord import send_risk_level

            send_risk_level(
                mode=order_mgr.mode,
                vol_severity=_kelly_wr_result.vol_severity,
                chop_severity=_kelly_wr_result.chop_severity,
                outcome_severity=_kelly_wr_result.outcome_severity,
                total_discount=_kelly_wr_result.total_discount,
                base_p=strategy.signal_cfg.conservative_p(
                    cfg.sizing.wilson_max_shrink_pct,
                ),
                adjusted_p=_kelly_wr_result.adjusted_p,
                entry_price=_entry if _entry > 0 else None,
                signal_id=strategy.signal_cfg.signal_id,
            )


# ---------------------------------------------------------------------------
# Shared helper (also used by main.py for WINDOW_DECISION log)
# ---------------------------------------------------------------------------


def _window_decision_tag(
    filled: bool,
    won: bool | None,
    fired: bool,
    order_placed: bool,
    idle_reason: str | None,
) -> tuple[str, str]:
    """Return (tag, skip_reason) for the WINDOW_DECISION log line."""
    if filled and won is True:
        tag = "[WIN]"
    elif filled and won is False:
        tag = "[LOSS]"
    elif filled and won is None:
        tag = "[FLAT]"
    else:
        tag = "[SKIP]"

    skip_why = ""
    if idle_reason:
        skip_why = f" (idle:{idle_reason})"
    elif not fired:
        skip_why = " (no fire)"
    elif not order_placed:
        skip_why = " (conditions not met)"
    elif not filled:
        skip_why = " (order not filled)"
    return tag, skip_why
