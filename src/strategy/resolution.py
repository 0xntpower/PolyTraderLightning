"""ResolutionManager — owns the full lifecycle of pending market resolutions.

Consolidates the previously-duplicated resolution logic from main.py:
- Confirmed resolution (Gamma API poll success)
- Timeout fallback (snapshot-based or conservative loss)
- Force-resolve (new fill arrives or signal swap while pending)

All three paths now go through a single `_resolve()` method that handles
P&L calculation, bankroll updates, journal recording, SPRT decay, and
Discord notifications.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import orjson

from strategy.signal import Direction

if TYPE_CHECKING:
    from collections import deque
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from config import Config, DataPaths
    from execution.base import KellyTelemetrySnapshot
    from market_data.state import MarketState
    from risk.fee_tracker import FeeTracker
    from risk.position_tracker import PositionTracker
    from shared.decay_detector import DecayDetector, DecayState
    from shared.trade_journal import TradeJournal
    from strategy.kelly import BankrollTracker
    from strategy.momentum_signal import MomentumSignalConfig
    from strategy.monitors import ConsecutiveLossTracker, SessionAccumulator
    from strategy.post_loss_cooldown import PostLossCooldown
    from strategy.window_tracker import WindowTracker

    BalanceRefresher = Callable[[], Awaitable[float | None]]

log = logging.getLogger(__name__)

# Timing constants — match the values previously in main.py
RESOLUTION_POLL_DELAY = 30.0  # seconds before first Gamma poll
RESOLUTION_POLL_INTERVAL = 5.0  # seconds between polls
RESOLUTION_TIMEOUT = 1200.0  # fall back after ~20 min


def _empty_kelly_telemetry() -> KellyTelemetrySnapshot:
    # Typed factory so the dataclass default_factory matches PendingResolution's
    # field type without a cast. A bare ``dict`` factory is inferred as
    # ``dict[Never, Never]`` and fails strict assignment.
    return {}


@dataclass
class PendingResolution:
    """Holds context for a live trade awaiting Gamma API resolution."""

    window_ts: int
    slug: str
    signal_cfg: MomentumSignalConfig
    entry_price: float
    size_usd: float
    signal_age_windows: int
    created_at: float  # time.time() when created
    last_poll_at: float = 0.0
    snapshot_outcome: str | None = None  # "up", "down", or None
    kelly_telemetry: KellyTelemetrySnapshot = field(default_factory=_empty_kelly_telemetry)
    window_delta_pct: float = 0.0
    # True if an optimistic entry was pushed to optimistic_outcomes
    optimistic_counted: bool = False
    # Early-exit realized state — set when the position was sold mid-window via
    # CUSUM erosion trigger. When present, _resolve uses these instead of the
    # $1/$0 resolution-based P&L calc. If ``early_exit_residual_shares`` is 0
    # the FAK fully filled and resolution short-circuits; if non-zero the FAK
    # partially filled and we still wait for Gamma to settle the residual.
    early_exit_pnl: float | None = None
    early_exit_sell_price: float | None = None
    early_exit_residual_shares: float = 0.0
    early_exit_residual_entry: float = 0.0
    # True if the originating entry order was maker (post-only GTC). Maker
    # entries pay no taker fee at entry, so _resolve skips the fee deduction.
    # For combined maker+taker entries this is False and the fee breakdown
    # lives on ``entry_taker_fee`` instead.
    is_maker_entry: bool = False
    # Per-leg capital breakdown for combined (maker partial + taker remainder)
    # entries — set by ``window_handler._finalize_previous_window`` when it
    # aggregates across ``state.live_fills``. Both default to 0.0 for paper
    # mode and for single-leg live entries (use ``size_usd`` + ``is_maker_entry``
    # in that case). Discord's WIN/LOSS embed renders a percent split when
    # both are positive.
    maker_usd: float = 0.0
    taker_usd: float = 0.0
    # Pre-computed entry-side taker fee, accumulated across every taker fill
    # in the window at aggregation time. Supersedes the boolean gate for
    # combined entries — ``_resolve`` uses this value directly so it handles
    # mixed maker/taker entries correctly.
    entry_taker_fee: float = 0.0
    # v3.7: orchestrator-tracked signal-family age at fire time and the
    # orchestrator's p80 lifetime estimate as of delivery. Observational —
    # surfaces on the WIN/LOSS Discord embed so operators can eyeball age-
    # vs-outcome correlation. All three may be None (bootstrap, tracker
    # disabled, older orchestrator).
    signal_age_at_fire_h: float | None = None
    est_max_lifetime_h: float | None = None
    lifetime_samples: int | None = None


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Returned when a resolution completes (confirmed, timeout, or forced)."""

    won: bool
    pnl: float
    verdict: str  # SPRT verdict: "ALIVE", "DEAD", or "INCONCLUSIVE"
    pending: PendingResolution  # the resolved pending (for caller context)


class ResolutionManager:
    """Owns the full lifecycle of a pending market resolution.

    Replaces three duplicated call sites in main.py with a single
    consistent pipeline: poll → resolve → bookkeeping → notify.
    """

    def __init__(
        self,
        window_tracker: WindowTracker,
        state: MarketState,
        fee_tracker: FeeTracker,
        bankroll_tracker: BankrollTracker,
        position_tracker: PositionTracker,
        journal: TradeJournal,
        decay_detector: DecayDetector,
        recent_outcomes: deque[int],
        results_dir: Path,
        session_stats: SessionAccumulator,
        loss_tracker: ConsecutiveLossTracker,
        cfg: Config,
        paths: DataPaths,
        optimistic_outcomes: deque[int],
        post_loss_cooldown: PostLossCooldown,
        balance_refresher: BalanceRefresher | None = None,
    ) -> None:
        self._window_tracker = window_tracker
        self._state = state
        self._fee_tracker = fee_tracker
        self._bankroll_tracker = bankroll_tracker
        self._position_tracker = position_tracker
        self._journal = journal
        self._decay_detector = decay_detector
        self._recent_outcomes = recent_outcomes
        self._optimistic_outcomes = optimistic_outcomes
        self._results_dir = results_dir
        self._session_stats = session_stats
        self._loss_tracker = loss_tracker
        self._cfg = cfg
        self._paths = paths
        self._balance_refresher = balance_refresher
        self._post_loss_cooldown = post_loss_cooldown
        self._pending: PendingResolution | None = None
        self._last_result: ResolutionResult | None = None

    @property
    def decay_detector(self) -> DecayDetector:
        return self._decay_detector

    @decay_detector.setter
    def decay_detector(self, value: DecayDetector) -> None:
        self._decay_detector = value

    @property
    def is_pending(self) -> bool:
        return self._pending is not None

    @property
    def pending(self) -> PendingResolution | None:
        return self._pending

    @property
    def last_result(self) -> ResolutionResult | None:
        return self._last_result

    def create_pending(
        self,
        window_ts: int,
        slug: str,
        signal_cfg: MomentumSignalConfig,
        entry_price: float,
        size_usd: float,
        signal_age_windows: int,
        snapshot_outcome: str | None,
        *,
        mode: str = "live",
        signal_id: str = "",
        kelly_telemetry: KellyTelemetrySnapshot | None = None,
        window_delta_pct: float = 0.0,
        early_exit_pnl: float | None = None,
        early_exit_sell_price: float | None = None,
        early_exit_residual_shares: float = 0.0,
        early_exit_residual_entry: float = 0.0,
        is_maker_entry: bool = False,
        maker_usd: float = 0.0,
        taker_usd: float = 0.0,
        entry_taker_fee: float = 0.0,
    ) -> ResolutionResult | None:
        """Create a new pending resolution, force-resolving any existing one first.

        Returns a ResolutionResult if an existing pending was force-resolved,
        None otherwise. The caller should check the verdict for SPRT decay.
        """
        force_result = None
        if self._pending is not None:
            force_result = self.force_resolve(
                context="previous resolution pending",
                signal_id=signal_id,
                mode=mode,
            )

        # Optimistic outcome feedback: if a live fire closes with a snapshot
        # outcome, feed it into the parallel optimistic_outcomes deque now so
        # the next window's Kelly sizing sees a fresh feedback signal instead
        # of one that's 15-20 min stale. The entry is drained and replaced
        # with the authoritative value when _resolve runs. Paper mode does
        # not need this — its feedback is already immediate.
        optimistic_counted = False
        if mode == "live" and snapshot_outcome is not None:
            won_guess = 1 if snapshot_outcome == signal_cfg.side.value else 0
            self._optimistic_outcomes.append(won_guess)
            optimistic_counted = True

        self._pending = PendingResolution(
            window_ts=window_ts,
            slug=slug,
            signal_cfg=signal_cfg,
            entry_price=entry_price,
            size_usd=size_usd,
            signal_age_windows=signal_age_windows,
            created_at=time.time(),
            snapshot_outcome=snapshot_outcome,
            kelly_telemetry=kelly_telemetry if kelly_telemetry is not None else {},
            window_delta_pct=window_delta_pct,
            optimistic_counted=optimistic_counted,
            early_exit_pnl=early_exit_pnl,
            early_exit_sell_price=early_exit_sell_price,
            early_exit_residual_shares=early_exit_residual_shares,
            early_exit_residual_entry=early_exit_residual_entry,
            is_maker_entry=is_maker_entry,
            maker_usd=maker_usd,
            taker_usd=taker_usd,
            entry_taker_fee=entry_taker_fee,
            # v3.7: snapshot the orchestrator-tracked family age at
            # fire time so the WIN/LOSS Discord embed reports what the
            # bot saw when it decided to fire (orchestrator re-delivers
            # every ~10 min, so the age may have grown by up to one
            # cycle by resolve; close enough for eyeballing age-vs-
            # outcome patterns).
            signal_age_at_fire_h=signal_cfg.signal_age_h,
            est_max_lifetime_h=signal_cfg.est_max_lifetime_h,
            lifetime_samples=signal_cfg.lifetime_samples,
        )
        log.info(
            "trade pending resolution: window=%d slug=%s "
            "side=%s entry=%.2f size=$%.2f (snapshot_outcome=%s optimistic=%s)",
            window_ts,
            slug,
            signal_cfg.side.value,
            entry_price,
            size_usd,
            snapshot_outcome,
            optimistic_counted,
        )
        return force_result

    async def tick(self, now: float, *, mode: str = "live") -> ResolutionResult | None:
        """Called every strategy tick. Polls Gamma API if timing conditions met.

        Returns ResolutionResult if resolved (confirmed or timeout), None if still pending.
        """
        if self._pending is None:
            return None

        pr = self._pending

        # Early-exit short-circuit: the position was sold mid-window via
        # ``exit_position_early``, so we already have realized P&L and don't
        # need to wait for Gamma — unless the FAK partially filled, in which
        # case the residual shares are still on-chain and must settle at the
        # canonical $1/$0 outcome. Fall through to normal Gamma polling in
        # that case; ``_resolve`` will blend realized + residual P&L.
        if pr.early_exit_pnl is not None and pr.early_exit_residual_shares <= 0:
            synthetic_outcome = self._force_resolve_outcome(pr)
            result = self._resolve(pr, synthetic_outcome, mode=mode)
            await self._reconcile_bankroll(mode)
            return result

        elapsed = now - pr.created_at
        since_last_poll = now - pr.last_poll_at

        # Not enough time elapsed for first poll
        if elapsed < RESOLUTION_POLL_DELAY:
            return None
        if since_last_poll < RESOLUTION_POLL_INTERVAL:
            return None

        # Poll Gamma API
        pr.last_poll_at = now
        resolution_data = await self._window_tracker.fetch_market_resolution(pr.slug)

        if resolution_data is not None:
            log.info(
                "market resolution confirmed: %s resolved %s (snapshot had predicted: %s)",
                pr.slug,
                resolution_data.outcome,
                pr.snapshot_outcome,
            )
            # finalPrice(N) = close of resolved window N = open of current window N+1
            self._try_oracle_upgrade(resolution_data.final_price)
            result = self._resolve(pr, resolution_data.outcome, mode=mode)
            await self._reconcile_bankroll(mode)
            return result

        # Timeout check
        if elapsed >= RESOLUTION_TIMEOUT:
            fallback = self._force_resolve_outcome(pr)
            if pr.snapshot_outcome is None:
                log.error(
                    "RESOLUTION TIMEOUT for %s after %.0fs (~%.0f min) — "
                    "Gamma API never confirmed AND no snapshot. "
                    "Conservatively resolving as LOSS.",
                    pr.slug,
                    elapsed,
                    elapsed / 60,
                )
            else:
                log.error(
                    "RESOLUTION TIMEOUT for %s after %.0fs (~%.0f min) — "
                    "Gamma API never confirmed. Falling back to snapshot: %s. "
                    "P&L may be incorrect!",
                    pr.slug,
                    elapsed,
                    elapsed / 60,
                    fallback,
                )
            result = self._resolve(pr, fallback, mode=mode)
            await self._reconcile_bankroll(mode)
            return result

        return None

    async def _reconcile_bankroll(self, mode: str) -> None:
        """Pull on-chain balance and sync local bankroll after a live resolve.

        Mirror of paper Fix 4 (window_handler.py sync_from_api after settled
        paper trade). In live mode the analogous drift source is CUSUM early
        exit + taker-fee model error: update_win/loss tracks entry-price P&L,
        but the real USDC delta on-chain includes realized sell proceeds from
        early exits and true taker fees. Without this reconcile, Kelly sizes
        off an increasingly stale bankroll.
        """
        if mode != "live" or self._balance_refresher is None:
            return
        api_bal = await self._balance_refresher()
        if api_bal is None or api_bal <= 0:
            return
        drift = api_bal - self._bankroll_tracker.bankroll
        if abs(drift) > 0.01:
            log.info(
                "post-resolve bankroll reconcile: kelly=$%.2f onchain=$%.2f drift=%+.2f",
                self._bankroll_tracker.bankroll,
                api_bal,
                drift,
            )
        self._bankroll_tracker.sync_from_api(api_bal)

    def force_resolve(
        self,
        context: str,
        signal_id: str,
        *,
        mode: str = "live",
    ) -> ResolutionResult:
        """Force-resolve the current pending resolution.

        Used when:
        - A new fill arrives while previous resolution is still pending
        - Signal swap while resolution is still pending

        Returns ResolutionResult so the caller can check SPRT verdict.
        """
        pr = self._pending
        assert pr is not None, "force_resolve called with no pending resolution"  # noqa: S101  # invariant check

        outcome = self._force_resolve_outcome(pr)
        if pr.snapshot_outcome is None:
            log.warning(
                "%s for %s with NO snapshot — force-resolving as LOSS (conservative)",
                context,
                pr.slug,
            )
        else:
            log.warning(
                "%s for %s — force-resolving with snapshot (%s)",
                context,
                pr.slug,
                outcome,
            )

        return self._resolve(pr, outcome, mode=mode, signal_id_override=signal_id)

    def clear(self) -> None:
        """Clear pending resolution without resolving (e.g., on signal reset)."""
        self._pending = None

    # ------------------------------------------------------------------
    # Internal — single resolution pipeline
    # ------------------------------------------------------------------

    def _try_oracle_upgrade(self, final_price: float | None) -> None:
        """Upgrade the current window's open price using finalPrice from the resolved event.

        finalPrice(N) = close of resolved window N = open of the current window N+1.
        It becomes available when the previous window resolves (~+196s).
        Note: priceToBeat(N) is the open of window N (the *resolved* window),
        which is NOT the current window — using it here would be wrong.
        """
        if final_price is None or final_price <= 0:
            return
        st = self._state
        if st.oracle_open_confirmed:
            return  # already upgraded this window

        old_price = st.window_open_price
        old_tier = st.open_price_tier
        if old_price <= 0:
            return  # no open price captured yet — nothing to upgrade

        diff = abs(final_price - old_price)
        diff_pct = (diff / old_price) * 100 if old_price > 0 else 0.0

        st.oracle_open_confirmed = True

        if diff < 0.01:
            log.info(
                "oracle open price confirmed: $%.2f matches tier %d capture (diff=$%.2f)",
                final_price,
                old_tier,
                diff,
            )
        else:
            st.window_open_price = final_price
            st.open_price_tier = 0
            log.warning(
                "oracle open price UPGRADED: $%.2f → $%.2f (tier %d→0, diff=$%.2f / %.4f%%)",
                old_price,
                final_price,
                old_tier,
                diff,
                diff_pct,
            )

    @staticmethod
    def _force_resolve_outcome(pr: PendingResolution) -> str:
        """Determine outcome for force-resolving. Snapshot if available, else conservative loss."""
        if pr.snapshot_outcome is not None:
            return pr.snapshot_outcome
        return "down" if pr.signal_cfg.side == Direction.UP else "up"

    def _resolve(
        self,
        pr: PendingResolution,
        outcome: str,
        *,
        mode: str = "live",
        signal_id_override: str = "",
    ) -> ResolutionResult:
        """Single resolution pipeline — handles P&L, bankroll, journal, SPRT, notifications.

        This replaces the three duplicated paths that were in main.py.
        """
        sc = pr.signal_cfg
        entry_price = pr.entry_price
        size_usd = pr.size_usd
        shares = size_usd / entry_price
        is_early_exit = pr.early_exit_pnl is not None

        if is_early_exit:
            # CUSUM sold the position mid-window. ``exit_position_early``
            # already recorded the sell-side taker fee on the fee tracker and
            # computed realized P&L on the filled portion. If the FAK partially
            # filled, the residual shares are still on-chain and settle at the
            # canonical $1/$0 outcome — add that in here. SPRT counts the
            # combined outcome by its sign.
            assert pr.early_exit_pnl is not None  # noqa: S101  # is_early_exit invariant
            realized_pnl = pr.early_exit_pnl
            residual_pnl = 0.0
            if pr.early_exit_residual_shares > 0 and pr.early_exit_residual_entry > 0:
                # Residual inherits the original entry tier's fee treatment:
                # maker entries paid no fee, taker entries already paid at
                # entry. No new fee at resolution.
                resid_shares = pr.early_exit_residual_shares
                resid_entry = pr.early_exit_residual_entry
                if outcome == sc.side.value:
                    residual_pnl = resid_shares * (1.0 - resid_entry)
                else:
                    residual_pnl = -(resid_shares * resid_entry)
            pnl = round(realized_pnl + residual_pnl, 4)
            won = pnl > 0
            log.info(
                "live early-exit resolution: window=%d side=%s sell=%.4f "
                "realized=$%.4f residual_shares=%.2f residual_pnl=$%.4f total=$%.4f "
                "(snapshot_outcome=%s market_outcome=%s)",
                pr.window_ts,
                sc.side.value,
                pr.early_exit_sell_price or 0.0,
                realized_pnl,
                pr.early_exit_residual_shares,
                residual_pnl,
                pnl,
                pr.snapshot_outcome,
                outcome,
            )
        else:
            # Entry-side taker fee: pre-computed at aggregation time by
            # ``window_handler._finalize_previous_window`` so combined
            # (maker partial + taker remainder) entries pay fee on only the
            # taker portion. ``entry_taker_fee`` is 0 for pure-maker entries
            # and equals ``compute_taker_fee(entry_price, shares)`` for
            # pure-taker entries, matching the old boolean semantic. Binary
            # resolution isn't a new trade, so no exit fee applies either.
            taker_fee = pr.entry_taker_fee

            bet_won = outcome == sc.side.value
            if bet_won:
                pnl = round(shares * (1.0 - entry_price) - taker_fee, 4)
                won = True
            else:
                pnl = round(-(shares * entry_price) - taker_fee, 4)
                won = False

            log.info(
                "live resolution: window=%d side=%s outcome=%s → %s pnl=$%.4f "
                "(entry=%.2f shares=%.1f taker_fee=$%.4f maker_entry=%s)",
                pr.window_ts,
                sc.side.value,
                outcome,
                "WIN" if won else "LOSS",
                pnl,
                entry_price,
                shares,
                taker_fee,
                pr.is_maker_entry,
            )

        # Drain the optimistic pre-resolution feedback entry before appending
        # the authoritative one. Pendings resolve in FIFO order so a single
        # popleft is correct. If the optimistic deque underflowed (e.g. a
        # race where the pending was created before optimistic tracking was
        # wired up), skip gracefully.
        if pr.optimistic_counted and self._optimistic_outcomes:
            self._optimistic_outcomes.popleft()

        # Bankroll update — for early exits, skip the $1/$0-based recalc in
        # update_win/update_loss and let ``_reconcile_bankroll`` (called after
        # _resolve in tick()/window_handler) pull the authoritative on-chain
        # balance. Still record the outcome in recent_outcomes for Kelly.
        # Capture bankroll BEFORE updates so the post-loss cooldown sees
        # the pre-loss denominator (v3.2 §5.8).
        bankroll_before = self._bankroll_tracker.bankroll
        if is_early_exit:
            if won:
                self._recent_outcomes.append(1)
            else:
                self._recent_outcomes.append(0)
        elif won:
            self._bankroll_tracker.update_win(size_usd, entry_price, fee=taker_fee)
            self._recent_outcomes.append(1)
        else:
            self._bankroll_tracker.update_loss(size_usd, entry_price, fee=taker_fee)
            self._recent_outcomes.append(0)

        # v3.2 §5.8 post-loss cooldown — arm after any settled loss (live or
        # early-exit) large enough to clear the configured threshold. Paper
        # mode arms via ``WindowEventHandler._process_trade_outcome``.
        # v3.6.2: pass the resolved window's ts so the absolute-ts freeze
        # semantics hold even when gamma poll arms the cooldown mid-window
        # (post-mortem 2026-04-22 T4 observed zero-window freeze bug).
        if not won and pnl < 0.0:
            self._post_loss_cooldown.register_loss(-pnl, bankroll_before, pr.window_ts)

        # Position tracker
        self._position_tracker.record_window(
            pr.window_ts,
            pnl,
            traded=True,
            mode=mode,
            balance=self._bankroll_tracker.bankroll,
        )

        # Discord notification — skip for early exits, since momentum_signal
        # already fired send_early_exit at the time of the sell.
        if not is_early_exit:
            from shared.discord import send_bet_result

            send_bet_result(
                mode=mode,
                outcome="WIN" if won else "LOSS",
                pnl=pnl,
                entry_price=entry_price,
                side=sc.side.value,
                size_usd=size_usd,
                balance=self._bankroll_tracker.bankroll,
                market_outcome=outcome,
                maker_usd=pr.maker_usd,
                taker_usd=pr.taker_usd,
                signal_age_at_fire_h=pr.signal_age_at_fire_h,
                est_max_lifetime_h=pr.est_max_lifetime_h,
                lifetime_samples=pr.lifetime_samples,
                obi_threshold=sc.obi_threshold,
                obi_depth=sc.obi_depth.value,
            )

        # Journal record
        from shared.trade_journal import TradeJournal, TradeRecord

        self._journal.record_trade(
            TradeRecord(
                timestamp=TradeJournal.now_iso(),
                signal_id=sc.signal_id,
                signal_side=sc.side.value,
                window_ts=pr.window_ts,
                fired=True,
                filled=True,
                won=won,
                entry_price=entry_price,
                pnl=pnl,
                source=mode,
                signal_age_windows=pr.signal_age_windows,
            )
        )

        # SPRT decay update
        ds = self._decay_detector.update(won)
        if ds.verdict == "DEAD":
            log.warning(
                "DECAY DETECTED via resolution: Signal %s — "
                "SPRT concluded after %d trades (LLR=%.2f)",
                sc.signal_id,
                ds.n_trades,
                ds.llr,
            )

        # WR checkpoint — same milestone logic
        self._send_wr_checkpoint(ds, sc, pr, mode)

        # Write JSONL record
        self._write_live_window_record(pr, outcome, pnl)

        # Bookkeeping — session stats + loss tracker + state persistence
        signal_id = signal_id_override or sc.signal_id
        self._apply_bookkeeping(won, pnl, entry_price, signal_id, mode)

        # Clear pending
        self._pending = None

        result = ResolutionResult(
            won=won,
            pnl=pnl,
            verdict=ds.verdict,
            pending=pr,
        )
        self._last_result = result
        return result

    def _apply_bookkeeping(
        self,
        won: bool,
        pnl: float,
        entry_price: float,
        signal_id: str,
        mode: str,
    ) -> None:
        """Common bookkeeping after a live trade resolves."""
        self._session_stats.record_trade(won, pnl)
        if won:
            self._loss_tracker.record_win()
        else:
            self._loss_tracker.check_and_alert(
                streak=self._position_tracker.consecutive_losses,
                mode=mode,
                max_allowed=self._cfg.risk.max_consecutive_losses,
                entry_price=entry_price,
                pnl=pnl,
                session_pnl=self._position_tracker.total_pnl,
                daily_pnl=self._position_tracker.daily_pnl,
            )
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        self._position_tracker.save_state(self._paths.state, today, signal_id=signal_id)

    def _send_wr_checkpoint(
        self,
        ds: DecayState,
        sc: MomentumSignalConfig,
        pr: PendingResolution,
        mode: str,
    ) -> None:
        """Send win-rate checkpoint at milestone fill counts."""
        fills = ds.n_trades
        send_wr = fills <= 20 or fills in (25, 30, 40, 50, 75, 100) or fills % 50 == 0
        if not send_wr:
            return

        wins = ds.n_wins
        losses = fills - wins
        live_wr = (wins / fills * 100) if fills > 0 else 0.0
        be_entry = sc.avg_entry_price or pr.entry_price
        be_wr = be_entry * 100

        from shared.discord import send_live_wr_checkpoint

        send_live_wr_checkpoint(
            mode=mode,
            signal_id=sc.signal_id,
            fills=fills,
            wins=wins,
            losses=losses,
            live_wr_pct=live_wr,
            expected_wr_pct=sc.oos_win_rate_pct,
            entry_avg=be_entry,
            breakeven_wr_pct=be_wr,
            net_pnl=self._position_tracker.total_pnl,
            llr=ds.llr,
            signal_age_windows=pr.signal_age_windows,
        )

    def _write_live_window_record(
        self,
        pr: PendingResolution,
        outcome: str | None,
        pnl: float | None,
    ) -> None:
        """Write a per-window JSONL record for a resolved live trade.

        Emits the exact field names used by paper's WindowRecord writer so
        downstream analysis tools parse both modes identically. Crash
        recovery between fill and resolution is handled by ``TradeJournal``,
        so this is a single definitive write per window — no pending
        pre-record.
        """
        t = pr.kelly_telemetry
        bankroll_after = round(self._bankroll_tracker.bankroll, 4)
        bankroll_before_tel = t.get("bankroll_before")
        bankroll_before = round(bankroll_before_tel, 4) if bankroll_before_tel is not None else None
        side = pr.signal_cfg.side.value
        data: dict[str, Any] = {
            "window_ts": pr.window_ts,
            "window_delta_pct": pr.window_delta_pct,
            "direction": side,
            "rule_triggered": t.get("rule_triggered"),
            "rule_direction": t.get("rule_direction", side),
            "rule_entry_price": pr.entry_price,
            "rule_simulated_fill": True,
            "rule_signal_features": t.get("rule_signal_features"),
            "actual_outcome": outcome,
            "pnl_rules": pnl if pnl is not None else 0.0,
            "pnl_total": pnl if pnl is not None else 0.0,
            "latency_signal_ms": None,
            "latency_order_ms": None,
            "balance_usd": bankroll_after,
            "kelly_adjusted_p": t.get("kelly_adjusted_p"),
            "kelly_vol_discount": t.get("kelly_vol_discount"),
            "kelly_chop_discount": t.get("kelly_chop_discount"),
            "kelly_outcome_discount": t.get("kelly_outcome_discount"),
            "kelly_total_discount": t.get("kelly_total_discount"),
            "kelly_feedback_adj": t.get("kelly_feedback_adj"),
            "kelly_raw_f": t.get("kelly_raw_f"),
            "kelly_fractional_f": t.get("kelly_fractional_f"),
            "kelly_bet_size": t.get("kelly_bet_size"),
            "kelly_entry_price": t.get("kelly_entry_price"),
            "kelly_has_edge": t.get("kelly_has_edge"),
            "bankroll_before": bankroll_before,
            "bankroll_after": bankroll_after,
            "sprt_factor": t.get("sprt_factor", 1.0),
            "final_bet_size": t.get("final_bet_size", pr.size_usd),
            "early_exit": pr.early_exit_pnl is not None,
            "early_exit_sell_price": pr.early_exit_sell_price,
        }
        self._append_jsonl(pr.window_ts, data)

    def write_skipped_window_record(
        self,
        window_ts: int,
        window_delta_pct: float,
        direction: str,
        kelly_telemetry: KellyTelemetrySnapshot,
        actual_outcome: str | None,
    ) -> None:
        """Write a skip/no-fill JSONL line for live mode.

        Paper writes one record per window (including skips); live now
        matches that so session-level analysis (SKIP rates, coverage
        counts) produces identical results across modes.
        """
        t = kelly_telemetry
        bankroll_after = round(self._bankroll_tracker.bankroll, 4)
        bankroll_before_tel = t.get("bankroll_before")
        bankroll_before = round(bankroll_before_tel, 4) if bankroll_before_tel is not None else None
        data: dict[str, Any] = {
            "window_ts": window_ts,
            "window_delta_pct": window_delta_pct,
            "direction": direction,
            "rule_triggered": t.get("rule_triggered"),
            "rule_direction": t.get("rule_direction", ""),
            "rule_entry_price": 0.0,
            "rule_simulated_fill": False,
            "rule_signal_features": t.get("rule_signal_features"),
            "actual_outcome": actual_outcome,
            "pnl_rules": 0.0,
            "pnl_total": 0.0,
            "latency_signal_ms": None,
            "latency_order_ms": None,
            "balance_usd": bankroll_after,
            "kelly_adjusted_p": t.get("kelly_adjusted_p"),
            "kelly_vol_discount": t.get("kelly_vol_discount"),
            "kelly_chop_discount": t.get("kelly_chop_discount"),
            "kelly_outcome_discount": t.get("kelly_outcome_discount"),
            "kelly_total_discount": t.get("kelly_total_discount"),
            "kelly_feedback_adj": t.get("kelly_feedback_adj"),
            "kelly_raw_f": t.get("kelly_raw_f"),
            "kelly_fractional_f": t.get("kelly_fractional_f"),
            "kelly_bet_size": t.get("kelly_bet_size"),
            "kelly_entry_price": t.get("kelly_entry_price"),
            "kelly_has_edge": t.get("kelly_has_edge"),
            "bankroll_before": bankroll_before,
            "bankroll_after": bankroll_after,
            "sprt_factor": t.get("sprt_factor", 1.0),
            "final_bet_size": t.get("final_bet_size", 0.0),
            "early_exit": False,
            "early_exit_sell_price": None,
        }
        self._append_jsonl(window_ts, data)

    def _append_jsonl(self, window_ts: int, data: dict[str, Any]) -> None:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.fromtimestamp(window_ts, tz=UTC).strftime("%Y-%m-%d")
        path = self._results_dir / f"{date_str}.jsonl"
        try:
            with open(path, "ab") as f:
                f.write(orjson.dumps(data))
                f.write(b"\n")
        except OSError as exc:
            log.warning("failed to write window record for %d: %s", window_ts, exc)
