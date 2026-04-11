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
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import orjson

from strategy.signal import Direction

if TYPE_CHECKING:
    from collections import deque
    from pathlib import Path

    from config import Config, DataPaths
    from market_data.state import MarketState
    from risk.fee_tracker import FeeTracker
    from risk.position_tracker import PositionTracker
    from shared.decay_detector import DecayDetector, DecayState
    from shared.trade_journal import TradeJournal
    from strategy.kelly import BankrollTracker
    from strategy.momentum_signal import MomentumSignalConfig
    from strategy.monitors import ConsecutiveLossTracker, SessionAccumulator
    from strategy.window_tracker import WindowTracker

log = logging.getLogger(__name__)

# Timing constants — match the values previously in main.py
RESOLUTION_POLL_DELAY = 30.0  # seconds before first Gamma poll
RESOLUTION_POLL_INTERVAL = 5.0  # seconds between polls
RESOLUTION_TIMEOUT = 1200.0  # fall back after ~20 min


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
    ) -> None:
        self._window_tracker = window_tracker
        self._state = state
        self._fee_tracker = fee_tracker
        self._bankroll_tracker = bankroll_tracker
        self._position_tracker = position_tracker
        self._journal = journal
        self._decay_detector = decay_detector
        self._recent_outcomes = recent_outcomes
        self._results_dir = results_dir
        self._session_stats = session_stats
        self._loss_tracker = loss_tracker
        self._cfg = cfg
        self._paths = paths
        self._pending: PendingResolution | None = None

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

        self._pending = PendingResolution(
            window_ts=window_ts,
            slug=slug,
            signal_cfg=signal_cfg,
            entry_price=entry_price,
            size_usd=size_usd,
            signal_age_windows=signal_age_windows,
            created_at=time.time(),
            snapshot_outcome=snapshot_outcome,
        )
        log.info(
            "trade pending resolution: window=%d slug=%s "
            "side=%s entry=%.2f size=$%.2f (snapshot_outcome=%s)",
            window_ts,
            slug,
            signal_cfg.side.value,
            entry_price,
            size_usd,
            snapshot_outcome,
        )
        return force_result

    async def tick(self, now: float, *, mode: str = "live") -> ResolutionResult | None:
        """Called every strategy tick. Polls Gamma API if timing conditions met.

        Returns ResolutionResult if resolved (confirmed or timeout), None if still pending.
        """
        if self._pending is None:
            return None

        pr = self._pending
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
            return self._resolve(pr, resolution_data.outcome, mode=mode)

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
            return self._resolve(pr, fallback, mode=mode)

        return None

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

        # Taker fee deduction
        taker_fee = self._fee_tracker.record_taker_fee(entry_price, shares)

        bet_won = outcome == sc.side.value
        if bet_won:
            pnl = round(shares * (1.0 - entry_price) - taker_fee, 4)
            won = True
        else:
            pnl = round(-(shares * entry_price) - taker_fee, 4)
            won = False

        log.info(
            "live resolution: window=%d side=%s outcome=%s → %s pnl=$%.4f "
            "(entry=%.2f shares=%.1f taker_fee=$%.4f)",
            pr.window_ts,
            sc.side.value,
            outcome,
            "WIN" if won else "LOSS",
            pnl,
            entry_price,
            shares,
            taker_fee,
        )

        # Bankroll update
        if won:
            self._bankroll_tracker.update_win(size_usd, entry_price, fee=taker_fee)
            self._recent_outcomes.append(1)
        else:
            self._bankroll_tracker.update_loss(size_usd, entry_price, fee=taker_fee)
            self._recent_outcomes.append(0)

        # Position tracker
        self._position_tracker.record_window(
            pr.window_ts,
            pnl,
            traded=True,
            mode=mode,
            balance=self._bankroll_tracker.bankroll,
        )

        # Discord notification
        from shared.discord import send_bet_result

        send_bet_result(
            mode=mode,
            outcome="WIN" if won else "LOSS",
            pnl=pnl,
            entry_price=entry_price,
            side=sc.side.value,
            size_usd=size_usd,
            balance=self._bankroll_tracker.bankroll,
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

        return ResolutionResult(
            won=won,
            pnl=pnl,
            verdict=ds.verdict,
            pending=pr,
        )

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

    def _write_live_window_record(self, pr: PendingResolution, outcome: str, pnl: float) -> None:
        """Write a per-window JSONL record for live mode."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.fromtimestamp(pr.window_ts, tz=UTC).strftime("%Y-%m-%d")
        path = self._results_dir / f"{date_str}.jsonl"

        data = {
            "window_ts": pr.window_ts,
            "window_delta_pct": 0.0,  # not available after deferral
            "direction": outcome,
            "actual_outcome": outcome,
            "fired": True,
            "filled": True,
            "pnl": pnl,
            "entry_price": pr.entry_price,
            "size_usd": pr.size_usd,
            "bankroll": round(self._bankroll_tracker.bankroll, 4),
            "signal_features": None,
            "kelly_adjusted_p": None,
            "kelly_vol_discount": None,
            "kelly_chop_discount": None,
            "kelly_outcome_discount": None,
            "kelly_total_discount": None,
            "kelly_feedback_adj": None,
            "kelly_raw_f": None,
            "kelly_fractional_f": None,
            "kelly_bet_size": None,
            "kelly_has_edge": None,
            "sprt_factor": 1.0,
        }

        try:
            with open(path, "ab") as f:
                f.write(orjson.dumps(data))
                f.write(b"\n")
        except OSError as exc:
            log.warning("failed to write live window record: %s", exc)
