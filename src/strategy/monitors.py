"""Self-contained monitoring state machines extracted from the strategy loop.

Each class owns its state and exposes simple update/check methods.
No logic changes — these are pure extractions to reduce local-variable sprawl
in ``_strategy_loop()``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.discord import (
    send_bet_scale_squeeze,
    send_consecutive_loss_warning,
    send_session_summary,
    send_skip_streak_alert,
)

if TYPE_CHECKING:
    from risk.position_tracker import PositionTracker
    from strategy.momentum_signal import MomentumSignalConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Daily accumulators
# ---------------------------------------------------------------------------


@dataclass
class SessionAccumulator:
    """Collects session-lifetime trading statistics and sends periodic summaries."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    net_pnl: float = 0.0
    gross_won: float = 0.0
    gross_lost: float = 0.0
    bet_scale_sum: float = 0.0
    bet_scale_count: int = 0
    vol_sum: float = 0.0
    chop_sum: float = 0.0
    windows: int = 0
    signals_received: int = 0
    blocked_score: int = 0
    blocked_folds: int = 0

    def record_trade(self, won: bool, pnl: float) -> None:
        self.trades += 1
        self.net_pnl += pnl
        if won:
            self.wins += 1
            self.gross_won += pnl
        else:
            self.losses += 1
            self.gross_lost += abs(pnl)

    def record_window(self, bet_scale: float, vol: float, chop: float) -> None:
        self.windows += 1
        self.bet_scale_sum += bet_scale
        self.bet_scale_count += 1
        self.vol_sum += vol
        self.chop_sum += chop

    def send_summary(
        self,
        mode: str,
        balance: float,
    ) -> None:
        """Send session summary notification (no reset — accumulates for the entire session)."""
        if self.windows > 0:
            avg_scale = (
                (self.bet_scale_sum / self.bet_scale_count * 100)
                if self.bet_scale_count > 0
                else 100.0
            )
            avg_vol = self.vol_sum / self.windows
            avg_chop = self.chop_sum / self.windows
            send_session_summary(
                mode=mode,
                trades_placed=self.trades,
                wins=self.wins,
                losses=self.losses,
                net_pnl=self.net_pnl,
                gross_won=self.gross_won,
                gross_lost=self.gross_lost,
                avg_bet_scale_pct=avg_scale,
                avg_vol_reading_pct=avg_vol,
                avg_chop_flips=avg_chop,
                signals_received=self.signals_received,
                signals_blocked_score=self.blocked_score,
                signals_blocked_folds=self.blocked_folds,
                windows_total=self.windows,
                balance=balance,
            )


# ---------------------------------------------------------------------------
# Skip streak tracker
# ---------------------------------------------------------------------------


class SkipStreakTracker:
    """Tracks consecutive window skips and alerts after a time threshold."""

    def __init__(self, alert_hours: float = 12.0) -> None:
        self._alert_hours = alert_hours
        self._last_trade_time: float = time.time()
        self._skip_streak_windows: int = 0
        self._skip_reasons: dict[str, int] = {}
        self._alert_sent: bool = False

    def record_fill(self) -> None:
        self._last_trade_time = time.time()
        self._skip_streak_windows = 0
        self._skip_reasons.clear()
        self._alert_sent = False

    def record_skip(self, reason: str) -> None:
        self._skip_streak_windows += 1
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1

    def check_alert(
        self,
        mode: str,
        position_tracker: PositionTracker,
        signal_cfg: MomentumSignalConfig,
        bot_start_time: float,
    ) -> None:
        hours_since = (time.time() - self._last_trade_time) / 3600.0
        if hours_since >= self._alert_hours and not self._alert_sent:
            self._alert_sent = True
            pt = position_tracker
            sc = signal_cfg
            send_skip_streak_alert(
                mode=mode,
                hours_since_last_trade=hours_since,
                windows_skipped=self._skip_streak_windows,
                skip_reasons=self._skip_reasons,
                session_pnl=pt.total_pnl,
                win_rate_pct=(pt.windows_won / pt.windows_traded * 100)
                if pt.windows_traded > 0
                else 0.0,
                wins=pt.windows_won,
                total=pt.windows_traded,
                current_signal=f"#{sc.rank} {sc.side.value}",
                signal_score=sc.smart_score if sc.smart_score > 0 else None,
                signal_folds=f"{sc.wf_folds_appeared}/{sc.wf_total_test_folds}"
                if sc.wf_total_test_folds > 0
                else None,
                uptime_hours=(time.time() - bot_start_time) / 3600.0,
            )


# ---------------------------------------------------------------------------
# Consecutive loss tracker
# ---------------------------------------------------------------------------


class ConsecutiveLossTracker:
    """Tracks consecutive losses and sends alerts at threshold."""

    def __init__(self, warn_at: int = 5) -> None:
        self._warn_at = warn_at
        self._alert_sent_at: int = 0  # streak count when last alert was sent

    def record_win(self) -> None:
        self._alert_sent_at = 0

    def check_and_alert(
        self,
        streak: int,
        mode: str,
        max_allowed: int,
        entry_price: float,
        pnl: float,
        session_pnl: float,
        daily_pnl: float,
    ) -> None:
        if streak >= self._warn_at and streak > self._alert_sent_at:
            self._alert_sent_at = streak
            send_consecutive_loss_warning(
                mode=mode,
                streak=streak,
                max_allowed=max_allowed,
                recent_losses=[{"entry": entry_price, "pnl": pnl}],
                session_pnl=session_pnl,
                daily_pnl=daily_pnl,
            )


# ---------------------------------------------------------------------------
# Bet scale squeeze tracker
# ---------------------------------------------------------------------------


class BetScaleSqueezeTracker:
    """Detects prolonged bet scale compression and alerts."""

    def __init__(
        self,
        threshold_pct: float = 30.0,
        alert_windows: int = 12,
    ) -> None:
        self._threshold_pct = threshold_pct
        self._alert_windows = alert_windows
        self._consecutive: int = 0
        self._alert_sent: bool = False

    def update(
        self,
        sprt_factor: float,
        mode: str,
        llr_conf_pct: float,
        age_tap_pct: float,
        vol_stddev_pct: float,
        chop_avg_flips: float,
        llr: float,
        signal_age_windows: int,
    ) -> None:
        if sprt_factor * 100 <= self._threshold_pct:
            self._consecutive += 1
            if self._consecutive >= self._alert_windows and not self._alert_sent:
                self._alert_sent = True
                send_bet_scale_squeeze(
                    mode=mode,
                    combined_scale_pct=sprt_factor * 100,
                    sprt_scale_pct=llr_conf_pct,
                    age_scale_pct=age_tap_pct,
                    vol_scale_pct=0.0,
                    chop_scale_pct=0.0,
                    vol_stddev_pct=vol_stddev_pct,
                    chop_avg_flips=chop_avg_flips,
                    llr=llr,
                    signal_age_windows=signal_age_windows,
                    consecutive_squeeze_windows=self._consecutive,
                )
        else:
            self._consecutive = 0
            self._alert_sent = False
