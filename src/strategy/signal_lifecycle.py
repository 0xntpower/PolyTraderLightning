"""SignalLifecycle — explicit state machine for signal age, fire-rate, and decay tracking.

Consolidates the implicit lifecycle transitions that were previously scattered
across _strategy_loop() in main.py:
- Signal age + fire-rate tracking
- Fire-stall detection → IDLE transition
- SPRT decay → IDLE transition
- Shadow tracking (continues monitoring a signal after it goes IDLE)
- Signal swap transitions (same signal refresh vs different signal reset)

States:
  ACTIVE → IDLE:fire_stall → (shadow tracking)
  ACTIVE → IDLE:decay      → (shadow tracking)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from strategy.signal import Direction

if TYPE_CHECKING:
    from config import RulesStrategyConfig
    from market_data.state import MarketState
    from shared.trade_journal import TradeJournal
    from strategy.momentum_signal import MomentumSignalConfig, MomentumSignalStrategy
    from strategy.signal import Signal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events emitted by lifecycle transitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """Emitted when a lifecycle state transition occurs."""

    kind: str  # "fire_stall", "decay", "shadow_complete", "alive_reset"
    detail: str = ""


# ---------------------------------------------------------------------------
# Shadow tracking state
# ---------------------------------------------------------------------------


@dataclass
class ShadowState:
    """Encapsulates all shadow tracking state."""

    signal: MomentumSignalStrategy
    signal_cfg: MomentumSignalConfig
    age_at_idle: int
    windows_tracked: int = 0
    fires: int = 0
    fills: int = 0
    wins: int = 0


# ---------------------------------------------------------------------------
# Signal lifecycle state machine
# ---------------------------------------------------------------------------


class SignalLifecycle:
    """Explicit state machine for signal lifecycle.

    Replaces the scattered lifecycle variables and transition logic
    that were previously locals/attrs in _strategy_loop().
    """

    def __init__(self) -> None:
        self.signal_age_windows: int = 0
        self.windows_since_last_fire: int = 0
        self.idle_reason: str | None = None
        self.running_entry_sum: float = 0.0
        self.running_entry_count: int = 0
        self._shadow: ShadowState | None = None

    @property
    def is_idle(self) -> bool:
        return self.idle_reason is not None

    @property
    def shadow(self) -> ShadowState | None:
        return self._shadow

    # ------------------------------------------------------------------
    # Window-complete lifecycle update
    # ------------------------------------------------------------------

    def on_window_complete(
        self,
        fired: bool,
        fire_stall_windows: int,
        signal_cfg: MomentumSignalConfig,
        rules_cfg: RulesStrategyConfig,
        state: MarketState,
    ) -> LifecycleEvent | None:
        """Process end-of-window lifecycle updates.

        Updates fire-rate tracking, signal age, and detects fire-stall.
        Returns a LifecycleEvent if a state transition occurred.
        """
        # Fire-rate tracking
        if fired:
            self.windows_since_last_fire = 0
        else:
            self.windows_since_last_fire += 1

        self.signal_age_windows += 1

        # Periodic status log
        if self.signal_age_windows % 50 == 0:
            log.info(
                "STATUS signal=%s age=%dw since_last_fire=%dw",
                signal_cfg.signal_id,
                self.signal_age_windows,
                self.windows_since_last_fire,
            )

        # Fire-stall detection
        if self.idle_reason is None and self.windows_since_last_fire >= fire_stall_windows:
            self.idle_reason = "fire_stall"
            log.warning(
                "FIRE STALL: Signal %s has not fired in %d windows. "
                "Market conditions may have shifted. Entering IDLE state.",
                signal_cfg.signal_id,
                self.windows_since_last_fire,
            )
            self._start_shadow(signal_cfg, rules_cfg, state)
            return LifecycleEvent(
                kind="fire_stall",
                detail=f"signal={signal_cfg.signal_id} windows={self.windows_since_last_fire}",
            )

        return None

    # ------------------------------------------------------------------
    # SPRT verdict handling
    # ------------------------------------------------------------------

    def on_sprt_verdict(
        self,
        verdict: str,
        signal_cfg: MomentumSignalConfig,
        rules_cfg: RulesStrategyConfig,
        state: MarketState,
    ) -> LifecycleEvent | None:
        """Process an SPRT verdict. Returns event if state transition occurred."""
        if verdict == "DEAD" and self.idle_reason is None:
            self.idle_reason = "decay"
            self._start_shadow(signal_cfg, rules_cfg, state)
            return LifecycleEvent(
                kind="decay",
                detail=f"signal={signal_cfg.signal_id}",
            )
        return None

    # ------------------------------------------------------------------
    # Running entry average (for dynamic p_dead)
    # ------------------------------------------------------------------

    def record_entry_price(self, entry_price: float) -> float:
        """Record an entry price and return the running average."""
        self.running_entry_sum += entry_price
        self.running_entry_count += 1
        return self.running_entry_sum / self.running_entry_count

    @property
    def avg_entry_price(self) -> float:
        if self.running_entry_count == 0:
            return 0.0
        return self.running_entry_sum / self.running_entry_count

    # ------------------------------------------------------------------
    # Shadow tracking
    # ------------------------------------------------------------------

    def _start_shadow(
        self,
        signal_cfg: MomentumSignalConfig,
        rules_cfg: RulesStrategyConfig,
        state: MarketState,
    ) -> None:
        """Initialize shadow tracking for the current signal."""
        from strategy.momentum_signal import MomentumSignalStrategy

        self._shadow = ShadowState(
            signal=MomentumSignalStrategy(rules_cfg, state, signal_cfg),
            signal_cfg=signal_cfg,
            age_at_idle=self.signal_age_windows,
        )

    def tick_shadow(self, signal: Signal, time_remaining: float) -> None:
        """Feed a tick to the shadow strategy during IDLE state.

        Accumulates samples in the observation window and detects fires.
        """
        if self._shadow is None:
            return

        sc = self._shadow.signal_cfg
        bn_dir_pct = signal.bn_direction_from_open_pct * 100.0

        if sc.observe_to_s <= time_remaining <= sc.observe_from_s:
            self._shadow.signal._accumulate(
                bn_dir_pct, time_remaining, self._shadow.signal._gate_obi(signal)
            )
        elif time_remaining < sc.observe_to_s and not self._shadow.signal._fired:
            self._shadow.signal._fired = True
            if self._shadow.signal._conditions_met():
                self._shadow.fires += 1

    def finalize_shadow_window(
        self,
        outcome_dir: str | None,
        journal: TradeJournal,
        window_ts: int,
        shadow_tracking_windows: int,
        mode: str,
    ) -> LifecycleEvent | None:
        """Finalize shadow tracking for a completed window.

        Records the shadow journal entry, checks for expiry.
        Returns LifecycleEvent("shadow_complete") if tracking is done.
        """
        if self._shadow is None:
            return None

        from shared.trade_journal import TradeRecord

        sh = self._shadow
        sh.windows_tracked += 1

        shadow_fired = sh.signal._fired and sh.signal._conditions_met()
        shadow_won: bool | None = None

        if shadow_fired and outcome_dir:
            shadow_won = (sh.signal_cfg.side == Direction.UP and outcome_dir == "up") or (
                sh.signal_cfg.side == Direction.DOWN and outcome_dir == "down"
            )
            if shadow_won:
                sh.wins += 1
            sh.fills += 1

        journal.record_trade(
            TradeRecord(
                timestamp=journal.now_iso(),
                signal_id=sh.signal_cfg.signal_id,
                signal_side=sh.signal_cfg.side.value,
                window_ts=window_ts,
                fired=shadow_fired,
                filled=shadow_fired,
                won=shadow_won,
                entry_price=0.0,
                pnl=0.0,
                source="shadow",
                signal_age_windows=sh.age_at_idle + sh.windows_tracked,
            )
        )

        # Check expiry
        if sh.windows_tracked >= shadow_tracking_windows:
            total_age = sh.age_at_idle + sh.windows_tracked
            shadow_wr = (sh.wins / sh.fills * 100) if sh.fills > 0 else 0.0
            decay_correct = sh.fills == 0 or shadow_wr < 81.0

            log.info(
                "Shadow tracking complete for %s: %d windows observed, "
                "%d fires, %d/%d wins (%.0f%%). "
                "Total signal lifetime: %d windows from activation to shadow end.",
                sh.signal_cfg.signal_id,
                sh.windows_tracked,
                sh.fires,
                sh.wins,
                sh.fills,
                shadow_wr,
                total_age,
            )

            from shared.discord import send_shadow_tracking_result

            send_shadow_tracking_result(
                mode=mode,
                signal_id=sh.signal_cfg.signal_id,
                windows_tracked=sh.windows_tracked,
                fires=sh.fires,
                fills=sh.fills,
                wins=sh.wins,
                shadow_win_rate_pct=shadow_wr,
                total_signal_age=total_age,
                decay_was_correct=decay_correct,
            )

            self._shadow = None
            return LifecycleEvent(
                kind="shadow_complete",
                detail=f"signal={sh.signal_cfg.signal_id} "
                f"wr={shadow_wr:.0f}% decay_correct={decay_correct}",
            )

        # Reset shadow strategy for next window
        sh.signal.reset()
        return None

    # ------------------------------------------------------------------
    # Signal swap transitions
    # ------------------------------------------------------------------

    def on_same_signal_refresh(self) -> None:
        """Preserve SPRT DEAD but clear other idle reasons."""
        if self.idle_reason != "decay":
            self.idle_reason = None
        self.windows_since_last_fire = 0

    def reset_for_new_signal(self) -> None:
        """Full reset when the orchestrator delivers a different signal."""
        self.signal_age_windows = 0
        self.windows_since_last_fire = 0
        self.idle_reason = None
        self.running_entry_sum = 0.0
        self.running_entry_count = 0
        # Shadow tracking continues independently (not reset)

    def clear_shadow(self) -> None:
        """Explicitly clear shadow tracking."""
        self._shadow = None
