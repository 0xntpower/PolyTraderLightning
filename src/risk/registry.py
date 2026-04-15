"""RiskRegistry — composable risk gate with pluggable checks.

Replaces the hardcoded if-chain in ``RiskLimits.can_trade()`` with a
registry of ``RiskCheck`` implementations. New risk checks can be added
without modifying existing code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from config import RiskConfig
    from risk.position_tracker import PositionTracker
    from strategy.kelly import BankrollTracker

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol + verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """Result of a single risk check."""

    allowed: bool
    reason: str = ""


@runtime_checkable
class RiskCheck(Protocol):
    """Interface for a composable risk check."""

    name: str

    def check(self, additional_usd: float) -> RiskVerdict: ...


# ---------------------------------------------------------------------------
# Built-in checks (extracted from RiskLimits)
# ---------------------------------------------------------------------------


class DailyLossLimit:
    """Halt trading when daily P&L exceeds the loss threshold."""

    name = "daily_loss_limit"

    def __init__(self, max_daily_loss_usd: float, tracker: PositionTracker) -> None:
        self._max = max_daily_loss_usd
        self._tracker = tracker

    def check(self, additional_usd: float = 0.0) -> RiskVerdict:  # noqa: ARG002  # required by RiskCheck Protocol
        if self._tracker.daily_pnl <= -self._max:
            return RiskVerdict(
                allowed=False,
                reason="daily loss limit hit: %.2f" % self._tracker.daily_pnl,
            )
        return RiskVerdict(allowed=True)


class ConsecutiveLossLimit:
    """Halt trading after too many consecutive losses."""

    name = "consecutive_loss_limit"

    def __init__(self, max_consecutive: int, tracker: PositionTracker) -> None:
        self._max = max_consecutive
        self._tracker = tracker

    def check(self, additional_usd: float = 0.0) -> RiskVerdict:  # noqa: ARG002  # required by RiskCheck Protocol
        if self._tracker.consecutive_losses >= self._max:
            return RiskVerdict(
                allowed=False,
                reason="consecutive loss limit: %d" % self._tracker.consecutive_losses,
            )
        return RiskVerdict(allowed=True)


class WindowExposureCap:
    """Block trades that would exceed the per-window position cap."""

    name = "window_exposure_cap"

    def __init__(self, max_per_window_usd: float, tracker: PositionTracker) -> None:
        self._max = max_per_window_usd
        self._tracker = tracker

    def check(self, additional_usd: float = 0.0) -> RiskVerdict:
        projected = self._tracker.window_exposure_usd + additional_usd
        if projected > self._max:
            return RiskVerdict(
                allowed=False,
                reason="window position cap: %.2f + %.2f > %.2f"
                % (self._tracker.window_exposure_usd, additional_usd, self._max),
            )
        return RiskVerdict(allowed=True)


class BankrollCorruptedLimit:
    """Halt trading when the bankroll tracker flags irrecoverable state.

    Trips on BankrollTracker.is_corrupted — see kelly.BankrollTracker for
    the conditions that set it (negative update_loss, rejected sync_from_api).
    """

    name = "bankroll_corrupted"

    def __init__(self, tracker: BankrollTracker) -> None:
        self._tracker = tracker

    def check(self, additional_usd: float = 0.0) -> RiskVerdict:  # noqa: ARG002  # required by RiskCheck Protocol
        if self._tracker.is_corrupted:
            return RiskVerdict(
                allowed=False,
                reason=f"bankroll corrupted: {self._tracker.corruption_reason}",
            )
        return RiskVerdict(allowed=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RiskRegistry:
    """Composable risk gate — runs all registered checks.

    Drop-in replacement for ``RiskLimits``: exposes the same ``can_trade()``,
    ``halted``, ``halt_reason``, and ``reset_halt()`` interface.
    """

    def __init__(self) -> None:
        self._checks: list[RiskCheck] = []
        self.halted: bool = False
        self._halt_reason: str = ""
        self.tracker: PositionTracker | None = None

    def register(self, check: RiskCheck) -> None:
        """Add a risk check to the registry."""
        self._checks.append(check)

    def can_trade(self, additional_usd: float = 0.0) -> bool:
        """Run all checks. First failure halts or blocks."""
        if self.halted:
            return False

        for check in self._checks:
            verdict = check.check(additional_usd)
            if not verdict.allowed:
                # Window exposure cap is a soft block (doesn't halt),
                # while daily loss and consecutive loss are kill switches
                if check.name == "window_exposure_cap":
                    log.warning("%s", verdict.reason)
                    return False
                self._halt(verdict.reason)
                return False

        return True

    def _halt(self, reason: str) -> None:
        self.halted = True
        self._halt_reason = reason
        log.error("KILL SWITCH: %s — trading halted", reason)

    def reset_halt(self) -> None:
        """Reset halt state (called at midnight UTC)."""
        self.halted = False
        self._halt_reason = ""

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def register_bankroll_check(self, bankroll_tracker: BankrollTracker) -> None:
        """Wire a BankrollCorruptedLimit after construction.

        RiskRegistry is built in ``run()`` before the strategy loop creates
        ``BankrollTracker``, so the bankroll check is attached post-hoc.
        """
        self.register(BankrollCorruptedLimit(bankroll_tracker))

    @classmethod
    def from_config(cls, cfg: RiskConfig, tracker: PositionTracker) -> RiskRegistry:
        """Build the default registry from a RiskConfig + PositionTracker."""
        registry = cls()
        registry.tracker = tracker
        registry.register(DailyLossLimit(cfg.max_daily_loss_usd, tracker))
        registry.register(ConsecutiveLossLimit(cfg.max_consecutive_losses, tracker))
        registry.register(WindowExposureCap(cfg.max_position_per_window_usd, tracker))
        return registry
