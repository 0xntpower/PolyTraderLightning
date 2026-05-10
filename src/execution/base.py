"""OrderExecutor protocol — shared interface for live and paper order managers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from strategy.signal import Signal


class KellyTelemetrySnapshot(TypedDict, total=False):
    """Point-in-time Kelly/rule capture recorded at fire, read at resolution.

    Shared schema so paper WindowRecord and live ResolutionManager write the
    same fields to JSONL — prevents analysis tooling from branching per mode.
    Every field is optional (total=False) because fires can occur before
    Kelly is fully populated (e.g. warmup) or when sizing is bypassed.
    """

    rule_triggered: int
    rule_direction: str
    rule_obi_threshold: float
    rule_obi_depth: str
    rule_signal_features: dict[str, float]
    kelly_adjusted_p: float | None
    kelly_vol_discount: float | None
    kelly_chop_discount: float | None
    kelly_outcome_discount: float | None
    kelly_total_discount: float | None
    kelly_feedback_adj: float | None
    kelly_raw_f: float | None
    kelly_fractional_f: float | None
    kelly_bet_size: float | None
    kelly_entry_price: float | None
    kelly_has_edge: bool | None
    bankroll_before: float | None
    sprt_factor: float | None
    final_bet_size: float | None


@runtime_checkable
class OrderExecutor(Protocol):
    """Minimal interface used by MomentumSignalStrategy to place/manage orders.

    Both ``OrderManager`` (live CLOB) and ``PaperOrderManager`` (simulator)
    implement this protocol.  Strategy code depends only on this interface,
    making it testable without CLOB API mocks.
    """

    mode: str  # "live" or "paper"

    async def place_maker_order(
        self,
        token_id: str,
        price: float,
        size_usd: float,
        tier: str,
    ) -> str | None: ...

    async def place_taker_order(
        self,
        token_id: str,
        price: float,
        size_usd: float,
        tier: str,
    ) -> str | None: ...

    async def cancel_order(self, order_id: str) -> bool: ...

    async def cancel_all_active(self) -> None: ...

    def filled_usd(self, order_id: str) -> float:
        """Cumulative USD filled against this order so far. 0.0 if unknown."""
        ...

    def is_order_fully_filled(self, order_id: str) -> bool:
        """True when cumulative fills cover the intended size (within tolerance).

        Replaces the older "any fill -> True" semantic, which silently accepted
        partial fills as complete (post-mortem 2026-04-22 §5.2) — at live-CLOB
        book depths the maker intent frequently matches only 5-20 % of size,
        and the strategy needs to distinguish "done" from "partial so escalate
        the remainder".
        """
        ...

    def has_filled_buys(self) -> bool:
        """True when at least one BUY fill is confirmed for the current window.

        Used by ``MomentumSignalStrategy._execute_early_exit`` to defer a CUSUM
        exit when the entry order's WS fill confirmation hasn't arrived yet —
        otherwise the exit no-ops on an empty position and ``_early_exit_triggered``
        latches True, leaving the unfilled entry on-book until window-close
        cancel sweep (post-mortem 2026-05-09 §6 H-cusum-pre-fill, T11).
        """
        ...

    def set_rule_triggered(
        self,
        rule_id: int,
        direction: str,
        signal: Signal | None,
        obi_threshold: float = 0.0,
        obi_depth: str = "none",
    ) -> None: ...

    def set_kelly_fields(self, **kwargs: float | bool | None) -> None: ...

    async def exit_position_early(self, sell_price: float) -> float | None: ...
