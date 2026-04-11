"""OrderExecutor protocol — shared interface for live and paper order managers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from strategy.signal import Signal


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

    def is_order_filled(self, order_id: str) -> bool: ...

    def set_rule_triggered(
        self,
        rule_id: int,
        direction: str,
        signal: Signal | None,
    ) -> None: ...

    def set_kelly_fields(self, **kwargs: float | bool | None) -> None: ...

    async def exit_position_early(self, sell_price: float) -> float | None: ...
