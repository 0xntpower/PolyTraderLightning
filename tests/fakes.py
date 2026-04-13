"""Shared test fakes — lightweight substitutes for production components.

These fakes implement the same interfaces as the real objects but record
calls and return canned responses, enabling isolated unit testing of
strategy and resolution logic without network, disk, or CLOB API access.
"""

from __future__ import annotations

from dataclasses import dataclass

from market_data.state import MarketState
from strategy.signal import Direction

# ---------------------------------------------------------------------------
# Fake OrderExecutor — records calls, returns configurable results
# ---------------------------------------------------------------------------


@dataclass
class OrderCall:
    """Record of a single order call for assertion."""

    method: str  # "place_maker_order" or "place_taker_order"
    token_id: str
    price: float
    size_usd: float
    tier: str


class FakeOrderExecutor:
    """Fake implementation of the OrderExecutor protocol for testing."""

    mode: str = "test"

    def __init__(self) -> None:
        self.calls: list[OrderCall] = []
        self.next_order_id: str | None = "fake-order-1"
        self.filled_orders: set[str] = set()
        self.cancelled_orders: set[str] = set()
        self._rule_triggered: tuple[int, str, object] | None = None
        self._kelly_fields: dict | None = None

    async def place_maker_order(
        self,
        token_id: str,
        price: float,
        size_usd: float,
        tier: str,
    ) -> str | None:
        self.calls.append(OrderCall("place_maker_order", token_id, price, size_usd, tier))
        return self.next_order_id

    async def place_taker_order(
        self,
        token_id: str,
        price: float,
        size_usd: float,
        tier: str,
    ) -> str | None:
        self.calls.append(OrderCall("place_taker_order", token_id, price, size_usd, tier))
        return self.next_order_id

    async def cancel_order(self, order_id: str) -> bool:
        self.cancelled_orders.add(order_id)
        return order_id not in self.filled_orders

    async def cancel_all_active(self) -> None:
        pass

    def is_order_filled(self, order_id: str) -> bool:
        return order_id in self.filled_orders

    def set_rule_triggered(self, rule_id: int, direction: str, signal: object) -> None:
        self._rule_triggered = (rule_id, direction, signal)

    def set_kelly_fields(self, **kwargs) -> None:
        self._kelly_fields = kwargs


# ---------------------------------------------------------------------------
# Helpers for building configured MarketState
# ---------------------------------------------------------------------------


def make_market_state(
    *,
    btc_binance: float = 100_000.0,
    btc_chainlink: float = 100_000.0,
    window_open_price: float = 100_000.0,
    binance_window_open_price: float = 100_000.0,
    best_ask_up: float = 0.85,
    best_ask_down: float = 0.15,
    best_bid_up: float = 0.84,
    best_bid_down: float = 0.14,
    up_token_id: str = "token_up",
    down_token_id: str = "token_down",
    time_remaining: float = 200.0,
    has_fresh_book_data: bool = True,
) -> MarketState:
    """Create a MarketState with sensible defaults for testing."""
    state = MarketState()
    state.btc_binance = btc_binance
    state.btc_chainlink = btc_chainlink
    state.window_open_price = window_open_price
    state.binance_window_open_price = binance_window_open_price
    state.open_price_captured = window_open_price > 0
    state.binance_open_price_captured = binance_window_open_price > 0
    state.best_ask_up = best_ask_up
    state.best_ask_down = best_ask_down
    state.best_bid_up = best_bid_up
    state.best_bid_down = best_bid_down
    state.up_token_id = up_token_id
    state.down_token_id = down_token_id
    state.time_remaining = time_remaining
    state.has_fresh_book_data = has_fresh_book_data
    return state


# ---------------------------------------------------------------------------
# Default signal configs for testing
# ---------------------------------------------------------------------------

from strategy.momentum_signal import MomentumSignalConfig


def make_signal_config(
    *,
    side: Direction = Direction.UP,
    observe_from_s: float = 240.0,
    observe_to_s: float = 180.0,
    min_delta_pct: float = 0.05,
    max_variance_pct: float = 0.10,
    oos_win_rate_pct: float = 90.0,
    oos_matches: int = 30,
    avg_entry_price: float = 0.85,
    ev_per_trade: float = 0.02,
    rank: int = 1,
) -> MomentumSignalConfig:
    """Create a MomentumSignalConfig with sensible test defaults."""
    return MomentumSignalConfig(
        rank=rank,
        side=side,
        observe_from_s=observe_from_s,
        observe_to_s=observe_to_s,
        min_delta_pct=min_delta_pct,
        max_variance_pct=max_variance_pct,
        train_win_rate_pct=92.0,
        oos_win_rate_pct=oos_win_rate_pct,
        bh_adjusted_p_value=0.001,
        oos_matches=oos_matches,
        avg_entry_price=avg_entry_price,
        ev_per_trade=ev_per_trade,
    )


from config import (
    ErosionConfig,
    RegimeConfig,
    RulesStrategyConfig,
    SignalLifecycleConfig,
    SizingConfig,
)


def make_rules_config(**overrides) -> RulesStrategyConfig:
    """Create a RulesStrategyConfig with test defaults.

    Skip-maker fast path is off by default here so tests that exercise the
    maker path remain deterministic regardless of the signal's oos_wr.
    Tests for the skip-maker path turn it on explicitly.
    """
    defaults = {
        "enabled": True,
        "entry_window_stop": 5,
        "min_win_rate": 0.50,
        "maker_timeout_s": 20.0,
        "skip_maker_min_oos_wr_pct": 0.0,
        "skip_maker_max_stddev_pct": 0.0,
    }
    defaults.update(overrides)
    return RulesStrategyConfig(**defaults)


def make_lifecycle_config(**overrides) -> SignalLifecycleConfig:
    """Create a SignalLifecycleConfig with test defaults."""
    return SignalLifecycleConfig(**overrides)


def make_sizing_config(**overrides) -> SizingConfig:
    """Create a SizingConfig with test defaults."""
    return SizingConfig(**overrides)


def make_regime_config(**overrides) -> RegimeConfig:
    """Create a RegimeConfig with test defaults."""
    return RegimeConfig(**overrides)


def make_erosion_config(**overrides) -> ErosionConfig:
    """Create an ErosionConfig with test defaults."""
    return ErosionConfig(**overrides)
