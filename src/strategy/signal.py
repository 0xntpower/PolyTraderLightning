"""Compute window delta, direction, feed agreement, and all backtest features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from market_data.state import MarketState, WindowSnapshot


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class Signal:
    # Core direction signal
    delta_pct: float  # (chainlink_now - open) / open * 100
    direction: Direction
    feeds_agree: bool

    # Extended backtest features (all computed when open prices are available)
    time_remaining: float = 0.0
    binance_obi: float = 0.0
    delta_bn_cl_pct: float = 0.0  # (binance - chainlink) / chainlink
    poly_implied_up_prob: float = 0.0  # (best_bid_up + best_ask_up) / 2
    bn_direction_from_open_pct: float = 0.0  # (binance_now - binance_open) / binance_open
    cl_direction_from_open_pct: float = 0.0  # (chainlink_now - chainlink_open) / chainlink_open
    poly_spread_up: float = 0.0  # best_ask_up - best_bid_up
    poly_spread_down: float = 0.0  # best_ask_down - best_bid_down

    def as_feature_dict(self) -> dict[str, float]:
        """Return all backtest features as a dict for logging/recording."""
        return {
            "time_remaining": round(self.time_remaining, 2),
            "binance_obi": round(self.binance_obi, 4),
            "delta_bn_cl_pct": round(self.delta_bn_cl_pct, 6),
            "poly_implied_up_prob": round(self.poly_implied_up_prob, 4),
            "bn_direction_from_open_pct": round(self.bn_direction_from_open_pct, 6),
            "cl_direction_from_open_pct": round(self.cl_direction_from_open_pct, 6),
            "poly_spread_up": round(self.poly_spread_up, 4),
            "poly_spread_down": round(self.poly_spread_down, 4),
        }


def compute_signal(state: MarketState) -> Signal:
    """Compute full signal from current live state."""
    if state.window_open_price <= 0 or state.btc_chainlink <= 0:
        return Signal(
            delta_pct=0.0,
            direction=Direction.NONE,
            feeds_agree=False,
            time_remaining=state.time_remaining,
            binance_obi=state.binance_obi,
        )

    return _compute(
        chainlink=state.btc_chainlink,
        binance=state.btc_binance,
        chainlink_open=state.window_open_price,
        binance_open=state.binance_window_open_price,
        best_bid_up=state.best_bid_up,
        best_ask_up=state.best_ask_up,
        best_bid_down=state.best_bid_down,
        best_ask_down=state.best_ask_down,
        binance_obi=state.binance_obi,
        time_remaining=state.time_remaining,
    )


def compute_signal_from_snapshot(snap: WindowSnapshot) -> Signal:
    """Compute full signal from a frozen end-of-window snapshot."""
    if snap.open_price <= 0 or snap.chainlink_price <= 0:
        return Signal(
            delta_pct=0.0, direction=Direction.NONE, feeds_agree=False, binance_obi=snap.binance_obi
        )

    return _compute(
        chainlink=snap.chainlink_price,
        binance=snap.binance_price,
        chainlink_open=snap.open_price,
        binance_open=snap.binance_open_price,
        best_bid_up=snap.best_bid_up,
        best_ask_up=snap.best_ask_up,
        best_bid_down=snap.best_bid_down,
        best_ask_down=snap.best_ask_down,
        binance_obi=snap.binance_obi,
        time_remaining=0.0,
    )


def _compute(
    chainlink: float,
    binance: float,
    chainlink_open: float,
    binance_open: float,  # noqa: ARG001
    best_bid_up: float,
    best_ask_up: float,
    best_bid_down: float,
    best_ask_down: float,
    binance_obi: float,
    time_remaining: float,
) -> Signal:
    delta_pct = (chainlink - chainlink_open) / chainlink_open * 100.0

    if abs(delta_pct) < 1e-9:
        direction = Direction.NONE
    elif delta_pct > 0:
        direction = Direction.UP
    else:
        direction = Direction.DOWN

    # H3 fix: >= instead of > for Binance (neutral is not disagreement)
    binance_up = binance >= chainlink_open
    chainlink_up = chainlink >= chainlink_open
    feeds_agree = binance_up == chainlink_up

    # Extended features
    delta_bn_cl_pct = (binance - chainlink) / chainlink if chainlink > 0 else 0.0

    if best_bid_up > 0 and best_ask_up > 0:
        poly_implied_up_prob = (best_bid_up + best_ask_up) / 2.0
    elif best_bid_up > 0:
        poly_implied_up_prob = best_bid_up
    else:
        poly_implied_up_prob = 0.0

    # Match the data collector's formula: Binance price vs Chainlink open.
    # The signal engine's thresholds were trained against this definition.
    bn_direction_from_open_pct = (
        (binance - chainlink_open) / chainlink_open if chainlink_open > 0 else 0.0
    )
    cl_direction_from_open_pct = (chainlink - chainlink_open) / chainlink_open

    poly_spread_up = (
        max(0.0, best_ask_up - best_bid_up) if best_ask_up > 0 and best_bid_up > 0 else 0.0
    )
    poly_spread_down = (
        max(0.0, best_ask_down - best_bid_down) if best_ask_down > 0 and best_bid_down > 0 else 0.0
    )

    return Signal(
        delta_pct=delta_pct,
        direction=direction,
        feeds_agree=feeds_agree,
        time_remaining=time_remaining,
        binance_obi=binance_obi,
        delta_bn_cl_pct=delta_bn_cl_pct,
        poly_implied_up_prob=poly_implied_up_prob,
        bn_direction_from_open_pct=bn_direction_from_open_pct,
        cl_direction_from_open_pct=cl_direction_from_open_pct,
        poly_spread_up=poly_spread_up,
        poly_spread_down=poly_spread_down,
    )
