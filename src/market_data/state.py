"""Shared in-memory state written by WebSocket consumers, read by strategy."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

# Max ticks to retain — ~10 minutes at ~1 tick/sec from RTDS
_TICK_BUFFER_MAXLEN = 600


class ChainlinkTick(NamedTuple):
    """A single Chainlink oracle tick with the oracle's own timestamp."""

    oracle_ts_ms: int  # payload.timestamp from RTDS (oracle epoch ms)
    price: float  # payload.value (USD)


@dataclass
class LiveFill:  # mutable: confirmed field updated on CLOB status transitions (MATCHED→CONFIRMED)
    """Tracks a confirmed fill from the CLOB user WebSocket."""

    order_id: str
    token_id: str
    side: str  # "BUY" or "SELL"
    price: float  # fill price
    size: float  # accumulated fill size (shares)
    size_usd: float  # accumulated USDC spent (price * size per partial fill)
    fill_time: float  # timestamp of first fill event
    confirmed: bool = False  # True once status reaches CONFIRMED
    # Whether the originating order was maker (post-only GTC) vs taker (FOK).
    # Read at resolution time to gate the entry-side taker fee in _resolve.
    is_maker: bool = False


@dataclass(frozen=True, slots=True)
class WindowSnapshot:
    """Frozen price state captured near window close for accurate finalization."""

    window_ts: int = 0
    chainlink_price: float = 0.0
    binance_price: float = 0.0
    open_price: float = 0.0
    binance_open_price: float = 0.0
    up_token_id: str = ""
    down_token_id: str = ""
    best_bid_up: float = 0.0
    best_ask_up: float = 0.0
    best_bid_down: float = 0.0
    best_ask_down: float = 0.0
    # Centered OBI at three depths, computed from btcusdt@depth20@100ms.
    binance_obi_d5: float = 0.0
    binance_obi_d10: float = 0.0
    binance_obi_d20: float = 0.0


@dataclass
class MarketState:
    btc_binance: float = 0.0
    btc_binance_ts: float = 0.0
    btc_chainlink: float = 0.0
    btc_chainlink_ts: float = 0.0

    # Centered Binance order book imbalance: (bid - ask) / (bid + ask),
    # computed at three depths from btcusdt@depth20@100ms. D5 is collected
    # for research parity with the data collector; engine signals gate on
    # D10 or D20 (see MomentumSignalConfig.obi_depth).
    binance_obi_d5: float = 0.0
    binance_obi_d10: float = 0.0
    binance_obi_d20: float = 0.0
    binance_obi_ts: float = 0.0

    window_open_price: float = 0.0  # Chainlink open price for current window
    binance_window_open_price: float = 0.0  # Binance open price for current window
    window_ts: int = 0
    time_remaining: float = 300.0

    best_bid_up: float = 0.0
    best_ask_up: float = 0.0
    best_bid_down: float = 0.0
    best_ask_down: float = 0.0

    position_up: float = 0.0
    position_down: float = 0.0

    up_token_id: str = ""
    down_token_id: str = ""

    # Tracks whether we've captured the open prices for current window
    open_price_captured: bool = False
    binance_open_price_captured: bool = False

    # Open price tier: -1=not captured, 0=Gamma priceToBeat, 1=RTDS boundary, 2=RTDS latest
    open_price_tier: int = -1
    oracle_open_confirmed: bool = False  # True once Gamma finalPrice confirms/upgrades
    chainlink_tick_buffer: deque[ChainlinkTick] = field(
        default_factory=lambda: deque(maxlen=_TICK_BUFFER_MAXLEN),
    )

    # Set True once fresh book data arrives after a window reset
    has_fresh_book_data: bool = False

    # Last-message timestamps for feed staleness detection (written by WS handlers)
    last_binance_msg_ts: float = 0.0
    last_chainlink_msg_ts: float = 0.0
    last_clob_market_msg_ts: float = 0.0

    # Active order IDs for this window
    active_order_ids: list[str] = field(default_factory=list)

    # Subset of active_order_ids that were placed as maker (post-only GTC).
    # The CLOB user WS stamps LiveFill.is_maker by looking up this set when
    # it matches a BUY fill, so resolution can gate the entry taker fee.
    maker_order_ids: set[str] = field(default_factory=set)

    # Confirmed fills from the CLOB user WebSocket, keyed by order_id
    live_fills: dict[str, LiveFill] = field(default_factory=dict)

    # Snapshot captured near window close for safe finalization
    end_snapshot: WindowSnapshot | None = None

    def select_boundary_tick(self, boundary_ts: int) -> ChainlinkTick | None:
        """Find the latest tick whose oracle timestamp is <= boundary epoch.

        This replicates Polymarket's price selection: the oracle price at
        or just before the window boundary is the official open price.

        Args:
            boundary_ts: window start epoch in seconds (window_ts).
        """
        boundary_ms = boundary_ts * 1000
        best: ChainlinkTick | None = None
        for tick in self.chainlink_tick_buffer:
            if tick.oracle_ts_ms <= boundary_ms and (
                best is None or tick.oracle_ts_ms > best.oracle_ts_ms
            ):
                best = tick
        return best

    def is_feed_stale(
        self,
        binance_threshold: float = 15.0,
        chainlink_threshold: float = 30.0,
        clob_book_threshold: float = 60.0,
    ) -> str | None:
        """Return the name of the first stale feed, or None if all fresh.

        Only checks feeds that have received at least one message
        (last_*_msg_ts > 0) so we don't flag during initial startup.
        """
        now = time.time()
        bn_stale = (
            self.last_binance_msg_ts > 0 and (now - self.last_binance_msg_ts) > binance_threshold
        )
        if bn_stale:
            return "binance"
        cl_stale = (
            self.last_chainlink_msg_ts > 0
            and (now - self.last_chainlink_msg_ts) > chainlink_threshold
        )
        if cl_stale:
            return "chainlink"
        clob_stale = (
            self.last_clob_market_msg_ts > 0
            and (now - self.last_clob_market_msg_ts) > clob_book_threshold
        )
        if clob_stale:
            return "clob_book"
        return None

    def snapshot(self) -> WindowSnapshot:
        """Capture current prices for end-of-window finalization."""
        snap = WindowSnapshot(
            window_ts=self.window_ts,
            chainlink_price=self.btc_chainlink,
            binance_price=self.btc_binance,
            open_price=self.window_open_price,
            binance_open_price=self.binance_window_open_price,
            up_token_id=self.up_token_id,
            down_token_id=self.down_token_id,
            best_bid_up=self.best_bid_up,
            best_ask_up=self.best_ask_up,
            best_bid_down=self.best_bid_down,
            best_ask_down=self.best_ask_down,
            binance_obi_d5=self.binance_obi_d5,
            binance_obi_d10=self.binance_obi_d10,
            binance_obi_d20=self.binance_obi_d20,
        )
        self.end_snapshot = snap
        return snap

    def reset_for_window(self, window_ts: int) -> None:
        """Clear per-window state when a new window starts."""
        self.window_ts = window_ts
        self.window_open_price = 0.0
        self.binance_window_open_price = 0.0
        self.open_price_captured = False
        self.binance_open_price_captured = False
        self.open_price_tier = -1
        self.oracle_open_confirmed = False
        # NOTE: do NOT clear chainlink_tick_buffer here — boundary lookup
        # needs ticks from the previous few seconds
        self.time_remaining = 300.0
        self.position_up = 0.0
        self.position_down = 0.0
        self.active_order_ids.clear()
        self.maker_order_ids.clear()
        self.live_fills.clear()
        self.best_bid_up = 0.0
        self.best_ask_up = 0.0
        self.best_bid_down = 0.0
        self.best_ask_down = 0.0
        self.has_fresh_book_data = False
        self.end_snapshot = None
