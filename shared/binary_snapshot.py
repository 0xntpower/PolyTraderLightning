"""Binary snapshot wire format — generated from schema.

Do not edit by hand. This module is the single source of truth
for the struct layout used by ``state_publisher.publish_binary``
and the matching C++ ``BinarySnapshot.hpp`` parser.
"""

from __future__ import annotations

import struct
from typing import Final, NamedTuple

# === GENERATED FILE — DO NOT EDIT ===
# Regenerate via shared/codegen/gen_binary_snapshot.py.
# Source schema: shared/binary_snapshot_schema.toml
# Schema SHA256 (first 16 hex): b5d16d4d75f74099

SCHEMA_VERSION: Final[int] = 4
STRUCT_FORMAT: Final[str] = "<BBBBBBBBBBBBQQdddddddddddiddddddddddiiddddddiidddddddddddddddddiiidi"
STRUCT: Final[struct.Struct] = struct.Struct(STRUCT_FORMAT)
SIZE: Final[int] = 424
if STRUCT.size != SIZE:  # pragma: no cover
    raise RuntimeError("binary_snapshot: struct.Struct size mismatch — regenerate")

FIELD_NAMES: Final[tuple[str, ...]] = (
    "version",
    "has_cold_dirty",
    "direction_code",
    "halted",
    "has_last_resolution",
    "feeds_agree",
    "fired_this_window",
    "has_edge",
    "warmup_active",
    "regime_ready",
    "last_bet_won",
    "pad_hdr_0",
    "seq",
    "ts_ms",
    "btc_binance",
    "btc_chainlink",
    "binance_obi_d5",
    "binance_obi_d10",
    "binance_obi_d20",
    "window_open_price",
    "time_remaining",
    "best_bid_up",
    "best_ask_up",
    "best_bid_down",
    "best_ask_down",
    "window_ts",
    "delta_pct",
    "poly_implied_up_prob",
    "bn_direction_from_open_pct",
    "cl_direction_from_open_pct",
    "poly_spread_up",
    "poly_spread_down",
    "delta_bn_cl_pct",
    "binance_age_s",
    "chainlink_age_s",
    "clob_age_s",
    "signal_age_windows",
    "windows_since_last_fire",
    "llr",
    "boundary_alive",
    "boundary_dead",
    "rolling_win_rate",
    "p_alive",
    "p_dead",
    "sprt_n_trades",
    "sprt_n_wins",
    "adjusted_p",
    "raw_kelly",
    "fractional_kelly",
    "bet_size",
    "implied_ev",
    "vol_discount",
    "chop_discount",
    "outcome_discount",
    "total_discount",
    "feedback_adjustment",
    "bankroll",
    "sprt_factor",
    "vol_stddev_pct",
    "chop_avg_flips",
    "daily_pnl",
    "total_pnl",
    "win_rate_pct",
    "windows_traded",
    "windows_won",
    "consecutive_losses",
    "last_bet_pnl",
    "last_bet_window_ts",
)


class HotSnapshot(NamedTuple):
    """Strongly-typed hot-path field tuple.

    Construction order matches the schema — the caller passes
    fields by keyword so reordering the schema breaks loudly.
    """

    version: int
    has_cold_dirty: int
    direction_code: int
    halted: int
    has_last_resolution: int
    feeds_agree: int
    fired_this_window: int
    has_edge: int
    warmup_active: int
    regime_ready: int
    last_bet_won: int
    pad_hdr_0: int
    seq: int
    ts_ms: int
    btc_binance: float
    btc_chainlink: float
    binance_obi_d5: float
    binance_obi_d10: float
    binance_obi_d20: float
    window_open_price: float
    time_remaining: float
    best_bid_up: float
    best_ask_up: float
    best_bid_down: float
    best_ask_down: float
    window_ts: int
    delta_pct: float
    poly_implied_up_prob: float
    bn_direction_from_open_pct: float
    cl_direction_from_open_pct: float
    poly_spread_up: float
    poly_spread_down: float
    delta_bn_cl_pct: float
    binance_age_s: float
    chainlink_age_s: float
    clob_age_s: float
    signal_age_windows: int
    windows_since_last_fire: int
    llr: float
    boundary_alive: float
    boundary_dead: float
    rolling_win_rate: float
    p_alive: float
    p_dead: float
    sprt_n_trades: int
    sprt_n_wins: int
    adjusted_p: float
    raw_kelly: float
    fractional_kelly: float
    bet_size: float
    implied_ev: float
    vol_discount: float
    chop_discount: float
    outcome_discount: float
    total_discount: float
    feedback_adjustment: float
    bankroll: float
    sprt_factor: float
    vol_stddev_pct: float
    chop_avg_flips: float
    daily_pnl: float
    total_pnl: float
    win_rate_pct: float
    windows_traded: int
    windows_won: int
    consecutive_losses: int
    last_bet_pnl: float
    last_bet_window_ts: int


def pack(snap: HotSnapshot) -> bytes:
    """Serialize ``snap`` to the wire format. One allocation."""
    return STRUCT.pack(*snap)


def pack_into(buf: bytearray, offset: int, snap: HotSnapshot) -> None:
    """Serialize ``snap`` directly into ``buf[offset:]`` — zero alloc."""
    STRUCT.pack_into(buf, offset, *snap)


def unpack(data: bytes | memoryview | bytearray) -> HotSnapshot:
    """Parse wire bytes back into a ``HotSnapshot``. Used by tests."""
    return HotSnapshot(*STRUCT.unpack(data))
