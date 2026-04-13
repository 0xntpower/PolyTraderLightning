"""Tests for the generated ``binary_snapshot`` wire format.

These tests cover three things:

1. The generated struct layout matches the golden fixtures shipped next to
   the codegen script — so a drift between ``shared/codegen/*.py`` and
   the C++ ``BinarySnapshot.hpp`` (or the Python module) is caught the
   moment a test runs.
2. ``pack``/``unpack`` round-trips a ``HotSnapshot`` losslessly for both
   integer and float fields (with appropriate tolerances).
3. ``unpack`` rejects truncated or over-long buffers with a clear
   ``struct.error`` — this is the Python-side analogue of the C++
   ``ProcessBinaryFrame`` size check and keeps the failure mode
   explicit.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Final

import pytest

from shared import binary_snapshot

_GOLDEN_DIR: Final = Path(__file__).resolve().parent.parent / "shared" / "codegen"
_GOLDEN_BIN: Final = _GOLDEN_DIR / "golden_snapshot.bin"
_GOLDEN_JSON: Final = _GOLDEN_DIR / "golden_snapshot.json"

_FLOAT_TOL: Final = 1e-9


def _load_golden() -> tuple[bytes, dict[str, object]]:
    data = _GOLDEN_BIN.read_bytes()
    payload = json.loads(_GOLDEN_JSON.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return data, payload


def test_golden_binary_size_matches_struct() -> None:
    data, _ = _load_golden()
    assert len(data) == binary_snapshot.SIZE


def test_golden_schema_version_matches_module() -> None:
    _, payload = _load_golden()
    assert payload["schema_version"] == binary_snapshot.SCHEMA_VERSION


def test_golden_struct_format_matches_module() -> None:
    _, payload = _load_golden()
    assert payload["struct_format"] == binary_snapshot.STRUCT_FORMAT


def test_golden_binary_unpacks_to_expected_field_values() -> None:
    data, payload = _load_golden()
    snap = binary_snapshot.unpack(data)
    expected_fields = payload["fields"]
    assert isinstance(expected_fields, list)

    for field in expected_fields:
        assert isinstance(field, dict)
        name = field["name"]
        assert isinstance(name, str)
        expected = field["value"]
        actual = getattr(snap, name)
        type_str = field["type"]
        assert isinstance(type_str, str)
        if type_str in ("f32", "f64"):
            assert isinstance(actual, float)
            assert isinstance(expected, (int, float))
            assert math.isclose(actual, float(expected), rel_tol=_FLOAT_TOL, abs_tol=_FLOAT_TOL), (
                f"field {name!r} float mismatch: {actual} != {expected}"
            )
        else:
            assert actual == expected, f"field {name!r} int mismatch: {actual} != {expected}"


def test_pack_unpack_roundtrip() -> None:
    # Build a non-trivial snapshot with every numeric-ish field populated.
    snap = binary_snapshot.HotSnapshot(
        version=binary_snapshot.SCHEMA_VERSION,
        has_cold_dirty=1,
        direction_code=2,
        halted=0,
        has_last_resolution=1,
        feeds_agree=1,
        fired_this_window=1,
        has_edge=1,
        warmup_active=0,
        regime_ready=1,
        last_bet_won=0,
        pad_hdr_0=0,
        seq=12345,
        ts_ms=1_700_000_000_000,
        btc_binance=67891.23,
        btc_chainlink=67890.12,
        binance_obi=0.5432,
        window_open_price=67850.00,
        time_remaining=217.4,
        best_bid_up=0.5510,
        best_ask_up=0.5525,
        best_bid_down=0.4475,
        best_ask_down=0.4490,
        window_ts=1_700_000_000,
        delta_pct=0.000612,
        poly_implied_up_prob=0.551,
        bn_direction_from_open_pct=0.000714,
        cl_direction_from_open_pct=0.000702,
        poly_spread_up=0.0015,
        poly_spread_down=0.0015,
        delta_bn_cl_pct=0.000012,
        binance_age_s=0.3,
        chainlink_age_s=1.1,
        clob_age_s=0.2,
        signal_age_windows=47,
        windows_since_last_fire=3,
        llr=0.875,
        boundary_alive=2.944,
        boundary_dead=-2.944,
        rolling_win_rate=0.612,
        p_alive=0.72,
        p_dead=0.28,
        sprt_n_trades=21,
        sprt_n_wins=13,
        adjusted_p=0.598,
        raw_kelly=0.1234,
        fractional_kelly=0.0617,
        bet_size=61.75,
        implied_ev=0.042,
        vol_discount=0.94,
        chop_discount=0.87,
        outcome_discount=0.98,
        total_discount=0.80,
        feedback_adjustment=1.00,
        bankroll=1023.45,
        sprt_factor=0.85,
        vol_stddev_pct=0.42,
        chop_avg_flips=3.1,
        daily_pnl=12.34,
        total_pnl=123.45,
        win_rate_pct=61.9,
        windows_traded=21,
        windows_won=13,
        consecutive_losses=2,
        last_bet_pnl=-1.75,
        last_bet_window_ts=1_699_999_700,
    )
    wire = binary_snapshot.pack(snap)
    assert len(wire) == binary_snapshot.SIZE
    roundtripped = binary_snapshot.unpack(wire)

    for name in binary_snapshot.FIELD_NAMES:
        original = getattr(snap, name)
        back = getattr(roundtripped, name)
        if isinstance(original, float):
            assert math.isclose(back, original, rel_tol=_FLOAT_TOL, abs_tol=_FLOAT_TOL), (
                f"{name}: {back} != {original}"
            )
        else:
            assert back == original, f"{name}: {back} != {original}"


def test_pack_into_matches_pack() -> None:
    snap = binary_snapshot.unpack(_GOLDEN_BIN.read_bytes())
    buf = bytearray(binary_snapshot.SIZE + 7)  # extra padding to catch offset bugs
    binary_snapshot.pack_into(buf, 3, snap)
    assert bytes(buf[3 : 3 + binary_snapshot.SIZE]) == binary_snapshot.pack(snap)


def test_unpack_rejects_truncated_buffer() -> None:
    data = _GOLDEN_BIN.read_bytes()
    with pytest.raises(struct.error):
        binary_snapshot.unpack(data[:-1])


def test_unpack_rejects_overlong_buffer() -> None:
    data = _GOLDEN_BIN.read_bytes() + b"\x00"
    with pytest.raises(struct.error):
        binary_snapshot.unpack(data)
