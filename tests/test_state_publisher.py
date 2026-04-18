"""Tests for the StatePublisher frame format and cold-dirty logic.

These tests avoid touching real sockets — they drive the publisher's
pure builders and its in-process mailbox/frame builders directly, then
assert the wire-level framing matches what the C++ client expects.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
from typing import Final

import pytest

from shared import binary_snapshot
from shared import state_publisher as state_publisher_module
from shared.state_publisher import StatePublisher

_TEST_HMAC_KEY: Final = b"\xaa" * 32


@pytest.fixture(autouse=True)
def _hermetic_hmac_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The publisher calls ``get_hmac_key`` eagerly in __init__. Patch the
    # imported reference in the state_publisher module so every test sees
    # a deterministic key, regardless of whatever is actually sitting in
    # the developer's Credential Manager.
    monkeypatch.setattr(state_publisher_module, "get_hmac_key", lambda: _TEST_HMAC_KEY)


def _unwrap_frame(frame: bytes, hmac_key: bytes) -> tuple[int, bytes]:
    """Reverse the publisher's framing and return (frame_type, payload)."""
    assert len(frame) >= 4 + 1 + 32
    (body_len,) = struct.unpack("!I", frame[:4])
    assert len(frame) == 4 + body_len
    frame_type = frame[4]
    mac = frame[5 : 5 + 32]
    payload = frame[5 + 32 :]
    expected = hmac.new(hmac_key, bytes((frame_type,)) + payload, hashlib.sha256).digest()
    assert hmac.compare_digest(mac, expected)
    return frame_type, payload


def _make_publisher() -> StatePublisher:
    return StatePublisher(host="127.0.0.1", port=0)


def test_build_hot_snapshot_has_schema_version() -> None:
    snap = StatePublisher.build_hot_snapshot(
        btc_binance=67890.0,
        btc_chainlink=67891.0,
        binance_obi_d5=0.1,
        binance_obi_d10=0.08,
        binance_obi_d20=0.06,
        window_open_price=67850.0,
        window_ts=1700000000,
        time_remaining=240.0,
        best_bid_up=0.55,
        best_ask_up=0.56,
        best_bid_down=0.44,
        best_ask_down=0.45,
        signal=None,
        signal_age_windows=2,
        windows_since_last_fire=5,
        fired_this_window=False,
        decay_state=None,
        kelly_result=None,
        wr_result=None,
        bankroll=1000.0,
        sprt_factor=1.0,
        warmup_active=False,
        vol_stddev_pct=0.4,
        chop_avg_flips=2.0,
        regime_ready=False,
        daily_pnl=0.0,
        total_pnl=0.0,
        windows_traded=0,
        windows_won=0,
        consecutive_losses=0,
        halted=False,
        last_resolution=None,
        last_binance_msg_ts=0.0,
        last_chainlink_msg_ts=0.0,
        last_clob_market_msg_ts=0.0,
        has_cold_dirty=False,
    )
    assert snap.version == binary_snapshot.SCHEMA_VERSION
    assert snap.direction_code == 0
    assert snap.halted == 0
    assert snap.feeds_agree == 0
    assert snap.seq == 0  # unstamped — publish_binary stamps the real value


def test_build_cold_snapshot_omits_direction_and_hot_numerics() -> None:
    cold = StatePublisher.build_cold_snapshot(
        signal_cfg=None,
        idle_reason="warmup",
        decay_verdict="INCONCLUSIVE",
        outcome_summary="W/L/W",
        halt_reason="",
    )
    # Direction is derived on the visualizer side from the binary
    # direction_code byte, so the cold dict must not leak it.
    assert "direction" not in cold
    assert cold["idle_reason"] == "warmup"
    assert cold["sprt_verdict"] == "INCONCLUSIVE"
    assert cold["outcome_summary"] == "W/L/W"
    assert cold["halt_reason"] == ""


def test_publish_cold_if_dirty_skips_unchanged_payload() -> None:
    pub = _make_publisher()
    cold = {"type": "bot_state_cold", "idle_reason": None, "sprt_verdict": "ALIVE"}

    now = 1_700_000_000.0
    assert pub.publish_cold_if_dirty(dict(cold), now) is True
    # Second identical publish within the heartbeat window is a no-op.
    assert pub.publish_cold_if_dirty(dict(cold), now + 1.0) is False
    # A different value triggers a send.
    changed = dict(cold)
    changed["sprt_verdict"] = "DEAD"
    assert pub.publish_cold_if_dirty(changed, now + 2.0) is True


def test_publish_cold_if_dirty_heartbeat_fires_after_30s() -> None:
    pub = _make_publisher()
    cold = {"type": "bot_state_cold", "idle_reason": None}
    now = 1_700_000_000.0
    assert pub.publish_cold_if_dirty(dict(cold), now) is True
    # Exactly at 30s the heartbeat should fire even for an unchanged payload.
    assert pub.publish_cold_if_dirty(dict(cold), now + 30.0) is True
    # And immediately after it goes quiet again.
    assert pub.publish_cold_if_dirty(dict(cold), now + 30.1) is False


def test_binary_frame_has_expected_layout() -> None:
    pub = _make_publisher()

    snap = binary_snapshot.HotSnapshot(
        version=binary_snapshot.SCHEMA_VERSION,
        has_cold_dirty=0,
        direction_code=1,
        halted=0,
        has_last_resolution=0,
        feeds_agree=1,
        fired_this_window=0,
        has_edge=1,
        warmup_active=0,
        regime_ready=1,
        last_bet_won=0,
        pad_hdr_0=0,
        seq=0,
        ts_ms=1_700_000_000_000,
        btc_binance=67890.0,
        btc_chainlink=67890.0,
        binance_obi_d5=0.1,
        binance_obi_d10=0.08,
        binance_obi_d20=0.06,
        window_open_price=67800.0,
        time_remaining=240.0,
        best_bid_up=0.55,
        best_ask_up=0.56,
        best_bid_down=0.44,
        best_ask_down=0.45,
        window_ts=1_700_000_000,
        delta_pct=0.0,
        poly_implied_up_prob=0.55,
        bn_direction_from_open_pct=0.0,
        cl_direction_from_open_pct=0.0,
        poly_spread_up=0.01,
        poly_spread_down=0.01,
        delta_bn_cl_pct=0.0,
        binance_age_s=0.2,
        chainlink_age_s=0.3,
        clob_age_s=0.5,
        signal_age_windows=0,
        windows_since_last_fire=0,
        llr=0.0,
        boundary_alive=2.944,
        boundary_dead=-2.944,
        rolling_win_rate=0.5,
        p_alive=0.5,
        p_dead=0.5,
        sprt_n_trades=0,
        sprt_n_wins=0,
        adjusted_p=0.5,
        raw_kelly=0.0,
        fractional_kelly=0.0,
        bet_size=0.0,
        implied_ev=0.0,
        vol_discount=1.0,
        chop_discount=1.0,
        outcome_discount=1.0,
        total_discount=1.0,
        feedback_adjustment=1.0,
        bankroll=1000.0,
        sprt_factor=1.0,
        vol_stddev_pct=0.4,
        chop_avg_flips=2.0,
        daily_pnl=0.0,
        total_pnl=0.0,
        win_rate_pct=0.0,
        windows_traded=0,
        windows_won=0,
        consecutive_losses=0,
        last_bet_pnl=0.0,
        last_bet_window_ts=0,
    )
    pub.publish_binary(snap)
    # Drain the internal mailbox without using sockets.
    queued = pub._pending_binary  # test-only introspection
    assert queued is not None

    key = _TEST_HMAC_KEY
    frame_type, payload = _unwrap_frame(queued, key)
    assert frame_type == 1  # binary
    assert len(payload) == binary_snapshot.SIZE
    parsed = binary_snapshot.unpack(payload)
    assert parsed.version == binary_snapshot.SCHEMA_VERSION
    assert parsed.direction_code == 1
    assert parsed.seq == 1  # publisher stamped its monotonic counter


def test_cold_frame_has_expected_layout() -> None:
    pub = _make_publisher()
    cold = {"type": "bot_state_cold", "idle_reason": "warmup", "sprt_verdict": "ALIVE"}
    assert pub.publish_cold_if_dirty(dict(cold), 1_700_000_000.0) is True
    queued = pub._pending_json  # test-only introspection
    assert queued is not None

    key = _TEST_HMAC_KEY
    frame_type, payload = _unwrap_frame(queued, key)
    assert frame_type == 0  # JSON
    parsed = json.loads(payload)
    assert parsed["idle_reason"] == "warmup"
    assert parsed["sprt_verdict"] == "ALIVE"
    assert parsed["seq"] == 1
    assert "ts" in parsed


def test_seq_is_monotonic_across_binary_and_cold() -> None:
    pub = _make_publisher()
    cold = {"type": "bot_state_cold", "idle_reason": None}
    pub.publish_cold_if_dirty(dict(cold), 1_700_000_000.0)  # seq = 1

    snap = StatePublisher.build_hot_snapshot(
        btc_binance=0.0,
        btc_chainlink=0.0,
        binance_obi_d5=0.0,
        binance_obi_d10=0.0,
        binance_obi_d20=0.0,
        window_open_price=0.0,
        window_ts=0,
        time_remaining=0.0,
        best_bid_up=0.0,
        best_ask_up=0.0,
        best_bid_down=0.0,
        best_ask_down=0.0,
        signal=None,
        signal_age_windows=0,
        windows_since_last_fire=0,
        fired_this_window=False,
        decay_state=None,
        kelly_result=None,
        wr_result=None,
        bankroll=0.0,
        sprt_factor=1.0,
        warmup_active=False,
        vol_stddev_pct=0.0,
        chop_avg_flips=0.0,
        regime_ready=False,
        daily_pnl=0.0,
        total_pnl=0.0,
        windows_traded=0,
        windows_won=0,
        consecutive_losses=0,
        halted=False,
        last_resolution=None,
        last_binance_msg_ts=0.0,
        last_chainlink_msg_ts=0.0,
        last_clob_market_msg_ts=0.0,
        has_cold_dirty=True,
    )
    pub.publish_binary(snap)  # seq = 2

    _, bin_payload = _unwrap_frame(pub._pending_binary or b"", _TEST_HMAC_KEY)
    assert binary_snapshot.unpack(bin_payload).seq == 2
