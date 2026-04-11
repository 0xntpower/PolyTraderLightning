"""TCP state publisher for the PolyLiveVisualizer.

Runs a background daemon thread that pushes HMAC-SHA256 signed JSON state
snapshots to connected visualizer clients.

Wire protocol:
  - 4-byte big-endian length header + UTF-8 JSON envelope.
  - Envelope: ``{"payload": "<canonical json>", "signature": "<hex>"}``.
  - The signature covers the exact ``payload`` bytes using HMAC-SHA256
    with the ``PLSLAB_HMAC_KEY`` pre-shared key.

The publisher accepts multiple clients.  Each client receives the latest
snapshot immediately on connect, then every subsequent snapshot as it is
published.  Slow/dead clients are dropped silently.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import socket
import struct
import threading
import time
from typing import TYPE_CHECKING

from shared.keystore import get_hmac_key

if TYPE_CHECKING:
    from shared.decay_detector import DecayState
    from strategy.kelly import AdjustedWinRateResult, KellyResult
    from strategy.momentum_signal import MomentumSignalConfig
    from strategy.signal import Signal

log = logging.getLogger(__name__)

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_SEND_TIMEOUT = 5.0
_MAX_PAYLOAD = 128 * 1024  # 128 KB safety cap


class StatePublisher:
    """TCP server that pushes bot state snapshots to visualizer clients.

    Usage::

        pub = StatePublisher(host="0.0.0.0", port=19732)
        pub.start()
        # ... in strategy loop every tick:
        pub.publish(snapshot_dict)
        # ... on shutdown:
        pub.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 19732) -> None:  # noqa: S104  # nosec B104  # intentional: visualizer connects over Tailscale
        self._host = host
        self._port = port
        self._hmac_key = get_hmac_key()
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()

        self._clients_lock = threading.Lock()
        self._clients: list[socket.socket] = []

        self._snapshot_lock = threading.Lock()
        self._snapshot_bytes: bytes = b""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.settimeout(2.0)
        self._server_sock.bind((self._host, self._port))
        self._server_sock.listen(4)
        self._thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="state-publisher"
        )
        self._thread.start()
        log.info("state publisher listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        self._shutdown.set()
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        with self._clients_lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()
        if self._thread:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Publishing (called from asyncio thread — must be fast)
    # ------------------------------------------------------------------

    def publish(self, snapshot: dict[str, object]) -> None:
        """Serialize and push a snapshot to all connected clients.

        This is called from the main asyncio event loop.  The JSON
        serialization + framing happens here (microseconds), then each
        client gets a non-blocking send.  Dead clients are pruned.
        """
        try:
            payload_str = json.dumps(snapshot, separators=(",", ":"))
        except (TypeError, ValueError):
            log.warning("state_publisher: failed to serialize snapshot")
            return

        payload_bytes = payload_str.encode("utf-8")
        signature = hmac.new(self._hmac_key, payload_bytes, hashlib.sha256).hexdigest()
        envelope = json.dumps(
            {"payload": payload_str, "signature": signature},
            separators=(",", ":"),
        ).encode("utf-8")

        if len(envelope) > _MAX_PAYLOAD:
            log.warning(
                "state_publisher: snapshot too large (%d bytes) — dropping",
                len(envelope),
            )
            return

        frame = struct.pack(_HEADER_FMT, len(envelope)) + envelope

        # Cache for new clients that connect between publishes
        with self._snapshot_lock:
            self._snapshot_bytes = frame

        with self._clients_lock:
            dead: list[socket.socket] = []
            for client in self._clients:
                try:
                    client.sendall(frame)
                except (OSError, BrokenPipeError, ConnectionResetError):
                    dead.append(client)
            for client in dead:
                try:
                    client.close()
                except OSError:
                    pass
                self._clients.remove(client)
            if dead:
                log.info(
                    "state_publisher: pruned %d dead client(s), %d active",
                    len(dead),
                    len(self._clients),
                )

    # ------------------------------------------------------------------
    # Accept loop (daemon thread)
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                assert self._server_sock is not None  # noqa: S101  # nosec B101  # set in start()
                client, addr = self._server_sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            client.settimeout(_SEND_TIMEOUT)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Send the most recent snapshot immediately so the visualizer
            # doesn't have to wait for the next tick.
            with self._snapshot_lock:
                cached = self._snapshot_bytes
            if cached:
                try:
                    client.sendall(cached)
                except (OSError, BrokenPipeError, ConnectionResetError):
                    try:
                        client.close()
                    except OSError:
                        pass
                    continue

            with self._clients_lock:
                self._clients.append(client)

            log.info("state_publisher: client connected from %s:%d", addr[0], addr[1])

    # ------------------------------------------------------------------
    # Snapshot builders (pure functions — no side effects)
    # ------------------------------------------------------------------

    @staticmethod
    def build_snapshot(
        *,
        # Market data
        btc_binance: float,
        btc_chainlink: float,
        binance_obi: float,
        window_open_price: float,
        window_ts: int,
        time_remaining: float,
        best_bid_up: float,
        best_ask_up: float,
        best_bid_down: float,
        best_ask_down: float,
        # Computed signal
        signal: Signal | None,
        # Active signal config
        signal_cfg: MomentumSignalConfig | None,
        # Lifecycle
        signal_age_windows: int,
        windows_since_last_fire: int,
        fired_this_window: bool,
        idle_reason: str | None,
        # SPRT
        decay_state: DecayState | None,
        # Kelly
        kelly_result: KellyResult | None,
        wr_result: AdjustedWinRateResult | None,
        bankroll: float,
        sprt_factor: float,
        warmup_active: bool,
        # Regime
        vol_stddev_pct: float,
        chop_avg_flips: float,
        outcome_summary: str,
        regime_ready: bool,
        # Performance
        daily_pnl: float,
        total_pnl: float,
        windows_traded: int,
        windows_won: int,
        consecutive_losses: int,
        # Risk
        halted: bool,
        halt_reason: str,
        # Feed health
        last_binance_msg_ts: float,
        last_chainlink_msg_ts: float,
        last_clob_market_msg_ts: float,
    ) -> dict[str, object]:
        """Build a snapshot dict from all bot components.

        Pure function — reads values passed in, no side effects.
        Returns a JSON-serializable dict.
        """
        now = time.time()

        snapshot: dict[str, object] = {
            "type": "bot_state",
            "ts": round(now, 3),
        }

        # -- Market --
        snapshot["market"] = {
            "btc_binance": round(btc_binance, 2),
            "btc_chainlink": round(btc_chainlink, 2),
            "binance_obi": round(binance_obi, 4),
            "window_open_price": round(window_open_price, 2),
            "window_ts": window_ts,
            "time_remaining": round(time_remaining, 1),
            "best_bid_up": round(best_bid_up, 4),
            "best_ask_up": round(best_ask_up, 4),
            "best_bid_down": round(best_bid_down, 4),
            "best_ask_down": round(best_ask_down, 4),
        }

        # -- Computed signal --
        if signal is not None:
            snapshot["signal_live"] = {
                "delta_pct": round(signal.delta_pct, 6),
                "direction": signal.direction.value,
                "feeds_agree": signal.feeds_agree,
                "poly_implied_up_prob": round(signal.poly_implied_up_prob, 4),
                "bn_direction_from_open_pct": round(signal.bn_direction_from_open_pct, 6),
                "cl_direction_from_open_pct": round(signal.cl_direction_from_open_pct, 6),
                "poly_spread_up": round(signal.poly_spread_up, 4),
                "poly_spread_down": round(signal.poly_spread_down, 4),
                "delta_bn_cl_pct": round(signal.delta_bn_cl_pct, 6),
            }

        # -- Active signal config --
        if signal_cfg is not None:
            snapshot["signal_config"] = {
                "signal_id": signal_cfg.signal_id,
                "rank": signal_cfg.rank,
                "side": signal_cfg.side.value,
                "observe_from_s": signal_cfg.observe_from_s,
                "observe_to_s": signal_cfg.observe_to_s,
                "min_delta_pct": round(signal_cfg.min_delta_pct, 4),
                "max_variance_pct": round(signal_cfg.max_variance_pct, 4),
                "oos_win_rate_pct": round(signal_cfg.oos_win_rate_pct, 1),
                "oos_matches": signal_cfg.oos_matches,
                "smart_score": round(signal_cfg.smart_score, 1),
                "cons_win_rate_pct": round(signal_cfg.conservative_win_rate_pct or 0.0, 1),
                "ev_per_trade": round(signal_cfg.ev_per_trade or 0.0, 4),
                "avg_entry_price": round(signal_cfg.avg_entry_price or 0.0, 4),
                "wf_folds_appeared": signal_cfg.wf_folds_appeared,
                "wf_total_test_folds": signal_cfg.wf_total_test_folds,
            }

        # -- Lifecycle --
        snapshot["lifecycle"] = {
            "signal_age_windows": signal_age_windows,
            "windows_since_last_fire": windows_since_last_fire,
            "fired_this_window": fired_this_window,
            "idle_reason": idle_reason,
        }

        # -- SPRT --
        if decay_state is not None:
            snapshot["sprt"] = {
                "verdict": decay_state.verdict,
                "llr": round(decay_state.llr, 3),
                "boundary_alive": round(decay_state.boundary_alive, 3),
                "boundary_dead": round(decay_state.boundary_dead, 3),
                "n_trades": decay_state.n_trades,
                "n_wins": decay_state.n_wins,
                "rolling_win_rate": round(decay_state.rolling_win_rate, 3),
                "p_alive": round(decay_state.p_alive, 4),
                "p_dead": round(decay_state.p_dead, 4),
            }

        # -- Kelly --
        kelly: dict[str, object] = {
            "bankroll": round(bankroll, 2),
            "sprt_factor": round(sprt_factor, 3),
            "warmup_active": warmup_active,
        }
        if kelly_result is not None:
            kelly["raw_kelly"] = round(kelly_result.raw_kelly, 4)
            kelly["fractional_kelly"] = round(kelly_result.fractional_kelly, 4)
            kelly["bet_size"] = round(kelly_result.bet_size, 2)
            kelly["has_edge"] = kelly_result.has_edge
            kelly["implied_ev"] = round(kelly_result.implied_ev, 4)
        if wr_result is not None:
            kelly["adjusted_p"] = round(wr_result.adjusted_p, 4)
            kelly["vol_discount"] = round(wr_result.vol_discount, 4)
            kelly["chop_discount"] = round(wr_result.chop_discount, 4)
            kelly["outcome_discount"] = round(wr_result.outcome_discount, 4)
            kelly["total_discount"] = round(wr_result.total_discount, 4)
            kelly["feedback_adjustment"] = round(wr_result.feedback_adjustment, 4)
            kelly["regime_ready"] = wr_result.regime_ready
        snapshot["kelly"] = kelly

        # -- Regime --
        snapshot["regime"] = {
            "vol_stddev_pct": round(vol_stddev_pct, 4),
            "chop_avg_flips": round(chop_avg_flips, 2),
            "outcome_summary": outcome_summary,
            "regime_ready": regime_ready,
        }

        # -- Performance --
        win_rate = round(windows_won / windows_traded * 100, 1) if windows_traded > 0 else 0.0
        snapshot["performance"] = {
            "daily_pnl": round(daily_pnl, 4),
            "total_pnl": round(total_pnl, 4),
            "windows_traded": windows_traded,
            "windows_won": windows_won,
            "win_rate_pct": win_rate,
            "consecutive_losses": consecutive_losses,
        }

        # -- Risk --
        snapshot["risk"] = {
            "halted": halted,
            "halt_reason": halt_reason,
        }

        # -- Feed health --
        snapshot["feeds"] = {
            "binance_age_s": round(now - last_binance_msg_ts, 1)
            if last_binance_msg_ts > 0
            else -1.0,
            "chainlink_age_s": round(now - last_chainlink_msg_ts, 1)
            if last_chainlink_msg_ts > 0
            else -1.0,
            "clob_age_s": round(now - last_clob_market_msg_ts, 1)
            if last_clob_market_msg_ts > 0
            else -1.0,
        }

        return snapshot
