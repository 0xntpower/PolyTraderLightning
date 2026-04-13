"""TCP state publisher for the PolyLiveVisualizer.

Publishes HMAC-SHA256 signed state snapshots to connected visualizer
clients over a framed binary protocol. Two frame types share the same
socket:

* ``frame_type = 0`` — cold-path JSON snapshot. Sent only when
  string/config fields change (signal rotation, resolution, idle-reason
  flip) or as a periodic heartbeat. Preserves the rich display state
  (signal id, halt reason, etc) that doesn't fit cleanly into a packed
  struct.
* ``frame_type = 1`` — hot-path binary snapshot. Fixed-layout packed
  struct generated from ``shared/binary_snapshot_schema.toml`` containing
  every numeric field that changes tick-to-tick. Sent every strategy
  tick. Eliminates the JSON DOM from the visualizer's render-path
  allocator and cuts the hot-path payload from ~2 KB to ~440 B.

Wire format (per frame)::

    [4-byte big-endian length][1-byte frame_type][32-byte HMAC-SHA256][body]

``length`` covers ``frame_type + HMAC + body``. The HMAC is computed
over ``frame_type || body`` using the ``PLSLAB_HMAC_KEY`` pre-shared
key, so an attacker cannot flip the type byte undetected.

Threading model
---------------
- ``publish_binary()`` is called from the asyncio strategy loop every
  tick and ``publish_cold_if_dirty()`` is called on the same wake but
  only queues a frame when the cold payload actually changed (or a
  heartbeat interval elapsed). Both must never block: the strategy
  tick places orders and we cannot afford a slow TCP client stalling
  it. Both only serialize/frame the snapshot (microseconds) and hand
  the bytes to a writer thread via a small mailbox — the latest frame
  of each type replaces any pending one.
- A dedicated writer thread drains the mailbox and runs ``sendall`` for
  every connected client. Slow clients stall the writer thread only, not
  the event loop.
- A separate accept thread handles new client connections.
- Dead clients are pruned on send failure.
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

from shared import binary_snapshot
from shared.keystore import get_hmac_key

if TYPE_CHECKING:
    from shared.decay_detector import DecayState
    from strategy.kelly import AdjustedWinRateResult, KellyResult
    from strategy.momentum_signal import MomentumSignalConfig
    from strategy.resolution import ResolutionResult
    from strategy.signal import Signal

log = logging.getLogger(__name__)

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_FRAME_TYPE_SIZE = 1
_HMAC_DIGEST_SIZE = 32
_SEND_TIMEOUT = 5.0
_MAX_PAYLOAD = 128 * 1024  # 128 KB — matches the C++ client cap
_ACCEPT_TIMEOUT = 2.0
_WRITER_IDLE_TIMEOUT = 1.0

# Frame type discriminator — first byte after the length prefix. Must
# match the C++ client's ``kFrameTypeJson`` / ``kFrameTypeBinary``.
_FRAME_TYPE_JSON = 0
_FRAME_TYPE_BINARY = 1

# Cold JSON heartbeat — force-send at this interval even when nothing
# has changed, so a late-attaching visualizer client always sees cold
# state refresh within the first ~30 s.
_COLD_HEARTBEAT_S = 30.0


class StatePublisher:
    """TCP server that pushes bot state snapshots to visualizer clients.

    Publishing never blocks the caller. ``publish_binary()`` and
    ``publish_cold_if_dirty()`` both build a frame in microseconds and
    hand it to a writer thread through a two-slot mailbox; when the
    mailbox already holds a frame of that type, the new one replaces it
    (visualizers only care about the latest state anyway).

    Usage::

        pub = StatePublisher(host="0.0.0.0", port=19732)
        pub.start()
        hot = StatePublisher.build_hot_snapshot(..., has_cold_dirty=False)
        pub.publish_binary(hot)
        cold = StatePublisher.build_cold_snapshot(...)
        pub.publish_cold_if_dirty(cold, time.time())
        pub.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 19732) -> None:  # noqa: S104  # nosec B104  # intentional: visualizer connects over Tailscale
        self._host = host
        self._port = port
        self._hmac_key = get_hmac_key()
        self._server_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

        self._clients_lock = threading.Lock()
        self._clients: list[socket.socket] = []

        # Two-slot frame mailbox: newest wins per frame type. Binary frames
        # are the hot path (every tick), JSON frames are cold (rare). A
        # single writer thread drains both slots on wake, binary first so
        # clients see the tick update before the heavier cold payload.
        self._frame_cond = threading.Condition()
        self._pending_binary: bytes | None = None
        self._pending_json: bytes | None = None

        # Monotonic frame sequence number — stamped onto every published
        # snapshot so the visualizer can detect drops, coalesced frames,
        # or a publisher restart (seq jumps backwards to 1). Shared across
        # both frame types so the visualizer's seq tracker sees one
        # continuous stream.
        self._seq = 0

        # Cached last-sent frames for instant hydration of new connections.
        # We replay the most recent binary frame (so the client sees fresh
        # numbers immediately) and the most recent JSON frame (so it has
        # the cold string/config state).
        self._snapshot_lock = threading.Lock()
        self._last_binary_bytes: bytes = b""
        self._last_json_bytes: bytes = b""

        # Cold change detection — JSON bytes of the last cold payload we
        # serialized and the wall-clock time we last sent one. Used by
        # ``publish_cold_if_dirty`` to decide whether the incoming cold
        # dict actually needs to go out.
        self._last_cold_signature: bytes = b""
        self._last_cold_sent_at: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.settimeout(_ACCEPT_TIMEOUT)
        self._server_sock.bind((self._host, self._port))
        self._server_sock.listen(4)

        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="state-publisher-writer"
        )
        self._writer_thread.start()

        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="state-publisher-accept"
        )
        self._accept_thread.start()

        log.info("state publisher listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        self._shutdown.set()

        # Wake the writer thread so it notices shutdown.
        with self._frame_cond:
            self._frame_cond.notify_all()

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

        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
        if self._accept_thread:
            self._accept_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Publishing (called from asyncio thread — never blocks)
    # ------------------------------------------------------------------

    def publish_binary(self, snap: binary_snapshot.HotSnapshot) -> None:
        """Build a hot binary frame from ``snap`` and hand it to the writer.

        Called once per strategy tick. Fast path: pack the struct into a
        preallocated bytearray (zero-alloc beyond the HMAC), hand off.
        ``snap.seq`` is ignored; the publisher stamps its own monotonic
        counter so the visualizer sees one continuous stream across
        binary and JSON frames.
        """
        self._seq += 1
        stamped = snap._replace(seq=self._seq)
        frame = self._build_binary_frame(stamped)
        with self._frame_cond:
            self._pending_binary = frame  # newest wins
            self._frame_cond.notify()

    def publish_cold_if_dirty(self, cold: dict[str, object], now: float) -> bool:
        """Send a cold JSON frame only if its content changed (or heartbeat).

        Returns ``True`` iff a frame was actually queued. The caller should
        pass the returned flag into ``build_hot_snapshot(has_cold_dirty=...)``
        so the companion binary frame advertises that the cold slot is
        being refreshed on the same wake-up.
        """
        # Serialize once with sorted keys so the signature is deterministic
        # regardless of dict insertion order.
        try:
            signature = json.dumps(cold, separators=(",", ":"), sort_keys=True).encode("utf-8")
        except (TypeError, ValueError):
            log.warning("state_publisher: failed to serialize cold snapshot")
            return False

        changed = signature != self._last_cold_signature
        stale = (now - self._last_cold_sent_at) >= _COLD_HEARTBEAT_S
        if not changed and not stale:
            return False

        self._seq += 1
        cold["seq"] = self._seq
        cold["ts"] = round(now, 3)
        frame = self._build_json_frame(cold)
        if frame is None:
            return False

        self._last_cold_signature = signature
        self._last_cold_sent_at = now
        with self._frame_cond:
            self._pending_json = frame  # newest wins
            self._frame_cond.notify()
        return True

    def _frame_with_type(self, frame_type: int, body: bytes) -> bytes:
        """Wrap ``body`` in the on-wire framing for ``frame_type``.

        Layout: ``[4B BE length][1B type][32B HMAC(type || body)][body]``.
        The HMAC covers the type byte so an attacker cannot flip the
        discriminator undetected.
        """
        type_byte = bytes((frame_type,))
        mac = hmac.new(self._hmac_key, type_byte + body, hashlib.sha256).digest()
        framed_body = type_byte + mac + body
        return struct.pack(_HEADER_FMT, len(framed_body)) + framed_body

    def _build_json_frame(self, snapshot: dict[str, object]) -> bytes | None:
        try:
            payload = json.dumps(snapshot, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            log.warning("state_publisher: failed to serialize snapshot")
            return None

        if len(payload) > _MAX_PAYLOAD:
            log.warning(
                "state_publisher: snapshot too large (%d bytes) — dropping",
                len(payload),
            )
            return None

        return self._frame_with_type(_FRAME_TYPE_JSON, payload)

    def _build_binary_frame(self, snap: binary_snapshot.HotSnapshot) -> bytes:
        payload = binary_snapshot.pack(snap)
        return self._frame_with_type(_FRAME_TYPE_BINARY, payload)

    # ------------------------------------------------------------------
    # Writer thread — owns all sendall() calls
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        while not self._shutdown.is_set():
            with self._frame_cond:
                while (
                    self._pending_binary is None
                    and self._pending_json is None
                    and not self._shutdown.is_set()
                ):
                    self._frame_cond.wait(timeout=_WRITER_IDLE_TIMEOUT)
                binary_frame = self._pending_binary
                json_frame = self._pending_json
                self._pending_binary = None
                self._pending_json = None

            if binary_frame is None and json_frame is None:
                continue

            # Binary first so the visualizer gets the latest numbers
            # ahead of the heavier cold payload on the same wake-up.
            frames_to_send: list[bytes] = []
            if binary_frame is not None:
                frames_to_send.append(binary_frame)
            if json_frame is not None:
                frames_to_send.append(json_frame)

            with self._clients_lock:
                clients = list(self._clients)

            dead: list[socket.socket] = []
            for client in clients:
                try:
                    for frame in frames_to_send:
                        client.sendall(frame)
                except (OSError, BrokenPipeError, ConnectionResetError):
                    dead.append(client)

            if dead:
                with self._clients_lock:
                    for client in dead:
                        try:
                            client.close()
                        except OSError:
                            pass
                        try:
                            self._clients.remove(client)
                        except ValueError:
                            pass
                    active = len(self._clients)
                log.info(
                    "state_publisher: pruned %d dead client(s), %d active",
                    len(dead),
                    active,
                )

            with self._snapshot_lock:
                if binary_frame is not None:
                    self._last_binary_bytes = binary_frame
                if json_frame is not None:
                    self._last_json_bytes = json_frame

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

            # Hydrate the new client with the most recent cached frames
            # so it doesn't have to wait for the next tick. Send JSON
            # first so cold string state (signal id, etc.) is populated
            # before the first binary-only numeric update.
            with self._snapshot_lock:
                cached_json = self._last_json_bytes
                cached_binary = self._last_binary_bytes
            hydrate_failed = False
            for cached in (cached_json, cached_binary):
                if not cached:
                    continue
                try:
                    client.sendall(cached)
                except (OSError, BrokenPipeError, ConnectionResetError):
                    hydrate_failed = True
                    break
            if hydrate_failed:
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
    def build_hot_snapshot(
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
        # Lifecycle
        signal_age_windows: int,
        windows_since_last_fire: int,
        fired_this_window: bool,
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
        regime_ready: bool,
        # Performance
        daily_pnl: float,
        total_pnl: float,
        windows_traded: int,
        windows_won: int,
        consecutive_losses: int,
        # Risk
        halted: bool,
        # Last bet resolution
        last_resolution: ResolutionResult | None,
        # Feed health
        last_binance_msg_ts: float,
        last_chainlink_msg_ts: float,
        last_clob_market_msg_ts: float,
        # Cold dirty flag — set by caller after ``publish_cold_if_dirty``
        # determines a JSON frame is being sent on the same wake.
        has_cold_dirty: bool,
    ) -> binary_snapshot.HotSnapshot:
        """Pack every hot numeric field into a ``HotSnapshot``.

        Pure function. The ``seq`` field is zero-filled here; the
        publisher stamps the monotonic value inside ``publish_binary``.
        """
        now = time.time()

        direction_code = 0
        delta_pct = 0.0
        feeds_agree = False
        poly_implied_up_prob = 0.0
        bn_direction_from_open_pct = 0.0
        cl_direction_from_open_pct = 0.0
        poly_spread_up = 0.0
        poly_spread_down = 0.0
        delta_bn_cl_pct = 0.0
        if signal is not None:
            delta_pct = signal.delta_pct
            feeds_agree = signal.feeds_agree
            poly_implied_up_prob = signal.poly_implied_up_prob
            bn_direction_from_open_pct = signal.bn_direction_from_open_pct
            cl_direction_from_open_pct = signal.cl_direction_from_open_pct
            poly_spread_up = signal.poly_spread_up
            poly_spread_down = signal.poly_spread_down
            delta_bn_cl_pct = signal.delta_bn_cl_pct
            dir_value = signal.direction.value
            if dir_value == "up":
                direction_code = 1
            elif dir_value == "down":
                direction_code = 2

        llr = 0.0
        boundary_alive = 0.0
        boundary_dead = 0.0
        rolling_win_rate = 0.0
        p_alive = 0.0
        p_dead = 0.0
        sprt_n_trades = 0
        sprt_n_wins = 0
        if decay_state is not None:
            llr = decay_state.llr
            boundary_alive = decay_state.boundary_alive
            boundary_dead = decay_state.boundary_dead
            rolling_win_rate = decay_state.rolling_win_rate
            p_alive = decay_state.p_alive
            p_dead = decay_state.p_dead
            sprt_n_trades = decay_state.n_trades
            sprt_n_wins = decay_state.n_wins

        raw_kelly = 0.0
        fractional_kelly = 0.0
        bet_size = 0.0
        has_edge = False
        implied_ev = 0.0
        if kelly_result is not None:
            raw_kelly = kelly_result.raw_kelly
            fractional_kelly = kelly_result.fractional_kelly
            bet_size = kelly_result.bet_size
            has_edge = kelly_result.has_edge
            implied_ev = kelly_result.implied_ev

        adjusted_p = 0.0
        vol_discount = 1.0
        chop_discount = 1.0
        outcome_discount = 1.0
        total_discount = 1.0
        feedback_adjustment = 1.0
        if wr_result is not None:
            adjusted_p = wr_result.adjusted_p
            vol_discount = wr_result.vol_discount
            chop_discount = wr_result.chop_discount
            outcome_discount = wr_result.outcome_discount
            total_discount = wr_result.total_discount
            feedback_adjustment = wr_result.feedback_adjustment

        win_rate_pct = (windows_won / windows_traded * 100.0) if windows_traded > 0 else 0.0

        has_last_resolution = last_resolution is not None
        last_bet_won = False
        last_bet_pnl = 0.0
        last_bet_window_ts = 0
        if last_resolution is not None:
            last_bet_won = last_resolution.won
            last_bet_pnl = last_resolution.pnl
            last_bet_window_ts = last_resolution.pending.window_ts

        binance_age_s = round(now - last_binance_msg_ts, 1) if last_binance_msg_ts > 0 else -1.0
        chainlink_age_s = (
            round(now - last_chainlink_msg_ts, 1) if last_chainlink_msg_ts > 0 else -1.0
        )
        clob_age_s = (
            round(now - last_clob_market_msg_ts, 1) if last_clob_market_msg_ts > 0 else -1.0
        )

        return binary_snapshot.HotSnapshot(
            version=binary_snapshot.SCHEMA_VERSION,
            has_cold_dirty=1 if has_cold_dirty else 0,
            direction_code=direction_code,
            halted=1 if halted else 0,
            has_last_resolution=1 if has_last_resolution else 0,
            feeds_agree=1 if feeds_agree else 0,
            fired_this_window=1 if fired_this_window else 0,
            has_edge=1 if has_edge else 0,
            warmup_active=1 if warmup_active else 0,
            regime_ready=1 if regime_ready else 0,
            last_bet_won=1 if last_bet_won else 0,
            pad_hdr_0=0,
            seq=0,  # stamped inside publish_binary
            ts_ms=int(now * 1000.0),
            btc_binance=btc_binance,
            btc_chainlink=btc_chainlink,
            binance_obi=binance_obi,
            window_open_price=window_open_price,
            time_remaining=time_remaining,
            best_bid_up=best_bid_up,
            best_ask_up=best_ask_up,
            best_bid_down=best_bid_down,
            best_ask_down=best_ask_down,
            window_ts=window_ts,
            delta_pct=delta_pct,
            poly_implied_up_prob=poly_implied_up_prob,
            bn_direction_from_open_pct=bn_direction_from_open_pct,
            cl_direction_from_open_pct=cl_direction_from_open_pct,
            poly_spread_up=poly_spread_up,
            poly_spread_down=poly_spread_down,
            delta_bn_cl_pct=delta_bn_cl_pct,
            binance_age_s=binance_age_s,
            chainlink_age_s=chainlink_age_s,
            clob_age_s=clob_age_s,
            signal_age_windows=signal_age_windows,
            windows_since_last_fire=windows_since_last_fire,
            llr=llr,
            boundary_alive=boundary_alive,
            boundary_dead=boundary_dead,
            rolling_win_rate=rolling_win_rate,
            p_alive=p_alive,
            p_dead=p_dead,
            sprt_n_trades=sprt_n_trades,
            sprt_n_wins=sprt_n_wins,
            adjusted_p=adjusted_p,
            raw_kelly=raw_kelly,
            fractional_kelly=fractional_kelly,
            bet_size=bet_size,
            implied_ev=implied_ev,
            vol_discount=vol_discount,
            chop_discount=chop_discount,
            outcome_discount=outcome_discount,
            total_discount=total_discount,
            feedback_adjustment=feedback_adjustment,
            bankroll=bankroll,
            sprt_factor=sprt_factor,
            vol_stddev_pct=vol_stddev_pct,
            chop_avg_flips=chop_avg_flips,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            win_rate_pct=win_rate_pct,
            windows_traded=windows_traded,
            windows_won=windows_won,
            consecutive_losses=consecutive_losses,
            last_bet_pnl=last_bet_pnl,
            last_bet_window_ts=last_bet_window_ts,
        )

    @staticmethod
    def build_cold_snapshot(
        *,
        signal_cfg: MomentumSignalConfig | None,
        idle_reason: str | None,
        decay_verdict: str,
        outcome_summary: str,
        halt_reason: str,
    ) -> dict[str, object]:
        """Build the cold JSON payload — strings and rarely-changing config.

        The shape is deliberately flat-ish so the visualizer can replace
        compound fields wholesale without cross-group merges. All numeric
        fields here are the ones that don't fit cleanly into a fixed
        struct (e.g., ``signal_config``'s parametric rows).

        The live direction string is derived on the visualizer side from
        the ``direction_code`` byte in the hot binary frame, so there is
        no need to send it here.
        """
        snapshot: dict[str, object] = {
            "type": "bot_state_cold",
            "idle_reason": idle_reason,
            "sprt_verdict": decay_verdict,
            "outcome_summary": outcome_summary,
            "halt_reason": halt_reason,
        }

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

        return snapshot
