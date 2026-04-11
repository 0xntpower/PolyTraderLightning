"""TCP-based IPC for signal delivery between orchestrator and trading bot.

Protocol:
  - TCP on a configurable port (default 19731).
  - Messages are JSON objects prefixed with a 4-byte big-endian length header.
  - Every message is self-authenticating via HMAC-SHA256.
  - The signed content is a canonical JSON string sent as the ``payload`` field
    so both sides verify against the exact same bytes (no re-serialisation).
  - Envelope: {"payload": "<canonical json>", "nonce": "...", "signature": "..."}.
  - Server replies with a signed ack using the same envelope format, bound to
    the request nonce so the client can verify the bot actually received it.
  - PSK read from PLSLAB_HMAC_KEY env var (minimum 32 bytes / 64 hex chars).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import socket
import struct
import threading
from collections import deque
from datetime import UTC, datetime
from typing import (
    TYPE_CHECKING,
    Any,  # IPC JSON messages are dynamically typed at wire boundary
)

from shared.keystore import get_hmac_key

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_RECV_TIMEOUT = 10.0
_SEND_TIMEOUT = 10.0

# Reject messages with timestamps older than this (seconds).
_TIMESTAMP_MAX_AGE = 60.0
# Maximum number of recent nonces to track for replay protection.
_NONCE_HISTORY_SIZE = 256


def _sign(key: bytes, nonce: str, payload_bytes: bytes) -> str:
    """Compute HMAC-SHA256(key, nonce || payload) and return hex digest."""
    msg = nonce.encode("utf-8") + payload_bytes
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def _send_message(sock: socket.socket, data: dict[str, Any]) -> None:
    payload = json.dumps(data).encode("utf-8")
    header = struct.pack(_HEADER_FMT, len(payload))
    sock.sendall(header + payload)


def _send_signed_message(
    sock: socket.socket,
    data: dict[str, Any],
    key: bytes,
    nonce: str,
) -> None:
    """Send a message wrapped in an HMAC-signed envelope.

    Uses the provided nonce (which may be the request nonce for acks)
    so the signature binds to the specific conversation.
    """
    payload_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    payload_bytes = payload_str.encode("utf-8")
    signature = _sign(key, nonce, payload_bytes)
    _send_message(sock, {"payload": payload_str, "nonce": nonce, "signature": signature})


def _recv_message(sock: socket.socket) -> dict[str, Any] | None:
    header = _recv_exact(sock, _HEADER_SIZE)
    if header is None:
        return None
    length = struct.unpack(_HEADER_FMT, header)[0]
    if length == 0 or length > 64 * 1024:  # 64KB max for signal updates
        return None
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    try:
        result: dict[str, Any] | None = json.loads(payload.decode("utf-8"))
        return result
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _recv_signed_message(
    sock: socket.socket,
    key: bytes,
    expected_nonce: str,
) -> dict[str, Any] | None:
    """Receive a message, verify its HMAC signature, and return the inner payload.

    The nonce in the envelope must match ``expected_nonce`` (binds the ack
    to the original request).  Returns None on any verification failure.
    """
    envelope = _recv_message(sock)
    if not envelope:
        return None

    payload_str = envelope.get("payload")
    nonce = envelope.get("nonce")
    signature = envelope.get("signature")

    if not payload_str or not nonce or not signature:
        return None
    if (
        not isinstance(payload_str, str)
        or not isinstance(nonce, str)
        or not isinstance(signature, str)
    ):
        return None

    # Nonce must match the request we sent
    if nonce != expected_nonce:
        log.warning("IPC: ack nonce mismatch — dropping")
        return None

    # Verify HMAC
    expected_sig = _sign(key, nonce, payload_str.encode("utf-8"))
    if not hmac.compare_digest(signature, expected_sig):
        log.warning("IPC: ack HMAC signature mismatch — dropping")
        return None

    try:
        result: dict[str, Any] | None = json.loads(payload_str)
        return result
    except (json.JSONDecodeError, ValueError):
        return None


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (TimeoutError, OSError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class SignalServer:
    """TCP server for the trading bot to receive signal updates.

    Runs its accept loop in a background thread. When a valid HMAC-signed
    signal arrives, calls the registered handler.
    """

    def __init__(
        self,
        handler: Callable[[dict[str, Any], str], None],
        host: str = "127.0.0.1",
        port: int = 19731,
    ) -> None:
        self._handler = handler
        self._host = host
        self._port = port
        self._key = get_hmac_key()
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._seen_nonces: deque[str] = deque(maxlen=_NONCE_HISTORY_SIZE)
        self._nonce_lock = threading.Lock()

    def start(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.settimeout(2.0)
        self._server_sock.bind((self._host, self._port))
        self._server_sock.listen(2)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True, name="ipc-server")
        self._thread.start()
        log.info("IPC server listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        self._shutdown.set()
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)

    def _accept_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                assert self._server_sock is not None  # noqa: S101  # set in start()
                client, _addr = self._server_sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                client.settimeout(_RECV_TIMEOUT)
                self._handle_client(client)
            # broad catch: server-side handler, unknown client errors
            except Exception:
                log.exception("IPC client handler error")
            finally:
                try:
                    client.close()
                except OSError:
                    pass

    def _handle_client(self, sock: socket.socket) -> None:
        msg = _recv_message(sock)
        if not msg:
            log.warning("IPC: empty or invalid message — dropping connection")
            return

        # --- Extract envelope fields ---
        payload_str = msg.get("payload")
        nonce = msg.get("nonce")
        signature = msg.get("signature")

        if not payload_str or not nonce or not signature:
            log.warning("IPC: missing envelope fields (payload/nonce/signature) — dropping")
            return

        # --- HMAC verification (against the exact payload bytes from the sender) ---
        payload_bytes = payload_str.encode("utf-8")
        expected = _sign(self._key, nonce, payload_bytes)

        if not hmac.compare_digest(signature, expected):
            log.warning("IPC: HMAC signature mismatch — dropping")
            return

        # --- Parse the authenticated payload ---
        try:
            inner = json.loads(payload_str)
        except (json.JSONDecodeError, ValueError):
            log.warning("IPC: payload is not valid JSON — dropping")
            return

        if inner.get("type") != "signal_update":
            log.warning("IPC: unexpected message type %r — dropping", inner.get("type"))
            return

        # --- Timestamp freshness ---
        timestamp_str = inner.get("timestamp")
        if not timestamp_str:
            log.warning("IPC: missing timestamp in payload — dropping")
            return

        try:
            msg_time = datetime.fromisoformat(timestamp_str)
            # Reject naive datetimes — require explicit timezone
            if msg_time.tzinfo is None:
                log.warning("IPC: timestamp has no timezone — dropping (must be UTC)")
                return
            age = abs((datetime.now(UTC) - msg_time).total_seconds())
        except (ValueError, TypeError):
            log.warning("IPC: invalid timestamp format — dropping")
            return

        if age > _TIMESTAMP_MAX_AGE:
            log.warning("IPC: message too old (%.1fs) — dropping", age)
            return

        # --- Replay protection ---
        with self._nonce_lock:
            if nonce in self._seen_nonces:
                log.warning("IPC: duplicate nonce — dropping (replay attempt?)")
                return
            self._seen_nonces.append(nonce)

        # --- Authenticated — process signal ---
        signal_data = inner.get("signal", {})
        summary_file = inner.get("summary_file", "")
        log.info("IPC: received authenticated signal_update (summary=%s)", summary_file)
        try:
            self._handler(signal_data, summary_file)
            # Signed ack — bound to the request nonce so it can't be spoofed
            _send_signed_message(sock, {"type": "ack"}, self._key, nonce)
        # broad catch: server-side handler, unknown client errors
        except Exception:
            log.exception("IPC: handler error processing signal_update")


class SignalClient:
    """TCP client for the orchestrator to push HMAC-signed signals to the bot."""

    def __init__(self, host: str = "127.0.0.1", port: int = 19731) -> None:
        self._host = host
        self._port = port
        self._key = get_hmac_key()

    def send_signal(self, signal_data: dict[str, Any], summary_file: str) -> bool:
        """Send an HMAC-signed signal update to the bot. Returns True on success."""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(_SEND_TIMEOUT)
            sock.connect((self._host, self._port))

            nonce = secrets.token_hex(16)
            timestamp = datetime.now(UTC).isoformat()

            # Build the inner payload and serialize to canonical JSON once.
            # This exact string is what both sides use for HMAC input.
            inner = {
                "type": "signal_update",
                "signal": signal_data,
                "summary_file": summary_file,
                "timestamp": timestamp,
            }
            payload_str = json.dumps(inner, sort_keys=True, separators=(",", ":"))
            payload_bytes = payload_str.encode("utf-8")

            signature = _sign(self._key, nonce, payload_bytes)

            # Send envelope: the canonical payload string + auth fields.
            _send_message(
                sock,
                {
                    "payload": payload_str,
                    "nonce": nonce,
                    "signature": signature,
                },
            )

            # Verify signed ack — nonce must match our request
            ack = _recv_signed_message(sock, self._key, nonce)
            if ack and ack.get("type") == "ack":
                log.info("IPC: signal delivered successfully (authenticated ack)")
                return True

            log.warning("IPC: no valid ack received from bot")
            return False

        except (TimeoutError, ConnectionRefusedError, ConnectionResetError) as exc:
            log.error("IPC: connection to bot failed — %s", exc)
            return False
        except (OSError, ConnectionError, ValueError) as exc:
            log.error("IPC: unexpected error sending signal — %s", exc)
            return False
        finally:
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass
