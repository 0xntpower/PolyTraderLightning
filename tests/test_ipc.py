"""Tests for the HMAC-authenticated signal IPC (SignalServer + SignalClient).

Covers both message types on the v3.0 channel:

* ``signal_update`` — orchestrator pushes a new signal to the bot, handler
  is invoked, authenticated ack returns.
* ``status_query`` — orchestrator asks the bot for its recent resolved
  fires (v3.0 P5 signal-identity dedupe feedback loop). The ack must
  carry the ``status_provider``'s dict merged into the signed envelope.

Every test spins up a real ``SignalServer`` on an ephemeral port so the
framing, HMAC signing, timestamp/nonce validation, and ack flow are all
exercised end-to-end without mocks.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import socket
import struct
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

import pytest

from shared import ipc as ipc_module
from shared.ipc import SignalClient, SignalServer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_TEST_HMAC_KEY: Final = b"\xbb" * 32


@pytest.fixture(autouse=True)
def _hermetic_hmac_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ipc.get_hmac_key so tests don't pull a real secret from the
    developer's Credential Manager / age store."""
    monkeypatch.setattr(ipc_module, "get_hmac_key", lambda: _TEST_HMAC_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_server(
    handler: Callable[[dict[str, Any], str], None],
    *,
    status_provider: Callable[[], dict[str, Any]] | None = None,
) -> tuple[SignalServer, int]:
    """Bind a server on an ephemeral port, start it, and return (server, port)."""
    server = SignalServer(handler, host="127.0.0.1", port=0, status_provider=status_provider)
    server.start()
    sock = server._server_sock
    assert sock is not None
    port = sock.getsockname()[1]
    return server, port


@pytest.fixture
def server_factory() -> Iterator[Callable[..., tuple[SignalServer, int]]]:
    """Factory that starts a server and cleans it up at test teardown."""
    started: list[SignalServer] = []

    def _factory(
        handler: Callable[[dict[str, Any], str], None] | None = None,
        *,
        status_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> tuple[SignalServer, int]:
        actual_handler: Callable[[dict[str, Any], str], None] = (
            handler if handler is not None else (lambda _s, _f: None)
        )
        server, port = _start_server(actual_handler, status_provider=status_provider)
        started.append(server)
        return server, port

    yield _factory

    for srv in started:
        srv.stop()


def _wait_for_port(port: int, timeout: float = 2.0) -> None:
    """Poll the port until the listen socket accepts connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError(f"server on port {port} never became reachable")


def _send_raw(port: int, envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Send a raw envelope and try to read back one framed response.

    Used by tests that need to forge bad envelopes — the production
    ``SignalClient`` always signs correctly.
    """
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as sock:
        body = json.dumps(envelope).encode("utf-8")
        header = struct.pack("!I", len(body))
        sock.sendall(header + body)
        try:
            hdr = _recv_exact(sock, 4)
            if hdr is None:
                return None
            (length,) = struct.unpack("!I", hdr)
            payload = _recv_exact(sock, length)
            if payload is None:
                return None
            result: dict[str, Any] = json.loads(payload.decode("utf-8"))
            return result
        except (TimeoutError, OSError):
            return None


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    sock.settimeout(1.0)
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (TimeoutError, OSError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _sign(nonce: str, payload_bytes: bytes) -> str:
    return hmac.new(
        _TEST_HMAC_KEY, nonce.encode("utf-8") + payload_bytes, hashlib.sha256
    ).hexdigest()


def _make_signed_envelope(inner: dict[str, Any], *, nonce: str | None = None) -> dict[str, Any]:
    n = nonce or secrets.token_hex(16)
    payload_str = json.dumps(inner, sort_keys=True, separators=(",", ":"))
    return {
        "payload": payload_str,
        "nonce": n,
        "signature": _sign(n, payload_str.encode("utf-8")),
    }


# ---------------------------------------------------------------------------
# signal_update happy-path
# ---------------------------------------------------------------------------


class TestSignalUpdate:
    def test_signed_signal_delivered_to_handler(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        received: list[tuple[dict[str, Any], str]] = []
        handler_done = threading.Event()

        def handler(sig: dict[str, Any], summary: str) -> None:
            received.append((sig, summary))
            handler_done.set()

        _, port = server_factory(handler)
        _wait_for_port(port)

        client = SignalClient(host="127.0.0.1", port=port)
        ok = client.send_signal({"rank": 1, "side": "up"}, "summary.json")

        assert ok is True
        assert handler_done.wait(timeout=2.0)
        assert len(received) == 1
        signal_data, summary_file = received[0]
        assert signal_data == {"rank": 1, "side": "up"}
        assert summary_file == "summary.json"

    def test_send_signal_returns_false_when_no_server(self) -> None:
        # A port the OS is almost certainly not listening on.
        client = SignalClient(host="127.0.0.1", port=1)
        assert client.send_signal({}, "") is False

    def test_tampered_signature_is_rejected(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        called = threading.Event()

        def handler(_s: dict[str, Any], _f: str) -> None:
            called.set()

        _, port = server_factory(handler)
        _wait_for_port(port)

        inner = {
            "type": "signal_update",
            "signal": {"rank": 1},
            "summary_file": "s.json",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        env = _make_signed_envelope(inner)
        env["signature"] = "0" * 64  # wipe real signature
        _send_raw(port, env)

        # Handler must not have been called — server dropped on HMAC mismatch.
        assert not called.wait(timeout=0.3)

    def test_stale_timestamp_is_rejected(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        called = threading.Event()

        def handler(_s: dict[str, Any], _f: str) -> None:
            called.set()

        _, port = server_factory(handler)
        _wait_for_port(port)

        old_ts = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        inner = {
            "type": "signal_update",
            "signal": {"rank": 1},
            "summary_file": "s.json",
            "timestamp": old_ts,
        }
        _send_raw(port, _make_signed_envelope(inner))

        assert not called.wait(timeout=0.3)

    def test_naive_timestamp_is_rejected(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        """A timestamp without timezone info must be rejected as ambiguous."""
        called = threading.Event()

        def handler(_s: dict[str, Any], _f: str) -> None:
            called.set()

        _, port = server_factory(handler)
        _wait_for_port(port)

        inner = {
            "type": "signal_update",
            "signal": {},
            "summary_file": "",
            "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        }
        _send_raw(port, _make_signed_envelope(inner))

        assert not called.wait(timeout=0.3)

    def test_replay_of_same_nonce_is_rejected(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        call_count = [0]
        event = threading.Event()

        def handler(_s: dict[str, Any], _f: str) -> None:
            call_count[0] += 1
            event.set()

        _, port = server_factory(handler)
        _wait_for_port(port)

        nonce = secrets.token_hex(16)
        inner = {
            "type": "signal_update",
            "signal": {"rank": 1},
            "summary_file": "s.json",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        env = _make_signed_envelope(inner, nonce=nonce)

        _send_raw(port, env)
        event.wait(timeout=1.0)
        assert call_count[0] == 1

        # Replay the identical envelope — server must drop it.
        _send_raw(port, env)
        time.sleep(0.1)
        assert call_count[0] == 1  # handler still called only once

    def test_unknown_message_type_is_dropped(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        called = threading.Event()

        def handler(_s: dict[str, Any], _f: str) -> None:
            called.set()

        _, port = server_factory(handler)
        _wait_for_port(port)

        inner = {
            "type": "something_evil",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _send_raw(port, _make_signed_envelope(inner))

        assert not called.wait(timeout=0.3)


# ---------------------------------------------------------------------------
# status_query — v3.0 feedback channel
# ---------------------------------------------------------------------------


class TestStatusQuery:
    def test_status_query_returns_provider_dict_in_ack(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        call_count = [0]

        def provider() -> dict[str, Any]:
            call_count[0] += 1
            return {
                "recent_fires": [
                    {"signal_id": "up_240_180", "won": True, "timestamp": "t1"},
                    {"signal_id": "up_240_180", "won": False, "timestamp": "t2"},
                ],
                "mode": "paper",
            }

        _, port = server_factory(status_provider=provider)
        _wait_for_port(port)

        client = SignalClient(host="127.0.0.1", port=port)
        ack = client.query_status()

        assert ack is not None
        assert call_count[0] == 1
        assert ack["mode"] == "paper"
        assert len(ack["recent_fires"]) == 2
        assert ack["recent_fires"][0]["signal_id"] == "up_240_180"
        # The "type" field is stripped by query_status before returning.
        assert "type" not in ack

    def test_status_query_without_provider_returns_empty_ack(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        _, port = server_factory()  # no provider
        _wait_for_port(port)

        client = SignalClient(host="127.0.0.1", port=port)
        ack = client.query_status()

        assert ack == {}  # no fields apart from the stripped "type"

    def test_status_provider_exception_returns_empty_ack(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        """A buggy provider must never bring down the IPC server — the
        handler catches and returns a plain ack so the orchestrator can
        distinguish 'no data' from 'unreachable'."""

        def broken_provider() -> dict[str, Any]:
            raise RuntimeError("boom")

        _, port = server_factory(status_provider=broken_provider)
        _wait_for_port(port)

        client = SignalClient(host="127.0.0.1", port=port)
        ack = client.query_status()

        assert ack == {}  # empty, but valid — the channel survived the error

    def test_status_query_does_not_invoke_signal_handler(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        """status_query must never trigger the signal_update handler —
        the two code paths are strictly separate."""
        handler_calls: list[tuple[dict[str, Any], str]] = []

        def handler(sig: dict[str, Any], summary: str) -> None:
            handler_calls.append((sig, summary))

        def provider() -> dict[str, Any]:
            return {"recent_fires": [], "mode": "live"}

        _, port = server_factory(handler, status_provider=provider)
        _wait_for_port(port)

        client = SignalClient(host="127.0.0.1", port=port)
        ack = client.query_status()
        time.sleep(0.1)

        assert ack is not None
        assert ack["mode"] == "live"
        assert handler_calls == []

    def test_query_status_returns_none_when_server_down(self) -> None:
        client = SignalClient(host="127.0.0.1", port=1)
        assert client.query_status() is None

    def test_status_provider_returning_non_dict_is_ignored(
        self,
        server_factory: Callable[..., tuple[SignalServer, int]],
    ) -> None:
        """If the provider accidentally returns something non-dict-like,
        the handler must not crash — it falls through to an empty ack."""

        def provider() -> dict[str, Any]:
            return ["not", "a", "dict"]  # type: ignore[return-value]  # deliberate misuse

        _, port = server_factory(status_provider=provider)
        _wait_for_port(port)

        client = SignalClient(host="127.0.0.1", port=port)
        ack = client.query_status()

        assert ack == {}


# ---------------------------------------------------------------------------
# Concurrency — many status_queries while signals are flowing
# ---------------------------------------------------------------------------


def test_interleaved_signal_and_status_traffic(
    server_factory: Callable[..., tuple[SignalServer, int]],
) -> None:
    """The IPC server must serve status_query and signal_update traffic
    without cross-contamination or dropped messages."""
    received: list[dict[str, Any]] = []
    received_lock = threading.Lock()

    def handler(sig: dict[str, Any], _f: str) -> None:
        with received_lock:
            received.append(sig)

    def provider() -> dict[str, Any]:
        return {"recent_fires": [], "mode": "paper"}

    _, port = server_factory(handler, status_provider=provider)
    _wait_for_port(port)

    client = SignalClient(host="127.0.0.1", port=port)

    for i in range(5):
        assert client.send_signal({"rank": i}, f"s_{i}.json") is True
        ack = client.query_status()
        assert ack is not None
        assert ack["mode"] == "paper"

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with received_lock:
            if len(received) == 5:
                break
        time.sleep(0.02)

    with received_lock:
        assert len(received) == 5
        assert [r["rank"] for r in received] == [0, 1, 2, 3, 4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
