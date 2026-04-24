"""Component-side helper for PSLAgent enrollment (spec §7).

A component calls ``enroll_if_requested`` early in its startup. When
``PSLAGENT_ENROLL_ENDPOINT`` + ``PSLAGENT_ENROLL_NONCE`` are set in the
environment the helper connects to the agent's one-shot localhost listener,
sends a JSON self-description, awaits the ack, and calls ``sys.exit(0)`` —
the component has no other job on an enrollment run.

When those env vars are **not** set, the helper returns immediately and the
component continues its normal startup path. This means dropping the call
at the top of every component's ``main()`` is safe at all times.

Stdlib-only (``socket``, ``json``, ``struct``, ``os``, ``sys``) so the
helper adds zero new dependencies to the components that adopt it.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import NoReturn

_ENDPOINT_VAR = "PSLAGENT_ENROLL_ENDPOINT"
_NONCE_VAR = "PSLAGENT_ENROLL_NONCE"
_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Timeouts for the handshake itself (not the component's own start). Short:
# the agent is running on the same host and we have already connected.
_CONNECT_TIMEOUT_SEC = 5.0
_IO_TIMEOUT_SEC = 5.0
_ACK_MAX_BYTES = 16 * 1024


class EnrollmentHelperError(RuntimeError):
    """Something went wrong with the enrollment handshake."""


def enroll_if_requested(
    *,
    id: str,  # noqa: A002  # field name mirrors the wire-protocol JSON key
    display_name: str,
    config_files: Iterable[str] = (),
    log_files: Iterable[str] = (),
    telemetry_endpoint: tuple[str, int] = ("auto", 0),
    stop_signal: str = "SIGTERM",
    stop_signal_windows: str = "CTRL_BREAK_EVENT",
    start_timeout_sec: int = 20,
    healthcheck_timeout_sec: int = 5,
) -> None:
    """If the agent is running enrollment, complete the handshake and exit.

    Call this as early as possible in the component's startup — before any
    config loading, network connections, or other initialization. When the
    agent is *not* running enrollment this function returns immediately with
    no side effects.

    Args:
        id: slug identifying this component; must match
            ``[a-z0-9][a-z0-9_-]{0,63}``.
        display_name: human-readable label for the manager panel.
        config_files: paths relative to the component's cwd that the agent
            may read/write via ``config_get``/``config_set``.
        log_files: paths relative to the component's cwd the agent may tail
            or grep via ``log_*`` messages.
        telemetry_endpoint: ``(host, port)`` where this component publishes
            its state stream. ``"auto"`` for host means the agent substitutes
            its own host_id when writing the manifest.
        stop_signal: POSIX signal name sent on ``component_stop`` (SIGTERM/SIGINT).
        stop_signal_windows: Windows control event sent on stop
            (CTRL_BREAK_EVENT/CTRL_C_EVENT).
        start_timeout_sec: how long the agent should wait for this component
            to become ``running`` after ``component_start``.
        healthcheck_timeout_sec: how long the agent should wait for health
            probe responses before marking the component ``crashed``.

    On success the function does **not** return; it calls ``sys.exit(0)``.
    """
    endpoint = os.environ.get(_ENDPOINT_VAR, "").strip()
    nonce = os.environ.get(_NONCE_VAR, "").strip()
    if not endpoint or not nonce:
        return

    host, port = _parse_endpoint(endpoint)
    payload = {
        "nonce": nonce,
        "id": id,
        "display_name": display_name,
        "config_files": list(config_files),
        "log_files": list(log_files),
        "telemetry_endpoint": {
            "host": telemetry_endpoint[0],
            "port": telemetry_endpoint[1],
        },
        "stop_signal": stop_signal,
        "stop_signal_windows": stop_signal_windows,
        "start_timeout_sec": start_timeout_sec,
        "healthcheck_timeout_sec": healthcheck_timeout_sec,
    }
    _perform_handshake(host=host, port=port, payload=payload)
    _exit_success()


# ---------------------------------------------------------------------------
# Internals (exposed with leading underscore so tests can drive them directly
# without having the helper call sys.exit).
# ---------------------------------------------------------------------------


def _parse_endpoint(raw: str) -> tuple[str, int]:
    host, _sep, port_str = raw.rpartition(":")
    if not host or not port_str:
        raise EnrollmentHelperError(f"{_ENDPOINT_VAR} must be host:port, got {raw!r}")
    try:
        port = int(port_str)
    except ValueError as exc:
        raise EnrollmentHelperError(
            f"{_ENDPOINT_VAR} port is not an integer: {port_str!r}"
        ) from exc
    if not (1 <= port <= 65535):
        raise EnrollmentHelperError(f"{_ENDPOINT_VAR} port out of range: {port}")
    return host, port


def _perform_handshake(*, host: str, port: int, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode("utf-8")
    frame = struct.pack(_HEADER_FMT, len(body)) + body

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_CONNECT_TIMEOUT_SEC)
    try:
        try:
            sock.connect((host, port))
        except (TimeoutError, ConnectionRefusedError, OSError) as exc:
            raise EnrollmentHelperError(
                f"could not connect to enrollment endpoint {host}:{port}: {exc}"
            ) from exc

        sock.settimeout(_IO_TIMEOUT_SEC)
        try:
            sock.sendall(frame)
        except OSError as exc:
            raise EnrollmentHelperError(f"send failed: {exc}") from exc

        header = _recv_exact(sock, _HEADER_SIZE)
        length = struct.unpack(_HEADER_FMT, header)[0]
        if not (0 < length <= _ACK_MAX_BYTES):
            raise EnrollmentHelperError(f"invalid ack length: {length}")
        ack_body = _recv_exact(sock, length)
    finally:
        try:
            sock.close()
        except OSError:
            pass

    try:
        ack = json.loads(ack_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EnrollmentHelperError(f"ack is not valid JSON: {exc}") from exc
    if not isinstance(ack, dict) or not isinstance(ack.get("ok"), bool):
        raise EnrollmentHelperError(f"ack missing boolean 'ok': {ack!r}")
    if not ack["ok"]:
        err = ack.get("error", "<no error message>")
        raise EnrollmentHelperError(f"agent rejected enrollment: {err}")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (TimeoutError, OSError) as exc:
            raise EnrollmentHelperError(f"recv failed: {exc}") from exc
        if not chunk:
            raise EnrollmentHelperError(f"connection closed mid-read: got {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def _exit_success() -> NoReturn:
    """Terminate the process cleanly after a successful handshake."""
    sys.exit(0)
