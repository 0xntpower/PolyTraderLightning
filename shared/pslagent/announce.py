"""One-shot 'I'm alive' announcement from a component to the local PSLAgent.

A supervised component calls :func:`announce_alive` once, early in ``main()``,
after ``enroll_if_requested``. The helper sends a single HMAC-signed UDP
datagram to the local agent containing the component's id + PID. If the agent
isn't running (or isn't listening), the send fails silently — announcement
is fire-and-forget by design.

The agent uses this message to *adopt* externally-started components: if it
sees a valid announce for a component it thinks is STOPPED / CRASHED or
whose last-known PID differs, it swaps to the new PID and tracks that
process's exit. This lets operators launch components via the existing
``start_*.bat`` scripts or Task Scheduler while still surfacing them as
RUNNING in the panel and allowing panel-side STOP / RESTART.

Packet layout (single UDP datagram, ≤ 4 KiB):

    [32-byte HMAC-SHA256][UTF-8 JSON body]

The HMAC is computed over the JSON body alone, using the shared
``HMAC_KEY`` already distributed to every component in the fleet. JSON
body is the canonical form Python emits with ``sort_keys=True,
separators=(",", ":")`` so the agent and component hash identical bytes.

Default target is ``127.0.0.1:19734`` — local-only by design; there is
no cross-host announce use case.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import socket
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 19734
_SEND_TIMEOUT_SEC = 0.5


def announce_alive(
    component_id: str,
    *,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    hmac_key: bytes | None = None,
    pid: int | None = None,
) -> bool:
    """Send a single 'I'm alive' UDP packet to the local PSLAgent.

    Args:
        component_id: Manifest id for this component (must match what was
            enrolled, e.g. ``"polydatacollector"``).
        host: Agent host. Defaults to localhost — cross-host announce is
            not a supported flow.
        port: Agent's UDP announce port. Defaults to 19734.
        hmac_key: 32+ byte HMAC key. Defaults to loading ``HMAC_KEY`` from
            the shared keystore (the same fleet key used by the existing
            component IPC).
        pid: Override the announced PID. Defaults to ``os.getpid()``.

    Returns:
        True if the datagram was handed to the OS socket layer; False on
        any error. Callers should treat the return value as advisory —
        the agent may be down entirely, and the component should continue
        normally regardless.
    """
    if hmac_key is None:
        try:
            from shared.keystore import get_hmac_key

            hmac_key = get_hmac_key()
        except (ImportError, ValueError) as exc:
            log.info("announce_alive: no HMAC key available (%s) — skipping", exc)
            return False

    if pid is None:
        pid = os.getpid()

    body = json.dumps(
        {"id": component_id, "pid": pid},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.new(hmac_key, body, hashlib.sha256).digest()
    packet = signature + body

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(_SEND_TIMEOUT_SEC)
            sock.sendto(packet, (host, port))
    except OSError as exc:
        log.info("announce_alive: send failed (%s) — agent likely not running", exc)
        return False
    log.info("announce_alive: sent for %s (pid=%d) to %s:%d", component_id, pid, host, port)
    return True
