"""WebSocket reconnection with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import websockets

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from websockets.asyncio.client import ClientConnection

log = logging.getLogger(__name__)


@dataclass
class WsHealthState:
    """Observable health state for a single WebSocket feed.

    Written by ``ws_connect_forever``, read by the strategy loop.
    """

    name: str = ""
    is_connected: bool = False
    last_connected_at: float = 0.0
    last_disconnected_at: float = 0.0
    consecutive_failures: int = 0

    def seconds_disconnected(self) -> float:
        """Return how long the feed has been disconnected, or 0.0 if connected."""
        if self.is_connected:
            return 0.0
        if self.last_disconnected_at <= 0:
            return 0.0
        return time.time() - self.last_disconnected_at


@dataclass
class FeedHealthMonitor:
    """Aggregates health of all WebSocket feeds for the strategy loop."""

    feeds: dict[str, WsHealthState] = field(default_factory=dict)

    def register(self, name: str) -> WsHealthState:
        """Create and register a health state for a named feed."""
        hs = WsHealthState(name=name)
        self.feeds[name] = hs
        return hs

    def critical_feed_down(self, threshold_sec: float = 30.0) -> str | None:
        """Return the name of the first critical feed that has been
        disconnected for longer than *threshold_sec*, or None if all OK.

        Critical feeds: binance, rtds (Chainlink), clob-market.
        """
        for name in ("binance", "rtds", "clob-market"):
            hs = self.feeds.get(name)
            if hs is None:
                continue
            if hs.seconds_disconnected() > threshold_sec:
                return name
        return None


async def ws_connect_forever(
    uri: str,
    handler: Callable[[ClientConnection], Awaitable[None]],
    *,
    name: str = "ws",
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    extra_headers: dict[str, str] | None = None,
    health: WsHealthState | None = None,
) -> None:
    """Connect to a WebSocket and call handler. Reconnect on failure with backoff + jitter."""
    delay = base_delay
    while True:
        try:
            async with websockets.connect(
                uri,
                additional_headers=extra_headers,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
            ) as ws:
                log.info("%s connected to %s", name, uri)
                delay = base_delay
                if health is not None:
                    health.is_connected = True
                    health.last_connected_at = time.time()
                    health.consecutive_failures = 0
                await handler(ws)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if health is not None:
                health.is_connected = False
                health.last_disconnected_at = time.time()
                health.consecutive_failures += 1
            jittered = delay * (0.5 + random.random())  # noqa: S311  # jitter delay, not cryptographic
            log.warning("%s disconnected: %s — reconnecting in %.1fs", name, exc, jittered)
            await asyncio.sleep(jittered)
            delay = min(delay * 2, max_delay)
