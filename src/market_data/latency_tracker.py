"""Passive latency tracker — instruments existing data flows, zero extra requests.

Each feed records message arrival times.  For Binance trades the exchange
timestamp lets us compute true wire latency.  For WebSocket feeds without
server timestamps we track message inter-arrival gaps (a proxy for feed
health).  REST calls record round-trip time.

All operations are O(1) amortised (fixed-size deques) and lock-free — safe
for a single-threaded asyncio event loop.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

_ROLLING_SIZE = 200  # keep last N samples per feed


@dataclass(frozen=True, slots=True)
class FeedStats:
    """Computed statistics for a single feed over the rolling window."""

    name: str
    samples: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    jitter_ms: float = 0.0  # mean abs difference between consecutive samples (RFC 3550)
    last_message_ago_s: float = 0.0  # seconds since last message


class _FeedAccumulator:
    """Fixed-size deque of latency samples (ms) for one feed."""

    __slots__ = ("_last_ts", "_samples", "name")

    def __init__(self, name: str) -> None:
        self.name = name
        self._samples: deque[float] = deque(maxlen=_ROLLING_SIZE)
        self._last_ts: float = 0.0

    def record(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)
        self._last_ts = time.time()

    def stats(self) -> FeedStats:
        n = len(self._samples)
        if n == 0:
            return FeedStats(name=self.name)
        vals = sorted(self._samples)
        # Jitter: mean absolute difference between consecutive samples
        # (RFC 3550 interarrival jitter, computed over arrival order not sorted)
        raw = self._samples
        jitter = 0.0
        if n >= 2:
            total = 0.0
            for i in range(1, n):
                total += abs(raw[i] - raw[i - 1])
            jitter = total / (n - 1)
        return FeedStats(
            name=self.name,
            samples=n,
            min_ms=vals[0],
            max_ms=vals[-1],
            median_ms=vals[n // 2],
            p95_ms=vals[int(n * 0.95)] if n >= 20 else vals[-1],
            mean_ms=sum(vals) / n,
            jitter_ms=jitter,
            last_message_ago_s=time.time() - self._last_ts if self._last_ts else 0.0,
        )

    def clear(self) -> None:
        self._samples.clear()


class LatencyTracker:
    """Singleton-style tracker shared across all feeds.

    Usage from WS handlers:
        tracker.record_ws("binance", latency_ms)
    Usage from REST wrappers:
        tracker.record_rest("gamma", elapsed_ms)
    """

    def __init__(self) -> None:
        self._feeds: dict[str, _FeedAccumulator] = {}

    def _acc(self, name: str) -> _FeedAccumulator:
        acc = self._feeds.get(name)
        if acc is None:
            acc = _FeedAccumulator(name)
            self._feeds[name] = acc
        return acc

    def record_ws(self, feed_name: str, latency_ms: float) -> None:
        """Record a WebSocket message latency sample."""
        self._acc(feed_name).record(latency_ms)

    def record_rest(self, endpoint_name: str, elapsed_ms: float) -> None:
        """Record a REST round-trip time sample."""
        self._acc(endpoint_name).record(elapsed_ms)

    def all_stats(self) -> list[FeedStats]:
        """Return stats for every tracked feed, sorted by name."""
        return sorted(
            (acc.stats() for acc in self._feeds.values()),
            key=lambda s: s.name,
        )

    def clear_all(self) -> None:
        """Reset all accumulators (called after reporting)."""
        for acc in self._feeds.values():
            acc.clear()
