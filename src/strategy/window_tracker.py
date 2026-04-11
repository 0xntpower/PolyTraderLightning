"""Track current and next 5-minute windows, fetch token IDs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import aiohttp

if TYPE_CHECKING:
    from config import Config
    from market_data.latency_tracker import LatencyTracker
    from market_data.state import MarketState

log = logging.getLogger(__name__)

WINDOW_DURATION = 300
GAMMA_MAX_RETRIES = 3
GAMMA_RETRY_DELAY = 2.0
GAMMA_OPEN_PRICE_RETRY_DELAY = 0.5  # faster retries for open price (non-blocking in main loop)
RESOLUTION_MAX_RETRIES = 3
RESOLUTION_RETRY_DELAY = 2.0


class ResolutionData(NamedTuple):
    """Result from a successful Gamma resolution poll."""

    outcome: Literal["up", "down"]
    price_to_beat: float | None  # eventMetadata.priceToBeat — open of resolved window
    final_price: float | None  # eventMetadata.finalPrice — close of resolved window


@dataclass(frozen=True, slots=True)
class WindowInfo:
    window_ts: int
    slug: str
    up_token_id: str
    down_token_id: str
    open_price: float | None  # eventMetadata.priceToBeat from Gamma API


class WindowTracker:
    def __init__(
        self,
        cfg: Config,
        state: MarketState,
        session: aiohttp.ClientSession,
        latency: LatencyTracker | None = None,
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.session = session
        self.latency = latency
        self._current_window_ts: int = 0
        self._resolution_consecutive_failures: int = 0
        # Non-blocking open price fetch — background task + result cache
        self._open_price_slug: str | None = None
        self._open_price_task: asyncio.Task[float | None] | None = None
        self._open_price_last_launch: float = 0.0
        self.open_price_fetched: bool = False
        # Pre-fetch for the next window's open price
        self._prefetch_next_ts: int = 0
        self._prefetch_task: asyncio.Task[float | None] | None = None
        self._prefetch_price: float | None = None
        self._prefetch_last_launch: float = 0.0

    def current_window_ts(self) -> int:
        now = int(time.time())
        return now - (now % WINDOW_DURATION)

    def time_remaining(self) -> float:
        now = time.time()
        window_end = self.current_window_ts() + WINDOW_DURATION
        return max(0.0, window_end - now)

    def make_slug(self, window_ts: int) -> str:
        return f"btc-updown-5m-{window_ts}"

    async def fetch_token_ids(self, slug: str) -> tuple[str, str] | None:
        """Fetch Up/Down token IDs from Gamma API with retries."""
        url = f"{self.cfg.connections.gamma_rest}/markets?slug={slug}"

        for attempt in range(GAMMA_MAX_RETRIES):
            try:
                t0 = time.time()
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        log.warning(
                            "gamma API returned %d for slug=%s (attempt %d/%d)",
                            resp.status,
                            slug,
                            attempt + 1,
                            GAMMA_MAX_RETRIES,
                        )
                        await asyncio.sleep(GAMMA_RETRY_DELAY)
                        continue
                    data = await resp.json()
                if self.latency is not None:
                    self.latency.record_rest("gamma_rest", (time.time() - t0) * 1000)
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                log.warning(
                    "gamma API error: %s (attempt %d/%d)", exc, attempt + 1, GAMMA_MAX_RETRIES
                )
                await asyncio.sleep(GAMMA_RETRY_DELAY)
                continue

            if not data:
                log.warning(
                    "gamma API returned empty for slug=%s (attempt %d/%d)",
                    slug,
                    attempt + 1,
                    GAMMA_MAX_RETRIES,
                )
                await asyncio.sleep(GAMMA_RETRY_DELAY)
                continue

            market = data[0] if isinstance(data, list) else data
            raw_ids = market.get("clobTokenIds", "[]")
            token_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            if len(token_ids) < 2:
                log.warning("market %s has fewer than 2 token IDs", slug)
                return None

            # Match token IDs to outcomes by name instead of assuming index order
            raw_outcomes = market.get("outcomes", "[]")
            outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
            if len(outcomes) >= 2:
                up_idx, down_idx = None, None
                for i, name in enumerate(outcomes):
                    lower = name.lower() if isinstance(name, str) else ""
                    if lower == "up":
                        up_idx = i
                    elif lower == "down":
                        down_idx = i
                if up_idx is not None and down_idx is not None:
                    return token_ids[up_idx], token_ids[down_idx]
                # Outcomes exist but don't contain "up"/"down" — fall through to index assumption
                log.warning(
                    "market %s: outcomes %r don't contain 'Up'/'Down', assuming index order",
                    slug,
                    outcomes,
                )
                return token_ids[0], token_ids[1]

            # Fallback: assume index 0=Up, 1=Down if outcomes unavailable
            log.warning("market %s: no outcomes field, assuming token order Up/Down", slug)
            return token_ids[0], token_ids[1]

        return None

    async def _fetch_price_to_beat(self, slug: str, timeout_s: float = 3.0) -> float | None:
        """Fetch priceToBeat from Gamma events API (single attempt).

        This is the shared HTTP helper used by both pre-fetch and main-loop
        retry paths. It runs inside an asyncio.Task so it never blocks the
        main loop.
        """
        url = f"{self.cfg.connections.gamma_rest}/events?slug={slug}&metadata=true"
        try:
            t0 = time.time()
            async with self.session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout_s)
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            if self.latency is not None:
                self.latency.record_rest("gamma_rest", (time.time() - t0) * 1000)
        except (aiohttp.ClientError, TimeoutError, OSError):
            return None

        if not data:
            return None

        event = data[0] if isinstance(data, list) else data
        event_metadata = event.get("eventMetadata")
        if not isinstance(event_metadata, dict):
            return None

        ptb_raw = event_metadata.get("priceToBeat")
        if ptb_raw is not None:
            try:
                price = float(ptb_raw)
                if price > 0:
                    return price
            except (ValueError, TypeError):
                pass
        return None

    def prefetch_next_open_price(self, time_remaining: float) -> None:
        """Pre-fetch the next window's priceToBeat in the last seconds of the current window.

        Launches a background task so the main loop is never blocked. Called
        every tick; rate-limited internally. The cached result is consumed by
        on_new_window() at the transition.
        """
        if time_remaining > 5.0 or time_remaining < 0.5:
            return
        if self._current_window_ts <= 0:
            return

        next_ts = self._current_window_ts + WINDOW_DURATION
        if self._prefetch_next_ts == next_ts and self._prefetch_price is not None:
            return  # already have it cached

        # Harvest completed task
        if self._prefetch_task is not None:
            if not self._prefetch_task.done():
                return  # still in-flight, don't launch another
            try:
                result = self._prefetch_task.result()
            except BaseException:  # task errors are non-fatal
                result = None
            self._prefetch_task = None
            if result is not None:
                self._prefetch_price = result
                self._prefetch_next_ts = next_ts
                log.info(
                    "pre-fetched open price for next window %d: $%.2f",
                    next_ts,
                    result,
                )
                return

        # Rate-limit launches
        now = time.time()
        if now - self._prefetch_last_launch < GAMMA_OPEN_PRICE_RETRY_DELAY:
            return

        self._prefetch_last_launch = now
        self._prefetch_next_ts = next_ts
        slug = self.make_slug(next_ts)
        self._prefetch_task = asyncio.create_task(self._fetch_price_to_beat(slug, timeout_s=2.0))

    def try_fetch_open_price(self) -> float | None:
        """Non-blocking attempt to get priceToBeat for the current window.

        Launches a background asyncio.Task and returns None immediately.
        On subsequent calls, checks whether the task completed and returns
        the result. Called every tick from the main loop until open_price_fetched
        is set.
        """
        if self._open_price_slug is None:
            return None

        # Harvest completed task
        if self._open_price_task is not None:
            if not self._open_price_task.done():
                return None  # still in-flight
            try:
                result = self._open_price_task.result()
            except BaseException:  # task errors are non-fatal
                result = None
            self._open_price_task = None
            if result is not None:
                return result
            # Task returned None — fall through to launch a new one

        # Rate-limit launches
        now = time.time()
        if now - self._open_price_last_launch < GAMMA_OPEN_PRICE_RETRY_DELAY:
            return None

        self._open_price_last_launch = now
        self._open_price_task = asyncio.create_task(
            self._fetch_price_to_beat(self._open_price_slug)
        )
        return None

    async def on_new_window(self) -> WindowInfo | None:
        """Called when a new window starts. Fetches token IDs and updates state."""
        wts = self.current_window_ts()
        if wts == self._current_window_ts:
            return None

        self._current_window_ts = wts
        slug = self.make_slug(wts)
        log.info("new window: %s (ts=%d)", slug, wts)

        tokens = await self.fetch_token_ids(slug)
        if not tokens:
            log.warning("skipping window %d — could not fetch token IDs after retries", wts)
            return None

        up_id, down_id = tokens
        self.state.reset_for_window(wts)
        self.state.up_token_id = up_id
        self.state.down_token_id = down_id

        # Check prefetch cache, else set up background retries in main loop
        self._open_price_slug = slug
        self._open_price_last_launch = 0.0
        self._open_price_task = None
        self.open_price_fetched = False

        open_price: float | None = None

        # Consume prefetch result if available
        if self._prefetch_next_ts == wts and self._prefetch_price is not None:
            open_price = self._prefetch_price
            log.info(
                "window open price captured: %.2f (pre-fetched Gamma priceToBeat)",
                open_price,
            )
        # Also harvest a prefetch task that completed but wasn't consumed yet
        elif (
            self._prefetch_next_ts == wts
            and self._prefetch_task is not None
            and self._prefetch_task.done()
        ):
            try:
                open_price = self._prefetch_task.result()
            except BaseException:  # task may have been cancelled
                open_price = None
            if open_price is not None:
                log.info(
                    "window open price captured: %.2f (pre-fetched Gamma priceToBeat)",
                    open_price,
                )

        # Clear prefetch state
        self._prefetch_next_ts = 0
        self._prefetch_price = None
        self._prefetch_last_launch = 0.0
        self._prefetch_task = None

        if open_price is not None:
            self.state.window_open_price = open_price
            self.state.open_price_captured = True
            self.state.open_price_tier = 0
            self.open_price_fetched = True
        else:
            log.info(
                "Gamma priceToBeat not yet available for %s — fetching in background",
                slug,
            )
            # Launch first background fetch immediately
            self.try_fetch_open_price()

        return WindowInfo(
            window_ts=wts,
            slug=slug,
            up_token_id=up_id,
            down_token_id=down_id,
            open_price=open_price,
        )

    async def fetch_market_resolution(
        self,
        slug: str,
    ) -> ResolutionData | None:
        """Query Gamma API for a resolved market's outcome and priceToBeat.

        Returns ResolutionData(outcome, price_to_beat) if resolved, None if
        not yet resolved or on error. Polymarket markets typically resolve
        23-45s after the window end time.

        The ``&metadata=true`` parameter unlocks ``eventMetadata.priceToBeat``
        which is the oracle's authoritative open price for the window.
        """
        # Use /events endpoint — /markets doesn't include umaResolutionStatus
        url = f"{self.cfg.connections.gamma_rest}/events?slug={slug}&metadata=true"

        for attempt in range(RESOLUTION_MAX_RETRIES):
            try:
                async with self.session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 429:
                        log.warning("gamma API rate limited (429) for %s — backing off", slug)
                        await asyncio.sleep(5.0)
                        continue
                    if resp.status != 200:
                        if attempt < RESOLUTION_MAX_RETRIES - 1:
                            await asyncio.sleep(RESOLUTION_RETRY_DELAY)
                            continue
                        return None
                    data = await resp.json()
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                log.warning(
                    "resolution query failed for %s (attempt %d/%d): %s",
                    slug,
                    attempt + 1,
                    RESOLUTION_MAX_RETRIES,
                    exc,
                )
                self._resolution_consecutive_failures += 1
                if self._resolution_consecutive_failures >= 10:
                    log.error(
                        "gamma API: %d consecutive resolution poll failures",
                        self._resolution_consecutive_failures,
                    )
                if attempt < RESOLUTION_MAX_RETRIES - 1:
                    await asyncio.sleep(RESOLUTION_RETRY_DELAY)
                    continue
                return None

            # Reset consecutive failure counter on successful HTTP response
            self._resolution_consecutive_failures = 0

            if not data:
                return None

            # /events returns [{..., "markets": [{...market...}]}]
            event = data[0] if isinstance(data, list) else data
            markets = event.get("markets", [])
            if not markets:
                return None
            market = markets[0]

            if market.get("umaResolutionStatus") != "resolved":
                return None

            outcomes_raw = market.get("outcomes", "[]")
            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
            prices_raw = market.get("outcomePrices", "[]")
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw

            if len(outcomes) < 2 or len(prices) < 2:
                return None

            # Extract oracle prices from eventMetadata (&metadata=true)
            # priceToBeat = open of THIS (resolved) window
            # finalPrice  = close of THIS window = open of the NEXT window
            price_to_beat: float | None = None
            final_price: float | None = None
            event_metadata = event.get("eventMetadata")
            if isinstance(event_metadata, dict):
                ptb_raw = event_metadata.get("priceToBeat")
                if ptb_raw is not None:
                    try:
                        price_to_beat = float(ptb_raw)
                    except (ValueError, TypeError):
                        pass
                fp_raw = event_metadata.get("finalPrice")
                if fp_raw is not None:
                    try:
                        final_price = float(fp_raw)
                    except (ValueError, TypeError):
                        pass

            for i, name in enumerate(outcomes):
                try:
                    if float(prices[i]) >= 0.99:
                        outcome: Literal["up", "down"] = "up" if name.lower() == "up" else "down"
                        return ResolutionData(
                            outcome=outcome,
                            price_to_beat=price_to_beat,
                            final_price=final_price,
                        )
                except (ValueError, TypeError):
                    continue

            return None

        return None
