"""Track fees paid and estimated maker rebates."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from market_data.latency_tracker import LatencyTracker

log = logging.getLogger(__name__)


class FeeTracker:
    def __init__(self, latency: LatencyTracker | None = None) -> None:
        self.total_taker_fees: float = 0.0
        self.total_estimated_rebates: float = 0.0
        self.fee_rate_bps: int = 0
        self._latency = latency

    async def fetch_fee_rate(
        self,
        session: aiohttp.ClientSession,
        clob_rest: str,
        token_id: str | None = None,
    ) -> int:
        """Fetch current fee rate from the V2 CLOB API.

        V2 fee model is per-token: ``GET /fee-rate?token_id=<id>`` returns
        ``{"base_fee": <bps>}``. UP and DOWN tokens of the same binary market
        share the same fee, so callers may pass either side's token_id.
        ``token_id=None`` skips the fetch and keeps the cached value (or the
        200 bps default) — used when token IDs aren't yet known (warmup).
        """
        if token_id is None:
            if self.fee_rate_bps == 0:
                self.fee_rate_bps = 200
            return self.fee_rate_bps
        try:
            t0 = time.time()
            async with session.get(
                f"{clob_rest}/fee-rate",
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    raw = data.get("base_fee")
                    if raw is None:
                        log.warning("base_fee missing from /fee-rate response — using default 200")
                        self.fee_rate_bps = 200
                    else:
                        self.fee_rate_bps = int(raw)
                    log.info("fee_rate_bps=%d (token=%s)", self.fee_rate_bps, token_id[:12])
                    if self._latency is not None:
                        self._latency.record_rest("clob_rest", (time.time() - t0) * 1000)
                    return self.fee_rate_bps
        except (aiohttp.ClientError, TimeoutError, OSError, KeyError, ValueError) as exc:
            log.warning("failed to fetch fee rate: %s", exc)

        if self.fee_rate_bps == 0:
            self.fee_rate_bps = 200
        return self.fee_rate_bps

    def compute_taker_fee(self, price: float, size: float) -> float:
        """Compute taker fee for a given price and size.

        Polymarket binary market fee: fee_rate * min(price, 1 - price) per share.
        """
        fee_per_share = self.fee_rate_bps / 10000 * min(price, 1 - price)
        return fee_per_share * size

    def record_taker_fee(self, price: float, size: float) -> float:
        fee = self.compute_taker_fee(price, size)
        self.total_taker_fees += fee
        return fee

    def record_maker_rebate(self, estimated_amount: float) -> None:
        self.total_estimated_rebates += estimated_amount
