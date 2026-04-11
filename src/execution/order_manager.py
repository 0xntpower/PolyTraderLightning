"""Place, cancel, and track orders via py-clob-client."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING, Any

from py_clob_client.clob_types import (  # type: ignore[import-untyped]  # no stubs available
    BalanceAllowanceParams,
    MarketOrderArgs,
    OrderArgs,
    OrderType,
)
from py_clob_client.order_builder.constants import (  # type: ignore[import-untyped]  # no stubs available
    BUY,
)

from shared.discord import send_bet_cancelled
from utils.circuit_breaker import CircuitBreaker

if TYPE_CHECKING:
    from py_clob_client.client import (  # type: ignore[import-untyped]  # no stubs available
        ClobClient,
    )

    from config import Config
    from market_data.state import MarketState
    from risk.registry import RiskRegistry
    from strategy.signal import Signal

log = logging.getLogger(__name__)

# Timeout for synchronous CLOB API calls run via executor.
# Prevents hung HTTP requests from blocking the event loop indefinitely.
_CLOB_CALL_TIMEOUT_SEC = 10.0


class OrderManager:
    mode: str = "live"

    def __init__(
        self,
        cfg: Config,
        state: MarketState,
        clob: ClobClient,
        risk: RiskRegistry,
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.clob = clob
        self.risk = risk
        self._cancel_in_progress: bool = False
        self._cached_balance_usd: float | None = None
        # Track placed order details for cancel notifications
        self._order_details: dict[str, dict[str, Any]] = {}
        # Circuit breaker — stops hammering CLOB API after consecutive failures
        self._breaker = CircuitBreaker("clob_orders", failure_threshold=3, cooldown_sec=60.0)

    # -- Paper-compatible no-ops (called unconditionally by MomentumSignalStrategy) --

    def set_rule_triggered(self, rule_id: int, direction: str, signal: Signal | None) -> None:
        """No-op in live mode. PaperOrderManager overrides to record in WindowRecord."""

    def set_kelly_fields(
        self,
        **kwargs: float | bool | None,
    ) -> None:
        """No-op in live mode. PaperOrderManager overrides to record in WindowRecord."""

    async def exit_position_early(self, sell_price: float) -> float | None:  # noqa: ARG002  # required by OrderExecutor Protocol
        """No-op in live mode. PaperOrderManager overrides to simulate early exit."""
        return None

    async def refresh_balance(self) -> float | None:
        """Fetch on-chain USDC balance from CLOB API and cache it.

        Called once per window boundary to keep the cached value fresh.
        Returns balance in USD or None on failure.
        """
        loop = asyncio.get_running_loop()
        try:
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.clob.get_balance_allowance(
                        BalanceAllowanceParams(
                            signature_type=self.cfg.connections.signature_type,
                        )
                    ),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
            # Response has 'balance' in raw token units (6 decimals for USDC)
            raw_balance = float(resp.get("balance", 0))
            self._cached_balance_usd = raw_balance / 1e6
            log.info("on-chain balance refreshed: $%.2f", self._cached_balance_usd)
            return self._cached_balance_usd
        except TimeoutError:
            log.warning("refresh_balance timed out after %.0fs", _CLOB_CALL_TIMEOUT_SEC)
            return self._cached_balance_usd
        except (OSError, ValueError, KeyError) as exc:
            log.warning("failed to refresh on-chain balance: %s", exc)
            return self._cached_balance_usd

    def is_order_filled(self, order_id: str) -> bool:
        """Check if an order was filled via the CLOB user WebSocket."""
        return order_id in self.state.live_fills

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID.

        Returns True if cancelled, False if already filled or failed.
        """
        assert self.risk.tracker is not None  # noqa: S101  # set in RiskRegistry.from_config()
        if order_id in self.state.live_fills:
            return False  # already filled

        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(self.clob.cancel, order_id),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
            try:
                self.state.active_order_ids.remove(order_id)
            except ValueError:
                pass
            detail = self._order_details.pop(order_id, None)
            if detail:
                self.risk.tracker.remove_exposure(detail["size_usd"])
            log.info("cancelled order %s", order_id[:12])
            return True
        except TimeoutError:
            log.warning("cancel_order timed out for %s", order_id[:12])
            return False
        except (OSError, ValueError, KeyError) as exc:
            # May have been filled while we were cancelling
            if order_id in self.state.live_fills:
                return False
            log.warning("cancel_order failed for %s: %s", order_id[:12], exc)
            return False

    def _best_ask_for(self, token_id: str) -> float:
        if token_id == self.state.up_token_id:
            return self.state.best_ask_up
        if token_id == self.state.down_token_id:
            return self.state.best_ask_down
        return 0.0

    async def place_maker_order(
        self,
        token_id: str,
        price: float,
        size_usd: float,
        tier: str,
    ) -> str | None:
        """Place a GTC maker (post-only) order. Returns order ID or None."""
        assert self.risk.tracker is not None  # noqa: S101  # set in RiskRegistry.from_config()
        if not self._breaker.can_attempt():
            log.warning("%s blocked: CLOB circuit breaker OPEN", tier)
            return None

        if not self.risk.can_trade(size_usd):
            log.warning("%s blocked by risk limits", tier)
            return None

        # On-chain balance gate — block if balance unknown or insufficient
        if self._cached_balance_usd is None:
            log.warning("%s blocked: on-chain balance unknown (refresh not yet succeeded)", tier)
            return None
        if size_usd > self._cached_balance_usd:
            log.warning(
                "%s blocked: insufficient on-chain balance $%.2f < order $%.2f",
                tier,
                self._cached_balance_usd,
                size_usd,
            )
            return None

        # Post-only safety: reject if our bid would cross the spread
        best_ask = self._best_ask_for(token_id)
        if best_ask > 0 and price >= best_ask:
            log.warning(
                "%s maker rejected: price %.2f would cross ask %.2f (taker fees)",
                tier,
                price,
                best_ask,
            )
            return None

        self.risk.tracker.add_exposure(size_usd)
        size = size_usd / price

        # Run sync CLOB call in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        try:
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(
                        self.clob.create_and_post_order,
                        OrderArgs(
                            price=price,
                            size=size,
                            side=BUY,
                            token_id=token_id,
                        ),
                    ),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
        except TimeoutError:
            self.risk.tracker.remove_exposure(size_usd)
            self._breaker.record_failure()
            log.warning("%s maker order timed out after %.0fs", tier, _CLOB_CALL_TIMEOUT_SEC)
            return None
        except (OSError, ValueError, KeyError) as exc:
            self.risk.tracker.remove_exposure(size_usd)
            self._breaker.record_failure()
            log.warning("%s maker order failed: %s", tier, exc)
            return None

        order_id = resp.get("orderID", "")
        if not order_id:
            self.risk.tracker.remove_exposure(size_usd)
            log.warning("%s maker order rejected: %s", tier, resp)
            return None

        self._breaker.record_success()
        self.state.active_order_ids.append(order_id)
        self._order_details[order_id] = {
            "tier": tier,
            "price": price,
            "size_usd": size_usd,
            "token_id": token_id,
        }
        self._cached_balance_usd -= size_usd
        log.info(
            "%s maker order placed: id=%s price=%.2f size=%.2f", tier, order_id[:12], price, size
        )
        return str(order_id)

    async def place_taker_order(
        self,
        token_id: str,
        price: float,
        size_usd: float,
        tier: str,
    ) -> str | None:
        """Place a FOK taker order. Returns order ID or None."""
        assert self.risk.tracker is not None  # noqa: S101  # set in RiskRegistry.from_config()
        if not self._breaker.can_attempt():
            log.warning("%s blocked: CLOB circuit breaker OPEN", tier)
            return None

        if not self.risk.can_trade(size_usd):
            log.warning("%s blocked by risk limits", tier)
            return None

        # On-chain balance gate — block if balance unknown or insufficient
        if self._cached_balance_usd is None:
            log.warning("%s blocked: on-chain balance unknown (refresh not yet succeeded)", tier)
            return None
        if size_usd > self._cached_balance_usd:
            log.warning(
                "%s blocked: insufficient on-chain balance $%.2f < order $%.2f",
                tier,
                self._cached_balance_usd,
                size_usd,
            )
            return None

        self.risk.tracker.add_exposure(size_usd)
        size = size_usd / price

        loop = asyncio.get_running_loop()
        try:
            # FOK orders require MarketOrderArgs + two-step flow per py-clob-client docs
            market_args = MarketOrderArgs(
                token_id=token_id,
                amount=size_usd,
                price=price,
            )
            signed_order = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(self.clob.create_market_order, market_args),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(self.clob.post_order, signed_order, OrderType.FOK),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
        except TimeoutError:
            self.risk.tracker.remove_exposure(size_usd)
            self._breaker.record_failure()
            log.warning("%s taker order timed out after %.0fs", tier, _CLOB_CALL_TIMEOUT_SEC)
            return None
        except (OSError, ValueError, KeyError) as exc:
            self.risk.tracker.remove_exposure(size_usd)
            self._breaker.record_failure()
            log.warning("%s taker order failed: %s", tier, exc)
            return None

        order_id = resp.get("orderID", "")
        if not order_id:
            self.risk.tracker.remove_exposure(size_usd)
            log.warning("%s taker order rejected: %s", tier, resp)
            return None

        self._breaker.record_success()
        self.state.active_order_ids.append(order_id)
        self._cached_balance_usd -= size_usd
        log.info(
            "%s taker order placed: id=%s price=%.2f size=%.2f", tier, order_id[:12], price, size
        )
        return str(order_id)

    async def cancel_all_active(self) -> None:
        """Cancel all unfilled orders for the current window.

        Skips orders that have already been filled (detected via the CLOB
        user WebSocket and recorded in ``state.live_fills``).
        """
        if not self.state.active_order_ids or self._cancel_in_progress:
            return
        self._cancel_in_progress = True
        loop = asyncio.get_running_loop()
        try:
            successfully_cancelled: list[str] = []
            already_filled: list[str] = []
            for order_id in list(self.state.active_order_ids):
                # Skip orders that the WS already confirmed as filled
                if order_id in self.state.live_fills:
                    already_filled.append(order_id)
                    continue
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(self.clob.cancel, order_id),
                        ),
                        timeout=_CLOB_CALL_TIMEOUT_SEC,
                    )
                    successfully_cancelled.append(order_id)
                    log.info("cancelled order %s", order_id[:12])
                except TimeoutError:
                    log.warning(
                        "cancel timed out for %s after %.0fs", order_id[:12], _CLOB_CALL_TIMEOUT_SEC
                    )
                except (OSError, ValueError, KeyError) as exc:
                    # Cancel failure on a filled order is expected — check
                    # if a fill arrived while we were trying to cancel
                    if order_id in self.state.live_fills:
                        already_filled.append(order_id)
                        log.info("cancel failed for %s (already filled): %s", order_id[:12], exc)
                    else:
                        log.warning("cancel failed for %s: %s", order_id[:12], exc)
            for oid in successfully_cancelled:
                try:
                    self.state.active_order_ids.remove(oid)
                except ValueError:
                    pass
            # Don't remove filled orders from active_order_ids here —
            # the main loop uses them at window finalization
            remaining = [
                oid for oid in self.state.active_order_ids if oid not in self.state.live_fills
            ]
            if remaining:
                log.error(
                    "orders still active after cancel sweep: %s", [oid[:12] for oid in remaining]
                )
            if already_filled:
                log.info(
                    "skipped cancel for %d filled order(s): %s",
                    len(already_filled),
                    [oid[:12] for oid in already_filled],
                )
            if successfully_cancelled:
                order_infos = []
                for oid in successfully_cancelled:
                    detail = self._order_details.pop(oid, None)
                    if detail:
                        info: dict[str, Any] = {
                            "tier": detail["tier"],
                            "price": detail["price"],
                            "size_usd": detail["size_usd"],
                        }
                        tid = detail.get("token_id", "")
                        if tid == self.state.up_token_id:
                            info["side"] = "up"
                            if self.state.best_ask_up > 0:
                                info["best_ask"] = self.state.best_ask_up
                        elif tid == self.state.down_token_id:
                            info["side"] = "down"
                            if self.state.best_ask_down > 0:
                                info["best_ask"] = self.state.best_ask_down
                        order_infos.append(info)
                send_bet_cancelled(
                    mode="live",
                    count=len(successfully_cancelled),
                    reason="window close (not filled)",
                    orders=order_infos or None,
                )
            # Clean up details for filled orders
            for oid in already_filled:
                self._order_details.pop(oid, None)
        finally:
            self._cancel_in_progress = False
