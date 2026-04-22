"""Place, cancel, and track orders via py-clob-client."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

from py_clob_client.clob_types import (  # type: ignore[import-untyped]  # no stubs available
    AssetType,
    BalanceAllowanceParams,
    MarketOrderArgs,
    OrderArgs,
    OrderType,
)
from py_clob_client.exceptions import (  # type: ignore[import-untyped]  # no stubs available
    PolyApiException,
)
from py_clob_client.order_builder.constants import (  # type: ignore[import-untyped]  # no stubs available
    BUY,
    SELL,
)

from shared.discord import send_bet_cancelled
from utils.circuit_breaker import CircuitBreaker

if TYPE_CHECKING:
    from py_clob_client.client import (  # type: ignore[import-untyped]  # no stubs available
        ClobClient,
    )

    from config import Config
    from execution.base import KellyTelemetrySnapshot
    from market_data.state import MarketState
    from risk.fee_tracker import FeeTracker
    from risk.registry import RiskRegistry
    from strategy.signal import Signal

log = logging.getLogger(__name__)

# Timeout for synchronous CLOB API calls run via executor.
# Prevents hung HTTP requests from blocking the event loop indefinitely.
_CLOB_CALL_TIMEOUT_SEC = 10.0

# Dedicated thread-pool size for blocking py-clob-client HTTP calls. Sized so a
# burst (cancel sweep + balance refresh + order placement) cannot starve the
# default executor that aiohttp / other run_in_executor consumers share.
_CLOB_EXECUTOR_WORKERS = 8


def _parse_fak_filled_shares(resp: dict[str, Any]) -> float:
    """Extract filled share count from a CLOB post_order FAK response.

    py-clob-client field names vary across versions; try the known variants
    in preference order. For a SELL FAK, ``takingAmount`` is the shares
    consumed from the book (the amount we received/gave up from our side).
    Returns 0.0 if nothing was filled or the response is malformed.
    """
    for key in ("takingAmount", "taking_amount", "size_matched", "sizeMatched"):
        v = resp.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


class OrderManager:
    mode: str = "live"

    def __init__(
        self,
        cfg: Config,
        state: MarketState,
        clob: ClobClient,
        risk: RiskRegistry,
        fee_tracker: FeeTracker,
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.clob = clob
        self.risk = risk
        self.fee_tracker = fee_tracker
        # Dedicated thread pool for blocking py-clob-client HTTP calls so
        # they never contend with the default run_in_executor pool used by
        # aiohttp and other stdlib consumers.
        self._clob_exec = ThreadPoolExecutor(
            max_workers=_CLOB_EXECUTOR_WORKERS,
            thread_name_prefix="clob",
        )
        self._cancel_in_progress: bool = False
        self._cached_balance_usd: float | None = None
        # Track placed order details for cancel notifications
        self._order_details: dict[str, dict[str, Any]] = {}
        # Circuit breaker — stops hammering CLOB API after consecutive failures
        self._breaker = CircuitBreaker("clob_orders", failure_threshold=3, cooldown_sec=60.0)
        # Per-window Kelly telemetry capture. PaperOrderManager stores these on
        # WindowRecord; live stores here and ResolutionManager reads the snapshot
        # at resolve time so the live JSONL matches paper's schema.
        self._kelly_telemetry: KellyTelemetrySnapshot = {}
        # Early-exit realized state. Set by exit_position_early when a live
        # SELL fills; read by window_handler to short-circuit the Gamma-based
        # resolution path (the trade is already realized — no need to wait).
        # ``residual_shares`` is non-zero when the FAK partially filled — the
        # unfilled portion is still on-chain and must be reconciled at Gamma
        # resolution with the canonical $1/$0 outcome.
        self._early_exit_sell_price: float | None = None
        self._early_exit_pnl: float | None = None
        self._early_exit_residual_shares: float = 0.0
        self._early_exit_residual_entry: float = 0.0

    def set_rule_triggered(
        self,
        rule_id: int,
        direction: str,
        signal: Signal | None,
        obi_threshold: float = 0.0,
        obi_depth: str = "none",
    ) -> None:
        self._kelly_telemetry["rule_triggered"] = rule_id
        self._kelly_telemetry["rule_direction"] = direction
        self._kelly_telemetry["rule_obi_threshold"] = obi_threshold
        self._kelly_telemetry["rule_obi_depth"] = obi_depth
        if signal is not None:
            self._kelly_telemetry["rule_signal_features"] = signal.as_feature_dict()

    def set_kelly_fields(
        self,
        kelly_adjusted_p: float | None = None,
        kelly_vol_discount: float | None = None,
        kelly_chop_discount: float | None = None,
        kelly_outcome_discount: float | None = None,
        kelly_total_discount: float | None = None,
        kelly_feedback_adj: float | None = None,
        kelly_raw_f: float | None = None,
        kelly_fractional_f: float | None = None,
        kelly_bet_size: float | None = None,
        kelly_entry_price: float | None = None,
        kelly_has_edge: bool | None = None,
        bankroll_before: float | None = None,
        sprt_factor: float | None = None,
        final_bet_size: float | None = None,
    ) -> None:
        t = self._kelly_telemetry
        t["kelly_adjusted_p"] = kelly_adjusted_p
        t["kelly_vol_discount"] = kelly_vol_discount
        t["kelly_chop_discount"] = kelly_chop_discount
        t["kelly_outcome_discount"] = kelly_outcome_discount
        t["kelly_total_discount"] = kelly_total_discount
        t["kelly_feedback_adj"] = kelly_feedback_adj
        t["kelly_raw_f"] = kelly_raw_f
        t["kelly_fractional_f"] = kelly_fractional_f
        t["kelly_bet_size"] = kelly_bet_size
        t["kelly_entry_price"] = kelly_entry_price
        t["kelly_has_edge"] = kelly_has_edge
        t["bankroll_before"] = bankroll_before
        t["sprt_factor"] = sprt_factor
        t["final_bet_size"] = final_bet_size

    def take_kelly_telemetry(self) -> KellyTelemetrySnapshot:
        """Return and clear the current window's Kelly telemetry capture."""
        snap = self._kelly_telemetry
        self._kelly_telemetry = {}
        return snap

    def close(self) -> None:
        """Shut down the dedicated CLOB thread pool.

        Called from ``main.run()``'s finally block during bot teardown so
        lingering HTTP workers don't keep the process alive after task
        cancellation. ``wait=False`` is intentional: any in-flight CLOB call
        is already protected by ``asyncio.wait_for`` with
        ``_CLOB_CALL_TIMEOUT_SEC``, so blocking shutdown on them serves no
        purpose at Ctrl+C time.
        """
        self._clob_exec.shutdown(wait=False, cancel_futures=True)

    async def exit_position_early(self, sell_price: float) -> float | None:
        """Place a live SELL taker order to exit the current filled position.

        Mirrors PaperOrderManager.exit_position_early: sums filled BUY fills
        from ``state.live_fills``, places a FAK SELL against ``sell_price`` on
        the same token, computes realized P&L (including taker fee), and
        stores the result so window-close bookkeeping can short-circuit the
        Gamma-based resolution path. Returns the realized P&L or None if no
        position existed or the CLOB SELL failed.
        """
        assert self.risk.tracker is not None  # noqa: S101  # set in RiskRegistry.from_config()
        filled_buys = [f for f in self.state.live_fills.values() if f.side == "BUY"]
        if not filled_buys:
            log.info("live exit_position_early: no filled BUY fills to sell")
            return None

        token_id = filled_buys[0].token_id
        # Guard against multi-token positions — our bot only buys one direction
        # per window, so mixed tokens would indicate a bug upstream.
        if any(f.token_id != token_id for f in filled_buys):
            log.error("live exit_position_early: mixed token_ids in live_fills — aborting sell")
            return None

        total_shares = sum(f.size for f in filled_buys)
        total_cost = sum(f.price * f.size for f in filled_buys)
        if total_shares <= 0:
            log.warning("live exit_position_early: total_shares <= 0, skipping")
            return None
        avg_entry = total_cost / total_shares

        if not self._breaker.can_attempt():
            log.warning("early exit blocked: CLOB circuit breaker OPEN")
            return None

        loop = asyncio.get_running_loop()
        try:
            # FAK SELL at the specified bid. OrderArgs side=SELL + OrderType.FAK
            # fills whatever the book supports at that price and kills the rest
            # — the taker equivalent of a buy-side FOK for sells.
            signed_order = await asyncio.wait_for(
                loop.run_in_executor(
                    self._clob_exec,
                    partial(
                        self.clob.create_order,
                        OrderArgs(
                            price=sell_price,
                            size=total_shares,
                            side=SELL,
                            token_id=token_id,
                        ),
                    ),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    self._clob_exec,
                    partial(self.clob.post_order, signed_order, OrderType.FAK),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
        except TimeoutError:
            self._breaker.record_failure()
            log.warning("early exit SELL timed out after %.0fs", _CLOB_CALL_TIMEOUT_SEC)
            return None
        except (OSError, ValueError, KeyError, PolyApiException) as exc:
            self._breaker.record_failure()
            log.warning("early exit SELL failed: %s", exc)
            return None

        order_id = resp.get("orderID", "")
        if not order_id:
            log.warning("early exit SELL rejected: %s", resp)
            return None

        self._breaker.record_success()
        filled_shares = _parse_fak_filled_shares(resp)
        if filled_shares <= 0:
            # FAK accepted but matched nothing (book moved away between quote
            # and post). Treat as no early-exit — leave the position open and
            # let normal Gamma resolution settle at $1/$0.
            log.warning(
                "early exit FAK matched no size: sell_price=%.4f shares_requested=%.2f resp=%s",
                sell_price,
                total_shares,
                resp,
            )
            return None

        # Clamp over-reported fills to the requested size (defensive — FAK
        # cannot fill more than the requested quantity, but keeps residual ≥ 0
        # if an off-by-rounding response comes back).
        filled_shares = min(filled_shares, total_shares)
        residual_shares = total_shares - filled_shares
        is_partial = residual_shares > 1e-6
        if not is_partial:
            filled_shares = total_shares
            residual_shares = 0.0

        taker_fee = self.fee_tracker.record_taker_fee(sell_price, filled_shares)
        pnl = round(filled_shares * (sell_price - avg_entry) - taker_fee, 4)
        self._early_exit_sell_price = sell_price
        self._early_exit_pnl = pnl
        self._early_exit_residual_shares = residual_shares
        self._early_exit_residual_entry = avg_entry if is_partial else 0.0

        if is_partial:
            log.warning(
                "LIVE EARLY EXIT PARTIAL: sell=%.4f filled=%.2f/%.2f residual=%.2f "
                "avg_entry=%.4f taker_fee=$%.4f realized_pnl=$%.4f order_id=%s "
                "(residual settles at Gamma resolution)",
                sell_price,
                filled_shares,
                total_shares,
                residual_shares,
                avg_entry,
                taker_fee,
                pnl,
                str(order_id)[:12],
            )
        else:
            log.info(
                "LIVE EARLY EXIT: sell_price=%.4f shares=%.2f avg_entry=%.4f "
                "taker_fee=$%.4f pnl=$%.4f order_id=%s",
                sell_price,
                total_shares,
                avg_entry,
                taker_fee,
                pnl,
                str(order_id)[:12],
            )
        return pnl

    def take_early_exit(self) -> tuple[float, float, float, float] | None:
        """Return (sell_price, realized_pnl, residual_shares, residual_entry).

        Called by window_handler at window-close to detect whether an early
        exit occurred this window. ``residual_shares > 0`` means the FAK
        partially filled and the unfilled portion must be resolved through
        Gamma; callers thread it into PendingResolution so _resolve combines
        realized early-exit P&L with residual $1/$0 outcome.
        """
        if self._early_exit_sell_price is None or self._early_exit_pnl is None:
            return None
        result = (
            self._early_exit_sell_price,
            self._early_exit_pnl,
            self._early_exit_residual_shares,
            self._early_exit_residual_entry,
        )
        self._early_exit_sell_price = None
        self._early_exit_pnl = None
        self._early_exit_residual_shares = 0.0
        self._early_exit_residual_entry = 0.0
        return result

    async def refresh_balance(self) -> float | None:
        """Fetch on-chain USDC balance from CLOB API and cache it.

        Called once per window boundary to keep the cached value fresh.
        Returns balance in USD or None on failure.
        """
        loop = asyncio.get_running_loop()
        try:
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    self._clob_exec,
                    lambda: self.clob.get_balance_allowance(
                        BalanceAllowanceParams(
                            asset_type=AssetType.COLLATERAL,
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
        except (OSError, ValueError, KeyError, PolyApiException) as exc:
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
                    self._clob_exec,
                    partial(self.clob.cancel, order_id),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
            try:
                self.state.active_order_ids.remove(order_id)
            except ValueError:
                pass
            self.state.maker_order_ids.discard(order_id)
            detail = self._order_details.pop(order_id, None)
            if detail:
                self.risk.tracker.remove_exposure(detail["size_usd"])
                # Refund reserved balance: place_maker/place_taker deducted
                # size_usd from _cached_balance_usd at order time. Without this
                # refund, maker→taker escalation within a window can falsely
                # trip the balance gate before the next on-chain refresh.
                if self._cached_balance_usd is not None:
                    self._cached_balance_usd += detail["size_usd"]
            log.info("cancelled order %s", order_id[:12])
            return True
        except TimeoutError:
            log.warning("cancel_order timed out for %s", order_id[:12])
            return False
        except (OSError, ValueError, KeyError, PolyApiException) as exc:
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
                    self._clob_exec,
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
        except (OSError, ValueError, KeyError, PolyApiException) as exc:
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
        self.state.maker_order_ids.add(order_id)
        self._order_details[order_id] = {
            "tier": tier,
            "price": price,
            "size_usd": size_usd,
            "token_id": token_id,
            "is_maker": True,
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

        # Commit flag + finally block guarantee exposure is rolled back on any
        # exit path that does not successfully place an order — including an
        # unanticipated exception escaping the narrow excepts below. Without
        # this, a TypeError from a py-clob-client constructor mismatch (see
        # post-mortem 2026-04-22 §5.3) leaks exposure indefinitely.
        committed = False
        loop = asyncio.get_running_loop()
        try:
            # FOK orders require MarketOrderArgs + two-step flow per py-clob-client docs.
            # ``side=BUY`` is mandatory in py_clob_client.clob_types.MarketOrderArgs;
            # omitting it raises TypeError at construction (post-mortem 2026-04-22 §5.1).
            # Every momentum entry is a BUY of the chosen outcome token; SELL exits go
            # through ``exit_position_early`` using OrderArgs, not this path.
            market_args = MarketOrderArgs(
                token_id=token_id,
                side=BUY,
                amount=size_usd,
                price=price,
            )
            signed_order = await asyncio.wait_for(
                loop.run_in_executor(
                    self._clob_exec,
                    partial(self.clob.create_market_order, market_args),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    self._clob_exec,
                    partial(self.clob.post_order, signed_order, OrderType.FOK),
                ),
                timeout=_CLOB_CALL_TIMEOUT_SEC,
            )
        except TimeoutError:
            self._breaker.record_failure()
            log.warning("%s taker order timed out after %.0fs", tier, _CLOB_CALL_TIMEOUT_SEC)
            return None
        except (OSError, ValueError, KeyError, PolyApiException) as exc:
            self._breaker.record_failure()
            log.warning("%s taker order failed: %s", tier, exc)
            return None
        else:
            order_id = resp.get("orderID", "")
            if not order_id:
                log.warning("%s taker order rejected: %s", tier, resp)
                return None

            self._breaker.record_success()
            self.state.active_order_ids.append(order_id)
            self._order_details[order_id] = {
                "tier": tier,
                "price": price,
                "size_usd": size_usd,
                "token_id": token_id,
                "is_maker": False,
            }
            self._cached_balance_usd -= size_usd
            committed = True
            log.info(
                "%s taker order placed: id=%s price=%.2f size=%.2f",
                tier,
                order_id[:12],
                price,
                size,
            )
            return str(order_id)
        finally:
            if not committed:
                self.risk.tracker.remove_exposure(size_usd)

    async def cancel_all_active(self) -> None:
        """Cancel all unfilled orders for the current window.

        Skips orders that have already been filled (detected via the CLOB
        user WebSocket and recorded in ``state.live_fills``).
        """
        assert self.risk.tracker is not None  # noqa: S101  # set in RiskRegistry.from_config()
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
                            self._clob_exec,
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
                except (OSError, ValueError, KeyError, PolyApiException) as exc:
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
                self.state.maker_order_ids.discard(oid)
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
            order_infos: list[dict[str, Any]] = []
            for oid in successfully_cancelled:
                detail = self._order_details.pop(oid, None)
                if not detail:
                    continue
                # Release reserved risk capacity and refund the cached on-chain
                # balance debit from place_*_order. Without this, exposure and
                # the balance gate leak until the next refresh_balance tick.
                self.risk.tracker.remove_exposure(detail["size_usd"])
                if self._cached_balance_usd is not None:
                    self._cached_balance_usd += detail["size_usd"]
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
            if successfully_cancelled:
                send_bet_cancelled(
                    mode="live",
                    count=len(successfully_cancelled),
                    reason="window close (not filled)",
                    orders=order_infos or None,
                )
            # Clean up details for filled orders — exposure/balance for these
            # is reconciled at resolution via _reconcile_bankroll, so we only
            # drop the bookkeeping here.
            for oid in already_filled:
                self._order_details.pop(oid, None)
        finally:
            self._cancel_in_progress = False
