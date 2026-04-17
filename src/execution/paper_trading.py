"""Paper trade interceptor — simulates order fills against live data."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from shared.discord import send_bet_cancelled, send_bet_result

if TYPE_CHECKING:
    from config import Config
    from market_data.state import MarketState, WindowSnapshot
    from risk.fee_tracker import FeeTracker
    from risk.registry import RiskRegistry
    from strategy.signal import Signal

log = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    order_id: str
    token_id: str
    price: float
    size: float
    size_usd: float
    tier: str
    is_maker: bool
    submitted_at: float = field(default_factory=time.time)
    filled: bool = False


@dataclass
class WindowRecord:
    window_ts: int = 0
    window_delta_pct: float = 0.0
    direction: str = ""
    rule_triggered: int | None = None
    rule_direction: str = ""
    rule_entry_price: float = 0.0
    rule_simulated_fill: bool = False
    rule_signal_features: dict[str, float] | None = None
    actual_outcome: str | None = None
    pnl_rules: float = 0.0
    pnl_total: float = 0.0
    latency_signal_ms: float | None = None
    latency_order_ms: float | None = None
    # Kelly fields
    kelly_adjusted_p: float | None = None
    kelly_vol_discount: float | None = None
    kelly_chop_discount: float | None = None
    kelly_outcome_discount: float | None = None
    kelly_total_discount: float | None = None
    kelly_feedback_adj: float | None = None
    kelly_raw_f: float | None = None
    kelly_fractional_f: float | None = None
    kelly_bet_size: float | None = None
    kelly_entry_price: float | None = None
    kelly_has_edge: bool | None = None
    bankroll_before: float | None = None
    bankroll_after: float | None = None
    sprt_factor: float | None = None
    final_bet_size: float | None = None
    early_exit: bool = False
    early_exit_sell_price: float | None = None


class PaperOrderManager:
    """Drop-in replacement for OrderManager that simulates fills."""

    mode: str = "paper"

    def __init__(
        self,
        cfg: Config,
        state: MarketState,
        risk: RiskRegistry,
        fee_tracker: FeeTracker,
        results_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.risk = risk
        self.fee_tracker = fee_tracker
        self._orders: list[PaperOrder] = []
        self._order_counter: int = 0
        self._current_record: WindowRecord = WindowRecord()
        self._tick_start: float = 0.0
        self._balance: float = cfg.paper.starting_balance_usd

        self._output_dir = results_dir or Path("data/paper/results")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def balance(self) -> float:
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        self._balance = value

    def record_tick_start(self, ts: float) -> None:
        """Called at the top of each strategy tick, before compute_signal()."""
        self._tick_start = ts

    def reset_window(self, window_ts: int) -> None:
        self._orders.clear()
        self._current_record = WindowRecord(window_ts=window_ts)

    def set_rule_triggered(self, rule_id: int, direction: str, signal: Signal | None) -> None:
        """Called by MomentumSignalStrategy after a successful order placement."""
        self._current_record.rule_triggered = rule_id
        self._current_record.rule_direction = direction
        if signal is not None:
            self._current_record.rule_signal_features = signal.as_feature_dict()

    def set_kelly_fields(
        self,
        kelly_adjusted_p: float | None,
        kelly_vol_discount: float | None,
        kelly_chop_discount: float | None,
        kelly_outcome_discount: float | None,
        kelly_total_discount: float | None,
        kelly_feedback_adj: float | None,
        kelly_raw_f: float | None,
        kelly_fractional_f: float | None,
        kelly_bet_size: float | None,
        kelly_entry_price: float | None,
        kelly_has_edge: bool | None,
        bankroll_before: float | None,
        sprt_factor: float | None,
        final_bet_size: float | None,
    ) -> None:
        """Populate Kelly Criterion fields on the current window record."""
        rec = self._current_record
        rec.kelly_adjusted_p = kelly_adjusted_p
        rec.kelly_vol_discount = kelly_vol_discount
        rec.kelly_chop_discount = kelly_chop_discount
        rec.kelly_outcome_discount = kelly_outcome_discount
        rec.kelly_total_discount = kelly_total_discount
        rec.kelly_feedback_adj = kelly_feedback_adj
        rec.kelly_raw_f = kelly_raw_f
        rec.kelly_fractional_f = kelly_fractional_f
        rec.kelly_bet_size = kelly_bet_size
        rec.kelly_entry_price = kelly_entry_price
        rec.kelly_has_edge = kelly_has_edge
        rec.bankroll_before = bankroll_before
        rec.sprt_factor = sprt_factor
        rec.final_bet_size = final_bet_size

    def is_order_filled(self, order_id: str) -> bool:
        """Check if a paper order has been filled."""
        return any(o.order_id == order_id and o.filled for o in self._orders)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a single paper order. Returns True if cancelled."""
        assert self.risk.tracker is not None  # noqa: S101  # set in RiskRegistry.from_config()
        for i, order in enumerate(self._orders):
            if order.order_id == order_id and not order.filled:
                self.risk.tracker.remove_exposure(order.size_usd)
                self._orders.pop(i)
                log.info("paper cancel: %s (maker escalation)", order_id)
                return True
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
        if not self.risk.can_trade(size_usd):
            log.warning("paper %s blocked by risk limits", tier)
            return None

        if size_usd > self._balance:
            log.warning(
                "paper %s blocked: insufficient balance %.2f < %.2f", tier, self._balance, size_usd
            )
            return None

        # Post-only safety: reject if our bid would cross the spread (same as live)
        best_ask = self._best_ask_for(token_id)
        if best_ask > 0 and price >= best_ask:
            log.warning(
                "paper %s maker rejected: price %.2f would cross ask %.2f", tier, price, best_ask
            )
            return None

        # NOTE: In paper mode, orders always "succeed" at creation, so exposure
        # rollback is not needed here. But we add it for consistency with live.
        assert self.risk.tracker is not None  # noqa: S101  # set in RiskRegistry.from_config()
        self.risk.tracker.add_exposure(size_usd)
        self._order_counter += 1
        order_id = f"paper-{self._order_counter}"
        size = size_usd / price

        order = PaperOrder(
            order_id=order_id,
            token_id=token_id,
            price=price,
            size=size,
            size_usd=size_usd,
            tier=tier,
            is_maker=True,
        )
        self._orders.append(order)

        # Delay enforced in _check_maker_fill — order cannot fill the same tick it was submitted
        filled = self._check_maker_fill(order)
        order.filled = filled

        self._current_record.rule_entry_price = price
        self._current_record.rule_simulated_fill = filled

        # Record latency for the first order this window
        if self._tick_start > 0 and self._current_record.latency_signal_ms is None:
            self._current_record.latency_signal_ms = (time.time() - self._tick_start) * 1000.0

        status = "filled" if filled else "resting"
        log.info(
            "paper %s maker: id=%s price=%.2f size=%.4f [%s]", tier, order_id, price, size, status
        )
        return order_id

    async def place_taker_order(
        self,
        token_id: str,
        price: float,
        size_usd: float,
        tier: str,
    ) -> str | None:
        """Simulate an immediate taker fill. Kept for future strategy use."""
        if not self.risk.can_trade(size_usd):
            log.warning("paper %s blocked by risk limits", tier)
            return None

        if size_usd > self._balance:
            log.warning(
                "paper %s blocked: insufficient balance %.2f < %.2f", tier, self._balance, size_usd
            )
            return None

        assert self.risk.tracker is not None  # noqa: S101  # set in RiskRegistry.from_config()
        self.risk.tracker.add_exposure(size_usd)
        self._order_counter += 1
        order_id = f"paper-{self._order_counter}"
        size = size_usd / price

        order = PaperOrder(
            order_id=order_id,
            token_id=token_id,
            price=price,
            size=size,
            size_usd=size_usd,
            tier=tier,
            is_maker=False,
            filled=True,
        )
        self._orders.append(order)

        self._current_record.rule_entry_price = price
        self._current_record.rule_simulated_fill = True

        if self._tick_start > 0 and self._current_record.latency_signal_ms is None:
            self._current_record.latency_signal_ms = (time.time() - self._tick_start) * 1000.0

        log.info("paper %s taker: id=%s price=%.2f size=%.4f [filled]", tier, order_id, price, size)
        return order_id

    async def cancel_all_active(self) -> None:
        unfilled = [o for o in self._orders if not o.filled]
        for order in unfilled:
            log.info("paper cancel: %s", order.order_id)
        if unfilled:
            order_details = []
            for o in unfilled:
                detail: dict[str, Any] = {
                    "tier": o.tier,
                    "price": o.price,
                    "size_usd": o.size_usd,
                }
                # Include current best ask for the relevant side
                if o.token_id == self.state.up_token_id and self.state.best_ask_up > 0:
                    detail["side"] = "up"
                    detail["best_ask"] = self.state.best_ask_up
                elif o.token_id == self.state.down_token_id and self.state.best_ask_down > 0:
                    detail["side"] = "down"
                    detail["best_ask"] = self.state.best_ask_down
                order_details.append(detail)
            send_bet_cancelled(
                mode="paper",
                count=len(unfilled),
                reason="window close (not filled)",
                orders=order_details,
            )
        self._orders = [o for o in self._orders if o.filled]

    def _check_maker_fill(self, order: PaperOrder) -> bool:
        """Simulate maker fill with a minimum resting delay.

        Fill condition: best_ask <= order.price AND the order has been resting
        for at least simulated_fill_delay_sec. This prevents the overly-optimistic
        scenario where an order fills in the same tick it was submitted.
        Queue depth is not modelled, but the delay provides a conservative proxy.
        """
        delay = self.cfg.paper.simulated_fill_delay_sec
        if time.time() - order.submitted_at < delay:
            return False
        if order.token_id == self.state.up_token_id:
            return self.state.best_ask_up > 0 and self.state.best_ask_up <= order.price
        if order.token_id == self.state.down_token_id:
            return self.state.best_ask_down > 0 and self.state.best_ask_down <= order.price
        return False

    def check_resting_fills(self) -> None:
        """Check all unfilled maker orders against current orderbook."""
        for order in [o for o in self._orders if not o.filled and o.is_maker]:
            if self._check_maker_fill(order):
                order.filled = True
                log.info(
                    "paper %s maker fill: id=%s price=%.2f", order.tier, order.order_id, order.price
                )
                self._current_record.rule_simulated_fill = True

    async def exit_position_early(self, sell_price: float) -> float | None:
        """Simulate selling the position at the given bid price.

        Marks the current window as an early exit so that finalize_window
        uses the sell P&L instead of binary resolution P&L.
        Returns the realized P&L, or None if no filled orders existed.
        """
        filled_orders = [o for o in self._orders if o.filled]
        if not filled_orders:
            log.info("exit_position_early: no filled orders to sell")
            return None

        pnl = 0.0
        total_size = 0.0
        for order in filled_orders:
            # P&L from selling tokens at bid: (sell - buy) * quantity
            pnl += order.size * (sell_price - order.price)
            total_size += order.size

        # Exit is always a taker action — the bid cross that fills the SELL pays
        # a taker fee regardless of whether the entry was maker or taker.
        fee = self.fee_tracker.record_taker_fee(sell_price, total_size)
        pnl -= fee

        pnl = round(pnl, 4)
        rec = self._current_record
        rec.early_exit = True
        rec.early_exit_sell_price = sell_price
        rec.pnl_rules = pnl
        rec.pnl_total = pnl
        self._balance += pnl

        log.info(
            "EARLY EXIT: sell_price=%.2f size=%.2f pnl=%.4f balance=$%.2f orders=%d",
            sell_price,
            total_size,
            pnl,
            self._balance,
            len(filled_orders),
        )
        return pnl

    def finalize_window(
        self,
        delta_pct: float,
        direction: str,
        outcome: str | None,
        snapshot: WindowSnapshot | None = None,
    ) -> WindowRecord:
        """Calculate P&L and write the window record to disk."""
        rec = self._current_record
        rec.window_delta_pct = delta_pct
        rec.direction = direction

        if outcome:
            rec.actual_outcome = outcome
            if not rec.early_exit:
                rec.pnl_rules = self._calc_pnl(outcome, snapshot)
                rec.pnl_total = rec.pnl_rules
                self._balance += rec.pnl_total

        # Compute bankroll_after from bankroll_before + pnl
        if rec.bankroll_before is not None and rec.rule_simulated_fill:
            rec.bankroll_after = round(rec.bankroll_before + rec.pnl_total, 4)

        self._write_record(rec, snapshot)
        return rec

    def _calc_pnl(self, outcome: str, snapshot: WindowSnapshot | None) -> float:
        filled_orders = [o for o in self._orders if o.filled]
        if not filled_orders:
            return 0.0

        up_id = snapshot.up_token_id if snapshot else self.state.up_token_id
        down_id = snapshot.down_token_id if snapshot else self.state.down_token_id

        pnl = 0.0
        for order in filled_orders:
            won = (order.token_id == up_id and outcome == "up") or (
                order.token_id == down_id and outcome == "down"
            )
            if won:
                pnl += order.size * (1.0 - order.price)
            else:
                pnl -= order.size * order.price

            if not order.is_maker:
                fee = self.fee_tracker.record_taker_fee(order.price, order.size)
                pnl -= fee

        return round(pnl, 4)

    def _write_record(self, rec: WindowRecord, snapshot: WindowSnapshot | None = None) -> None:
        from datetime import datetime

        date_str = datetime.fromtimestamp(rec.window_ts, tz=UTC).strftime("%Y-%m-%d")
        path = self._output_dir / f"{date_str}.jsonl"

        data = {
            "window_ts": rec.window_ts,
            "window_delta_pct": rec.window_delta_pct,
            "direction": rec.direction,
            "rule_triggered": rec.rule_triggered,
            "rule_direction": rec.rule_direction,
            "rule_entry_price": rec.rule_entry_price,
            "rule_simulated_fill": rec.rule_simulated_fill,
            "rule_signal_features": rec.rule_signal_features,
            "actual_outcome": rec.actual_outcome,
            "pnl_rules": rec.pnl_rules,
            "pnl_total": rec.pnl_total,
            "latency_signal_ms": rec.latency_signal_ms,
            "latency_order_ms": rec.latency_order_ms,
            "balance_usd": round(self._balance, 4),
            "kelly_adjusted_p": rec.kelly_adjusted_p,
            "kelly_vol_discount": rec.kelly_vol_discount,
            "kelly_chop_discount": rec.kelly_chop_discount,
            "kelly_outcome_discount": rec.kelly_outcome_discount,
            "kelly_total_discount": rec.kelly_total_discount,
            "kelly_feedback_adj": rec.kelly_feedback_adj,
            "kelly_raw_f": rec.kelly_raw_f,
            "kelly_fractional_f": rec.kelly_fractional_f,
            "kelly_bet_size": rec.kelly_bet_size,
            "kelly_entry_price": rec.kelly_entry_price,
            "kelly_has_edge": rec.kelly_has_edge,
            "bankroll_before": rec.bankroll_before,
            "bankroll_after": rec.bankroll_after,
            "sprt_factor": rec.sprt_factor,
            "final_bet_size": rec.final_bet_size,
            "early_exit": rec.early_exit,
            "early_exit_sell_price": rec.early_exit_sell_price,
        }

        with open(path, "ab") as f:
            f.write(orjson.dumps(data))
            f.write(b"\n")

        if not rec.rule_simulated_fill:
            label = "SKIP"
        elif rec.pnl_total > 0:
            label = "WIN"
        elif rec.pnl_total < 0:
            label = "LOSS"
        else:
            label = "FLAT"

        if snapshot:
            bid_up, ask_up = snapshot.best_bid_up, snapshot.best_ask_up
            bid_dn, ask_dn = snapshot.best_bid_down, snapshot.best_ask_down
        else:
            bid_up, ask_up = self.state.best_bid_up, self.state.best_ask_up
            bid_dn, ask_dn = self.state.best_bid_down, self.state.best_ask_down

        rule_str = (
            f"rule#{rec.rule_triggered}({'fill' if rec.rule_simulated_fill else 'rest'})"
            if rec.rule_triggered
            else "-"
        )
        log.info(
            "paper window_ts=%d [%s] pnl=%.4f direction=%s outcome=%s "
            "%s balance=$%.2f bid_up=%.2f ask_up=%.2f bid_dn=%.2f ask_dn=%.2f",
            rec.window_ts,
            label,
            rec.pnl_total,
            rec.direction,
            rec.actual_outcome,
            rule_str,
            self._balance,
            bid_up,
            ask_up,
            bid_dn,
            ask_dn,
        )

        if label in ("WIN", "LOSS"):
            send_bet_result(
                mode="paper",
                outcome=label,
                pnl=rec.pnl_total,
                entry_price=rec.rule_entry_price,
                side=rec.rule_direction or rec.direction,
                size_usd=rec.final_bet_size or 0.0,
                balance=self._balance,
                market_outcome=rec.actual_outcome,
            )
