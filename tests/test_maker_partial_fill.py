"""Tests for partial-fill escalation in _monitor_maker_entry (PR-C).

Post-mortem 2026-04-22 §5.2: the previous ``is_order_filled`` method
returned True on any non-zero fill, so a maker order that filled 5% of
intended size was treated as fully filled and the unfilled 95% was
silently forfeited.

These tests exercise the new three-way branch in ``_monitor_maker_entry``:
fully filled (finalize as today), zero fill after timeout (existing
full-size taker), and the new **partial** branch (cancel remainder,
book the partial, escalate a taker for the unfilled quantity).
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from config import ErosionConfig, SizingConfig
from fakes import (
    FakeOrderExecutor,
    make_market_state,
    make_rules_config,
    make_signal_config,
)
from strategy.kelly import AdjustedWinRateResult
from strategy.momentum_signal import MomentumSignalStrategy
from strategy.signal import Direction, Signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(bn_pct: float = 0.0008) -> Signal:
    """Signal with a moderate up-direction delta. Matches the existing pattern."""
    return Signal(
        delta_pct=0.0,
        direction=Direction.UP,
        feeds_agree=True,
        bn_direction_from_open_pct=bn_pct,
        cl_direction_from_open_pct=bn_pct,
        poly_spread_up=0.0,
        poly_spread_down=0.0,
        binance_obi_d5=0.0,
        binance_obi_d10=0.0,
        binance_obi_d20=0.0,
        time_remaining=200.0,
    )


def _build_maker_strategy(
    *, maker_timeout_s: float = 20.0
) -> tuple[MomentumSignalStrategy, FakeOrderExecutor]:
    """Build a strategy wired to take the maker path (skip_maker disabled)."""
    cfg = make_rules_config(
        skip_maker_min_oos_wr_pct=0.0,  # force maker path
        maker_timeout_s=maker_timeout_s,
    )
    state = make_market_state(
        best_ask_up=0.85,
        best_ask_down=0.15,
        best_bid_up=0.84,
        best_bid_down=0.14,
    )
    sc = make_signal_config(
        side=Direction.UP,
        observe_from_s=240.0,
        observe_to_s=180.0,
        min_delta_pct=0.05,
        max_variance_pct=1.0,
        oos_win_rate_pct=90.0,
    )
    strategy = MomentumSignalStrategy(cfg, state, sc)
    strategy.kelly_wr_result = AdjustedWinRateResult(
        adjusted_p=0.90,
        vol_discount=0,
        chop_discount=0,
        outcome_discount=0,
        total_discount=0,
        feedback_adjustment=0,
        regime_ready=True,
    )
    strategy.sizing_cfg = SizingConfig()
    strategy.erosion_cfg = ErosionConfig()
    strategy.bankroll = 1000.0
    executor = FakeOrderExecutor()
    return strategy, executor


async def _fire_maker(
    strategy: MomentumSignalStrategy,
    executor: FakeOrderExecutor,
) -> None:
    """Accumulate samples and trigger a fire that takes the maker path."""
    for i in range(10):
        sig = _make_signal()
        await strategy.evaluate(sig, 220.0 - i * 4, executor)
    sig = _make_signal()
    await strategy.evaluate(sig, 170.0, executor)
    # Sanity: one order was placed and it was a maker.
    assert len(executor.calls) == 1
    assert executor.calls[0].method == "place_maker_order"
    assert strategy._maker_order_id is not None


# ---------------------------------------------------------------------------
# Branch 1 — full fill: existing behaviour, uses actual filled_usd
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maker_fully_filled_finalizes_with_actual_fill() -> None:
    strategy, executor = _build_maker_strategy()
    await _fire_maker(strategy, executor)

    maker_id = strategy._maker_order_id
    assert maker_id is not None
    # Simulate a full fill: executor reports the order as filled.
    executor.filled_orders.add(maker_id)

    # Next tick dispatches to _monitor_maker_entry.
    await strategy.evaluate(_make_signal(), 150.0, executor)

    # _entry_complete set, last_size_usd reflects actual fill (= intent here
    # because full), no additional order placed.
    assert strategy._entry_complete is True
    assert strategy.last_size_usd == pytest.approx(strategy._maker_size_usd)
    # Only the maker order — no taker escalation.
    assert len(executor.calls) == 1


# ---------------------------------------------------------------------------
# Branch 2 — zero fill after timeout: full-size taker escalation (existing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maker_zero_fill_after_timeout_escalates_full_taker() -> None:
    strategy, executor = _build_maker_strategy(maker_timeout_s=0.001)
    await _fire_maker(strategy, executor)
    maker_intent = strategy._maker_size_usd

    # Wait past the tiny maker_timeout — no fills recorded.
    await asyncio.sleep(0.01)

    await strategy.evaluate(_make_signal(), 150.0, executor)

    # Maker cancelled, taker placed for the full intent.
    assert strategy._maker_order_id in executor.cancelled_orders
    assert len(executor.calls) == 2
    assert executor.calls[1].method == "place_taker_order"
    assert executor.calls[1].size_usd == pytest.approx(maker_intent)
    assert strategy._entry_complete is True
    # Entry type accounting: pure taker (no maker partial).
    assert strategy.last_size_usd == pytest.approx(maker_intent)
    assert strategy.last_entry_price == pytest.approx(0.85)  # best_ask_up


# ---------------------------------------------------------------------------
# Branch 3 — PARTIAL fill: new behaviour. Cancel remainder, finalize partial,
# escalate taker for the unfilled quantity only.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maker_partial_escalates_only_the_remainder() -> None:
    strategy, executor = _build_maker_strategy(maker_timeout_s=0.001)
    await _fire_maker(strategy, executor)

    maker_id = strategy._maker_order_id
    assert maker_id is not None
    maker_intent = strategy._maker_size_usd
    maker_price = strategy._maker_entry_price
    # 30% filled, 70% remaining — matches the thin-book pattern seen live.
    partial_usd = maker_intent * 0.30
    executor.set_partial_fill(maker_id, partial_usd)

    await asyncio.sleep(0.01)  # past the tiny maker_timeout
    await strategy.evaluate(_make_signal(), 150.0, executor)

    # Taker placed for EXACTLY the unfilled remainder, not the full intent.
    assert len(executor.calls) == 2
    assert executor.calls[1].method == "place_taker_order"
    expected_remainder = maker_intent - partial_usd
    assert executor.calls[1].size_usd == pytest.approx(expected_remainder, abs=0.01)

    # Maker was cancelled.
    assert maker_id in executor.cancelled_orders

    # Combined entry recorded: total USD = filled + remainder = intent.
    assert strategy._entry_complete is True
    assert strategy.last_size_usd == pytest.approx(maker_intent, abs=0.01)

    # Weighted-average entry price: (filled*maker_price + remainder*best_ask) / total.
    best_ask = 0.85
    expected_avg = (partial_usd * maker_price + expected_remainder * best_ask) / maker_intent
    assert strategy.last_entry_price == pytest.approx(expected_avg, abs=0.001)


@pytest.mark.asyncio
async def test_maker_partial_books_partial_only_when_remainder_below_min_bet() -> None:
    """If the unfilled remainder is below kelly_min_bet, just book the partial."""
    strategy, executor = _build_maker_strategy(maker_timeout_s=0.001)
    await _fire_maker(strategy, executor)

    maker_id = strategy._maker_order_id
    assert maker_id is not None
    maker_intent = strategy._maker_size_usd
    # Fill almost everything — leave only a sub-$1 tail that's below
    # kelly_min_bet=$1 default.
    partial_usd = maker_intent - 0.50
    executor.set_partial_fill(maker_id, partial_usd)

    await asyncio.sleep(0.01)
    await strategy.evaluate(_make_signal(), 150.0, executor)

    # No taker — remainder $0.50 < kelly_min_bet $1.0.
    assert len(executor.calls) == 1
    assert strategy._entry_complete is True
    # Only the partial is booked.
    assert strategy.last_size_usd == pytest.approx(partial_usd, abs=0.01)
    assert strategy.last_entry_price == pytest.approx(strategy._maker_entry_price)


@pytest.mark.asyncio
async def test_maker_partial_survives_taker_failure() -> None:
    """If the taker escalation fails, the maker partial is still booked."""
    strategy, executor = _build_maker_strategy(maker_timeout_s=0.001)
    await _fire_maker(strategy, executor)

    maker_id = strategy._maker_order_id
    assert maker_id is not None
    maker_intent = strategy._maker_size_usd
    partial_usd = maker_intent * 0.25
    executor.set_partial_fill(maker_id, partial_usd)

    # Arrange for the taker to fail by returning None on placement.
    # The executor places one maker (next_order_id=fake-order-1), then we flip
    # to None before it places the taker.
    executor.next_order_id = None

    await asyncio.sleep(0.01)
    await strategy.evaluate(_make_signal(), 150.0, executor)

    # Taker was attempted but returned None.
    assert len(executor.calls) == 2
    assert executor.calls[1].method == "place_taker_order"
    # Partial is still booked on the maker side.
    assert strategy._entry_complete is True
    assert strategy.last_size_usd == pytest.approx(partial_usd, abs=0.01)


# ---------------------------------------------------------------------------
# Log-hygiene: MAKER FILLED log reports actual filled_usd, not intended
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maker_filled_log_reports_actual_not_intended(caplog) -> None:
    import logging

    strategy, executor = _build_maker_strategy()
    await _fire_maker(strategy, executor)

    maker_id = strategy._maker_order_id
    assert maker_id is not None
    # "Fully filled" via the executor's binary filled_orders set — but the
    # fake's filled_usd reports the ORDER INTENT, which happens to equal
    # maker_size_usd in this happy path. The log should surface that value.
    executor.filled_orders.add(maker_id)

    caplog.set_level(logging.INFO, logger="strategy.momentum_signal")
    await strategy.evaluate(_make_signal(), 150.0, executor)

    msgs = [r.message for r in caplog.records]
    maker_filled_lines = [m for m in msgs if m.startswith("MAKER FILLED")]
    assert len(maker_filled_lines) == 1
    # The size in the log should match the actual filled_usd (= intent here).
    expected_size = executor.filled_usd(maker_id)
    assert f"size=${expected_size:.2f}" in maker_filled_lines[0]


@pytest.mark.asyncio
async def test_partial_escalation_emits_maker_partial_log(caplog) -> None:
    import logging

    strategy, executor = _build_maker_strategy(maker_timeout_s=0.001)
    await _fire_maker(strategy, executor)

    maker_id = strategy._maker_order_id
    assert maker_id is not None
    maker_intent = strategy._maker_size_usd
    partial_usd = maker_intent * 0.20
    executor.set_partial_fill(maker_id, partial_usd)

    caplog.set_level(logging.INFO, logger="strategy.momentum_signal")
    await asyncio.sleep(0.01)
    await strategy.evaluate(_make_signal(), 150.0, executor)

    msgs = [r.message for r in caplog.records]
    partial_lines = [m for m in msgs if m.startswith("MAKER PARTIAL")]
    assert len(partial_lines) == 1
    assert f"filled=${partial_usd:.2f}" in partial_lines[0]
    assert f"intended=${maker_intent:.2f}" in partial_lines[0]


# ---------------------------------------------------------------------------
# Discord notification — combined maker+taker entry carries the capital split
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_escalation_sends_discord_split() -> None:
    """The Discord bet-placed payload for a combined entry must include
    ``maker_usd`` and ``taker_usd`` so the embed can render the percent split.
    """
    strategy, executor = _build_maker_strategy(maker_timeout_s=0.001)
    await _fire_maker(strategy, executor)

    maker_id = strategy._maker_order_id
    assert maker_id is not None
    maker_intent = strategy._maker_size_usd
    partial_usd = maker_intent * 0.30
    executor.set_partial_fill(maker_id, partial_usd)

    with patch("strategy.momentum_signal.send_bet_placed") as mock_send:
        await asyncio.sleep(0.01)
        await strategy.evaluate(_make_signal(), 150.0, executor)

    # send_bet_placed was called once with both maker_usd and taker_usd > 0.
    assert mock_send.call_count == 1
    kwargs = mock_send.call_args.kwargs
    assert kwargs["entry_type"] == "maker+taker"
    expected_remainder = maker_intent - partial_usd
    assert kwargs["maker_usd"] == pytest.approx(partial_usd, abs=0.01)
    assert kwargs["taker_usd"] == pytest.approx(expected_remainder, abs=0.01)


@pytest.mark.asyncio
async def test_pure_maker_fill_sends_zero_split() -> None:
    """Pure maker fills must NOT include a split — the single-label form
    ("Maker Fill") is the correct presentation for uncombined entries.
    """
    strategy, executor = _build_maker_strategy()
    await _fire_maker(strategy, executor)

    maker_id = strategy._maker_order_id
    assert maker_id is not None
    executor.filled_orders.add(maker_id)

    with patch("strategy.momentum_signal.send_bet_placed") as mock_send:
        await strategy.evaluate(_make_signal(), 150.0, executor)

    assert mock_send.call_count == 1
    kwargs = mock_send.call_args.kwargs
    assert kwargs["entry_type"] == "maker"
    # Split params default to 0 on non-combined paths.
    assert kwargs["maker_usd"] == 0.0
    assert kwargs["taker_usd"] == 0.0


@pytest.mark.asyncio
async def test_pure_taker_escalation_sends_zero_split() -> None:
    """Zero-fill maker → full-size taker escalation is a pure taker entry,
    not a combined one — no split to render.
    """
    strategy, executor = _build_maker_strategy(maker_timeout_s=0.001)
    await _fire_maker(strategy, executor)

    with patch("strategy.momentum_signal.send_bet_placed") as mock_send:
        await asyncio.sleep(0.01)
        await strategy.evaluate(_make_signal(), 150.0, executor)

    assert mock_send.call_count == 1
    kwargs = mock_send.call_args.kwargs
    assert kwargs["entry_type"] == "taker"
    assert kwargs["maker_usd"] == 0.0
    assert kwargs["taker_usd"] == 0.0


# ---------------------------------------------------------------------------
# Discord embed rendering — Entry field formatting
# ---------------------------------------------------------------------------


def _extract_entry_field(embeds: list[dict]) -> str:
    """Pull the "Entry" field value out of a bet-placed embed payload."""
    for e in embeds:
        for f in e.get("fields", []):
            if f.get("name") == "Entry":
                return str(f.get("value", ""))
    return ""


def test_discord_entry_field_renders_split_for_combined_entry() -> None:
    """Combined maker+taker entry renders a percent split."""
    from shared.discord import send_bet_placed

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_bet_placed(
            mode="live",
            side="down",
            price=0.78,
            size_usd=3.12,
            rank=1,
            entry_type="maker+taker",
            maker_usd=0.17,
            taker_usd=2.95,
        )

    assert mock_send.call_count == 1
    embeds = mock_send.call_args.args[1]
    # 0.17 / 3.12 = 5.4 %, 2.95 / 3.12 = 94.6 % → rounded "5%" / "95%".
    assert _extract_entry_field(embeds) == "`Maker 5% / Taker 95%`"


def test_discord_entry_field_single_label_for_pure_maker() -> None:
    from shared.discord import send_bet_placed

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_bet_placed(
            mode="live",
            side="up",
            price=0.85,
            size_usd=3.12,
            rank=1,
            entry_type="maker",
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_entry_field(embeds) == "`Maker Fill`"


def test_discord_entry_field_single_label_for_pure_taker() -> None:
    from shared.discord import send_bet_placed

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_bet_placed(
            mode="live",
            side="up",
            price=0.85,
            size_usd=3.12,
            rank=1,
            entry_type="taker",
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_entry_field(embeds) == "`Taker`"
