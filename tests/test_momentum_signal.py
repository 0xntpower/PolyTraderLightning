"""Tests for MomentumSignalStrategy — signal evaluation, conditions, order placement."""

from __future__ import annotations

import pytest

from config import ErosionConfig, SizingConfig
from fakes import (
    FakeOrderExecutor,
    make_market_state,
    make_rules_config,
    make_signal_config,
)
from strategy.kelly import AdjustedWinRateResult
from strategy.momentum_signal import MomentumSignalConfig, MomentumSignalStrategy
from strategy.signal import Direction, Signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    side: Direction = Direction.UP,
    observe_from_s: float = 240.0,
    observe_to_s: float = 180.0,
    min_delta_pct: float = 0.05,
    max_variance_pct: float = 0.10,
    **state_overrides,
) -> tuple[MomentumSignalStrategy, FakeOrderExecutor]:
    """Build a strategy + fake executor with sensible defaults."""
    cfg = make_rules_config()
    state = make_market_state(**state_overrides)
    sc = make_signal_config(
        side=side,
        observe_from_s=observe_from_s,
        observe_to_s=observe_to_s,
        min_delta_pct=min_delta_pct,
        max_variance_pct=max_variance_pct,
    )
    strategy = MomentumSignalStrategy(cfg, state, sc)
    executor = FakeOrderExecutor()
    return strategy, executor


def _make_signal(
    delta_pct: float = 0.0,
    direction: Direction = Direction.NONE,
    bn_direction_from_open_pct: float = 0.0,
) -> Signal:
    return Signal(
        delta_pct=delta_pct,
        direction=direction,
        feeds_agree=True,
        bn_direction_from_open_pct=bn_direction_from_open_pct,
        cl_direction_from_open_pct=bn_direction_from_open_pct,
        poly_spread_up=0.0,
        poly_spread_down=0.0,
        binance_obi=0.0,
        time_remaining=200.0,
    )


# ---------------------------------------------------------------------------
# MomentumSignalConfig tests
# ---------------------------------------------------------------------------


class TestMomentumSignalConfig:
    def test_signal_id_deterministic(self):
        sc = make_signal_config(side=Direction.UP, observe_from_s=240.0, observe_to_s=180.0)
        assert sc.signal_id == "up_240.0_180.0_0.05_0.1"

    def test_invalid_observe_window_raises(self):
        with pytest.raises(ValueError, match="observe_from_s"):
            make_signal_config(observe_from_s=100, observe_to_s=200)

    def test_conservative_p_uses_wilson(self):
        sc = make_signal_config(oos_win_rate_pct=90.0, oos_matches=30)
        p = sc.conservative_p()
        # Wilson lower bound should reduce from 0.90
        assert 0.85 < p < 0.90

    def test_conservative_p_prefers_engine_value(self):
        sc = MomentumSignalConfig(
            rank=1,
            side=Direction.UP,
            observe_from_s=240,
            observe_to_s=180,
            min_delta_pct=0.05,
            max_variance_pct=0.10,
            train_win_rate_pct=92.0,
            oos_win_rate_pct=90.0,
            bh_adjusted_p_value=0.001,
            oos_matches=30,
            conservative_win_rate_pct=87.5,
        )
        assert sc.conservative_p() == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# Accumulation + conditions tests
# ---------------------------------------------------------------------------


class TestConditionsMet:
    def test_insufficient_ticks(self):
        """Need at least 5 samples to fire."""
        strategy, _ = _make_strategy()
        # Add only 3 samples
        for _ in range(3):
            strategy._accumulate(0.10, 200.0, 0.0)
        assert not strategy._conditions_met()

    def test_up_signal_fires_when_delta_above_threshold(self):
        strategy, _ = _make_strategy(side=Direction.UP, min_delta_pct=0.05)
        for i in range(10):
            strategy._accumulate(0.08, 230.0 - i, 0.0)
        assert strategy._conditions_met()

    def test_up_signal_skips_when_delta_below_threshold(self):
        strategy, _ = _make_strategy(side=Direction.UP, min_delta_pct=0.05)
        for i in range(10):
            strategy._accumulate(0.03, 230.0 - i, 0.0)
        assert not strategy._conditions_met()

    def test_down_signal_fires_when_delta_negative(self):
        strategy, _ = _make_strategy(side=Direction.DOWN, min_delta_pct=0.05)
        for i in range(10):
            strategy._accumulate(-0.08, 230.0 - i, 0.0)
        assert strategy._conditions_met()

    def test_down_signal_skips_when_delta_positive(self):
        strategy, _ = _make_strategy(side=Direction.DOWN, min_delta_pct=0.05)
        for i in range(10):
            strategy._accumulate(0.08, 230.0 - i, 0.0)
        assert not strategy._conditions_met()

    def test_high_variance_blocks_fire(self):
        strategy, _ = _make_strategy(max_variance_pct=0.05)
        # Wildly varying samples → high stddev
        values = [0.20, -0.15, 0.25, -0.10, 0.30, -0.05, 0.35, 0.10, 0.08, 0.15]
        for i, v in enumerate(values):
            strategy._accumulate(v, 230.0 - i, 0.0)
        assert not strategy._conditions_met()

    def test_population_stddev_calculation(self):
        strategy, _ = _make_strategy()
        # Known values: [2, 4, 4, 4, 5, 5, 7, 9]
        # Population stddev = sqrt(32/8) = 2.0
        # time_remaining decreases by 1.0s per call so each sample passes the
        # default variance_subsample_interval_s=1.0 gate.
        for i, v in enumerate([2, 4, 4, 4, 5, 5, 7, 9]):
            strategy._accumulate(float(v), 200.0 - i, 0.0)
        assert strategy._population_stddev() == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# Evaluate flow tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        cfg = make_rules_config(enabled=False)
        state = make_market_state()
        sc = make_signal_config()
        strategy = MomentumSignalStrategy(cfg, state, sc)
        executor = FakeOrderExecutor()
        await strategy.evaluate(_make_signal(), 200.0, executor)
        assert len(executor.calls) == 0

    @pytest.mark.asyncio
    async def test_skips_when_entry_complete(self):
        strategy, executor = _make_strategy()
        strategy._entry_complete = True
        await strategy.evaluate(_make_signal(), 200.0, executor)
        assert len(executor.calls) == 0

    @pytest.mark.asyncio
    async def test_skips_before_observation_window(self):
        """time_remaining > observe_from_s → haven't entered window yet."""
        strategy, executor = _make_strategy(observe_from_s=240.0)
        await strategy.evaluate(_make_signal(), 250.0, executor)
        assert len(executor.calls) == 0
        assert not strategy._fired

    @pytest.mark.asyncio
    async def test_accumulates_inside_observation_window(self):
        """Inside [observe_to_s, observe_from_s] → accumulate only, no fire."""
        strategy, executor = _make_strategy(observe_from_s=240.0, observe_to_s=180.0)
        sig = _make_signal(bn_direction_from_open_pct=0.001)  # 0.1% as fraction
        await strategy.evaluate(sig, 200.0, executor)
        assert strategy._n == 1
        assert len(executor.calls) == 0
        assert not strategy._fired

    @pytest.mark.asyncio
    async def test_skips_when_no_market_data(self):
        strategy, executor = _make_strategy(window_open_price=0.0)
        await strategy.evaluate(_make_signal(), 170.0, executor)
        assert len(executor.calls) == 0

    @pytest.mark.asyncio
    async def test_fires_and_places_maker_when_conditions_met(self):
        """Full happy path: accumulate → fire → maker order placed."""
        strategy, executor = _make_strategy(
            side=Direction.UP,
            min_delta_pct=0.05,
            max_variance_pct=1.0,
            observe_from_s=240.0,
            observe_to_s=180.0,
        )
        # Set up Kelly context so sizing works
        strategy.kelly_wr_result = AdjustedWinRateResult(
            adjusted_p=0.88,
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

        # Accumulate enough ticks inside observation window
        for i in range(10):
            sig = _make_signal(bn_direction_from_open_pct=0.001)  # 0.1% → 0.10 in pct units
            await strategy.evaluate(sig, 220.0 - i * 4, executor)

        # Now time drops below observe_to_s → should fire
        sig = _make_signal(bn_direction_from_open_pct=0.001)
        await strategy.evaluate(sig, 170.0, executor)

        assert strategy._fired
        assert len(executor.calls) == 1
        assert executor.calls[0].method == "place_maker_order"

    @pytest.mark.asyncio
    async def test_kelly_no_edge_skips_order(self):
        """If Kelly says no edge at current price, skip."""
        strategy, executor = _make_strategy(
            side=Direction.UP,
            min_delta_pct=0.01,
            max_variance_pct=1.0,
            observe_from_s=240.0,
            observe_to_s=180.0,
            best_ask_up=0.95,  # very expensive → no edge
        )
        strategy.kelly_wr_result = AdjustedWinRateResult(
            adjusted_p=0.88,
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

        # Accumulate
        for i in range(10):
            sig = _make_signal(bn_direction_from_open_pct=0.001)
            await strategy.evaluate(sig, 220.0 - i * 4, executor)

        # Fire
        sig = _make_signal(bn_direction_from_open_pct=0.001)
        await strategy.evaluate(sig, 170.0, executor)

        assert strategy._fired
        assert len(executor.calls) == 0  # no order placed

    @pytest.mark.asyncio
    async def test_conditions_not_met_no_order(self):
        """Signal fires but conditions not met (delta too low)."""
        strategy, executor = _make_strategy(
            side=Direction.UP,
            min_delta_pct=0.50,  # very high threshold
            observe_from_s=240.0,
            observe_to_s=180.0,
        )
        strategy.kelly_wr_result = AdjustedWinRateResult(
            adjusted_p=0.88,
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

        # Accumulate with tiny delta (0.01% → below 0.50% threshold)
        for i in range(10):
            sig = _make_signal(bn_direction_from_open_pct=0.0001)
            await strategy.evaluate(sig, 220.0 - i * 4, executor)

        sig = _make_signal(bn_direction_from_open_pct=0.0001)
        await strategy.evaluate(sig, 170.0, executor)

        assert strategy._fired
        assert len(executor.calls) == 0

    @pytest.mark.asyncio
    async def test_maker_rejected_falls_through_to_taker(self):
        """If maker order is rejected (returns None), falls through to taker."""
        strategy, executor = _make_strategy(
            side=Direction.UP,
            min_delta_pct=0.01,
            max_variance_pct=1.0,
            observe_from_s=240.0,
            observe_to_s=180.0,
        )
        strategy.kelly_wr_result = AdjustedWinRateResult(
            adjusted_p=0.88,
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

        # Maker will return None (rejected)
        executor.next_order_id = None

        # Accumulate
        for i in range(10):
            sig = _make_signal(bn_direction_from_open_pct=0.001)
            await strategy.evaluate(sig, 220.0 - i * 4, executor)

        sig = _make_signal(bn_direction_from_open_pct=0.001)
        await strategy.evaluate(sig, 170.0, executor)

        assert strategy._fired
        # Should have tried maker first, then taker
        assert len(executor.calls) == 2
        assert executor.calls[0].method == "place_maker_order"
        assert executor.calls[1].method == "place_taker_order"

    @pytest.mark.asyncio
    async def test_only_fires_once_per_window(self):
        """After firing once, subsequent ticks don't re-evaluate."""
        strategy, executor = _make_strategy(
            side=Direction.UP,
            min_delta_pct=0.01,
            max_variance_pct=1.0,
            observe_from_s=240.0,
            observe_to_s=180.0,
        )
        strategy.kelly_wr_result = AdjustedWinRateResult(
            adjusted_p=0.88,
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

        # Accumulate
        for i in range(10):
            sig = _make_signal(bn_direction_from_open_pct=0.001)
            await strategy.evaluate(sig, 220.0 - i * 4, executor)

        # First fire
        sig = _make_signal(bn_direction_from_open_pct=0.001)
        await strategy.evaluate(sig, 170.0, executor)
        first_calls = len(executor.calls)
        assert first_calls > 0

        # Second tick at same time — should be no-op (maker is pending)
        await strategy.evaluate(sig, 169.0, executor)
        # No additional order calls (just monitoring existing maker)
        # _entry_complete or _maker_order_id guards prevent re-entry

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        strategy, executor = _make_strategy()
        strategy._order_placed = True
        strategy._fired = True
        strategy._n = 10
        strategy._mean = 5.0
        strategy._entry_complete = True

        strategy.reset()

        assert not strategy._order_placed
        assert not strategy._fired
        assert strategy._n == 0
        assert strategy._mean == 0.0
        assert not strategy._entry_complete


# ---------------------------------------------------------------------------
# Skip-maker high-confidence fast path tests
# ---------------------------------------------------------------------------


def _prime_strategy_for_fire(strategy: MomentumSignalStrategy) -> None:
    """Wire Kelly context + bankroll so a fire can actually size a bet."""
    strategy.kelly_wr_result = AdjustedWinRateResult(
        adjusted_p=0.88,
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


async def _run_fire(
    strategy: MomentumSignalStrategy,
    executor: FakeOrderExecutor,
    *,
    bn_pct: float = 0.001,
) -> None:
    """Accumulate 10 samples inside the observation window, then fire."""
    for i in range(10):
        sig = _make_signal(bn_direction_from_open_pct=bn_pct)
        await strategy.evaluate(sig, 220.0 - i * 4, executor)
    sig = _make_signal(bn_direction_from_open_pct=bn_pct)
    await strategy.evaluate(sig, 170.0, executor)


def _build_strategy_skip_maker(
    *,
    skip_maker_min_oos_wr_pct: float,
    skip_maker_max_stddev_pct: float,
    oos_win_rate_pct: float = 90.0,
    max_variance_pct: float = 1.0,
) -> tuple[MomentumSignalStrategy, FakeOrderExecutor]:
    cfg = make_rules_config(
        skip_maker_min_oos_wr_pct=skip_maker_min_oos_wr_pct,
        skip_maker_max_stddev_pct=skip_maker_max_stddev_pct,
    )
    state = make_market_state()
    sc = make_signal_config(
        side=Direction.UP,
        observe_from_s=240.0,
        observe_to_s=180.0,
        min_delta_pct=0.05,
        max_variance_pct=max_variance_pct,
        oos_win_rate_pct=oos_win_rate_pct,
    )
    strategy = MomentumSignalStrategy(cfg, state, sc)
    executor = FakeOrderExecutor()
    _prime_strategy_for_fire(strategy)
    return strategy, executor


class TestSkipMakerHighConfidence:
    """Cross-spread-on-fire path for ultra-high-confidence signals.

    With `skip_maker_min_oos_wr_pct > 0`, fires whose `oos_win_rate_pct`
    clears the threshold (and whose realised stddev clears the optional
    stddev gate) should bypass the maker quote and go straight to taker.
    """

    @pytest.mark.asyncio
    async def test_skip_maker_when_both_gates_pass(self):
        strategy, executor = _build_strategy_skip_maker(
            skip_maker_min_oos_wr_pct=96.0,
            skip_maker_max_stddev_pct=0.035,
            oos_win_rate_pct=97.0,
        )
        await _run_fire(strategy, executor)

        assert strategy._fired
        assert len(executor.calls) == 1
        assert executor.calls[0].method == "place_taker_order"

    @pytest.mark.asyncio
    async def test_keep_maker_when_oos_wr_below_threshold(self):
        strategy, executor = _build_strategy_skip_maker(
            skip_maker_min_oos_wr_pct=96.0,
            skip_maker_max_stddev_pct=0.035,
            oos_win_rate_pct=95.0,  # below gate
        )
        await _run_fire(strategy, executor)

        assert strategy._fired
        assert len(executor.calls) == 1
        assert executor.calls[0].method == "place_maker_order"

    @pytest.mark.asyncio
    async def test_keep_maker_when_stddev_above_gate(self):
        strategy, executor = _build_strategy_skip_maker(
            skip_maker_min_oos_wr_pct=96.0,
            skip_maker_max_stddev_pct=0.01,  # very tight gate
            oos_win_rate_pct=99.0,
        )
        # Samples vary enough that population stddev exceeds 0.01
        for i in range(10):
            val = 0.002 if i % 2 == 0 else 0.0005
            sig = _make_signal(bn_direction_from_open_pct=val)
            await strategy.evaluate(sig, 220.0 - i * 4, executor)
        sig = _make_signal(bn_direction_from_open_pct=0.002)
        await strategy.evaluate(sig, 170.0, executor)

        assert strategy._fired
        assert len(executor.calls) == 1
        assert executor.calls[0].method == "place_maker_order"

    @pytest.mark.asyncio
    async def test_disabled_when_min_oos_is_zero(self):
        """Setting `skip_maker_min_oos_wr_pct=0` fully disables the fast path."""
        strategy, executor = _build_strategy_skip_maker(
            skip_maker_min_oos_wr_pct=0.0,
            skip_maker_max_stddev_pct=0.035,
            oos_win_rate_pct=100.0,  # would otherwise trip the gate
        )
        await _run_fire(strategy, executor)

        assert strategy._fired
        assert len(executor.calls) == 1
        assert executor.calls[0].method == "place_maker_order"

    @pytest.mark.asyncio
    async def test_stddev_gate_disabled_with_zero(self):
        """`skip_maker_max_stddev_pct=0` means 'no stddev gate' — oos_wr alone decides."""
        strategy, executor = _build_strategy_skip_maker(
            skip_maker_min_oos_wr_pct=96.0,
            skip_maker_max_stddev_pct=0.0,
            oos_win_rate_pct=97.0,
        )
        # Noisy samples — stddev would fail a tight gate, but the gate is off
        for i in range(10):
            val = 0.005 if i % 2 == 0 else 0.0005
            sig = _make_signal(bn_direction_from_open_pct=val)
            await strategy.evaluate(sig, 220.0 - i * 4, executor)
        sig = _make_signal(bn_direction_from_open_pct=0.005)
        await strategy.evaluate(sig, 170.0, executor)

        assert strategy._fired
        assert len(executor.calls) == 1
        assert executor.calls[0].method == "place_taker_order"


# ---------------------------------------------------------------------------
# Maker monitoring tests
# ---------------------------------------------------------------------------


class TestMakerMonitoring:
    @pytest.mark.asyncio
    async def test_maker_filled_finalizes_entry(self):
        strategy, executor = _make_strategy()
        strategy._maker_order_id = "fake-order-1"
        strategy._maker_placed_at = 0.0  # placed long ago
        strategy._maker_entry_price = 0.84
        strategy._maker_token_id = "token_up"
        strategy._maker_size_usd = 5.0
        strategy._maker_tier = "momentum1"
        strategy._fire_signal = _make_signal()
        strategy.kelly_wr_result = AdjustedWinRateResult(
            adjusted_p=0.88,
            vol_discount=0,
            chop_discount=0,
            outcome_discount=0,
            total_discount=0,
            feedback_adjustment=0,
            regime_ready=True,
        )

        executor.filled_orders.add("fake-order-1")

        await strategy._monitor_maker_entry(200.0, executor)

        assert strategy._entry_complete
        assert strategy._order_placed
        assert strategy.last_entry_price == 0.84

    @pytest.mark.asyncio
    async def test_maker_expired_at_deadline(self):
        strategy, executor = _make_strategy()
        strategy._maker_order_id = "fake-order-1"
        strategy._maker_placed_at = 0.0
        strategy._maker_entry_price = 0.84
        strategy._maker_token_id = "token_up"
        strategy._maker_size_usd = 5.0

        # entry_window_stop is 5 → time_remaining < 5 means deadline
        await strategy._monitor_maker_entry(3.0, executor)

        assert strategy._entry_complete
        assert "fake-order-1" in executor.cancelled_orders
