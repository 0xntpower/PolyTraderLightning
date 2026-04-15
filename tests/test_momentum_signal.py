"""Tests for MomentumSignalStrategy — signal evaluation, conditions, order placement."""

from __future__ import annotations

from dataclasses import replace

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
# v2.9 CUSUM sustain gate + price-aware suppression tests
# ---------------------------------------------------------------------------


class _FakeErosionOrderMgr:
    """Minimal OrderExecutor for post-fire erosion monitoring tests.

    Records calls to exit_position_early so tests can assert whether the
    CUSUM path actually fired an exit.
    """

    mode = "paper"

    def __init__(self) -> None:
        self.exit_calls: list[float] = []

    async def exit_position_early(self, sell_price: float) -> float:
        self.exit_calls.append(sell_price)
        return 0.0


class _Clock:
    """Monotonic-clock stand-in with explicit advance() control."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def advance(self, dt: float) -> None:
        self.t += dt

    def __call__(self) -> float:
        return self.t


def _prime_erosion_state(
    *,
    side: Direction = Direction.UP,
    best_bid_up: float = 0.20,
    best_bid_down: float = 0.20,
    threshold: float = 0.5,
) -> tuple[MomentumSignalStrategy, _FakeErosionOrderMgr]:
    """Build a 'fired' strategy with CUSUM primed just at the limit.

    - fire_delta = ±1.0% so an incoming tick with bn=0 gives raw erosion = 1.0
      (which is below panic_threshold = 1.1 at the default 2.2x multiplier,
      so we stay in the CUSUM path)
    - erosion_ema starts at 0.80 (above threshold+tolerance = 0.55)
    - _erosion_cusum starts at the limit (0.80) so the next tick's excess
      pushes it over
    - last_entry_price = 0.80 so _execute_early_exit has a valid entry to
      sell against if the exit path fires
    """
    cfg = make_rules_config()
    state = make_market_state(best_bid_up=best_bid_up, best_bid_down=best_bid_down)
    base_sc = make_signal_config(side=side)
    sc = replace(base_sc, post_fire_max_safe_erosion_pct=threshold)
    strategy = MomentumSignalStrategy(cfg, state, sc)
    strategy.erosion_cfg = ErosionConfig()
    strategy._fire_delta_pct = 1.0 if side == Direction.UP else -1.0
    strategy._erosion_ema = 0.80
    strategy._erosion_ema_initialized = True
    strategy._erosion_cusum = 0.80
    strategy.last_entry_price = 0.80
    return strategy, _FakeErosionOrderMgr()


def _patch_erosion_env(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    """Pin time.monotonic to the fake clock and neutralise Discord sends."""
    monkeypatch.setattr("strategy.momentum_signal.time.monotonic", clock)
    monkeypatch.setattr("strategy.momentum_signal.send_early_exit", lambda **kw: None)


class TestCusumSustainGate:
    """v2.9 CUSUM sustain gate — single-tick breaches must not trigger exit.

    v2.8 had 2 hard-false + 1 partial-false CUSUM exits out of 9 (33% false
    rate), all firing on a single tick. The sustain gate requires the breach
    to persist for cusum_sustain_s (default 4.0s) before firing.
    """

    @pytest.mark.asyncio
    async def test_single_tick_breach_does_not_exit(self, monkeypatch):
        """First tick over the limit starts the sustain timer and holds."""
        strategy, order_mgr = _prime_erosion_state()
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        # CUSUM went above the limit but the sustain gate held the exit.
        assert strategy._erosion_cusum >= strategy.erosion_cfg.cusum_limit
        assert strategy._cusum_breach_started_at == 1000.0
        assert not strategy._early_exit_triggered
        assert order_mgr.exit_calls == []

    @pytest.mark.asyncio
    async def test_sustained_breach_exits_after_required_duration(self, monkeypatch):
        """Continued breach beyond cusum_sustain_s triggers the exit."""
        strategy, order_mgr = _prime_erosion_state(best_bid_up=0.20)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0)

        # Tick 1: sustain timer starts, no exit.
        await strategy._monitor_post_fire_erosion(sig, order_mgr)
        assert order_mgr.exit_calls == []

        # Tick 2: 5.0s later — past the 4.0s sustain window, breach continues.
        clock.advance(5.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1
        assert order_mgr.exit_calls[0] == pytest.approx(0.20)

    @pytest.mark.asyncio
    async def test_dropout_below_limit_resets_timer(self, monkeypatch):
        """CUSUM dropping below limit mid-stream must clear the sustain timer
        so a later re-breach gets a fresh clock."""
        strategy, order_mgr = _prime_erosion_state()
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0)

        # Tick 1: breach → timer starts.
        await strategy._monitor_post_fire_erosion(sig, order_mgr)
        assert strategy._cusum_breach_started_at == 1000.0

        # Force internal state below the limit so the next tick's cusum
        # update lands in the decay branch (ema below threshold+tolerance
        # means excess==0 → cusum *= decay).
        strategy._erosion_cusum = 0.50
        strategy._erosion_ema = 0.40

        clock.advance(1.0)
        # bn=0.006 → current_pct=0.6 → raw erosion=(1.0-0.6)/1.0=0.4 (below
        # threshold+tolerance=0.55 so no fresh excess).
        recover_sig = _make_signal(bn_direction_from_open_pct=0.006)
        await strategy._monitor_post_fire_erosion(recover_sig, order_mgr)

        assert strategy._erosion_cusum < strategy.erosion_cfg.cusum_limit
        assert strategy._cusum_breach_started_at is None
        assert order_mgr.exit_calls == []

    @pytest.mark.asyncio
    async def test_sustain_disabled_with_zero_fires_on_first_tick(self, monkeypatch):
        """cusum_sustain_s=0 disables the gate — single-tick exit allowed."""
        strategy, order_mgr = _prime_erosion_state(best_bid_up=0.20)
        strategy.erosion_cfg = ErosionConfig(cusum_sustain_s=0.0)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1


class TestCusumSuppression:
    """v2.9 price-aware CUSUM suppression — refuse to exit when the market
    already agrees with us.

    v2.8 Trade 19 exited via CUSUM while bid_up=0.99 (market fully priced in
    the winning outcome). Suppression holds when the top bid on our side is
    already >= cusum_suppress_top_bid (default 0.85), preserving those wins.
    """

    @staticmethod
    def _pre_sustained(
        *,
        side: Direction = Direction.UP,
        best_bid_up: float = 0.20,
        best_bid_down: float = 0.20,
    ) -> tuple[MomentumSignalStrategy, _FakeErosionOrderMgr]:
        """Same as _prime_erosion_state but with sustain already satisfied —
        the next tick proceeds straight to the suppression check."""
        strategy, order_mgr = _prime_erosion_state(
            side=side, best_bid_up=best_bid_up, best_bid_down=best_bid_down
        )
        # Sustain started far in the past — any clock reading past 4.0s
        # (via the test clock starting at 1000.0) clears the sustain gate.
        strategy._cusum_breach_started_at = 0.0
        return strategy, order_mgr

    @pytest.mark.asyncio
    async def test_suppression_holds_when_top_bid_agrees(self, monkeypatch):
        """top_bid >= cusum_suppress_top_bid → hold position, no exit."""
        strategy, order_mgr = self._pre_sustained(best_bid_up=0.90)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert not strategy._early_exit_triggered
        assert order_mgr.exit_calls == []

    @pytest.mark.asyncio
    async def test_suppression_boundary_equal_still_holds(self, monkeypatch):
        """top_bid exactly equal to cusum_suppress_top_bid is suppressed (>=)."""
        strategy, order_mgr = self._pre_sustained(best_bid_up=0.85)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert not strategy._early_exit_triggered

    @pytest.mark.asyncio
    async def test_no_suppression_when_top_bid_disagrees(self, monkeypatch):
        """top_bid < cusum_suppress_top_bid → exit proceeds."""
        strategy, order_mgr = self._pre_sustained(best_bid_up=0.20)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1

    @pytest.mark.asyncio
    async def test_suppression_disabled_with_zero(self, monkeypatch):
        """cusum_suppress_top_bid=0 disables suppression — exit even at 0.99."""
        strategy, order_mgr = self._pre_sustained(best_bid_up=0.99)
        strategy.erosion_cfg = ErosionConfig(cusum_suppress_top_bid=0.0)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1

    @pytest.mark.asyncio
    async def test_down_signal_reads_down_side_bid(self, monkeypatch):
        """DOWN signals must check best_bid_down, not best_bid_up."""
        strategy, order_mgr = self._pre_sustained(
            side=Direction.DOWN,
            best_bid_up=0.05,
            best_bid_down=0.92,
        )
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        # For DOWN, fire_delta is -1.0%; bn=0 → erosion = (-1 - 0) / -1 = 1.0
        sig = _make_signal(bn_direction_from_open_pct=0.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        # best_bid_down=0.92 >= 0.85 → suppression holds the exit.
        assert not strategy._early_exit_triggered
        assert order_mgr.exit_calls == []


# ---------------------------------------------------------------------------
# v3.0 P1 — CUSUM delta-reversal gate
# ---------------------------------------------------------------------------


class TestCusumReversalGate:
    """The v3.0 reversal gate suppresses CUSUM exits when the live BTC
    delta has not actually reversed far enough from the fire delta. v2.9
    had 4/6 CUSUM exits fire while |current - fire| <= 0.14pp (premature
    exits on noise); both valid exits had reversal >= 0.16pp.
    """

    @staticmethod
    def _sustained_primed(
        *,
        best_bid_up: float = 0.20,
        min_reversal_pp: float = 0.15,
    ) -> tuple[MomentumSignalStrategy, _FakeErosionOrderMgr]:
        """Build a strategy where sustain has elapsed so the reversal gate
        is the next check on any tick that keeps the CUSUM breached. The
        v3.1 cusum override is disabled here so the gate is exercised in
        isolation; TestCusumOverride covers the bypass behavior."""
        strategy, order_mgr = _prime_erosion_state(best_bid_up=best_bid_up)
        strategy.erosion_cfg = ErosionConfig(
            cusum_min_reversal_pp=min_reversal_pp,
            cusum_override_multiplier=0.0,
        )
        # Pre-expire the sustain timer so the reversal check runs first.
        strategy._cusum_breach_started_at = 0.0
        # Push the CUSUM well above the limit so even a decayed tick stays
        # over the limit and we reach the reversal gate.
        strategy._erosion_cusum = 5.0
        return strategy, order_mgr

    @pytest.mark.asyncio
    async def test_shallow_reversal_suppresses_exit(self, monkeypatch):
        """|current - fire| < min_reversal_pp → exit blocked."""
        strategy, order_mgr = self._sustained_primed()
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        # fire_delta = 1.0 (pp). current_pct = 0.0095 * 100 = 0.95 pp.
        # reversal = |0.95 - 1.0| = 0.05 pp — below default gate of 0.15 pp.
        sig = _make_signal(bn_direction_from_open_pct=0.0095)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert not strategy._early_exit_triggered
        assert order_mgr.exit_calls == []

    @pytest.mark.asyncio
    async def test_deep_reversal_allows_exit(self, monkeypatch):
        """|current - fire| >= min_reversal_pp → exit proceeds."""
        strategy, order_mgr = self._sustained_primed()
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        # bn=0.0 → current_pct=0 → reversal = |0 - 1.0| = 1.0 pp >> 0.15
        sig = _make_signal(bn_direction_from_open_pct=0.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1

    @pytest.mark.asyncio
    async def test_reversal_just_above_threshold_allows_exit(self, monkeypatch):
        """Reversal just above min_reversal_pp clears the strict-less-than
        gate and the exit proceeds."""
        strategy, order_mgr = self._sustained_primed(min_reversal_pp=0.15)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        # current = 0.0084 * 100 = 0.84 pp → reversal = 0.16 pp > 0.15
        sig = _make_signal(bn_direction_from_open_pct=0.0084)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1

    @pytest.mark.asyncio
    async def test_gate_disabled_with_zero(self, monkeypatch):
        """cusum_min_reversal_pp=0 disables the gate — shallow reversal still exits."""
        strategy, order_mgr = self._sustained_primed(min_reversal_pp=0.0)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        # Would normally be blocked (reversal = 0.05 pp) but the gate is off.
        sig = _make_signal(bn_direction_from_open_pct=0.0095)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1


# ---------------------------------------------------------------------------
# v3.0 P6 — hostile-regime double cap
# ---------------------------------------------------------------------------


class TestHostileRegimeCap:
    """When vol_discount or chop_discount exceeds hostile_regime_threshold,
    the final Kelly bet is scaled by hostile_regime_multiplier on top of
    the regime cap. Applied once inside the sizing block before SPRT.
    """

    @staticmethod
    def _build(
        *,
        vol_discount: float = 0.0,
        chop_discount: float = 0.0,
        hostile_threshold: float = 0.20,
        hostile_multiplier: float = 0.5,
        hostile_skip_threshold: float = 0.0,
    ) -> tuple[MomentumSignalStrategy, FakeOrderExecutor]:
        strategy, executor = _make_strategy(
            side=Direction.UP,
            min_delta_pct=0.05,
            max_variance_pct=1.0,
            observe_from_s=240.0,
            observe_to_s=180.0,
        )
        strategy.kelly_wr_result = AdjustedWinRateResult(
            adjusted_p=0.88,
            vol_discount=vol_discount,
            chop_discount=chop_discount,
            outcome_discount=0,
            total_discount=0,
            feedback_adjustment=0,
            regime_ready=True,
        )
        strategy.sizing_cfg = SizingConfig(
            hostile_regime_threshold=hostile_threshold,
            hostile_regime_multiplier=hostile_multiplier,
            hostile_regime_skip_threshold=hostile_skip_threshold,
        )
        strategy.erosion_cfg = ErosionConfig()
        strategy.bankroll = 1000.0
        return strategy, executor

    @pytest.mark.asyncio
    async def test_benign_regime_leaves_bet_unchanged(self):
        strategy, executor = self._build(vol_discount=0.05, chop_discount=0.05)
        await _run_fire(strategy, executor)

        assert strategy._fired
        assert strategy.last_kelly_result is not None
        # Bet size matches the raw Kelly (no hostile cap, no multiplier).
        # Compare to fractional Kelly clamped by min/max bet. We only need
        # to assert the hostile code path did NOT multiply.
        assert len(executor.calls) == 1
        ordered = executor.calls[0].size_usd
        # With adjusted_p=0.88 and entry ~0.85 bankroll=1000, the Kelly bet
        # lands in the mid two-digit dollar range. What matters is it's
        # *not* been halved — so above 1.5x the minimum bet floor.
        assert ordered > strategy.sizing_cfg.kelly_min_bet * 1.5

    @pytest.mark.asyncio
    async def test_vol_above_threshold_halves_bet(self):
        benign_strat, benign_exec = self._build(vol_discount=0.05)
        await _run_fire(benign_strat, benign_exec)
        assert len(benign_exec.calls) == 1
        benign_size = benign_exec.calls[0].size_usd

        hostile_strat, hostile_exec = self._build(vol_discount=0.35)
        await _run_fire(hostile_strat, hostile_exec)
        assert len(hostile_exec.calls) == 1
        hostile_size = hostile_exec.calls[0].size_usd

        # Hostile size must be strictly smaller; ratio ~0.5 (within rounding
        # caused by the min-bet floor and the $0.01 round in size_usd).
        assert hostile_size < benign_size
        assert hostile_size == pytest.approx(benign_size * 0.5, rel=0.02)

    @pytest.mark.asyncio
    async def test_chop_above_threshold_halves_bet(self):
        """chop_discount drives the cap identically to vol_discount."""
        benign_strat, benign_exec = self._build(chop_discount=0.05)
        await _run_fire(benign_strat, benign_exec)
        benign_size = benign_exec.calls[0].size_usd

        hostile_strat, hostile_exec = self._build(chop_discount=0.30)
        await _run_fire(hostile_strat, hostile_exec)
        hostile_size = hostile_exec.calls[0].size_usd

        assert hostile_size == pytest.approx(benign_size * 0.5, rel=0.02)

    @pytest.mark.asyncio
    async def test_max_metric_wins_between_vol_and_chop(self):
        """The gate uses max(vol, chop); either one above the threshold trips it."""
        # vol benign, chop hostile → still triggers.
        strategy, executor = self._build(vol_discount=0.05, chop_discount=0.40)
        await _run_fire(strategy, executor)
        assert strategy.last_kelly_result is not None

        benign, benign_exec = self._build(vol_discount=0.05, chop_discount=0.05)
        await _run_fire(benign, benign_exec)
        assert executor.calls[0].size_usd == pytest.approx(
            benign_exec.calls[0].size_usd * 0.5, rel=0.02
        )

    @pytest.mark.asyncio
    async def test_threshold_exactly_equal_does_not_trip(self):
        """The check is strictly greater-than, so equality stays benign."""
        at_threshold, at_exec = self._build(vol_discount=0.20, hostile_threshold=0.20)
        benign, benign_exec = self._build(vol_discount=0.05)
        await _run_fire(at_threshold, at_exec)
        await _run_fire(benign, benign_exec)

        # Same bet size when strictly-greater-than gate is not tripped.
        assert at_exec.calls[0].size_usd == pytest.approx(benign_exec.calls[0].size_usd)

    @pytest.mark.asyncio
    async def test_custom_multiplier_applies(self):
        """A non-default multiplier (e.g. 0.25) also composes correctly."""
        benign_strat, benign_exec = self._build(vol_discount=0.05)
        await _run_fire(benign_strat, benign_exec)
        benign_size = benign_exec.calls[0].size_usd

        hostile_strat, hostile_exec = self._build(
            vol_discount=0.35,
            hostile_multiplier=0.25,
        )
        await _run_fire(hostile_strat, hostile_exec)
        hostile_size = hostile_exec.calls[0].size_usd

        assert hostile_size == pytest.approx(benign_size * 0.25, rel=0.03)


# ---------------------------------------------------------------------------
# v3.1 — hostile-regime SKIP gate
# ---------------------------------------------------------------------------


class TestHostileRegimeSkip:
    """The v3.1 hostile-regime skip aborts the fire entirely when severity
    exceeds hostile_regime_skip_threshold, leaving the
    [hostile_regime_threshold, hostile_regime_skip_threshold) band to the
    existing v3.0 P6 halving. Skip threshold default 0.25 — driven by the
    v3.0 paper session loss at vol_sev=0.274."""

    @pytest.mark.asyncio
    async def test_vol_above_skip_threshold_aborts_fire(self):
        strategy, executor = TestHostileRegimeCap._build(
            vol_discount=0.30,
            hostile_skip_threshold=0.25,
        )
        await _run_fire(strategy, executor)
        # No order placed at all — the skip exits the sizing block early.
        assert len(executor.calls) == 0

    @pytest.mark.asyncio
    async def test_chop_above_skip_threshold_aborts_fire(self):
        strategy, executor = TestHostileRegimeCap._build(
            chop_discount=0.40,
            hostile_skip_threshold=0.25,
        )
        await _run_fire(strategy, executor)
        assert len(executor.calls) == 0

    @pytest.mark.asyncio
    async def test_severity_inside_halving_band_still_fires_halved(self):
        """Severity in (hostile_threshold, skip_threshold] takes the halving
        path and still places an order — the skip must not steal that band."""
        benign, benign_exec = TestHostileRegimeCap._build(
            vol_discount=0.05,
            hostile_skip_threshold=0.25,
        )
        await _run_fire(benign, benign_exec)
        benign_size = benign_exec.calls[0].size_usd

        halved, halved_exec = TestHostileRegimeCap._build(
            vol_discount=0.23,  # > 0.20 (halve), <= 0.25 (skip threshold)
            hostile_skip_threshold=0.25,
        )
        await _run_fire(halved, halved_exec)
        assert len(halved_exec.calls) == 1
        assert halved_exec.calls[0].size_usd == pytest.approx(benign_size * 0.5, rel=0.02)

    @pytest.mark.asyncio
    async def test_severity_exactly_equal_does_not_skip(self):
        """The skip uses strict greater-than so equality stays in the halving
        band."""
        strategy, executor = TestHostileRegimeCap._build(
            vol_discount=0.25,
            hostile_skip_threshold=0.25,
        )
        await _run_fire(strategy, executor)
        # Did not skip — still produced an order (halved by the v3.0 P6 path).
        assert len(executor.calls) == 1

    @pytest.mark.asyncio
    async def test_skip_disabled_with_zero(self):
        """hostile_regime_skip_threshold=0 disables the skip — even a wildly
        hostile severity still fires (subject to the halving path)."""
        strategy, executor = TestHostileRegimeCap._build(
            vol_discount=0.50,
            hostile_skip_threshold=0.0,
        )
        await _run_fire(strategy, executor)
        assert len(executor.calls) == 1


# ---------------------------------------------------------------------------
# v3.1 — CUSUM overwhelming-breach override
# ---------------------------------------------------------------------------


class TestCusumOverride:
    """The v3.1 override bypasses the reversal-pp gate and the top-bid
    suppression once the CUSUM accumulator climbs to
    cusum_override_multiplier * cusum_limit. Covers the v3.0 01:47 loss
    case (cusum=1.608 = 2.01x of 0.80, blocked by the reversal gate)."""

    @staticmethod
    def _primed(
        *,
        best_bid_up: float = 0.20,
        cusum_value: float = 1.80,
        override_multiplier: float = 2.0,
        min_reversal_pp: float = 0.15,
        suppress_top_bid: float = 0.85,
    ) -> tuple[MomentumSignalStrategy, _FakeErosionOrderMgr]:
        strategy, order_mgr = _prime_erosion_state(best_bid_up=best_bid_up)
        strategy.erosion_cfg = ErosionConfig(
            cusum_min_reversal_pp=min_reversal_pp,
            cusum_suppress_top_bid=suppress_top_bid,
            cusum_override_multiplier=override_multiplier,
        )
        # Sustain already cleared so we walk straight to the suppressions.
        strategy._cusum_breach_started_at = 0.0
        strategy._erosion_cusum = cusum_value
        return strategy, order_mgr

    @pytest.mark.asyncio
    async def test_override_bypasses_shallow_reversal(self, monkeypatch):
        """cusum >= 2x limit and shallow reversal — exit fires anyway."""
        # 2x of default cusum_limit (0.80) = 1.60. 1.80 clears it.
        strategy, order_mgr = self._primed(cusum_value=1.80)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        # Shallow reversal: fire=1.0, current=0.95, |diff|=0.05 < 0.15
        sig = _make_signal(bn_direction_from_open_pct=0.0095)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1

    @pytest.mark.asyncio
    async def test_override_bypasses_top_bid_suppression(self, monkeypatch):
        """cusum >= 2x limit and friendly top bid — exit fires anyway."""
        strategy, order_mgr = self._primed(
            best_bid_up=0.95,
            cusum_value=1.80,
        )
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        # Deep reversal so we are not blocked by the reversal gate either.
        sig = _make_signal(bn_direction_from_open_pct=0.0)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1

    @pytest.mark.asyncio
    async def test_below_override_threshold_still_suppressed(self, monkeypatch):
        """Just below 2x limit — reversal gate still blocks the exit."""
        # 2x of 0.80 = 1.60. 1.40 is above limit, below override threshold.
        strategy, order_mgr = self._primed(cusum_value=1.40)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0095)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert not strategy._early_exit_triggered
        assert order_mgr.exit_calls == []

    @pytest.mark.asyncio
    async def test_override_disabled_with_zero(self, monkeypatch):
        """cusum_override_multiplier=0 disables the override — even a huge
        breach is still blocked by the reversal gate."""
        strategy, order_mgr = self._primed(
            cusum_value=10.0,
            override_multiplier=0.0,
        )
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0095)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert not strategy._early_exit_triggered
        assert order_mgr.exit_calls == []

    @pytest.mark.asyncio
    async def test_v3_0_loss_replay(self, monkeypatch):
        """Replay the v3.0 01:47 loss numbers: cusum=1.608 (2.01x of 0.80)
        with shallow reversal — override fires the exit."""
        strategy, order_mgr = self._primed(cusum_value=1.608)
        clock = _Clock(start=1000.0)
        _patch_erosion_env(monkeypatch, clock)

        sig = _make_signal(bn_direction_from_open_pct=0.0095)
        await strategy._monitor_post_fire_erosion(sig, order_mgr)

        assert strategy._early_exit_triggered
        assert len(order_mgr.exit_calls) == 1


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
