"""Tests for RiskRegistry — composable risk checks.

Covers:
- Individual check behavior (DailyLossLimit, ConsecutiveLossLimit, WindowExposureCap)
- Registry composition (first failure blocks, halt vs soft block)
- Kill switch halt/reset
- from_config factory
"""

from __future__ import annotations

from risk.registry import (
    ConsecutiveLossLimit,
    DailyLossLimit,
    RiskRegistry,
    WindowExposureCap,
)

# ---------------------------------------------------------------------------
# Fake tracker
# ---------------------------------------------------------------------------


class FakeTracker:
    def __init__(self, daily_pnl=0.0, consecutive_losses=0, window_exposure_usd=0.0):
        self.daily_pnl = daily_pnl
        self.consecutive_losses = consecutive_losses
        self.window_exposure_usd = window_exposure_usd


class FakeRiskConfig:
    max_daily_loss_usd = 50.0
    max_consecutive_losses = 5
    max_position_per_window_usd = 25.0


# ---------------------------------------------------------------------------
# Individual check tests
# ---------------------------------------------------------------------------


class TestDailyLossLimit:
    def test_allows_when_under_limit(self):
        tracker = FakeTracker(daily_pnl=-10.0)
        check = DailyLossLimit(50.0, tracker)
        assert check.check().allowed

    def test_blocks_at_limit(self):
        tracker = FakeTracker(daily_pnl=-50.0)
        check = DailyLossLimit(50.0, tracker)
        v = check.check()
        assert not v.allowed
        assert "daily loss" in v.reason

    def test_blocks_beyond_limit(self):
        tracker = FakeTracker(daily_pnl=-100.0)
        check = DailyLossLimit(50.0, tracker)
        assert not check.check().allowed

    def test_allows_positive_pnl(self):
        tracker = FakeTracker(daily_pnl=100.0)
        check = DailyLossLimit(50.0, tracker)
        assert check.check().allowed


class TestConsecutiveLossLimit:
    def test_allows_under_limit(self):
        tracker = FakeTracker(consecutive_losses=3)
        check = ConsecutiveLossLimit(5, tracker)
        assert check.check().allowed

    def test_blocks_at_limit(self):
        tracker = FakeTracker(consecutive_losses=5)
        check = ConsecutiveLossLimit(5, tracker)
        v = check.check()
        assert not v.allowed
        assert "consecutive" in v.reason

    def test_allows_zero_losses(self):
        tracker = FakeTracker(consecutive_losses=0)
        check = ConsecutiveLossLimit(5, tracker)
        assert check.check().allowed


class TestWindowExposureCap:
    def test_allows_under_limit(self):
        tracker = FakeTracker(window_exposure_usd=10.0)
        check = WindowExposureCap(25.0, tracker)
        assert check.check(additional_usd=10.0).allowed

    def test_blocks_over_limit(self):
        tracker = FakeTracker(window_exposure_usd=20.0)
        check = WindowExposureCap(25.0, tracker)
        v = check.check(additional_usd=10.0)
        assert not v.allowed
        assert "position cap" in v.reason

    def test_allows_exact_limit(self):
        tracker = FakeTracker(window_exposure_usd=15.0)
        check = WindowExposureCap(25.0, tracker)
        assert check.check(additional_usd=10.0).allowed

    def test_allows_zero_additional(self):
        tracker = FakeTracker(window_exposure_usd=20.0)
        check = WindowExposureCap(25.0, tracker)
        assert check.check(additional_usd=0.0).allowed


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRiskRegistry:
    def test_empty_registry_allows(self):
        reg = RiskRegistry()
        assert reg.can_trade(5.0)
        assert not reg.halted

    def test_all_checks_pass(self):
        tracker = FakeTracker(daily_pnl=10.0, consecutive_losses=0, window_exposure_usd=0.0)
        reg = RiskRegistry()
        reg.register(DailyLossLimit(50.0, tracker))
        reg.register(ConsecutiveLossLimit(5, tracker))
        reg.register(WindowExposureCap(25.0, tracker))
        assert reg.can_trade(10.0)

    def test_daily_loss_halts(self):
        tracker = FakeTracker(daily_pnl=-60.0, consecutive_losses=0)
        reg = RiskRegistry()
        reg.register(DailyLossLimit(50.0, tracker))
        reg.register(ConsecutiveLossLimit(5, tracker))

        assert not reg.can_trade()
        assert reg.halted
        assert "daily loss" in reg.halt_reason

    def test_consecutive_loss_halts(self):
        tracker = FakeTracker(daily_pnl=0.0, consecutive_losses=5)
        reg = RiskRegistry()
        reg.register(DailyLossLimit(50.0, tracker))
        reg.register(ConsecutiveLossLimit(5, tracker))

        assert not reg.can_trade()
        assert reg.halted

    def test_window_cap_blocks_without_halting(self):
        tracker = FakeTracker(daily_pnl=0.0, consecutive_losses=0, window_exposure_usd=20.0)
        reg = RiskRegistry()
        reg.register(DailyLossLimit(50.0, tracker))
        reg.register(WindowExposureCap(25.0, tracker))

        assert not reg.can_trade(additional_usd=10.0)
        assert not reg.halted  # soft block, not a halt

    def test_halt_persists_until_reset(self):
        tracker = FakeTracker(daily_pnl=-60.0)
        reg = RiskRegistry()
        reg.register(DailyLossLimit(50.0, tracker))

        reg.can_trade()
        assert reg.halted

        # Even if PnL recovers, still halted
        tracker.daily_pnl = 100.0
        assert not reg.can_trade()
        assert reg.halted

        # Reset clears halt
        reg.reset_halt()
        assert not reg.halted
        assert reg.can_trade()

    def test_from_config_factory(self):
        tracker = FakeTracker(daily_pnl=0.0, consecutive_losses=0, window_exposure_usd=0.0)
        cfg = FakeRiskConfig()
        reg = RiskRegistry.from_config(cfg, tracker)

        assert len(reg._checks) == 3
        assert reg.can_trade(5.0)

    def test_from_config_respects_limits(self):
        tracker = FakeTracker(daily_pnl=-60.0, consecutive_losses=0)
        cfg = FakeRiskConfig()
        reg = RiskRegistry.from_config(cfg, tracker)

        assert not reg.can_trade()
        assert reg.halted
