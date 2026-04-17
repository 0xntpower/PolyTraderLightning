"""Tests for PostLossCooldown — v3.2 §5.8 post-loss freeze gate."""

from __future__ import annotations

from strategy.post_loss_cooldown import PostLossCooldown


class TestEnabledDisabled:
    def test_disabled_never_freezes(self):
        tracker = PostLossCooldown(enabled=False, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.register_loss(100.0, 1000.0)  # 10% loss, far above threshold
        assert tracker.is_frozen is False
        assert tracker.windows_remaining == 0

    def test_zero_windows_disables_gate(self):
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=0)
        tracker.register_loss(100.0, 1000.0)
        assert tracker.is_frozen is False


class TestRegisterLoss:
    def test_loss_below_threshold_does_not_arm(self):
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        armed = tracker.register_loss(15.0, 1000.0)  # 1.5% < 2.0%
        assert armed is False
        assert tracker.is_frozen is False

    def test_loss_at_threshold_does_not_arm(self):
        """Strictly > threshold required. 2.0% exactly should NOT arm."""
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        armed = tracker.register_loss(20.0, 1000.0)  # exactly 2.0%
        assert armed is False
        assert tracker.is_frozen is False

    def test_loss_above_threshold_arms(self):
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        armed = tracker.register_loss(25.0, 1000.0)  # 2.5%
        assert armed is True
        assert tracker.is_frozen is True
        assert tracker.windows_remaining == 1

    def test_zero_loss_does_not_arm(self):
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        assert tracker.register_loss(0.0, 1000.0) is False
        assert tracker.is_frozen is False

    def test_zero_bankroll_does_not_arm(self):
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        assert tracker.register_loss(10.0, 0.0) is False

    def test_multiple_windows_arm_for_full_duration(self):
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=3)
        tracker.register_loss(100.0, 1000.0)
        assert tracker.windows_remaining == 3


class TestWindowBoundary:
    def test_boundary_decrements_counter(self):
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=2)
        tracker.register_loss(100.0, 1000.0)
        assert tracker.windows_remaining == 2

        tracker.on_window_boundary()
        assert tracker.windows_remaining == 1
        assert tracker.is_frozen is True

        tracker.on_window_boundary()
        assert tracker.windows_remaining == 0
        assert tracker.is_frozen is False

    def test_boundary_noop_when_not_armed(self):
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.on_window_boundary()
        assert tracker.windows_remaining == 0

    def test_boundary_cannot_go_negative(self):
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.on_window_boundary()
        tracker.on_window_boundary()
        tracker.on_window_boundary()
        assert tracker.windows_remaining == 0

    def test_boundary_disabled_is_noop(self):
        tracker = PostLossCooldown(enabled=False, loss_pct_threshold=2.0, cooldown_windows=1)
        # Even if we could somehow arm it, boundary should do nothing when disabled.
        tracker.on_window_boundary()
        assert tracker.windows_remaining == 0


class TestRearming:
    def test_subsequent_loss_resets_counter(self):
        """A second qualifying loss while still frozen resets the full counter."""
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=2)
        tracker.register_loss(100.0, 1000.0)
        tracker.on_window_boundary()  # remaining=1
        assert tracker.windows_remaining == 1

        # Another large loss re-arms to the full cooldown window count.
        tracker.register_loss(100.0, 1000.0)
        assert tracker.windows_remaining == 2
