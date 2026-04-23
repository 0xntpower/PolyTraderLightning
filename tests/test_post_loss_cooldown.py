"""Tests for PostLossCooldown — v3.2 §5.8 post-loss freeze gate.

v3.6.2 rewrote the internal state from a countdown to an absolute
``freeze_until_window_ts`` marker after the 2026-04-22 session showed
the old semantic freezing zero windows when the gamma resolve fired
mid-window. These tests exercise the new API directly, including the
specific regression case from that session.

Arming formula: freeze from ``max(resolved_window_ts, current_window_ts) + span``
for ``cooldown_windows`` windows. This handles both callsite shapes:

- Paper (boundary-driven): register_loss fires BEFORE on_window_boundary
  advances the pointer, so current = old window. freeze starts at old+1.
- Live (gamma-poll driven): register_loss fires mid-window-N+1 after the
  boundary already advanced the pointer. current = N+1. freeze starts
  at N+2.
"""

from __future__ import annotations

from strategy.post_loss_cooldown import PostLossCooldown

# Window timestamps spaced 300 s apart (matches btc-updown-5m).
_W0 = 1_776_000_000
_W1 = _W0 + 300
_W2 = _W0 + 600
_W3 = _W0 + 900
_W4 = _W0 + 1200
_W5 = _W0 + 1500


class TestEnabledDisabled:
    def test_disabled_never_freezes(self) -> None:
        tracker = PostLossCooldown(enabled=False, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.on_window_boundary(_W0)
        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W0)
        tracker.on_window_boundary(_W1)
        assert tracker.is_frozen is False
        assert tracker.windows_remaining == 0

    def test_zero_windows_disables_gate(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=0)
        tracker.on_window_boundary(_W0)
        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W0)
        tracker.on_window_boundary(_W1)
        assert tracker.is_frozen is False


class TestRegisterLoss:
    def test_loss_below_threshold_does_not_arm(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.on_window_boundary(_W0)
        armed = tracker.register_loss(15.0, 1000.0, resolved_window_ts=_W0)  # 1.5 %
        assert armed is False
        tracker.on_window_boundary(_W1)
        assert tracker.is_frozen is False

    def test_loss_at_threshold_does_not_arm(self) -> None:
        """Strictly > threshold required. 2.0 % exactly should NOT arm."""
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.on_window_boundary(_W0)
        armed = tracker.register_loss(20.0, 1000.0, resolved_window_ts=_W0)
        assert armed is False
        tracker.on_window_boundary(_W1)
        assert tracker.is_frozen is False

    def test_loss_above_threshold_arms(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.on_window_boundary(_W0)
        armed = tracker.register_loss(25.0, 1000.0, resolved_window_ts=_W0)  # 2.5 %
        assert armed is True
        tracker.on_window_boundary(_W1)
        assert tracker.is_frozen is True
        assert tracker.windows_remaining == 1

    def test_zero_loss_does_not_arm(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        assert tracker.register_loss(0.0, 1000.0, resolved_window_ts=_W0) is False

    def test_zero_bankroll_does_not_arm(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        assert tracker.register_loss(10.0, 0.0, resolved_window_ts=_W0) is False

    def test_zero_resolved_window_ts_does_not_arm(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        assert tracker.register_loss(100.0, 1000.0, resolved_window_ts=0) is False


# ---------------------------------------------------------------------------
# Paper-mode shape: arm fires while current_window_ts still = old window
# ---------------------------------------------------------------------------


class TestPaperModeArmShape:
    def test_cooldown_of_one_freezes_exactly_next_window(self) -> None:
        """Paper flow: on_window_boundary runs AFTER register_loss in the
        transition, so at arm time current = W0 (the window that just
        closed). Freeze covers W1 and W1 only."""
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.on_window_boundary(_W0)
        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W0)
        # Then the transition's tail advances the cursor to W1.
        tracker.on_window_boundary(_W1)
        assert tracker.is_frozen is True
        assert tracker.windows_remaining == 1

        tracker.on_window_boundary(_W2)
        assert tracker.is_frozen is False

    def test_cooldown_of_three_freezes_next_three_windows(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=3)
        tracker.on_window_boundary(_W0)
        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W0)

        tracker.on_window_boundary(_W1)
        assert tracker.is_frozen is True
        tracker.on_window_boundary(_W2)
        assert tracker.is_frozen is True
        tracker.on_window_boundary(_W3)
        assert tracker.is_frozen is True
        tracker.on_window_boundary(_W4)
        assert tracker.is_frozen is False


# ---------------------------------------------------------------------------
# Live-mode shape: arm fires mid-window, current_window_ts already advanced
# ---------------------------------------------------------------------------


class TestLiveModeMidWindowArm:
    def test_gamma_resolve_in_next_window_still_freezes_following_window(self) -> None:
        """The exact live-mode timing from the 2026-04-22 T4 session.

        T4 fires in window N. Window N closes, pointer advances to N+1.
        Gamma polls mid-N+1 and register_loss fires with current = N+1
        and resolved = N. Under the OLD countdown the cooldown decremented
        to zero at the N+1→N+2 boundary and window N+2 was not frozen.
        Under the new semantics, freeze anchor = max(N, N+1) = N+1, so
        freeze covers N+2 (the first tradable window the strategy will
        see after T4's resolution).
        """
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        # Window N: trade fired (no cooldown yet).
        tracker.on_window_boundary(_W0)
        assert tracker.is_frozen is False

        # Boundary N → N+1 advances the pointer. Trade still pending.
        tracker.on_window_boundary(_W1)
        assert tracker.is_frozen is False

        # Mid-N+1: gamma resolves the loss that was booked in N.
        armed = tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W0)
        assert armed is True

        # Boundary N+1 → N+2. Under the NEW semantics N+2 IS frozen.
        tracker.on_window_boundary(_W2)
        assert tracker.is_frozen is True, "N+2 must be frozen — regression from 2026-04-22 T4"

        # Boundary N+2 → N+3. Cooldown expires.
        tracker.on_window_boundary(_W3)
        assert tracker.is_frozen is False

    def test_live_cooldown_of_two_covers_two_windows_after_current(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=2)
        tracker.on_window_boundary(_W0)
        tracker.on_window_boundary(_W1)  # advanced to N+1 before mid-window resolve

        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W0)

        tracker.on_window_boundary(_W2)
        assert tracker.is_frozen is True
        tracker.on_window_boundary(_W3)
        assert tracker.is_frozen is True
        tracker.on_window_boundary(_W4)
        assert tracker.is_frozen is False


# ---------------------------------------------------------------------------
# Re-arming
# ---------------------------------------------------------------------------


class TestRearming:
    def test_rearm_with_later_window_extends_freeze(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=2)
        tracker.on_window_boundary(_W0)
        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W0)
        # freeze_until = W0 + 3*300 = W3

        tracker.on_window_boundary(_W1)
        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W1)
        # freeze_until = max(W3, W1 + 3*300) = W4. Extended by one window.

        tracker.on_window_boundary(_W3)
        assert tracker.is_frozen is True
        tracker.on_window_boundary(_W4)
        assert tracker.is_frozen is False

    def test_rearm_with_earlier_window_does_not_shrink_freeze(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=2)
        tracker.on_window_boundary(_W2)
        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W2)
        # freeze_until = W2 + 3*300 = W5.

        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W0)
        # anchor = max(W0, W2) = W2; proposed = W5 — unchanged.

        tracker.on_window_boundary(_W3)
        assert tracker.is_frozen is True
        tracker.on_window_boundary(_W4)
        assert tracker.is_frozen is True
        tracker.on_window_boundary(_W5)
        assert tracker.is_frozen is False


class TestBoundaryEdgeCases:
    def test_boundary_with_zero_ts_keeps_not_frozen(self) -> None:
        tracker = PostLossCooldown(enabled=True, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.on_window_boundary(0)
        assert tracker.is_frozen is False

    def test_boundary_disabled_is_noop(self) -> None:
        tracker = PostLossCooldown(enabled=False, loss_pct_threshold=2.0, cooldown_windows=1)
        tracker.on_window_boundary(_W1)
        tracker.register_loss(100.0, 1000.0, resolved_window_ts=_W0)
        assert tracker.is_frozen is False
