"""Tests for pure helper functions extracted from main.py."""

from __future__ import annotations

from market_data.state import WindowSnapshot
from strategy.momentum_signal import MomentumSignalConfig
from strategy.resolution import PendingResolution, ResolutionManager
from strategy.signal import Direction
from strategy.window_handler import _compute_snapshot_outcome, _window_decision_tag

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_signal_cfg(side: Direction = Direction.UP) -> MomentumSignalConfig:
    return MomentumSignalConfig(
        rank=1, side=side, observe_from_s=120.0, observe_to_s=60.0,
        min_delta_pct=0.05, max_variance_pct=0.10,
        train_win_rate_pct=90.0, oos_win_rate_pct=87.3,
        bh_adjusted_p_value=0.001, oos_matches=100,
    )


def _make_pending(side: Direction = Direction.UP, snapshot_outcome: str | None = "up"):
    return PendingResolution(
        window_ts=1700000000, slug="test", signal_cfg=_make_signal_cfg(side),
        entry_price=0.55, size_usd=10.0, signal_age_windows=5,
        created_at=0, snapshot_outcome=snapshot_outcome,
    )


# ======================================================================
# _window_decision_tag
# ======================================================================


class TestWindowDecisionTag:

    def test_win(self):
        tag, why = _window_decision_tag(True, True, True, True, None)
        assert tag == "[WIN]"
        assert why == ""

    def test_loss(self):
        tag, why = _window_decision_tag(True, False, True, True, None)
        assert tag == "[LOSS]"

    def test_flat(self):
        tag, why = _window_decision_tag(True, None, True, True, None)
        assert tag == "[FLAT]"

    def test_skip_idle(self):
        tag, why = _window_decision_tag(False, None, False, False, "decay")
        assert tag == "[SKIP]"
        assert why == " (idle:decay)"

    def test_skip_no_fire(self):
        tag, why = _window_decision_tag(False, None, False, False, None)
        assert tag == "[SKIP]"
        assert why == " (no fire)"

    def test_skip_conditions_not_met(self):
        tag, why = _window_decision_tag(False, None, True, False, None)
        assert tag == "[SKIP]"
        assert why == " (conditions not met)"

    def test_skip_not_filled(self):
        tag, why = _window_decision_tag(False, None, True, True, None)
        assert tag == "[SKIP]"
        assert why == " (order not filled)"


# ======================================================================
# _force_resolve_outcome
# ======================================================================


class TestForceResolveOutcome:

    def test_with_snapshot(self):
        pr = _make_pending(side=Direction.UP, snapshot_outcome="up")
        assert ResolutionManager._force_resolve_outcome(pr) == "up"

    def test_with_snapshot_down(self):
        pr = _make_pending(side=Direction.UP, snapshot_outcome="down")
        assert ResolutionManager._force_resolve_outcome(pr) == "down"

    def test_no_snapshot_up_side_defaults_loss(self):
        pr = _make_pending(side=Direction.UP, snapshot_outcome=None)
        assert ResolutionManager._force_resolve_outcome(pr) == "down"  # opposite of UP

    def test_no_snapshot_down_side_defaults_loss(self):
        pr = _make_pending(side=Direction.DOWN, snapshot_outcome=None)
        assert ResolutionManager._force_resolve_outcome(pr) == "up"  # opposite of DOWN


# ======================================================================
# _compute_snapshot_outcome
# ======================================================================


class TestComputeSnapshotOutcome:

    def test_none_snapshot(self):
        assert _compute_snapshot_outcome(None) is None

    def test_snapshot_with_movement(self):
        snap = WindowSnapshot(
            window_ts=1700000000,
            chainlink_price=100.10,
            binance_price=100.10,
            open_price=100.00,
            binance_open_price=100.00,
        )
        result = _compute_snapshot_outcome(snap)
        # Prices moved up from open → should be "up"
        assert result == "up"

    def test_snapshot_no_movement(self):
        snap = WindowSnapshot(
            window_ts=1700000000,
            chainlink_price=100.00,
            binance_price=100.00,
            open_price=100.00,
            binance_open_price=100.00,
        )
        result = _compute_snapshot_outcome(snap)
        # No movement → Direction.NONE → None
        assert result is None
