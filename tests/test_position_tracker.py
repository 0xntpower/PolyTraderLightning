"""Tests for PositionTracker — window P&L accounting and state persistence.

Covers:
- Traded win/loss/flat window accounting
- Untraded windows do not touch P&L or consecutive losses (Item 1 hardening)
- Untraded windows with non-zero pnl log an error and leave state untouched
- Exposure helpers
- save_state / load_state roundtrip
- Date rollover resets daily counters
- Signal-id mismatch refuses to load
- Stale-state (>24h) refuses to load
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from risk.position_tracker import PositionTracker

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _silence_discord():
    """Stop Discord webhooks from firing during tests."""
    with patch("risk.position_tracker.send_window_summary"):
        yield


# ---------------------------------------------------------------------------
# Window accounting — traded
# ---------------------------------------------------------------------------


class TestTradedWindow:
    def test_winning_window(self) -> None:
        t = PositionTracker()
        t.record_window(1, pnl=2.50, traded=True)

        assert t.daily_pnl == 2.50
        assert t.total_pnl == 2.50
        assert t.total_winnings == 2.50
        assert t.total_losses == 0.0
        assert t.windows_traded == 1
        assert t.windows_won == 1
        assert t.consecutive_losses == 0

    def test_losing_window(self) -> None:
        t = PositionTracker()
        t.record_window(1, pnl=-1.25, traded=True)

        assert t.daily_pnl == -1.25
        assert t.total_pnl == -1.25
        assert t.total_winnings == 0.0
        assert t.total_losses == 1.25
        assert t.windows_traded == 1
        assert t.windows_won == 0
        assert t.consecutive_losses == 1

    def test_flat_traded_window_does_not_change_streak(self) -> None:
        """A fill that broke exactly even (after fees) is neither win nor loss."""
        t = PositionTracker()
        t.consecutive_losses = 3
        t.record_window(1, pnl=0.0, traded=True)

        # Counted as traded...
        assert t.windows_traded == 1
        # ...but doesn't increment OR reset the streak.
        assert t.windows_won == 0
        assert t.consecutive_losses == 3
        assert t.total_winnings == 0.0
        assert t.total_losses == 0.0

    def test_win_resets_consecutive_loss_streak(self) -> None:
        t = PositionTracker()
        t.record_window(1, pnl=-1.0, traded=True)
        t.record_window(2, pnl=-1.0, traded=True)
        t.record_window(3, pnl=-1.0, traded=True)
        assert t.consecutive_losses == 3

        t.record_window(4, pnl=0.5, traded=True)
        assert t.consecutive_losses == 0
        assert t.windows_won == 1
        assert t.windows_traded == 4

    def test_cumulative_session_totals(self) -> None:
        t = PositionTracker()
        t.record_window(1, pnl=1.0, traded=True)
        t.record_window(2, pnl=-0.4, traded=True)
        t.record_window(3, pnl=0.8, traded=True)

        assert t.total_pnl == pytest.approx(1.4)
        assert t.daily_pnl == pytest.approx(1.4)
        assert t.total_winnings == pytest.approx(1.8)
        assert t.total_losses == pytest.approx(0.4)
        assert t.windows_traded == 3
        assert t.windows_won == 2


# ---------------------------------------------------------------------------
# Window accounting — untraded (Item 1 hardening)
# ---------------------------------------------------------------------------


class TestUntradedWindow:
    def test_untraded_zero_pnl_is_noop(self) -> None:
        """The common case: no fill, pnl=0 — tracker state is untouched."""
        t = PositionTracker()
        t.daily_pnl = 5.0
        t.total_pnl = 5.0
        t.windows_traded = 3
        t.windows_won = 2
        t.consecutive_losses = 0

        t.record_window(1, pnl=0.0, traded=False)

        # No field should have moved.
        assert t.daily_pnl == 5.0
        assert t.total_pnl == 5.0
        assert t.windows_traded == 3
        assert t.windows_won == 2
        assert t.consecutive_losses == 0

    def test_untraded_does_not_increment_consecutive_losses(self) -> None:
        """Regression: prior code incremented streak on untraded negative pnl."""
        t = PositionTracker()
        t.consecutive_losses = 0

        t.record_window(1, pnl=0.0, traded=False)
        t.record_window(2, pnl=0.0, traded=False)
        t.record_window(3, pnl=0.0, traded=False)

        assert t.consecutive_losses == 0

    def test_untraded_with_nonzero_pnl_logs_error_and_ignores(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An upstream bug should surface — we must not silently absorb it."""
        t = PositionTracker()
        baseline_pnl = 2.0
        t.total_pnl = baseline_pnl
        t.daily_pnl = baseline_pnl

        with caplog.at_level("ERROR", logger="risk.position_tracker"):
            t.record_window(1, pnl=-0.75, traded=False)

        # State unchanged — the bogus pnl was NOT added to totals.
        assert t.total_pnl == baseline_pnl
        assert t.daily_pnl == baseline_pnl
        assert t.windows_traded == 0
        assert t.consecutive_losses == 0
        # Error was logged.
        assert any("traded=False" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Exposure helpers
# ---------------------------------------------------------------------------


class TestExposure:
    def test_add_and_remove(self) -> None:
        t = PositionTracker()
        t.add_exposure(50.0)
        t.add_exposure(25.0)
        assert t.window_exposure_usd == 75.0

        t.remove_exposure(30.0)
        assert t.window_exposure_usd == 45.0

    def test_remove_clamps_at_zero(self) -> None:
        """Removing more than current exposure must not go negative."""
        t = PositionTracker()
        t.add_exposure(10.0)
        t.remove_exposure(50.0)
        assert t.window_exposure_usd == 0.0

    def test_reset_window_exposure(self) -> None:
        t = PositionTracker()
        t.add_exposure(100.0)
        t.reset_window_exposure()
        assert t.window_exposure_usd == 0.0


# ---------------------------------------------------------------------------
# Daily reset
# ---------------------------------------------------------------------------


class TestResetDaily:
    def test_reset_daily_clears_daily_and_streak_only(self) -> None:
        t = PositionTracker()
        t.record_window(1, pnl=-1.0, traded=True)
        t.record_window(2, pnl=-1.0, traded=True)
        t.total_pnl = 10.0  # pretend previous days exist

        t.reset_daily()

        assert t.daily_pnl == 0.0
        assert t.consecutive_losses == 0
        # Session totals preserved across day boundary.
        assert t.total_pnl == 10.0
        assert t.windows_traded == 2


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        today = datetime.now(UTC).strftime("%Y-%m-%d")

        t1 = PositionTracker()
        t1.record_window(1, pnl=1.5, traded=True)
        t1.record_window(2, pnl=-0.5, traded=True)
        t1.save_state(state_file, today, signal_id="up_240_180")

        t2 = PositionTracker()
        t2.load_state(state_file, today, current_signal_id="up_240_180")

        assert t2.daily_pnl == pytest.approx(1.0)
        assert t2.total_pnl == pytest.approx(1.0)
        assert t2.total_winnings == pytest.approx(1.5)
        assert t2.total_losses == pytest.approx(0.5)
        assert t2.windows_traded == 2
        assert t2.windows_won == 1

    def test_load_nonexistent_is_noop(self, tmp_path: Path) -> None:
        t = PositionTracker()
        t.load_state(tmp_path / "missing.json", "2026-04-15")
        # All defaults preserved.
        assert t.total_pnl == 0.0
        assert t.daily_pnl == 0.0

    def test_load_different_date_clears_daily_preserves_totals(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")

        t1 = PositionTracker()
        t1.record_window(1, pnl=3.0, traded=True)
        t1.save_state(state_file, yesterday, signal_id="up_240_180")

        t2 = PositionTracker()
        t2.load_state(
            state_file,
            datetime.now(UTC).strftime("%Y-%m-%d"),
            current_signal_id="up_240_180",
        )

        # Daily cleared (new day)...
        assert t2.daily_pnl == 0.0
        assert t2.consecutive_losses == 0
        # ...but session totals preserved.
        assert t2.total_pnl == pytest.approx(3.0)
        assert t2.windows_traded == 1

    def test_load_signal_id_mismatch_refuses(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        today = datetime.now(UTC).strftime("%Y-%m-%d")

        t1 = PositionTracker()
        t1.record_window(1, pnl=5.0, traded=True)
        t1.save_state(state_file, today, signal_id="old_signal")

        t2 = PositionTracker()
        t2.load_state(state_file, today, current_signal_id="new_signal")

        # Refused — defaults preserved.
        assert t2.total_pnl == 0.0
        assert t2.windows_traded == 0

    def test_load_stale_state_over_24h_refuses(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        stale_ts = (datetime.now(UTC) - timedelta(hours=30)).isoformat()
        state_file.write_text(
            json.dumps(
                {
                    "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                    "signal_id": "up_240_180",
                    "last_updated": stale_ts,
                    "daily_pnl": 5.0,
                    "total_pnl": 5.0,
                    "total_winnings": 5.0,
                    "total_losses": 0.0,
                    "windows_traded": 2,
                    "windows_won": 2,
                    "consecutive_losses": 0,
                }
            )
        )

        t = PositionTracker()
        t.load_state(state_file, datetime.now(UTC).strftime("%Y-%m-%d"))

        # Refused — defaults preserved.
        assert t.total_pnl == 0.0
        assert t.windows_traded == 0
