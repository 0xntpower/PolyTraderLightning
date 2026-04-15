"""Track current exposure and P&L per window."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from shared.discord import send_window_summary

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class PositionTracker:
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    total_winnings: float = 0.0
    total_losses: float = 0.0
    windows_traded: int = 0
    windows_won: int = 0
    consecutive_losses: int = 0
    window_exposure_usd: float = 0.0

    def record_window(
        self,
        window_ts: int,
        pnl: float,
        traded: bool = True,
        mode: str = "paper",
        balance: float | None = None,
    ) -> None:
        # Untraded windows must not touch P&L or streak counters. All current
        # call sites pass pnl=0.0 when traded=False; a non-zero value here
        # points at an upstream accounting bug (partial fill, early-exit
        # divergence) and is an error we want to surface rather than silently
        # absorb into daily totals.
        if not traded:
            if pnl != 0.0:
                log.error(
                    "record_window called with traded=False but pnl=%.4f "
                    "(window_ts=%d) — ignoring; investigate upstream",
                    pnl,
                    window_ts,
                )
            # Still log a summary line so the window is represented in logs.
            win_pct = (
                (self.windows_won / self.windows_traded * 100) if self.windows_traded > 0 else 0.0
            )
            log.info(
                "WINDOW_SUMMARY window_pnl=$0.0000 session_pnl=$%.4f "
                "won=$%.4f lost=$%.4f wr=%.1f%% (%d/%d) consec_losses=%d (not traded)",
                self.total_pnl,
                self.total_winnings,
                self.total_losses,
                win_pct,
                self.windows_won,
                self.windows_traded,
                self.consecutive_losses,
            )
            return

        self.daily_pnl += pnl
        self.total_pnl += pnl

        self.windows_traded += 1
        if pnl > 0:
            self.windows_won += 1
            self.total_winnings += pnl
            self.consecutive_losses = 0
        elif pnl < 0:
            self.total_losses += abs(pnl)
            self.consecutive_losses += 1
        # pnl == 0 on a traded window (flat fill after fees) leaves
        # consecutive_losses unchanged — neither win nor loss.

        win_pct = (self.windows_won / self.windows_traded * 100) if self.windows_traded > 0 else 0.0

        log.info(
            "WINDOW_SUMMARY window_pnl=$%.4f session_pnl=$%.4f "
            "won=$%.4f lost=$%.4f wr=%.1f%% (%d/%d) consec_losses=%d",
            pnl,
            self.total_pnl,
            self.total_winnings,
            self.total_losses,
            win_pct,
            self.windows_won,
            self.windows_traded,
            self.consecutive_losses,
        )

        # Only send Discord summary when a trade actually resolved with P&L.
        # (Untraded windows already returned above.)
        if pnl != 0:
            send_window_summary(
                mode=mode,
                window_pnl=pnl,
                session_pnl=self.total_pnl,
                session_won=self.total_winnings,
                session_lost=self.total_losses,
                win_rate_pct=win_pct,
                wins=self.windows_won,
                total=self.windows_traded,
                consecutive_losses=self.consecutive_losses,
                balance=balance,
            )

    def add_exposure(self, usd: float) -> None:
        self.window_exposure_usd += usd

    def remove_exposure(self, usd: float) -> None:
        self.window_exposure_usd = max(0.0, self.window_exposure_usd - usd)

    def reset_window_exposure(self) -> None:
        self.window_exposure_usd = 0.0

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        log.info("daily counters reset")

    def save_state(self, path: Path, date_str: str, signal_id: str = "") -> None:
        """Persist tracker state to a JSON file so restarts don't lose context."""
        state = {
            "date": date_str,
            "signal_id": signal_id,
            "last_updated": datetime.now(UTC).isoformat(),
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "total_winnings": self.total_winnings,
            "total_losses": self.total_losses,
            "windows_traded": self.windows_traded,
            "windows_won": self.windows_won,
            "consecutive_losses": self.consecutive_losses,
        }
        try:
            path.write_text(json.dumps(state, indent=2))
        except OSError as exc:
            log.warning("failed to save tracker state: %s", exc)

    def load_state(self, path: Path, today_str: str, current_signal_id: str = "") -> None:
        """Restore tracker state from a JSON file. Daily fields are reset if date differs.

        Warns if the saved signal_id doesn't match the current signal, or if
        the state is older than 24 hours.
        """
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
            log.warning("failed to load tracker state from %s: %s", path, exc)
            return

        # Stale state detection
        saved_signal = state.get("signal_id", "")
        if current_signal_id and saved_signal and saved_signal != current_signal_id:
            log.warning(
                "STATE SIGNAL MISMATCH: saved signal_id=%s but current=%s — "
                "state may be stale, resetting totals",
                saved_signal,
                current_signal_id,
            )
            return  # don't load mismatched state

        last_updated = state.get("last_updated", "")
        if last_updated:
            try:
                updated_dt = datetime.fromisoformat(last_updated)
                age_hours = (datetime.now(UTC) - updated_dt).total_seconds() / 3600
                if age_hours > 24:
                    log.warning(
                        "state file is %.1f hours old (last_updated=%s) — discarding stale state",
                        age_hours,
                        last_updated,
                    )
                    return
            except (ValueError, TypeError):
                pass

        self.total_pnl = state.get("total_pnl", 0.0)
        self.total_winnings = state.get("total_winnings", 0.0)
        self.total_losses = state.get("total_losses", 0.0)
        self.windows_traded = state.get("windows_traded", 0)
        self.windows_won = state.get("windows_won", 0)

        if state.get("date") == today_str:
            self.daily_pnl = state.get("daily_pnl", 0.0)
            self.consecutive_losses = state.get("consecutive_losses", 0)
            log.info(
                "resumed state: daily_pnl=%.4f total_pnl=%.4f consecutive_losses=%d signal=%s",
                self.daily_pnl,
                self.total_pnl,
                self.consecutive_losses,
                saved_signal,
            )
        else:
            log.info(
                "new day — daily counters reset; total_pnl=%.4f from previous sessions",
                self.total_pnl,
            )
