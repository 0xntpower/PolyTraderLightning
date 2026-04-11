"""Kelly Criterion sizing for Polymarket binary outcome trades.

Replaces the old multiplicative vol_factor * chop_factor regime scaling
with a mathematically grounded approach:
  1. Regime metrics (vol stddev, chop flips) discount the base win rate.
  2. Kelly formula converts (adjusted_p, entry_price) → optimal bet fraction.
  3. Fractional Kelly + hard limits → final dollar bet size.

Kelly for binaries:  f* = p / entry - (1 - p) / (1 - entry)
If f* ≤ 0, the trade has no edge at this price — don't fire.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import deque
    from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KELLY_WIN_RATE_FLOOR = 0.50  # never estimate p below coin flip
KELLY_WIN_RATE_CEILING_BONUS = 0.05  # never estimate p more than 5% above base
KELLY_FEEDBACK_DAMPENING = 0.3  # how much recent performance influences p
KELLY_FEEDBACK_CLAMP = 0.05  # max +/- adjustment from feedback
KELLY_MIN_OUTCOMES_FOR_FEEDBACK = 10
KELLY_OUTCOME_WINDOW_SIZE = 20  # rolling deque size for recent outcomes


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KellyResult:
    raw_kelly: float  # f* from the formula (can be negative)
    fractional_kelly: float  # f* * kelly_fraction
    bet_size: float  # actual dollar amount to bet (0 if no edge)
    has_edge: bool  # True if raw_kelly > 0
    implied_ev: float  # p * (1 - entry) - (1 - p) * entry


@dataclass(frozen=True, slots=True)
class AdjustedWinRateResult:
    adjusted_p: float  # final adjusted win probability
    vol_discount: float  # individual vol contribution (for logging)
    chop_discount: float  # individual chop contribution (for logging)
    outcome_discount: float  # individual outcome bias contribution (for logging)
    total_discount: float  # combined discount actually applied
    feedback_adjustment: float  # adjustment from recent performance
    regime_ready: bool
    vol_severity: float = 0.0  # 0-1 severity for logging
    chop_severity: float = 0.0
    outcome_severity: float = 0.0


# ---------------------------------------------------------------------------
# Small-sample win rate correction
# ---------------------------------------------------------------------------

_WILSON_Z = 1.5  # Must match kWilsonZ in engine Config.hpp


def wilson_lower_bound(wins: int, n: int) -> float:
    """Wilson score interval lower bound for a binomial proportion.

    With small OOS sample sizes (e.g. 16/16 = 100%) the observed win rate
    overstates the true probability.  The Wilson lower bound gives a
    conservative estimate that converges to the observed rate as n grows.

    Uses z = 1.5 — conservative enough that Kelly naturally rejects
    expensive entries (entry > ~0.93) for typical sample sizes, while
    preserving edge on cheap entries with strong signals.

    Examples (z=1.5):
        16/16  → 93.6%   (observed 100%, shrink 6.4%)
        29/30  → 92.7%   (observed 96.7%, shrink 4.0%)
        50/50  → 96.4%   (observed 100%, shrink 3.6%)
        87/100 → 82.8%   (observed 87%, shrink 4.2%)
    """
    if n <= 0:
        return 0.5
    z = _WILSON_Z
    z2 = z * z
    p_hat = wins / n
    denom = 1.0 + z2 / n
    centre = p_hat + z2 / (2.0 * n)
    spread = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - spread) / denom)


def conservative_win_rate(
    oos_win_rate_pct: float,
    oos_matches: int,
    max_shrink_pct: float = 3.0,
) -> float:
    """Convert an observed OOS win rate to a conservative probability estimate.

    Returns a fraction (0-1).  Uses Wilson lower bound when sample size is
    small; converges to observed rate for large samples.  The correction is
    capped at ``max_shrink_pct`` percentage points so small-sample signals
    are never penalised too aggressively.
    """
    observed = oos_win_rate_pct / 100.0
    wins = round(observed * oos_matches)
    wilson = wilson_lower_bound(wins, oos_matches)
    floor = observed - max_shrink_pct / 100.0
    return max(wilson, floor)


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------


def kelly_size(
    p: float,
    entry_price: float,
    bankroll: float,
    kelly_fraction: float,
    min_bet: float,
    max_bet: float,
) -> KellyResult:
    """Compute Kelly-optimal bet size for a binary outcome trade.

    Parameters
    ----------
    p : float
        Regime-adjusted win probability.
    entry_price : float
        Price we're paying (best_ask for taker orders).
    bankroll : float
        Current bankroll.
    kelly_fraction : float
        Fractional Kelly (e.g. 0.25 = quarter-Kelly).
    min_bet : float
        Minimum bet size — below this, don't trade.
    max_bet : float
        Maximum bet size (hard cap).
    """
    if bankroll <= 0:
        log.warning("kelly_size: bankroll=$%.2f — cannot size bets", bankroll)
        return KellyResult(0.0, 0.0, 0.0, False, 0.0)

    # Guard: entry_price must be in (0, 1) — Polymarket binary market constraint.
    # At price >= 1.0 the denominator (1 - entry_price) is zero or negative.
    if entry_price <= 0.0 or entry_price >= 1.0:
        log.warning("kelly_size: invalid entry_price=%.4f — must be in (0,1)", entry_price)
        return KellyResult(0.0, 0.0, 0.0, False, 0.0)

    raw_kelly = (p - entry_price) / (1.0 - entry_price)
    implied_ev = p * (1.0 - entry_price) - (1.0 - p) * entry_price

    if raw_kelly <= 0.0:
        return KellyResult(raw_kelly, 0.0, 0.0, False, implied_ev)

    fractional = raw_kelly * kelly_fraction
    bet_size = bankroll * fractional

    # Apply hard limits
    bet_size = min(bet_size, max_bet)
    if bet_size < min_bet:
        bet_size = 0.0  # not worth the execution overhead

    return KellyResult(raw_kelly, fractional, bet_size, True, implied_ev)


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


def _linear_severity(value: float, baseline: float, elevated: float) -> float:
    """Compute severity 0-1 via linear interpolation.

    At baseline → 0 (calm), at elevated → 1 (extreme).
    """
    if elevated <= baseline:
        return 0.0
    return max(0.0, min(1.0, (value - baseline) / (elevated - baseline)))


def _linear_severity_inverted(value: float, baseline: float, elevated: float) -> float:
    """Inverted severity: low value = high severity.

    For outcome_agreement: baseline=0.50 (50% agreement, calm),
    elevated=0.15 (15% agreement, extreme). Lower value → higher severity.
    """
    if baseline <= elevated:
        return 0.0
    return max(0.0, min(1.0, (baseline - value) / (baseline - elevated)))


# ---------------------------------------------------------------------------
# Regime-adjusted win rate estimation
# ---------------------------------------------------------------------------


def estimate_adjusted_win_rate(
    base_win_rate: float,
    vol_stddev: float,
    chop_avg_flips: float,
    outcome_agreement: float,
    vol_baseline: float,
    vol_elevated: float,
    chop_baseline: float,
    chop_elevated: float,
    outcome_baseline: float,
    outcome_elevated: float,
    max_discount: float,
    vol_weight: float,
    chop_weight: float,
    outcome_weight: float,
    regime_ready: bool,
    recent_outcomes: deque[int],
    min_outcomes_for_feedback: int,
) -> AdjustedWinRateResult:
    """Convert regime metrics into an adjusted win probability.

    Uses a max-weighted severity formula — the worst single factor
    drives the discount.  Moderate readings across multiple factors
    do not compound into false alarms:

        total_discount = max_discount * max(severity_i * weight_i)

    Parameters
    ----------
    base_win_rate : float
        Signal's OOS win rate (e.g. 0.873).
    vol_stddev : float
        Current volatility stddev from regime tracker.
    chop_avg_flips : float
        Current chop avg flips from regime tracker.
    outcome_agreement : float
        Fraction of recent windows resolving in the signal's direction (0-1).
    vol_baseline / vol_elevated : float
        Calm / elevated volatility thresholds.
    chop_baseline / chop_elevated : float
        Calm / elevated chop thresholds.
    outcome_baseline / outcome_elevated : float
        Agreement thresholds (inverted: baseline is high, elevated is low).
    max_discount : float
        Maximum win rate reduction (e.g. 0.25).
    vol_weight / chop_weight / outcome_weight : float
        Per-factor weight (0-1) controlling how much of the budget each factor
        can consume independently.
    regime_ready : bool
        Whether regime has enough data.
    recent_outcomes : deque
        Last N resolved trade outcomes (1=win, 0=loss) for feedback.
    min_outcomes_for_feedback : int
        Minimum trades before using feedback.
    """
    if not regime_ready:
        return AdjustedWinRateResult(base_win_rate, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    # Step 1: Compute individual severities (0-1)
    vol_severity = _linear_severity(vol_stddev, vol_baseline, vol_elevated)
    chop_severity = _linear_severity(chop_avg_flips, chop_baseline, chop_elevated)
    outcome_severity = _linear_severity_inverted(
        outcome_agreement,
        outcome_baseline,
        outcome_elevated,
    )

    # Step 2: Max-weighted combination — worst factor drives the discount,
    # moderate factors don't compound into false alarms.
    vol_contrib = vol_severity * vol_weight
    chop_contrib = chop_severity * chop_weight
    outcome_contrib = outcome_severity * outcome_weight
    total_discount = max_discount * max(vol_contrib, chop_contrib, outcome_contrib)

    # Step 4: Performance feedback (optional, requires enough data)
    feedback_adjustment = 0.0
    if len(recent_outcomes) >= min_outcomes_for_feedback:
        recent_wr = sum(recent_outcomes) / len(recent_outcomes)
        feedback_adjustment = (recent_wr - base_win_rate) * KELLY_FEEDBACK_DAMPENING
        feedback_adjustment = max(
            -KELLY_FEEDBACK_CLAMP,
            min(KELLY_FEEDBACK_CLAMP, feedback_adjustment),
        )

    adjusted_p = base_win_rate - total_discount + feedback_adjustment

    # Safety clamp: never below coin flip, never above base + ceiling bonus
    adjusted_p = max(
        KELLY_WIN_RATE_FLOOR,
        min(base_win_rate + KELLY_WIN_RATE_CEILING_BONUS, adjusted_p),
    )

    return AdjustedWinRateResult(
        adjusted_p=adjusted_p,
        vol_discount=vol_contrib,
        chop_discount=chop_contrib,
        outcome_discount=outcome_contrib,
        total_discount=total_discount,
        feedback_adjustment=feedback_adjustment,
        regime_ready=True,
        vol_severity=vol_severity,
        chop_severity=chop_severity,
        outcome_severity=outcome_severity,
    )


# ---------------------------------------------------------------------------
# Bankroll persistence
# ---------------------------------------------------------------------------


class BankrollTracker:
    """Tracks bankroll across trades. Persists to disk atomically.

    Bankroll updates are called only from the main asyncio event loop —
    no locks needed.
    """

    def __init__(self, initial_bankroll: float, path: Path) -> None:
        self._bankroll = initial_bankroll
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    @property
    def bankroll(self) -> float:
        return self._bankroll

    def update_win(self, size: float, entry_price: float, fee: float = 0.0) -> float:
        """Update bankroll after a winning trade. Returns new bankroll."""
        shares = size / entry_price
        profit = shares * (1.0 - entry_price) - fee
        self._bankroll += profit
        self._save()
        return self._bankroll

    def update_loss(self, size: float, entry_price: float, fee: float = 0.0) -> float:
        """Update bankroll after a losing trade. Returns new bankroll."""
        shares = size / entry_price
        loss = shares * entry_price + fee
        self._bankroll -= loss
        self._save()
        return self._bankroll

    def sync_from_api(self, api_balance: float) -> float:
        """Sync bankroll with on-chain balance. Returns drift amount.

        If there is meaningful drift (>$0.01), updates bankroll to match
        the API value and persists to disk.
        """
        drift = api_balance - self._bankroll
        if abs(drift) > 0.01:
            log.info(
                "bankroll sync: local=$%.2f api=$%.2f drift=%+.2f — updating",
                self._bankroll,
                api_balance,
                drift,
            )
            self._bankroll = api_balance
            self._save()
        return drift

    def reset(self, new_bankroll: float) -> None:
        """Reset bankroll to a new value."""
        self._bankroll = new_bankroll
        self._save()

    def _load(self) -> None:
        """Load bankroll from disk if file exists."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._bankroll = float(data.get("bankroll", self._bankroll))
                log.info("Loaded bankroll from %s: $%.2f", self._path, self._bankroll)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                log.warning(
                    "Failed to load bankroll from %s: %s — using initial value $%.2f",
                    self._path,
                    e,
                    self._bankroll,
                )

    def _save(self) -> None:
        """Write bankroll to disk atomically (write tmp, then rename)."""
        data = json.dumps({"bankroll": round(self._bankroll, 4)})
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent),
                prefix=".bankroll_",
                suffix=".tmp",
            )
            try:
                os.write(fd, data.encode())
                os.close(fd)
                # On Windows, target must not exist for os.rename
                if os.name == "nt" and self._path.exists():
                    os.replace(tmp_path, str(self._path))
                else:
                    os.rename(tmp_path, str(self._path))
            except OSError:
                with suppress(OSError):
                    os.close(fd)
                with suppress(OSError):
                    os.unlink(tmp_path)
                raise
        except OSError as e:
            log.warning("Failed to save bankroll to %s: %s", self._path, e)
