"""Property-based tests for Kelly sizing math (HRP 8.3).

These tests use Hypothesis to fuzz the mathematical functions with random
inputs, verifying invariants that must hold for ALL valid inputs — not just
the handful of examples in test_kelly.py.
"""

from __future__ import annotations

from collections import deque

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from strategy.kelly import (
    KELLY_OUTCOME_WINDOW_SIZE,
    KELLY_WIN_RATE_CEILING_BONUS,
    KELLY_WIN_RATE_FLOOR,
    _linear_severity,
    _linear_severity_inverted,
    conservative_win_rate,
    estimate_adjusted_win_rate,
    kelly_size,
    wilson_lower_bound,
)

# ---------------------------------------------------------------------------
# Wilson lower bound properties
# ---------------------------------------------------------------------------


@given(
    wins=st.integers(min_value=0, max_value=1000),
    n=st.integers(min_value=1, max_value=1000),
)
def test_wilson_bounded_0_1(wins: int, n: int) -> None:
    """Wilson lower bound must always be in [0, 1]."""
    assume(wins <= n)
    p = wilson_lower_bound(wins, n)
    assert 0.0 <= p <= 1.0


@given(
    wins=st.integers(min_value=0, max_value=1000),
    n=st.integers(min_value=1, max_value=1000),
)
def test_wilson_at_most_observed(wins: int, n: int) -> None:
    """Wilson lower bound must be <= observed win rate (it's conservative)."""
    assume(wins <= n)
    observed = wins / n
    p = wilson_lower_bound(wins, n)
    assert p <= observed + 1e-9  # small epsilon for float imprecision


@given(n=st.integers(min_value=1, max_value=500))
def test_wilson_perfect_record_below_1(n: int) -> None:
    """Even with 100% observed, Wilson should be < 1.0 for finite samples."""
    p = wilson_lower_bound(n, n)
    assert p < 1.0


@given(n=st.integers(min_value=2, max_value=500))
def test_wilson_monotonic_in_wins(n: int) -> None:
    """More wins (same total) → higher Wilson bound."""
    p_low = wilson_lower_bound(n // 2, n)
    p_high = wilson_lower_bound(n - 1, n)
    assert p_high >= p_low


@given(
    wins_frac=st.floats(min_value=0.5, max_value=1.0),
    n1=st.integers(min_value=5, max_value=50),
    n2=st.integers(min_value=100, max_value=500),
)
def test_wilson_tighter_with_more_samples(
    wins_frac: float,
    n1: int,
    n2: int,
) -> None:
    """Same observed rate, larger n → Wilson closer to observed."""
    assume(n2 > n1)
    wins1 = round(wins_frac * n1)
    wins2 = round(wins_frac * n2)
    p1 = wilson_lower_bound(wins1, n1)
    p2 = wilson_lower_bound(wins2, n2)
    obs1 = wins1 / n1
    obs2 = wins2 / n2
    # The gap (observed - wilson) should shrink with more data
    gap1 = obs1 - p1
    gap2 = obs2 - p2
    # Allow some tolerance since rounding can shift the observed rate
    assert gap2 <= gap1 + 0.05


# ---------------------------------------------------------------------------
# Kelly sizing properties
# ---------------------------------------------------------------------------


@given(
    p=st.floats(min_value=0.01, max_value=0.99),
    entry=st.floats(min_value=0.01, max_value=0.99),
    bankroll=st.floats(min_value=1.0, max_value=1e6),
    fraction=st.floats(min_value=0.01, max_value=1.0),
    min_bet=st.floats(min_value=0.0, max_value=10.0),
    max_bet=st.floats(min_value=10.0, max_value=1e6),
)
def test_kelly_bet_never_negative(
    p: float,
    entry: float,
    bankroll: float,
    fraction: float,
    min_bet: float,
    max_bet: float,
) -> None:
    """Bet size must never be negative."""
    kr = kelly_size(p, entry, bankroll, fraction, min_bet, max_bet)
    assert kr.bet_size >= 0.0


@given(
    p=st.floats(min_value=0.01, max_value=0.99),
    entry=st.floats(min_value=0.01, max_value=0.99),
    bankroll=st.floats(min_value=1.0, max_value=1e6),
    fraction=st.floats(min_value=0.01, max_value=1.0),
    max_bet=st.floats(min_value=10.0, max_value=1e6),
)
def test_kelly_bet_never_exceeds_max(
    p: float,
    entry: float,
    bankroll: float,
    fraction: float,
    max_bet: float,
) -> None:
    """Bet size must never exceed max_bet."""
    kr = kelly_size(p, entry, bankroll, fraction, 0.0, max_bet)
    assert kr.bet_size <= max_bet + 1e-9


@given(
    p=st.floats(min_value=0.51, max_value=0.99),
    entry=st.floats(min_value=0.01, max_value=0.50),
    bankroll=st.floats(min_value=100.0, max_value=1e6),
)
def test_kelly_has_edge_when_p_above_entry(
    p: float,
    entry: float,
    bankroll: float,
) -> None:
    """When p > entry, Kelly should find an edge."""
    assume(p > entry + 0.01)
    kr = kelly_size(p, entry, bankroll, 0.25, 0.0, 1e6)
    assert kr.has_edge
    assert kr.raw_kelly > 0


@given(
    p=st.floats(min_value=0.01, max_value=0.49),
    entry=st.floats(min_value=0.50, max_value=0.99),
    bankroll=st.floats(min_value=100.0, max_value=1e6),
)
def test_kelly_no_edge_when_p_below_entry(
    p: float,
    entry: float,
    bankroll: float,
) -> None:
    """When p < entry, Kelly should find no edge."""
    assume(entry > p + 0.01)
    kr = kelly_size(p, entry, bankroll, 0.25, 1.0, 1e6)
    assert not kr.has_edge or kr.raw_kelly <= 0


@given(
    p=st.floats(min_value=0.51, max_value=0.99),
    entry=st.floats(min_value=0.01, max_value=0.50),
    bankroll=st.floats(min_value=100.0, max_value=1e6),
)
def test_kelly_implied_ev_sign_matches_edge(
    p: float,
    entry: float,
    bankroll: float,
) -> None:
    """Implied EV should be positive iff there's an edge."""
    assume(abs(p - entry) > 0.01)
    kr = kelly_size(p, entry, bankroll, 0.25, 0.0, 1e6)
    if kr.has_edge:
        assert kr.implied_ev > -1e-9


# ---------------------------------------------------------------------------
# Severity function properties
# ---------------------------------------------------------------------------


@given(
    value=st.floats(min_value=-10.0, max_value=100.0),
    baseline=st.floats(min_value=0.0, max_value=50.0),
    elevated=st.floats(min_value=0.0, max_value=100.0),
)
def test_severity_bounded_0_1(value: float, baseline: float, elevated: float) -> None:
    """Severity must always be in [0, 1]."""
    assume(elevated > baseline)
    s = _linear_severity(value, baseline, elevated)
    assert 0.0 <= s <= 1.0


@given(
    value=st.floats(min_value=-10.0, max_value=100.0),
    baseline=st.floats(min_value=0.0, max_value=100.0),
    elevated=st.floats(min_value=0.0, max_value=50.0),
)
def test_inverted_severity_bounded_0_1(
    value: float,
    baseline: float,
    elevated: float,
) -> None:
    """Inverted severity must always be in [0, 1]."""
    assume(baseline > elevated)
    s = _linear_severity_inverted(value, baseline, elevated)
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Adjusted win rate properties
# ---------------------------------------------------------------------------


@given(
    base_wr=st.floats(min_value=0.55, max_value=0.99),
    vol=st.floats(min_value=0.0, max_value=1.0),
    chop=st.floats(min_value=0.0, max_value=20.0),
    outcome=st.floats(min_value=0.0, max_value=1.0),
    max_disc=st.floats(min_value=0.01, max_value=0.50),
)
@settings(max_examples=200)
def test_adjusted_wr_floor_at_coin_flip(
    base_wr: float,
    vol: float,
    chop: float,
    outcome: float,
    max_disc: float,
) -> None:
    """Adjusted win rate must never drop below 0.50 (coin flip floor)."""
    result = estimate_adjusted_win_rate(
        base_win_rate=base_wr,
        vol_stddev=vol,
        chop_avg_flips=chop,
        outcome_agreement=outcome,
        vol_baseline=0.10,
        vol_elevated=0.30,
        chop_baseline=3.0,
        chop_elevated=10.0,
        outcome_baseline=0.50,
        outcome_elevated=0.15,
        max_discount=max_disc,
        vol_weight=1.0,
        chop_weight=1.0,
        outcome_weight=0.8,
        regime_ready=True,
        recent_outcomes=deque(maxlen=KELLY_OUTCOME_WINDOW_SIZE),
        min_outcomes_for_feedback=10,
    )
    assert result.adjusted_p >= KELLY_WIN_RATE_FLOOR


@given(
    base_wr=st.floats(min_value=0.55, max_value=0.99),
    vol=st.floats(min_value=0.0, max_value=1.0),
    chop=st.floats(min_value=0.0, max_value=20.0),
    outcome=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200)
def test_adjusted_wr_ceiling(
    base_wr: float,
    vol: float,
    chop: float,
    outcome: float,
) -> None:
    """Adjusted win rate must never exceed base + ceiling bonus."""
    result = estimate_adjusted_win_rate(
        base_win_rate=base_wr,
        vol_stddev=vol,
        chop_avg_flips=chop,
        outcome_agreement=outcome,
        vol_baseline=0.10,
        vol_elevated=0.30,
        chop_baseline=3.0,
        chop_elevated=10.0,
        outcome_baseline=0.50,
        outcome_elevated=0.15,
        max_discount=0.15,
        vol_weight=1.0,
        chop_weight=1.0,
        outcome_weight=0.8,
        regime_ready=True,
        recent_outcomes=deque(maxlen=KELLY_OUTCOME_WINDOW_SIZE),
        min_outcomes_for_feedback=10,
    )
    assert result.adjusted_p <= base_wr + KELLY_WIN_RATE_CEILING_BONUS + 1e-9


@given(base_wr=st.floats(min_value=0.55, max_value=0.99))
def test_calm_regime_returns_base(base_wr: float) -> None:
    """At baseline values (severity=0), adjusted should equal base."""
    result = estimate_adjusted_win_rate(
        base_win_rate=base_wr,
        vol_stddev=0.10,  # at baseline
        chop_avg_flips=3.0,  # at baseline
        outcome_agreement=0.50,  # at baseline
        vol_baseline=0.10,
        vol_elevated=0.30,
        chop_baseline=3.0,
        chop_elevated=10.0,
        outcome_baseline=0.50,
        outcome_elevated=0.15,
        max_discount=0.15,
        vol_weight=1.0,
        chop_weight=1.0,
        outcome_weight=0.8,
        regime_ready=True,
        recent_outcomes=deque(maxlen=KELLY_OUTCOME_WINDOW_SIZE),
        min_outcomes_for_feedback=10,
    )
    assert result.adjusted_p == base_wr


# ---------------------------------------------------------------------------
# Conservative win rate properties
# ---------------------------------------------------------------------------


@given(
    wr_pct=st.floats(min_value=50.0, max_value=100.0),
    n=st.integers(min_value=1, max_value=500),
    max_shrink=st.floats(min_value=0.1, max_value=10.0),
)
def test_conservative_wr_bounded(wr_pct: float, n: int, max_shrink: float) -> None:
    """Conservative win rate must be in [0, 1] and <= observed."""
    p = conservative_win_rate(wr_pct, n, max_shrink)
    assert 0.0 <= p <= 1.0
    assert p <= wr_pct / 100.0 + 1e-9
