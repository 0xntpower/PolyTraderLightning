"""Adaptive bet sizing based on SPRT confidence and signal age.

Pure functions — no side effects, no global state.

SPRT is a **safety net only** — it stays dormant (1.0) while the orchestrator
is delivering signal updates on schedule.  It only activates when the signal
has gone stale (no update for longer than ``sprt_activation_minutes``).  This
prevents SPRT from second-guessing the orchestrator's statistical validation
during normal operation.

Age taper provides a gentle secondary reduction for very long-lived signals.
"""

from __future__ import annotations


def llr_confidence(llr: float, boundary_alive: float, boundary_dead: float) -> float:
    """Map the SPRT's LLR to a [0, 1] confidence score.

    Returns 1.0 when LLR is at or below the alive boundary (full confidence).
    Returns 0.0 when LLR is at or above the dead boundary (no confidence).
    Linear interpolation between.
    """
    if llr <= boundary_alive:
        return 1.0
    if llr >= boundary_dead:
        return 0.0
    if boundary_dead == boundary_alive:
        return 1.0
    return (boundary_dead - llr) / (boundary_dead - boundary_alive)


def age_taper(
    signal_age_windows: int,
    taper_start: int,
    taper_end: int,
    floor: float,
) -> float:
    """Gentle age-based confidence reduction. Returns [floor, 1.0]."""
    if taper_end <= taper_start:
        return 1.0 if signal_age_windows <= taper_start else floor
    if signal_age_windows <= taper_start:
        return 1.0
    if signal_age_windows >= taper_end:
        return floor
    progress = (signal_age_windows - taper_start) / (taper_end - taper_start)
    return 1.0 - progress * (1.0 - floor)


def compute_bet_scale(
    *,
    llr: float,
    boundary_alive: float,
    boundary_dead: float,
    signal_age_windows: int,
    signal_is_stale: bool,
    taper_start: int = 200,
    taper_end: int = 400,
    age_floor: float = 0.5,
    min_total_scale: float = 0.10,
) -> float:
    """Combined bet scale factor from SPRT confidence and age taper.

    Returns a value in [min_total_scale, 1.0].

    SPRT is dormant (1.0) during normal operation.  It only affects sizing
    when ``signal_is_stale`` is True — meaning the orchestrator hasn't
    delivered a signal update for longer than the configured threshold.
    """
    sprt_conf = llr_confidence(llr, boundary_alive, boundary_dead) if signal_is_stale else 1.0

    age_tap = age_taper(signal_age_windows, taper_start, taper_end, age_floor)
    return max(min_total_scale, sprt_conf * age_tap)
