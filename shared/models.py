"""Shared data structures for signal data across PolySignalLab components."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _optional_float(raw: object) -> float | None:
    """Parse a JSON value as a non-NaN finite float, or return None.

    The IPC payload may carry explicit nulls for fields the orchestrator
    couldn't compute (bootstrap phase, tracker failure, older orchestrator
    version pre-v3.7). Treat missing / None / non-numeric as None rather
    than silently coercing to 0.0 — downstream code branches on None to
    distinguish "unknown" from "zero".
    """
    if raw is None or isinstance(raw, bool):
        return None
    if not isinstance(raw, (int, float)):
        return None
    f = float(raw)
    if not math.isfinite(f):
        return None
    return f


def _optional_int(raw: object) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not math.isfinite(raw):
            return None
        return int(raw)
    return None


@dataclass
class SignalConfig:
    """A momentum signal configuration as produced by PolySignalEngine."""

    rank: int
    side: str
    observe_from_s: float
    observe_to_s: float
    min_delta_pct: float
    max_variance_pct: float
    train_wins: int = 0
    train_matches: int = 0
    train_win_rate_pct: float = 0.0
    oos_wins: int = 0
    oos_matches: int = 0
    oos_win_rate_pct: float = 0.0
    bh_adjusted_p_value: float = 1.0
    oos_bh_adjusted_p_value: float = 1.0
    lag1_autocorrelation: float = 0.0
    effective_n: int = 0
    wf_folds_appeared: int = 0
    wf_total_test_folds: int = 0
    wf_fold_indices: list[int] = field(default_factory=list)
    wf_min_fold_win_rate_pct: float = 0.0
    wf_max_fold_win_rate_pct: float = 0.0
    conservative_win_rate_pct: float = 0.0
    avg_entry_price: float = 0.0
    ev_per_trade: float = 0.0
    composite_score: float = 0.0
    smart_score: float = 0.0
    tier: str = ""
    post_fire_max_safe_erosion_pct: float = 0.0
    post_fire_win_median_erosion_pct: float = 0.0
    post_fire_win_p90_erosion_pct: float = 0.0
    post_fire_loss_median_erosion_pct: float = 0.0
    post_fire_win_samples: int = 0
    post_fire_loss_samples: int = 0
    # v4: per-signal OBI gate carried end-to-end from engine → orchestrator →
    # bot IPC. Both fields must round-trip so the bot's signal_loader can
    # apply the same gate the engine trained on.
    obi_threshold: float = 0.0
    obi_depth: str = "none"
    # v3.7: signal-family lifetime tracking. Populated by the orchestrator
    # from LifetimeTracker state; passed through for bot Discord surfacing.
    # None during bootstrap phase or when the orchestrator's tracker
    # failed this cycle.
    signal_age_h: float | None = None
    # v3.7 phase-2: typical (median) signal lifetime over the eligible
    # samples in the orchestrator's rolling buffer. Replaces the legacy
    # ``est_max_lifetime_h`` (p80) for the same Discord slot — bot
    # accepts both for one release for backwards compatibility.
    typical_lifetime_h: float | None = None
    typical_lifetime_samples: int | None = None
    typical_lifetime_status: str = "unavailable"  # 'unavailable' | 'tentative' | 'stable'
    # Deprecated v3.7 phase-1 fields — retained while in-flight payloads
    # may still carry them. Read by ``from_engine_json`` for backwards
    # compat; new orchestrators populate ``typical_lifetime_*`` instead.
    est_max_lifetime_h: float | None = None
    lifetime_samples: int | None = None
    # v3.7 phase-1: pool-anchored age. ``first_fire_window_ts`` is the Unix
    # second of the chronologically-earliest window in the engine's rolling
    # pool where this exact signal fired. ``first_fire_window_saturated``
    # is true when that timestamp equals the pool's oldest window — the
    # value is then a *lower bound*, not a measurement (the signal may have
    # been firing before the pool's oldest window was loaded). None when
    # the engine pre-dates v3.7 or the field was missing for any reason.
    first_fire_window_ts: int | None = None
    first_fire_window_saturated: bool = False
    # v3.7 phase-3: when the orchestrator's age-aware selector chose this
    # signal over a top-by-score candidate, ``selected_over`` is the label
    # of the runner-up. Surfaced on the signal-updated Discord embed.
    # None when this signal was the top by score.
    selected_over: str | None = None

    @classmethod
    def from_engine_json(cls, data: dict[str, Any]) -> SignalConfig:
        """Construct from PolySignalEngine JSON signal object."""
        return cls(
            rank=data.get("rank", 0),
            side=data.get("side", ""),
            observe_from_s=data.get("observeFromS", 0.0),
            observe_to_s=data.get("observeToS", 0.0),
            min_delta_pct=data.get("minDeltaPct", 0.0),
            max_variance_pct=data.get("maxVariancePct", 0.0),
            obi_threshold=float(data.get("obiThreshold", 0.0)),
            obi_depth=str(data.get("obiDepth", "none")),
            train_wins=data.get("trainWins", 0),
            train_matches=data.get("trainMatches", 0),
            train_win_rate_pct=data.get("trainWinRatePct", 0.0),
            oos_wins=data.get("oosWins", 0),
            oos_matches=data.get("oosMatches", 0),
            oos_win_rate_pct=data.get("oosWinRatePct", 0.0),
            bh_adjusted_p_value=data.get("bhAdjustedPValue", 1.0),
            oos_bh_adjusted_p_value=data.get("oosBhAdjustedPValue", 1.0),
            lag1_autocorrelation=data.get("lag1Autocorrelation", 0.0),
            effective_n=data.get("effectiveN", 0),
            wf_folds_appeared=data.get("wfFoldsAppeared", 0),
            wf_total_test_folds=data.get("wfTotalTestFolds", 0),
            wf_fold_indices=data.get("wfFoldIndices", []),
            wf_min_fold_win_rate_pct=data.get("wfMinFoldWinRatePct", 0.0),
            wf_max_fold_win_rate_pct=data.get("wfMaxFoldWinRatePct", 0.0),
            conservative_win_rate_pct=data.get("conservativeWinRatePct", 0.0),
            avg_entry_price=data.get("avgEntryPrice", 0.0),
            ev_per_trade=data.get("evPerTrade", 0.0),
            composite_score=data.get("compositeScore", 0.0),
            signal_age_h=_optional_float(data.get("signalAgeH")),
            typical_lifetime_h=_optional_float(data.get("typicalLifetimeH")),
            typical_lifetime_samples=_optional_int(data.get("typicalLifetimeSamples")),
            typical_lifetime_status=str(data.get("typicalLifetimeStatus", "unavailable")),
            est_max_lifetime_h=_optional_float(data.get("estMaxLifetimeH")),
            lifetime_samples=_optional_int(data.get("lifetimeSamples")),
            first_fire_window_ts=_optional_int(data.get("firstFireWindowTs")),
            first_fire_window_saturated=bool(data.get("firstFireWindowSaturated", False)),
            selected_over=(
                str(data["selectedOver"]) if isinstance(data.get("selectedOver"), str) else None
            ),
            **cls._parse_post_fire(data),
        )

    @staticmethod
    def _parse_post_fire(
        data: dict[str, Any],  # Any: engine JSON with externally-defined schema
    ) -> dict[str, Any]:  # Any: mixed float/int values for dataclass fields
        """Extract postFire fields from the nested engine JSON object."""
        pf = data.get("postFire")
        if not isinstance(pf, dict):
            return {}
        return {
            "post_fire_max_safe_erosion_pct": float(pf.get("maxSafeErosionPct", 0.0)),
            "post_fire_win_median_erosion_pct": float(pf.get("winMedianErosionPct", 0.0)),
            "post_fire_win_p90_erosion_pct": float(pf.get("winP90ErosionPct", 0.0)),
            "post_fire_loss_median_erosion_pct": float(pf.get("lossMedianErosionPct", 0.0)),
            "post_fire_win_samples": int(pf.get("winSamples", 0)),
            "post_fire_loss_samples": int(pf.get("lossSamples", 0)),
        }

    def to_ipc_dict(self) -> dict[str, Any]:
        """Serialize for IPC transmission to the trading bot."""
        return {
            "rank": self.rank,
            "side": self.side,
            "observeFromS": self.observe_from_s,
            "observeToS": self.observe_to_s,
            "minDeltaPct": self.min_delta_pct,
            "maxVariancePct": self.max_variance_pct,
            "obiThreshold": self.obi_threshold,
            "obiDepth": self.obi_depth,
            "trainWins": self.train_wins,
            "trainMatches": self.train_matches,
            "trainWinRatePct": self.train_win_rate_pct,
            "oosWins": self.oos_wins,
            "oosMatches": self.oos_matches,
            "oosWinRatePct": self.oos_win_rate_pct,
            "bhAdjustedPValue": self.bh_adjusted_p_value,
            "oosBhAdjustedPValue": self.oos_bh_adjusted_p_value,
            "lag1Autocorrelation": self.lag1_autocorrelation,
            "effectiveN": self.effective_n,
            "wfFoldsAppeared": self.wf_folds_appeared,
            "wfTotalTestFolds": self.wf_total_test_folds,
            "wfFoldIndices": self.wf_fold_indices,
            "wfMinFoldWinRatePct": self.wf_min_fold_win_rate_pct,
            "wfMaxFoldWinRatePct": self.wf_max_fold_win_rate_pct,
            "conservativeWinRatePct": self.conservative_win_rate_pct,
            "avgEntryPrice": self.avg_entry_price,
            "evPerTrade": self.ev_per_trade,
            "compositeScore": self.composite_score,
            "smartScore": self.smart_score,
            "tier": self.tier,
            "postFire": {
                "maxSafeErosionPct": self.post_fire_max_safe_erosion_pct,
                "winMedianErosionPct": self.post_fire_win_median_erosion_pct,
                "winP90ErosionPct": self.post_fire_win_p90_erosion_pct,
                "lossMedianErosionPct": self.post_fire_loss_median_erosion_pct,
                "winSamples": self.post_fire_win_samples,
                "lossSamples": self.post_fire_loss_samples,
            },
            "signalAgeH": self.signal_age_h,
            "typicalLifetimeH": self.typical_lifetime_h,
            "typicalLifetimeSamples": self.typical_lifetime_samples,
            "typicalLifetimeStatus": self.typical_lifetime_status,
            "estMaxLifetimeH": self.est_max_lifetime_h,
            "lifetimeSamples": self.lifetime_samples,
            "firstFireWindowTs": self.first_fire_window_ts,
            "firstFireWindowSaturated": self.first_fire_window_saturated,
            "selectedOver": self.selected_over,
        }

    @property
    def signal_id(self) -> str:
        """Deterministic identifier from signal parameters.

        Side is lowercased so the ID matches the bot's canonical form
        (the bot's signal_loader normalizes side to lowercase on receipt).
        Use ``self.side`` directly for display — ``label`` preserves case.
        """
        return (
            f"{self.side.lower()}_{self.observe_from_s}_{self.observe_to_s}_"
            f"{self.min_delta_pct}_{self.max_variance_pct}"
        )

    @property
    def obi_label(self) -> str:
        """Compact OBI gate label for logs and Discord. ``off`` when the
        gate is disabled; ``{threshold:.2f}@{depth}`` otherwise."""
        if self.obi_threshold <= 0.0:
            return "off"
        return f"{self.obi_threshold:.2f}@{self.obi_depth}"

    @property
    def label(self) -> str:
        return (
            f"#{self.rank} {self.side} [{self.observe_from_s:.0f}s->{self.observe_to_s:.0f}s] "
            f"d>={self.min_delta_pct}% v<={self.max_variance_pct}% obi={self.obi_label}"
        )
