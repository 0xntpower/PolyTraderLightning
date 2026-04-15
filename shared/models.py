"""Shared data structures for signal data across PolySignalLab components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    def label(self) -> str:
        return (
            f"#{self.rank} {self.side} [{self.observe_from_s:.0f}s->{self.observe_to_s:.0f}s] "
            f"d>={self.min_delta_pct}% v<={self.max_variance_pct}%"
        )
