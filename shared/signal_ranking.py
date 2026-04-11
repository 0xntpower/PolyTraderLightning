"""Signal ranking logic shared across PolySignalLab.

Used by the standalone smart_ranker CLI tool, psl.py, and as a fallback
when the engine JSON does not yet contain a pre-computed compositeScore.
The formula mirrors the C++ composite score in CrossFoldAggregationStage.
"""

from __future__ import annotations

import math
from typing import Any

# Must match core::kMaxCompositeConfidence in Config.hpp
_MAX_CONFIDENCE = 10.0


# Any: engine JSON with many optional fields, schema defined externally
def calculate_smart_score(signal: dict[str, Any]) -> float:
    """Composite score: EV * sqrt(minFoldWR * foldStrength * confidence) * sampleDepth * 100.

    Mirrors the engine's composite score in CrossFoldAggregationStage.
    If the engine already computed ``compositeScore`` it should be preferred;
    this function exists for standalone tools that rank engine JSON without
    re-running the engine.

    Returns 0 for signals with < 2 folds, no OOS matches, weak p-values,
    or non-positive EV.
    """
    ev = max(0.0, signal.get("evPerTrade", 0))
    if ev <= 0:
        return 0.0

    oos_matches = signal.get("oosMatches", 0)
    bh_p_value = signal.get("oosBhAdjustedPValue", 1.0)

    if oos_matches < 1 or bh_p_value >= 0.10:
        return 0.0

    folds_appeared = signal.get("wfFoldsAppeared", 1)
    total_folds = signal.get("wfTotalTestFolds", 3)
    if total_folds <= 0 or folds_appeared < 2:
        return 0.0

    min_win_rate = signal.get("wfMinFoldWinRatePct", 0) / 100.0

    # Confidence: log10(p-value) normalized against cap of 3 (p=0.001),
    # capped at 1.0 so extreme p-values don't inflate the score.
    confidence_norm_cap = 3.0
    raw_confidence = min(abs(math.log10(max(bh_p_value, 1e-20))), _MAX_CONFIDENCE)
    confidence = min(raw_confidence / confidence_norm_cap, 1.0)

    # Recency-weighted fold strength: each test fold gets linearly
    # increasing weight (oldest=1, newest=N).  Fold indices start at
    # kMinTrain (=2), so position = index - 2.
    first_test_fold = 2  # must match core::kWalkForwardMinTrainFolds
    max_fold_weight_sum = total_folds * (total_folds + 1) / 2.0
    fold_indices = signal.get("wfFoldIndices", [])
    fold_weight_sum = sum(
        (idx - first_test_fold) + 1.0 for idx in fold_indices
    )
    fold_strength = fold_weight_sum / max_fold_weight_sum if max_fold_weight_sum > 0 else 0.0

    stat_strength = min_win_rate * fold_strength * confidence
    sample_depth = min(oos_matches / 20.0, 1.0)

    score = ev * math.sqrt(stat_strength) * sample_depth * 100
    return float(round(score, 2))


# Tier priority — lower number = higher quality.  Used for display/logging only;
# delivery eligibility is determined solely by smart_score >= min_score.
TIER_RANK: dict[str, int] = {
    "GOLDEN": 0,
    "SILVER": 1,
    "BRONZE": 2,
    "OVERFIT_DANGER": 3,
}


def assign_tier(signal: dict[str, Any]) -> str:  # Any: engine JSON, schema defined externally
    """Classify a signal into a quality tier based on survival metrics.

    Tiers:
        GOLDEN       — survived all folds, high matches, strong EV, extreme significance
        SILVER       — survived most folds, decent matches and EV
        OVERFIT_DANGER — appeared in only 1 fold
        BRONZE       — everything else
    """
    folds = signal.get("wfFoldsAppeared", 1)
    total = signal.get("wfTotalTestFolds", 3)
    matches = signal.get("oosMatches", 0)
    ev = signal.get("evPerTrade", 0)
    bh_p = signal.get("oosBhAdjustedPValue", 1.0)

    if folds == total and matches >= 30 and ev > 0.02 and bh_p < 0.0001:
        return "GOLDEN"
    if folds >= total - 1 and matches >= 20 and ev > 0.01 and bh_p < 0.01:
        return "SILVER"
    if folds == 1:
        return "OVERFIT_DANGER"
    return "BRONZE"
