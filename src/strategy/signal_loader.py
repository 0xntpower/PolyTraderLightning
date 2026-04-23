"""Loader and validator for PolySignalEngine momentum signal JSON files.

Signal files (signal_NNN.json) are produced by the PolySignalEngine and look like:

    {
        "rank":             1,
        "side":             "Up",
        "observeFromS":     250.0,
        "observeToS":       130.0,
        "minDeltaPct":      0.10,
        "maxVariancePct":   0.050,
        "trainWins":        14,
        "trainMatches":     15,
        "trainWinRatePct":  93.33,
        "oosWins":          16,
        "oosMatches":       16,
        "oosWinRatePct":    100.00,
        "bhAdjustedPValue": 3.13e-07
    }

All threshold values (minDeltaPct, maxVariancePct) are in PERCENT units, matching
the engine's bnDirectionFromOpenPct field (also in percent). The bot's
signal.bn_direction_from_open_pct is a fraction; the strategy multiplies by 100.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.errors import SignalError
from strategy.momentum_signal import MomentumSignalConfig
from strategy.signal import Direction, ObiDepth

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------


class SignalValidationError(SignalError):
    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"signal validation failed [{field}]: {message}")


@dataclass
class SignalValidationWarning:
    field: str
    message: str


@dataclass
class SignalValidationResult:
    signal: MomentumSignalConfig
    warnings: list[SignalValidationWarning]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def is_momentum_signal(data: dict[str, Any]) -> bool:  # Any: engine JSON, schema defined externally
    """Return True if data is a PolySignalEngine momentum signal."""
    return "observeFromS" in data and "observeToS" in data and "rank" in data


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_momentum_signal(
    path: Path | str,
    min_oos_win_rate_pct: float = 50.0,
) -> SignalValidationResult:
    """Load and validate a PolySignalEngine signal JSON file.

    Args:
        path:                 Path to signal_NNN.json.
        min_oos_win_rate_pct: Hard-reject signals whose OOS win rate is below this.

    Returns:
        SignalValidationResult.

    Raises:
        SignalValidationError on hard failures.
        FileNotFoundError if path does not exist.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_momentum_signal(data, min_oos_win_rate_pct)


def validate_momentum_signal(
    data: dict[str, Any],  # Any: engine JSON with many optional fields, schema defined externally
    min_oos_win_rate_pct: float = 50.0,
) -> SignalValidationResult:
    warnings: list[SignalValidationWarning] = []

    # Required fields
    for key in (
        "rank",
        "side",
        "observeFromS",
        "observeToS",
        "minDeltaPct",
        "maxVariancePct",
        "oosWinRatePct",
        "bhAdjustedPValue",
    ):
        if key not in data:
            raise SignalValidationError(key, "required field missing")

    # rank
    rank = data["rank"]
    if not isinstance(rank, int) or rank <= 0:
        raise SignalValidationError("rank", f"must be a positive integer, got {rank!r}")

    # side
    side_raw = str(data["side"]).lower().strip()
    if side_raw not in ("up", "down"):
        raise SignalValidationError("side", f"must be 'Up' or 'Down', got {data['side']!r}")
    side = Direction.UP if side_raw == "up" else Direction.DOWN

    # observeFromS / observeToS (seconds remaining — from > to)
    observe_from = _finite_float("observeFromS", data["observeFromS"])
    observe_to = _finite_float("observeToS", data["observeToS"])

    if observe_from <= 0:
        raise SignalValidationError("observeFromS", f"must be positive, got {observe_from}")
    if observe_to <= 0:
        raise SignalValidationError("observeToS", f"must be positive, got {observe_to}")
    if observe_to >= observe_from:
        raise SignalValidationError(
            "observeToS",
            f"observeToS ({observe_to}s) must be less than observeFromS ({observe_from}s)",
        )
    if observe_from > 295:
        warnings.append(
            SignalValidationWarning(
                "observeFromS",
                f"observeFromS={observe_from}s — bot may miss ticks if the open price is not "
                "captured before this point in the window",
            )
        )
    if observe_to < 10:
        warnings.append(
            SignalValidationWarning(
                "observeToS",
                f"observeToS={observe_to}s is very close to window end — "
                "the maker order may not fill in time",
            )
        )

    # minDeltaPct — in percent units (0.10 = 0.10% move)
    min_delta = _finite_float("minDeltaPct", data["minDeltaPct"])
    if min_delta < 0:
        raise SignalValidationError("minDeltaPct", "must be non-negative")
    if min_delta > 10.0:
        warnings.append(
            SignalValidationWarning(
                "minDeltaPct",
                f"minDeltaPct={min_delta}% is very large — this signal may rarely fire",
            )
        )

    # maxVariancePct — in percent units (stddev of bnDirectionFromOpenPct)
    max_variance = _finite_float("maxVariancePct", data["maxVariancePct"])
    if max_variance < 0:
        raise SignalValidationError("maxVariancePct", "must be non-negative")

    # oosWinRatePct — hard reject below threshold
    oos_wr = _finite_float("oosWinRatePct", data["oosWinRatePct"])
    if oos_wr < min_oos_win_rate_pct:
        raise SignalValidationError(
            "oosWinRatePct",
            f"{oos_wr:.1f}% is below the minimum {min_oos_win_rate_pct:.1f}% — "
            "signal rejected to protect capital",
        )

    # bhAdjustedPValue — warning if not significant
    bh_p = _finite_float("bhAdjustedPValue", data["bhAdjustedPValue"])
    if bh_p > 0.05:
        warnings.append(
            SignalValidationWarning(
                "bhAdjustedPValue",
                f"BH-adjusted p={bh_p:.4g} > 0.05 — "
                "signal may not be statistically distinguishable from noise",
            )
        )

    # Optional fields
    train_wr = float(data.get("trainWinRatePct", 0.0))
    oos_matches = int(data.get("oosMatches", 0))
    if oos_matches < 5:
        warnings.append(
            SignalValidationWarning(
                "oosMatches",
                f"only {oos_matches} OOS matching windows — win rate estimate has high variance",
            )
        )

    # Optional entry pricing fields from the engine
    avg_entry_price: float | None = None
    if "avgEntryPrice" in data:
        avg_entry_price = _finite_float("avgEntryPrice", data["avgEntryPrice"])
        if avg_entry_price <= 0 or avg_entry_price >= 1.0:
            warnings.append(
                SignalValidationWarning(
                    "avgEntryPrice",
                    f"avgEntryPrice={avg_entry_price} outside (0, 1) — will be ignored",
                )
            )
            avg_entry_price = None

    ev_per_trade: float | None = None
    if "evPerTrade" in data:
        ev_per_trade = _finite_float("evPerTrade", data["evPerTrade"])

    conservative_win_rate_pct: float | None = None
    if "conservativeWinRatePct" in data:
        conservative_win_rate_pct = _finite_float(
            "conservativeWinRatePct",
            data["conservativeWinRatePct"],
        )

    smart_score = float(data.get("smartScore", data.get("smart_score", 0.0)) or 0.0)
    wf_folds_appeared = int(data.get("wfFoldsAppeared", 0) or 0)
    wf_total_test_folds = int(data.get("wfTotalTestFolds", 0) or 0)
    wf_fold_indices = [int(i) for i in data.get("wfFoldIndices", [])]

    # v3.4: per-signal OBI threshold from engine. 0.0 = gate disabled;
    # otherwise fire requires |bnObi| >= this with the right sign.
    obi_threshold = _finite_float("obiThreshold", data["obiThreshold"])

    # Per-signal OBI depth: "D10", "D20", or "none". Engine emits "none"
    # whenever obiThreshold == 0.0 (depth is meaningless when gate is off).
    # Older signals without the field default to "none" and the threshold
    # must therefore also be 0 — reject the combination where threshold is
    # enabled but depth is missing, since the bot would not know which
    # depth column to read from the live Binance feed.
    obi_depth_raw = data.get("obiDepth", "none")
    if not isinstance(obi_depth_raw, str):
        raise SignalValidationError(
            "obiDepth",
            f"must be a string ('D10', 'D20', or 'none'), got {type(obi_depth_raw).__name__}",
        )
    obi_depth_key = obi_depth_raw.strip().upper()
    if obi_depth_key in ("NONE", ""):
        obi_depth = ObiDepth.NONE
    elif obi_depth_key == "D10":
        obi_depth = ObiDepth.D10
    elif obi_depth_key == "D20":
        obi_depth = ObiDepth.D20
    else:
        raise SignalValidationError(
            "obiDepth",
            f"unrecognized value {obi_depth_raw!r} — expected 'D10', 'D20', or 'none'",
        )
    if obi_threshold > 0.0 and obi_depth is ObiDepth.NONE:
        raise SignalValidationError(
            "obiDepth",
            f"obiThreshold={obi_threshold} > 0 requires obiDepth of 'D10' or 'D20'",
        )
    if obi_threshold == 0.0 and obi_depth is not ObiDepth.NONE:
        warnings.append(
            SignalValidationWarning(
                "obiDepth",
                f"obiThreshold=0 with obiDepth={obi_depth.value} — depth is ignored "
                "because the OBI gate is disabled",
            )
        )

    # Post-fire erosion threshold from engine (nested in "postFire" object)
    post_fire_max_safe_erosion_pct: float | None = None
    post_fire_raw = data.get("postFire")
    if isinstance(post_fire_raw, dict) and "maxSafeErosionPct" in post_fire_raw:
        post_fire_max_safe_erosion_pct = _finite_float(
            "postFire.maxSafeErosionPct",
            post_fire_raw["maxSafeErosionPct"],
        )
        if post_fire_max_safe_erosion_pct <= 0:
            warnings.append(
                SignalValidationWarning(
                    "postFire.maxSafeErosionPct",
                    f"value={post_fire_max_safe_erosion_pct} is non-positive — will be ignored",
                )
            )
            post_fire_max_safe_erosion_pct = None

    # v3.7: orchestrator-tracked family age and aggregate lifetime p80.
    # All three fields are optional and may be None (bootstrap phase,
    # disabled tracker, older orchestrator). Accept missing / null /
    # non-numeric gracefully — there is nothing the bot does with these
    # values except surface them on Discord.
    signal_age_h = _optional_float_field(data.get("signalAgeH"))
    est_max_lifetime_h = _optional_float_field(data.get("estMaxLifetimeH"))
    lifetime_samples = _optional_int_field(data.get("lifetimeSamples"))

    cfg = MomentumSignalConfig(
        rank=rank,
        side=side,
        observe_from_s=observe_from,
        observe_to_s=observe_to,
        min_delta_pct=min_delta,
        max_variance_pct=max_variance,
        train_win_rate_pct=train_wr,
        oos_win_rate_pct=oos_wr,
        bh_adjusted_p_value=bh_p,
        oos_matches=oos_matches,
        conservative_win_rate_pct=conservative_win_rate_pct,
        avg_entry_price=avg_entry_price,
        ev_per_trade=ev_per_trade,
        smart_score=smart_score,
        wf_folds_appeared=wf_folds_appeared,
        wf_total_test_folds=wf_total_test_folds,
        wf_fold_indices=wf_fold_indices,
        post_fire_max_safe_erosion_pct=post_fire_max_safe_erosion_pct,
        obi_threshold=obi_threshold,
        obi_depth=obi_depth,
        signal_age_h=signal_age_h,
        est_max_lifetime_h=est_max_lifetime_h,
        lifetime_samples=lifetime_samples,
    )

    # --- Completeness check: every field the engine produces must arrive ---
    # If any expected field is None, the IPC or engine pipeline dropped data.
    expected_fields: dict[str, object] = {
        "conservative_win_rate_pct": conservative_win_rate_pct,
        "avg_entry_price": avg_entry_price,
        "ev_per_trade": ev_per_trade,
        "post_fire_max_safe_erosion_pct": post_fire_max_safe_erosion_pct,
    }
    missing = [name for name, val in expected_fields.items() if val is None]
    if missing:
        log.warning(
            "SIGNAL INCOMPLETE — the following fields are None for rank=%d %s: [%s]. "
            "This means the engine or IPC pipeline failed to deliver them. "
            "Features that depend on these fields are DISABLED.",
            rank,
            side_raw,
            ", ".join(missing),
        )

    return SignalValidationResult(signal=cfg, warnings=warnings)


def _finite_float(field: str, value: object) -> float:
    if not isinstance(value, (int, float)):
        raise SignalValidationError(field, f"must be a number, got {type(value).__name__}")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise SignalValidationError(field, "must be a finite number")
    return f


def _optional_float_field(value: object) -> float | None:
    """Lenient float parser for v3.7 lifetime fields.

    Accepts missing (None), null JSON, or any non-numeric input by
    returning None rather than raising — these fields are observational
    and the bot must never reject a delivery because the orchestrator
    couldn't compute them.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _optional_int_field(value: object) -> int | None:
    """Lenient int parser for v3.7 lifetime_samples."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    return None
