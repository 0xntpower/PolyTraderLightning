"""Tests for the v3.7 lifetime-field IPC round-trip.

Phase 2 introduces ``typicalLifetimeH`` / ``typicalLifetimeSamples`` /
``typicalLifetimeStatus`` to replace the legacy ``estMaxLifetimeH`` /
``lifetimeSamples`` slot. The bot accepts both for one release of
orchestrator-bot version skew.

Covers (a) values arrive intact when present, (b) absent / null /
invalid values gracefully become None, (c) legacy payloads still
populate the new fields via fallback.
"""

from __future__ import annotations

import pytest

from shared.models import SignalConfig
from strategy.signal_loader import validate_momentum_signal


def _base_payload(**overrides: object) -> dict[str, object]:
    """Minimal valid signal dict; overrides let tests vary fields."""
    p: dict[str, object] = {
        "rank": 1,
        "side": "up",
        "observeFromS": 240.0,
        "observeToS": 180.0,
        "minDeltaPct": 0.06,
        "maxVariancePct": 0.10,
        "trainWinRatePct": 92.0,
        "oosWinRatePct": 96.0,
        "bhAdjustedPValue": 0.001,
        "oosBhAdjustedPValue": 0.001,
        "oosMatches": 25,
        "conservativeWinRatePct": 93.0,
        "avgEntryPrice": 0.80,
        "evPerTrade": 0.08,
        "wfFoldsAppeared": 4,
        "wfTotalTestFolds": 6,
        "wfFoldIndices": [4, 5, 6, 7],
        "postFire": {"maxSafeErosionPct": 0.5},
        "obiThreshold": 0.0,
        "obiDepth": "none",
    }
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# signal_loader: bot-side ingestion parses new fields
# ---------------------------------------------------------------------------


class TestSignalLoaderTypicalLifetimeFields:
    def test_values_parsed_when_present(self) -> None:
        data = _base_payload(
            signalAgeH=5.2,
            typicalLifetimeH=14.3,
            typicalLifetimeSamples=127,
            typicalLifetimeStatus="stable",
        )
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h == pytest.approx(5.2)
        assert result.signal.typical_lifetime_h == pytest.approx(14.3)
        assert result.signal.typical_lifetime_samples == 127
        assert result.signal.typical_lifetime_status == "stable"

    def test_fields_default_to_none_when_absent(self) -> None:
        """Older orchestrator that doesn't yet send these fields still works."""
        data = _base_payload()
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h is None
        assert result.signal.typical_lifetime_h is None
        assert result.signal.typical_lifetime_samples is None
        assert result.signal.typical_lifetime_status == "unavailable"

    def test_null_values_become_none(self) -> None:
        """Orchestrator explicitly sends None during bootstrap."""
        data = _base_payload(
            signalAgeH=None,
            typicalLifetimeH=None,
            typicalLifetimeSamples=None,
            typicalLifetimeStatus="unavailable",
        )
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h is None
        assert result.signal.typical_lifetime_h is None
        assert result.signal.typical_lifetime_samples is None

    def test_nan_and_infinity_rejected_as_none(self) -> None:
        data = _base_payload(
            signalAgeH=float("nan"),
            typicalLifetimeH=float("inf"),
        )
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h is None
        assert result.signal.typical_lifetime_h is None

    def test_non_numeric_becomes_none(self) -> None:
        data = _base_payload(
            signalAgeH="five hours",
            typicalLifetimeH={"nested": 5},
            typicalLifetimeSamples="lots",
        )
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h is None
        assert result.signal.typical_lifetime_h is None
        assert result.signal.typical_lifetime_samples is None

    def test_int_samples_accepted(self) -> None:
        data = _base_payload(typicalLifetimeSamples=42)
        result = validate_momentum_signal(data)
        assert result.signal.typical_lifetime_samples == 42

    def test_float_samples_truncated_to_int(self) -> None:
        data = _base_payload(typicalLifetimeSamples=42.9)
        result = validate_momentum_signal(data)
        assert result.signal.typical_lifetime_samples == 42

    def test_boolean_rejected_as_none(self) -> None:
        """``True`` is technically an ``int`` in Python — reject to avoid
        surprising typical_lifetime_samples=1 from a misserialized boolean."""
        data = _base_payload(typicalLifetimeSamples=True)
        result = validate_momentum_signal(data)
        assert result.signal.typical_lifetime_samples is None

    def test_legacy_est_max_lifetime_h_falls_back_to_typical(self) -> None:
        """A legacy orchestrator that only sends ``estMaxLifetimeH`` and
        ``lifetimeSamples`` populates ``typical_lifetime_h`` via fallback,
        with status downgraded to ``tentative`` so consumers know the
        publication path is the legacy one."""
        data = _base_payload(
            estMaxLifetimeH=18.4,
            lifetimeSamples=64,
        )
        result = validate_momentum_signal(data)
        assert result.signal.typical_lifetime_h == pytest.approx(18.4)
        assert result.signal.typical_lifetime_samples == 64
        assert result.signal.typical_lifetime_status == "tentative"

    def test_typical_takes_precedence_over_legacy_when_both_present(self) -> None:
        """During the cross-version window both fields may be set. New
        orchestrator's ``typicalLifetimeH`` wins."""
        data = _base_payload(
            typicalLifetimeH=12.0,
            typicalLifetimeSamples=80,
            typicalLifetimeStatus="stable",
            estMaxLifetimeH=18.4,
            lifetimeSamples=64,
        )
        result = validate_momentum_signal(data)
        assert result.signal.typical_lifetime_h == pytest.approx(12.0)
        assert result.signal.typical_lifetime_samples == 80
        assert result.signal.typical_lifetime_status == "stable"


# ---------------------------------------------------------------------------
# SignalConfig round-trip — orchestrator serialises, bot deserialises
# ---------------------------------------------------------------------------


class TestSharedModelRoundTrip:
    def test_ipc_roundtrip_preserves_typical_lifetime_fields(self) -> None:
        cfg = SignalConfig(
            rank=1,
            side="up",
            observe_from_s=240.0,
            observe_to_s=180.0,
            min_delta_pct=0.06,
            max_variance_pct=0.10,
            oos_win_rate_pct=96.0,
            avg_entry_price=0.80,
            ev_per_trade=0.08,
            signal_age_h=5.2,
            typical_lifetime_h=14.3,
            typical_lifetime_samples=127,
            typical_lifetime_status="stable",
        )
        payload = cfg.to_ipc_dict()
        assert payload["signalAgeH"] == pytest.approx(5.2)
        assert payload["typicalLifetimeH"] == pytest.approx(14.3)
        assert payload["typicalLifetimeSamples"] == 127
        assert payload["typicalLifetimeStatus"] == "stable"

        restored = SignalConfig.from_engine_json(payload)
        assert restored.signal_age_h == pytest.approx(5.2)
        assert restored.typical_lifetime_h == pytest.approx(14.3)
        assert restored.typical_lifetime_samples == 127
        assert restored.typical_lifetime_status == "stable"

    def test_ipc_roundtrip_preserves_none_values(self) -> None:
        cfg = SignalConfig(
            rank=1,
            side="up",
            observe_from_s=240.0,
            observe_to_s=180.0,
            min_delta_pct=0.06,
            max_variance_pct=0.10,
            signal_age_h=None,
            typical_lifetime_h=None,
            typical_lifetime_samples=None,
        )
        payload = cfg.to_ipc_dict()
        assert payload["signalAgeH"] is None
        assert payload["typicalLifetimeH"] is None
        assert payload["typicalLifetimeSamples"] is None
        assert payload["typicalLifetimeStatus"] == "unavailable"

        restored = SignalConfig.from_engine_json(payload)
        assert restored.signal_age_h is None
        assert restored.typical_lifetime_h is None
        assert restored.typical_lifetime_samples is None

    def test_legacy_payload_without_lifetime_fields_parses_as_none(self) -> None:
        """Pre-v3.7 orchestrator produces payloads with no lifetime keys —
        load cleanly rather than raising."""
        legacy_payload: dict[str, object] = {
            "rank": 1,
            "side": "up",
            "observeFromS": 240.0,
            "observeToS": 180.0,
            "minDeltaPct": 0.06,
            "maxVariancePct": 0.10,
            "oosWinRatePct": 96.0,
        }
        restored = SignalConfig.from_engine_json(legacy_payload)
        assert restored.signal_age_h is None
        assert restored.typical_lifetime_h is None
        assert restored.typical_lifetime_samples is None
        assert restored.typical_lifetime_status == "unavailable"
