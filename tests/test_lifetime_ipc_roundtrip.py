"""Tests for the v3.7 lifetime-field IPC round-trip.

Covers the signal_loader parsing path (orchestrator → bot) and the
shared/models.py round-trip (bot → orchestrator, used in tests and
replay). Bot consumers don't yet use these values for anything beyond
Discord notifications, so these tests focus on: (a) values arrive
intact when present, (b) absent / null / invalid values gracefully
become None rather than raising.
"""

from __future__ import annotations

import pytest

from shared.models import SignalConfig
from strategy.signal_loader import validate_momentum_signal

# ---------------------------------------------------------------------------
# Minimal engine-shape payload — mirrors orchestrator-to-bot IPC
# ---------------------------------------------------------------------------


def _base_payload(**overrides: object) -> dict[str, object]:
    """Build a minimal valid signal dict; overrides let tests vary fields."""
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


class TestSignalLoaderLifetimeFields:
    def test_values_parsed_when_present(self) -> None:
        data = _base_payload(
            signalAgeH=5.2,
            estMaxLifetimeH=14.3,
            lifetimeSamples=127,
        )
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h == pytest.approx(5.2)
        assert result.signal.est_max_lifetime_h == pytest.approx(14.3)
        assert result.signal.lifetime_samples == 127

    def test_fields_default_to_none_when_absent(self) -> None:
        """Older orchestrator that doesn't yet send these fields must still work."""
        data = _base_payload()
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h is None
        assert result.signal.est_max_lifetime_h is None
        assert result.signal.lifetime_samples is None

    def test_null_values_become_none(self) -> None:
        """Orchestrator explicitly sends None during bootstrap (no samples yet)."""
        data = _base_payload(
            signalAgeH=None,
            estMaxLifetimeH=None,
            lifetimeSamples=None,
        )
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h is None
        assert result.signal.est_max_lifetime_h is None
        assert result.signal.lifetime_samples is None

    def test_nan_and_infinity_rejected_as_none(self) -> None:
        """Protect against upstream pipeline bugs injecting non-finite floats."""
        data = _base_payload(signalAgeH=float("nan"), estMaxLifetimeH=float("inf"))
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h is None
        assert result.signal.est_max_lifetime_h is None

    def test_non_numeric_becomes_none(self) -> None:
        data = _base_payload(
            signalAgeH="five hours",
            estMaxLifetimeH={"nested": 5},
            lifetimeSamples="lots",
        )
        result = validate_momentum_signal(data)
        assert result.signal.signal_age_h is None
        assert result.signal.est_max_lifetime_h is None
        assert result.signal.lifetime_samples is None

    def test_int_samples_accepted(self) -> None:
        data = _base_payload(lifetimeSamples=42)
        result = validate_momentum_signal(data)
        assert result.signal.lifetime_samples == 42

    def test_float_samples_truncated_to_int(self) -> None:
        data = _base_payload(lifetimeSamples=42.9)
        result = validate_momentum_signal(data)
        assert result.signal.lifetime_samples == 42

    def test_boolean_rejected_as_none(self) -> None:
        """``True`` is technically an ``int`` in Python — reject to avoid
        surprising lifetime_samples=1 from a misserialized boolean."""
        data = _base_payload(lifetimeSamples=True)
        result = validate_momentum_signal(data)
        assert result.signal.lifetime_samples is None


# ---------------------------------------------------------------------------
# SignalConfig round-trip — orchestrator serialises, bot deserialises
# ---------------------------------------------------------------------------


class TestSharedModelRoundTrip:
    def test_ipc_roundtrip_preserves_lifetime_fields(self) -> None:
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
            est_max_lifetime_h=14.3,
            lifetime_samples=127,
        )
        payload = cfg.to_ipc_dict()
        assert payload["signalAgeH"] == pytest.approx(5.2)
        assert payload["estMaxLifetimeH"] == pytest.approx(14.3)
        assert payload["lifetimeSamples"] == 127

        restored = SignalConfig.from_engine_json(payload)
        assert restored.signal_age_h == pytest.approx(5.2)
        assert restored.est_max_lifetime_h == pytest.approx(14.3)
        assert restored.lifetime_samples == 127

    def test_ipc_roundtrip_preserves_none_values(self) -> None:
        cfg = SignalConfig(
            rank=1,
            side="up",
            observe_from_s=240.0,
            observe_to_s=180.0,
            min_delta_pct=0.06,
            max_variance_pct=0.10,
            signal_age_h=None,
            est_max_lifetime_h=None,
            lifetime_samples=None,
        )
        payload = cfg.to_ipc_dict()
        assert payload["signalAgeH"] is None
        assert payload["estMaxLifetimeH"] is None
        assert payload["lifetimeSamples"] is None

        restored = SignalConfig.from_engine_json(payload)
        assert restored.signal_age_h is None
        assert restored.est_max_lifetime_h is None
        assert restored.lifetime_samples is None

    def test_legacy_payload_without_lifetime_fields_parses_as_none(self) -> None:
        """Pre-v3.7 orchestrator produces payloads with no lifetime keys —
        must load cleanly rather than raising KeyError."""
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
        assert restored.est_max_lifetime_h is None
        assert restored.lifetime_samples is None
