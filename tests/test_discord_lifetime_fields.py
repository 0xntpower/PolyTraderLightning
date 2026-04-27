"""Tests for the v3.7 phase-2 typical-lifetime fields on Discord embeds.

The orchestrator publishes a median-of-eligible-lifetimes value (with a
status: ``unavailable`` / ``tentative`` / ``stable``) alongside every
signal delivery. The bot stashes that value at fire time and surfaces it
on signal-updated and bet-result embeds.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch


def _extract_field(embeds: list[dict[str, Any]], name: str) -> str:
    for e in embeds:
        for f in e.get("fields", []):
            if f.get("name") == name:
                return str(f.get("value", ""))
    return ""


# ---------------------------------------------------------------------------
# send_signal_updated
# ---------------------------------------------------------------------------


def test_signal_updated_renders_age_and_typical_lifetime_when_provided() -> None:
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=1,
            new_side="down",
            old_rank=1,
            old_side="up",
            signal_age_h=4.2,
            typical_lifetime_h=14.3,
            typical_lifetime_samples=127,
            typical_lifetime_status="stable",
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Signal Age") == "`4.2h`"
    assert _extract_field(embeds, "Typical Lifetime") == "`14.3h (median, n=127)`"


def test_signal_updated_marks_tentative_status() -> None:
    """``tentative`` status should annotate the displayed value."""
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=1,
            new_side="down",
            old_rank=1,
            old_side="up",
            signal_age_h=4.2,
            typical_lifetime_h=12.0,
            typical_lifetime_samples=18,
            typical_lifetime_status="tentative",
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Typical Lifetime") == "`12.0h (median, n=18, tentative)`"


def test_signal_updated_omits_lifetime_fields_when_absent() -> None:
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=1,
            new_side="down",
            old_rank=1,
            old_side="up",
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Signal Age") == ""
    assert _extract_field(embeds, "Typical Lifetime") == ""


def test_signal_updated_renders_age_without_lifetime_during_bootstrap() -> None:
    """Bootstrap phase: age is known, but ``typical_lifetime_h`` is None."""
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=1,
            new_side="down",
            old_rank=1,
            old_side="up",
            signal_age_h=0.5,
            typical_lifetime_h=None,
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Signal Age") == "`0.5h`"
    assert _extract_field(embeds, "Typical Lifetime") == ""


def test_signal_updated_renders_selected_over_when_provided() -> None:
    """v3.7 phase-3: orchestrator picked this signal over a younger
    runner-up. Show the runner-up's label."""
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=3,
            new_side="up",
            old_rank=1,
            old_side="down",
            signal_age_h=2.1,
            typical_lifetime_h=18.0,
            typical_lifetime_status="stable",
            selected_over="#1 down [240->180] d>=0.06% v<=0.05%",
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Selected Over") == "`#1 down [240->180] d>=0.06% v<=0.05%`"


def test_signal_updated_omits_selected_over_when_none() -> None:
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=1,
            new_side="down",
            old_rank=1,
            old_side="up",
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Selected Over") == ""


def test_signal_updated_omits_sample_count_when_absent() -> None:
    """Lifetime value provided but sample count not — drop the "n=" suffix."""
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=1,
            new_side="down",
            old_rank=1,
            old_side="up",
            signal_age_h=2.0,
            typical_lifetime_h=10.0,
            typical_lifetime_status="stable",
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Typical Lifetime") == "`10.0h (median)`"


# ---------------------------------------------------------------------------
# send_bet_result
# ---------------------------------------------------------------------------


def test_bet_result_renders_age_at_fire_and_typical_lifetime() -> None:
    from shared.discord import send_bet_result

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_bet_result(
            mode="live",
            outcome="WIN",
            pnl=0.46,
            entry_price=0.77,
            side="down",
            size_usd=3.12,
            signal_age_at_fire_h=4.5,
            typical_lifetime_h=14.3,
            typical_lifetime_samples=127,
            typical_lifetime_status="stable",
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Signal Age at Fire") == "`4.5h`"
    assert _extract_field(embeds, "Typical Lifetime") == "`14.3h (median, n=127)`"


def test_bet_result_omits_lifetime_fields_when_absent() -> None:
    from shared.discord import send_bet_result

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_bet_result(
            mode="live",
            outcome="LOSS",
            pnl=-3.20,
            entry_price=0.83,
            side="down",
            size_usd=3.19,
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Signal Age at Fire") == ""
    assert _extract_field(embeds, "Typical Lifetime") == ""


def test_bet_result_preserves_existing_fields_alongside_lifetime() -> None:
    """Adding lifetime fields must not change existing fields."""
    from shared.discord import send_bet_result

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_bet_result(
            mode="live",
            outcome="WIN",
            pnl=0.46,
            entry_price=0.77,
            side="down",
            size_usd=3.12,
            balance=62.73,
            maker_usd=0.17,
            taker_usd=2.99,
            signal_age_at_fire_h=4.5,
            typical_lifetime_h=14.3,
            typical_lifetime_samples=127,
            typical_lifetime_status="stable",
        )

    embeds = mock_send.call_args.args[1]
    # Core fields still present.
    assert _extract_field(embeds, "PnL") == "`$+0.4600`"
    assert _extract_field(embeds, "Side") == "`DOWN`"
    assert _extract_field(embeds, "Size") == "`$3.12`"
    assert _extract_field(embeds, "Balance") == "`$62.73`"
    # Combined-fill split still rendered.
    assert _extract_field(embeds, "Fill") == "`Maker 5% / Taker 95%`"
    # New lifetime fields present.
    assert _extract_field(embeds, "Signal Age at Fire") == "`4.5h`"
    assert _extract_field(embeds, "Typical Lifetime") == "`14.3h (median, n=127)`"
