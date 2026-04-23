"""Tests for the signal-age + est-max-lifetime fields added to Discord
``send_signal_updated`` and ``send_bet_result`` in v3.7.

These fields are orchestrator-sourced: the orchestrator threads
``signal_age_h`` and ``est_max_lifetime_h`` through the IPC signal
payload, the bot stashes both at fire time, and both surface on the
signal-delivered and bet-resolve embeds.
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


def test_signal_updated_renders_age_and_lifetime_when_provided() -> None:
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=1,
            new_side="down",
            old_rank=1,
            old_side="up",
            signal_age_h=4.2,
            est_max_lifetime_h=14.3,
            lifetime_samples=127,
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Signal Age") == "`4.2h`"
    assert _extract_field(embeds, "Est. Max Lifetime") == "`14.3h (p80, n=127)`"


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
    assert _extract_field(embeds, "Est. Max Lifetime") == ""


def test_signal_updated_renders_age_without_lifetime_during_bootstrap() -> None:
    """Bootstrap phase: age is known, but the estimator has no samples yet."""
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=1,
            new_side="down",
            old_rank=1,
            old_side="up",
            signal_age_h=0.5,
            est_max_lifetime_h=None,
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Signal Age") == "`0.5h`"
    assert _extract_field(embeds, "Est. Max Lifetime") == ""


def test_signal_updated_omits_sample_count_when_absent() -> None:
    """``est_max_lifetime_h`` provided but sample count not — drop the "n=" suffix."""
    from shared.discord import send_signal_updated

    with patch("shared.discord._send", return_value=True) as mock_send:
        send_signal_updated(
            new_rank=1,
            new_side="down",
            old_rank=1,
            old_side="up",
            signal_age_h=2.0,
            est_max_lifetime_h=10.0,
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Est. Max Lifetime") == "`10.0h (p80)`"


# ---------------------------------------------------------------------------
# send_bet_result
# ---------------------------------------------------------------------------


def test_bet_result_renders_age_at_fire_and_lifetime() -> None:
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
            est_max_lifetime_h=14.3,
            lifetime_samples=127,
        )

    embeds = mock_send.call_args.args[1]
    assert _extract_field(embeds, "Signal Age at Fire") == "`4.5h`"
    assert _extract_field(embeds, "Est. Max Lifetime") == "`14.3h (p80, n=127)`"


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
    assert _extract_field(embeds, "Est. Max Lifetime") == ""


def test_bet_result_preserves_existing_fields_alongside_lifetime() -> None:
    """Backward compat: adding lifetime fields must not change existing fields."""
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
            est_max_lifetime_h=14.3,
            lifetime_samples=127,
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
    assert _extract_field(embeds, "Est. Max Lifetime") == "`14.3h (p80, n=127)`"
