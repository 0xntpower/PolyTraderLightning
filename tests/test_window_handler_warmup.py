"""Tests for WindowEventHandler's warmup-transition tracking (PR-D).

v3.4 §5.1 documented a race: ``_was_warming_up`` was read from
``strategy.warmup_active``, and when a signal swap at the first
post-warmup window boundary replaced the strategy with a freshly-built
one whose ``warmup_active`` defaults to False, the True→False transition
was swallowed and the "WARMUP complete" alert never fired.

PR-D moves the transition flag onto the handler (``_warmup_was_active``)
so it survives strategy swaps. The strategy attribute is still written
every window so the fire-time sizing clamp continues to work.
"""

from __future__ import annotations

import time
from unittest.mock import patch

# The handler test module defines its own fakes (FakeConfig, FakeVolTracker, etc)
# and a ``_make_handler`` factory; reuse them rather than duplicating.
from test_window_handler import (
    FakeConfig,
    FakeDecayDetector,
    FakeOrderManager,
    FakeStrategy,
    _make_handler,
)

# ---------------------------------------------------------------------------
# __init__ — _warmup_was_active is initialized from bot_start_time
# ---------------------------------------------------------------------------


def test_init_detects_active_warmup() -> None:
    """bot_start_time in the recent past (within warmup window) → flag True."""
    now = time.time()
    # bot_start_time = now − 5 min; default warmup_minutes = 30 → still warming up
    handler = _make_handler(bot_start_time=now - 300.0)
    assert handler._warmup_was_active is True


def test_init_detects_expired_warmup() -> None:
    """bot_start_time beyond warmup window → flag False."""
    now = time.time()
    # bot_start_time = now − 2 h; default warmup_minutes = 30 → expired
    handler = _make_handler(bot_start_time=now - 7200.0)
    assert handler._warmup_was_active is False


def test_init_with_warmup_disabled() -> None:
    """warmup_minutes=0 → flag always False regardless of bot_start_time."""
    from fakes import make_sizing_config

    cfg = FakeConfig()
    cfg.sizing = make_sizing_config(warmup_minutes=0.0)
    now = time.time()
    handler = _make_handler(cfg=cfg, bot_start_time=now)
    assert handler._warmup_was_active is False


# ---------------------------------------------------------------------------
# State placement — _warmup_was_active lives on the handler (the bug fix)
# ---------------------------------------------------------------------------


def test_warmup_flag_is_handler_state_not_strategy_state() -> None:
    """Regression: the transition flag must be on the handler so it survives
    ``_handle_signal_swap`` replacing the strategy with a fresh one.
    """
    now = time.time()
    handler = _make_handler(bot_start_time=now - 300.0)
    # The flag we care about lives on the handler instance itself.
    assert hasattr(handler, "_warmup_was_active")
    # A freshly-built strategy has warmup_active=False by default; reading
    # transition state from it would lose the True→False edge.
    fresh_strategy = FakeStrategy()
    assert fresh_strategy.warmup_active is False
    # But the handler still remembers the pre-swap truth.
    assert handler._warmup_was_active is True


# ---------------------------------------------------------------------------
# Integration — the alert fires exactly once even when a swap coincides
# with warmup expiry
# ---------------------------------------------------------------------------


def test_warmup_alert_fires_when_expiry_coincides_with_strategy_swap() -> None:
    """The exact v3.4 §5.1 bug: swap at first post-warmup boundary swallows
    the alert. Under PR-D this must fire exactly once.
    """
    from strategy.kelly import AdjustedWinRateResult

    now_start = 1_000_000.0
    warmup_min = 30.0  # default in SizingConfig

    # Initialize handler mid-warmup so _warmup_was_active=True.
    with patch("strategy.window_handler.time.time", return_value=now_start + 300.0):
        handler = _make_handler(bot_start_time=now_start)
    assert handler._warmup_was_active is True

    # Simulate a signal swap: replace the strategy with a fresh one whose
    # warmup_active defaults to False (this is the bug condition).
    fresh_strategy = FakeStrategy()
    assert fresh_strategy.warmup_active is False

    # Advance time past warmup expiry. Tick _compute_kelly_context; patch the
    # heavy pipeline and time.time so only the warmup block does real work.
    expired_now = now_start + (warmup_min * 60.0) + 5.0

    fake_wr = AdjustedWinRateResult(
        adjusted_p=0.90,
        vol_discount=0.0,
        chop_discount=0.0,
        outcome_discount=0.0,
        total_discount=0.0,
        feedback_adjustment=0.0,
        regime_ready=True,
    )

    with (
        patch("strategy.window_handler.time.time", return_value=expired_now),
        patch.object(
            handler,
            "_build_kelly_wr_result",
            return_value=(fake_wr, 0.05, 0.05, 2.0),
        ),
        patch("shared.discord.send_warmup_complete") as mock_discord,
    ):
        handler._compute_kelly_context(
            fresh_strategy, FakeOrderManager(mode="paper"), FakeDecayDetector()
        )

    # Alert fires exactly once at the boundary.
    assert mock_discord.call_count == 1
    # Handler flag updated to reflect current state.
    assert handler._warmup_was_active is False
    # Fire-time clamp on the strategy is also updated.
    assert fresh_strategy.warmup_active is False
    # Handler's "already sent" latch is engaged so a subsequent window does
    # not re-fire.
    assert handler._warmup_alert_sent is True


def test_warmup_alert_does_not_fire_before_expiry() -> None:
    """If warmup is still active, no alert fires regardless of swaps."""
    from strategy.kelly import AdjustedWinRateResult

    now_start = 1_000_000.0

    with patch("strategy.window_handler.time.time", return_value=now_start + 300.0):
        handler = _make_handler(bot_start_time=now_start)

    fresh_strategy = FakeStrategy()

    # Still within warmup window: 10 min after start, warmup ends at 30 min.
    still_in_warmup = now_start + 600.0

    fake_wr = AdjustedWinRateResult(
        adjusted_p=0.90,
        vol_discount=0.0,
        chop_discount=0.0,
        outcome_discount=0.0,
        total_discount=0.0,
        feedback_adjustment=0.0,
        regime_ready=True,
    )

    with (
        patch("strategy.window_handler.time.time", return_value=still_in_warmup),
        patch.object(
            handler,
            "_build_kelly_wr_result",
            return_value=(fake_wr, 0.05, 0.05, 2.0),
        ),
        patch("shared.discord.send_warmup_complete") as mock_discord,
    ):
        handler._compute_kelly_context(
            fresh_strategy, FakeOrderManager(mode="paper"), FakeDecayDetector()
        )

    assert mock_discord.call_count == 0
    assert handler._warmup_was_active is True
    assert handler._warmup_alert_sent is False


def test_warmup_alert_fires_exactly_once_across_multiple_ticks() -> None:
    """Once the alert fires, subsequent post-expiry windows do not re-fire."""
    from strategy.kelly import AdjustedWinRateResult

    now_start = 1_000_000.0

    with patch("strategy.window_handler.time.time", return_value=now_start + 300.0):
        handler = _make_handler(bot_start_time=now_start)

    fake_wr = AdjustedWinRateResult(
        adjusted_p=0.90,
        vol_discount=0.0,
        chop_discount=0.0,
        outcome_discount=0.0,
        total_discount=0.0,
        feedback_adjustment=0.0,
        regime_ready=True,
    )

    # First tick: still in warmup
    with (
        patch("strategy.window_handler.time.time", return_value=now_start + 600.0),
        patch.object(
            handler,
            "_build_kelly_wr_result",
            return_value=(fake_wr, 0.05, 0.05, 2.0),
        ),
        patch("shared.discord.send_warmup_complete") as mock_discord,
    ):
        handler._compute_kelly_context(
            FakeStrategy(), FakeOrderManager(mode="paper"), FakeDecayDetector()
        )
        assert mock_discord.call_count == 0

    # Second tick: warmup just expired — alert fires
    with (
        patch("strategy.window_handler.time.time", return_value=now_start + 1805.0),
        patch.object(
            handler,
            "_build_kelly_wr_result",
            return_value=(fake_wr, 0.05, 0.05, 2.0),
        ),
        patch("shared.discord.send_warmup_complete") as mock_discord,
    ):
        handler._compute_kelly_context(
            FakeStrategy(), FakeOrderManager(mode="paper"), FakeDecayDetector()
        )
        assert mock_discord.call_count == 1

    # Third tick: 10 min later — no re-fire
    with (
        patch("strategy.window_handler.time.time", return_value=now_start + 2400.0),
        patch.object(
            handler,
            "_build_kelly_wr_result",
            return_value=(fake_wr, 0.05, 0.05, 2.0),
        ),
        patch("shared.discord.send_warmup_complete") as mock_discord,
    ):
        handler._compute_kelly_context(
            FakeStrategy(), FakeOrderManager(mode="paper"), FakeDecayDetector()
        )
        assert mock_discord.call_count == 0
