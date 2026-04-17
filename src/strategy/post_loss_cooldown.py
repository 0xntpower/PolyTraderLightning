"""Post-loss cooldown tracker — v3.2 §5.8.

After a settled loss whose magnitude exceeds a fraction of bankroll,
freeze trading for the next N windows. Called at two points:

* ``register_loss(loss_abs_usd, bankroll)`` — at trade settlement.
* ``on_window_boundary()`` — at the start of every new window.

``is_frozen`` is queried per-tick by the strategy loop to gate evaluate().
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class PostLossCooldown:
    """Single-counter freeze gate for post-loss cooldowns."""

    def __init__(
        self,
        enabled: bool,
        loss_pct_threshold: float,
        cooldown_windows: int,
    ) -> None:
        self._enabled = enabled and cooldown_windows > 0
        self._threshold = loss_pct_threshold / 100.0
        self._cooldown_windows = cooldown_windows
        self._remaining = 0

    @property
    def is_frozen(self) -> bool:
        """True while the cooldown counter is non-zero."""
        return self._enabled and self._remaining > 0

    @property
    def windows_remaining(self) -> int:
        return self._remaining

    def register_loss(self, loss_abs_usd: float, bankroll: float) -> bool:
        """Record a settled loss; arm the cooldown if magnitude exceeds threshold.

        ``loss_abs_usd`` is the absolute loss size (pass a positive number).
        Returns True if the cooldown was (re-)armed.
        """
        if not self._enabled or loss_abs_usd <= 0.0 or bankroll <= 0.0:
            return False
        loss_frac = loss_abs_usd / bankroll
        if loss_frac <= self._threshold:
            return False
        self._remaining = self._cooldown_windows
        log.warning(
            "POST_LOSS_COOLDOWN armed: loss=$%.2f (%.2f%% of $%.2f bankroll) > %.2f%% "
            "-> freezing %d window(s)",
            loss_abs_usd,
            loss_frac * 100.0,
            bankroll,
            self._threshold * 100.0,
            self._cooldown_windows,
        )
        return True

    def on_window_boundary(self) -> None:
        """Advance one window. Decrements the counter but never below zero."""
        if not self._enabled or self._remaining <= 0:
            return
        self._remaining -= 1
        if self._remaining == 0:
            log.info("POST_LOSS_COOLDOWN expired — trading re-enabled")
        else:
            log.info("POST_LOSS_COOLDOWN active — %d window(s) remaining", self._remaining)
