"""Post-loss cooldown tracker — v3.2 §5.8.

After a settled loss whose magnitude exceeds a fraction of bankroll,
freeze trading for the next N windows.

The original design used a countdown decremented at every window
boundary. That worked in paper mode, where ``register_loss`` fires
synchronously during the window transition (after the boundary
decrement), but it silently failed in live mode: gamma polling
resolves a window's outcome in the middle of the *following* window,
so the arm happened between boundaries and the very next transition
decremented the counter to zero — effectively freezing zero windows.
The 2026-04-22 v3.6.1 session hit exactly this on T4 (armed at
19:59:57, expired at 20:00:00, window 20:00-20:05 was NOT frozen).

v3.6.2 replaces the countdown with an absolute-ts marker. Arming
records the smallest ``window_ts`` at which trading re-opens; the
periodic boundary callback just advances ``_current_window_ts``
without any ordering dependency on when ``register_loss`` fires.

Called from two points:

* ``register_loss(loss_abs_usd, bankroll, resolved_window_ts)`` — at
  trade settlement, with the ``window_ts`` of the resolved window.
* ``on_window_boundary(new_window_ts)`` — at the start of every new
  window, with the ``window_ts`` of the window now opening.

``is_frozen`` is queried per-tick by the strategy loop to gate
``evaluate()``.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Market window length. btc-updown-5m = 300 s; the post-loss cooldown
# tracker only cares about multiples of this, so it's fine as a constant.
_WINDOW_SPAN_S = 300


class PostLossCooldown:
    """Absolute-ts freeze gate for post-loss cooldowns."""

    def __init__(
        self,
        enabled: bool,
        loss_pct_threshold: float,
        cooldown_windows: int,
    ) -> None:
        self._enabled = enabled and cooldown_windows > 0
        self._threshold = loss_pct_threshold / 100.0
        self._cooldown_windows = cooldown_windows
        self._freeze_until_window_ts: int = 0
        self._current_window_ts: int = 0
        # Tracks the previous is_frozen state so ``on_window_boundary`` can
        # log the "expired" transition exactly once.
        self._was_frozen: bool = False

    @property
    def is_frozen(self) -> bool:
        """True while the current window is strictly before the freeze-until ts."""
        return (
            self._enabled
            and self._current_window_ts > 0
            and self._current_window_ts < self._freeze_until_window_ts
        )

    @property
    def windows_remaining(self) -> int:
        """Number of windows still frozen from the current one onward."""
        if not self.is_frozen:
            return 0
        return (self._freeze_until_window_ts - self._current_window_ts) // _WINDOW_SPAN_S

    def register_loss(
        self,
        loss_abs_usd: float,
        bankroll: float,
        resolved_window_ts: int,
    ) -> bool:
        """Record a settled loss; arm the cooldown if it clears the threshold.

        ``loss_abs_usd`` is a positive magnitude. ``resolved_window_ts`` is
        the ``window_ts`` of the window where the losing trade was booked.

        The freeze extends across ``cooldown_windows`` windows starting
        from ``max(resolved_window_ts, current_window_ts) + span`` — i.e.
        from the window strictly after the later of "where the trade was
        booked" and "where we are now." This handles both arm-shapes
        uniformly:

        - Paper mode: ``register_loss`` fires during the boundary transition
          for window N+1, but ``on_window_boundary`` is called at the END
          of the transition, so ``current_window_ts`` is still N at arm
          time. freeze starts at N+1. ✓
        - Live mode: ``register_loss`` fires via gamma poll mid-window
          N+1, so ``current_window_ts = N+1`` already. freeze starts at
          N+2. Matches the "first window the strategy would be re-tradable
          after notice of the loss" intent. ✓

        Returns True if the cooldown was (re-)armed.
        """
        if not self._enabled or loss_abs_usd <= 0.0 or bankroll <= 0.0 or resolved_window_ts <= 0:
            return False
        loss_frac = loss_abs_usd / bankroll
        if loss_frac <= self._threshold:
            return False
        anchor = max(resolved_window_ts, self._current_window_ts)
        proposed = anchor + (self._cooldown_windows + 1) * _WINDOW_SPAN_S
        # max() so re-arming doesn't shrink a longer existing freeze.
        self._freeze_until_window_ts = max(self._freeze_until_window_ts, proposed)
        self._was_frozen = self.is_frozen
        log.warning(
            "POST_LOSS_COOLDOWN armed: loss=$%.2f (%.2f%% of $%.2f bankroll) "
            "> %.2f%% -> freezing %d window(s) until ts=%d",
            loss_abs_usd,
            loss_frac * 100.0,
            bankroll,
            self._threshold * 100.0,
            self._cooldown_windows,
            self._freeze_until_window_ts,
        )
        return True

    def on_window_boundary(self, new_window_ts: int) -> None:
        """Advance the current-window pointer. No decrement, no race conditions.

        Pass the ``window_ts`` of the window that just started. ``0`` is
        accepted for pre-first-window calls and leaves the frozen check
        inactive.
        """
        if not self._enabled:
            return
        self._current_window_ts = new_window_ts
        now_frozen = self.is_frozen
        if self._was_frozen and not now_frozen:
            log.info("POST_LOSS_COOLDOWN expired — trading re-enabled")
        elif now_frozen:
            log.info(
                "POST_LOSS_COOLDOWN active — %d window(s) remaining",
                self.windows_remaining,
            )
        self._was_frozen = now_frozen
