"""Simple three-state circuit breaker for external API calls.

States:
    CLOSED   — normal operation, tracking consecutive failures.
    OPEN     — too many failures; reject calls until cooldown expires.
    HALF_OPEN — cooldown expired; allow one probe call to test recovery.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class CircuitBreaker:
    """Protects against hammering a broken external service."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_sec: float = 60.0,
        max_cooldown_sec: float = 300.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self._base_cooldown = cooldown_sec
        self._max_cooldown = max_cooldown_sec

        self._consecutive_failures: int = 0
        self._state: str = "CLOSED"
        self._opened_at: float = 0.0
        self._current_cooldown: float = cooldown_sec

    @property
    def state(self) -> str:
        # Auto-transition OPEN → HALF_OPEN when cooldown expires
        if self._state == "OPEN" and (time.time() - self._opened_at) >= self._current_cooldown:
            self._state = "HALF_OPEN"
            log.info("circuit breaker [%s]: OPEN → HALF_OPEN (probing)", self.name)
        return self._state

    def can_attempt(self) -> bool:
        """Return True if a call should be attempted."""
        s = self.state  # triggers auto-transition
        if s == "CLOSED":
            return True
        return s == "HALF_OPEN"

    def record_success(self) -> None:
        """Call after a successful API call."""
        if self._state != "CLOSED":
            log.info("circuit breaker [%s]: %s → CLOSED (success)", self.name, self._state)
        self._consecutive_failures = 0
        self._state = "CLOSED"
        self._current_cooldown = self._base_cooldown

    def record_failure(self) -> None:
        """Call after a failed API call."""
        self._consecutive_failures += 1

        if self._state == "HALF_OPEN":
            # Probe failed — back to OPEN with doubled cooldown
            self._current_cooldown = min(self._current_cooldown * 2, self._max_cooldown)
            self._state = "OPEN"
            self._opened_at = time.time()
            log.warning(
                "circuit breaker [%s]: HALF_OPEN → OPEN (probe failed, cooldown=%.0fs)",
                self.name,
                self._current_cooldown,
            )
        elif self._consecutive_failures >= self.failure_threshold:
            self._state = "OPEN"
            self._opened_at = time.time()
            log.warning(
                "circuit breaker [%s]: CLOSED -> OPEN after %d failures (cooldown=%.0fs)",
                self.name,
                self._consecutive_failures,
                self._current_cooldown,
            )
