"""Sequential Probability Ratio Test (SPRT) for signal decay detection.

Accumulates evidence from resolved trade outcomes to determine whether a
signal is still performing at its expected win rate (ALIVE) or has degraded
to breakeven or worse (DEAD). This is the sole performance-based kill switch
for live trading.

The LLR (log-likelihood ratio) tracks evidence for H0 (dead) vs H1 (alive):
  - Positive LLR = evidence toward DEAD
  - Negative LLR = evidence toward ALIVE
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)

_ROLLING_WINDOW = 20


@dataclass
class DecayState:
    signal_id: str
    p_alive: float
    p_dead: float
    llr: float = 0.0
    boundary_dead: float = 0.0
    boundary_alive: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    rolling_win_rate: float = 0.0
    verdict: str = "INCONCLUSIVE"


class DecayDetector:
    """SPRT-based signal decay detector."""

    @staticmethod
    def _validated_p_dead(p_dead: float, p_alive: float, signal_id: str) -> float:
        """Ensure p_dead < p_alive and > 0 to avoid math errors."""
        if p_dead >= p_alive:
            log.warning(
                "p_dead (%.4f) >= p_alive (%.4f) for %s — using fallback p_dead = p_alive - 0.05",
                p_dead,
                p_alive,
                signal_id,
            )
            p_dead = p_alive - 0.05
        return max(p_dead, 0.01)  # clamp to avoid log(0)

    def __init__(
        self,
        signal_id: str,
        p_alive: float,
        p_dead: float,
        alpha: float = 0.05,
        beta: float = 0.10,
    ) -> None:
        self._signal_id = signal_id
        self._p_alive = p_alive
        self._p_dead = self._validated_p_dead(p_dead, p_alive, signal_id)
        self._alpha = alpha
        self._beta = beta

        self._boundary_dead = math.log((1.0 - beta) / alpha)
        self._boundary_alive = math.log(beta / (1.0 - alpha))

        self._llr = 0.0
        self._n_trades = 0
        self._n_wins = 0
        self._verdict = "INCONCLUSIVE"
        self._rolling: deque[bool] = deque(maxlen=_ROLLING_WINDOW)

    def update(self, won: bool) -> DecayState:
        """Feed one resolved trade outcome. Returns updated state."""
        self._n_trades += 1
        if won:
            self._n_wins += 1
        self._rolling.append(won)

        try:
            if won:
                increment = math.log(self._p_dead / self._p_alive)
            else:
                increment = math.log((1.0 - self._p_dead) / (1.0 - self._p_alive))

            if not math.isfinite(increment):
                log.error("SPRT increment is non-finite for %s — skipping", self._signal_id)
                return self.state
        except (ValueError, ZeroDivisionError):
            log.error("SPRT math error for %s — skipping update", self._signal_id)
            return self.state

        self._llr += increment

        if self._llr >= self._boundary_dead:
            self._verdict = "DEAD"
        elif self._llr <= self._boundary_alive:
            self._verdict = "ALIVE"
        else:
            self._verdict = "INCONCLUSIVE"

        # Log every 10 trades
        if self._n_trades % 10 == 0:
            log.info(
                "Signal health: %d trades, rolling WR: %.0f%%, SPRT: %s "
                "(LLR: %.2f, bounds: [%.2f, %.2f])",
                self._n_trades,
                self._rolling_wr * 100,
                self._verdict,
                self._llr,
                self._boundary_alive,
                self._boundary_dead,
            )

        return self.state

    def reset(self, p_alive: float | None = None, p_dead: float | None = None) -> None:
        """Reset LLR for continued monitoring after ALIVE conclusion."""
        if p_alive is not None:
            self._p_alive = p_alive
        if p_dead is not None:
            self._p_dead = self._validated_p_dead(p_dead, self._p_alive, self._signal_id)

        self._llr = 0.0
        self._n_trades = 0
        self._n_wins = 0
        self._verdict = "INCONCLUSIVE"
        self._rolling.clear()

    @property
    def _rolling_wr(self) -> float:
        if not self._rolling:
            return 0.0
        return sum(self._rolling) / len(self._rolling)

    @property
    def state(self) -> DecayState:
        return DecayState(
            signal_id=self._signal_id,
            p_alive=self._p_alive,
            p_dead=self._p_dead,
            llr=self._llr,
            boundary_dead=self._boundary_dead,
            boundary_alive=self._boundary_alive,
            n_trades=self._n_trades,
            n_wins=self._n_wins,
            rolling_win_rate=self._rolling_wr,
            verdict=self._verdict,
        )
