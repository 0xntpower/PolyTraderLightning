"""Momentum signal strategy — evaluates PolySignalEngine signal_NNN.json files.

The engine discovers time-range momentum patterns of the form:
  "If BTC moves >= minDeltaPct% from open in direction <side>, with stddev of
   that move <= maxVariancePct%, observed between observeFromS and observeToS
   seconds remaining, then the window resolves <side> with high probability."

Unit note
---------
The engine stores bnDirectionFromOpenPct in PERCENT units (e.g. 0.10 means a
0.10% move). The bot's signal.bn_direction_from_open_pct is a FRACTION (e.g.
0.001 for a 0.10% move). MomentumSignalStrategy multiplies the bot's value by
100 before comparing against min_delta_pct and max_variance_pct.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

from shared.discord import send_bet_placed, send_early_exit
from strategy.kelly import AdjustedWinRateResult, KellyResult, conservative_win_rate, kelly_size
from strategy.signal import Direction, ObiDepth, Signal

if TYPE_CHECKING:
    from config import ErosionConfig, RulesStrategyConfig, SizingConfig
    from execution.base import OrderExecutor
    from market_data.state import MarketState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MomentumSignalConfig:
    rank: int
    side: Direction
    observe_from_s: float  # Upper bound of observation window (seconds remaining)
    observe_to_s: float  # Lower bound of observation window (seconds remaining)
    min_delta_pct: float  # Min |bn_direction_from_open_pct| in percent at window end
    max_variance_pct: float  # Max population stddev of bn_direction_from_open_pct in percent
    train_win_rate_pct: float
    oos_win_rate_pct: float
    bh_adjusted_p_value: float
    oos_matches: int
    conservative_win_rate_pct: float | None = None  # Wilson-adjusted OOS win rate from engine
    avg_entry_price: float | None = None  # Historical avg best_ask when signal fires
    ev_per_trade: float | None = None  # Expected value per trade
    smart_score: float = 0.0  # Composite quality score
    wf_folds_appeared: int = 0  # WF folds where signal appeared
    wf_total_test_folds: int = 0  # Total WF test folds
    wf_fold_indices: list[int] = field(default_factory=list)
    post_fire_max_safe_erosion_pct: float | None = None  # P90 win erosion threshold from engine
    # v3.4: per-signal OBI threshold from engine sweep. 0.0 = gate disabled;
    # otherwise fire requires |latest_obi| >= threshold with the right sign.
    obi_threshold: float = 0.0
    # Depth level the engine trained the OBI gate at (D10 or D20). NONE
    # whenever ``obi_threshold == 0.0`` — depth is meaningless with no gate.
    obi_depth: ObiDepth = ObiDepth.NONE
    # v3.7: engine-anchored signal age (hours) at delivery time and the
    # orchestrator's typical-lifetime publication (median of completed
    # family lifetimes, with eligibility filters applied). None during
    # bootstrap or when the orchestrator tracker is disabled / failed.
    # Observational on the bot — surfaces on Discord embeds so operators
    # can eyeball age-vs-outcome correlation; the orchestrator's selector
    # uses the same numbers to choose between candidates upstream.
    signal_age_h: float | None = None
    typical_lifetime_h: float | None = None
    typical_lifetime_samples: int | None = None
    typical_lifetime_status: str = "unavailable"
    # v3.7 phase-3: orchestrator delivered this signal because the age-aware
    # selector chose it over a top-by-score runner-up — the runner-up's
    # label, surfaced on the signal-updated Discord embed. None when this
    # signal was the top by score.
    selected_over: str | None = None

    def __post_init__(self) -> None:
        if self.observe_from_s <= self.observe_to_s:
            raise ValueError(
                f"observe_from_s ({self.observe_from_s}) must be "
                f"> observe_to_s ({self.observe_to_s})"
            )
        if self.min_delta_pct < 0:
            raise ValueError(f"min_delta_pct must be >= 0, got {self.min_delta_pct}")
        if self.oos_win_rate_pct < 0 or self.oos_win_rate_pct > 100:
            raise ValueError(f"oos_win_rate_pct must be 0-100, got {self.oos_win_rate_pct}")
        if self.obi_threshold > 0.0 and self.obi_depth is ObiDepth.NONE:
            raise ValueError(
                f"obi_threshold={self.obi_threshold} > 0 requires "
                f"obi_depth of D10 or D20, got {self.obi_depth}"
            )

    @property
    def signal_id(self) -> str:
        """Deterministic identifier from signal parameters."""
        return (
            f"{self.side.value}_{self.observe_from_s}_{self.observe_to_s}_"
            f"{self.min_delta_pct}_{self.max_variance_pct}"
        )

    def conservative_p(self, max_shrink_pct: float = 3.0) -> float:
        """Wilson lower-bound win probability (small-sample corrected).

        Prefers the engine's pre-computed conservative win rate when available,
        which uses the same Wilson formula with z=1.5.  Falls back to the bot's
        own calculation for signals from older engine versions.
        """
        if self.conservative_win_rate_pct is not None and self.conservative_win_rate_pct > 0:
            return self.conservative_win_rate_pct / 100.0
        return conservative_win_rate(self.oos_win_rate_pct, self.oos_matches, max_shrink_pct)


class EarlyExitTelemetry(TypedDict):
    """Trigger snapshot captured at the moment ``exit_position_early`` fired.

    Carried forward to the postmortem Discord notification so operators can
    see by how much the erosion exceeded the threshold (the exact tuning
    knob delta needed to flip the decision) once the window resolves.
    """

    erosion: float
    threshold: float
    reason: str
    fire_delta_pct: float
    current_delta_pct: float


class MomentumSignalStrategy:
    """Evaluates a MomentumSignalConfig against live tick data each strategy tick.

    Accumulates bn_direction_from_open_pct (converted to percent) while
    time_remaining is inside [observe_to_s, observe_from_s]. Once time_remaining
    drops below observe_to_s, evaluates conditions and places one maker order if
    the signal fires. One order per window maximum.
    """

    def __init__(
        self,
        cfg: RulesStrategyConfig,
        state: MarketState,
        signal_cfg: MomentumSignalConfig,
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.signal_cfg = signal_cfg
        self._order_placed: bool = False
        self._logged_no_data: bool = False
        self._fired: bool = False  # conditions were met and order attempted
        self.last_entry_price: float = 0.0  # actual entry price of placed order
        self.last_size_usd: float = 0.0  # actual size of placed order
        self.bet_scale: float = 1.0  # legacy — kept for daily summary compatibility
        # Kelly sizing context — set by main loop each window
        self.sprt_factor: float = 1.0
        self.kelly_wr_result: AdjustedWinRateResult | None = None
        self.bankroll: float = 1000.0
        self.sizing_cfg: SizingConfig | None = None
        self.erosion_cfg: ErosionConfig | None = None
        # Last Kelly result for logging/paper output
        self.last_kelly_result: KellyResult | None = None
        # Warmup mode — set by main loop each window
        self.warmup_active: bool = False
        # v3.2 §5.2: directional regime t-stat from vol_tracker, set by
        # WindowEventHandler._compute_kelly_context each window. 0 when
        # tracker has insufficient data.
        self.directional_t: float = 0.0
        # Welford online variance (accumulates values in percent units)
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        # Subsample state: throttle variance sampling to cfg.variance_subsample_interval_s
        # so live stddev matches the engine's backtest cadence. time_remaining
        # decreases as the window progresses, so we sample when the elapsed
        # drop since the last sample is >= the configured interval.
        self._last_var_sample_t: float = 0.0
        self._var_sampled_once: bool = False
        # Latest value at the lowest time_remaining seen within the window
        self._latest_pct: float = 0.0
        self._latest_obi: float = 0.0
        self._lowest_t: float = signal_cfg.observe_from_s
        # v3.2 §5.7: rolling OBI buffer — (wall_ts, obi) samples fed from
        # _accumulate and trimmed to the last obi_rolling_window_s seconds at
        # fire-time. wall time (not time_remaining) so the rolling window
        # semantics are stable even if the strategy tick cadence changes.
        self._obi_samples: deque[tuple[float, float]] = deque()
        # Maker-first entry tracking
        self._maker_order_id: str | None = None
        self._maker_placed_at: float = 0.0
        self._maker_entry_price: float = 0.0
        self._maker_token_id: str = ""
        self._maker_size_usd: float = 0.0
        self._maker_tier: str = ""
        self._entry_complete: bool = False  # final state: filled or gave up
        self._fire_signal: Signal | None = None  # signal snapshot at fire time
        # Post-fire erosion monitoring
        self._fire_delta_pct: float = 0.0  # bn_direction_from_open_pct * 100 at fire time
        self._early_exit_triggered: bool = False
        # Trigger snapshot at the moment ``exit_position_early`` actually
        # fires (skipped on the no-bid path where no sell happens). Read by
        # window_handler at window-end and shipped to the postmortem Discord
        # embed so the user can see how far erosion overshot the threshold.
        self._exit_telemetry: EarlyExitTelemetry | None = None
        # CUSUM erosion detector state
        self._erosion_ema: float = 0.0
        self._erosion_cusum: float = 0.0
        self._erosion_ema_initialized: bool = False
        self._erosion_last_log_time: float = 0.0
        self._erosion_last_logged_val: float = 0.0
        self._erosion_last_logged_cusum: float = 0.0
        # Monotonic timestamp when erosion first breached the panic line.
        # Reset to None whenever erosion drops below panic threshold. Panic
        # only fires once the breach has persisted >= panic_min_duration_s;
        # a brief noise blip on 2026-04-12 14:41 turned a winning trade
        # into a -$20 loss under the previous single-tick panic path.
        self._panic_breach_started_at: float | None = None
        # v2.9: monotonic timestamp when CUSUM first crossed its limit.
        # Mirrors the panic sustain timer. Prevents single-tick CUSUM blips
        # from triggering exits — v2.8 had 2 hard-false + 1 partial-false
        # CUSUM exits out of 9 (33% false rate) that the sustain gate
        # would have filtered.
        self._cusum_breach_started_at: float | None = None

    @property
    def fired(self) -> bool:
        """Whether the signal fired (conditions met) during this window."""
        return self._fired

    def reset(self) -> None:
        """Call at the start of each new 5-minute window."""
        self._order_placed = False
        self._logged_no_data = False
        self._fired = False
        self.last_entry_price = 0.0
        self.last_size_usd = 0.0
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._last_var_sample_t = 0.0
        self._var_sampled_once = False
        self._latest_pct = 0.0
        self._latest_obi = 0.0
        self._lowest_t = self.signal_cfg.observe_from_s
        self._obi_samples.clear()
        # Maker-first entry reset
        self._maker_order_id = None
        self._maker_placed_at = 0.0
        self._maker_entry_price = 0.0
        self._maker_token_id = ""
        self._maker_size_usd = 0.0
        self._maker_tier = ""
        self._entry_complete = False
        self._fire_signal = None
        # Post-fire erosion reset
        self._fire_delta_pct = 0.0
        self._early_exit_triggered = False
        self._erosion_ema = 0.0
        self._erosion_cusum = 0.0
        self._erosion_ema_initialized = False
        self._erosion_last_log_time = 0.0
        self._erosion_last_logged_val = 0.0
        self._erosion_last_logged_cusum = 0.0
        self._panic_breach_started_at = None
        self._cusum_breach_started_at = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gate_obi(self, signal: Signal) -> float:
        """Centered OBI at the depth the engine trained this signal at.

        When ``obi_depth`` is ``NONE`` (gate disabled), the rolling-OBI
        regime filter still runs, and D20 is the canonical "deepest book
        reading available" to drive that broader skip check.
        """
        depth = self.signal_cfg.obi_depth
        if depth is ObiDepth.NONE:
            return signal.binance_obi_d20
        return signal.obi_at(depth)

    def _accumulate(self, bn_dir_pct: float, time_remaining: float, obi: float) -> None:
        """Add one sample (in percent) using Welford's online algorithm.

        The variance accumulator is subsampled to cfg.variance_subsample_interval_s
        so the bot's live stddev matches the engine's historical cadence
        (collector snapshot_interval_s). Latest-value tracking is NOT subsampled
        — we always want the freshest delta/OBI at the end of the window.
        """
        interval = self.cfg.variance_subsample_interval_s
        # time_remaining decreases over the window; sample when the drop since
        # the last accepted sample is >= the configured interval (or on the
        # very first call so the accumulator has a seed).
        if (
            interval <= 0.0
            or not self._var_sampled_once
            or (self._last_var_sample_t - time_remaining) >= interval
        ):
            self._n += 1
            delta = bn_dir_pct - self._mean
            self._mean += delta / self._n
            self._m2 += delta * (bn_dir_pct - self._mean)
            self._last_var_sample_t = time_remaining
            self._var_sampled_once = True

        if time_remaining <= self._lowest_t:
            self._lowest_t = time_remaining
            self._latest_pct = bn_dir_pct
            self._latest_obi = obi

        # v3.2 §5.7: rolling-OBI buffer. Always record (even if variance
        # subsampling skipped this tick) — the rolling gate benefits from
        # every fresh OBI reading. Trim to the configured window.
        window_s = self.cfg.obi_rolling_window_s
        if window_s > 0.0:
            now = time.time()
            self._obi_samples.append((now, obi))
            cutoff = now - window_s
            while self._obi_samples and self._obi_samples[0][0] < cutoff:
                self._obi_samples.popleft()

    def _mean_recent_obi(self, window_s: float) -> tuple[float, int]:
        """Mean Binance OBI over the last ``window_s`` seconds.

        Returns (mean, sample_count). (0.0, 0) when the buffer is empty.
        """
        if window_s <= 0.0 or not self._obi_samples:
            return 0.0, 0
        cutoff = time.time() - window_s
        total = 0.0
        n = 0
        for ts, val in self._obi_samples:
            if ts >= cutoff:
                total += val
                n += 1
        if n == 0:
            return 0.0, 0
        return total / n, n

    def _population_stddev(self) -> float:
        if self._n < 2:
            return 0.0
        return math.sqrt(self._m2 / self._n)

    def _conditions_met(self) -> bool:
        if self._n < 5:  # match engine's kSweepMinTicksInRange
            return False
        sc = self.signal_cfg
        if self._population_stddev() > sc.max_variance_pct:
            return False
        if sc.side == Direction.UP:
            if self._latest_pct < sc.min_delta_pct:
                return False
        elif self._latest_pct > -sc.min_delta_pct:
            return False
        # v3.4: per-signal OBI threshold (0.0 = gate disabled). Fire requires
        # |_latest_obi| >= threshold with the right sign. Matches the engine
        # sweeper's per-config gate (core/Config.hpp kObiThresholdLevels).
        # ``_latest_obi`` holds the value at ``signal_cfg.obi_depth`` (D10 or
        # D20) — set in ``_accumulate`` via ``_gate_obi(signal)``.
        if sc.obi_threshold > 0.0:
            if sc.side == Direction.UP and self._latest_obi < sc.obi_threshold:
                return False
            if sc.side == Direction.DOWN and self._latest_obi > -sc.obi_threshold:
                return False
        return True

    # ------------------------------------------------------------------
    # Post-fire erosion monitoring
    # ------------------------------------------------------------------

    async def _monitor_post_fire_erosion(
        self,
        signal: Signal,
        order_mgr: OrderExecutor,
    ) -> None:
        """Monitor post-fire erosion using CUSUM change detection.

        Instead of exiting on a single tick above threshold (which triggers
        false exits on transient BTC spikes), we use a Cumulative Sum
        (CUSUM) detector that requires sustained exceedance before acting.

        Erosion = (fireDelta - currentDelta) / fireDelta.
        0 means no erosion, 1 means fully reversed to open.

        Three exit paths:
        1. PANIC - raw erosion exceeds panic_multiplier * threshold,
           sustained for panic_min_duration_s. Catastrophic reversal.
        2. CUSUM - EMA-smoothed erosion accumulates excess above threshold.
           When cumulative excess reaches the limit AND the breach has
           persisted for cusum_sustain_s (v2.9 sustain gate), exit. If the
           Polymarket top bid on our side is already >= cusum_suppress_top_bid
           (v2.9 price-aware suppression), the orderbook disagrees with the
           BTC delta signal and the exit is suppressed — the market is
           telling us we are right.
        3. Sub-threshold - CUSUM bleeds off via decay factor, preventing
           false triggers from brief spikes that recover.
        """
        threshold = self.signal_cfg.post_fire_max_safe_erosion_pct
        if threshold is None or self.erosion_cfg is None:
            return

        fire_delta = self._fire_delta_pct
        if abs(fire_delta) < 1e-9:
            return

        current_pct = signal.bn_direction_from_open_pct * 100.0
        erosion = (fire_delta - current_pct) / fire_delta
        if erosion < 0:
            erosion = 0.0

        ecfg = self.erosion_cfg

        # --- EMA smoothing ---
        if not self._erosion_ema_initialized:
            self._erosion_ema = erosion
            self._erosion_ema_initialized = True
        else:
            alpha = ecfg.ema_alpha
            self._erosion_ema = alpha * erosion + (1.0 - alpha) * self._erosion_ema

        # --- Path 1: panic exit on catastrophic reversal ---
        # Panic requires a *sustained* breach of panic_threshold — not a single
        # tick — so that a short-duration price noise burst cannot short-circuit
        # the CUSUM path. See 2026-04-12 14:41 Trade #4 false panic.
        panic_threshold = threshold * ecfg.panic_multiplier
        now_mono = time.monotonic()
        if erosion > panic_threshold:
            if self._panic_breach_started_at is None:
                self._panic_breach_started_at = now_mono
            breach_duration = now_mono - self._panic_breach_started_at
            if breach_duration >= ecfg.panic_min_duration_s:
                log.warning(
                    "EROSION PANIC: rank=%d erosion=%.4f > panic=%.4f (%.1fx threshold=%.4f) "
                    "duration=%.2fs fire=%.4f%% current=%.4f%%",
                    self.signal_cfg.rank,
                    erosion,
                    panic_threshold,
                    ecfg.panic_multiplier,
                    threshold,
                    breach_duration,
                    fire_delta,
                    current_pct,
                )
                await self._execute_early_exit(
                    erosion,
                    threshold,
                    fire_delta,
                    current_pct,
                    order_mgr,
                    reason="post-fire erosion PANIC — sustained catastrophic reversal",
                )
                return
        else:
            # Breach cleared — reset the sustained-duration timer so a later
            # transient re-breach does not inherit an already-expired clock.
            self._panic_breach_started_at = None

        # --- Path 2: CUSUM accumulation ---
        excess = max(0.0, self._erosion_ema - threshold - ecfg.cusum_tolerance)
        if excess > 0:
            self._erosion_cusum += excess
        else:
            # Below threshold+tolerance - bleed off accumulated CUSUM
            self._erosion_cusum *= ecfg.cusum_decay

        if self._erosion_cusum >= ecfg.cusum_limit:
            # v2.9 sustain gate: require the CUSUM breach to persist for
            # cusum_sustain_s before firing. Single-tick CUSUM breaches in
            # v2.8 produced 2 hard + 1 partial false exits out of 9 (33%).
            if self._cusum_breach_started_at is None:
                self._cusum_breach_started_at = now_mono
            cusum_breach_duration = now_mono - self._cusum_breach_started_at
            if ecfg.cusum_sustain_s > 0.0 and cusum_breach_duration < ecfg.cusum_sustain_s:
                log.info(
                    "EROSION CUSUM BREACH (not yet sustained): rank=%d cusum=%.3f "
                    "limit=%.3f duration=%.2fs need=%.1fs",
                    self.signal_cfg.rank,
                    self._erosion_cusum,
                    ecfg.cusum_limit,
                    cusum_breach_duration,
                    ecfg.cusum_sustain_s,
                )
                return

            # v3.1 overwhelming-breach override. When the accumulator climbs
            # to cusum_override_multiplier * cusum_limit the excess is too
            # large to attribute to noise, so the reversal-pp and top-bid
            # suppressions below are bypassed. v3.0 01:47 loss had
            # cusum=1.608 (2.01x of 0.80) blocked by the reversal gate.
            override_active = (
                ecfg.cusum_override_multiplier > 0.0
                and self._erosion_cusum >= ecfg.cusum_override_multiplier * ecfg.cusum_limit
            )
            if override_active:
                log.warning(
                    "EROSION CUSUM OVERRIDE: rank=%d cusum=%.3f >= %.2fx "
                    "limit %.3f — bypassing reversal/top-bid suppressions",
                    self.signal_cfg.rank,
                    self._erosion_cusum,
                    ecfg.cusum_override_multiplier,
                    ecfg.cusum_limit,
                )

            # v3.0 delta-reversal gate: require the live delta to have
            # reversed by at least cusum_min_reversal_pp vs fire_delta before
            # an exit is allowed. In the v2.9 session, 4 of 6 CUSUM exits
            # were premature: every one had |current - fire| <= 0.14 pp,
            # while both valid exits had >= 0.16 pp. This gate preserves the
            # valid saves (e.g. v2.9 fire #14, reversal 0.32 pp) and blocks
            # the false ones (#4/#7/#8/#15, reversal <= 0.14 pp). The v3.1
            # override above bypasses this gate when the breach is enormous.
            reversal_pp = abs(current_pct - fire_delta)
            if (
                not override_active
                and ecfg.cusum_min_reversal_pp > 0.0
                and reversal_pp < ecfg.cusum_min_reversal_pp
            ):
                log.info(
                    "EROSION CUSUM SUPPRESSED (reversal too shallow): rank=%d "
                    "reversal=%.4f%% < min=%.4f%% cusum=%.3f erosion=%.4f "
                    "fire=%.4f%% current=%.4f%%",
                    self.signal_cfg.rank,
                    reversal_pp,
                    ecfg.cusum_min_reversal_pp,
                    self._erosion_cusum,
                    erosion,
                    fire_delta,
                    current_pct,
                )
                return

            # v2.9 price-aware suppression: if the Polymarket top bid on our
            # position's side already reflects a winning outcome, suppress
            # the CUSUM exit regardless of BTC delta erosion. v2.8 Trade 19
            # exited via CUSUM while bid_up=0.99; the orderbook was telling
            # us we were right and the BTC wobble was noise. The v3.1
            # override above bypasses this gate when the breach is enormous.
            sc = self.signal_cfg
            our_top_bid = (
                self.state.best_bid_up if sc.side == Direction.UP else self.state.best_bid_down
            )
            if (
                not override_active
                and ecfg.cusum_suppress_top_bid > 0.0
                and our_top_bid > 0.0
                and our_top_bid >= ecfg.cusum_suppress_top_bid
            ):
                log.info(
                    "EROSION CUSUM SUPPRESSED (market agrees): rank=%d side=%s "
                    "top_bid=%.3f >= suppress=%.3f cusum=%.3f erosion=%.4f",
                    sc.rank,
                    sc.side.value,
                    our_top_bid,
                    ecfg.cusum_suppress_top_bid,
                    self._erosion_cusum,
                    erosion,
                )
                return

            log.info(
                "EROSION CUSUM TRIGGERED: rank=%d cusum=%.3f >= limit=%.3f "
                "sustained=%.2fs ema=%.4f erosion=%.4f threshold=%.4f "
                "fire=%.4f%% current=%.4f%% top_bid=%.3f",
                self.signal_cfg.rank,
                self._erosion_cusum,
                ecfg.cusum_limit,
                cusum_breach_duration,
                self._erosion_ema,
                erosion,
                threshold,
                fire_delta,
                current_pct,
                our_top_bid,
            )
            await self._execute_early_exit(
                erosion,
                threshold,
                fire_delta,
                current_pct,
                order_mgr,
                reason="post-fire erosion exceeded safe threshold (sustained)",
            )
            return

        # CUSUM below limit — reset the sustain timer so a later re-breach
        # doesn't inherit an already-expired clock.
        self._cusum_breach_started_at = None

        # --- Below threshold or accumulating (throttled: 10s interval or meaningful change) ---
        now = time.monotonic()
        erosion_shifted = abs(erosion - self._erosion_last_logged_val) >= 0.15
        cusum_shifted = abs(self._erosion_cusum - self._erosion_last_logged_cusum) >= 0.05
        if now - self._erosion_last_log_time >= 10.0 or erosion_shifted or cusum_shifted:
            self._erosion_last_log_time = now
            self._erosion_last_logged_val = erosion
            self._erosion_last_logged_cusum = self._erosion_cusum
            log.info(
                "post-fire erosion: rank=%d erosion=%.4f ema=%.4f cusum=%.3f/%.3f "
                "threshold=%.4f fire=%.4f%% current=%.4f%%",
                self.signal_cfg.rank,
                erosion,
                self._erosion_ema,
                self._erosion_cusum,
                ecfg.cusum_limit,
                threshold,
                fire_delta,
                current_pct,
            )

    async def _execute_early_exit(
        self,
        erosion: float,
        threshold: float,
        fire_delta: float,
        current_pct: float,
        order_mgr: OrderExecutor,
        *,
        reason: str,
    ) -> None:
        """Sell the position and send notifications for an early exit."""
        self._early_exit_triggered = True
        sc = self.signal_cfg
        sell_price = self.state.best_bid_up if sc.side == Direction.UP else self.state.best_bid_down

        if sell_price <= 0:
            log.warning(
                "EARLY EXIT SKIP (no bid): rank=%d side=%s erosion=%.4f threshold=%.4f "
                "fire=%.4f%% current=%.4f%% entry=%.2f",
                sc.rank,
                sc.side.value,
                erosion,
                threshold,
                fire_delta,
                current_pct,
                self.last_entry_price,
            )
            send_early_exit(
                mode=order_mgr.mode,
                side=sc.side.value,
                rank=sc.rank,
                reason=f"{reason} (no bid available)",
                erosion=erosion,
                threshold=threshold,
                fire_delta_pct=fire_delta,
                current_delta_pct=current_pct,
                entry_price=self.last_entry_price,
                sell_price=0.0,
                signal_age_h=sc.signal_age_h,
            )
            return

        log.info(
            "EARLY EXIT TRIGGERED: rank=%d side=%s erosion=%.4f threshold=%.4f "
            "cusum=%.3f fire=%.4f%% current=%.4f%% sell_bid=%.2f entry=%.2f — %s",
            sc.rank,
            sc.side.value,
            erosion,
            threshold,
            self._erosion_cusum,
            fire_delta,
            current_pct,
            sell_price,
            self.last_entry_price,
            reason,
        )
        pnl = await order_mgr.exit_position_early(sell_price)
        self._exit_telemetry = EarlyExitTelemetry(
            erosion=erosion,
            threshold=threshold,
            reason=reason,
            fire_delta_pct=fire_delta,
            current_delta_pct=current_pct,
        )
        send_early_exit(
            mode=order_mgr.mode,
            side=sc.side.value,
            rank=sc.rank,
            reason=reason,
            erosion=erosion,
            threshold=threshold,
            fire_delta_pct=fire_delta,
            current_delta_pct=current_pct,
            entry_price=self.last_entry_price,
            sell_price=sell_price,
            pnl=pnl,
            signal_age_h=sc.signal_age_h,
        )

    # ------------------------------------------------------------------
    # Maker-first entry helpers
    # ------------------------------------------------------------------

    def _finalize_entry(
        self,
        order_id: str,
        entry_price: float,
        size_usd: float,
        entry_type: str,
        order_mgr: OrderExecutor,
        *,
        maker_usd: float = 0.0,
        taker_usd: float = 0.0,
    ) -> None:
        """Common finalization after a fill is confirmed (maker or taker).

        ``maker_usd`` / ``taker_usd`` default to 0.0 for the pure-maker and
        pure-taker paths — their entry_type label carries the full story.
        The partial-fill escalation branch passes both values so Discord can
        render the capital split (post-mortem 2026-04-22 §5.2).
        """
        sc = self.signal_cfg
        self._entry_complete = True
        self._order_placed = True
        self._fire_delta_pct = self._latest_pct
        self.last_entry_price = entry_price
        self.last_size_usd = size_usd
        self.bet_scale = self.sprt_factor

        order_mgr.set_rule_triggered(
            sc.rank,
            sc.side.value,
            self._fire_signal,
            obi_threshold=sc.obi_threshold,
            obi_depth=sc.obi_depth.value,
        )
        _kr = self.last_kelly_result
        _wr = self.kelly_wr_result
        order_mgr.set_kelly_fields(
            kelly_adjusted_p=_wr.adjusted_p if _wr else None,
            kelly_vol_discount=_wr.vol_discount if _wr else None,
            kelly_chop_discount=_wr.chop_discount if _wr else None,
            kelly_outcome_discount=_wr.outcome_discount if _wr else None,
            kelly_total_discount=_wr.total_discount if _wr else None,
            kelly_feedback_adj=_wr.feedback_adjustment if _wr else None,
            kelly_raw_f=_kr.raw_kelly if _kr else None,
            kelly_fractional_f=_kr.fractional_kelly if _kr else None,
            kelly_bet_size=_kr.bet_size if _kr else None,
            kelly_entry_price=entry_price,
            kelly_has_edge=_kr.has_edge if _kr else None,
            bankroll_before=self.bankroll,
            sprt_factor=self.sprt_factor,
            final_bet_size=size_usd,
        )
        send_bet_placed(
            mode=order_mgr.mode,
            side=sc.side.value,
            price=entry_price,
            size_usd=size_usd,
            rank=sc.rank,
            order_id=order_id,
            entry_type=entry_type,
            maker_usd=maker_usd,
            taker_usd=taker_usd,
            obi_threshold=sc.obi_threshold,
            obi_depth=sc.obi_depth.value,
            obi_observed=self._latest_obi,
            signal_age_h=sc.signal_age_h,
        )

    async def _monitor_maker_entry(
        self,
        time_remaining: float,
        order_mgr: OrderExecutor,
    ) -> None:
        """Check pending maker order: fully filled, timed out, or deadline?

        Partial-fill handling (post-mortem 2026-04-22 §5.2): a maker order
        that fills only part of the intended size must cancel the remainder
        and escalate a taker order for the unfilled quantity, rather than
        treating the partial as complete and silently forfeiting the rest.
        """
        sc = self.signal_cfg

        # Fully filled — finalize and done.
        assert self._maker_order_id is not None  # noqa: S101  # set when maker order placed
        if order_mgr.is_order_fully_filled(self._maker_order_id):
            filled = order_mgr.filled_usd(self._maker_order_id)
            self._finalize_entry(
                self._maker_order_id,
                self._maker_entry_price,
                filled,
                "maker",
                order_mgr,
            )
            log.info(
                "MAKER FILLED rank=%d side=%s price=%.2f size=$%.2f",
                sc.rank,
                sc.side.value,
                self._maker_entry_price,
                filled,
            )
            return

        elapsed = time.time() - self._maker_placed_at

        # Deadline: not enough time left for anything. Cancel and book any
        # partial that did land before close.
        if time_remaining < self.cfg.entry_window_stop:
            await order_mgr.cancel_order(self._maker_order_id)
            filled_at_deadline = order_mgr.filled_usd(self._maker_order_id)
            if filled_at_deadline > 0:
                self._finalize_entry(
                    self._maker_order_id,
                    self._maker_entry_price,
                    filled_at_deadline,
                    "maker",
                    order_mgr,
                )
                log.info(
                    "MAKER PARTIAL EXPIRED (deadline) rank=%d side=%s price=%.2f "
                    "filled=$%.2f of intended=$%.2f elapsed=%.1fs",
                    sc.rank,
                    sc.side.value,
                    self._maker_entry_price,
                    filled_at_deadline,
                    self._maker_size_usd,
                    elapsed,
                )
            else:
                self._entry_complete = True
                log.info(
                    "MAKER EXPIRED (deadline) rank=%d elapsed=%.1fs",
                    sc.rank,
                    elapsed,
                )
            return

        # Timeout: escalate to taker.
        if elapsed < self.cfg.maker_timeout_s:
            return  # still waiting

        # Snapshot filled portion BEFORE cancel (cancel may race with a
        # last-moment fill).
        filled_before_cancel = order_mgr.filled_usd(self._maker_order_id)
        cancelled = await order_mgr.cancel_order(self._maker_order_id)

        # Race: fully filled in the gap between our check and the cancel.
        if not cancelled and order_mgr.is_order_fully_filled(self._maker_order_id):
            filled = order_mgr.filled_usd(self._maker_order_id)
            self._finalize_entry(
                self._maker_order_id,
                self._maker_entry_price,
                filled,
                "maker",
                order_mgr,
            )
            log.info(
                "MAKER FILLED (during cancel) rank=%d price=%.2f size=$%.2f",
                sc.rank,
                self._maker_entry_price,
                filled,
            )
            return

        # Re-read filled after cancel — the cancel itself may have raced with
        # a fill, so use the larger value.
        filled_usd = max(
            filled_before_cancel,
            order_mgr.filled_usd(self._maker_order_id),
        )
        remaining_usd = max(0.0, self._maker_size_usd - filled_usd)
        partial = filled_usd > 0 and remaining_usd > 0

        if partial:
            log.info(
                "MAKER PARTIAL rank=%d side=%s price=%.2f filled=$%.2f of "
                "intended=$%.2f escalating=$%.2f elapsed=%.1fs",
                sc.rank,
                sc.side.value,
                self._maker_entry_price,
                filled_usd,
                self._maker_size_usd,
                remaining_usd,
                elapsed,
            )

        # Remainder below the min-bet floor — book the partial (if any) and stop.
        min_taker_bet = self.sizing_cfg.kelly_min_bet if self.sizing_cfg is not None else 1.0
        if remaining_usd < min_taker_bet:
            if filled_usd > 0:
                self._finalize_entry(
                    self._maker_order_id,
                    self._maker_entry_price,
                    filled_usd,
                    "maker",
                    order_mgr,
                )
            else:
                self._entry_complete = True
            return

        # Get current best ask for the taker leg.
        best_ask = self.state.best_ask_up if sc.side == Direction.UP else self.state.best_ask_down
        if best_ask <= 0:
            if filled_usd > 0:
                self._finalize_entry(
                    self._maker_order_id,
                    self._maker_entry_price,
                    filled_usd,
                    "maker",
                    order_mgr,
                )
            else:
                self._entry_complete = True
            log.warning(
                "TAKER SKIP (no ask) rank=%d after maker timeout",
                sc.rank,
            )
            return

        # Re-check Kelly edge at the current taker price — the ask may have
        # walked while we were resting the maker.
        if self.kelly_wr_result is not None and best_ask < 1.0:
            p = self.kelly_wr_result.adjusted_p
            b = (1.0 - best_ask) / best_ask
            raw_f = p - (1.0 - p) / b
            if raw_f <= 0:
                if filled_usd > 0:
                    self._finalize_entry(
                        self._maker_order_id,
                        self._maker_entry_price,
                        filled_usd,
                        "maker",
                        order_mgr,
                    )
                else:
                    self._entry_complete = True
                log.info(
                    "[SKIP] rank=%d reason=TAKER_NO_EDGE_AT_NEW_PRICE "
                    "ask=%.2f p=%.3f raw_f=%.3f maker_was=%.2f filled=$%.2f",
                    sc.rank,
                    best_ask,
                    p,
                    raw_f,
                    self._maker_entry_price,
                    filled_usd,
                )
                return

        # Place taker for the unfilled remainder only.
        taker_id = await order_mgr.place_taker_order(
            token_id=self._maker_token_id,
            price=best_ask,
            size_usd=remaining_usd,
            tier=self._maker_tier,
        )

        if taker_id is not None:
            # Combined finalization: maker partial + taker remainder become a
            # single volume-weighted entry for downstream window accounting.
            total_usd = filled_usd + remaining_usd
            if filled_usd > 0:
                weighted_price = (
                    filled_usd * self._maker_entry_price + remaining_usd * best_ask
                ) / total_usd
                entry_type = "maker+taker"
                self._finalize_entry(
                    taker_id,
                    weighted_price,
                    total_usd,
                    entry_type,
                    order_mgr,
                    maker_usd=filled_usd,
                    taker_usd=remaining_usd,
                )
            else:
                # Pure taker escalation (zero maker fill) — no split to render.
                self._finalize_entry(
                    taker_id,
                    best_ask,
                    total_usd,
                    "taker",
                    order_mgr,
                )
            log.info(
                "TAKER ESCALATION rank=%d side=%s maker_price=%.2f "
                "taker_price=%.2f elapsed=%.1fs maker_filled=$%.2f "
                "taker_size=$%.2f total=$%.2f",
                sc.rank,
                sc.side.value,
                self._maker_entry_price,
                best_ask,
                elapsed,
                filled_usd,
                remaining_usd,
                total_usd,
            )
        elif filled_usd > 0:
            self._finalize_entry(
                self._maker_order_id,
                self._maker_entry_price,
                filled_usd,
                "maker",
                order_mgr,
            )
            log.warning(
                "TAKER FAILED rank=%d booking maker partial $%.2f only",
                sc.rank,
                filled_usd,
            )
        else:
            self._entry_complete = True
            log.warning(
                "TAKER FAILED rank=%d after maker timeout",
                sc.rank,
            )

    # ------------------------------------------------------------------
    # Main evaluation — called every strategy tick
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        signal: Signal,
        time_remaining: float,
        order_mgr: OrderExecutor,
    ) -> None:
        if not self.cfg.enabled:
            return

        # Post-fire erosion monitoring — runs after entry is complete
        if self._entry_complete and self._order_placed and not self._early_exit_triggered:
            await self._monitor_post_fire_erosion(signal, order_mgr)
            return

        # Terminal states (entry complete but no fill, or early exit already triggered)
        if self._entry_complete:
            return

        # Monitor pending maker order (takes priority over everything)
        if self._maker_order_id is not None:
            await self._monitor_maker_entry(time_remaining, order_mgr)
            return

        # Already fired conditions (no maker pending = conditions failed or taker-only completed)
        if self._order_placed or self._fired:
            return
        if time_remaining < self.cfg.entry_window_stop:
            return

        # Wait until we have both price feeds and fresh orderbook data
        if (
            self.state.window_open_price <= 0
            or self.state.binance_window_open_price <= 0
            or not self.state.has_fresh_book_data
        ):
            if not self._logged_no_data:
                log.info("momentum_signal: waiting for full market data")
                self._logged_no_data = True
            return

        sc = self.signal_cfg
        # Convert fraction to percent to match engine units
        bn_dir_pct = signal.bn_direction_from_open_pct * 100.0

        if sc.observe_to_s <= time_remaining <= sc.observe_from_s:
            # Inside observation window — accumulate, don't fire yet
            self._accumulate(bn_dir_pct, time_remaining, self._gate_obi(signal))
            return

        if time_remaining >= sc.observe_from_s:
            # Haven't entered the observation window yet
            return

        # Narrow-window grace: keep accumulating for a few seconds past
        # observe_to_s so a signal whose delta crossed the threshold just
        # after window close still gets a chance to fire.
        window_span = sc.observe_from_s - sc.observe_to_s
        if (
            self.cfg.post_observe_grace_s > 0.0
            and window_span < self.cfg.narrow_observe_window_threshold_s
        ):
            grace_floor = sc.observe_to_s - self.cfg.post_observe_grace_s
            if time_remaining >= grace_floor:
                self._accumulate(bn_dir_pct, time_remaining, self._gate_obi(signal))
                return

        # Window (+ grace if applicable) has elapsed — evaluate once.
        self._fired = True  # only evaluate once per window
        if self._n < 5:
            log.info(
                "momentum_signal rank #%d: insufficient ticks (%d < 5) in window "
                "[%.0f→%.0f]s — observation window may have been missed at bot start",
                sc.rank,
                self._n,
                sc.observe_from_s,
                sc.observe_to_s,
            )
            return

        stddev = self._population_stddev()
        log.info(
            "signal_eval rank=%d side=%s latest_pct=%.4f stddev=%.4f n=%d "
            "min_delta=%.2f max_var=%.3f",
            sc.rank,
            sc.side.value,
            self._latest_pct,
            stddev,
            self._n,
            sc.min_delta_pct,
            sc.max_variance_pct,
        )

        if not self._conditions_met():
            log.info(
                "[SKIP] rank=%d side=%s reason=conditions_not_met "
                "latest_pct=%.4f need=%s%.2f stddev=%.4f max_var=%.3f",
                sc.rank,
                sc.side.value,
                self._latest_pct,
                ">=" if sc.side == Direction.UP else "<=",
                sc.min_delta_pct if sc.side == Direction.UP else -sc.min_delta_pct,
                stddev,
                sc.max_variance_pct,
            )
            return

        token_id = self.state.up_token_id if sc.side == Direction.UP else self.state.down_token_id
        if not token_id:
            log.warning(
                "momentum_signal rank #%d: no token_id for side=%s — market tokens not yet fetched",
                sc.rank,
                sc.side.value,
            )
            return

        # Dynamic entry pricing: current best ask
        best_ask = self.state.best_ask_up if sc.side == Direction.UP else self.state.best_ask_down
        if best_ask <= 0:
            log.warning(
                "momentum_signal rank #%d: no best_ask for side=%s — cannot price order",
                sc.rank,
                sc.side.value,
            )
            return

        entry_price = best_ask

        # v3.2 §5.2: directional-regime gate. t-stat of the rolling signed
        # 5-min returns from the volatility tracker. Positive = up-trending
        # regime, negative = down-trending. Skip when magnitude exceeds
        # threshold AND the trend opposes the signal side — catches the
        # hostile "BTC drifting against this signal for hours" regime the
        # direction-agnostic vol/chop axes cannot see.
        if self.cfg.directional_gate_enabled and self.cfg.directional_threshold_t > 0.0:
            t = self.directional_t
            thresh = self.cfg.directional_threshold_t
            opposes = (sc.side == Direction.UP and t <= -thresh) or (
                sc.side == Direction.DOWN and t >= thresh
            )
            if opposes:
                log.info(
                    "[SKIP] rank=%d side=%s reason=DIRECTIONAL_OPPOSES t_stat=%.2f thresh=%.2f",
                    sc.rank,
                    sc.side.value,
                    t,
                    thresh,
                )
                return

        # v3.2 §5.7: rolling-OBI gate. The engine's spot-OBI confirmation
        # (obi_threshold → _latest_obi) flips on a single depth update;
        # averaging over the last obi_rolling_window_s gives a flow-direction
        # read that survives individual-tick noise. Skip if the smoothed OBI
        # opposes signal direction by ≥ skip_threshold.
        if self.cfg.obi_rolling_gate_enabled and self.cfg.obi_rolling_window_s > 0.0:
            mean_obi, n_obi = self._mean_recent_obi(self.cfg.obi_rolling_window_s)
            if n_obi >= self.cfg.obi_rolling_min_samples:
                thresh = self.cfg.obi_rolling_skip_threshold
                opposes = (sc.side == Direction.UP and mean_obi <= -thresh) or (
                    sc.side == Direction.DOWN and mean_obi >= thresh
                )
                if opposes:
                    log.info(
                        "[SKIP] rank=%d side=%s reason=OBI_ROLLING_OPPOSES "
                        "mean_obi=%.3f n=%d window=%.1fs thresh=%.3f",
                        sc.rank,
                        sc.side.value,
                        mean_obi,
                        n_obi,
                        self.cfg.obi_rolling_window_s,
                        thresh,
                    )
                    return

        # Market-agreement filter: skip when live price deviates too far from
        # the signal's historical avg_entry_price (market strongly disagrees)
        if (
            self.cfg.max_entry_gap_pct > 0
            and sc.avg_entry_price is not None
            and sc.avg_entry_price > 0
        ):
            discount_pct = (sc.avg_entry_price - entry_price) / sc.avg_entry_price * 100.0
            if discount_pct > self.cfg.max_entry_gap_pct:
                log.info(
                    "[SKIP] rank=%d side=%s reason=market_disagrees "
                    "discount=%.1f%% max=%.1f%% ask=%.2f avg_entry=%.2f",
                    sc.rank,
                    sc.side.value,
                    discount_pct,
                    self.cfg.max_entry_gap_pct,
                    entry_price,
                    sc.avg_entry_price,
                )
                return

        # Kelly Criterion sizing — compute bet size from adjusted win rate + entry price
        if self.kelly_wr_result is not None and self.sizing_cfg is not None:
            max_bet = self.bankroll * self.sizing_cfg.kelly_max_bet_pct / 100.0

            kr = kelly_size(
                p=self.kelly_wr_result.adjusted_p,
                entry_price=entry_price,
                bankroll=self.bankroll,
                kelly_fraction=self.sizing_cfg.kelly_fraction,
                min_bet=self.sizing_cfg.kelly_min_bet,
                max_bet=max_bet,
            )
            self.last_kelly_result = kr

            # Kelly no-edge gate: if raw_kelly ≤ 0, trade has no edge at this price
            if not kr.has_edge:
                log.info(
                    "[SKIP] rank=%d side=%s reason=KELLY_NO_EDGE "
                    "p_adj=%.3f entry=%.2f raw_f=%.3f "
                    "total_disc=%.3f vol=%.3f chop=%.3f outcome=%.3f",
                    sc.rank,
                    sc.side.value,
                    self.kelly_wr_result.adjusted_p,
                    entry_price,
                    kr.raw_kelly,
                    self.kelly_wr_result.total_discount,
                    self.kelly_wr_result.vol_discount,
                    self.kelly_wr_result.chop_discount,
                    self.kelly_wr_result.outcome_discount,
                )
                return

            # v3.2 §5.6 Polymarket-implied probability cross-check. best_ask is the
            # market's price for YES — i.e. the implied probability the side resolves.
            # Kelly's has_edge only requires adjusted_p > entry_price; this tightens
            # to a minimum edge in probability points, skipping fires where the market
            # has already priced in most of our signal. Set kelly_min_edge_pp=0 to disable.
            if self.sizing_cfg.kelly_min_edge_pp > 0.0:
                edge_pp = (self.kelly_wr_result.adjusted_p - entry_price) * 100.0
                if edge_pp < self.sizing_cfg.kelly_min_edge_pp:
                    log.info(
                        "[SKIP] rank=%d side=%s reason=IMPLIED_PROB_TOO_CLOSE "
                        "p_adj=%.3f entry=%.3f edge=%.2fpp min=%.2fpp",
                        sc.rank,
                        sc.side.value,
                        self.kelly_wr_result.adjusted_p,
                        entry_price,
                        edge_pp,
                        self.sizing_cfg.kelly_min_edge_pp,
                    )
                    return

            # v3.0 P6 / v3.1 / v3.2 hostile-regime stack. The regime cap already
            # reduces adjusted_p (and therefore raw Kelly); on top of that:
            #   - skip-metric > skip_threshold: abort the fire entirely
            #   - halve-metric > hostile_regime_threshold: halve bet dollars
            # v3.2: outcome_discount participates in the halve metric (v3.1
            # T4 lost $22 with outcome_sev=0.381 that the halve gate ignored)
            # but NOT in the skip metric — outcome's empirical cap sits near
            # the skip threshold, and skipping on outcome alone kills too many
            # real wins (v3.1 T5/T7/T9). Vol/chop keep their full skip path.
            _hostile_halve_metric = max(
                self.kelly_wr_result.vol_discount,
                self.kelly_wr_result.chop_discount,
                self.kelly_wr_result.outcome_discount,
            )
            _hostile_skip_metric = max(
                self.kelly_wr_result.vol_discount,
                self.kelly_wr_result.chop_discount,
            )
            _hostile_threshold = self.sizing_cfg.hostile_regime_threshold
            _hostile_skip_threshold = self.sizing_cfg.hostile_regime_skip_threshold
            if _hostile_skip_threshold > 0.0 and _hostile_skip_metric > _hostile_skip_threshold:
                log.info(
                    "[SKIP] rank=%d side=%s reason=HOSTILE_REGIME_SKIP "
                    "vol=%.3f chop=%.3f max=%.3f > skip_thresh=%.3f",
                    sc.rank,
                    sc.side.value,
                    self.kelly_wr_result.vol_discount,
                    self.kelly_wr_result.chop_discount,
                    _hostile_skip_metric,
                    _hostile_skip_threshold,
                )
                return
            if _hostile_halve_metric > _hostile_threshold:
                _hostile_mult = self.sizing_cfg.hostile_regime_multiplier
                _pre = kr.bet_size
                kr = KellyResult(
                    raw_kelly=kr.raw_kelly,
                    fractional_kelly=kr.fractional_kelly,
                    bet_size=kr.bet_size * _hostile_mult,
                    has_edge=kr.has_edge,
                    implied_ev=kr.implied_ev,
                )
                self.last_kelly_result = kr
                log.info(
                    "HOSTILE REGIME: vol=%.3f chop=%.3f outcome=%.3f max=%.3f > thresh=%.3f "
                    "— bet $%.2f -> $%.2f (x%.2f)",
                    self.kelly_wr_result.vol_discount,
                    self.kelly_wr_result.chop_discount,
                    self.kelly_wr_result.outcome_discount,
                    _hostile_halve_metric,
                    _hostile_threshold,
                    _pre,
                    kr.bet_size,
                    _hostile_mult,
                )

            # Apply SPRT factor on top of Kelly (signal confidence, separate concern)
            size_usd = round(kr.bet_size * self.sprt_factor, 2)

            # Warmup clamp — cap to minimum bet while safety mechanisms collect data
            if self.warmup_active and size_usd > self.sizing_cfg.kelly_min_bet:
                log.info(
                    "WARMUP clamping bet=$%.2f to min=$%.2f",
                    size_usd,
                    self.sizing_cfg.kelly_min_bet,
                )
                size_usd = self.sizing_cfg.kelly_min_bet

            if size_usd < self.sizing_cfg.kelly_min_bet:
                log.info(
                    "[SKIP] rank=%d side=%s reason=KELLY_below_min "
                    "kelly_bet=$%.2f sprt=%.2f final=$%.2f min=$%.2f",
                    sc.rank,
                    sc.side.value,
                    kr.bet_size,
                    self.sprt_factor,
                    size_usd,
                    self.sizing_cfg.kelly_min_bet,
                )
                return

            _wr = self.kelly_wr_result
            _base_p = sc.conservative_p(self.sizing_cfg.wilson_max_shrink_pct)
            log.info(
                "KELLY sizing: p_adj=%.3f base=%.3f total_disc=%.3f "
                "vol=%.3f chop=%.3f outcome=%.3f feedback=%+.3f",
                _wr.adjusted_p,
                _base_p,
                _wr.total_discount,
                _wr.vol_discount,
                _wr.chop_discount,
                _wr.outcome_discount,
                _wr.feedback_adjustment,
            )
            log.info(
                "KELLY bet: entry=%.2f raw_f=%.3f frac_f=%.3f "
                "bankroll=$%.2f kelly_bet=$%.2f sprt=%.2f final=$%.2f",
                entry_price,
                kr.raw_kelly,
                kr.fractional_kelly,
                self.bankroll,
                kr.bet_size,
                self.sprt_factor,
                size_usd,
            )
        else:
            # Fallback: Kelly context not yet available (should not happen in steady state)
            base_size = self.sizing_cfg.kelly_min_bet if self.sizing_cfg else 1.0
            size_usd = round(base_size * self.bet_scale, 2)
            self.last_kelly_result = None
            win_rate = (
                sc.conservative_p(self.sizing_cfg.wilson_max_shrink_pct)
                if self.sizing_cfg
                else sc.oos_win_rate_pct / 100.0
            )
            implied_ev = win_rate * (1.0 - entry_price) - (1.0 - win_rate) * entry_price
            log.info(
                "FIRED (fallback) rank=%d side=%s t=%.1fs ask=%.2f size=$%.2f implied_ev=%.4f",
                sc.rank,
                sc.side.value,
                time_remaining,
                entry_price,
                size_usd,
                implied_ev,
            )

        # Store signal snapshot and bet_scale for finalization
        self._fire_signal = signal
        self.bet_scale = self.sprt_factor
        tier = f"momentum{sc.rank}"

        obi_gate = (
            f"{sc.obi_threshold:.2f}@{sc.obi_depth.value}" if sc.obi_threshold > 0.0 else "off"
        )
        log.info(
            "FIRED rank=%d side=%s t=%.1fs "
            "dir=%.4f%% stddev=%.4f%% n=%d "
            "oos_wr=%.1f%% bh_p=%.3g "
            "avg_entry=%.2f ask=%.2f size=$%.2f "
            "obi_gate=%s obs_obi=%+.4f",
            sc.rank,
            sc.side.value,
            time_remaining,
            self._latest_pct,
            stddev,
            self._n,
            sc.oos_win_rate_pct,
            sc.bh_adjusted_p_value,
            sc.avg_entry_price or 0.0,
            entry_price,
            size_usd,
            obi_gate,
            self._latest_obi,
        )

        # --- Maker-first entry ---
        # Ultra-high-confidence fast path: cross the spread on fire instead
        # of resting a maker. On strong, tight signals the ask can walk away
        # within the maker timeout and turn a correct prediction into a
        # TAKER_NO_EDGE_AT_NEW_PRICE no-fill; paying ~1¢ on entry is cheap
        # insurance against that. Gated by oos_wr and (optionally) stddev;
        # setting either threshold to 0 disables that gate.
        skip_maker = (
            self.cfg.skip_maker_min_oos_wr_pct > 0.0
            and sc.oos_win_rate_pct >= self.cfg.skip_maker_min_oos_wr_pct
            and (
                self.cfg.skip_maker_max_stddev_pct <= 0.0
                or stddev <= self.cfg.skip_maker_max_stddev_pct
            )
        )
        if skip_maker:
            log.info(
                "SKIP_MAKER (high conf) rank=%d oos_wr=%.1f%% stddev=%.4f%% \u2192 cross on fire",
                sc.rank,
                sc.oos_win_rate_pct,
                stddev,
            )

        maker_timeout = 0.0 if skip_maker else self.cfg.maker_timeout_s
        if maker_timeout > 0:
            maker_price = round(best_ask - 0.01, 2)
            if maker_price <= 0:
                maker_price = best_ask  # safety fallback

            order_id = await order_mgr.place_maker_order(
                token_id=token_id,
                price=maker_price,
                size_usd=size_usd,
                tier=tier,
            )

            if order_id is not None:
                # Maker placed — monitor for fill on subsequent ticks
                self._maker_order_id = order_id
                self._maker_placed_at = time.time()
                self._maker_entry_price = maker_price
                self._maker_token_id = token_id
                self._maker_size_usd = size_usd
                self._maker_tier = tier
                log.info(
                    "MAKER PLACED rank=%d side=%s price=%.2f (ask=%.2f -1tick) "
                    "size=$%.2f timeout=%ds",
                    sc.rank,
                    sc.side.value,
                    maker_price,
                    best_ask,
                    size_usd,
                    int(maker_timeout),
                )
                return
            # Maker rejected (price would cross spread) — fall through to taker
            log.info(
                "maker rejected rank=%d (price=%.2f ask=%.2f), falling through to taker",
                sc.rank,
                maker_price,
                best_ask,
            )

        # --- Taker entry (maker disabled, rejected, or not enough time) ---
        order_id = await order_mgr.place_taker_order(
            token_id=token_id,
            price=entry_price,
            size_usd=size_usd,
            tier=tier,
        )

        if order_id is not None:
            self._finalize_entry(order_id, entry_price, size_usd, "taker", order_mgr)
