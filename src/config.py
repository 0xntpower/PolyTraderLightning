"""Load, validate, and expose bot configuration."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import (
    Any,  # YAML boundary: untyped config values narrowed via dataclasses
)

import yaml

_log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yml"

DEFAULT_CONFIG = """\
# =============================================================================
# Polymarket BTC 5-Minute Trading Bot - Configuration
# =============================================================================

# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------
mode:
  trading: "paper"         # "paper" or "live"
  log_level: "INFO"        # DEBUG, INFO, WARNING, ERROR
  log_dir: "data/logs"     # directory for log files

# ---------------------------------------------------------------------------
# Strategy - signal execution and entry filters
# ---------------------------------------------------------------------------
rules_strategy:
  enabled: true
  max_position_usd: 75.0    # Max USDC to risk per window
  entry_window_stop: 5      # Stop entering within N seconds of close
  min_win_rate: 0.50         # Reject signals below this OOS win rate (fraction)
  maker_timeout_s: 3.0       # Maker order timeout before taker fallback (0=taker only)
  max_chop_flips: 0          # Skip trade if avg chop flips >= this (0=disabled)
  max_entry_gap_pct: 15.0    # Skip if bid >N% below signal avg_entry (0=disabled)
  skip_maker_min_oos_wr_pct: 95.0   # Cross spread on fire when oos_wr >= this (0=disabled)
  skip_maker_max_stddev_pct: 0.035  # AND-gate: also require stddev <= this (0=no stddev gate)
  # Subsample the variance accumulator to this cadence (seconds) so the bot's
  # live stddev lines up with the engine's historical backtest values. Must
  # match PolyDataCollector's snapshot_interval_s — the engine's max_variance
  # thresholds were measured on data sampled at that rate, and the bot sees
  # Binance updates faster than the collector records them. Without this,
  # live stddev runs systematically higher than the backtest and marginal
  # cells get rejected live that would have passed historically. 0 disables
  # subsampling (accept every strategy tick).
  variance_subsample_interval_s: 1.0
  # Post-window grace for narrow observation windows. Signals whose window
  # span is < narrow_observe_window_threshold_s are re-evaluated up to
  # post_observe_grace_s after window close, so that price movement barely
  # missing the window (e.g. delta crossing threshold ~5s late) still fires.
  narrow_observe_window_threshold_s: 60.0
  post_observe_grace_s: 10.0
  # v3.2 §5.7: rolling-window Binance OBI gate. The per-signal obi_threshold
  # reads _latest_obi (spot, noisy, can flip on a single depth update). This
  # gate averages OBI over the last N seconds and skips when the smoothed
  # value opposes signal direction. Complements, does not replace, the spot
  # confirmation.
  obi_rolling_gate_enabled: true
  obi_rolling_window_s: 20.0     # 10-30s recommendation range
  obi_rolling_min_samples: 4     # min samples required before gating activates
  # v3.4: default matches the mid engine bucket in kObiThresholdLevels.
  obi_rolling_skip_threshold: 0.05
  # v3.2 §5.2: directional-regime gate. Skips the fire when the t-stat of
  # the last N 5-min signed returns shows BTC significantly drifting against
  # the signal side. Catches the "trending against us" regime that the
  # direction-agnostic vol/chop axes miss.
  directional_gate_enabled: true
  directional_threshold_t: 2.0
  directional_min_samples: 6

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
risk:
  max_daily_loss_usd: 100.0
  max_consecutive_losses: 10
  cancel_unfilled_at_sec: 5
  max_position_per_window_usd: 75.0
  # v3.2 §5.8: post-loss cooldown. After a settled loss whose magnitude
  # exceeds ``post_loss_cooldown_loss_pct`` of bankroll, freeze trading
  # for the next ``post_loss_cooldown_windows`` window(s). Meant to
  # break tail-loss clusters where successive fires compound into the
  # same regime shift. Set windows to 0 to disable.
  post_loss_cooldown_enabled: true
  post_loss_cooldown_loss_pct: 2.0
  post_loss_cooldown_windows: 1

# ---------------------------------------------------------------------------
# Data connections
# ---------------------------------------------------------------------------
connections:
  binance_ws: "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@depth20@100ms"
  rtds_ws: "wss://ws-live-data.polymarket.com"
  clob_market_ws: "wss://ws-subscriptions-clob.polymarket.com/ws/market"
  clob_user_ws: "wss://ws-subscriptions-clob.polymarket.com/ws/user"
  clob_rest: "https://clob.polymarket.com"
  gamma_rest: "https://gamma-api.polymarket.com"
  chain_id: 137
  signature_type: 1          # Polymarket CLOB: 0=EOA, 1=Magic/proxy, 2=Gnosis Safe
  rtds_ping_interval_sec: 5
  reconnect_base_delay_sec: 1.0
  reconnect_max_delay_sec: 30.0
  binance_stale_sec: 15.0
  chainlink_stale_sec: 30.0
  clob_book_stale_sec: 60.0

# ---------------------------------------------------------------------------
# Paper trading
# ---------------------------------------------------------------------------
paper:
  log_every_window: true
  simulated_fill_delay_sec: 2.5
  starting_balance_usd: 1000.0

# ---------------------------------------------------------------------------
# IPC - orchestrator signal delivery
# ---------------------------------------------------------------------------
ipc:
  host: "127.0.0.1"
  port: 19731
  stale_signal_warning_hours: 6
  visualizer_enabled: true
  visualizer_host: "127.0.0.1"
  visualizer_port: 19732

# ---------------------------------------------------------------------------
# Signal lifecycle - signal health, decay, and age scaling
# ---------------------------------------------------------------------------
signal_lifecycle:
  fire_stall_windows: 50         # ~4 hours with no fires = fire-rate stall
  shadow_tracking_windows: 500   # ~42 hours of continued observation after decay
  bet_scaling_enabled: true
  age_taper_start_windows: 300   # ~25 hours - no age effect before this
  age_taper_end_windows: 500     # ~42 hours - age taper fully applied
  age_floor: 0.5                 # age alone never reduces below 50%
  min_bet_scale: 0.10            # absolute floor - never bet less than 10% of base
  sprt_activation_minutes: 45    # SPRT only affects sizing after this long without update

# ---------------------------------------------------------------------------
# Bet sizing - Kelly criterion and bankroll management
# ---------------------------------------------------------------------------
sizing:
  kelly_fraction: 0.25           # quarter-Kelly (conservative start)
  kelly_min_bet: 1.00            # minimum $1 bet, below this skip
  kelly_max_bet_pct: 5.0         # max bet as % of bankroll
  bankroll: 1000.00              # starting/current bankroll for Kelly sizing
  warmup_minutes: 30.0           # cap bets to min for this long after start (0=off)
  wilson_max_shrink_pct: 3.0     # max Wilson win rate correction (percentage points)
  kelly_regime_cap: 0.12         # max regime penalty on win rate (all factors combined)
  vol_weight: 1.0                # regime weight for volatility (0-1)
  chop_weight: 1.0               # regime weight for chop (0-1)
  outcome_weight: 0.8            # regime weight for outcome bias (0-1)
  feedback_min_trades: 10        # trades before performance feedback activates
  # v3.0 P6 / v3.2: hostile regime double-cap. When max(vol_disc, chop_disc,
  # outcome_disc) exceeds hostile_regime_threshold, bet dollars are
  # multiplied by hostile_regime_multiplier in addition to the regime cap
  # reducing adjusted_p. Threshold lowered from 0.20 to 0.15 after v3.1
  # paper loss (T3 at 0.161 went untouched); outcome_disc added to the max
  # after T4 lost $22 with outcome_sev=0.381 that the v3.1 gate ignored.
  hostile_regime_threshold: 0.15
  hostile_regime_multiplier: 0.5
  # v3.1: hostile-regime SKIP gate. When max(vol_disc, chop_disc) exceeds
  # this value the fire is aborted entirely instead of merely halved.
  # outcome_disc is deliberately excluded from the skip metric (it only
  # participates in halving) because outcome's empirical cap sits near the
  # skip threshold and skipping on it alone killed v3.1 T5/T7/T9 wins in
  # the counterfactual. Leaves a halving band intact. 0 disables.
  hostile_regime_skip_threshold: 0.25
  # v3.2 §5.3: soft-OR combine of per-axis severities instead of max(...)
  # — compounds moderate readings across vol/chop/outcome into a larger
  # total_discount (v3.1 T4 failure mode). Per-axis contribs unchanged.
  kelly_soft_or_combine: true
  # v3.2 §5.10: adaptive discount cap. 1 hot axis → kelly_regime_cap (0.12);
  # 2 hot axes → kelly_regime_cap_2_axes (0.20); 3 hot axes →
  # kelly_regime_cap_3_axes (0.30). Hot = severity*weight ≥ threshold.
  kelly_regime_cap_2_axes: 0.20
  kelly_regime_cap_3_axes: 0.30
  kelly_hot_axis_threshold: 0.33
  # v3.2 §5.6: Polymarket-implied probability cross-check. Require
  # adjusted_p - ask >= kelly_min_edge_pp/100 before firing, so the signal
  # has meaningful edge above what the market has already priced in. 0
  # disables (Kelly's no-edge check remains as the floor).
  kelly_min_edge_pp: 2.0

# ---------------------------------------------------------------------------
# Regime detection - volatility, chop, and outcome bias
# ---------------------------------------------------------------------------
regime:
  vol_lookback_windows: 24       # ~2 hours of 5-min returns
  vol_normal_pct: 0.10           # normal vol (stddev) - no bet reduction
  vol_high_pct: 0.30             # high vol (stddev) - full severity
  vol_min_samples: 6             # min returns before vol scaling is active
  chop_lookback_windows: 6       # ~30 min of recent windows
  chop_normal_flips: 5.0         # normal direction flips per window
  chop_high_flips: 10.0          # high chop - full severity
  chop_min_samples: 3            # min windows before chop scaling is active
  outcome_lookback_windows: 6    # rolling window of recent outcomes
  outcome_normal_agreement: 0.50 # 50% agreement = no concern
  outcome_high_agreement: 0.15   # 15% agreement = full severity
  # v3.2 §5.9: magnitude-weighted outcome agreement. Instead of treating
  # every resolved window equally, weight by |close_delta_pct| so tiny-
  # move windows don't drag the regime indicator around. Set to false
  # to fall back to the legacy count-based fraction.
  outcome_magnitude_weighted: true
  outcome_min_magnitude_pct: 0.01   # treat deltas below this as floor weight
  cache_staleness_minutes: 30.0  # discard cached regime data older than this (0=off)
  # v3.2 short-horizon EWMA volatility (RiskMetrics λ≈0.94)
  vol_fast_enabled: true             # feed EWMA vol from strategy tick loop
  vol_fast_sample_interval_s: 10.0   # one kept sample per N seconds
  vol_fast_decay_lambda: 0.94        # RiskMetrics default (~2min half-life at 10s)
  vol_fast_min_samples: 6            # min updates before fast vol is valid
  vol_fast_horizon_s: 300.0          # horizon-scale per-sample stddev to 5-min equivalent
  intra_window_refresh_s: 240.0      # refresh Kelly context every tick in last N seconds (0=off)

# ---------------------------------------------------------------------------
# Erosion - post-fire CUSUM exit detection
# ---------------------------------------------------------------------------
erosion:
  ema_alpha: 0.10                # EMA smoothing on raw erosion (half-life ~1.7s at 4Hz)
  cusum_tolerance: 0.05          # ignore exceedances < 5% above threshold
  cusum_limit: 0.80              # cumulative excess needed to trigger exit
  cusum_decay: 0.95              # CUSUM bleed-off rate when erosion dips below threshold
  panic_multiplier: 2.20         # panic only on genuine catastrophic reversal
  panic_min_duration_s: 3.0      # breach must persist N seconds before panic fires
  cusum_sustain_s: 4.0           # v2.9: CUSUM breach must persist N seconds before exit
  cusum_suppress_top_bid: 0.85   # v2.9: suppress CUSUM exit if our side's top bid >= this
  # v3.0 delta-reversal gate. Only permit a CUSUM exit when the live delta
  # has reversed by at least this many percentage points versus the fire-time
  # delta. v2.9 session post-mortem: 4 of 6 CUSUM exits were premature and
  # every one of them had |current - fire| <= 0.14 pp; the 2 valid exits both
  # had reversals >= 0.16 pp. 0 disables the gate.
  cusum_min_reversal_pp: 0.15
  # v3.1: overwhelming-breach override. When the CUSUM accumulator reaches
  # cusum_override_multiplier * cusum_limit, the reversal-pp and top-bid
  # suppressions are bypassed because the breach is too large to attribute
  # to noise. 0 disables the override (always honor suppressions).
  # v3.2: raised 2.0 -> 3.8. v3.1 trade #10 was a false exit at cusum=2.87
  # (3.58x limit); v3.1 trades #3 and #4 were correct exits at cusum 7.08
  # and 4.07 (8.85x and 5.09x). Threshold 3.8 (=3.04 cusum) blocks the
  # false exit while preserving both true-positive catches.
  cusum_override_multiplier: 3.8
"""


@dataclass(frozen=True, slots=True)
class ModeConfig:
    trading: str = "paper"
    log_level: str = "INFO"
    log_dir: str = "data/logs"


@dataclass(frozen=True, slots=True)
class RulesStrategyConfig:
    enabled: bool = True
    max_position_usd: float = 75.0
    entry_window_stop: int = 5
    min_win_rate: float = 0.50
    maker_timeout_s: float = 3.0
    max_chop_flips: int = 0  # skip if avg chop flips >= this (0=disabled)
    max_entry_gap_pct: float = 15.0  # skip if bid >N% below avg_entry (0=disabled)
    # High-confidence fast path: when oos_wr is high enough (and optionally
    # stddev is low enough), skip the maker quote and cross the spread on
    # fire. Prevents ask walk-away from turning correct predictions into
    # no-fills via TAKER_NO_EDGE_AT_NEW_PRICE. 0 disables either gate.
    skip_maker_min_oos_wr_pct: float = 95.0
    skip_maker_max_stddev_pct: float = 0.035
    # Subsample the live variance accumulator to this cadence (seconds) so the
    # bot's stddev matches the engine's backtest values. Must match
    # PolyDataCollector's snapshot_interval_s on the lab machine — the engine's
    # max_variance thresholds were measured on data sampled at that rate. The
    # bot cannot read this from the collector config because it runs on a
    # separate VPS, so it's set here explicitly. 0 disables subsampling.
    variance_subsample_interval_s: float = 1.0
    # Narrow-window grace: signals with (observe_from_s - observe_to_s) <
    # narrow_observe_window_threshold_s get an extra grace period after the
    # observation window closes before being evaluated. Motivated by the
    # 2026-04-12 16:05 UP signal (window [200,160], delta crossed threshold
    # ~5s post-window close and was missed). Grace only fires if the window
    # was narrow; wide windows evaluate at close as before.
    narrow_observe_window_threshold_s: float = 60.0
    post_observe_grace_s: float = 10.0
    # v3.2 §5.7: rolling-window Binance OBI gate. The per-signal
    # ``obi_threshold`` check reads ``_latest_obi`` (spot value at fire-time)
    # which is noisy and can flip on a single depth update. This gate averages
    # OBI over the last N seconds and skips when that smoothed value opposes
    # the signal direction by more than ``obi_rolling_skip_threshold``.
    # Complements, does not replace, the spot OBI check.
    obi_rolling_gate_enabled: bool = True
    obi_rolling_window_s: float = 20.0  # 10-30s typical (recommendation range)
    obi_rolling_min_samples: int = 4  # require this many samples before gating
    # Skip magnitude: an UP fire is aborted when mean_obi <=
    # -obi_rolling_skip_threshold; DOWN fire when mean_obi >=
    # +obi_rolling_skip_threshold. 0.05 matches the mid engine bucket in
    # core::kObiThresholdLevels.
    obi_rolling_skip_threshold: float = 0.05
    # v3.2 §5.2: directional-regime fire gate. Computes the t-statistic of
    # recent 5-min signed BTC returns from the volatility tracker. Large
    # positive t = up-trending regime; large negative t = down-trending. A
    # fire is vetoed when |t| exceeds ``directional_threshold_t`` AND the
    # trend direction opposes the signal side — hostile-regime variant that
    # captures "BTC has been drifting against this signal for hours".
    # Complements vol/chop/outcome (which are direction-agnostic).
    directional_gate_enabled: bool = True
    directional_threshold_t: float = 2.0  # |t| above which trend is "significant"
    directional_min_samples: int = 6  # min completed returns before gate activates


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_daily_loss_usd: float = 100.0
    max_consecutive_losses: int = 10
    cancel_unfilled_at_sec: int = 5
    max_position_per_window_usd: float = 75.0
    # v3.2 §5.8: post-loss cooldown. After a settled loss whose magnitude
    # exceeds ``post_loss_cooldown_loss_pct`` of bankroll, freeze trading
    # for the next ``post_loss_cooldown_windows`` window(s). Intended to
    # break tail-loss clusters where successive fires compound into the
    # same regime shift (v3.1 T3/T4 back-to-back losses). Set windows to
    # 0 or enabled=False to disable.
    post_loss_cooldown_enabled: bool = True
    post_loss_cooldown_loss_pct: float = 2.0
    post_loss_cooldown_windows: int = 1


@dataclass(frozen=True, slots=True)
class ConnectionsConfig:
    binance_ws: str = (
        "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@depth20@100ms"
    )
    rtds_ws: str = "wss://ws-live-data.polymarket.com"
    clob_market_ws: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    clob_user_ws: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    clob_rest: str = "https://clob.polymarket.com"
    gamma_rest: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137
    signature_type: int = 1  # Polymarket CLOB: 0=EOA, 1=Magic/proxy, 2=Gnosis Safe
    rtds_ping_interval_sec: int = 5
    reconnect_base_delay_sec: float = 1.0
    reconnect_max_delay_sec: float = 30.0
    binance_stale_sec: float = 15.0
    chainlink_stale_sec: float = 30.0
    clob_book_stale_sec: float = 60.0


@dataclass(frozen=True, slots=True)
class PaperConfig:
    log_every_window: bool = True
    simulated_fill_delay_sec: float = 2.5
    starting_balance_usd: float = 1000.0


@dataclass(frozen=True, slots=True)
class IpcConfig:
    # Defaults match the winlab Tailscale setup so a fresh config.yml on
    # winlab works without manual edits. Override for other machines.
    host: str = "127.0.0.1"
    port: int = 19731
    stale_signal_warning_hours: float = 6.0
    visualizer_enabled: bool = True
    visualizer_host: str = "127.0.0.1"
    visualizer_port: int = 19732


@dataclass(frozen=True, slots=True)
class SignalLifecycleConfig:
    fire_stall_windows: int = 50
    shadow_tracking_windows: int = 500
    bet_scaling_enabled: bool = True
    age_taper_start_windows: int = 300
    age_taper_end_windows: int = 500
    age_floor: float = 0.5
    min_bet_scale: float = 0.10
    sprt_activation_minutes: float = 45.0


@dataclass(frozen=True, slots=True)
class SizingConfig:
    kelly_fraction: float = 0.25
    kelly_min_bet: float = 1.00
    kelly_max_bet_pct: float = 5.0
    bankroll: float = 1000.00
    warmup_minutes: float = 30.0
    wilson_max_shrink_pct: float = 3.0  # max Wilson win rate correction (%)
    kelly_regime_cap: float = 0.12  # max regime penalty on win rate
    vol_weight: float = 1.0  # regime weight for volatility (0-1)
    chop_weight: float = 1.0  # regime weight for chop (0-1)
    outcome_weight: float = 0.8  # regime weight for outcome bias (0-1)
    feedback_min_trades: int = 10  # trades before performance feedback
    # v3.0 P6 / v3.2: double regime cap when vol, chop, OR outcome severity
    # is hostile. The regime cap alone only tips adjusted_p; hostile
    # conditions additionally scale the final bet dollars to avoid sizing
    # through a storm. Threshold tightened from 0.20 to 0.15 after v3.1
    # T3 loss at severity 0.161 bypassed the gate.
    hostile_regime_threshold: float = 0.15  # severity*weight above which hostile
    hostile_regime_multiplier: float = 0.5  # multiplier applied to bet dollars
    # v3.1: hostile-regime SKIP gate. When vol or chop severity exceeds this
    # value the fire is aborted entirely instead of being halved. outcome
    # severity does NOT participate in this metric — the halve path already
    # covers it, and skipping on outcome alone would have killed real wins
    # (v3.1 T5/T7/T9 all had outcome_sev=0.381). Must exceed
    # hostile_regime_threshold to leave a halving band intact. 0 disables.
    hostile_regime_skip_threshold: float = 0.25
    # v3.2 §5.3: soft-OR combine replaces max(vol, chop, outcome). When
    # multiple axes are moderately elevated together (v3.1 T4: vol=0.161,
    # chop=0.133, outcome=0.381 all individually below the 0.15 hostile
    # gate), max-combine reports only the worst single axis; soft-OR
    # compounds them into a meaningfully larger total_discount. Per-axis
    # contribs (feeding hostile/SKIP gates) are unchanged — only the final
    # total_discount differs.
    kelly_soft_or_combine: bool = True
    # v3.2 §5.10: adaptive max_discount cap. kelly_regime_cap (0.12) stays
    # the default for a single hot axis; when 2 or 3 axes clear
    # kelly_hot_axis_threshold the cap widens so the compounded soft-OR
    # severity can actually bite into adjusted_p. Without this scaling, the
    # 0.12 ceiling neutralises most of the soft-OR benefit.
    kelly_regime_cap_2_axes: float = 0.20
    kelly_regime_cap_3_axes: float = 0.30
    kelly_hot_axis_threshold: float = 0.33  # severity*weight counted as "hot"
    # v3.2 §5.6: Polymarket-implied probability cross-check. ``best_ask`` is
    # the market's price for YES — i.e. the implied probability we resolve
    # that side. Kelly's existing no-edge check (raw_kelly > 0) requires
    # only ``adjusted_p > ask``; this tightens to ``adjusted_p - ask >=
    # kelly_min_edge_pp / 100``, skipping fires where the market has
    # already priced in most of our signal (thin margin → expected value
    # barely covers fees/slippage). Set to 0 to disable.
    kelly_min_edge_pp: float = 2.0


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    vol_lookback_windows: int = 24
    vol_normal_pct: float = 0.10
    vol_high_pct: float = 0.30
    vol_min_samples: int = 6
    chop_lookback_windows: int = 6
    chop_normal_flips: float = 5.0
    chop_high_flips: float = 10.0
    chop_min_samples: int = 3
    outcome_lookback_windows: int = 6
    outcome_normal_agreement: float = 0.50
    outcome_high_agreement: float = 0.15
    # v3.2 §5.9: magnitude-weighted outcome agreement. Instead of treating
    # every resolved window equally, the tracker weights each historical
    # window by ``|close_delta_pct|`` so tiny-move windows (noise) don't
    # drag the regime indicator as hard as real sustained moves. Set to
    # False to fall back to count-based fraction.
    outcome_magnitude_weighted: bool = True
    # Floor weight to keep zero-magnitude windows from dropping out of the
    # average entirely (interpreted as percent, same units as delta_pct).
    outcome_min_magnitude_pct: float = 0.01
    cache_staleness_minutes: float = 30.0
    # v3.2 short-horizon EWMA volatility (RiskMetrics λ≈0.94). Feeds BTC
    # price once per vol_fast_sample_interval_s from the strategy tick
    # loop. The horizon-scaled stddev (per-sample → 5-min equivalent) is
    # max-combined with the slow close-to-close tracker so a mid-window
    # squeeze reaches the hostile-regime gate on the same tick it appears,
    # instead of waiting for the next window boundary. v3.1 T4 is the
    # canonical failure case — see docs/strategy/v3.0_v3.1_signal_analysis.md.
    vol_fast_enabled: bool = True
    vol_fast_sample_interval_s: float = 10.0
    vol_fast_decay_lambda: float = 0.94
    vol_fast_min_samples: int = 6
    # Scale the per-sample EWMA stddev up to a 5-min-equivalent horizon so
    # it is directly comparable to vol_normal_pct / vol_high_pct. Under IID
    # returns, stddev scales with sqrt(horizon / sample_interval). Default
    # 300 s / 10 s → factor sqrt(30) ≈ 5.48.
    vol_fast_horizon_s: float = 300.0
    # Tick-frequency regime refresh: during the final N seconds of the
    # window, recompute the Kelly/hostile context every strategy tick so
    # pre-fire checks see live vol/chop/outcome instead of a value frozen
    # at window-open. 0 disables. Default 240 s covers the full observe
    # window across every currently-deployed signal.
    intra_window_refresh_s: float = 240.0


@dataclass(frozen=True, slots=True)
class ErosionConfig:
    ema_alpha: float = 0.10
    cusum_tolerance: float = 0.05
    cusum_limit: float = 0.80
    cusum_decay: float = 0.95
    panic_multiplier: float = 2.20
    panic_min_duration_s: float = 3.0
    # v2.9 CUSUM sustain gate: the CUSUM exit must stay above limit for at
    # least this many seconds before firing. v2.8 had 9 CUSUM exits of which
    # 2 were hard false (market bid_up ≥ 0.96 at close) + 1 partial (Trade
    # 19 sold at profit while bid_up=0.99). A sustain gate filters single-
    # tick CUSUM blips. 0 disables the gate (legacy single-tick behavior).
    cusum_sustain_s: float = 4.0
    # v2.9 price-aware CUSUM suppression: if the Polymarket top bid on our
    # position's side is >= this threshold, the orderbook is telling us we
    # are right regardless of BTC delta wiggles, so the CUSUM exit is
    # suppressed. Trade 19 in v2.8 exited via CUSUM while bid_up was 0.99.
    # 0 disables the suppression (exit regardless of orderbook).
    cusum_suppress_top_bid: float = 0.85
    # v3.0 delta-reversal gate: require the live delta to have reversed by
    # at least this many pp vs the fire-time delta before a CUSUM exit is
    # allowed. v2.9 post-mortem showed every premature CUSUM exit had a
    # reversal depth of <= 0.14 pp while both valid exits had >= 0.16 pp.
    # 0 disables the gate.
    cusum_min_reversal_pp: float = 0.15
    # v3.1: overwhelming-breach override. When _erosion_cusum reaches
    # cusum_override_multiplier * cusum_limit the reversal-pp and top-bid
    # suppressions are bypassed because the accumulated excess is too large
    # to attribute to noise. v3.0 01:47 loss had cusum=1.608 (2.01x limit)
    # blocked by the reversal gate; an override would have cut ~$16 off the
    # full $25.18 loss. 0 disables the override (always honor suppressions).
    # v3.2: raised 2.0 -> 3.8. v3.1 trade #10 was a false exit at cusum=2.87
    # (3.58x limit) that turned a correct UP bet into a -$6.31 loss (would
    # have been +$10.76 if held). v3.1 trades #3 and #4 were correct
    # override exits at cusum 7.08 and 4.07 (8.85x and 5.09x); threshold
    # 3.8 (=3.04 cusum) blocks #10 while preserving both true positives.
    cusum_override_multiplier: float = 3.8


@dataclass(frozen=True, slots=True)
class Config:
    mode: ModeConfig = field(default_factory=ModeConfig)
    rules_strategy: RulesStrategyConfig = field(default_factory=RulesStrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    connections: ConnectionsConfig = field(default_factory=ConnectionsConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    ipc: IpcConfig = field(default_factory=IpcConfig)
    signal_lifecycle: SignalLifecycleConfig = field(default_factory=SignalLifecycleConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)

    @property
    def is_paper(self) -> bool:
        return self.mode.trading == "paper"

    def data_paths(self, repo_root: Path) -> DataPaths:
        """Compute all mode-dependent data paths from a single root."""
        mode = self.mode.trading  # "paper" or "live"
        base = repo_root / "data" / mode
        return DataPaths(
            bankroll=base / "bankroll.json",
            journal=base / "journal.jsonl",
            state=base / "state.json",
            results=base / "results",
            logs=repo_root / self.mode.log_dir,
            vol_cache=base / "vol_cache.json",
            chop_cache=base / "chop_cache.json",
            outcome_cache=base / "outcome_cache.json",
            fast_vol_cache=base / "fast_vol_cache.json",
        )


@dataclass(frozen=True, slots=True)
class DataPaths:
    """All mode-dependent file paths, computed from Config."""

    bankroll: Path
    journal: Path
    state: Path
    results: Path
    logs: Path
    vol_cache: Path = Path("data/paper/vol_cache.json")
    chop_cache: Path = Path("data/paper/chop_cache.json")
    outcome_cache: Path = Path("data/paper/outcome_cache.json")
    fast_vol_cache: Path = Path("data/paper/fast_vol_cache.json")

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        self.bankroll.parent.mkdir(parents=True, exist_ok=True)
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)


# Section name -> dataclass class mapping for auto-update
_SECTION_MAP: dict[str, type] = {
    "mode": ModeConfig,
    "rules_strategy": RulesStrategyConfig,
    "risk": RiskConfig,
    "connections": ConnectionsConfig,
    "paper": PaperConfig,
    "ipc": IpcConfig,
    "signal_lifecycle": SignalLifecycleConfig,
    "sizing": SizingConfig,
    "regime": RegimeConfig,
    "erosion": ErosionConfig,
}


def _auto_update_config(raw: dict[str, Any], config_path: Path) -> None:
    """Sync config.yml schema with current dataclass definitions.

    Adds missing fields with their defaults, removes fields that no longer
    exist in the dataclass. Preserves all user-set values. Writes back to
    disk only if changes are needed.
    """
    changes: list[str] = []

    for section_name, dc_cls in _SECTION_MAP.items():
        dc_fields = {f.name for f in fields(dc_cls)}
        dc_defaults = dc_cls()
        section = raw.get(section_name)

        if section is None:
            # Entire section missing - add it with all defaults
            raw[section_name] = {f.name: getattr(dc_defaults, f.name) for f in fields(dc_cls)}
            changes.append(f"added missing section [{section_name}]")
            continue

        yaml_keys = set(section.keys())

        # Add new fields (in dataclass but not in YAML)
        added = dc_fields - yaml_keys
        for key in sorted(added):
            section[key] = getattr(dc_defaults, key)
            changes.append(f"[{section_name}] added '{key}' = {section[key]!r}")

        # Remove old fields (in YAML but not in dataclass)
        removed = yaml_keys - dc_fields
        for key in sorted(removed):
            del section[key]
            changes.append(f"[{section_name}] removed obsolete '{key}'")

    # Also detect top-level sections in YAML that don't map to any dataclass
    known_sections = set(_SECTION_MAP.keys())
    for top_key in list(raw.keys()):
        if top_key not in known_sections:
            del raw[top_key]
            changes.append(f"removed unknown top-level section [{top_key}]")

    if not changes:
        return

    # Write updated config back to disk
    try:
        # Preserve section order matching _SECTION_MAP
        ordered: dict[str, Any] = {}
        for section_name in _SECTION_MAP:
            if section_name in raw:
                ordered[section_name] = raw[section_name]

        output = yaml.dump(
            ordered,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
        config_path.write_text(output, encoding="utf-8")
        _log.info("config.yml auto-updated with %d change(s):", len(changes))
        for c in changes:
            _log.info("  config: %s", c)
    except OSError as exc:
        _log.warning("failed to write updated config.yml: %s", exc)


def _parse_section(raw: dict[str, Any], key: str, cls: type) -> Any:
    """Parse a YAML section into a dataclass, filtering unknown keys."""
    section = raw.get(key, {})
    if not section:
        return cls()
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in section.items() if k in known})


def load_config() -> Config:
    """Load config.yml and return validated Config."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG)
        print(f"config.yml created with defaults at {CONFIG_PATH} -- review it before running.")
        sys.exit(0)

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    if not raw:
        print("config.yml is empty -- delete it and restart to regenerate defaults.")
        sys.exit(1)

    # Auto-update config.yml schema (add new fields, remove obsolete ones)
    _auto_update_config(raw, CONFIG_PATH)

    cfg = Config(
        mode=_parse_section(raw, "mode", ModeConfig),
        rules_strategy=_parse_section(raw, "rules_strategy", RulesStrategyConfig),
        risk=_parse_section(raw, "risk", RiskConfig),
        connections=_parse_section(raw, "connections", ConnectionsConfig),
        paper=_parse_section(raw, "paper", PaperConfig),
        ipc=_parse_section(raw, "ipc", IpcConfig),
        signal_lifecycle=_parse_section(raw, "signal_lifecycle", SignalLifecycleConfig),
        sizing=_parse_section(raw, "sizing", SizingConfig),
        regime=_parse_section(raw, "regime", RegimeConfig),
        erosion=_parse_section(raw, "erosion", ErosionConfig),
    )

    _validate(cfg)

    return cfg


def _validate(cfg: Config) -> None:
    if cfg.mode.trading not in ("paper", "live"):
        print(f"Invalid mode.trading: {cfg.mode.trading!r} -- must be 'paper' or 'live'")
        sys.exit(1)

    if cfg.mode.log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        print(f"Invalid log_level: {cfg.mode.log_level!r}")
        sys.exit(1)
