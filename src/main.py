"""Entry point for the Polymarket BTC 5-minute trading bot."""

from __future__ import annotations

import asyncio
import gc
import json as _json
import logging
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add submodule root to path for shared imports
_SUBMODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SUBMODULE_ROOT))

from dotenv import load_dotenv

load_dotenv(_SUBMODULE_ROOT / ".env", override=True)

import aiohttp

from config import Config, DataPaths, load_config
from execution.base import OrderExecutor
from execution.order_manager import OrderManager
from execution.paper_trading import PaperOrderManager
from market_data.binance_ws import handle_binance
from market_data.clob_market_ws import handle_clob_market
from market_data.clob_user_ws import handle_clob_user
from market_data.latency_tracker import LatencyTracker
from market_data.rtds_ws import handle_rtds
from market_data.state import MarketState
from risk.fee_tracker import FeeTracker
from risk.position_tracker import PositionTracker
from risk.registry import RiskRegistry
from shared.decay_detector import DecayDetector
from shared.discord import (
    send_latency_report,
    send_presignal_warmup,
    send_presignal_warmup_done,
    send_sprt_decay_alert,
)
from shared.state_publisher import StatePublisher
from shared.trade_journal import RecentFireMailbox, TradeJournal
from strategy.kelly import KELLY_OUTCOME_WINDOW_SIZE, BankrollTracker
from strategy.momentum_signal import MomentumSignalConfig, MomentumSignalStrategy
from strategy.monitors import (
    BetScaleSqueezeTracker,
    ConsecutiveLossTracker,
    SessionAccumulator,
    SkipStreakTracker,
)
from strategy.regime import RegimeManager
from strategy.resolution import ResolutionManager
from strategy.signal import compute_signal
from strategy.signal_lifecycle import SignalLifecycle
from strategy.signal_loader import SignalValidationError, validate_momentum_signal
from strategy.window_handler import WindowEventHandler
from strategy.window_tracker import WindowTracker
from utils.log import setup_logging
from utils.reconnect import FeedHealthMonitor, ws_connect_forever

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from py_clob_client.client import (  # type: ignore[import-untyped]  # no stubs available
        ClobClient,
    )

    from shared.chop_detector import ChopDetector
    from shared.outcome_tracker import OutcomeTracker
    from shared.volatility_tracker import VolatilityTracker

log = logging.getLogger(__name__)

STRATEGY_TICK_INTERVAL = 0.25


# ---------------------------------------------------------------------------
# Pending signal manager — handles IPC signal transition
# ---------------------------------------------------------------------------


class PendingSignalManager:
    """Thread-safe container for a signal received via IPC, awaiting transition."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, Any] | None = None
        self._summary_file: str = ""
        self._last_signal_time: float = 0.0

    def set_pending(self, signal_data: dict[str, Any], summary_file: str) -> None:
        with self._lock:
            self._pending = signal_data
            self._summary_file = summary_file
            self._last_signal_time = time.time()

    def take_pending(self) -> tuple[dict[str, Any], str] | None:
        with self._lock:
            if self._pending is None:
                return None
            data = self._pending
            summary = self._summary_file
            self._pending = None
            self._summary_file = ""
            return data, summary

    @property
    def last_signal_time(self) -> float:
        with self._lock:
            return self._last_signal_time


# ---------------------------------------------------------------------------
# Pure helper functions (no side effects — safe to extract and test)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Strategy loading — PolySignalEngine signal files only
# ---------------------------------------------------------------------------


def _signal_cfg_to_dict(sc: MomentumSignalConfig) -> dict[str, Any]:
    """Serialize a MomentumSignalConfig to a dict for logging."""
    return {
        "rank": sc.rank,
        "side": sc.side.value,
        "observeFromS": sc.observe_from_s,
        "observeToS": sc.observe_to_s,
        "minDeltaPct": sc.min_delta_pct,
        "maxVariancePct": sc.max_variance_pct,
        "trainWinRatePct": sc.train_win_rate_pct,
        "oosWinRatePct": sc.oos_win_rate_pct,
        "oosBhAdjustedPValue": sc.bh_adjusted_p_value,
        "oosMatches": sc.oos_matches,
        "avgEntryPrice": sc.avg_entry_price or 0.0,
        "evPerTrade": sc.ev_per_trade or 0.0,
        "smartScore": sc.smart_score,
        "wfFoldsAppeared": sc.wf_folds_appeared,
        "wfTotalTestFolds": sc.wf_total_test_folds,
        "wfFoldIndices": sc.wf_fold_indices,
    }


def _build_strategy(
    cfg: Config,
    state: MarketState,
    signal_path: str,
) -> MomentumSignalStrategy:
    """Load a PolySignalEngine signal file and return the strategy."""
    path = Path(signal_path)

    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.error("signal file not found: %s", path)
        sys.exit(1)
    except (OSError, _json.JSONDecodeError, ValueError) as exc:
        log.error("failed to read %s: %s", path, exc)
        sys.exit(1)

    min_oos_wr = cfg.rules_strategy.min_win_rate * 100.0
    try:
        result = validate_momentum_signal(data, min_oos_win_rate_pct=min_oos_wr)
    except SignalValidationError as exc:
        log.error("signal rejected — %s", exc)
        sys.exit(1)

    for w in result.warnings:
        log.warning("signal warning [%s]: %s", w.field, w.message)

    sc = result.signal
    log.info(
        "momentum signal accepted: rank=%d side=%s "
        "observe=[%.0f->%.0f]s min_delta=%.2f%% max_var=%.3f%% "
        "oos_wr=%.1f%% (%d matches) conservative_p=%.1f%% bh_p=%.3g "
        "avg_entry=%.2f ev=%.4f",
        sc.rank,
        sc.side.value,
        sc.observe_from_s,
        sc.observe_to_s,
        sc.min_delta_pct,
        sc.max_variance_pct,
        sc.oos_win_rate_pct,
        sc.oos_matches,
        sc.conservative_p(cfg.sizing.wilson_max_shrink_pct) * 100,
        sc.bh_adjusted_p_value,
        sc.avg_entry_price or 0.0,
        sc.ev_per_trade or 0.0,
    )
    log.info("SIGNAL_SWAP_ACTIVE: %s", _json.dumps(_signal_cfg_to_dict(sc), separators=(",", ":")))
    return MomentumSignalStrategy(cfg.rules_strategy, state, sc)


def _build_strategy_from_dict(
    cfg: Config,
    state: MarketState,
    data: dict[str, Any],
) -> MomentumSignalStrategy | None:
    """Build a strategy from a raw signal dict received via IPC. Returns None on failure."""
    min_oos_wr = cfg.rules_strategy.min_win_rate * 100.0
    try:
        result = validate_momentum_signal(data, min_oos_win_rate_pct=min_oos_wr)
    except SignalValidationError as exc:
        log.error("IPC signal rejected: %s", exc)
        return None

    for w in result.warnings:
        log.warning("IPC signal warning [%s]: %s", w.field, w.message)

    return MomentumSignalStrategy(cfg.rules_strategy, state, result.signal)


async def _wait_for_first_signal(
    cfg: Config,
    state: MarketState,
    pending_signal_mgr: PendingSignalManager,
    vol_tracker: VolatilityTracker | None = None,
    chop_detector: ChopDetector | None = None,
    outcome_tracker: OutcomeTracker | None = None,
    paths: DataPaths | None = None,
) -> MomentumSignalStrategy:
    """Block until the orchestrator delivers a valid signal via IPC.

    If vol_tracker / chop_detector are provided, collects baseline data
    while waiting so warmup is already progressing when the signal arrives.
    Uses only local state for window tracking — never mutates MarketState
    fields that the strategy loop manages (window_open_price, snapshots, etc.).
    """
    _collect = vol_tracker is not None and chop_detector is not None
    _last_wts = 0
    _local_open = 0.0  # local-only open price for direction calc
    _warmup_complete_notified = False

    while True:
        pending = pending_signal_mgr.take_pending()
        if pending:
            data, _summary_file = pending
            strategy = _build_strategy_from_dict(cfg, state, data)
            if strategy:
                sc = strategy.signal_cfg
                log.info(
                    "first signal received from orchestrator: rank=%d side=%s signal_id=%s",
                    sc.rank,
                    sc.side.value,
                    sc.signal_id,
                )
                log.info(
                    "SIGNAL_SWAP_ACTIVE: %s",
                    _json.dumps(_signal_cfg_to_dict(sc), separators=(",", ":")),
                )
                return strategy
            log.warning("orchestrator signal rejected — continuing to wait...")

        # Lightweight vol/chop collection while waiting for the first signal.
        # Window boundary detection uses simple arithmetic (same as WindowTracker).
        if _collect and state.btc_chainlink > 0:
            now_ts = int(time.time())
            wts = now_ts - (now_ts % 300)
            if wts != _last_wts:
                if (
                    _last_wts > 0
                    and vol_tracker is not None
                    and chop_detector is not None
                    and paths is not None
                ):
                    vol_tracker.record_close(state.btc_chainlink)
                    vol_tracker.save_cache(paths.vol_cache)
                    chop_detector.finalize_window()
                    chop_detector.save_cache(paths.chop_cache)
                    # Record outcome direction for the completed warmup window
                    if outcome_tracker is not None and _local_open > 0:
                        _warmup_outcome = "up" if state.btc_chainlink >= _local_open else "down"
                        outcome_tracker.record_outcome(_warmup_outcome)
                        outcome_tracker.save_cache(paths.outcome_cache)
                    _credited = min(vol_tracker.n_returns, chop_detector.n_windows) * 5.0
                    _warmup_done = _credited >= cfg.sizing.warmup_minutes
                    if _warmup_done:
                        log.info(
                            "warmup complete, waiting for signal from orchestrator "
                            "(vol=%d prices, chop=%d windows, %.0f min credited)",
                            vol_tracker.n_returns,
                            chop_detector.n_windows,
                            _credited,
                        )
                        if not _warmup_complete_notified:
                            send_presignal_warmup_done(
                                vol_prices=vol_tracker.n_returns,
                                chop_windows=chop_detector.n_windows,
                                credited_min=_credited,
                            )
                            _warmup_complete_notified = True
                    else:
                        log.info(
                            "pre-signal warmup: vol=%d prices chop=%d windows (%.0f min credited)",
                            vol_tracker.n_returns,
                            chop_detector.n_windows,
                            _credited,
                        )
                        send_presignal_warmup(
                            vol_prices=vol_tracker.n_returns,
                            chop_windows=chop_detector.n_windows,
                            credited_min=_credited,
                            warmup_minutes=cfg.sizing.warmup_minutes,
                        )
                _last_wts = wts
                _local_open = state.btc_chainlink

            if _local_open > 0 and chop_detector is not None:
                delta = (state.btc_chainlink - _local_open) / _local_open * 100
                chop_detector.tick("up" if delta >= 0 else "down", delta)

        await asyncio.sleep(STRATEGY_TICK_INTERVAL)


# ---------------------------------------------------------------------------
# CLOB client (live mode only)
# ---------------------------------------------------------------------------


def _build_clob_client(cfg: Config) -> ClobClient:
    from py_clob_client.client import ClobClient

    from shared.keystore import get_secret

    private_key = get_secret("private_key", env_var="POLY_PRIVATE_KEY") or ""
    funder_address = get_secret("funder_address", env_var="POLY_FUNDER_ADDRESS") or ""

    client = ClobClient(
        host=cfg.connections.clob_rest,
        key=private_key,
        chain_id=cfg.connections.chain_id,
        signature_type=cfg.connections.signature_type,
        funder=funder_address,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pre-flight checks — fast, synchronous, no network
# ---------------------------------------------------------------------------

_RED = "\033[91m"
_GREEN = "\033[92m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _preflight(
    cfg: Config,
    paths: DataPaths,
    signal_path: str | None,
    standalone: bool,
) -> bool:
    """Run pre-flight checks before any connections are made.

    Returns True if all checks pass. Prints red error messages for failures.
    """
    errors: list[str] = []

    # 1. Config file exists
    from config import CONFIG_PATH

    if not CONFIG_PATH.exists():
        errors.append(f"config.yml not found at {CONFIG_PATH}")

    # 2. Signal file (standalone requires it)
    if standalone and signal_path:
        p = Path(signal_path)
        if not p.exists():
            errors.append(f"signal file not found: {p}")
        elif p.stat().st_size == 0:
            errors.append(f"signal file is empty: {p}")

    # 3. Live mode credentials
    if cfg.mode.trading == "live":
        from shared.keystore import get_secret

        required_creds = {
            "private_key": "POLY_PRIVATE_KEY",
            "api_key": "POLY_API_KEY",
            "api_secret": "POLY_API_SECRET",
            "api_passphrase": "POLY_API_PASSPHRASE",
            "funder_address": "POLY_FUNDER_ADDRESS",
        }
        missing = [
            name for name, env in required_creds.items() if not get_secret(name, env_var=env)
        ]
        if missing:
            errors.append(f"live mode missing credentials: {', '.join(missing)}")

    # 4. IPC HMAC key (non-standalone)
    if not standalone:
        from shared.keystore import get_hmac_key

        try:
            get_hmac_key()
        except ValueError as exc:
            errors.append(str(exc))

    # 5. Data directory write permissions
    test_dirs = [
        ("results", paths.results),
        ("logs", paths.logs),
        ("state", paths.state.parent),
        ("bankroll", paths.bankroll.parent),
        ("journal", paths.journal.parent),
    ]
    for label, dir_path in test_dirs:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                errors.append(f"cannot create {label} directory {dir_path}: {exc}")
                continue
        # Test write permission
        test_file = dir_path / ".preflight_write_test"
        try:
            test_file.write_text("ok")
            test_file.unlink()
        except OSError as exc:
            errors.append(f"no write permission for {label} directory {dir_path}: {exc}")

    # 6. IPC port availability (non-standalone)
    if not standalone:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((cfg.ipc.host, cfg.ipc.port))
        except OSError:
            errors.append(
                f"IPC port {cfg.ipc.host}:{cfg.ipc.port} is already in use — "
                "another bot instance may be running"
            )
        finally:
            sock.close()

    # 7. Config value sanity
    if cfg.sizing.kelly_fraction <= 0 or cfg.sizing.kelly_fraction > 1:
        errors.append(f"kelly_fraction={cfg.sizing.kelly_fraction} — must be between 0 and 1")
    if cfg.sizing.kelly_max_bet_pct <= 0 or cfg.sizing.kelly_max_bet_pct > 100:
        errors.append(
            f"kelly_max_bet_pct={cfg.sizing.kelly_max_bet_pct} — must be between 0 and 100"
        )
    if cfg.risk.max_daily_loss_usd <= 0:
        errors.append(f"max_daily_loss_usd={cfg.risk.max_daily_loss_usd} — must be positive")
    if cfg.risk.max_consecutive_losses <= 0:
        errors.append(
            f"max_consecutive_losses={cfg.risk.max_consecutive_losses} — must be positive"
        )

    # Print results
    if errors:
        print(f"\n{_RED}{_BOLD}PRE-FLIGHT FAILED{_RESET}\n")
        for err in errors:
            print(f"  {_RED}✗{_RESET} {err}")
        print()
        return False

    print(f"\n{_GREEN}pre-flight checks passed{_RESET}\n")
    return True


async def _validate_startup(
    cfg: Config,
    session: aiohttp.ClientSession,
    clob: object | None = None,
) -> bool:
    """Validate external dependencies before entering main loop.

    If *clob* is provided (live mode), skips building a throwaway CLOB
    client — the caller already built one.
    """
    ok = True

    try:
        async with session.get(
            f"{cfg.connections.gamma_rest}/markets?slug=btc-updown-5m-0",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status not in (200, 404):
                log.error("gamma API returned unexpected status %d", resp.status)
                ok = False
            else:
                log.info("gamma API: reachable")
    except (aiohttp.ClientError, TimeoutError, OSError) as exc:
        log.error("gamma API unreachable: %s", exc)
        ok = False

    if cfg.mode.trading == "live" and clob is None:
        try:
            _build_clob_client(cfg)
            log.info("CLOB auth: ok")
        except (OSError, ValueError, KeyError) as exc:
            log.error("CLOB auth failed: %s", exc)
            ok = False

    return ok


# ---------------------------------------------------------------------------
# WebSocket feeds
# ---------------------------------------------------------------------------


async def _run_price_feeds(
    cfg: Config,
    state: MarketState,
    health_monitor: FeedHealthMonitor,
    latency: LatencyTracker | None = None,
) -> list[asyncio.Task[None]]:
    """Start Binance + RTDS (Chainlink) feeds — the two price feeds needed
    for pre-signal warmup.  CLOB feeds are deferred until token IDs are known
    (see ``_run_clob_feeds``)."""
    tasks: list[asyncio.Task[None]] = []

    tasks.append(
        asyncio.create_task(
            ws_connect_forever(
                cfg.connections.binance_ws,
                lambda ws: handle_binance(ws, state, latency=latency),
                name="binance",
                base_delay=cfg.connections.reconnect_base_delay_sec,
                max_delay=cfg.connections.reconnect_max_delay_sec,
                health=health_monitor.register("binance"),
            ),
            name="ws-binance",
        )
    )

    tasks.append(
        asyncio.create_task(
            ws_connect_forever(
                cfg.connections.rtds_ws,
                lambda ws: handle_rtds(
                    ws, state, cfg.connections.rtds_ping_interval_sec, latency=latency
                ),
                name="rtds",
                base_delay=cfg.connections.reconnect_base_delay_sec,
                max_delay=cfg.connections.reconnect_max_delay_sec,
                health=health_monitor.register("rtds"),
            ),
            name="ws-rtds",
        )
    )

    return tasks


async def _run_clob_feeds(
    cfg: Config,
    state: MarketState,
    health_monitor: FeedHealthMonitor,
    api_creds: dict[str, str] | None = None,
    latency: LatencyTracker | None = None,
) -> list[asyncio.Task[None]]:
    """Start CLOB WebSocket feeds.  Call only after token IDs are available
    (i.e. after the first signal is received), otherwise the Polymarket CLOB
    server drops idle connections every ~10 seconds."""
    tasks: list[asyncio.Task[None]] = []

    tasks.append(
        asyncio.create_task(
            ws_connect_forever(
                cfg.connections.clob_market_ws,
                lambda ws: handle_clob_market(ws, state, latency=latency),
                name="clob-market",
                base_delay=cfg.connections.reconnect_base_delay_sec,
                max_delay=cfg.connections.reconnect_max_delay_sec,
                extra_headers={"Origin": "https://clob.polymarket.com"},
                health=health_monitor.register("clob-market"),
            ),
            name="ws-clob-market",
        )
    )

    if not cfg.is_paper:
        # Use derived API creds from the CLOB client (not raw env vars)
        if api_creds is None:
            from shared.keystore import get_secret

            api_creds = {
                "apiKey": get_secret("api_key", env_var="POLY_API_KEY") or "",
                "secret": get_secret("api_secret", env_var="POLY_API_SECRET") or "",
                "passphrase": get_secret("api_passphrase", env_var="POLY_API_PASSPHRASE") or "",
            }
        creds = api_creds
        tasks.append(
            asyncio.create_task(
                ws_connect_forever(
                    cfg.connections.clob_user_ws,
                    lambda ws: handle_clob_user(ws, state, creds, latency=latency),
                    name="clob-user",
                    base_delay=cfg.connections.reconnect_base_delay_sec,
                    max_delay=cfg.connections.reconnect_max_delay_sec,
                    extra_headers={"Origin": "https://clob.polymarket.com"},
                    health=health_monitor.register("clob-user"),
                ),
                name="ws-clob-user",
            )
        )

    return tasks


async def _wait_for_prices(state: MarketState, timeout: float = 30.0) -> bool:  # noqa: ASYNC109  # timeout used for deadline, not asyncio.timeout
    """Wait until both price feeds AND the Binance depth (OBI) stream are live."""
    deadline = time.time() + timeout
    prices_logged = False
    while time.time() < deadline:
        has_prices = state.btc_binance > 0 and state.btc_chainlink > 0
        has_obi = state.binance_obi_ts > 0

        if has_prices and not prices_logged:
            log.info(
                "price feeds live: binance=%.2f chainlink=%.2f — waiting for depth stream",
                state.btc_binance,
                state.btc_chainlink,
            )
            prices_logged = True

        if has_prices and has_obi:
            log.info(
                "all feeds live: binance=%.2f chainlink=%.2f obi=%.4f",
                state.btc_binance,
                state.btc_chainlink,
                state.binance_obi,
            )
            return True

        await asyncio.sleep(0.5)

    log.error(
        "timed out waiting for feeds — binance=%.2f chainlink=%.2f obi=%.4f",
        state.btc_binance,
        state.btc_chainlink,
        state.binance_obi,
    )
    return False


# ---------------------------------------------------------------------------
# Strategy loop
# ---------------------------------------------------------------------------


async def _strategy_loop(
    cfg: Config,
    state: MarketState,
    window_tracker: WindowTracker,
    rules_strategy: MomentumSignalStrategy,
    order_mgr: OrderManager | PaperOrderManager,
    position_tracker: PositionTracker,
    risk: RiskRegistry,
    fee_tracker: FeeTracker,
    session: aiohttp.ClientSession,
    paths: DataPaths,
    pending_signal_mgr: PendingSignalManager | None = None,
    latency_tracker: LatencyTracker | None = None,
    health_monitor: FeedHealthMonitor | None = None,
    vol_tracker: VolatilityTracker | None = None,
    chop_detector: ChopDetector | None = None,
    outcome_tracker: OutcomeTracker | None = None,
    state_publisher: StatePublisher | None = None,
    fire_mailbox: RecentFireMailbox | None = None,
) -> None:
    last_window_ts = 0
    gc_disabled = False
    snapshot_taken = False
    last_status_log = 0.0
    last_stale_warn = 0.0
    _stale_warn_ts = 0.0  # throttle for feed staleness warnings
    _health_warn_ts = 0.0  # throttle for WS health monitor warnings
    STATUS_LOG_INTERVAL = 30.0  # noqa: N806  # constant defined in function scope
    STALE_WARN_INTERVAL = 3600.0  # noqa: N806  # constant defined in function scope
    last_utc_date = datetime.now(UTC).date()

    # Signal lifecycle tracking. The optional fire_mailbox is a shared
    # in-memory ring buffer: when the journal records a resolved fire, it
    # also pushes a RecentFire into the mailbox so the IPC status_query
    # handler can answer without rereading the JSONL file from disk.
    journal = TradeJournal(paths.journal, fire_mailbox=fire_mailbox)
    lifecycle = SignalLifecycle()

    # SPRT decay detector
    _init_sc = rules_strategy.signal_cfg
    _p_alive = _init_sc.conservative_p(cfg.sizing.wilson_max_shrink_pct)
    # SPRT H0 (dead) threshold: avg_entry_price + 0.02
    # See docs/reference/system_pipeline.md (SPRT section) for rationale.
    _p_dead = (_init_sc.avg_entry_price or 0.85) + 0.02
    if _p_dead >= _p_alive:
        _p_dead = _p_alive - 0.05
    decay_detector = DecayDetector(
        signal_id=_init_sc.signal_id,
        p_alive=_p_alive,
        p_dead=_p_dead,
    )

    # Regime trackers — use pre-warmed instances if provided, otherwise create fresh
    if vol_tracker is None or chop_detector is None or outcome_tracker is None:
        _regime = RegimeManager.create(cfg.regime, paths)
        vol_tracker = vol_tracker or _regime.vol_tracker
        chop_detector = chop_detector or _regime.chop_detector
        outcome_tracker = outcome_tracker or _regime.outcome_tracker

    # Kelly Criterion bankroll tracking and recent outcomes
    bankroll_tracker = BankrollTracker(
        initial_bankroll=cfg.sizing.bankroll,
        path=paths.bankroll,
    )
    recent_outcomes: deque[int] = deque(maxlen=KELLY_OUTCOME_WINDOW_SIZE)
    # Parallel deque of in-flight optimistic live outcomes (from end-of-window
    # snapshots, before Gamma API confirms). Shared between ResolutionManager
    # (appends on fire, drains on resolve) and WindowEventHandler (merges into
    # the Kelly feedback view). Always empty in paper mode.
    optimistic_outcomes: deque[int] = deque(maxlen=KELLY_OUTCOME_WINDOW_SIZE)

    # Warmup credit — backdate bot_start_time using pre-existing tracker data
    _bot_start_time = time.time()
    _regime_for_credit = RegimeManager(vol_tracker, chop_detector, outcome_tracker)
    _bot_start_time -= _regime_for_credit.compute_warmup_credit(
        cfg.sizing.warmup_minutes,
    )
    # Monitoring trackers (extracted state machines)
    skip_tracker = SkipStreakTracker(alert_hours=12.0)
    loss_tracker = ConsecutiveLossTracker(warn_at=5)
    squeeze_tracker = BetScaleSqueezeTracker(threshold_pct=30.0, alert_windows=12)
    session_stats = SessionAccumulator()

    # Resolution manager — single pipeline for all resolution paths
    # Live-only balance refresher: pulled by ResolutionManager after every
    # confirmed live resolve to keep Kelly bankroll aligned with on-chain USDC.
    _balance_refresher: Callable[[], Awaitable[float | None]] | None = None
    if isinstance(order_mgr, OrderManager) and order_mgr.mode == "live":
        _live_mgr = order_mgr
        _balance_refresher = _live_mgr.refresh_balance

    resolution_mgr = ResolutionManager(
        window_tracker=window_tracker,
        state=state,
        fee_tracker=fee_tracker,
        bankroll_tracker=bankroll_tracker,
        position_tracker=position_tracker,
        journal=journal,
        decay_detector=decay_detector,
        recent_outcomes=recent_outcomes,
        results_dir=paths.results,
        session_stats=session_stats,
        loss_tracker=loss_tracker,
        cfg=cfg,
        paths=paths,
        optimistic_outcomes=optimistic_outcomes,
        balance_refresher=_balance_refresher,
    )

    # Window event handler — consolidates the ~570-line window transition block
    window_handler = WindowEventHandler(
        cfg=cfg,
        state=state,
        window_tracker=window_tracker,
        resolution_mgr=resolution_mgr,
        lifecycle=lifecycle,
        position_tracker=position_tracker,
        fee_tracker=fee_tracker,
        bankroll_tracker=bankroll_tracker,
        recent_outcomes=recent_outcomes,
        optimistic_outcomes=optimistic_outcomes,
        session_stats=session_stats,
        skip_tracker=skip_tracker,
        loss_tracker=loss_tracker,
        squeeze_tracker=squeeze_tracker,
        journal=journal,
        vol_tracker=vol_tracker,
        chop_detector=chop_detector,
        outcome_tracker=outcome_tracker,
        paths=paths,
        session=session,
        pending_signal_mgr=pending_signal_mgr,
        bot_start_time=_bot_start_time,
        build_strategy_fn=_build_strategy_from_dict,
        signal_cfg_to_dict_fn=_signal_cfg_to_dict,
    )

    # Latency report timer (every 60 minutes)
    _LATENCY_REPORT_INTERVAL = 3600.0  # noqa: N806  # constant defined in function scope
    _last_latency_report = time.time()

    # v3.0: reconcile Kelly bankroll with the authoritative balance BEFORE the
    # strategy loop starts. Paper mode's authoritative source is
    # PaperOrderManager.balance (resumed from state.json). Live mode's
    # authoritative source is the on-chain USDC balance from the CLOB API. In
    # both modes, the cached bankroll.json can drift from the authoritative
    # source across session boundaries — v2.9 paper showed a $28 startup drift
    # that only got healed after the first window opened. Reconciling here,
    # before any fire decisions, eliminates the window where Kelly sizes off a
    # stale base.
    _startup_authoritative: float | None = None
    _startup_source: str | None = None
    if order_mgr.mode == "live" and isinstance(order_mgr, OrderManager):
        _api_bal = await order_mgr.refresh_balance()
        if _api_bal is not None and _api_bal > 0:
            _startup_authoritative = _api_bal
            _startup_source = "onchain"
        else:
            log.warning(
                "could not fetch on-chain balance at startup — using cached bankroll $%.2f",
                bankroll_tracker.bankroll,
            )
    elif isinstance(order_mgr, PaperOrderManager):
        _startup_authoritative = order_mgr.balance
        _startup_source = "paper"

    if _startup_authoritative is not None and _startup_source is not None:
        _local_bal = bankroll_tracker.bankroll
        _drift = _startup_authoritative - _local_bal
        if abs(_drift) > 1.0:
            log.warning(
                "BANKROLL_STARTUP_DRIFT kelly_br=$%.2f %s_br=$%.2f drift=%+.2f "
                "— stale bankroll.json, reconciling to %s balance",
                _local_bal,
                _startup_source,
                _startup_authoritative,
                _drift,
                _startup_source,
            )
        else:
            log.info(
                "bankroll startup check: kelly_br=$%.2f %s_br=$%.2f drift=%+.2f OK",
                _local_bal,
                _startup_source,
                _startup_authoritative,
                _drift,
            )
        bankroll_tracker.sync_from_api(_startup_authoritative)
        window_handler._last_bankroll_sync = time.time()

    while True:
        try:
            time_remaining = window_tracker.time_remaining()
            state.time_remaining = time_remaining

            now = time.time()
            current_utc_date = datetime.now(UTC).date()
            if current_utc_date != last_utc_date:
                log.info("midnight UTC — sending session summary, resetting risk counters")
                _balance = (
                    order_mgr.balance
                    if isinstance(order_mgr, PaperOrderManager)
                    else bankroll_tracker.bankroll
                )
                session_stats.send_summary(
                    mode=order_mgr.mode,
                    balance=_balance,
                )
                position_tracker.reset_daily()
                risk.reset_halt()
                last_utc_date = current_utc_date

            # Stale signal warning
            if pending_signal_mgr and pending_signal_mgr.last_signal_time > 0:
                hours_since = (now - pending_signal_mgr.last_signal_time) / 3600
                if (
                    hours_since >= cfg.ipc.stale_signal_warning_hours
                    and now - last_stale_warn >= STALE_WARN_INTERVAL
                ):
                    log.warning(
                        "no signal update from orchestrator in %.1f hours — continuing with current signal",
                        hours_since,
                    )
                    last_stale_warn = now

            if now - last_status_log >= STATUS_LOG_INTERVAL and last_window_ts > 0:
                signal = compute_signal(state)
                log.info(
                    "STATUS t=%.0fs delta=%.4f%% dir=%s agree=%s "
                    "chain=%.2f bin=%.2f open=%.2f obi=%.4f "
                    "ask_up=%.2f ask_dn=%.2f bid_up=%.2f bid_dn=%.2f halted=%s",
                    time_remaining,
                    signal.delta_pct,
                    signal.direction.value,
                    signal.feeds_agree,
                    state.btc_chainlink,
                    state.btc_binance,
                    state.window_open_price,
                    state.binance_obi,
                    state.best_ask_up,
                    state.best_ask_down,
                    state.best_bid_up,
                    state.best_bid_down,
                    risk.halted,
                )
                last_status_log = now

            # Latency report — every 60 minutes, fire-and-forget
            if (
                latency_tracker is not None
                and now - _last_latency_report >= _LATENCY_REPORT_INTERVAL
            ):
                _source = order_mgr.mode
                stats = latency_tracker.all_stats()
                send_latency_report(
                    mode=_source,
                    feed_stats=[
                        {
                            "name": s.name,
                            "samples": s.samples,
                            "min_ms": s.min_ms,
                            "max_ms": s.max_ms,
                            "median_ms": s.median_ms,
                            "p95_ms": s.p95_ms,
                            "mean_ms": s.mean_ms,
                            "jitter_ms": s.jitter_ms,
                            "last_message_ago_s": s.last_message_ago_s,
                        }
                        for s in stats
                    ],
                )
                latency_tracker.clear_all()
                _last_latency_report = now

            # Poll for pending market resolution (live mode)
            if resolution_mgr.is_pending:
                _res = await resolution_mgr.tick(now, mode=order_mgr.mode)
                if _res is not None and _res.verdict == "DEAD" and lifecycle.idle_reason is None:
                    lifecycle.idle_reason = "decay"
                    _sc = rules_strategy.signal_cfg
                    _ds = decay_detector.state
                    send_sprt_decay_alert(
                        mode=order_mgr.mode,
                        signal_id=_sc.signal_id,
                        n_trades=_ds.n_trades,
                        llr=_ds.llr,
                        rolling_win_rate_pct=_ds.rolling_win_rate * 100,
                        signal_age_windows=lifecycle.signal_age_windows,
                        p_alive=_ds.p_alive,
                        p_dead=_ds.p_dead,
                    )

            # Pre-fetch next window's open price from Gamma in the last seconds
            window_tracker.prefetch_next_open_price(time_remaining)

            # Capture end-of-window snapshot at T-1s
            if time_remaining <= 1.0 and not snapshot_taken and last_window_ts > 0:
                state.snapshot()
                snapshot_taken = True

            # New window detection — delegate to WindowEventHandler
            wts = window_tracker.current_window_ts()
            if wts != last_window_ts:
                _wt_result = await window_handler.on_window_transition(
                    last_window_ts=last_window_ts,
                    strategy=rules_strategy,
                    order_mgr=order_mgr,
                    decay_detector=decay_detector,
                )
                rules_strategy = _wt_result.strategy
                decay_detector = _wt_result.decay_detector
                last_window_ts = _wt_result.last_window_ts
                snapshot_taken = _wt_result.snapshot_taken

                if gc_disabled:
                    gc.enable()
                    gc_disabled = False

            # Open price capture: Gamma authoritative → RTDS boundary → RTDS latest
            if not state.open_price_captured and state.btc_chainlink > 0 and state.window_ts > 0:
                # No price at all yet — set RTDS fallback immediately so trading can start
                boundary_tick = state.select_boundary_tick(state.window_ts)
                if boundary_tick is not None:
                    state.window_open_price = boundary_tick.price
                    state.open_price_captured = True
                    state.open_price_tier = 1
                    age_ms = state.window_ts * 1000 - boundary_tick.oracle_ts_ms
                    log.warning(
                        "window open price captured: %.2f (RTDS boundary fallback, "
                        "oracle_age=%dms)",
                        state.window_open_price,
                        age_ms,
                    )
                else:
                    state.window_open_price = state.btc_chainlink
                    state.open_price_captured = True
                    state.open_price_tier = 2
                    log.warning(
                        "window open price captured: %.2f (RTDS latest fallback)",
                        state.window_open_price,
                    )

            # Keep retrying Gamma for the authoritative price until we get it
            if not window_tracker.open_price_fetched and state.window_ts > 0:
                gamma_price = window_tracker.try_fetch_open_price()
                if gamma_price is not None:
                    old_price = state.window_open_price
                    old_tier = state.open_price_tier
                    state.window_open_price = gamma_price
                    state.open_price_captured = True
                    state.open_price_tier = 0
                    window_tracker.open_price_fetched = True
                    if old_price > 0 and abs(gamma_price - old_price) >= 0.01:
                        log.info(
                            "open price UPGRADED: $%.2f → $%.2f (tier %d → Gamma, diff=$%.2f)",
                            old_price,
                            gamma_price,
                            old_tier,
                            abs(gamma_price - old_price),
                        )
                    else:
                        log.info(
                            "open price confirmed by Gamma: $%.2f (was tier %d)",
                            gamma_price,
                            old_tier,
                        )

            if (
                not state.binance_open_price_captured
                and state.btc_binance > 0
                and state.window_ts > 0
            ):
                state.binance_window_open_price = state.btc_binance
                state.binance_open_price_captured = True
                log.info(
                    "window open price captured: %.2f (binance)", state.binance_window_open_price
                )

            # Disable GC during hot window
            if time_remaining <= 60 and not gc_disabled:
                gc.disable()
                gc_disabled = True
            elif time_remaining > 60 and gc_disabled:
                gc.enable()
                gc.collect()
                gc_disabled = False

            # Publish state snapshot for visualizer (fast — microseconds).
            # Hot binary frame every tick; cold JSON only when dirty (or
            # heartbeat). We build the cold dict and offer it to the
            # publisher first so the resulting "dirty" flag can be stamped
            # into the hot frame on the same wake-up.
            if state_publisher is not None:
                _live_sig = compute_signal(state) if state.btc_chainlink > 0 else None
                _cold_snap = StatePublisher.build_cold_snapshot(
                    signal_cfg=rules_strategy.signal_cfg,
                    idle_reason=lifecycle.idle_reason,
                    decay_verdict=decay_detector.state.verdict,
                    outcome_summary=outcome_tracker.summary(),
                    halt_reason=risk.halt_reason,
                )
                _cold_dirty = state_publisher.publish_cold_if_dirty(_cold_snap, now)
                _hot_snap = StatePublisher.build_hot_snapshot(
                    btc_binance=state.btc_binance,
                    btc_chainlink=state.btc_chainlink,
                    binance_obi=state.binance_obi,
                    window_open_price=state.window_open_price,
                    window_ts=state.window_ts,
                    time_remaining=time_remaining,
                    best_bid_up=state.best_bid_up,
                    best_ask_up=state.best_ask_up,
                    best_bid_down=state.best_bid_down,
                    best_ask_down=state.best_ask_down,
                    signal=_live_sig,
                    signal_age_windows=lifecycle.signal_age_windows,
                    windows_since_last_fire=lifecycle.windows_since_last_fire,
                    fired_this_window=rules_strategy.fired,
                    decay_state=decay_detector.state,
                    kelly_result=rules_strategy.last_kelly_result,
                    wr_result=rules_strategy.kelly_wr_result,
                    bankroll=bankroll_tracker.bankroll,
                    sprt_factor=rules_strategy.sprt_factor,
                    warmup_active=rules_strategy.warmup_active,
                    vol_stddev_pct=vol_tracker.current_stddev_pct,
                    chop_avg_flips=chop_detector.avg_flips,
                    regime_ready=vol_tracker.n_returns >= 6,
                    daily_pnl=position_tracker.daily_pnl,
                    total_pnl=position_tracker.total_pnl,
                    windows_traded=position_tracker.windows_traded,
                    windows_won=position_tracker.windows_won,
                    consecutive_losses=position_tracker.consecutive_losses,
                    halted=risk.halted,
                    last_resolution=resolution_mgr.last_result,
                    last_binance_msg_ts=state.last_binance_msg_ts,
                    last_chainlink_msg_ts=state.last_chainlink_msg_ts,
                    last_clob_market_msg_ts=state.last_clob_market_msg_ts,
                    has_cold_dirty=_cold_dirty,
                )
                state_publisher.publish_binary(_hot_snap)

            if risk.halted:
                if now - last_status_log >= STATUS_LOG_INTERVAL:
                    log.warning("risk halted: %s — skipping all evaluations", risk.halt_reason)
                    last_status_log = now
                await asyncio.sleep(STRATEGY_TICK_INTERVAL)
                continue

            if isinstance(order_mgr, PaperOrderManager):
                order_mgr.check_resting_fills()
                order_mgr.record_tick_start(now)

            signal = compute_signal(state)

            # Feed chop detector with every tick
            chop_detector.tick(signal.direction.value, signal.delta_pct)

            if time_remaining <= cfg.risk.cancel_unfilled_at_sec:
                await order_mgr.cancel_all_active()

            # Skip strategy evaluation if any price feed is stale
            _stale_feed = state.is_feed_stale(
                binance_threshold=cfg.connections.binance_stale_sec,
                chainlink_threshold=cfg.connections.chainlink_stale_sec,
                clob_book_threshold=cfg.connections.clob_book_stale_sec,
            )
            if _stale_feed is not None:
                if _stale_warn_ts == 0.0 or (time.time() - _stale_warn_ts) > 30.0:
                    log.warning("feed stale: %s — suppressing trading", _stale_feed)
                    _stale_warn_ts = time.time()
                await asyncio.sleep(STRATEGY_TICK_INTERVAL)
                continue

            # Skip if a critical WebSocket feed has been disconnected too long
            if health_monitor is not None:
                _down_feed = health_monitor.critical_feed_down(threshold_sec=30.0)
                if _down_feed is not None:
                    if _health_warn_ts == 0.0 or (time.time() - _health_warn_ts) > 30.0:
                        log.warning("WS feed down: %s — suppressing trading", _down_feed)
                        _health_warn_ts = time.time()
                    await asyncio.sleep(STRATEGY_TICK_INTERVAL)
                    continue

            # Skip strategy evaluation when in IDLE state (fire stall or decay)
            if lifecycle.is_idle:
                lifecycle.tick_shadow(signal, time_remaining)
                await asyncio.sleep(STRATEGY_TICK_INTERVAL)
                continue

            assert isinstance(order_mgr, OrderExecutor)  # noqa: S101  # invariant: both OrderManager and PaperOrderManager implement OrderExecutor
            await rules_strategy.evaluate(signal, time_remaining, order_mgr)

            await asyncio.sleep(STRATEGY_TICK_INTERVAL)

        except asyncio.CancelledError:
            raise
        except (TimeoutError, aiohttp.ClientError, ConnectionError, OSError) as exc:
            # Transient network/IO errors — retry quickly
            log.warning("strategy loop transient error: %s", exc)
            await asyncio.sleep(1.0)
        except Exception:  # broad catch intentional: live trading loop must pause, not crash
            log.exception("strategy loop UNEXPECTED error")
            if not cfg.is_paper:
                # In live mode, pause longer to avoid trading with bad state
                log.error("live mode: pausing 30s after unexpected error")
                await asyncio.sleep(30.0)
            else:
                await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run(signal_path_override: str | None = None, standalone: bool = False) -> None:
    cfg = load_config()

    # Compute mode-dependent data paths and create directories
    paths = cfg.data_paths(_SUBMODULE_ROOT)
    paths.ensure_dirs()

    # Pre-flight checks — fast, synchronous, no network
    if not _preflight(cfg, paths, signal_path_override, standalone):
        sys.exit(1)

    setup_logging(cfg.mode.log_level, str(paths.logs), cfg.mode.log_retention_days)

    log.info("starting polymarket bot — mode=%s", cfg.mode.trading)
    log.info(
        "data paths: bankroll=%s journal=%s state=%s results=%s",
        paths.bankroll,
        paths.journal,
        paths.state,
        paths.results,
    )

    # IPC setup
    pending_signal_mgr: PendingSignalManager | None = None
    ipc_server = None

    # v3.0 signal-identity dedupe feedback: an in-memory ring buffer of
    # recently resolved fires, populated by the strategy loop's TradeJournal
    # whenever it records a fired+resolved outcome. The IPC status_query
    # handler reads a snapshot from here instead of rereading the JSONL file
    # on disk, so the orchestrator's feedback channel never touches the hot
    # path's I/O subsystem. Constructed unconditionally so _strategy_loop
    # can always wire it into the journal; only the IPC handler actually
    # reads from it.
    fire_mailbox = RecentFireMailbox(maxlen=64)
    _status_source = "paper" if cfg.is_paper else "live"

    if standalone:
        log.info("running in standalone mode — IPC disabled")
    else:
        from shared.ipc import SignalServer

        pending_signal_mgr = PendingSignalManager()

        def _on_signal(signal_data: dict[str, Any], summary_file: str) -> None:
            log.info(
                "received new signal from orchestrator: rank=%s side=%s",
                signal_data.get("rank", "?"),
                signal_data.get("side", "?"),
            )
            pending_signal_mgr.set_pending(signal_data, summary_file)

        def _status_provider() -> dict[str, Any]:
            fires = fire_mailbox.snapshot(source=_status_source, limit=20)
            return {
                "recent_fires": [
                    {
                        "signal_id": f.signal_id,
                        "won": f.won,
                        "timestamp": f.timestamp,
                    }
                    for f in fires
                ],
                "mode": _status_source,
            }

        ipc_server = SignalServer(
            _on_signal,
            host=cfg.ipc.host,
            port=cfg.ipc.port,
            status_provider=_status_provider,
        )
        ipc_server.start()

    # Visualizer state publisher (disabled via ipc.visualizer_enabled=false)
    state_publisher: StatePublisher | None = None
    if cfg.ipc.visualizer_enabled:
        state_publisher = StatePublisher(host=cfg.ipc.visualizer_host, port=cfg.ipc.visualizer_port)
        state_publisher.start()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:
        # In live mode, build CLOB client early so we can reuse it for
        # both startup validation and the user WS creds (avoids a
        # duplicate create_or_derive_api_creds() HTTP call).
        clob = None
        derived_creds = None
        if not cfg.is_paper:
            try:
                clob = _build_clob_client(cfg)
                log.info("CLOB auth: ok")
                if hasattr(clob, "creds") and clob.creds:
                    derived_creds = {
                        "apiKey": clob.creds.api_key,
                        "secret": clob.creds.api_secret,
                        "passphrase": clob.creds.api_passphrase,
                    }
            except (OSError, ValueError, KeyError) as exc:
                log.error("CLOB auth failed: %s", exc)
                sys.exit(1)

        if not await _validate_startup(cfg, session, clob=clob):
            log.error("startup validation failed")
            sys.exit(1)

        latency = LatencyTracker()
        state = MarketState()
        position_tracker = PositionTracker()
        risk = RiskRegistry.from_config(cfg.risk, position_tracker)
        fee_tracker = FeeTracker(latency=latency)

        window_tracker = WindowTracker(cfg, state, session, latency=latency)

        # Create regime trackers (vol/chop/outcome) with cache loading
        regime = RegimeManager.create(cfg.regime, paths)
        vol_tracker = regime.vol_tracker
        chop_detector = regime.chop_detector
        outcome_tracker = regime.outcome_tracker

        # Start price feeds (Binance + Chainlink) before signal wait so
        # pre-signal vol/chop collection has live market data.  CLOB feeds
        # are deferred until after the signal provides token IDs — the
        # Polymarket CLOB server drops idle connections that haven't subscribed.
        health_monitor = FeedHealthMonitor()
        ws_tasks = await _run_price_feeds(cfg, state, health_monitor, latency=latency)

        if not await _wait_for_prices(state):
            for t in ws_tasks:
                t.cancel()
            sys.exit(1)

        # Build strategy — from file if provided, otherwise wait for orchestrator
        if signal_path_override:
            rules_strategy = _build_strategy(cfg, state, signal_path_override)
        else:
            assert pending_signal_mgr is not None, "no signal file and no IPC — nothing to do"  # noqa: S101  # invariant check, not input validation
            log.info(
                "no signal file provided -- waiting for orchestrator to deliver first signal..."
            )
            rules_strategy = await _wait_for_first_signal(
                cfg,
                state,
                pending_signal_mgr,
                vol_tracker,
                chop_detector,
                outcome_tracker,
                paths,
            )

        # Now that we have a signal (with token IDs), start CLOB feeds
        clob_tasks = await _run_clob_feeds(
            cfg, state, health_monitor, api_creds=derived_creds, latency=latency
        )
        ws_tasks.extend(clob_tasks)

        _signal_id = rules_strategy.signal_cfg.signal_id

        if cfg.is_paper:
            paper_mgr = PaperOrderManager(
                cfg,
                state,
                risk,
                fee_tracker,
                results_dir=paths.results,
            )
            today_str = datetime.now(UTC).strftime("%Y-%m-%d")
            position_tracker.load_state(paths.state, today_str, current_signal_id=_signal_id)
            if paths.state.exists():
                try:
                    saved = _json.loads(paths.state.read_text())
                    if saved.get("date") == today_str and "balance_usd" in saved:
                        paper_mgr.balance = saved["balance_usd"]
                        log.info("resumed paper balance: $%.2f", paper_mgr.balance)
                except (_json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
                    log.warning("failed to resume paper balance: %s", exc)
            log.info(
                "paper trading mode — orders will be simulated (balance=$%.2f)", paper_mgr.balance
            )
            order_mgr: OrderManager | PaperOrderManager = paper_mgr
        else:
            order_mgr = OrderManager(cfg, state, clob, risk, fee_tracker)
            today_str = datetime.now(UTC).strftime("%Y-%m-%d")
            position_tracker.load_state(paths.state, today_str, current_signal_id=_signal_id)
            log.info("LIVE trading mode")

        strategy_task = asyncio.create_task(
            _strategy_loop(
                cfg,
                state,
                window_tracker,
                rules_strategy,
                order_mgr,
                position_tracker,
                risk,
                fee_tracker,
                session,
                paths=paths,
                pending_signal_mgr=pending_signal_mgr,
                latency_tracker=latency,
                health_monitor=health_monitor,
                vol_tracker=vol_tracker,
                chop_detector=chop_detector,
                outcome_tracker=outcome_tracker,
                state_publisher=state_publisher,
                fire_mailbox=fire_mailbox,
            ),
            name="strategy",
        )

        all_tasks = [*ws_tasks, strategy_task]

        try:
            done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                if task.exception():
                    log.error("task %s crashed: %s", task.get_name(), task.exception())
            # Cancel surviving tasks immediately so they don't run headless
            for task in pending:
                task.cancel()
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("shutting down...")
        finally:
            if ipc_server:
                ipc_server.stop()
            if state_publisher is not None:
                state_publisher.stop()
            for task in all_tasks:
                task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*all_tasks)
            log.info(
                "bot stopped — daily pnl=%.4f total=%.4f",
                position_tracker.daily_pnl,
                position_tracker.total_pnl,
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Polymarket BTC 5-minute trading bot",
    )
    parser.add_argument(
        "signal_file",
        nargs="?",
        metavar="SIGNAL_FILE",
        default=None,
        help="Path to a PolySignalEngine signal_NNN.json file. "
        "Required with --standalone; otherwise waits for orchestrator.",
    )
    parser.add_argument(
        "--signal",
        metavar="PATH",
        default=None,
        dest="signal_flag",
        help="(alias) Path to signal file — same as the positional argument.",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        default=False,
        help="Run without IPC server — no orchestrator connection, signal file required.",
    )
    args = parser.parse_args()

    signal_path = args.signal_file or args.signal_flag
    if args.standalone and not signal_path:
        parser.error(
            "standalone mode requires a signal file: python main.py --standalone path/to/signal.json"
        )

    try:
        import uvloop  # type: ignore[import-not-found]  # optional uvloop, stdlib fallback

        uvloop.install()
    except ImportError:
        pass
    asyncio.run(run(signal_path_override=signal_path, standalone=args.standalone))


if __name__ == "__main__":
    main()
