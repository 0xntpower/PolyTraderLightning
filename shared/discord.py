"""Non-blocking Discord webhook notifications with per-channel routing.

Each channel is configured via its own environment variable holding the full
webhook URL.  Messages are queued and sent from a single background thread so
callers never block on HTTP I/O.

Environment variables (one per channel):
    DISCORD_WEBHOOK_PAPER_SUMMARIES
    DISCORD_WEBHOOK_PAPER_BETS
    DISCORD_WEBHOOK_LIVE_SUMMARIES
    DISCORD_WEBHOOK_LIVE_BETS
    DISCORD_WEBHOOK_SIGNAL_UPDATED
    DISCORD_WEBHOOK_GENERATED_SIGNALS
    DISCORD_WEBHOOK_MISC_UPDATES
    DISCORD_WEBHOOK_LOG_ANALYSIS
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any  # Any: Discord webhook JSON payloads have mixed-type embed fields
from urllib import request
from urllib.error import URLError

log = logging.getLogger(__name__)

_QUEUE_MAXSIZE = int(os.environ.get("DISCORD_NOTIFIER_QUEUE_SIZE", "1024"))

# Colours
_GREEN = 0x2ECC71
_RED = 0xE74C3C
_GOLD = 0xF1C40F
_BLUE = 0x3498DB
_PURPLE = 0x9B59B6
_GREY = 0x95A5A6
_BLURPLE = 0x5865F2


# ---------------------------------------------------------------------------
# Core notifier — singleton background sender
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Payload:
    webhook_url: str
    body: dict[str, Any]


class _DiscordSender:
    """Single background thread that drains a shared queue of webhook payloads."""

    def __init__(self, maxsize: int = 1024) -> None:
        self._queue: queue.Queue[_Payload] = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name="discord-webhook",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    def enqueue(self, payload: _Payload) -> bool:
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            log.warning("Discord queue full (%d) — dropping message", self._queue.maxsize)
            return False

    def close(self) -> None:
        self._stop.set()

    _MAX_RETRIES = 2
    _RETRY_DELAY = 1.0

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._post_with_retry(payload)
            except (OSError, URLError, TimeoutError, ValueError) as exc:
                log.warning("Discord webhook failed after retries: %s", exc)
            finally:
                self._queue.task_done()

    def _post_with_retry(self, payload: _Payload) -> None:
        last_exc: Exception | None = None
        for attempt in range(1 + self._MAX_RETRIES):
            try:
                self._post(payload)
                return
            except (OSError, URLError, TimeoutError, ValueError) as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    import time as _time

                    _time.sleep(self._RETRY_DELAY)
        if last_exc:
            raise last_exc

    @staticmethod
    def _post(payload: _Payload) -> None:
        body = json.dumps(payload.body, ensure_ascii=False).encode("utf-8")
        req = request.Request(  # noqa: S310  # Discord webhook URL, not user input
            payload.webhook_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "PolySignalLab/1.0",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=5) as resp:  # noqa: S310  # Discord webhook URL, not user input
            resp.read()


_sender: _DiscordSender | None = None
_sender_lock = threading.Lock()


def _get_sender() -> _DiscordSender:
    global _sender  # noqa: PLW0603  # singleton pattern for Discord sender
    if _sender is None:
        with _sender_lock:
            if _sender is None:
                _sender = _DiscordSender(maxsize=_QUEUE_MAXSIZE)
    return _sender


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _send(webhook_url: str, embeds: list[dict[str, Any]]) -> bool:
    """Queue a message with embeds to a specific webhook URL. Fire-and-forget."""
    if not webhook_url:
        return False
    body = {"embeds": embeds, "allowed_mentions": {"parse": []}}
    return _get_sender().enqueue(_Payload(webhook_url=webhook_url, body=body))


def _embed(
    title: str,
    description: str = "",
    colour: int = _BLURPLE,
    fields: list[dict[str, Any]] | None = None,
    footer: str = "",
) -> dict[str, Any]:
    e: dict[str, Any] = {
        "title": title,
        "color": colour,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if description:
        desc = description[:3900] + "\n…" if len(description) > 3900 else description
        e["description"] = desc
    if fields:
        e["fields"] = fields
    if footer:
        e["footer"] = {"text": footer}
    return e


def _field(name: str, value: str, inline: bool = True) -> dict[str, str | bool]:
    return {"name": name, "value": value, "inline": inline}


# ---------------------------------------------------------------------------
# Channel webhook URLs (lazily read from env)
# ---------------------------------------------------------------------------


def _url(env_var: str) -> str:
    return os.environ.get(env_var, "").strip()


# ---------------------------------------------------------------------------
# PUBLIC API — one function per channel / event type
# ---------------------------------------------------------------------------

# ── Window summaries ──────────────────────────────────────────────────────


def send_window_summary(
    *,
    mode: str,
    window_pnl: float,
    session_pnl: float,
    session_won: float,
    session_lost: float,
    win_rate_pct: float,
    wins: int,
    total: int,
    consecutive_losses: int,
    balance: float | None = None,
) -> bool:
    """Send a window summary to the appropriate summaries channel."""
    is_paper = mode == "paper"
    url = _url("DISCORD_WEBHOOK_PAPER_SUMMARIES" if is_paper else "DISCORD_WEBHOOK_LIVE_SUMMARIES")
    tag = "PAPER" if is_paper else "LIVE"

    colour = _GREEN if window_pnl > 0 else _RED if window_pnl < 0 else _GREY

    fields = [
        _field("Window PnL", f"`${window_pnl:+.4f}`"),
        _field("Session PnL", f"`${session_pnl:+.4f}`"),
        _field("Won / Lost", f"`${session_won:.2f}` / `${session_lost:.2f}`"),
        _field("Win Rate", f"`{win_rate_pct:.1f}%` ({wins}/{total})"),
    ]
    if consecutive_losses > 0:
        fields.append(_field("Consec Losses", f"`{consecutive_losses}`"))
    if balance is not None:
        fields.append(_field("Balance", f"`${balance:.2f}`"))

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Window Summary",
                colour=colour,
                fields=fields,
            )
        ],
    )


# ── Bet events ────────────────────────────────────────────────────────────


def send_bet_placed(
    *,
    mode: str,
    side: str,
    price: float,
    size_usd: float,
    rank: int,
    order_id: str = "",
    entry_type: str = "taker",
    maker_usd: float = 0.0,
    taker_usd: float = 0.0,
    obi_threshold: float | None = None,
    obi_depth: str | None = None,
    obi_observed: float | None = None,
) -> bool:
    """Notify that a bet was submitted.

    When both ``maker_usd`` and ``taker_usd`` are positive, the entry is a
    maker-partial + taker-remainder combo (post-mortem 2026-04-22 §5.2); the
    "Entry" field shows the capital split as percentages so operators can
    see at a glance whether the maker captured most of the position or the
    taker escalation did. For pure maker or pure taker fills the existing
    single-label form ("Maker Fill" / "Taker") is preserved.
    """
    is_paper = mode == "paper"
    url = _url("DISCORD_WEBHOOK_PAPER_BETS" if is_paper else "DISCORD_WEBHOOK_LIVE_BETS")
    tag = "PAPER" if is_paper else "LIVE"

    entry_label: str
    if maker_usd > 0.0 and taker_usd > 0.0:
        total = maker_usd + taker_usd
        maker_pct = 100.0 * maker_usd / total
        taker_pct = 100.0 * taker_usd / total
        entry_label = f"Maker {maker_pct:.0f}% / Taker {taker_pct:.0f}%"
    elif entry_type == "maker":
        entry_label = "Maker Fill"
    else:
        entry_label = "Taker"

    fields = [
        _field("Side", f"`{side.upper()}`"),
        _field("Entry Price", f"`{price:.2f}`"),
        _field("Size", f"`${size_usd:.2f}`"),
        _field("Signal", f"`#{rank}`"),
        _field("Entry", f"`{entry_label}`"),
    ]
    if order_id:
        fields.append(_field("Order", f"`{order_id[:16]}`"))
    if obi_threshold is not None:
        if obi_threshold > 0.0 and obi_depth:
            gate = f"{obi_threshold:.2f}@{obi_depth}"
            if obi_observed is not None:
                fields.append(_field("OBI Gate", f"`{gate}` (obs `{obi_observed:+.3f}`)"))
            else:
                fields.append(_field("OBI Gate", f"`{gate}`"))
        elif obi_observed is not None:
            fields.append(_field("OBI Gate", f"`off` (obs `{obi_observed:+.3f}`)"))

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Bet Placed",
                colour=_BLUE,
                fields=fields,
            )
        ],
    )


def send_bet_result(
    *,
    mode: str,
    outcome: str,
    pnl: float,
    entry_price: float,
    side: str,
    size_usd: float,
    balance: float | None = None,
    market_outcome: str | None = None,
    maker_usd: float = 0.0,
    taker_usd: float = 0.0,
    signal_age_at_fire_h: float | None = None,
    typical_lifetime_h: float | None = None,
    typical_lifetime_samples: int | None = None,
    typical_lifetime_status: str = "unavailable",
    obi_threshold: float | None = None,
    obi_depth: str | None = None,
) -> bool:
    """Notify bet outcome — WIN, LOSS, SKIP, or FLAT.

    ``market_outcome`` is the side the window resolved to (e.g. "up"/"down").
    Shown alongside the bot's bet side so LOSS notifications reveal whether
    the market went against us or a correct bet was exited early.

    When both ``maker_usd`` and ``taker_usd`` are positive the entry was a
    combined maker-partial + taker-remainder fill — an extra "Fill" field
    renders the capital split so operators can see how the position was
    actually built (post-mortem 2026-04-22 §5.2 follow-up).
    """
    is_paper = mode == "paper"
    url = _url("DISCORD_WEBHOOK_PAPER_BETS" if is_paper else "DISCORD_WEBHOOK_LIVE_BETS")
    tag = "PAPER" if is_paper else "LIVE"

    colour_map = {"WIN": _GREEN, "LOSS": _RED, "SKIP": _GREY, "FLAT": _GREY}
    colour = colour_map.get(outcome, _GREY)

    fields = [
        _field("PnL", f"`${pnl:+.4f}`"),
        _field("Side", f"`{side.upper()}`"),
        _field("Entry", f"`{entry_price:.2f}`"),
        _field("Size", f"`${size_usd:.2f}`"),
    ]
    if maker_usd > 0.0 and taker_usd > 0.0:
        total = maker_usd + taker_usd
        maker_pct = 100.0 * maker_usd / total
        taker_pct = 100.0 * taker_usd / total
        fields.append(_field("Fill", f"`Maker {maker_pct:.0f}% / Taker {taker_pct:.0f}%`"))
    if signal_age_at_fire_h is not None:
        fields.append(_field("Signal Age at Fire", f"`{signal_age_at_fire_h:.1f}h`"))
    if typical_lifetime_h is not None:
        n_suffix = f", n={typical_lifetime_samples}" if typical_lifetime_samples is not None else ""
        status_suffix = ", tentative" if typical_lifetime_status == "tentative" else ""
        fields.append(
            _field(
                "Typical Lifetime",
                f"`{typical_lifetime_h:.1f}h (median{n_suffix}{status_suffix})`",
            )
        )
    if market_outcome:
        fields.append(_field("Market", f"`{market_outcome.upper()}`"))
    if obi_threshold is not None:
        if obi_threshold > 0.0 and obi_depth:
            fields.append(_field("OBI Gate", f"`{obi_threshold:.2f}@{obi_depth}`"))
        else:
            fields.append(_field("OBI Gate", "`off`"))
    if balance is not None:
        fields.append(_field("Balance", f"`${balance:.2f}`"))

    return _send(
        url,
        [
            _embed(
                title=f"{tag} {outcome}",
                colour=colour,
                fields=fields,
            )
        ],
    )


def send_bet_cancelled(
    *,
    mode: str,
    count: int,
    reason: str = "window close",
    orders: list[dict[str, Any]] | None = None,
) -> bool:
    """Notify that unfilled orders were cancelled.

    ``orders`` is an optional list of dicts with keys like
    ``side``, ``price``, ``size_usd``, ``best_ask`` — only present keys
    are shown.
    """
    if count == 0:
        return False
    is_paper = mode == "paper"
    url = _url("DISCORD_WEBHOOK_PAPER_BETS" if is_paper else "DISCORD_WEBHOOK_LIVE_BETS")
    tag = "PAPER" if is_paper else "LIVE"

    fields: list[dict[str, Any]] = [
        _field("Reason", f"`{reason}`", inline=False),
    ]
    if orders:
        for o in orders[:5]:  # cap at 5
            parts = []
            if "side" in o:
                parts.append(f"side=`{o['side']}`")
            if "price" in o:
                parts.append(f"bid=`{o['price']:.2f}`")
            if "size_usd" in o:
                parts.append(f"size=`${o['size_usd']:.2f}`")
            if "best_ask" in o:
                parts.append(f"ask=`{o['best_ask']:.2f}`")
            if parts:
                fields.append(
                    _field(
                        f"Order {o.get('tier', '')}".strip(),
                        " ".join(parts),
                        inline=False,
                    )
                )

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Order Cancelled",
                description=f"`{count}` unfilled order(s) cancelled.",
                colour=_GOLD,
                fields=fields,
            )
        ],
    )


def send_early_exit(
    *,
    mode: str,
    side: str,
    rank: int,
    reason: str,
    erosion: float,
    threshold: float,
    fire_delta_pct: float,
    current_delta_pct: float,
    entry_price: float,
    sell_price: float,
    pnl: float | None = None,
    balance: float | None = None,
) -> bool:
    """Notify that the bot sold a position early due to post-fire erosion."""
    is_paper = mode == "paper"
    url = _url("DISCORD_WEBHOOK_PAPER_BETS" if is_paper else "DISCORD_WEBHOOK_LIVE_BETS")
    tag = "PAPER" if is_paper else "LIVE"

    colour = _RED if (pnl is not None and pnl < 0) else _GOLD

    fields = [
        _field("Side", f"`{side.upper()}`"),
        _field("Signal", f"`#{rank}`"),
        _field("Reason", f"`{reason}`", inline=False),
        _field("Erosion", f"`{erosion:.4f}`"),
        _field("Threshold", f"`{threshold:.4f}`"),
        _field("Fire Delta", f"`{fire_delta_pct:+.4f}%`"),
        _field("Current Delta", f"`{current_delta_pct:+.4f}%`"),
        _field("Entry Price", f"`{entry_price:.2f}`"),
        _field("Sell Price", f"`{sell_price:.2f}`"),
    ]
    if pnl is not None:
        fields.append(_field("PnL", f"`${pnl:+.4f}`"))
    if balance is not None:
        fields.append(_field("Balance", f"`${balance:.2f}`"))

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Early Exit",
                colour=colour,
                fields=fields,
            )
        ],
    )


# ── Active signal updated ────────────────────────────────────────────────


def send_signal_updated(
    *,
    new_rank: int,
    new_side: str,
    old_rank: int,
    old_side: str,
    score: float | None = None,
    _tier: str | None = None,
    ev: float | None = None,
    avg_entry: float | None = None,
    conservative_wr_pct: float | None = None,
    folds_appeared: int | None = None,
    total_folds: int | None = None,
    _fold_indices: list[int] | None = None,
    obi_threshold: float | None = None,
    obi_depth: str | None = None,
    signal_age_h: float | None = None,
    typical_lifetime_h: float | None = None,
    typical_lifetime_samples: int | None = None,
    typical_lifetime_status: str = "unavailable",
    selected_over: str | None = None,
) -> bool:
    """Notify that the bot switched to a new active signal.

    ``signal_age_h`` is the engine-anchored age of this signal at delivery.
    ``typical_lifetime_h`` is the median of completed family lifetimes in
    the orchestrator's eligible-samples ring buffer; None during bootstrap.
    ``selected_over`` is the label of the top-by-score runner-up when the
    orchestrator's age-aware selector chose this signal instead.
    """
    url = _url("DISCORD_WEBHOOK_SIGNAL_UPDATED")

    fields = [
        _field("New Signal", f"`#{new_rank} {new_side.upper()}`"),
        _field("Previous", f"`#{old_rank} {old_side.upper()}`"),
    ]
    if score is not None:
        fields.append(_field("Score", f"`{score:.1f}`"))
    if ev is not None:
        fields.append(_field("EV", f"`{ev:.4f}`"))
    if avg_entry is not None:
        fields.append(_field("Entry", f"`${avg_entry:.2f}`"))
    if conservative_wr_pct is not None:
        fields.append(_field("Cons. WR", f"`{conservative_wr_pct:.1f}%`"))
    if folds_appeared is not None and total_folds is not None and total_folds > 0:
        fields.append(_field("Folds", f"`{folds_appeared}/{total_folds}`"))
    if obi_threshold is not None:
        if obi_threshold > 0.0 and obi_depth:
            fields.append(_field("OBI Gate", f"`{obi_threshold:.2f}@{obi_depth}`"))
        else:
            fields.append(_field("OBI Gate", "`off`"))
    if signal_age_h is not None:
        fields.append(_field("Signal Age", f"`{signal_age_h:.1f}h`"))
    if typical_lifetime_h is not None:
        n_suffix = f", n={typical_lifetime_samples}" if typical_lifetime_samples is not None else ""
        status_suffix = ", tentative" if typical_lifetime_status == "tentative" else ""
        fields.append(
            _field(
                "Typical Lifetime",
                f"`{typical_lifetime_h:.1f}h (median{n_suffix}{status_suffix})`",
            )
        )
    if selected_over:
        fields.append(_field("Selected Over", f"`{selected_over}`"))

    return _send(
        url,
        [
            _embed(
                title="Signal Updated",
                colour=_PURPLE,
                fields=fields,
            )
        ],
    )


# ── Generated signals (orchestrator rankings) ────────────────────────────


def send_generated_signals(
    *,
    signals: list[dict[str, Any]],
    best_rank: int | None = None,
    best_tier: str | None = None,
    best_score: float | None = None,
) -> bool:
    """Send the ranked signals table from the orchestrator."""
    url = _url("DISCORD_WEBHOOK_GENERATED_SIGNALS")
    if not url:
        return False

    # Build compact table — show top 10 max
    lines = ["```"]
    lines.append(
        f"{'#':<4} {'SIDE':<5} {'SCORE':>6} {'EV':>7} {'ENTRY':>6} {'WR':>5} "
        f"{'CWR':>5} {'FOLDS':>6} {'OBI':>9}"
    )
    lines.append("-" * 63)
    for sig in signals[:10]:
        side = sig.get("side", "?")[:4].upper()
        score = sig.get("_smart_score", sig.get("smartScore", 0))
        ev = sig.get("evPerTrade", 0)
        entry = sig.get("avgEntryPrice", 0)
        wr = sig.get("oosWinRatePct", 0)
        cwr = sig.get("conservativeWinRatePct", 0)
        folds = f"{sig.get('wfFoldsAppeared', 0)}/{sig.get('wfTotalTestFolds', 0)}"
        obi_t = float(sig.get("obiThreshold", 0.0) or 0.0)
        obi_d = str(sig.get("obiDepth", "none") or "none")
        obi_str = f"{obi_t:.2f}@{obi_d}" if obi_t > 0.0 else "off"
        lines.append(
            f"{sig.get('rank', 0):<4} {side:<5} {score:>6.1f} {ev:>7.4f} "
            f"${entry:>4.2f} {wr:>4.0f}% {cwr:>4.0f}% {folds:>6} {obi_str:>9}"
        )
    lines.append("```")
    table = "\n".join(lines)

    footer_parts = []
    if best_rank is not None:
        footer_parts.append(f"Best: #{best_rank}")
    if best_tier:
        footer_parts.append(best_tier)
    if best_score is not None:
        footer_parts.append(f"score={best_score:.1f}")

    return _send(
        url,
        [
            _embed(
                title=f"Signal Rankings — {len(signals)} signals",
                description=table,
                colour=_BLURPLE,
                footer=" · ".join(footer_parts) if footer_parts else "",
            )
        ],
    )


# ── Misc updates (health / skip streak alerts) ─────────────────────────


def send_skip_streak_alert(
    *,
    mode: str,
    hours_since_last_trade: float,
    windows_skipped: int,
    skip_reasons: dict[str, int],
    session_pnl: float,
    win_rate_pct: float,
    wins: int,
    total: int,
    current_signal: str,
    signal_score: float | None = None,
    signal_folds: str | None = None,
    uptime_hours: float | None = None,
) -> bool:
    """Alert that the bot has been skipping trades for an extended period."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    # Build reason breakdown
    reason_lines = []
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        reason_lines.append(f"`{reason}`: {count}")
    reasons_str = "\n".join(reason_lines) if reason_lines else "`no data`"

    fields = [
        _field("Hours Since Last Trade", f"`{hours_since_last_trade:.1f}h`"),
        _field("Windows Skipped", f"`{windows_skipped}`"),
        _field("Session PnL", f"`${session_pnl:+.2f}`"),
        _field("Win Rate", f"`{win_rate_pct:.1f}%` ({wins}/{total})"),
        _field("Current Signal", f"`{current_signal}`"),
    ]
    if signal_score is not None:
        fields.append(_field("Signal Score", f"`{signal_score:.1f}`"))
    if signal_folds:
        fields.append(_field("Signal Folds", f"`{signal_folds}`"))
    if uptime_hours is not None:
        fields.append(_field("Bot Uptime", f"`{uptime_hours:.1f}h`"))
    fields.append(_field("Skip Reasons", reasons_str, inline=False))

    return _send(
        url,
        [
            _embed(
                title=f"{tag} No Trades — {hours_since_last_trade:.1f}h",
                description="Bot is running but no trades have executed recently.",
                colour=_GOLD,
                fields=fields,
            )
        ],
    )


def send_risk_level(
    *,
    mode: str,
    vol_severity: float,
    chop_severity: float,
    outcome_severity: float,
    total_discount: float,
    base_p: float,
    adjusted_p: float,
    entry_price: float | None = None,
    signal_id: str,
) -> bool:
    """Unified risk level notification — sent each window when discount > 0.

    Shows a 10-char severity bar for each risk component, the combined
    discount, and the resulting win rate adjustment.
    """
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    # 10-char severity bar (same style as the old regime shift notification)
    def _bar(severity: float) -> str:
        filled = round(severity * 10)
        return "\u2588" * filled + "\u2591" * (10 - filled)

    # Overall risk level label + colour
    worst = max(vol_severity, chop_severity, outcome_severity)
    if worst >= 0.7:
        level, colour = "HIGH", _RED
    elif worst >= 0.3:
        level, colour = "MODERATE", _GOLD
    else:
        level, colour = "LOW", _GREEN

    # Build description with component bars, each on its own line
    lines = [
        f"**Volatility** `{_bar(vol_severity)}` **{vol_severity * 100:.0f}%**",
        f"**Chop**\u2003\u2003\u2003\u2003`{_bar(chop_severity)}` **{chop_severity * 100:.0f}%**",
        f"**Outcome**\u2003 `{_bar(outcome_severity)}` **{outcome_severity * 100:.0f}%**",
        "",
        f"**Total Discount** \u2014 **\u2212{total_discount * 100:.1f}%**",
        f"**Win Rate** \u2014 `{base_p * 100:.1f}%` \u2192 `{adjusted_p * 100:.1f}%`",
    ]
    if entry_price is not None and entry_price > 0:
        lines.append(f"**Entry Price** \u2014 `${entry_price:.2f}`")

    fields = [
        _field("Signal ID", f"`{signal_id}`", inline=False),
    ]

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Risk Level \u2014 {level}",
                description="\n".join(lines),
                colour=colour,
                fields=fields,
            )
        ],
    )


def send_sprt_decay_alert(
    *,
    mode: str,
    signal_id: str,
    n_trades: int,
    llr: float,
    rolling_win_rate_pct: float,
    signal_age_windows: int,
    p_alive: float,
    p_dead: float,
) -> bool:
    """Alert that SPRT has concluded a signal has decayed."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    fields = [
        _field("Signal", f"`{signal_id[:40]}`", inline=False),
        _field("Trades Observed", f"`{n_trades}`"),
        _field("Rolling WR", f"`{rolling_win_rate_pct:.1f}%`"),
        _field("LLR", f"`{llr:.2f}`"),
        _field(
            "Signal Age", f"`{signal_age_windows}` windows (~{signal_age_windows * 5 / 60:.1f}h)"
        ),
        _field("SPRT Thresholds", f"p_alive=`{p_alive:.2f}` p_dead=`{p_dead:.2f}`"),
    ]

    return _send(
        url,
        [
            _embed(
                title=f"{tag} SPRT Signal Decay Detected",
                description="SPRT concluded signal underperformance. Bot is now idle and shadow-tracking.",
                colour=_RED,
                fields=fields,
            )
        ],
    )


def send_consecutive_loss_warning(
    *,
    mode: str,
    streak: int,
    max_allowed: int,
    recent_losses: list[dict[str, Any]],
    session_pnl: float,
    daily_pnl: float,
) -> bool:
    """Warn that consecutive losses are approaching the risk limit."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    # Format recent losses
    loss_lines = [
        f"entry=`{loss.get('entry', 0):.2f}` pnl=`${loss.get('pnl', 0):+.4f}`"
        for loss in recent_losses[-5:]
    ]
    loss_str = "\n".join(loss_lines) if loss_lines else "`—`"

    fields = [
        _field("Streak", f"`{streak}` / `{max_allowed}` max"),
        _field("Session PnL", f"`${session_pnl:+.2f}`"),
        _field("Daily PnL", f"`${daily_pnl:+.2f}`"),
        _field("Recent Losses", loss_str, inline=False),
    ]

    return _send(
        url,
        [
            _embed(
                title=f"{tag} {streak} Consecutive Losses",
                description=f"Trading halts at `{max_allowed}`.",
                colour=_RED,
                fields=fields,
            )
        ],
    )


def send_warmup_complete(
    *,
    mode: str,
    warmup_minutes: float,
    signal_id: str,
    session_pnl: float,
    windows_traded: int,  # noqa: ARG001
    win_rate_pct: float,
    wins: int,
    total: int,
    bankroll: float,
) -> bool:
    """Notify that warmup period has ended and full Kelly sizing is now active."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    fields = [
        _field("Warmup Duration", f"`{warmup_minutes:.0f}` min"),
        _field("Signal", f"`{signal_id[:40]}`", inline=False),
        _field("Session PnL", f"`${session_pnl:+.2f}`"),
        _field("Win Rate", f"`{win_rate_pct:.1f}%` ({wins}/{total})"),
        _field("Bankroll", f"`${bankroll:.2f}`"),
    ]

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Warmup Complete",
                description="Full Kelly sizing is now active.",
                colour=_GREEN,
                fields=fields,
            )
        ],
    )


def send_presignal_warmup(
    *,
    vol_prices: int,
    chop_windows: int,
    credited_min: float,
    warmup_minutes: float,
) -> bool:
    """Periodic update showing pre-signal warmup progress."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    remaining = max(0.0, warmup_minutes - credited_min)

    fields = [
        _field("Vol Prices", f"`{vol_prices}`"),
        _field("Chop Windows", f"`{chop_windows}`"),
        _field("Credited", f"`{credited_min:.0f}` / `{warmup_minutes:.0f}` min"),
        _field("Remaining", f"`{remaining:.0f}` min"),
    ]

    return _send(
        url,
        [
            _embed(
                title="Pre-Signal Warmup",
                description="Collecting vol/chop baseline while waiting for signal.",
                colour=_BLUE,
                fields=fields,
            )
        ],
    )


def send_presignal_warmup_done(
    *,
    vol_prices: int,
    chop_windows: int,
    credited_min: float,
) -> bool:
    """One-shot notification: warmup baseline is complete, waiting for signal."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")

    fields = [
        _field("Vol Prices", f"`{vol_prices}`"),
        _field("Chop Windows", f"`{chop_windows}`"),
        _field("Credited", f"`{credited_min:.0f}` min"),
    ]

    return _send(
        url,
        [
            _embed(
                title="Warmup Complete \u2014 Waiting for Signal",
                description=(
                    "Vol/chop baseline collected. Standing by for first signal from orchestrator."
                ),
                colour=_GREEN,
                fields=fields,
            )
        ],
    )


def send_cache_loaded(
    *,
    mode: str,
    vol_prices: int,
    chop_windows: int,
    vol_stddev_pct: float,
    chop_avg_flips: float,
    cache_age_min: float,
    warmup_credit_min: float,
    warmup_remaining_min: float,
) -> bool:
    """Notify that vol/chop context was restored from disk."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    fields = [
        _field("Vol Prices", f"`{vol_prices}`"),
        _field("Chop Windows", f"`{chop_windows}`"),
        _field("Vol Stddev", f"`{vol_stddev_pct:.3f}%`"),
        _field("Chop Avg Flips", f"`{chop_avg_flips:.1f}`"),
        _field("Cache Age", f"`{cache_age_min:.1f}` min"),
        _field("Warmup Credit", f"`{warmup_credit_min:.0f}` min"),
        _field("Warmup Remaining", f"`{warmup_remaining_min:.0f}` min"),
    ]

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Regime Cache Loaded",
                description="Vol/chop context restored from disk — warmup shortened.",
                colour=_GREEN,
                fields=fields,
            )
        ],
    )


def send_bet_scale_squeeze(
    *,
    mode: str,
    combined_scale_pct: float,
    sprt_scale_pct: float,
    age_scale_pct: float,
    vol_scale_pct: float,
    chop_scale_pct: float,
    vol_stddev_pct: float,
    chop_avg_flips: float,
    llr: float,
    signal_age_windows: int,
    consecutive_squeeze_windows: int,
) -> bool:
    """Alert that bet sizing has been squeezed for multiple windows."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    # Find the dominant reducer
    components = {
        "SPRT confidence": sprt_scale_pct,
        "Age taper": age_scale_pct,
        "Volatility": vol_scale_pct,
        "Chop": chop_scale_pct,
    }
    bottleneck = min(components, key=lambda k: components[k])

    fields = [
        _field("Combined Scale", f"`{combined_scale_pct:.0f}%`"),
        _field("Bottleneck", f"`{bottleneck}` at `{components[bottleneck]:.0f}%`"),
        _field("SPRT", f"`{sprt_scale_pct:.0f}%` (LLR: {llr:.2f})"),
        _field("Age", f"`{age_scale_pct:.0f}%` ({signal_age_windows}w)"),
        _field("Volatility", f"`{vol_scale_pct:.0f}%` ({vol_stddev_pct:.3f}%)"),
        _field("Chop", f"`{chop_scale_pct:.0f}%` ({chop_avg_flips:.1f} flips)"),
        _field(
            "Duration",
            f"`{consecutive_squeeze_windows}w` (~{consecutive_squeeze_windows * 5 / 60:.1f}h)",
        ),
    ]

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Bet Squeeze — {combined_scale_pct:.0f}%",
                description=f"Bottleneck: **{bottleneck}**. Duration: `{consecutive_squeeze_windows}` windows.",
                colour=_GOLD,
                fields=fields,
            )
        ],
    )


def send_session_summary(
    *,
    mode: str,
    trades_placed: int,
    wins: int,
    losses: int,
    net_pnl: float,
    gross_won: float,
    gross_lost: float,
    avg_bet_scale_pct: float,
    avg_vol_reading_pct: float,
    avg_chop_flips: float,
    signals_received: int,
    signals_blocked_score: int,
    signals_blocked_folds: int,
    windows_total: int,
    balance: float | None = None,
) -> bool:
    """Send session-lifetime performance summary."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    wr_pct = (wins / trades_placed * 100) if trades_placed > 0 else 0
    colour = _GREEN if net_pnl > 0 else _RED if net_pnl < 0 else _GREY

    fields = [
        _field("Net PnL", f"`${net_pnl:+.2f}`"),
        _field("Win Rate", f"`{wr_pct:.0f}%` ({wins}W / {losses}L)"),
        _field("Gross Won", f"`${gross_won:.2f}`"),
        _field("Gross Lost", f"`${gross_lost:.2f}`"),
        _field("Trades / Windows", f"`{trades_placed}` / `{windows_total}`"),
        _field("Avg Bet Scale", f"`{avg_bet_scale_pct:.0f}%`"),
        _field("Avg Vol", f"`{avg_vol_reading_pct:.3f}%`"),
        _field("Avg Chop", f"`{avg_chop_flips:.1f}` flips"),
        _field("Signals Received", f"`{signals_received}`"),
    ]
    if signals_blocked_score > 0 or signals_blocked_folds > 0:
        fields.append(
            _field(
                "Blocked", f"score: `{signals_blocked_score}` · folds: `{signals_blocked_folds}`"
            )
        )
    if balance is not None:
        fields.append(_field("Balance", f"`${balance:.2f}`"))

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Session Summary",
                colour=colour,
                fields=fields,
            )
        ],
    )


def send_signal_drought(
    *,
    consecutive_cycles_rejected: int,
    hours_without_signal: float,
    last_best_score: float,
    min_score: float,
    last_best_folds: str,
    min_folds: int,
    rejection_reason: str,
) -> bool:
    """Alert that the orchestrator can't find signals above delivery threshold."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")

    fields = [
        _field("Cycles Without Delivery", f"`{consecutive_cycles_rejected}`"),
        _field("Hours", f"`{hours_without_signal:.1f}h`"),
        _field("Best Score Seen", f"`{last_best_score:.2f}`"),
        _field("Min Score Required", f"`{min_score:.1f}`"),
        _field("Best Signal Folds", f"`{last_best_folds}`"),
        _field("Min Folds Required", f"`{min_folds}`"),
        _field("Last Rejection", f"`{rejection_reason}`", inline=False),
    ]

    return _send(
        url,
        [
            _embed(
                title=f"Signal Drought — {hours_without_signal:.1f}h",
                description="No signal meets delivery criteria. Bot continues with last known signal.",
                colour=_GOLD,
                fields=fields,
            )
        ],
    )


def send_shadow_tracking_result(
    *,
    mode: str,
    signal_id: str,
    windows_tracked: int,
    fires: int,
    fills: int,
    wins: int,
    shadow_win_rate_pct: float,
    total_signal_age: int,
    decay_was_correct: bool,
) -> bool:
    """Report results when shadow tracking completes after a decay/stall event."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    verdict = "CONFIRMED" if decay_was_correct else "PREMATURE"
    colour = _GREEN if decay_was_correct else _RED

    fields = [
        _field("Signal", f"`{signal_id[:40]}`", inline=False),
        _field("Verdict", f"Decay was **{verdict}**"),
        _field("Shadow WR", f"`{shadow_win_rate_pct:.0f}%` ({wins}/{fills})"),
        _field("Windows Tracked", f"`{windows_tracked}` (~{windows_tracked * 5 / 60:.1f}h)"),
        _field("Fires / Fills", f"`{fires}` / `{fills}`"),
        _field(
            "Total Signal Age", f"`{total_signal_age}` windows (~{total_signal_age * 5 / 60:.1f}h)"
        ),
    ]

    if decay_was_correct:
        description = (
            f"Shadow WR `{shadow_win_rate_pct:.0f}%` — below breakeven. **Decay confirmed.**"
        )
    else:
        description = (
            f"Shadow WR `{shadow_win_rate_pct:.0f}%` — still profitable. **Decay was premature.**"
        )

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Shadow Tracking — {verdict}",
                description=description,
                colour=colour,
                fields=fields,
            )
        ],
    )


def send_live_wr_checkpoint(
    *,
    mode: str,
    signal_id: str,
    fills: int,
    wins: int,
    losses: int,
    live_wr_pct: float,
    expected_wr_pct: float,
    entry_avg: float,
    breakeven_wr_pct: float,
    net_pnl: float,
    llr: float,
    signal_age_windows: int,
) -> bool:
    """Report live win rate progress — the single most important ground truth metric."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    tag = "PAPER" if mode == "paper" else "LIVE"

    wr_gap = live_wr_pct - breakeven_wr_pct
    if fills < 10:
        colour = _BLUE
        status = "WARMING UP"
        verdict = "Too early to judge — need more data."
    elif live_wr_pct >= expected_wr_pct - 5:
        colour = _GREEN
        status = "ON TRACK"
        verdict = f"Live WR tracking within 5% of expected ({expected_wr_pct:.0f}%)."
    elif live_wr_pct >= breakeven_wr_pct:
        colour = _GOLD
        status = "BELOW EXPECTED"
        verdict = (
            f"Live WR is {expected_wr_pct - live_wr_pct:.0f}% below expected "
            f"({expected_wr_pct:.0f}%) but still above breakeven ({breakeven_wr_pct:.0f}%)."
        )
    else:
        colour = _RED
        status = "BELOW BREAKEVEN"
        verdict = (
            f"Live WR is below breakeven ({breakeven_wr_pct:.0f}%). "
            "Currently losing money on fills."
        )

    fields = [
        _field("Live WR", f"**`{live_wr_pct:.1f}%`** ({wins}W / {losses}L)", inline=False),
        _field("Expected WR", f"`{expected_wr_pct:.0f}%`"),
        _field("Breakeven WR", f"`{breakeven_wr_pct:.0f}%`"),
        _field("Edge", f"`{wr_gap:+.1f}%` vs breakeven"),
        _field("Avg Entry", f"`{entry_avg:.2f}`"),
        _field("Net PnL", f"`${net_pnl:+.4f}`"),
        _field("SPRT LLR", f"`{llr:+.3f}`"),
        _field("Signal Age", f"`{signal_age_windows}w` (~{signal_age_windows * 5 / 60:.1f}h)"),
    ]

    return _send(
        url,
        [
            _embed(
                title=f"{tag} SPRT WR Checkpoint — {status} ({fills} fills)",
                description=verdict,
                colour=colour,
                fields=fields,
                footer=signal_id[:40],
            )
        ],
    )


# ── Log analysis results ────────────────────────────────────────────────


def send_log_analysis(
    *,
    executive_summary: str,
    session_stats: str,
    recommendations: str,
    version: str = "",
    analysis_file: str = "",
) -> bool:
    """Send 12-hour log analysis to Discord."""
    url = _url("DISCORD_WEBHOOK_LOG_ANALYSIS")
    if not url:
        return False

    # Determine colour from executive summary tone
    lower_summary = executive_summary.lower()
    if any(w in lower_summary for w in ("concern", "degraded", "below breakeven", "loss")):
        colour = _RED
    elif any(w in lower_summary for w in ("mixed", "caution", "below expected")):
        colour = _GOLD
    else:
        colour = _BLUE

    embeds: list[dict[str, Any]] = []

    # 1. Executive summary embed
    footer = ""
    if version:
        footer = version
    if analysis_file:
        footer = f"{footer}  |  {analysis_file}" if footer else analysis_file

    embeds.append(
        _embed(
            title="Log Analysis — Executive Summary",
            description=executive_summary[:4000] or "(No summary available.)",
            colour=colour,
            footer=footer,
        )
    )

    # 2. Session stats embed
    if session_stats:
        embeds.append(
            _embed(
                title="Session Statistics",
                description=session_stats[:4000],
                colour=_GREY,
            )
        )

    # 3. Recommendations embed
    if recommendations:
        rec_text = recommendations
        if len(rec_text) > 4000:
            rec_text = rec_text[:3950] + "\n\n... see full report"
            if analysis_file:
                rec_text += f": `{analysis_file}`"
        embeds.append(
            _embed(
                title="Tuning Recommendations",
                description=rec_text,
                colour=_GOLD,
            )
        )

    return _send(url, embeds)


def send_bot_inactive_notice(
    *,
    reason: str = "Bot was inactive — no trading activity detected in log.",
) -> bool:
    """Send a notice when the bot was idle and analysis was skipped."""
    url = _url("DISCORD_WEBHOOK_LOG_ANALYSIS")
    if not url:
        return False

    return _send(
        url,
        [
            _embed(
                title="Log Analysis — Skipped",
                description=reason,
                colour=_GREY,
            )
        ],
    )


def send_log_archived_notice(
    *,
    filename: str,
    size_kb: float,
) -> bool:
    """Send a notice when a log has been archived (backup-only mode)."""
    url = _url("DISCORD_WEBHOOK_LOG_ANALYSIS")
    if not url:
        return False

    return _send(
        url,
        [
            _embed(
                title="Log Archived",
                description=f"`{filename}` ({size_kb:.0f} KB)",
                colour=_GREY,
            )
        ],
    )


def send_orchestrator_disabled(
    *,
    consecutive_failures: int,
    error: str,
) -> bool:
    """Critical alert: orchestrator has self-disabled after repeated engine failures."""
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    if not url:
        return False

    return _send(
        url,
        [
            _embed(
                title="ORCHESTRATOR DISABLED",
                description=(
                    f"Engine failed **{consecutive_failures}** consecutive times. "
                    "Orchestrator has halted — **manual intervention required**."
                ),
                colour=_RED,
                fields=[
                    _field("Last Error", f"```{error[:500]}```", inline=False),
                ],
            )
        ],
    )


# ── Latency report ────────────────────────────────────────────────────────


def send_latency_report(
    *,
    mode: str,
    feed_stats: list[dict[str, Any]],
) -> bool:
    """Send a compact hourly latency snapshot to misc-updates.

    feed_stats is a list of dicts with keys:
        name, samples, min_ms, max_ms, median_ms, p95_ms, mean_ms,
        jitter_ms, last_message_ago_s
    """
    url = _url("DISCORD_WEBHOOK_MISC_UPDATES")
    if not url:
        return False

    tag = "PAPER" if mode == "paper" else "LIVE"

    if not feed_stats:
        return _send(
            url,
            [
                _embed(
                    title=f"{tag} Latency",
                    description="No latency data collected.",
                    colour=_GREY,
                )
            ],
        )

    # Staleness thresholds (seconds since last message). clob_user is
    # event-driven — messages arrive only on our own order / fill events,
    # so hours of silence are normal during quiet sessions. Threshold
    # kept well above the 5-min window cadence so a genuinely dead
    # channel still trips eventually, but routine "no fills this hour"
    # does not.
    _STALE_THRESHOLDS = {  # noqa: N806  # constant defined in function scope
        "binance": 15,
        "gamma_rest": 600,
        "clob_rest": 600,
        "rtds": 120,
        "clob_market": 30,
        "clob_user": 3600,
    }

    def _status(name: str, samples: int, ago: float) -> str:
        if samples == 0:
            return "DEAD"
        return "STALE" if ago > _STALE_THRESHOLDS.get(name, 60) else "OK"

    # Determine overall health colour
    has_problem = False
    has_dead = False
    for s in feed_stats:
        st = _status(s["name"], s.get("samples", 0), s.get("last_message_ago_s", 0))
        if st == "DEAD":
            has_dead = True
        elif st == "STALE":
            has_problem = True

    if has_dead:
        colour = _RED
    elif has_problem:
        colour = _GOLD
    else:
        colour = _GREEN

    # Build compact table: one line per feed
    lines = []
    for s in feed_stats:
        name = s["name"]
        samples = s.get("samples", 0)
        ago = s.get("last_message_ago_s", 0)
        st = _status(name, samples, ago)

        if samples == 0:
            lines.append(f"\u274c **{name}** — no data")
            continue

        icon = "\u26a0\ufe0f" if st == "STALE" else "\u2705"
        med = s["median_ms"]
        p95 = s["p95_ms"]
        lines.append(
            f"{icon} **{name}** — `{med:.0f}ms` med \u00b7 `{p95:.0f}ms` p95 \u00b7 {samples} samples"
        )

    return _send(
        url,
        [
            _embed(
                title=f"{tag} Latency (1h)",
                description="\n".join(lines),
                colour=colour,
            )
        ],
    )
