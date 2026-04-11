#!/usr/bin/env python3
"""Analyze paper trading results from paper_results/*.jsonl.

Usage:
    python scripts/analyze_paper.py                   # all days
    python scripts/analyze_paper.py 2026-03-27        # single day
    python scripts/analyze_paper.py 2026-03-25 2026-03-27  # date range (inclusive)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "src" / "paper_results"


def load_records(start_date: str | None = None, end_date: str | None = None) -> list[dict]:
    records = []
    for f in sorted(RESULTS_DIR.glob("*.jsonl")):
        date_str = f.stem
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def _pct(num: int, denom: int) -> str:
    return f"{num / denom * 100:5.1f}%" if denom else "  n/a"


def _fmt(value: float) -> str:
    return f"{'+' if value >= 0 else ''}{value:.4f}"


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def analyze_overall(records: list[dict]) -> None:
    total = len(records)
    traded = [r for r in records if r.get("rule_simulated_fill")]
    wins = [r for r in traded if r.get("pnl_total", 0) > 0]
    total_pnl = sum(r.get("pnl_total", 0) for r in traded)

    print(f"  Windows total:     {total}")
    print(f"  Windows traded:    {len(traded)} ({_pct(len(traded), total)})")
    print(f"  Win rate:          {_pct(len(wins), len(traded))}")
    print(f"  Total P&L:         {_fmt(total_pnl)}")
    if traded:
        print(f"  Avg P&L / trade:   {_fmt(total_pnl / len(traded))}")

    last_balance = next(
        (r.get("balance_usd") for r in reversed(records) if r.get("balance_usd") is not None),
        None,
    )
    if last_balance is not None:
        print(f"  Current balance:   ${last_balance:.2f}")

    latencies = [r["latency_signal_ms"] for r in records if r.get("latency_signal_ms") is not None]
    if latencies:
        print(f"  Signal latency:    avg={sum(latencies)/len(latencies):.1f}ms  "
              f"max={max(latencies):.1f}ms  (n={len(latencies)})")


def analyze_rules(records: list[dict]) -> None:
    rule_stats: dict[int, dict] = defaultdict(lambda: {
        "triggered": 0, "filled": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "pnl_values": [],
    })

    for r in records:
        rule_id = r.get("rule_triggered")
        if rule_id is None:
            continue
        s = rule_stats[rule_id]
        s["triggered"] += 1
        if r.get("rule_simulated_fill"):
            s["filled"] += 1
            pnl = r.get("pnl_rules", 0.0)
            s["pnl_values"].append(pnl)
            s["total_pnl"] += pnl
            if pnl > 0:
                s["wins"] += 1
            elif pnl < 0:
                s["losses"] += 1

    if not rule_stats:
        print("  No rule trades recorded yet.")
        return

    total_pnl = sum(s["total_pnl"] for s in rule_stats.values())

    print(f"  {'Rule':>6}  {'Triggered':>9}  {'Filled':>6}  {'WinRate':>8}  "
          f"{'TotalPnL':>10}  {'AvgPnL':>8}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}")
    for rule_id in sorted(rule_stats):
        s = rule_stats[rule_id]
        avg = s["total_pnl"] / s["filled"] if s["filled"] else 0.0
        print(f"  #{rule_id:>4}   {s['triggered']:>9}  {s['filled']:>6}  "
              f"{_pct(s['wins'], s['filled']):>8}  "
              f"{_fmt(s['total_pnl']):>10}  {_fmt(avg):>8}")

    total_triggered = sum(s["triggered"] for s in rule_stats.values())
    total_filled = sum(s["filled"] for s in rule_stats.values())
    print(f"  {'TOTAL':>6}  {total_triggered:>9}  {total_filled:>6}  "
          f"{'':>8}  {_fmt(total_pnl):>10}")

    # Feature distributions for filled trades
    print()
    for rule_id in sorted(rule_stats):
        s = rule_stats[rule_id]
        if not s["filled"]:
            continue
        filled_recs = [r for r in records
                       if r.get("rule_triggered") == rule_id and r.get("rule_simulated_fill")]
        features = [r["rule_signal_features"] for r in filled_recs if r.get("rule_signal_features")]
        if not features:
            continue
        print(f"  Rule #{rule_id} — feature ranges at entry (n={len(features)}):")
        for key in features[0]:
            vals = [f[key] for f in features if key in f]
            if vals:
                print(f"    {key:<32}: [{min(vals):.4f} .. {max(vals):.4f}]  "
                      f"mean={sum(vals)/len(vals):.4f}")


def main() -> None:
    args = sys.argv[1:]
    start_date = args[0] if len(args) >= 1 else None
    end_date = args[1] if len(args) >= 2 else (start_date if start_date else None)

    if start_date and end_date and start_date == end_date:
        date_label = start_date
    elif start_date or end_date:
        date_label = f"{start_date or 'all'} → {end_date or 'all'}"
    else:
        date_label = "all dates"

    records = load_records(start_date, end_date)
    if not records:
        print(f"No records found in {RESULTS_DIR} for {date_label}")
        sys.exit(0)

    print(f"\nPaper trading analysis — {date_label} ({len(records)} windows)")
    print_section("OVERALL")
    analyze_overall(records)
    print_section("BACKTEST RULES")
    analyze_rules(records)
    print()


if __name__ == "__main__":
    main()
