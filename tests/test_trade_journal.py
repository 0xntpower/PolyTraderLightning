"""Tests for the append-only trade journal — JSONL persistence and read-back."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from shared.trade_journal import TradeJournal, TradeRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    signal_id: str = "up_240.0_180.0_0.05_0.1",
    source: str = "paper",
    fired: bool = True,
    won: bool | None = True,
    timestamp: str = "2026-04-14T12:00:00+00:00",
    window_ts: int = 1_700_000_000,
    signal_side: str = "up",
) -> TradeRecord:
    return TradeRecord(
        timestamp=timestamp,
        signal_id=signal_id,
        signal_side=signal_side,
        window_ts=window_ts,
        fired=fired,
        filled=True,
        won=won,
        entry_price=0.85,
        pnl=0.15 if won else -0.85,
        source=source,
        signal_age_windows=1,
    )


# ---------------------------------------------------------------------------
# TradeJournal — JSONL persistence
# ---------------------------------------------------------------------------


class TestTradeJournalPersistence:
    def test_record_trade_writes_jsonl_line(self, tmp_path: Path) -> None:
        journal = TradeJournal(tmp_path / "journal.jsonl")
        rec = _make_record()
        assert journal.record_trade(rec) is True
        lines = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["signal_id"] == rec.signal_id
        assert parsed["source"] == rec.source
        assert parsed["won"] is True

    def test_record_trade_creates_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dir" / "journal.jsonl"
        journal = TradeJournal(target)
        assert journal.record_trade(_make_record()) is True
        assert target.exists()

    def test_record_trade_appends_without_overwriting(self, tmp_path: Path) -> None:
        journal = TradeJournal(tmp_path / "journal.jsonl")
        journal.record_trade(_make_record(signal_id="sig_a"))
        journal.record_trade(_make_record(signal_id="sig_b"))
        lines = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_read_trades_round_trip(self, tmp_path: Path) -> None:
        journal = TradeJournal(tmp_path / "journal.jsonl")
        journal.record_trade(_make_record(signal_id="sig_a", source="paper"))
        journal.record_trade(_make_record(signal_id="sig_b", source="live"))

        all_records = journal.read_trades()
        assert len(all_records) == 2
        assert {r.signal_id for r in all_records} == {"sig_a", "sig_b"}

    def test_read_trades_filters_by_signal_id(self, tmp_path: Path) -> None:
        journal = TradeJournal(tmp_path / "journal.jsonl")
        journal.record_trade(_make_record(signal_id="sig_a"))
        journal.record_trade(_make_record(signal_id="sig_b"))

        filtered = journal.read_trades(signal_id="sig_a")
        assert len(filtered) == 1
        assert filtered[0].signal_id == "sig_a"

    def test_read_trades_filters_by_source(self, tmp_path: Path) -> None:
        journal = TradeJournal(tmp_path / "journal.jsonl")
        journal.record_trade(_make_record(source="paper"))
        journal.record_trade(_make_record(source="live"))
        journal.record_trade(_make_record(source="shadow"))

        assert len(journal.read_trades(source="live")) == 1
        assert len(journal.read_trades(source="paper")) == 1

    def test_read_trades_filters_by_since(self, tmp_path: Path) -> None:
        journal = TradeJournal(tmp_path / "journal.jsonl")
        journal.record_trade(_make_record(signal_id="sig_a", timestamp="2026-01-15T12:00:00+00:00"))
        journal.record_trade(_make_record(signal_id="sig_b", timestamp="2026-01-16T12:00:00+00:00"))

        filtered = journal.read_trades(since="2026-01-16T00:00:00+00:00")
        assert len(filtered) == 1
        assert filtered[0].signal_id == "sig_b"

    def test_read_trades_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.jsonl"
        journal = TradeJournal(path)
        journal.record_trade(_make_record(signal_id="sig_a"))
        # Corrupt the file with garbage in between.
        with open(path, "a", encoding="utf-8") as f:
            f.write("this is not json\n")
            f.write("\n")
            f.write('{"incomplete": true}\n')
        journal.record_trade(_make_record(signal_id="sig_b"))

        records = journal.read_trades()
        ids = {r.signal_id for r in records}
        assert ids == {"sig_a", "sig_b"}

    def test_read_trades_on_missing_file_returns_empty(self, tmp_path: Path) -> None:
        journal = TradeJournal(tmp_path / "does_not_exist.jsonl")
        assert journal.read_trades() == []

    def test_read_trades_skips_line_with_unexpected_field(self, tmp_path: Path) -> None:
        """A JSON line with a field TradeRecord doesn't accept must be skipped, not crash."""
        path = tmp_path / "journal.jsonl"
        journal = TradeJournal(path)
        journal.record_trade(_make_record(signal_id="sig_a"))

        bad = json.dumps(
            {
                "timestamp": "2026-04-14T12:00:00+00:00",
                "signal_id": "sig_future",
                "signal_side": "up",
                "window_ts": 99,
                "fired": True,
                "filled": True,
                "won": True,
                "entry_price": 0.55,
                "pnl": 0.45,
                "source": "paper",
                "signal_age_windows": 5,
                "extra_field_from_future": "surprise",
            }
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(bad + "\n")

        journal.record_trade(_make_record(signal_id="sig_b"))

        records = journal.read_trades()
        ids = {r.signal_id for r in records}
        assert ids == {"sig_a", "sig_b"}


# ---------------------------------------------------------------------------
# now_iso helper
# ---------------------------------------------------------------------------


def test_now_iso_returns_utc_iso_string() -> None:
    from datetime import datetime

    ts = TradeJournal.now_iso()
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
