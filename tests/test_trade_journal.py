"""Tests for the append-only trade journal and the RecentFireMailbox.

Covers the v3.0 P5 feedback path plumbing: every resolved fire must be
persisted to the JSONL journal AND published into the in-memory mailbox
so the IPC status_query handler can answer without rereading the file.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from shared.trade_journal import (
    RecentFire,
    RecentFireMailbox,
    TradeJournal,
    TradeRecord,
)

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
# RecentFireMailbox
# ---------------------------------------------------------------------------


class TestRecentFireMailbox:
    def test_empty_snapshot_returns_empty_list(self) -> None:
        mbox = RecentFireMailbox()
        assert mbox.snapshot(source="paper", limit=10) == []

    def test_record_then_snapshot_returns_matching_fire(self) -> None:
        mbox = RecentFireMailbox()
        fire = RecentFire(
            signal_id="up_240_180",
            source="paper",
            won=True,
            timestamp="2026-04-14T12:00:00+00:00",
        )
        mbox.record(fire)
        snap = mbox.snapshot(source="paper", limit=10)
        assert snap == [fire]

    def test_snapshot_filters_by_source(self) -> None:
        mbox = RecentFireMailbox()
        mbox.record(RecentFire("sig_a", "paper", True, "2026-04-14T12:00:00+00:00"))
        mbox.record(RecentFire("sig_b", "live", False, "2026-04-14T12:01:00+00:00"))
        mbox.record(RecentFire("sig_c", "paper", False, "2026-04-14T12:02:00+00:00"))

        paper = mbox.snapshot(source="paper", limit=10)
        live = mbox.snapshot(source="live", limit=10)

        assert [f.signal_id for f in paper] == ["sig_a", "sig_c"]
        assert [f.signal_id for f in live] == ["sig_b"]

    def test_snapshot_respects_limit(self) -> None:
        mbox = RecentFireMailbox()
        for i in range(10):
            mbox.record(RecentFire(f"sig_{i}", "paper", True, f"t{i}"))

        snap = mbox.snapshot(source="paper", limit=3)
        # Returns the most recent `limit` — the last 3 appended.
        assert [f.signal_id for f in snap] == ["sig_7", "sig_8", "sig_9"]

    def test_snapshot_limit_zero_returns_empty(self) -> None:
        mbox = RecentFireMailbox()
        mbox.record(RecentFire("sig_a", "paper", True, "t"))
        assert mbox.snapshot(source="paper", limit=0) == []

    def test_snapshot_negative_limit_returns_empty(self) -> None:
        mbox = RecentFireMailbox()
        mbox.record(RecentFire("sig_a", "paper", True, "t"))
        assert mbox.snapshot(source="paper", limit=-5) == []

    def test_snapshot_unknown_source_returns_empty(self) -> None:
        mbox = RecentFireMailbox()
        mbox.record(RecentFire("sig_a", "paper", True, "t"))
        assert mbox.snapshot(source="shadow", limit=10) == []

    def test_bounded_maxlen_evicts_oldest(self) -> None:
        mbox = RecentFireMailbox(maxlen=3)
        for i in range(5):
            mbox.record(RecentFire(f"sig_{i}", "paper", True, f"t{i}"))
        snap = mbox.snapshot(source="paper", limit=10)
        # Only the last 3 survive.
        assert [f.signal_id for f in snap] == ["sig_2", "sig_3", "sig_4"]

    def test_snapshot_is_a_copy_not_a_view(self) -> None:
        """Mutating the returned list must not affect mailbox state."""
        mbox = RecentFireMailbox()
        mbox.record(RecentFire("sig_a", "paper", True, "t"))
        snap = mbox.snapshot(source="paper", limit=10)
        snap.clear()
        # Second snapshot should still see the original record.
        assert len(mbox.snapshot(source="paper", limit=10)) == 1

    def test_concurrent_record_and_snapshot_is_safe(self) -> None:
        """Hammer the mailbox from multiple threads and ensure no exceptions
        escape and no records are lost beyond the bounded maxlen."""
        mbox = RecentFireMailbox(maxlen=256)
        errors: list[BaseException] = []

        def writer(tag: str) -> None:
            try:
                for i in range(100):
                    mbox.record(RecentFire(f"{tag}_{i}", "paper", True, "t"))
            except BaseException as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(100):
                    mbox.snapshot(source="paper", limit=50)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=("A",)),
            threading.Thread(target=writer, args=("B",)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert errors == []
        # At most 256 entries, all from the writers.
        final = mbox.snapshot(source="paper", limit=1000)
        assert 0 < len(final) <= 256
        assert all(f.signal_id.startswith(("A_", "B_")) for f in final)


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


# ---------------------------------------------------------------------------
# TradeJournal ↔ RecentFireMailbox integration
# ---------------------------------------------------------------------------


class TestTradeJournalMailboxIntegration:
    def test_fired_and_resolved_record_populates_mailbox(self, tmp_path: Path) -> None:
        mbox = RecentFireMailbox()
        journal = TradeJournal(tmp_path / "journal.jsonl", fire_mailbox=mbox)

        journal.record_trade(
            _make_record(
                signal_id="up_240_180",
                source="paper",
                fired=True,
                won=True,
                timestamp="2026-04-14T12:00:00+00:00",
            )
        )

        snap = mbox.snapshot(source="paper", limit=10)
        assert len(snap) == 1
        assert snap[0].signal_id == "up_240_180"
        assert snap[0].source == "paper"
        assert snap[0].won is True
        assert snap[0].timestamp == "2026-04-14T12:00:00+00:00"

    def test_unfired_record_does_not_populate_mailbox(self, tmp_path: Path) -> None:
        mbox = RecentFireMailbox()
        journal = TradeJournal(tmp_path / "journal.jsonl", fire_mailbox=mbox)
        # fired=False with won=None — an observation of a window that
        # didn't trigger an order; must not leak into the fire mailbox.
        journal.record_trade(_make_record(fired=False, won=None))
        assert mbox.snapshot(source="paper", limit=10) == []

    def test_unresolved_fire_does_not_populate_mailbox(self, tmp_path: Path) -> None:
        """A fire whose outcome is still unknown (won=None) must not leak
        into the mailbox — the feedback loop requires resolved outcomes."""
        mbox = RecentFireMailbox()
        journal = TradeJournal(tmp_path / "journal.jsonl", fire_mailbox=mbox)
        journal.record_trade(_make_record(fired=True, won=None))
        assert mbox.snapshot(source="paper", limit=10) == []

    def test_mailbox_and_file_stay_consistent_across_many_writes(self, tmp_path: Path) -> None:
        mbox = RecentFireMailbox(maxlen=100)
        journal = TradeJournal(tmp_path / "journal.jsonl", fire_mailbox=mbox)

        for i in range(10):
            journal.record_trade(
                _make_record(
                    signal_id=f"sig_{i}",
                    source="live",
                    fired=True,
                    won=(i % 2 == 0),
                )
            )

        # File has everything.
        assert len(journal.read_trades()) == 10
        # Mailbox has the resolved live fires only (all 10 are resolved here).
        snap = mbox.snapshot(source="live", limit=100)
        assert len(snap) == 10
        assert [f.signal_id for f in snap] == [f"sig_{i}" for i in range(10)]

    def test_mailbox_respects_source_separation(self, tmp_path: Path) -> None:
        """Paper fires must not surface to a 'live' snapshot and vice versa."""
        mbox = RecentFireMailbox()
        journal = TradeJournal(tmp_path / "journal.jsonl", fire_mailbox=mbox)

        journal.record_trade(_make_record(signal_id="p_1", source="paper", won=True))
        journal.record_trade(_make_record(signal_id="l_1", source="live", won=False))
        journal.record_trade(_make_record(signal_id="p_2", source="paper", won=False))

        paper = mbox.snapshot(source="paper", limit=10)
        live = mbox.snapshot(source="live", limit=10)
        assert [f.signal_id for f in paper] == ["p_1", "p_2"]
        assert [f.signal_id for f in live] == ["l_1"]

    def test_mailbox_is_optional(self, tmp_path: Path) -> None:
        """A journal without a mailbox still writes to disk normally."""
        journal = TradeJournal(tmp_path / "journal.jsonl")  # no mailbox
        assert journal.record_trade(_make_record()) is True
        assert len(journal.read_trades()) == 1


# ---------------------------------------------------------------------------
# now_iso helper
# ---------------------------------------------------------------------------


def test_now_iso_returns_utc_iso_string() -> None:
    from datetime import datetime

    ts = TradeJournal.now_iso()
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    utc_offset = parsed.utcoffset()
    assert utc_offset is not None
    assert utc_offset.total_seconds() == 0
