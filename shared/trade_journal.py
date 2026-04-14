"""Append-only trade outcome journal for signal decay analysis.

Records every window evaluation (fired or not) as a JSONL file. Both the
live bot and shadow paper tracker write here. The decay detector reads
from the same file concurrently — safe because writes are append-only
with flush-after-write on a single writer.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_JOURNAL_PATH = Path("data/paper/journal.jsonl")


@dataclass(frozen=True, slots=True)
class TradeRecord:
    timestamp: str
    signal_id: str
    signal_side: str
    window_ts: int
    fired: bool
    filled: bool
    won: bool | None
    entry_price: float
    pnl: float
    source: str
    signal_age_windows: int


@dataclass(frozen=True, slots=True)
class RecentFire:
    signal_id: str
    source: str
    won: bool
    timestamp: str


class RecentFireMailbox:
    """Thread-safe ring buffer of recently resolved fires.

    Mirrors the fire-and-forget pattern used by ``StatePublisher`` and the
    Discord sender: the strategy loop pushes every resolved fire into this
    in-memory buffer, and the IPC ``status_query`` handler (which runs on
    a background thread) reads a snapshot without touching the disk.

    The bounded deque never allocates beyond ``maxlen`` and is guarded by
    a single lock, so ``record`` and ``snapshot`` are O(maxlen) and cannot
    block the asyncio hot path on file I/O.
    """

    def __init__(self, maxlen: int = 64) -> None:
        self._deque: deque[RecentFire] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, fire: RecentFire) -> None:
        with self._lock:
            self._deque.append(fire)

    def snapshot(self, *, source: str, limit: int) -> list[RecentFire]:
        if limit <= 0:
            return []
        with self._lock:
            items = list(self._deque)
        matching = [f for f in items if f.source == source]
        return matching[-limit:]


class TradeJournal:
    """Persistent JSONL journal for trade outcomes."""

    def __init__(
        self,
        path: Path | str = _DEFAULT_JOURNAL_PATH,
        *,
        fire_mailbox: RecentFireMailbox | None = None,
    ) -> None:
        self._path = Path(path)
        self._fire_mailbox = fire_mailbox

    def record_trade(self, record: TradeRecord) -> bool:
        """Append a single trade record. Creates the directory on first write.

        Returns True on success, False on failure. When a ``RecentFireMailbox``
        was supplied to ``__init__`` and the record represents a resolved fire
        (``fired=True`` and ``won is not None``), the in-memory mailbox is
        updated so the IPC status handler can answer queries without rereading
        the file from disk.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            log.exception("trade journal write failed")
            return False

        if self._fire_mailbox is not None and record.fired and record.won is not None:
            self._fire_mailbox.record(
                RecentFire(
                    signal_id=record.signal_id,
                    source=record.source,
                    won=record.won,
                    timestamp=record.timestamp,
                )
            )
        return True

    def read_trades(
        self,
        *,
        signal_id: str | None = None,
        source: str | None = None,
        since: str | None = None,
    ) -> list[TradeRecord]:
        """Read records with optional filtering.

        Args:
            signal_id: Filter by signal_id.
            source: Filter by source ("live", "paper", "shadow").
            since: ISO8601 timestamp — only return records at or after this time.
        """
        if not self._path.exists():
            return []

        records: list[TradeRecord] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if signal_id and data.get("signal_id") != signal_id:
                        continue
                    if source and data.get("source") != source:
                        continue
                    if since and data.get("timestamp", "") < since:
                        continue

                    try:
                        records.append(TradeRecord(**data))
                    except (TypeError, KeyError) as exc:
                        log.warning("skipping malformed trade record: %s", exc)
                        continue
        except OSError:
            log.exception("trade journal read failed")

        return records

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat()
