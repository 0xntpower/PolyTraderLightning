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


class TradeJournal:
    """Persistent JSONL journal for trade outcomes."""

    def __init__(self, path: Path | str = _DEFAULT_JOURNAL_PATH) -> None:
        self._path = Path(path)

    def record_trade(self, record: TradeRecord) -> bool:
        """Append a single trade record. Creates the directory on first write.

        Returns True on success, False on failure.
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
