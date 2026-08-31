"""Append-only SQLite event bus. Point-in-time safe by construction.

Every row carries:
  ts          — event time as claimed by the SOURCE (used by backtests)
  ingested_at — wall-clock time we stored it (audit/freshness)
Backtests must filter on ts <= as_of, never ingested_at.
Dedup: UNIQUE(source, dedup_key) — collectors set a stable dedup_key per fact.
"""
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import ROOT

DB_PATH = Path(ROOT) / "data" / "events.db"


@dataclass
class Event:
    source: str            # e.g. "agentservices", "lse", "binance"
    event_type: str        # e.g. "price", "fear_greed", "news", "insider_trade"
    ts: float              # event time (unix seconds, UTC)
    payload: dict          # arbitrary JSON
    dedup_key: str         # stable id for this fact, e.g. "fear_greed:2026-08-31"
    symbols: list = field(default_factory=list)   # optional ticker tags for indexing

    def to_row(self):
        return (
            self.source, self.event_type, self.ts, time.time(),
            json.dumps(self.payload, separators=(",", ":")),
            self.dedup_key, json.dumps(self.symbols),
        )


class EventBus:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY,
              source TEXT NOT NULL,
              event_type TEXT NOT NULL,
              ts REAL NOT NULL,
              ingested_at REAL NOT NULL,
              payload TEXT NOT NULL,
              dedup_key TEXT NOT NULL,
              symbols TEXT NOT NULL DEFAULT '[]'
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_dedup
              ON events(source, dedup_key);
            CREATE INDEX IF NOT EXISTS ix_type_ts ON events(event_type, ts);
            CREATE INDEX IF NOT EXISTS ix_source_ts ON events(source, ts);
            """
        )
        self.conn.commit()

    def publish(self, events) -> int:
        """Insert events, skipping duplicates. Returns count of NEW rows."""
        rows = [e.to_row() for e in events]
        before = self._count()
        self.conn.executemany(
            "INSERT OR IGNORE INTO events"
            " (source, event_type, ts, ingested_at, payload, dedup_key, symbols)"
            " VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return self._count() - before

    def read(self, event_type: str | None = None, source: str | None = None,
             symbol: str | None = None, as_of: float | None = None,
             limit: int = 500) -> list[Event]:
        """Point-in-time read: only events with ts <= as_of."""
        q = "SELECT source, event_type, ts, payload, dedup_key, symbols FROM events WHERE 1=1"
        args: list = []
        if event_type:
            q += " AND event_type=?"; args.append(event_type)
        if source:
            q += " AND source=?"; args.append(source)
        if symbol:
            q += " AND symbols LIKE ?"; args.append(f'%"{symbol}"%')
        if as_of is not None:
            q += " AND ts<=?"; args.append(as_of)
        q += " ORDER BY ts DESC LIMIT ?"; args.append(limit)
        out = []
        for src, et, ts, pl, dk, syms in self.conn.execute(q, args):
            out.append(Event(src, et, ts, json.loads(pl), dk, json.loads(syms)))
        return out

    def stats(self) -> list[tuple]:
        return list(self.conn.execute(
            "SELECT source, event_type, COUNT(*), MAX(ts) FROM events"
            " GROUP BY source, event_type ORDER BY source, event_type"))

    def _count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def close(self):
        self.conn.close()
