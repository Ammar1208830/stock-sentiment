"""Persistent checkpoints, request accounting and exact content versions."""
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .contracts import ContractError


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class BudgetExceeded(ContractError):
    pass


class Store(AbstractContextManager):
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY, request_key TEXT NOT NULL, source TEXT NOT NULL,
                parameters TEXT NOT NULL, payload TEXT NOT NULL, retrieved_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS snapshot_request ON snapshots(request_key, retrieved_at);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, source TEXT NOT NULL, day TEXT NOT NULL,
                request_key TEXT NOT NULL, status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS texts (
                id TEXT NOT NULL, source TEXT NOT NULL, ticker TEXT NOT NULL,
                origin_id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(id, ticker));
            CREATE TABLE IF NOT EXISTS imports (
                id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS deletions (
                source TEXT, origin_hash TEXT, deleted_at TEXT, records INTEGER);
        ''')
        self.db.commit()

    def __exit__(self, *args):
        self.db.close()

    def bind(self, config: dict, synthetic: bool = False) -> None:
        # Never mix synthetic and unknown/real source caches or universes.
        identity = json.dumps({"tickers": list(config["tickers"]), "synthetic": synthetic}, sort_keys=True)
        row = self.db.execute("SELECT value FROM settings WHERE key='identity'").fetchone()
        if row and row[0] != identity:
            raise ContractError("Cache belongs to a different universe or synthetic mode; use a separate cache")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO settings VALUES ('identity', ?)", (identity,))

    def reserve(self, source: str, key: str, limit: int, now: datetime) -> int:
        day = now.date().isoformat()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            count = self.db.execute("SELECT count(*) FROM attempts WHERE source=? AND day=?", (source, day)).fetchone()[0]
            if count >= limit:
                raise BudgetExceeded(f"{source} daily request/call budget exhausted; cached progress is preserved")
            cursor = self.db.execute("INSERT INTO attempts(source,day,request_key,status) VALUES (?,?,?,'started')",
                                     (source, day, key))
            self.db.commit()
            return cursor.lastrowid
        except Exception:
            self.db.rollback()
            raise

    def finish_attempt(self, attempt: int, status: str):
        with self.db:
            self.db.execute("UPDATE attempts SET status=? WHERE id=?", (status, attempt))

    def latest(self, key: str) -> dict | None:
        row = self.db.execute("SELECT * FROM snapshots WHERE request_key=? ORDER BY retrieved_at DESC, rowid DESC LIMIT 1", (key,)).fetchone()
        return dict(row) if row else None

    def save_snapshot(self, source: str, key: str, parameters: dict, payload: object, retrieved_at: str) -> dict:
        snapshot_id = digest([source, key, payload, retrieved_at])
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?,?)",
                            (snapshot_id, key, source, json.dumps(parameters, sort_keys=True),
                             json.dumps(payload, sort_keys=True, allow_nan=False), retrieved_at))
        return dict(self.db.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone())

    def put_text(self, row: dict) -> None:
        existing = self.db.execute("SELECT payload FROM texts WHERE id=? AND ticker=?", (row['record_id'], row['ticker'])).fetchone()
        if existing:
            previous = json.loads(existing[0])
            row = {**row, "available_at": min(previous['available_at'], row['available_at']),
                   "retrieved_at": max(previous['retrieved_at'], row['retrieved_at']),
                   "first_snapshot_id": previous.get('first_snapshot_id') if previous['available_at'] <= row['available_at'] else row.get('first_snapshot_id'),
                   "snapshot_id": previous.get('snapshot_id') if previous['retrieved_at'] > row['retrieved_at'] else row.get('snapshot_id')}
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO texts VALUES (?,?,?,?,?)",
                (row['record_id'], row['source'], row['ticker'], row['origin_id'], json.dumps(row, sort_keys=True)))

    def texts(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT payload FROM texts ORDER BY source,ticker,id")]

    def add_import(self, metadata: dict):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO imports VALUES (?,?)", (digest(metadata), json.dumps(metadata, sort_keys=True)))

    def imports(self) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT metadata FROM imports ORDER BY id")]

    def usage(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT source,day,status,count(*) AS attempts FROM attempts GROUP BY source,day,status ORDER BY day,source,status")]

    def purge(self, source: str, origin_id: str) -> int:
        with self.db:
            cursor = self.db.execute("DELETE FROM texts WHERE source=? AND origin_id=?", (source, origin_id))
            count = cursor.rowcount
            self.db.execute("INSERT INTO deletions VALUES (?,?,?,?)", (source, digest(origin_id), utc_now().isoformat(), count))
        return count
