"""Scores keyed by exact model/runtime settings and normalized text hash."""
from contextlib import AbstractContextManager
import json
from pathlib import Path
import sqlite3

from .contracts import ContractError
from .finbert import normalize_text, validate_prediction
from .store import digest


class SentimentCache(AbstractContextManager):
    def __init__(self, path: Path, synthetic: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS scores (
                model_signature TEXT NOT NULL, text_hash TEXT NOT NULL, prediction TEXT NOT NULL,
                PRIMARY KEY(model_signature, text_hash));
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        value = json.dumps(synthetic)
        existing = self.db.execute("SELECT value FROM metadata WHERE key='synthetic'").fetchone()
        if existing and existing[0] != value:
            self.db.close()
            raise ContractError('Sentiment cache is bound to a different synthetic mode; use a separate cache')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('synthetic',?)", (value,))
        self.db.commit()

    def __exit__(self, *args):
        self.db.close()

    def get(self, model_signature: str, text: str) -> dict | None:
        row = self.db.execute('SELECT prediction FROM scores WHERE model_signature=? AND text_hash=?',
                              (model_signature, digest(normalize_text(text)))).fetchone()
        if row is None:
            return None
        try:
            return validate_prediction(json.loads(row[0]))
        except (ValueError, TypeError) as exc:
            raise ContractError('Cached sentiment prediction is invalid; rebuild the derived score cache') from exc

    def put_batch(self, model_signature: str, rows: list[tuple[str, dict]]):
        # Validate the complete batch, then commit it atomically. Interrupted runs reuse it.
        normalized = [(model_signature, digest(normalize_text(text)), json.dumps(validate_prediction(result), sort_keys=True))
                      for text, result in rows]
        with self.db:
            self.db.executemany('INSERT OR REPLACE INTO scores VALUES (?,?,?)', normalized)

    def clear(self) -> int:
        with self.db:
            count = self.db.execute('SELECT count(*) FROM scores').fetchone()[0]
            self.db.execute('DELETE FROM scores')
        return count


def score_unique(texts: list[str], scorer, cache: SentimentCache, batch_size: int) -> tuple[dict, dict]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ContractError('Scoring batch size must be a positive integer')
    unique = sorted({normalize_text(text) for text in texts})
    scored, pending = {}, []
    for text in unique:
        value = cache.get(scorer.signature, text)
        if value is None:
            pending.append(text)
        else:
            scored[text] = value
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        predictions = scorer.predict(batch)
        if len(predictions) != len(batch):
            raise ContractError('Scorer returned the wrong number of predictions')
        results = [(text, validate_prediction(row)) for text, row in zip(batch, predictions, strict=True)]
        cache.put_batch(scorer.signature, results)
        scored.update(results)
    return scored, {'unique_texts': len(unique), 'cache_hits': len(unique) - len(pending),
                    'inferred_texts': len(pending), 'truncated_unique_texts': sum(r['truncated'] for r in scored.values())}
