"""Conservative entity mapping and source/version provenance."""
from collections import Counter
from dataclasses import asdict
from datetime import timezone
import json
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import Config
from .contracts import ContractError, TextRecord, nonempty
from .io import file_hash
from .store import Store, digest, utc_now


ALIASES = {"AAPL": ("Apple",), "MSFT": ("Microsoft",), "NVDA": ("Nvidia",),
           "AMZN": ("Amazon",), "TSLA": ("Tesla",)}


def canonical_url(value: str) -> str:
    parsed = urlsplit(nonempty(value, "url"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ContractError("Source URL must be an http(s) URL without credentials")
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in {'fbclid', 'gclid'}]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or '/', urlencode(sorted(query)), ''))


def match_tickers(text: str, universe: tuple[str, ...]) -> list[str]:
    # Bare symbols must be uppercase; company names need context for ambiguous words.
    found = []
    for symbol in universe:
        explicit = re.search(r'(?<![\w])\$?' + re.escape(symbol) + r'(?![\w])', text)
        alias = any(re.search(r'\b' + re.escape(name) + r'\b', text, re.I) for name in ALIASES.get(symbol, ()))
        finance_context = re.search(r'\b(stock|shares|earnings|revenue|company|nasdaq|ipo|dividend|inc|corp|ceo|iphone|aws|gpu)\b', text, re.I)
        if explicit or (alias and finance_context):
            found.append(symbol)
    return found


def normalized_version(row: dict, origin_id: str, url: str | None = None, snapshot_id: str | None = None) -> dict:
    parsed = TextRecord.from_dict(row)
    result = asdict(parsed)
    for key in ('published_at', 'available_at', 'retrieved_at'):
        result[key] = result[key].astimezone(timezone.utc).isoformat()
    result['text'] = ' '.join(result['text'].split())
    content_hash = digest([result['text'], result['published_at']])
    result.update(origin_id=origin_id, content_sha256=content_hash,
                  record_id=digest([parsed.source, origin_id, content_hash, parsed.availability_basis]),
                  url=canonical_url(url) if url else None, snapshot_id=snapshot_id,
                  first_snapshot_id=snapshot_id)
    return result


def normalize_news(payload: dict, retrieved_at: str, universe: tuple[str, ...], snapshot_id: str) -> tuple[list[dict], dict]:
    if not isinstance(payload, dict) or payload.get('status') != 'ok' or not isinstance(payload.get('articles'), list):
        raise ContractError("NewsAPI response is not a successful article listing")
    result, counts = [], Counter()
    for article in payload['articles']:
        try:
            if not isinstance(article, dict):
                raise ContractError("Article must be an object")
            title = nonempty(article.get('title'), 'title')
            if title == '[Removed]':
                counts['removed'] += 1
                continue
            url = canonical_url(article.get('url'))
            # Title only: descriptions/full text can be revised without version timestamps.
            symbols = match_tickers(title, universe)
            if not symbols:
                counts['unmatched_or_ambiguous'] += 1
                continue
            for symbol in symbols:
                result.append(normalized_version({
                    'record_id': 'pending', 'source': 'newsapi', 'ticker': symbol, 'text': title,
                    'published_at': article['publishedAt'], 'available_at': retrieved_at,
                    'retrieved_at': retrieved_at, 'availability_basis': 'observed'}, url, url, snapshot_id))
        except (KeyError, ValueError, TypeError, AttributeError):
            counts['invalid_article'] += 1
    return result, dict(counts)


def deduplicate_export(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Dedup identical text per publication day/ticker/basis, with lineage retained."""
    selected, audit = [], []
    seen = {}
    for row in sorted(rows, key=lambda r: (r['available_at'], r['record_id'], r['ticker'])):
        # Keep observed and assumed records separate so an assumption cannot suppress evidence.
        key = (row['ticker'], ' '.join(row['text'].casefold().split()), row['published_at'][:10], row['availability_basis'])
        duplicate = seen.get(key)
        audit.append({k: row.get(k) for k in ('record_id', 'source', 'ticker', 'origin_id', 'url', 'content_sha256', 'snapshot_id', 'first_snapshot_id')} | {'duplicate_of': duplicate})
        if duplicate is None:
            seen[key] = row['record_id']
            selected.append(row)
    return sorted(selected, key=lambda r: (r['published_at'], r['ticker'], r['record_id'])), audit


def import_text(path: Path, provenance_path: Path, config: Config, store: Store) -> dict:
    try:
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise ContractError('Import provenance must be valid UTF-8 JSON') from exc
    if not isinstance(provenance, dict):
        raise ContractError("Import provenance must be an object")
    source = provenance.get('source')
    expected = {'newsapi': 'licensed_export', 'reddit': 'approved_research_export'}
    if not isinstance(source, str) or source not in expected or provenance.get('access_basis') != expected[source]:
        raise ContractError("Provide licensed_export provenance for newsapi or approved_research_export for reddit")
    nonempty(provenance.get('reference'), 'provenance reference')
    if provenance.get('synthetic_data') is not False:
        raise ContractError("Real-data imports require synthetic_data=false attestation; use m2-demo for fixtures")
    # Validate the whole batch before any records are committed.
    normalized = []
    for line, text in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not text.strip():
            continue
        try:
            raw = json.loads(text)
            parsed = TextRecord.from_dict(raw)
            if parsed.source != source or parsed.ticker not in config.tickers:
                raise ContractError("Import source/ticker is inconsistent with provenance/config")
            if parsed.retrieved_at > utc_now():
                raise ContractError("Import retrieval timestamp cannot be in the future")
            normalized.append(normalized_version(raw, nonempty(raw.get('origin_id', raw['record_id']), 'origin_id'), raw.get('url')))
        except (KeyError, ValueError, TypeError) as exc:
            raise ContractError(f"{path.name}:{line}: {exc}") from exc
    if not normalized:
        raise ContractError("Import has no text records")
    metadata = {"source": source, "access_basis": provenance['access_basis'],
                "reference": provenance['reference'], "synthetic_data": False,
                "input_sha256": file_hash(path), "provenance_sha256": file_hash(provenance_path),
                "imported_at": utc_now().isoformat(), "records": len(normalized),
                "verification": "user_attested_not_independently_authenticated"}
    # The cache stores only the normalized fields above, never author/profile data.
    for row in normalized:
        store.put_text(row)
    store.add_import(metadata)
    return metadata
