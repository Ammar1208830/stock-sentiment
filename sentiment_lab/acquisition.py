"""M2 orchestration: acquire permitted data and report whether a study is feasible."""
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from importlib.metadata import PackageNotFoundError, version
import csv
import json
import math
import os
from pathlib import Path
import platform
import tomllib

from . import __version__
from .calendar import generate_calendar
from .config import Config
from .contracts import ContractError, PriceBar, TextRecord, session_date
from .io import file_hash, write_json, write_jsonl
from .normalization import deduplicate_export
from .pipeline import chronological_split, text_windows
from .providers import NewsClient, collect_news, collect_prices
from .store import Store, digest, utc_now


@dataclass(frozen=True)
class AcquisitionConfig:
    schema_version: int = 1
    start: str = '2023-10-23'
    end: str = '2026-10-23'
    news_lookback_days: int = 28
    news_daily_request_budget: int = 90
    news_max_pages_per_day: int = 3
    news_page_size: int = 100
    http_attempts: int = 3
    http_timeout_seconds: int = 20
    price_availability: str = 'historical_assumption'
    price_latency_minutes: int = 1
    price_daily_call_budget: int = 20
    minimum_usable_sessions: int = 630
    minimum_test_sessions: int = 126
    minimum_source_coverage: float = .8

    @classmethod
    def load(cls, path: Path):
        with path.open('rb') as stream:
            raw = tomllib.load(stream)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ContractError(f'Unknown acquisition settings: {sorted(unknown)}')
        result = cls(**raw)
        result.validate()
        return result

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ContractError('Acquisition schema_version must be 1')
        if session_date(self.end) < session_date(self.start):
            raise ContractError('Acquisition end precedes start')
        limits = {'news_lookback_days': (1, 1825), 'news_daily_request_budget': (1, 100000),
                  'news_max_pages_per_day': (1, 100), 'news_page_size': (1, 100),
                  'http_attempts': (1, 5), 'http_timeout_seconds': (1, 60),
                  'price_latency_minutes': (1, 60), 'price_daily_call_budget': (1, 1000),
                  'minimum_usable_sessions': (3, 100000), 'minimum_test_sessions': (1, 100000)}
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ContractError(f'{key} must be an integer in [{low}, {high}]')
        if self.price_availability not in {'observed', 'historical_assumption'}:
            raise ContractError('price_availability must be observed or historical_assumption')
        value = self.minimum_source_coverage
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ContractError('minimum_source_coverage must be in (0, 1]')


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def feasibility(config, settings, sessions, price_rows, text_rows, collection, synthetic=False) -> tuple[dict, list[dict]]:
    """Coverage counts only cutoff-eligible, deduplicated text and usable price pairs."""
    buckets, excluded = text_windows(config, sessions, [TextRecord.from_dict(r) for r in text_rows])
    bars = {(r['ticker'], r['session']): PriceBar.from_dict(r) for r in price_rows}
    provisional, coverage = [], []
    closed = utc_now()
    for previous, target in zip(sessions, sessions[1:]):
        if target.close_at > closed:
            continue
        decision = target.open_at - timedelta(minutes=config.decision_minutes_before_open)
        for symbol in config.tickers:
            prior, outcome = bars.get((symbol, str(previous.date))), bars.get((symbol, str(target.date)))
            usable = bool(prior and outcome and prior.available_at < decision and outcome.split_ratio == 1)
            counts = Counter(r.source for r in buckets.get((symbol, target.date), []))
            coverage.append({'ticker': symbol, 'target_session': str(target.date), 'price_pair_usable': usable,
                             'newsapi_count': counts['newsapi'], 'reddit_count': counts['reddit']})
            if usable:
                provisional.append({'ticker': symbol, 'target_session': str(target.date),
                                    'decision_at': decision.isoformat(), 'label_available_at': outcome.available_at.isoformat()})
    partitioned, split = [], {}
    try:
        partitioned, split = chronological_split(provisional, config)
    except ContractError:
        pass
    kept = {(r['ticker'], r['target_session']): r['partition'] for r in partitioned}
    for row in coverage:
        row['partition'] = kept.get((row['ticker'], row['target_session']), '')
    usable_days = {symbol: {r['target_session'] for r in partitioned if r['ticker'] == symbol} for symbol in config.tickers}
    shared = set.intersection(*usable_days.values()) if usable_days else set()
    shared_test = {r['target_session'] for r in partitioned if r['partition'] == 'test'} & shared
    per_ticker = {}
    for symbol in config.tickers:
        rows = [r for r in coverage if r['ticker'] == symbol and r['partition']]
        per_ticker[symbol] = {'usable_sessions': len(rows), 'sources': {}}
        for source in ('newsapi', 'reddit'):
            count = sum(r[f'{source}_count'] > 0 for r in rows)
            partition_coverage = {}
            for partition in ('train', 'validation', 'test'):
                subset = [r for r in rows if r['partition'] == partition]
                partition_coverage[partition] = sum(r[f'{source}_count'] > 0 for r in subset) / len(subset) if subset else 0
            per_ticker[symbol]['sources'][source] = {'covered_sessions': count,
                 'coverage': count / len(rows) if rows else 0, 'partition_coverage': partition_coverage}
    enough_history = len(shared) >= settings.minimum_usable_sessions and len(shared_test) >= settings.minimum_test_sessions
    source_pass = {source: all(
        info['sources'][source]['coverage'] >= settings.minimum_source_coverage and
        all(v >= settings.minimum_source_coverage for v in info['sources'][source]['partition_coverage'].values())
        for info in per_ticker.values()) for source in ('newsapi', 'reddit')}
    reasons = []
    if synthetic:
        reasons.append('Synthetic data validates engineering only; it cannot establish predictive value.')
    if not enough_history:
        reasons.append(f'Need >= {settings.minimum_usable_sessions} shared usable sessions and >= {settings.minimum_test_sessions} shared test sessions.')
    for source, passed in source_pass.items():
        if not passed:
            reasons.append(f'{source} lacks required cutoff-eligible coverage for every ticker and partition.')
    if not synthetic and enough_history and all(source_pass.values()):
        decision = 'full_study_feasible'
    elif not synthetic and enough_history and source_pass['newsapi']:
        decision = 'reduced_news_only_study_feasible'
    else:
        decision = 'engineering_demonstration_only'
    return {'decision': decision, 'reasons': reasons, 'synthetic_data': synthetic if synthetic else None,
            'shared_usable_sessions': len(shared), 'shared_test_sessions': len(shared_test),
            'thresholds': {k: getattr(settings, k) for k in ('minimum_usable_sessions', 'minimum_test_sessions', 'minimum_source_coverage')},
            'split': split, 'per_ticker': per_ticker, 'source_coverage_pass': source_pass,
            'text_exclusions': dict(excluded), 'collection': collection,
            'availability_assumptions': {'prices': settings.price_availability,
                 'allow_historical_text_assumptions': config.allow_historical_assumptions},
            'interpretation': 'Feasibility is a data gate, not model performance or statistical power. Historical assumptions do not prove live tradability.'}, coverage


def export_dataset(output: Path, config, settings, sessions, calendar_metadata, prices, store, collection,
                   synthetic=False) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ContractError('Output directory must be empty; reuse the cache with a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    texts, audit = deduplicate_export(store.texts())
    gate, coverage = feasibility(config, settings, sessions, prices, texts, collection, synthetic)
    write_csv(output / 'sessions.csv', [{'session': str(s.date), 'open_at': s.open_at.isoformat(), 'close_at': s.close_at.isoformat()} for s in sessions], ['session', 'open_at', 'close_at'])
    write_csv(output / 'prices.csv', prices, ['ticker', 'session', 'open', 'high', 'low', 'close', 'volume', 'available_at', 'split_ratio', 'dividends', 'availability_basis', 'retrieved_at', 'snapshot_id'])
    write_jsonl(output / 'texts.jsonl', texts)
    write_jsonl(output / 'text_audit.jsonl', audit)
    write_csv(output / 'coverage.csv', coverage, ['ticker', 'target_session', 'partition', 'price_pair_usable', 'newsapi_count', 'reddit_count'])
    write_json(output / 'feasibility.json', gate)
    write_json(output / 'calendar.json', calendar_metadata)
    # Source IDs and short texts for a human matching audit; no author profiles.
    write_jsonl(output / 'matching_review.jsonl', [{k: r.get(k) for k in ('record_id', 'ticker', 'source', 'text', 'url')} for r in texts[:25]])
    card = f'''# M2 data card

Decision: **{gate['decision']}**

Target period: {settings.start} through {settings.end} (inclusive). Calendar includes the preceding session.

Prices: {len(prices)} bars. Text: {len(texts)} exported versions. Shared usable target sessions: {gate['shared_usable_sessions']}. Shared test sessions: {gate['shared_test_sessions']}.

Synthetic data: {synthetic}. For local/provider sources this is a declaration, not independent authentication.

Price availability basis: {settings.price_availability}. Historical price availability uses close + {settings.price_latency_minutes} minute(s), an explicit assumption. Yahoo vendor OHLC can contain retrospective split adjustments and revisions. It is not an authenticated archive of contemporaneous raw prices.

News collection timestamps are first observed access, never publication backdating. Historical text assumptions are {'enabled' if config.allow_historical_assumptions else 'disabled'}. Coverage counts only text passing the actual overnight cutoff and price-pair filters; raw mentions are not eligible observations.

The calendar is generated from exchange_calendars {calendar_metadata.get('version', 'unknown')}. Its completeness is verified against the pinned library, which is not a guarantee against all exchange changes.

Read feasibility.json for collection gaps and thresholds, coverage.csv for every ticker/date, text_audit.jsonl for duplicates/version lineage, and matching_review.jsonl for a manual review sample. The sample has been exported, not automatically certified as manually reviewed.

Reddit collection requires approved research access; no generic PRAW scraper is enabled. Imported provenance is a user attestation. Reddit author/profile fields are not stored. Purging cached text does not remove external imports or older exported artifacts; remove/regenerate affected exports under your source's retention requirements.

No FinBERT scores, trained models, financial forecasts or predictive claims are produced in M2.
'''
    (output / 'DATA_CARD.md').write_text(card, encoding='utf-8')
    dependencies = {}
    for name in ('exchange-calendars', 'yfinance', 'pandas', 'numpy'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    hashes = {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    supporting_hashes = {name: file_hash(Path(__file__).parent.parent / name)
                         for name in ('requirements-acquisition.lock', 'pyproject.toml')
                         if (Path(__file__).parent.parent / name).exists()}
    identity = {'config': config.to_dict(), 'acquisition': asdict(settings), 'outputs': hashes, 'code': code,
                'supporting_sha256': supporting_hashes,
                'python_version': platform.python_version(), 'dependencies': dependencies}
    manifest = {**identity, 'run_fingerprint': digest(identity), 'created_at': utc_now().isoformat(),
                'package_version': __version__, 'synthetic_data': True if synthetic else None,
                'imports': store.imports(), 'request_usage': store.usage(),
                'collection': collection, 'calendar': calendar_metadata,
                'price_snapshot_ids': sorted({r['snapshot_id'] for r in prices if r.get('snapshot_id')}),
                'text_snapshot_ids': sorted({r[key] for r in texts for key in ('snapshot_id', 'first_snapshot_id') if r.get(key)})}
    write_json(output / 'manifest.json', manifest)
    return {'output': str(output.resolve()), 'price_bars': len(prices), 'text_versions': len(texts),
            'feasibility': gate['decision'], 'shared_usable_sessions': gate['shared_usable_sessions']}


def collect(config_path: Path, settings_path: Path, cache: Path, output: Path,
            refresh_prices=False, refresh_news=False, offline=False) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ContractError('Output directory must be empty; use a fresh directory with the same cache')
    config, settings = Config.load(config_path), AcquisitionConfig.load(settings_path)
    if offline and (refresh_prices or refresh_news):
        raise ContractError('Offline mode cannot be combined with refresh flags')
    sessions, metadata = generate_calendar(session_date(settings.start), session_date(settings.end))
    with Store(cache) as store:
        store.bind(config.to_dict())
        prices, price_status = collect_prices(store, config, settings, sessions, refresh=refresh_prices,
                                             cache_dir=cache.parent / 'yfinance', offline=offline)
        news_status = ({'status': 'cached_only', 'reason': 'Offline mode; only previously normalized text is exported.'}
                       if offline else collect_news(NewsClient(store, os.environ.get('NEWSAPI_API_KEY', ''), settings),
                                   config, settings, sessions, refresh=refresh_news))
        imports = store.imports()
        reddit_status = {'status': 'imported' if any(r['source'] == 'reddit' for r in imports) else 'unavailable',
                         'reason': 'Approved research export importer supported; live research interface is not configured.'}
        result = export_dataset(output, config, settings, sessions, metadata, prices, store,
                                 {'prices': price_status, 'newsapi': news_status, 'reddit': reddit_status})
        result['manifest'] = str((output / 'manifest.json').resolve())
        return result
