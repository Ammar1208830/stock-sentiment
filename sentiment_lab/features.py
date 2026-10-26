"""M4 matched feature sets, train-only transforms and temporal lineage audits."""
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform

from . import __version__
from .acquisition import write_csv
from .calendar import verify_calendar
from .config import Config
from .contracts import ContractError, session_date, timestamp
from .finbert import normalize_text, validate_prediction
from .indicators import FeatureConfig, MIN_HISTORY, PRICE_FEATURES, known_history, technical_values
from .io import file_hash, read_inputs, write_json, write_jsonl
from .pipeline import assemble
from .preprocessing import TrainPreprocessor
from .sentiment import SOURCES, aggregate_windows, verify_dataset
from .store import digest, utc_now


KEYS = ('ticker', 'target_session', 'partition', 'decision_at')
SOURCE_SUFFIXES = ('count', 'missing', 'mean_positive', 'mean_negative', 'mean_neutral', 'mean_polarity', 'polarity_dispersion')
NEWS_FEATURES = tuple('newsapi_' + key for key in SOURCE_SUFFIXES)
REDDIT_FEATURES = tuple('reddit_' + key for key in SOURCE_SUFFIXES)
SENTIMENT_FEATURES = NEWS_FEATURES + REDDIT_FEATURES
FEATURE_SETS = {'price_only': PRICE_FEATURES, 'price_news': PRICE_FEATURES + NEWS_FEATURES,
                'price_reddit': PRICE_FEATURES + REDDIT_FEATURES, 'sentiment_only': SENTIMENT_FEATURES,
                'combined': PRICE_FEATURES + SENTIMENT_FEATURES}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('expected an object')
        return value
    except (ValueError, UnicodeError) as exc:
        raise ContractError(f'{path.name}: invalid JSON object') from exc


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open(encoding='utf-8') as stream:
        for line, content in enumerate(stream, 1):
            if not content.strip():
                continue
            try:
                row = json.loads(content)
                if not isinstance(row, dict):
                    raise ValueError('expected an object')
                result.append(row)
            except (ValueError, UnicodeError) as exc:
                raise ContractError(f'{path.name}:{line}: invalid JSON record') from exc
    return result


def same_structure(actual, expected) -> bool:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        return not isinstance(actual, bool) and isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(same_structure(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(same_structure(a, e) for a, e in zip(actual, expected))
    return False


def checked_sentiment(dataset: Path, sentiment: Path, config_path: Path, acquisition: dict, windows: list[dict]) -> tuple[dict, list[dict]]:
    manifest = read_json(sentiment / 'manifest.json')
    try:
        if manifest['mode'] != 'ticker_session_sentiment':
            raise ContractError('M4 requires ticker/session M3 output, not a free-text smoke report')
        if manifest.get('synthetic_data') != acquisition.get('synthetic_data'):
            raise ContractError('M2 and M3 synthetic provenance differs')
        for name in ('manifest.json', 'sessions.csv', 'prices.csv', 'texts.jsonl', 'calendar.json', 'feasibility.json'):
            if manifest['input_sha256'][name] != file_hash(dataset / name):
                raise ContractError(f'M3 output is bound to a different acquisition input: {name}')
        if manifest['input_sha256']['project_config'] != file_hash(config_path):
            raise ContractError('M3 output was produced with a different project config file')
        for name in ('daily_sentiment.jsonl', 'scored_texts.jsonl', 'labels.jsonl', 'summary.json'):
            if manifest['output_sha256'][name] != file_hash(sentiment / name):
                raise ContractError(f'M3 output hash mismatch: {name}')
        gate = read_json(dataset / 'feasibility.json')['decision']
        if manifest['source_data_gate'] != gate:
            raise ContractError('M3 source data gate differs from M2')
    except (KeyError, TypeError) as exc:
        raise ContractError('M3 manifest is missing required lineage/hash fields') from exc
    # Reconcile the per-record scores with M1/M2 cutoff/version selection, then
    # rebuild aggregates. A hash-valid file can still contain an alignment bug.
    expected_records = {(w['ticker'], w['target_session'], r['source'], r['record_id']): (w, r)
                        for w in windows for r in w['texts']}
    observed, scores = set(), {}
    for row in read_jsonl(sentiment / 'scored_texts.jsonl'):
        try:
            key = tuple(row[field] for field in ('ticker', 'target_session', 'source', 'record_id'))
            if key in observed or key not in expected_records:
                raise ContractError('M3 contains duplicate, unexpected or cutoff-ineligible score records')
            observed.add(key)
            window, record = expected_records[key]
            for field in ('source', 'record_id', 'origin_id', 'published_at', 'available_at', 'availability_basis'):
                if row.get(field) != record.get(field):
                    raise ContractError('M3 text version/timestamp lineage differs from acquisition alignment')
            if row['partition'] != window['partition']:
                raise ContractError('M3 changed a chronological partition')
            normalized = normalize_text(record['text'])
            if row['text_sha256'] != digest(normalized):
                raise ContractError('M3 score content hash does not match its eligible text version')
            value = validate_prediction(row)
            if not same_structure(row['polarity'], value['polarity']) or row['sentiment_label'] != value['label']:
                raise ContractError('M3 polarity/label differs from its probabilities')
            if normalized in scores and not same_structure(scores[normalized], value):
                raise ContractError('Identical content has inconsistent M3 model scores')
            scores[normalized] = value
        except (KeyError, TypeError) as exc:
            raise ContractError('M3 score record is missing required fields') from exc
    if observed != set(expected_records):
        raise ContractError('M3 lacks scores for one or more eligible acquisition records')
    expected_daily, _ = aggregate_windows(windows, scores)
    actual_daily = read_jsonl(sentiment / 'daily_sentiment.jsonl')
    expected_labels = [{field: w[field] for field in ('ticker', 'target_session', 'partition', 'label_up', 'target_return', 'label_available_at')} for w in windows]
    if not same_structure(read_jsonl(sentiment / 'labels.jsonl'), expected_labels):
        raise ContractError('M3 labels/keys do not match the acquisition outcomes')
    if not same_structure(actual_daily, expected_daily):
        raise ContractError('M3 aggregates differ from recomputed eligible-source probabilities/counts')
    return manifest, expected_daily


def feature_schema(settings: FeatureConfig) -> dict:
    definitions = {
        'return_1d': 'C[s] / C[s-1] - 1',
        'return_5d': 'C[s] / C[s-5] - 1',
        'momentum_10d': 'C[s] / mean(C[s-9:s]) - 1; endpoints inclusive',
        'volatility_10d': 'sample standard deviation (ddof=1) of the last 10 simple daily returns; not annualized',
        'rsi_5d': 'Wilder RSI, SMA seed of first 5 gains/losses after reset, then recursive 1/5 smoothing; flat=50, gain-only=100, loss-only=0',
    }
    return {'schema_version': 1, 'policy': asdict(settings), 'row_keys_and_audit_metadata': list(KEYS),
            'feature_sets': {name: list(fields) for name, fields in FEATURE_SETS.items()},
            'price_definitions': definitions, 'price_anchor': 's is the previous exchange session; target-session prices are never inputs',
            'minimum_consecutive_known_closes': MIN_HISTORY,
            'rsi_dependency': 'All consecutive known closes from the latest split/gap/availability barrier; not only a five-close rolling window',
            'sentiment_definitions': 'Source-separated M3 equal-weight means/counts/population dispersion and explicit missingness; absent means remain null',
            'forbidden_features': ['label_up', 'target_return', 'label_available_at', 'target_close_at', 'partition', 'ticker', 'target_session', 'decision_at'],
            'selection_rule': 'Use only a named frozen feature set. Never select model columns by numeric dtype.'}


def build_feature_rows(config, sessions, prices, windows, daily) -> tuple[list[dict], list[dict], list[dict]]:
    bars = {(p.ticker, p.session): p for p in prices}
    by_date = {(r['ticker'], r['target_session']): r for r in daily}
    combined, lineage, excluded = [], [], []
    for window in windows:
        symbol, target, decision = window['ticker'], session_date(window['target_session']), timestamp(window['decision_at'])
        history, barrier = known_history(symbol, target, decision, sessions, bars)
        if len(history) < MIN_HISTORY:
            excluded.append({field: window[field] for field in KEYS} | {
                'reason': 'insufficient_consecutive_known_history', 'history_bars': len(history), 'barrier': barrier})
            continue
        if history[-1].session.isoformat() != window['previous_session'] or any(p.session >= target or p.available_at >= decision for p in history):
            raise ContractError('Price feature lineage violates the historical decision cutoff')
        source = by_date[(symbol, str(target))]
        technical = technical_values(history)
        row = {field: window[field] for field in KEYS} | technical | {field: source[field] for field in SENTIMENT_FEATURES}
        combined.append(row)
        lineage.append({field: window[field] for field in KEYS} | {
            'price_history_start_session': str(history[0].session), 'price_history_end_session': str(history[-1].session),
            'price_history_bars': len(history), 'price_max_available_at': max(p.available_at for p in history).isoformat(),
            'price_barrier': barrier, 'price_split_anchor': history[0].split_ratio != 1,
            'newsapi_latest_available_at': source['newsapi_latest_available_at'],
            'reddit_latest_available_at': source['reddit_latest_available_at'],
            'text_records': source['newsapi_count'] + source['reddit_count'],
            'past_only_checks': True})
    return combined, lineage, excluded


def project_rows(rows: list[dict], columns: tuple[str, ...]) -> list[dict]:
    return [{field: row[field] for field in KEYS + columns} for row in rows]


def retained_gate(rows: list[dict], acquisition: dict, upstream_gate: str) -> dict:
    thresholds = acquisition['acquisition']
    tickers = acquisition['config']['tickers']
    dates = [{r['target_session'] for r in rows if r['ticker'] == symbol} for symbol in tickers]
    shared = set.intersection(*dates) if dates else set()
    test_dates = {r['target_session'] for r in rows if r['partition'] == 'test'} & shared
    coverage = {}
    for symbol in tickers:
        coverage[symbol] = {}
        for partition in ('train', 'validation', 'test'):
            subset = [r for r in rows if r['ticker'] == symbol and r['partition'] == partition]
            coverage[symbol][partition] = {source: sum(not r[source + '_missing'] for r in subset) / len(subset) if subset else 0.0 for source in SOURCES}
    enough = len(shared) >= thresholds['minimum_usable_sessions'] and len(test_dates) >= thresholds['minimum_test_sessions']
    source_pass = {source: all(values[source] >= thresholds['minimum_source_coverage'] for parts in coverage.values() for values in parts.values()) for source in SOURCES}
    decision = 'engineering_demonstration_only'
    if acquisition.get('synthetic_data') is not True and enough:
        if upstream_gate == 'full_study_feasible' and all(source_pass.values()):
            decision = 'full_study_feasible'
        elif upstream_gate in {'full_study_feasible', 'reduced_news_only_study_feasible'} and source_pass['newsapi']:
            decision = 'reduced_news_only_study_feasible'
    return {'decision': decision, 'upstream_decision': upstream_gate, 'shared_usable_sessions': len(shared),
            'shared_test_sessions': len(test_dates), 'coverage_by_ticker_partition': coverage,
            'source_coverage_pass': source_pass, 'history_thresholds_pass': enough,
            'thresholds': {name: thresholds[name] for name in ('minimum_usable_sessions', 'minimum_test_sessions', 'minimum_source_coverage')}}


def run_features(dataset: Path, sentiment: Path, config_path: Path, feature_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ContractError('Feature output directory must be empty; choose a fresh path')
    config, settings = Config.load(config_path), FeatureConfig.load(feature_path)
    acquisition = verify_dataset(dataset, config)
    sessions, prices, texts = read_inputs(dataset / 'sessions.csv', dataset / 'prices.csv', dataset / 'texts.jsonl')
    sessions = sorted(sessions, key=lambda session: session.date)
    calendar = read_json(dataset / 'calendar.json')
    try:
        verified = verify_calendar(sessions, session_date(calendar['requested_target_start']), session_date(calendar['requested_target_end']))
        if verified['version'] != calendar['version']:
            raise ContractError('Calendar library version differs from acquisition; recreate a matching environment or regenerate upstream data')
    except KeyError as exc:
        raise ContractError('Acquisition calendar metadata is incomplete') from exc
    windows, alignment = assemble(config, sessions, prices, texts)
    source_manifest, daily = checked_sentiment(dataset, sentiment, config_path, acquisition, windows)
    rows, lineage, excluded = build_feature_rows(config, sessions, prices, windows, daily)
    kept = {(r['ticker'], r['target_session']) for r in rows}
    label_rows = [{field: w[field] for field in ('ticker', 'target_session', 'partition', 'label_up', 'target_return', 'label_available_at')}
                  for w in windows if (w['ticker'], w['target_session']) in kept]
    parts = {partition: [row for row in rows if row['partition'] == partition] for partition in ('train', 'validation', 'test')}
    original_eval = min((w['decision_at'] for w in windows if w['partition'] == 'validation'), default=None)
    original_test = min((w['decision_at'] for w in windows if w['partition'] == 'test'), default=None)
    # Preserve upstream purges and dates, even after technical warmup removes rows.
    boundary_failures = sum((r['partition'] == 'train' and r['label_available_at'] >= original_eval) or
                            (r['partition'] == 'validation' and r['label_available_at'] >= original_test) for r in label_rows)
    if boundary_failures:
        raise ContractError('A retained label is unresolved at the next original partition boundary')
    processor, scaled = None, []
    if all(parts.values()):
        processor = TrainPreprocessor.fit(parts['train'], FEATURE_SETS['combined'], original_eval)
        scaled = processor.transform(rows)
    gate = retained_gate(rows, acquisition, source_manifest['source_data_gate'])
    ids = [tuple(row[key] for key in KEYS) for row in rows]
    if len(set(ids)) != len(ids) or len(rows) != len(label_rows):
        raise ContractError('Feature/label keys are duplicate or mismatched')
    status = 'ready' if processor else 'insufficient_history_or_partitions'
    audit = {'status': 'passed', 'calendar': verified,
             'checks': {'m2_m3_hashes_and_source_binding': True, 'm3_scores_and_aggregates_reconciled': True,
                        'complete_exchange_calendar': True, 'past_only_price_history': True,
                        'latest_before_cutoff_text_versions': True, 'labels_separate': True,
                        'original_partitions_preserved': True, 'matched_rows_across_feature_sets': True,
                        'frozen_feature_allowlists': True, 'split_and_gap_resets': True},
             'preprocessing': 'train_only' if processor else 'not_fitted_incomplete_partitions',
             'future_price_inputs': 0, 'future_text_inputs': 0, 'unresolved_boundary_labels': boundary_failures,
             'input_windows': len(windows), 'retained_rows': len(rows), 'excluded_rows': len(excluded),
             'excluded_barriers': dict(Counter(r['barrier'] for r in excluded)),
             'matched_key_sha256': digest(ids), 'upstream_alignment': alignment,
             'limits': ['Validates declared availability and internal provenance, not independent provider point-in-time truth.',
                        'Yahoo vendor history may retain retrospective adjustments/revisions.',
                        'No fitted predictive model or test-performance analysis occurs in M4.']}
    output.mkdir(parents=True, exist_ok=True)
    for name, columns in FEATURE_SETS.items():
        projected = project_rows(rows, columns)
        write_jsonl(output / f'{name}.jsonl', projected)
        write_csv(output / f'{name}.csv', projected, list(KEYS + columns))
        if processor:
            write_jsonl(output / f'scaled_{name}.jsonl', project_rows(scaled, columns))
    write_jsonl(output / 'labels.jsonl', label_rows)
    # Keep holdout outcomes physically separate so model selection need not parse them.
    for partition in ('train', 'validation', 'test'):
        write_jsonl(output / f'labels_{partition}.jsonl', [r for r in label_rows if r['partition'] == partition])
    write_jsonl(output / 'lineage.jsonl', lineage)
    write_jsonl(output / 'excluded_rows.jsonl', excluded)
    write_json(output / 'feature_schema.json', feature_schema(settings))
    write_json(output / 'preprocessing.json', processor.to_dict() if processor else {'status': 'not_fitted_incomplete_partitions'})
    write_json(output / 'leakage_audit.json', audit)
    write_json(output / 'feature_feasibility.json', gate)
    summary = {'status': status, 'input_rows': len(windows), 'retained_rows': len(rows), 'dropped_rows': len(excluded),
               'partition_rows': {name: len(value) for name, value in parts.items()},
               'feature_counts': {name: len(columns) for name, columns in FEATURE_SETS.items()},
               'source_data_gate': gate['decision'], 'synthetic_data': acquisition.get('synthetic_data'),
               'sentiment_covered_rows': {source: sum(not r[source + '_missing'] for r in rows) for source in SOURCES},
               'all_missing_training_features': [name for name, value in processor.statistics.items() if value['all_missing_train']] if processor else [],
               'trained_directional_models': False}
    write_json(output / 'summary.json', summary)
    identity = {'schema_version': 1, 'package_version': __version__, 'python_version': platform.python_version(),
                'config': config.to_dict(), 'features': feature_schema(settings), 'synthetic_data': acquisition.get('synthetic_data'),
                'source_data_gate': gate['decision'],
                'input_sha256': {'m2_manifest': file_hash(dataset / 'manifest.json'), 'm3_manifest': file_hash(sentiment / 'manifest.json'),
                                 'project_config': file_hash(config_path), 'feature_config': file_hash(feature_path)},
                'code_sha256': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                'model_provenance': source_manifest['model'],
                'environment_lock_sha256': file_hash(Path(__file__).parent.parent / 'requirements-acquisition.lock'),
                'output_sha256': {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / 'manifest.json', {**identity, 'created_at': utc_now().isoformat(), 'run_fingerprint': digest(identity)})
    return {'output': str(output.resolve()), **summary}
