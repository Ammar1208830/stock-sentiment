"""M3 scoring and daily source aggregates; no directional model training."""
from dataclasses import asdict
from importlib.resources import files
import json
from pathlib import Path
import platform
import statistics

from . import __version__
from .acquisition import write_csv
from .config import Config
from .contracts import ContractError
from .finbert import FinBERTScorer, LABELS, SentimentConfig, normalize_text, validate_prediction
from .io import file_hash, read_inputs, write_json, write_jsonl
from .pipeline import assemble
from .sentiment_cache import SentimentCache, score_unique
from .store import digest, utc_now


SOURCES = ('newsapi', 'reddit')
AGGREGATES = ('count', 'missing', 'mean_positive', 'mean_negative', 'mean_neutral',
              'mean_polarity', 'polarity_dispersion', 'truncated_count', 'latest_available_at')


def assert_empty(output: Path):
    if output.exists() and any(output.iterdir()):
        raise ContractError('Sentiment output directory must be empty; use a fresh output path')


def aggregate_windows(windows: list[dict], scores: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    daily, scored = [], []
    seen = set()
    for window in windows:
        row = {key: window[key] for key in ('ticker', 'target_session', 'partition', 'decision_at', 'window_start')}
        for source in SOURCES:
            texts = [record for record in window['texts'] if record['source'] == source]
            values = []
            for record in texts:
                if record['available_at'] >= window['decision_at'] or record['published_at'] < window['window_start']:
                    raise ContractError('Aggregation encountered text outside the permitted time window')
                key = normalize_text(record['text'])
                if key not in scores:
                    raise ContractError('Every included text must have a validated model score')
                result = validate_prediction(scores[key])
                values.append(result)
                identity = (record['source'], window['ticker'], record['record_id'], window['target_session'])
                if identity in seen:
                    raise ContractError('A text record was counted twice in the same ticker/session')
                seen.add(identity)
                scored.append({'ticker': window['ticker'], 'target_session': window['target_session'],
                    'partition': window['partition'], 'source': source, 'record_id': record['record_id'],
                    'origin_id': record.get('origin_id'), 'published_at': record['published_at'],
                    'available_at': record['available_at'], 'availability_basis': record['availability_basis'],
                    'text_sha256': digest(key), 'positive': result['positive'], 'negative': result['negative'],
                    'neutral': result['neutral'], 'polarity': result['polarity'],
                    'sentiment_label': result['label'], 'token_count': result['token_count'], 'truncated': result['truncated']})
            prefix = source + '_'
            row.update({prefix + 'count': len(values), prefix + 'missing': not values,
                        prefix + 'truncated_count': sum(value['truncated'] for value in values),
                        prefix + 'latest_available_at': max((r['available_at'] for r in texts), default=None)})
            for label in LABELS:
                row[prefix + 'mean_' + label] = statistics.fmean(value[label] for value in values) if values else None
            row[prefix + 'mean_polarity'] = statistics.fmean(value['polarity'] for value in values) if values else None
            row[prefix + 'polarity_dispersion'] = statistics.pstdev(value['polarity'] for value in values) if values else None
        daily.append(row)
    return daily, scored


def verify_dataset(dataset: Path, config: Config) -> dict:
    try:
        manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
        if digest(manifest['config']) != digest(config.to_dict()):
            raise ContractError('Project configuration differs from the acquisition manifest; use the matching config')
        required = ('sessions.csv', 'prices.csv', 'texts.jsonl', 'feasibility.json', 'calendar.json')
        for name in required:
            if file_hash(dataset / name) != manifest['outputs'][name]:
                raise ContractError(f'Acquisition input hash mismatch: {name}')
        if not isinstance(manifest.get('synthetic_data'), (bool, type(None))):
            raise ContractError('Acquisition manifest has invalid synthetic_data metadata')
        return manifest
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f'Invalid acquisition dataset/manifest: {exc}') from exc


def run_sentiment(dataset: Path, config_path: Path, sentiment_path: Path, model_root: Path,
                  cache_path: Path, output: Path, offline_model=True, scorer_factory=FinBERTScorer) -> dict:
    assert_empty(output)
    config, settings = Config.load(config_path), SentimentConfig.load(sentiment_path)
    acquisition = verify_dataset(dataset, config)
    synthetic = acquisition.get('synthetic_data') is True
    sessions, prices, texts = read_inputs(dataset / 'sessions.csv', dataset / 'prices.csv', dataset / 'texts.jsonl')
    windows, alignment = assemble(config, sessions, prices, texts)
    inputs = [record['text'] for row in windows for record in row['texts']]
    scorer = None
    scores, cache_report = {}, {'unique_texts': 0, 'cache_hits': 0, 'inferred_texts': 0, 'truncated_unique_texts': 0}
    if inputs:
        scorer = scorer_factory(settings, model_root, offline=offline_model)
        with SentimentCache(cache_path, synthetic=synthetic) as cache:
            scores, cache_report = score_unique(inputs, scorer, cache, settings.batch_size)
    daily, scored = aggregate_windows(windows, scores)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'daily_sentiment.jsonl', daily)
    write_csv(output / 'daily_sentiment.csv', daily,
              ['ticker', 'target_session', 'partition', 'decision_at', 'window_start'] +
              [source + '_' + field for source in SOURCES for field in AGGREGATES])
    write_jsonl(output / 'scored_texts.jsonl', scored)
    # Outcomes remain in their own file. Aggregates never contain target returns or direction.
    write_jsonl(output / 'labels.jsonl', [{key: row[key] for key in
        ('ticker', 'target_session', 'partition', 'label_up', 'target_return', 'label_available_at')} for row in windows])
    by_identity = {(r.source, r.ticker, r.record_id): r.text for r in texts}
    review = [{**row, 'text': by_identity[(row['source'], row['ticker'], row['record_id'])]} for row in scored[:25]]
    write_jsonl(output / 'review_sample.jsonl', review)
    source_gate = json.loads((dataset / 'feasibility.json').read_text())
    summary = {'status': 'scored' if inputs else 'no_eligible_text', 'synthetic_data': True if synthetic else None,
               'ticker_sessions': len(daily), 'included_text_records': len(scored), 'inference': cache_report,
               'source_coverage': {source: {'covered_rows': sum(not row[source + '_missing'] for row in daily),
                    'missing_rows': sum(row[source + '_missing'] for row in daily)} for source in SOURCES},
               'source_data_gate': source_gate['decision'], 'alignment': alignment,
               'manual_review_status': 'sample_exported_not_yet_annotated',
               'interpretation': 'Softmax sentiment scores are not calibrated sentiment or stock-direction probabilities. No predictive model was trained.'}
    write_json(output / 'summary.json', summary)
    metadata = scorer.metadata if scorer else {'status': 'not_loaded_no_eligible_text', 'config': asdict(settings)}
    write_manifest(output, metadata, {name: file_hash(dataset / name) for name in
        ('manifest.json', 'sessions.csv', 'prices.csv', 'texts.jsonl', 'calendar.json', 'feasibility.json')} |
        {'project_config': file_hash(config_path), 'sentiment_config': file_hash(sentiment_path)},
        synthetic, 'ticker_session_sentiment', source_gate['decision'])
    return {'output': str(output.resolve()), 'status': summary['status'], 'ticker_sessions': len(daily),
            'included_text_records': len(scored), **cache_report, 'source_data_gate': source_gate['decision']}


def write_manifest(output: Path, model: dict, inputs: dict, synthetic: bool, mode: str, source_gate=None):
    lock = Path(__file__).parent.parent / 'requirements-sentiment.lock'
    identity = {'schema_version': 1, 'package_version': __version__, 'python_version': platform.python_version(),
                'mode': mode, 'synthetic_data': True if synthetic else None, 'source_data_gate': source_gate,
                'model': model, 'input_sha256': inputs,
                'code_sha256': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                'environment_lock_sha256': file_hash(lock) if lock.exists() else None,
                'output_sha256': {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'}}
    write_json(output / 'manifest.json', {**identity, 'created_at': utc_now().isoformat(), 'run_fingerprint': digest(identity)})


def run_smoke(sentiment_path: Path, model_root: Path, cache_path: Path, output: Path,
              offline_model=True, scorer_factory=FinBERTScorer) -> dict:
    assert_empty(output)
    settings = SentimentConfig.load(sentiment_path)
    fixture = files('sentiment_lab').joinpath('fixtures/sentiment_smoke.jsonl')
    fixtures = [json.loads(line) for line in fixture.read_text().splitlines()]
    scorer = scorer_factory(settings, model_root, offline=offline_model)
    with SentimentCache(cache_path, synthetic=True) as cache:
        scores, report = score_unique([r['text'] for r in fixtures], scorer, cache, settings.batch_size)
    rows = [{**r, **scores[normalize_text(r['text'])]} for r in fixtures]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'scores.jsonl', rows)
    summary = {'synthetic_input': True, 'inference_backend': scorer.metadata['backend'], 'examples': len(rows),
               'illustrative_label_matches': sum(row['label'] == row['expected_sentiment'] for row in rows),
               'cache': report, 'interpretation': 'Authored examples are a software smoke test, not an independent benchmark or predictive study.'}
    write_json(output / 'summary.json', summary)
    write_manifest(output, scorer.metadata, {'fixture': digest(fixtures), 'sentiment_config': file_hash(sentiment_path)},
                   True, 'authored_examples_model_smoke', 'engineering_demonstration_only')
    return {'output': str(output.resolve()), **summary}
