from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from io import StringIO
import importlib.util
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

from sentiment_lab.acquisition import AcquisitionConfig, export_dataset
from sentiment_lab.calendar import generate_calendar
from sentiment_lab.cli import main
from sentiment_lab.config import Config
from sentiment_lab.contracts import ContractError, PriceBar, Session
from sentiment_lab.features import (FEATURE_SETS, KEYS, PRICE_FEATURES, build_feature_rows,
                                   read_json, read_jsonl, run_features)
from sentiment_lab.finbert import validate_prediction
from sentiment_lab.indicators import FeatureConfig, known_history, technical_values, wilder_rsi
from sentiment_lab.io import file_hash
from sentiment_lab.normalization import normalized_version
from sentiment_lab.preprocessing import TrainPreprocessor
from sentiment_lab.sentiment import run_sentiment
from sentiment_lab.store import Store


ROOT = Path(__file__).resolve().parents[1]
CALENDAR_AVAILABLE = bool(importlib.util.find_spec('exchange_calendars'))


def toy_history(closes, symbol='AAPL'):
    sessions, bars = [], []
    day = date(2024, 3, 4)
    for value in closes:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        opened = datetime(day.year, day.month, day.day, 14, 30, tzinfo=timezone.utc)
        closed = opened + timedelta(hours=6, minutes=30)
        sessions.append(Session(day, opened, closed))
        bars.append(PriceBar(symbol, day, value, value + 1, max(value - 1, .01), value, 1000,
                             closed + timedelta(minutes=1), 1))
        day += timedelta(days=1)
    return sessions, bars


class IndicatorTests(unittest.TestCase):
    def test_exact_returns_momentum_and_constant_return_volatility(self):
        _, history = toy_history([2.0**i for i in range(11)])
        result = technical_values(history)
        self.assertEqual(result['return_1d'], 1)
        self.assertEqual(result['return_5d'], 31)
        self.assertAlmostEqual(result['momentum_10d'], 1024 / 204.6 - 1)
        self.assertEqual(result['volatility_10d'], 0)
        self.assertEqual(result['rsi_5d'], 100)

    def test_sample_volatility_uses_ten_returns_and_ddof_one(self):
        returns = [.01, -.02, .03, -.01, .02, -.03, .04, -.02, .01, -.01]
        closes = [100.0]
        for value in returns:
            closes.append(closes[-1] * (1 + value))
        _, history = toy_history(closes)
        expected = math.sqrt(sum((value - .002) ** 2 for value in returns) / 9)
        self.assertAlmostEqual(technical_values(history)['volatility_10d'], expected, places=12)

    def test_wilder_seed_smoothing_and_flat_extremes(self):
        self.assertAlmostEqual(wilder_rsi([10, 11, 10, 12, 11, 13]), 71.42857142857143)
        self.assertAlmostEqual(wilder_rsi([10, 11, 10, 12, 11, 13, 12]), 60.6060606060606)
        self.assertEqual(wilder_rsi([100] * 20), 50)
        self.assertEqual(wilder_rsi(list(range(20, 0, -1))), 0)
        self.assertEqual(wilder_rsi(list(range(1, 21))), 100)
        with self.assertRaises(ContractError):
            wilder_rsi([1, 2, 3])

    def test_rsi_keeps_recursive_history_instead_of_only_latest_five_changes(self):
        first = [10, 15, 14, 16, 13, 18, 17, 16, 17, 16, 18]
        second = [30, 29, 28, 27, 26, 18, 17, 16, 17, 16, 18]
        self.assertNotEqual(wilder_rsi(first), wilder_rsi(second))

    def test_minimum_history_and_frozen_schema(self):
        _, history = toy_history([100] * 10)
        with self.assertRaisesRegex(ContractError, 'eleven'):
            technical_values(history)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.toml'
            path.write_text('volatility_ddof = 0\n')
            with self.assertRaisesRegex(ContractError, 'frozen'):
                FeatureConfig.load(path)
            path.write_text('random_shuffle = true\n')
            with self.assertRaisesRegex(ContractError, 'Unknown'):
                FeatureConfig.load(path)

    def test_future_and_target_prices_or_splits_cannot_change_past_features(self):
        sessions, bars = toy_history([100 + i + math.sin(i) for i in range(35)])
        target = sessions[20]
        mapping = {(p.ticker, p.session): p for p in bars}
        history, _ = known_history('AAPL', target.date, target.open_at, sessions, mapping)
        expected = technical_values(history)
        altered = {(p.ticker, p.session): (replace(p, close=p.close * 10, split_ratio=2)
                   if p.session >= target.date else p) for p in bars}
        future, _ = known_history('AAPL', target.date, target.open_at, sessions, altered)
        self.assertEqual(expected, technical_values(future))
        prefix, _ = known_history('AAPL', target.date, target.open_at, sessions[:21], mapping)
        self.assertEqual(expected, technical_values(prefix))
        self.assertEqual(history[-1].session, sessions[19].date)

    def test_ticker_histories_are_isolated(self):
        sessions, bars = toy_history([100 + i for i in range(20)])
        mapping = {(p.ticker, p.session): p for p in bars}
        target = sessions[-1]
        before, _ = known_history('AAPL', target.date, target.open_at, sessions, mapping)
        mapping.update({('MSFT', p.session): replace(p, ticker='MSFT', close=1000000, split_ratio=3) for p in bars})
        after, _ = known_history('AAPL', target.date, target.open_at, sessions, mapping)
        self.assertEqual(before, after)

    def test_missing_bar_resets_instead_of_bridging_across_session_gap(self):
        sessions, bars = toy_history([100 + i for i in range(30)])
        mapping = {(p.ticker, p.session): p for p in bars}
        del mapping[('AAPL', sessions[15].date)]
        history, barrier = known_history('AAPL', sessions[20].date, sessions[20].open_at, sessions, mapping)
        self.assertEqual(barrier, 'missing_history_bar')
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].session, sessions[16].date)

    def test_bar_at_or_after_cutoff_is_excluded_but_can_be_used_once_known(self):
        sessions, bars = toy_history([100 + i for i in range(30)])
        cutoff = sessions[20].open_at
        bars[15] = replace(bars[15], available_at=cutoff)
        mapping = {(p.ticker, p.session): p for p in bars}
        early, barrier = known_history('AAPL', sessions[20].date, cutoff, sessions, mapping)
        later, _ = known_history('AAPL', sessions[21].date, sessions[21].open_at, sessions, mapping)
        self.assertEqual(barrier, 'unavailable_history_bar')
        self.assertEqual(len(early), 4)
        self.assertEqual(len(later), 21)

    def test_raw_or_already_adjusted_split_series_both_restart_at_split_close(self):
        for raw in (True, False):
            with self.subTest(raw=raw):
                sessions, bars = toy_history([100 if i < 20 or not raw else 50 for i in range(35)])
                bars[20] = replace(bars[20], split_ratio=2)
                mapping = {(p.ticker, p.session): p for p in bars}
                early, barrier = known_history('AAPL', sessions[21].date, sessions[21].open_at, sessions, mapping)
                self.assertEqual(len(early), 1)
                self.assertEqual(barrier, 'split_reset')
                later, _ = known_history('AAPL', sessions[31].date, sessions[31].open_at, sessions, mapping)
                self.assertEqual(len(later), 11)
                self.assertEqual(technical_values(later)['return_5d'], 0)
                self.assertEqual(technical_values(later)['rsi_5d'], 50)

    def test_preclose_daily_availability_is_rejected(self):
        sessions, bars = toy_history([100] * 20)
        bars[5] = replace(bars[5], available_at=sessions[5].open_at)
        with self.assertRaisesRegex(ContractError, 'before its exchange close'):
            known_history('AAPL', sessions[-1].date, sessions[-1].open_at, sessions, {(p.ticker, p.session): p for p in bars})

    def test_unordered_indicator_calendar_is_not_silently_treated_as_chronological(self):
        sessions, bars = toy_history([100] * 20)
        with self.assertRaisesRegex(ContractError, 'sorted unique'):
            known_history('AAPL', sessions[-1].date, sessions[-1].open_at, list(reversed(sessions)),
                          {(p.ticker, p.session): p for p in bars})


class PreprocessingTests(unittest.TestCase):
    def row(self, value, day, partition='train'):
        return {'ticker': 'AAPL', 'target_session': f'2024-03-{day:02d}', 'partition': partition,
                'decision_at': f'2024-03-{day:02d}T14:25:00+00:00', 'return_1d': value}

    def test_fit_uses_training_median_and_population_scale(self):
        rows = [self.row(value, i + 1) for i, value in enumerate([1, 3, None])]
        processor = TrainPreprocessor.fit(rows, ('return_1d',), '2024-03-10T14:25:00Z')
        stats = processor.statistics['return_1d']
        self.assertEqual(stats['fill_value'], 2)
        self.assertEqual(stats['mean'], 2)
        self.assertAlmostEqual(stats['scale'], math.sqrt(2 / 3))
        heldout = self.row(10000, 12, 'test')
        self.assertGreater(processor.transform([heldout])[0]['return_1d'], 1000)
        self.assertEqual(processor.statistics['return_1d'], stats)

    def test_refuses_validation_rows_evaluation_time_rows_duplicates_and_extra_labels(self):
        original = self.row(1, 1)
        candidates = [[self.row(2, 12, 'validation')], [self.row(1, 10)], [original, original],
                      [{**original, 'label_up': 1}]]
        for rows in candidates:
            with self.subTest(rows=rows), self.assertRaises(ContractError):
                TrainPreprocessor.fit(rows, ('return_1d',), '2024-03-10T14:25:00Z')

    def test_all_missing_and_constant_columns_remain_finite_and_are_disclosed(self):
        rows = [self.row(None, 1), self.row(None, 2)]
        processor = TrainPreprocessor.fit(rows, ('return_1d',), '2024-03-10T14:25:00Z')
        self.assertTrue(processor.statistics['return_1d']['all_missing_train'])
        self.assertEqual(processor.transform(rows)[0]['return_1d'], 0)
        self.assertIsNone(rows[0]['return_1d'])
        constant = TrainPreprocessor.fit([self.row(3, 1), self.row(3, 2)], ('return_1d',), '2024-03-10T14:25:00Z')
        self.assertEqual(constant.statistics['return_1d']['scale'], 1)

    def test_nonfinite_or_nonallowlisted_transform_input_fails(self):
        processor = TrainPreprocessor.fit([self.row(1, 1)], ('return_1d',), '2024-03-10T14:25:00Z')
        for row in (self.row(float('nan'), 12, 'test'), {**self.row(2, 12, 'test'), 'target_return': .1}):
            with self.assertRaises(ContractError):
                processor.transform([row])

    def test_new_fold_can_refit_without_reusing_outer_statistics(self):
        first = TrainPreprocessor.fit([self.row(1, 1), self.row(3, 2)], ('return_1d',), '2024-03-10T14:25:00Z')
        second = TrainPreprocessor.fit([self.row(1, 1)], ('return_1d',), '2024-03-02T14:25:00Z')
        self.assertEqual(first.statistics['return_1d']['mean'], 2)
        self.assertEqual(second.statistics['return_1d']['mean'], 1)
        self.assertNotEqual(first.fit_metadata['training_feature_sha256'], second.fit_metadata['training_feature_sha256'])


class FixtureScorer:
    """A test-only scorer used to construct synthetic M3 integration inputs."""
    signature = 'm4-test-only-scores'
    metadata = {'backend': 'test-double-not-finbert'}

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, texts):
        return [validate_prediction({'positive': .6, 'negative': .2, 'neutral': .2,
                                     'token_count': 12, 'truncated': False}) for _ in texts]


@unittest.skipUnless(CALENDAR_AVAILABLE, 'Install acquisition dependencies for full feature integration')
class FeaturePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.config_path = ROOT / 'sentiment_lab/fixtures/demo.toml'
        config = Config.load(self.config_path)
        settings = AcquisitionConfig(start='2024-01-02', end='2024-04-05')
        sessions, metadata = generate_calendar(date(2024, 1, 2), date(2024, 4, 5))
        prices = []
        with Store(self.path / 'sources.sqlite') as store:
            for i, session in enumerate(sessions):
                for j, symbol in enumerate(config.tickers):
                    value = 100 + j * 100 + 1.2 * i + 3 * math.sin(i / 4)
                    prices.append({'ticker': symbol, 'session': str(session.date), 'open': value,
                                   'high': value + 2, 'low': value - 2, 'close': value, 'volume': 1000 + i,
                                   'available_at': (session.close_at + timedelta(minutes=1)).isoformat(),
                                   'split_ratio': 2 if symbol == 'MSFT' and i == 25 else 1})
                    if i and i % 3 == 0:
                        previous = sessions[i - 1]
                        for source in ('newsapi', 'reddit'):
                            published = previous.close_at + timedelta(minutes=1)
                            row = {'record_id': f'{symbol}-{source}-{i}', 'source': source, 'ticker': symbol,
                                   'text': f'Synthetic {symbol} {source} financial statement {i}.',
                                   'published_at': published.isoformat(), 'available_at': published.isoformat(),
                                   'retrieved_at': published.isoformat(), 'availability_basis': 'observed'}
                            store.put_text(normalized_version(row, row['record_id']))
            self.dataset, self.sentiment = self.path / 'm2', self.path / 'm3'
            export_dataset(self.dataset, config, settings, sessions, metadata, prices, store, {}, synthetic=True)
        run_sentiment(self.dataset, self.config_path, ROOT / 'configs/sentiment.toml', self.path / 'model',
                      self.path / 'scores.sqlite', self.sentiment, scorer_factory=FixtureScorer)

    def run_features(self, name='features'):
        return run_features(self.dataset, self.sentiment, self.config_path, ROOT / 'configs/features.toml', self.path / name)

    def rewrite_m3_hash(self, filename):
        path = self.sentiment / 'manifest.json'
        manifest = read_json(path)
        manifest['output_sha256'][filename] = file_hash(self.sentiment / filename)
        path.write_text(json.dumps(manifest))

    def test_matched_allowlisted_matrices_separate_labels_and_preserved_partitions(self):
        result = self.run_features()
        self.assertEqual(result['status'], 'ready')
        self.assertGreater(result['dropped_rows'], 20)
        original = {(r['ticker'], r['target_session']): r for r in read_jsonl(self.sentiment / 'daily_sentiment.jsonl')}
        expected_ids = None
        for name, columns in FEATURE_SETS.items():
            rows = read_jsonl(self.path / f'features/{name}.jsonl')
            ids = [tuple(row[k] for k in KEYS) for row in rows]
            self.assertEqual(set(rows[0]), set(KEYS + columns))
            expected_ids = expected_ids or ids
            self.assertEqual(ids, expected_ids)
            for row in rows:
                self.assertEqual(row['partition'], original[(row['ticker'], row['target_session'])]['partition'])
                self.assertNotIn('label_up', row)
                self.assertNotIn('target_return', row)
        labels = read_jsonl(self.path / 'features/labels.jsonl')
        self.assertEqual(len(labels), len(expected_ids))
        self.assertTrue(all('label_up' in row for row in labels))

    def test_lineage_prices_and_texts_are_before_decision_and_schema_has_no_outcomes(self):
        self.run_features()
        for row in read_jsonl(self.path / 'features/lineage.jsonl'):
            self.assertLess(row['price_history_end_session'], row['target_session'])
            self.assertLess(row['price_max_available_at'], row['decision_at'])
            self.assertGreaterEqual(row['price_history_bars'], 11)
            for source in ('newsapi', 'reddit'):
                value = row[source + '_latest_available_at']
                self.assertTrue(value is None or value < row['decision_at'])
        schema = read_json(self.path / 'features/feature_schema.json')
        self.assertEqual(len(schema['feature_sets']['combined']), 19)
        self.assertTrue(set(schema['forbidden_features']).isdisjoint(schema['feature_sets']['combined']))
        self.assertTrue(read_json(self.path / 'features/leakage_audit.json')['checks']['m3_scores_and_aggregates_reconciled'])

    def test_raw_sentiment_missingness_is_preserved_and_transforms_fit_train_only(self):
        self.run_features()
        rows = read_jsonl(self.path / 'features/combined.jsonl')
        self.assertTrue(any(row['newsapi_mean_polarity'] is None and row['newsapi_missing'] for row in rows))
        self.assertTrue(any(row['newsapi_mean_polarity'] is not None for row in rows))
        processor = read_json(self.path / 'features/preprocessing.json')
        train = [r for r in rows if r['partition'] == 'train']
        self.assertEqual(processor['fit']['fit_rows'], len(train))
        expected = statistics.median(r['return_1d'] for r in train)
        self.assertEqual(processor['statistics']['return_1d']['fill_value'], expected)
        for row in read_jsonl(self.path / 'features/scaled_combined.jsonl'):
            self.assertTrue(all(math.isfinite(row[name]) for name in FEATURE_SETS['combined']))

    def test_repeat_outputs_identical_and_all_manifest_hashes_match(self):
        self.run_features('one')
        self.run_features('two')
        first, second = self.path / 'one', self.path / 'two'
        for name in ('combined.jsonl', 'scaled_combined.jsonl', 'price_only.jsonl', 'preprocessing.json', 'lineage.jsonl'):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        for name, expected in read_json(first / 'manifest.json')['output_sha256'].items():
            self.assertEqual(file_hash(first / name), expected)

    def test_tampered_m3_hash_fails(self):
        with (self.sentiment / 'daily_sentiment.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ContractError, 'hash mismatch'):
            self.run_features()

    def test_rehashed_wrong_sentiment_aggregate_still_fails_reconciliation(self):
        path = self.sentiment / 'daily_sentiment.jsonl'
        rows = read_jsonl(path)
        rows[0]['newsapi_count'] += 1
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.rewrite_m3_hash(path.name)
        with self.assertRaisesRegex(ContractError, 'aggregates differ'):
            self.run_features()

    def test_rehashed_future_text_timestamp_is_rejected(self):
        path = self.sentiment / 'scored_texts.jsonl'
        rows = read_jsonl(path)
        rows[0]['available_at'] = '2030-01-01T00:00:00+00:00'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.rewrite_m3_hash(path.name)
        with self.assertRaisesRegex(ContractError, 'timestamp lineage differs'):
            self.run_features()

    def test_rehashed_labels_cannot_be_swapped_or_repartitioned(self):
        path = self.sentiment / 'labels.jsonl'
        rows = read_jsonl(path)
        rows[0]['label_up'] = 1 - rows[0]['label_up']
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.rewrite_m3_hash(path.name)
        with self.assertRaisesRegex(ContractError, 'labels/keys'):
            self.run_features()

    def test_wrong_source_binding_or_synthetic_mode_is_rejected(self):
        path = self.sentiment / 'manifest.json'
        manifest = read_json(path)
        manifest['synthetic_data'] = None
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ContractError, 'synthetic provenance'):
            self.run_features()

    def test_existing_output_and_cli_errors_are_clear(self):
        self.run_features()
        with self.assertRaisesRegex(ContractError, 'must be empty'):
            self.run_features()
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(main(['build-features', '--dataset', str(self.dataset), '--sentiment', str(self.sentiment),
                '--config', str(self.config_path), '--out', str(self.path / 'cli')]), 0)

    def test_short_history_exports_audited_empty_matrices_without_fitting(self):
        import csv
        fixture = ROOT / 'sentiment_lab/fixtures'
        sessions, metadata = generate_calendar(date(2024, 11, 26), date(2024, 12, 4))
        with (fixture / 'prices.csv').open(newline='') as stream:
            prices = list(csv.DictReader(stream))
        dataset, sentiment = self.path / 'short-m2', self.path / 'short-m3'
        with Store(self.path / 'short.sqlite') as store:
            export_dataset(dataset, Config.load(self.config_path), AcquisitionConfig(start='2024-11-26', end='2024-12-04'),
                           sessions, metadata, prices, store, {}, synthetic=True)
        run_sentiment(dataset, self.config_path, ROOT / 'configs/sentiment.toml', self.path / 'model',
                      self.path / 'unused.sqlite', sentiment, scorer_factory=FixtureScorer)
        result = run_features(dataset, sentiment, self.config_path, ROOT / 'configs/features.toml', self.path / 'short-features')
        self.assertEqual(result['status'], 'insufficient_history_or_partitions')
        self.assertEqual(result['retained_rows'], 0)
        self.assertEqual(read_jsonl(self.path / 'short-features/combined.jsonl'), [])
        self.assertFalse((self.path / 'short-features/scaled_combined.jsonl').exists())
        self.assertEqual(read_json(self.path / 'short-features/preprocessing.json')['status'], 'not_fitted_incomplete_partitions')

    def test_calendar_input_order_is_normalized_before_computing_features(self):
        self.run_features('before')
        path = self.dataset / 'sessions.csv'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join([lines[0], *reversed(lines[1:])]) + '\n')
        manifest_path = self.dataset / 'manifest.json'
        manifest = read_json(manifest_path)
        manifest['outputs']['sessions.csv'] = file_hash(path)
        manifest_path.write_text(json.dumps(manifest))
        self.sentiment = self.path / 'm3-reordered'
        run_sentiment(self.dataset, self.config_path, ROOT / 'configs/sentiment.toml', self.path / 'model',
                      self.path / 'scores.sqlite', self.sentiment, scorer_factory=FixtureScorer)
        self.run_features('after')
        self.assertEqual((self.path / 'before/combined.jsonl').read_bytes(), (self.path / 'after/combined.jsonl').read_bytes())


if __name__ == '__main__':
    unittest.main()
