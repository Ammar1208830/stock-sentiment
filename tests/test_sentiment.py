from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sentiment_lab.acquisition import AcquisitionConfig, export_dataset
from sentiment_lab.cli import main
from sentiment_lab.config import Config
from sentiment_lab.contracts import ContractError
from sentiment_lab.finbert import (FinBERTScorer, LABELS, MODEL_FILES, MODEL_ID, REVISION,
    SentimentConfig, label_mapping, normalize_text, prepare_model, validate_prediction)
from sentiment_lab.io import file_hash, read_inputs
from sentiment_lab.normalization import normalized_version
from sentiment_lab.sentiment import aggregate_windows, run_sentiment, run_smoke, verify_dataset
from sentiment_lab.sentiment_cache import SentimentCache, score_unique
from sentiment_lab.store import Store, digest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'sentiment_lab' / 'fixtures'


def prediction(positive=.7, negative=.2, neutral=.1, tokens=10, truncated=False):
    return validate_prediction({'positive': positive, 'negative': negative, 'neutral': neutral,
                                'token_count': tokens, 'truncated': truncated})


class TestScorer:
    """Only a test double; no product command can choose this backend."""
    signature = 'unit-test-model-v1'
    metadata = {'backend': 'test-double-not-finbert'}

    def __init__(self, *args, **kwargs):
        self.calls = []

    def predict(self, texts):
        self.calls.append(list(texts))
        return [prediction() for _ in texts]


class SentimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.settings = SentimentConfig()
        self.config = Config.load(FIXTURES / 'demo.toml')
        self.sessions, self.prices, _ = read_inputs(FIXTURES / 'sessions.csv', FIXTURES / 'prices.csv', FIXTURES / 'texts.jsonl')

    def dataset(self, empty=False):
        import csv
        with (FIXTURES / 'prices.csv').open(newline='') as stream:
            prices = list(csv.DictReader(stream))
        dataset = self.path / ('empty-dataset' if empty else 'dataset')
        with Store(self.path / 'acquisition.sqlite') as store:
            if not empty:
                for line in (FIXTURES / 'texts.jsonl').read_text().splitlines():
                    row = json.loads(line)
                    store.put_text(normalized_version(row, row['record_id']))
            export_dataset(dataset, self.config, AcquisitionConfig(start='2024-11-26', end='2024-12-04'),
                           self.sessions, {'version': 'test-calendar'}, prices, store, {}, synthetic=True)
        return dataset

    def run_dataset(self, dataset, out='out', scorer_factory=TestScorer):
        return run_sentiment(dataset, FIXTURES / 'demo.toml', ROOT / 'configs/sentiment.toml', self.path / 'models',
                             self.path / 'scores.sqlite', self.path / out, scorer_factory=scorer_factory)

    def test_config_requires_immutable_model_revision_and_valid_runtime_settings(self):
        for settings in (replace(self.settings, revision='main'), replace(self.settings, model_id='other/model'),
                         replace(self.settings, max_length=513), replace(self.settings, batch_size=0),
                         replace(self.settings, cpu_threads=True), replace(self.settings, device='cuda')):
            with self.subTest(settings=settings), self.assertRaises(ContractError):
                settings.validate()
        self.assertEqual(SentimentConfig.load(ROOT / 'configs/sentiment.toml'), self.settings)
        bad = self.path / 'bad.toml'
        bad.write_text('max_lenght = 512\n')
        with self.assertRaisesRegex(ContractError, 'Unknown sentiment'):
            SentimentConfig.load(bad)

    def test_label_mapping_accepts_permutation_but_rejects_missing_or_duplicate_classes(self):
        self.assertEqual(label_mapping({'0': 'negative', '1': 'neutral', '2': 'positive'})[0], 'negative')
        for raw in ({'0': 'positive', '1': 'negative'}, {'0': 'positive', '1': 'positive', '2': 'neutral'},
                    {'0': 'LABEL_0', '1': 'LABEL_1', '2': 'LABEL_2'}, [], {'a': 'positive'}):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                label_mapping(raw)

    def test_probabilities_and_polarity_are_validated_without_silent_renormalization(self):
        self.assertAlmostEqual(prediction()['polarity'], .5)
        self.assertEqual(prediction()['label'], 'positive')
        for update in ({'positive': float('nan')}, {'negative': -1}, {'neutral': .7},
                       {'positive': True}, {'token_count': 0}, {'truncated': 1}):
            with self.subTest(update=update), self.assertRaises(ContractError):
                validate_prediction({**prediction(), **update})

    def test_normalization_preserves_case_and_punctuation(self):
        self.assertEqual(normalize_text('  EPS\n grew  10%!\t'), 'EPS grew 10%!')
        for value in (' ', None, 'x' * 100001):
            with self.assertRaises(ContractError):
                normalize_text(value)

    def test_offline_model_preparation_never_downloads_missing_artifacts(self):
        with self.assertRaisesRegex(ContractError, 'not available locally'):
            prepare_model(self.settings, self.path / 'models', offline=True)

    def test_pinned_artifacts_are_rehashed_and_modified_weights_fail(self):
        directory = self.path / 'models' / REVISION
        directory.mkdir(parents=True)
        for name in MODEL_FILES:
            (directory / name).write_text('test artifact, never loaded as an actual model')
        mapping = {'0': 'positive', '1': 'negative', '2': 'neutral'}
        (directory / 'config.json').write_text(json.dumps({'id2label': mapping}))
        receipt = {'model_id': MODEL_ID, 'revision': REVISION, 'id2label': mapping,
                   'files_sha256': {name: file_hash(directory / name) for name in MODEL_FILES}}
        (directory / 'model_manifest.json').write_text(json.dumps(receipt))
        self.assertEqual(prepare_model(self.settings, self.path / 'models', offline=True)['revision'], REVISION)
        (directory / 'model.safetensors').write_text('changed')
        with self.assertRaisesRegex(ContractError, 'missing or changed'):
            prepare_model(self.settings, self.path / 'models', offline=True)

    def test_cache_reuses_normalized_content_across_records_and_restarts(self):
        scorer = TestScorer()
        path = self.path / 'cache.sqlite'
        with SentimentCache(path, synthetic=True) as cache:
            scores, report = score_unique(['Revenue rose', ' Revenue\nrose ', 'Costs fell'], scorer, cache, 1)
            self.assertEqual(report['unique_texts'], 2)
            self.assertEqual(report['inferred_texts'], 2)
            self.assertEqual(len(scorer.calls), 2)
            self.assertEqual(set(scores), {'Revenue rose', 'Costs fell'})
        with SentimentCache(path, synthetic=True) as cache:
            _, second = score_unique(['Revenue rose', 'Costs fell'], scorer, cache, 1)
            self.assertEqual(second['cache_hits'], 2)
            self.assertEqual(second['inferred_texts'], 0)
            self.assertEqual(len(scorer.calls), 2)

    def test_model_signature_change_invalidates_cached_scores(self):
        with SentimentCache(self.path / 'cache.sqlite', synthetic=True) as cache:
            cache.put_batch('model-one', [('Revenue rose', prediction())])
            self.assertIsNone(cache.get('model-two', 'Revenue rose'))
            self.assertIsNotNone(cache.get('model-one', 'Revenue rose'))
            self.assertNotIn('Revenue rose', '\n'.join(cache.db.iterdump()))

    def test_corrupted_score_cache_fails_instead_of_supplying_features(self):
        with SentimentCache(self.path / 'cache.sqlite', synthetic=True) as cache:
            cache.put_batch('model', [('Revenue rose', prediction())])
            with cache.db:
                cache.db.execute("UPDATE scores SET prediction='invalid-json'")
            with self.assertRaisesRegex(ContractError, 'Cached sentiment prediction is invalid'):
                cache.get('model', 'Revenue rose')

    def test_invalid_prediction_does_not_partially_commit_batch(self):
        with SentimentCache(self.path / 'cache.sqlite', synthetic=True) as cache:
            with self.assertRaises(ContractError):
                cache.put_batch('model', [('first', prediction()), ('second', {'positive': 2})])
            self.assertIsNone(cache.get('model', 'first'))

    def test_completed_batches_survive_interruption_and_resume(self):
        class InterruptScorer(TestScorer):
            def predict(self, texts):
                if self.calls:
                    raise RuntimeError('simulated interruption')
                return super().predict(texts)
        path = self.path / 'cache.sqlite'
        with SentimentCache(path, synthetic=True) as cache:
            with self.assertRaises(RuntimeError):
                score_unique(['a', 'b'], InterruptScorer(), cache, 1)
        with SentimentCache(path, synthetic=True) as cache:
            _, report = score_unique(['a', 'b'], TestScorer(), cache, 1)
            self.assertEqual(report['cache_hits'], 1)
            self.assertEqual(report['inferred_texts'], 1)

    def test_wrong_scorer_result_count_is_an_error(self):
        scorer = TestScorer()
        scorer.predict = lambda texts: []
        with SentimentCache(self.path / 'cache.sqlite', synthetic=True) as cache:
            with self.assertRaisesRegex(ContractError, 'wrong number'):
                score_unique(['a'], scorer, cache, 1)
            with self.assertRaisesRegex(ContractError, 'positive integer'):
                score_unique(['a'], scorer, cache, 0)

    def test_synthetic_cache_cannot_be_reused_for_research_data(self):
        path = self.path / 'cache.sqlite'
        with SentimentCache(path, synthetic=True):
            pass
        with self.assertRaisesRegex(ContractError, 'different synthetic mode'):
            SentimentCache(path, synthetic=False)

    def test_cache_clear_removes_all_derived_scores(self):
        with SentimentCache(self.path / 'cache.sqlite', synthetic=False) as cache:
            cache.put_batch('model', [('a', prediction()), ('b', prediction())])
            self.assertEqual(cache.clear(), 2)
            self.assertIsNone(cache.get('model', 'a'))

    def test_purge_command_also_clears_derived_score_cache(self):
        path = self.path / 'derived.sqlite'
        with SentimentCache(path, synthetic=False) as cache:
            cache.put_batch('model', [('Financial text', prediction())])
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(['purge-text', '--source', 'reddit', '--origin-id', 'removed-origin',
                                   '--cache', str(self.path / 'source.sqlite'), '--sentiment-cache', str(path)]), 0)
        self.assertEqual(json.loads(output.getvalue())['cleared_sentiment_scores'], 1)
        with SentimentCache(path, synthetic=False) as cache:
            self.assertIsNone(cache.get('model', 'Financial text'))

    def window(self, texts):
        return {'ticker': 'AAPL', 'target_session': '2024-11-26', 'partition': 'train',
                'window_start': '2024-11-25T21:00:00+00:00', 'decision_at': '2024-11-26T14:25:00+00:00', 'texts': texts}

    def text(self, text='First headline', record_id='one', source='newsapi'):
        return {'text': text, 'record_id': record_id, 'source': source, 'published_at': '2024-11-25T22:00:00+00:00',
                'available_at': '2024-11-25T22:01:00+00:00', 'availability_basis': 'observed'}

    def test_aggregate_source_means_population_dispersion_and_latest_availability(self):
        one, two, social = self.text(), self.text('Second headline', 'two'), self.text('Social', 'three', 'reddit')
        two['available_at'] = '2024-11-25T23:00:00+00:00'
        scores = {'First headline': prediction(.8, .1, .1), 'Second headline': prediction(.1, .8, .1, truncated=True),
                  'Social': prediction(.2, .3, .5)}
        rows, records = aggregate_windows([self.window([one, two, social])], scores)
        row = rows[0]
        self.assertEqual(row['newsapi_count'], 2)
        self.assertAlmostEqual(row['newsapi_mean_positive'], .45)
        self.assertAlmostEqual(row['newsapi_mean_polarity'], 0)
        self.assertAlmostEqual(row['newsapi_polarity_dispersion'], .7)
        self.assertEqual(row['newsapi_truncated_count'], 1)
        self.assertEqual(row['newsapi_latest_available_at'], two['available_at'])
        self.assertEqual(row['reddit_count'], 1)
        self.assertEqual(row['reddit_polarity_dispersion'], 0)
        self.assertEqual(len(records), 3)
        self.assertFalse({'label_up', 'target_return'} & set(row))

    def test_empty_windows_remain_missing_instead_of_neutral(self):
        rows, records = aggregate_windows([self.window([])], {})
        self.assertEqual(records, [])
        self.assertEqual(rows[0]['newsapi_count'], 0)
        self.assertTrue(rows[0]['newsapi_missing'])
        self.assertIsNone(rows[0]['newsapi_mean_polarity'])
        self.assertIsNone(rows[0]['newsapi_mean_neutral'])
        self.assertIsNone(rows[0]['newsapi_polarity_dispersion'])

    def test_aggregate_rejects_missing_scores_duplicate_records_and_cutoff_text(self):
        record = self.text()
        for texts, scores in (([record], {}), ([record, record], {'First headline': prediction()}),
              ([{**record, 'available_at': '2024-11-26T14:25:00+00:00'}], {'First headline': prediction()})):
            with self.subTest(texts=texts), self.assertRaises(ContractError):
                aggregate_windows([self.window(texts)], scores)

    def test_full_pipeline_scores_only_eligible_texts_and_preserves_partitions(self):
        result = self.run_dataset(self.dataset())
        self.assertEqual(result['included_text_records'], 4)
        self.assertEqual(result['ticker_sessions'], 12)
        scored = [json.loads(line) for line in (self.path / 'out/scored_texts.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['available_at'] < next(r['decision_at'] for r in
            [json.loads(line) for line in (self.path / 'out/daily_sentiment.jsonl').read_text().splitlines()]
            if r['ticker'] == row['ticker'] and r['target_session'] == row['target_session']) for row in scored))
        manifest = json.loads((self.path / 'out/manifest.json').read_text())
        self.assertTrue(manifest['synthetic_data'])
        self.assertEqual(manifest['model']['backend'], 'test-double-not-finbert')
        for name, expected in manifest['output_sha256'].items():
            self.assertEqual(file_hash(self.path / 'out' / name), expected)

    def test_second_run_uses_cache_and_produces_identical_feature_bytes(self):
        dataset = self.dataset()
        self.run_dataset(dataset, 'first')
        second = self.run_dataset(dataset, 'second')
        self.assertEqual(second['inferred_texts'], 0)
        self.assertEqual(second['cache_hits'], 4)
        for name in ('daily_sentiment.jsonl', 'daily_sentiment.csv', 'scored_texts.jsonl', 'labels.jsonl'):
            self.assertEqual((self.path / 'first' / name).read_bytes(), (self.path / 'second' / name).read_bytes())

    def test_dataset_tampering_or_wrong_config_fails_before_scoring(self):
        dataset = self.dataset()
        with self.assertRaisesRegex(ContractError, 'configuration differs'):
            verify_dataset(dataset, Config())
        with (dataset / 'texts.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ContractError, 'hash mismatch'):
            self.run_dataset(dataset)

    def test_no_eligible_text_does_not_load_model_or_fabricate_scores(self):
        def forbidden(*args, **kwargs):
            self.fail('Empty input must not load/download a model')
        result = self.run_dataset(self.dataset(empty=True), scorer_factory=forbidden)
        self.assertEqual(result['status'], 'no_eligible_text')
        self.assertEqual(result['inferred_texts'], 0)
        self.assertEqual((self.path / 'out/scored_texts.jsonl').read_text(), '')
        metadata = json.loads((self.path / 'out/manifest.json').read_text())
        self.assertEqual(metadata['model']['status'], 'not_loaded_no_eligible_text')

    def test_existing_output_is_not_overwritten(self):
        dataset = self.dataset()
        self.run_dataset(dataset)
        with self.assertRaisesRegex(ContractError, 'must be empty'):
            self.run_dataset(dataset)

    def test_smoke_examples_are_labeled_synthetic_and_use_the_scorer(self):
        result = run_smoke(ROOT / 'configs/sentiment.toml', self.path / 'models', self.path / 'smoke.sqlite',
                           self.path / 'smoke', scorer_factory=TestScorer)
        self.assertTrue(result['synthetic_input'])
        self.assertEqual(result['examples'], 6)
        self.assertEqual(result['inference_backend'], 'test-double-not-finbert')

    def test_cli_scores_empty_dataset_and_reports_offline_model_error(self):
        dataset = self.dataset(empty=True)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            self.assertEqual(main(['score-sentiment', '--dataset', str(dataset), '--config', str(FIXTURES / 'demo.toml'),
                '--out', str(self.path / 'cli-output')]), 0)
            self.assertEqual(main(['prepare-finbert', '--offline', '--model-root', str(self.path / 'missing')]), 2)
        self.assertIn('not available locally', errors.getvalue())


@unittest.skipUnless(os.environ.get('RUN_FINBERT_INTEGRATION') == '1', 'Set RUN_FINBERT_INTEGRATION=1 after preparing local model')
class RealFinBERTTests(unittest.TestCase):
    def test_cpu_inference_probabilities_repeatability_and_truncation(self):
        scorer = FinBERTScorer(SentimentConfig(), ROOT / 'data/cache/models/finbert', offline=True)
        texts = ['The company reported record profits and raised its earnings forecast.',
                 'The company reported substantial losses and cut its earnings forecast.']
        first, repeated = scorer.predict(texts), scorer.predict(texts)
        self.assertEqual(first, repeated)
        self.assertEqual(first[0]['label'], 'positive')
        self.assertEqual(first[1]['label'], 'negative')
        for row in first:
            self.assertAlmostEqual(sum(row[k] for k in LABELS), 1, places=5)
        truncated = scorer.predict(['Revenue rose. ' * 600])[0]
        self.assertTrue(truncated['truncated'])
        self.assertGreater(truncated['token_count'], 512)


if __name__ == '__main__':
    unittest.main()
