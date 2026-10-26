from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date, timedelta
from io import StringIO
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sentiment_lab.classifiers import (baseline_predictions, baseline_spec, export_estimator, fit_classifier,
    matrix, metrics, predict_state, probabilities)
from sentiment_lab.cli import main
from sentiment_lab.contracts import ContractError
from sentiment_lab.features import FEATURE_SETS, KEYS, feature_schema, read_json, read_jsonl
from sentiment_lab.indicators import FeatureConfig
from sentiment_lab.io import file_hash, write_json, write_jsonl
from sentiment_lab.modeling import candidate_rank, predict_holdout, train_models, verified_freeze
from sentiment_lab.modeling_data import ModelingConfig, expanding_folds, load_features, row_key


ROOT = Path(__file__).resolve().parents[1]
SKLEARN_AVAILABLE = bool(importlib.util.find_spec('sklearn'))


def synthetic_features(directory: Path):
    """Model-contract fixture only; no actual prices, news or research result."""
    directory.mkdir()
    combined, labels = [], []
    day = date(2024, 1, 2)
    for index in range(60):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        partition = 'train' if index < 40 else 'validation' if index < 50 else 'test'
        for ticker_index, symbol in enumerate(('AAPL', 'MSFT')):
            signal = math.sin(index * .8) + .2 * ticker_index
            row = {'ticker': symbol, 'target_session': day.isoformat(), 'partition': partition,
                   'decision_at': day.isoformat() + 'T14:25:00+00:00',
                   'return_1d': .01 * signal, 'return_5d': .02 * math.cos(index),
                   'momentum_10d': .03 * signal, 'volatility_10d': .02 + .01 * abs(signal),
                   'rsi_5d': 50 + 20 * signal}
            for source in ('newsapi', 'reddit'):
                absent = index % 4 == 0
                positive, negative = (.7, .1) if signal > 0 else (.1, .7)
                values = {'count': int(not absent), 'missing': absent, 'mean_positive': positive,
                          'mean_negative': negative, 'mean_neutral': .2,
                          'mean_polarity': positive - negative, 'polarity_dispersion': 0.}
                for key, value in values.items():
                    row[source + '_' + key] = None if absent and key not in ('count', 'missing') else value
            combined.append(row)
            labels.append({'ticker': symbol, 'target_session': day.isoformat(), 'partition': partition,
                           'label_up': int(signal > 0), 'target_return': .01 * signal,
                           'label_available_at': day.isoformat() + 'T21:01:00+00:00'})
        day += timedelta(days=1)
    for name, columns in FEATURE_SETS.items():
        write_jsonl(directory / f'{name}.jsonl', [{key: row[key] for key in KEYS + columns} for row in combined])
    for partition in ('train', 'validation', 'test'):
        write_jsonl(directory / f'labels_{partition}.jsonl', [row for row in labels if row['partition'] == partition])
    write_jsonl(directory / 'labels.jsonl', labels)
    write_json(directory / 'feature_schema.json', feature_schema(FeatureConfig()))
    write_json(directory / 'leakage_audit.json', {'status': 'passed', 'test_fixture': True})
    write_json(directory / 'summary.json', {'status': 'ready', 'synthetic_data': True})
    write_json(directory / 'feature_feasibility.json', {'decision': 'engineering_demonstration_only'})
    write_json(directory / 'manifest.json', {'synthetic_data': True,
        'output_sha256': {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}})
    return combined, labels


def small_config(path):
    path.write_text('''schema_version = 1
cv_folds = 2
minimum_fold_train_sessions = 3
logistic_c = [0.1, 1.0]
forest_depths = [2, 3]
forest_min_leaf = [2]
forest_trees = 10
coin_seeds = 3
permutation_repeats = 1
''')


class ModelingContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.features = self.path / 'features'
        self.rows, self.labels = synthetic_features(self.features)
        self.config = self.path / 'modeling.toml'
        small_config(self.config)
        self.settings = ModelingConfig.load(self.config)

    def test_configuration_search_limits_and_frozen_threshold(self):
        for config in (replace(self.settings, threshold=.7), replace(self.settings, cv_folds=True),
                       replace(self.settings, logistic_c=tuple(range(1, 10))),
                       replace(self.settings, forest_depths=(1, 2, 3), forest_min_leaf=(1, 2, 3)),
                       replace(self.settings, logistic_c=(float('nan'),)),
                       replace(self.settings, forest_depths=(2, 2))):
            with self.subTest(config=config), self.assertRaises(ContractError):
                config.validate()
        self.assertEqual(len(ModelingConfig().candidates('logistic_regression')), 4)
        self.assertEqual(len(ModelingConfig().candidates('random_forest')), 4)

    def test_loader_does_not_parse_all_partition_or_test_label_files(self):
        (self.features / 'labels_test.jsonl').write_text('not-json deliberately unread')
        (self.features / 'labels.jsonl').write_text('not-json deliberately unread')
        bundle = load_features(self.features)
        self.assertEqual(len(bundle['labels']), 100)
        self.assertTrue(all(row['partition'] != 'test' for row in bundle['labels'].values()))

    def test_changed_development_hash_fails(self):
        with (self.features / 'labels_train.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ContractError, 'hash mismatch'):
            load_features(self.features)

    def test_old_feature_export_requests_partition_label_rebuild(self):
        manifest = read_json(self.features / 'manifest.json')
        del manifest['output_sha256']['labels_train.jsonl']
        write_json(self.features / 'manifest.json', manifest)
        with self.assertRaisesRegex(ContractError, 'rebuild M4'):
            load_features(self.features)

    def test_expanding_folds_preserve_whole_dates_and_refuse_insufficient_history(self):
        bundle = load_features(self.features)
        folds = expanding_folds(bundle['parts']['train'], bundle['labels'], self.settings)
        self.assertEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(max(r['target_session'] for r in fold['training']), min(r['target_session'] for r in fold['evaluation']))
            self.assertEqual(len(fold['evaluation']) % 2, 0)
            self.assertTrue(all(r['partition'] == 'train' for r in fold['training'] + fold['evaluation']))
        self.assertGreater(len(folds[1]['training']), len(folds[0]['training']))
        with self.assertRaisesRegex(ContractError, 'Insufficient'):
            expanding_folds(bundle['parts']['train'], bundle['labels'], ModelingConfig())

    def test_fold_boundary_unresolved_labels_are_purged_at_equality(self):
        bundle = load_features(self.features)
        fold = expanding_folds(bundle['parts']['train'], bundle['labels'], self.settings)[0]
        last = fold['training'][-1]
        bundle['labels'][row_key(last)]['label_available_at'] = fold['boundary']
        changed = expanding_folds(bundle['parts']['train'], bundle['labels'], self.settings)[0]
        self.assertEqual(changed['purged_rows'], 1)
        self.assertNotIn(last, changed['training'])

    def test_frozen_rank_uses_validation_then_cv_then_grid_order(self):
        first = {'validation': {'roc_auc': .6}, 'cv_mean_auc': .5, 'candidate_index': 0}
        second = {'validation': {'roc_auc': .6}, 'cv_mean_auc': .7, 'candidate_index': 1}
        self.assertGreater(candidate_rank(second), candidate_rank(first))
        tied = {**second, 'candidate_index': 2}
        self.assertGreater(candidate_rank(second), candidate_rank(tied))
        self.assertGreater(candidate_rank({**first, 'validation': {'roc_auc': None}}),
                           candidate_rank({**second, 'validation': {'roc_auc': None}}))

    def test_coin_scores_are_seeded_order_invariant_and_label_independent(self):
        spec = baseline_spec([0, 1, 1, 1], self.settings)
        first = baseline_predictions(self.rows[:10], spec)
        second = baseline_predictions(list(reversed(self.rows[:10])), spec)
        for name in first:
            self.assertEqual(first[name], list(reversed(second[name])))
        self.assertEqual(first['training_majority'], [.75] * 10)
        self.assertEqual(first['always_up'], [1.] * 10)
        self.assertEqual(first['zero_drift_no_change'], [0.] * 10)
        self.assertNotEqual(first['coin_seed_42'], first['coin_seed_43'])

    def test_rehashed_swapped_feature_keys_or_values_fail(self):
        filename = 'price_only.jsonl'
        path = self.features / filename
        rows = read_jsonl(path)
        rows[0]['return_1d'] = 999
        write_jsonl(path, rows)
        manifest = read_json(self.features / 'manifest.json')
        manifest['output_sha256'][filename] = file_hash(path)
        write_json(self.features / 'manifest.json', manifest)
        with self.assertRaisesRegex(ContractError, 'changes values'):
            load_features(self.features)


@unittest.skipUnless(SKLEARN_AVAILABLE, 'Install modeling dependencies for estimator integration')
class EstimatorTests(unittest.TestCase):
    def test_f1_auc_constant_scores_and_undefined_single_class_auc(self):
        result = metrics([0, 1, 0, 1], [0., 1., 0., 1.])
        self.assertEqual(result['roc_auc'], 1.)
        self.assertEqual(result['f1'], 1.)
        self.assertEqual(metrics([0, 1], [0., 0.])['roc_auc'], .5)
        self.assertEqual(metrics([0, 1], [0., 0.])['f1'], 0.)
        self.assertIsNone(metrics([1, 1], [.1, .9])['roc_auc'])
        with self.assertRaises(ContractError):
            metrics([0, 1], [float('nan'), .5])

    def test_portable_logistic_and_forest_match_native_predictions(self):
        settings = replace(ModelingConfig(), forest_trees=10)
        x = [[-2.], [-1.], [0.], [1.], [2.], [3.]]
        y = [0, 0, 0, 1, 1, 1]
        for family, params in (('logistic_regression', {'C': 1.}),
                               ('random_forest', {'max_depth': 3, 'min_samples_leaf': 1})):
            with self.subTest(family=family):
                estimator = fit_classifier(family, params, settings, x, y)
                native = probabilities(estimator, x)
                state = json.loads(json.dumps(export_estimator(estimator, family, ('return_1d',))))
                portable = predict_state(state, x)
                for left, right in zip(native, portable):
                    self.assertAlmostEqual(left, right, places=12)
        with self.assertRaisesRegex(ContractError, 'both target classes'):
            fit_classifier('logistic_regression', {'C': 1.}, settings, x, [1] * len(x))


@unittest.skipUnless(SKLEARN_AVAILABLE, 'Install modeling dependencies for training integration')
class ModelRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.features = self.path / 'features'
        self.rows, self.labels = synthetic_features(self.features)
        self.config = self.path / 'modeling.toml'
        small_config(self.config)

    def train(self, name='run'):
        return train_models(self.features, self.config, self.path / name)

    def test_training_freezes_ten_groups_equal_budgets_and_keeps_holdout_unpredicted(self):
        (self.features / 'labels_test.jsonl').write_text('invalid JSON: training must never parse this')
        (self.features / 'labels.jsonl').write_text('invalid JSON: training must never parse this')
        report = self.train()
        self.assertEqual(report['trained_models'], 10)
        self.assertEqual(report['attempted_candidates'], 20)
        self.assertFalse(report['test_labels_parsed'])
        self.assertFalse((self.path / 'run/test_predictions.jsonl').exists())
        freeze = verified_freeze(self.path / 'run')
        self.assertEqual(len(freeze['model_files']), 10)
        predicted = read_jsonl(self.path / 'run/validation_predictions.jsonl')
        self.assertTrue(all(row['partition'] == 'validation' for row in predicted))
        self.assertEqual({r['model_id'] for r in predicted if not r['model_id'].startswith('baseline:')}, set(freeze['model_files']))

    def test_fold_processors_have_their_own_earlier_training_receipts(self):
        self.train()
        model = read_json(self.path / 'run/model_logistic_regression_combined.json')
        outer = model['preprocessing']['fit']
        for fold in model['fold_preprocessing']:
            fit = fold['preprocessing']['fit']
            self.assertLess(fit['latest_fit_decision'], fold['boundary'])
            self.assertLess(fit['fit_rows'], outer['fit_rows'])
            self.assertNotEqual(fit['training_feature_sha256'], outer['training_feature_sha256'])
        self.assertEqual(model['interpretation']['method'], 'signed_logistic_coefficients')
        forest = read_json(self.path / 'run/model_random_forest_combined.json')
        self.assertEqual(forest['interpretation']['method'], 'validation_permutation_auc')

    def test_single_class_validation_auc_is_null_and_first_setting_is_explicitly_untuned(self):
        filename = 'labels_validation.jsonl'
        path = self.features / filename
        rows = read_jsonl(path)
        for row in rows:
            row['label_up'] = 1
        write_jsonl(path, rows)
        manifest = read_json(self.features / 'manifest.json')
        manifest['output_sha256'][filename] = file_hash(path)
        write_json(self.features / 'manifest.json', manifest)
        self.train()
        model = read_json(self.path / 'run/model_logistic_regression_price_only.json')
        self.assertIsNone(model['selected_candidate']['validation']['roc_auc'])
        self.assertEqual(model['selected_candidate']['candidate_index'], 0)
        self.assertEqual(model['selection'], 'first_declared_candidate_untuned_single_class_validation')
        forest = read_json(self.path / 'run/model_random_forest_price_only.json')
        self.assertEqual(forest['interpretation']['status'], 'unavailable_single_class_validation')

    def test_single_class_outer_training_fails_before_freezing(self):
        filename = 'labels_train.jsonl'
        path = self.features / filename
        rows = read_jsonl(path)
        for row in rows:
            row['label_up'] = 1
        write_jsonl(path, rows)
        manifest = read_json(self.features / 'manifest.json')
        manifest['output_sha256'][filename] = file_hash(path)
        write_json(self.features / 'manifest.json', manifest)
        with self.assertRaisesRegex(ContractError, 'both target classes'):
            self.train()
        self.assertFalse((self.path / 'run/freeze.json').exists())

    def test_post_freeze_prediction_works_without_any_label_files(self):
        self.train()
        for name in ('labels.jsonl', 'labels_train.jsonl', 'labels_validation.jsonl', 'labels_test.jsonl'):
            (self.features / name).write_text('do not parse any labels for prediction')
        result = predict_holdout(self.features, self.path / 'run', self.path / 'predicted')
        self.assertFalse(result['test_labels_parsed'])
        self.assertEqual(result['rows_per_model'], 20)
        rows = read_jsonl(self.path / 'predicted/test_predictions.jsonl')
        self.assertTrue(all(r['partition'] == 'test' for r in rows))
        self.assertTrue(all(0 <= r['up_probability'] <= 1 for r in rows))
        self.assertTrue(all('label_up' not in r for r in rows))
        self.assertFalse((self.path / 'predicted/test_metrics.json').exists())

    def test_modified_frozen_artifact_blocks_holdout_prediction(self):
        self.train()
        path = self.path / 'run/model_logistic_regression_price_only.json'
        with path.open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ContractError, 'missing or changed'):
            predict_holdout(self.features, self.path / 'run', self.path / 'blocked')

    def test_rerun_matches_selected_models_and_validation_prediction_bytes(self):
        self.train('one')
        self.train('two')
        for name in ('validation_predictions.jsonl', 'candidate_metrics.json', 'model_random_forest_combined.json',
                     'model_logistic_regression_combined.json'):
            self.assertEqual((self.path / 'one' / name).read_bytes(), (self.path / 'two' / name).read_bytes())

    def test_changed_holdout_values_cannot_change_development_fits_or_selection(self):
        self.train('one')
        manifest = read_json(self.features / 'manifest.json')
        for name, columns in FEATURE_SETS.items():
            filename = name + '.jsonl'
            path = self.features / filename
            rows = read_jsonl(path)
            for row in rows:
                if row['partition'] == 'test':
                    for column in columns:
                        if column.endswith('_missing'):
                            continue
                        if row[column] is not None:
                            row[column] += 5000
            write_jsonl(path, rows)
            manifest['output_sha256'][filename] = file_hash(path)
        write_json(self.features / 'manifest.json', manifest)
        self.train('two')
        for name in ('candidate_metrics.json', 'validation_predictions.jsonl', 'model_logistic_regression_combined.json', 'model_random_forest_combined.json'):
            self.assertEqual((self.path / 'one' / name).read_bytes(), (self.path / 'two' / name).read_bytes())

    def test_cli_invalid_old_artifacts_fail_clearly_and_existing_output_is_preserved(self):
        self.train()
        with self.assertRaisesRegex(ContractError, 'must be empty'):
            self.train()
        (self.features / 'labels_train.jsonl').write_text('invalid')
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            self.assertEqual(main(['train-models', '--features', str(self.features), '--modeling', str(self.config),
                                   '--out', str(self.path / 'cli')]), 2)
        self.assertIn('hash mismatch', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
