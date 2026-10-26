from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from sentiment_lab.contracts import ContractError
from sentiment_lab.evaluation import WeightedScores, evaluate, moving_block_dates, paired_bootstrap, verify_holdout
from sentiment_lab.features import read_json
from sentiment_lab.io import file_hash, write_json
from sentiment_lab.modeling import predict_holdout, train_models
from sentiment_lab.release import finalize, html_report, verified_evaluation
from test_modeling import small_config, synthetic_features

NUMPY = bool(importlib.util.find_spec('numpy'))
SKLEARN = bool(importlib.util.find_spec('sklearn'))
MPL = bool(importlib.util.find_spec('matplotlib'))


@unittest.skipUnless(NUMPY, 'NumPy needed for numerical uncertainty')
class BootstrapTests(unittest.TestCase):
    def test_tied_rank_auc_and_weighted_f1_known_example(self):
        engine = WeightedScores([0, 1, 0, 1], [.1, .5, .5, .9])
        auc, f1 = engine.calculate([1, 1, 1, 1])
        self.assertAlmostEqual(auc, .875)
        self.assertAlmostEqual(f1, .8)
        self.assertIsNone(WeightedScores([1, 1], [.1, .9]).calculate([1, 1])[0])

    @unittest.skipUnless(SKLEARN, 'scikit-learn oracle')
    def test_weighted_auc_matches_sklearn_sample_weight(self):
        from sklearn.metrics import roc_auc_score
        y, p, weights = [0, 1, 0, 1, 0], [.2, .2, .8, .9, .5], [2, 3, 1, 0, 4]
        self.assertAlmostEqual(WeightedScores(y, p).calculate(weights)[0], roc_auc_score(y, p, sample_weight=weights))

    def test_paired_identical_predictions_produce_exact_zero_interval(self):
        y = [0, 1] * 12
        scores = [.1, .8] * 12
        dates = [str(i // 2) for i in range(24)]
        comp = [{'id': 'primary', 'augmented': 'a', 'control': 'b', 'role': 'primary'}]
        result = paired_bootstrap(y, {'a': scores, 'b': scores}, dates, comp, replicates=100, block_length=5)
        primary = result['comparisons'][0]
        self.assertEqual(primary['delta_auc']['interval'], [0, 0])
        self.assertEqual(primary['delta_f1']['interval'], [0, 0])
        self.assertEqual(primary['delta_auc']['valid_replicates'], 100)

    def test_perfect_vs_reversed_remains_paired_with_positive_delta(self):
        y = [0, 1] * 12
        comp = [{'id': 'primary', 'augmented': 'a', 'control': 'b'}]
        result = paired_bootstrap(y, {'a': y, 'b': [1 - v for v in y]}, [f'{i//2:02d}' for i in range(24)], comp,
                                  replicates=100, block_length=3)
        self.assertEqual(result['comparisons'][0]['delta_auc']['interval'], [1, 1])

    def test_single_class_replicates_are_counted_without_fake_auc(self):
        result = paired_bootstrap([1] * 10, {'a': [.9] * 10, 'b': [.1] * 10}, [str(i) for i in range(10)],
            [{'id': 'p', 'augmented': 'a', 'control': 'b'}], replicates=20, block_length=2)
        self.assertEqual(result['invalid_single_class_auc_replicates'], 20)
        self.assertIsNone(result['comparisons'][0]['delta_auc']['interval'])
        self.assertEqual(result['comparisons'][0]['delta_f1']['valid_replicates'], 20)

    def test_block_sampler_has_contiguous_blocks_and_fixed_sample_length(self):
        import numpy as np
        result = moving_block_dates(12, 5, np.random.default_rng(5))
        self.assertEqual(len(result), 12)
        self.assertTrue(all(result[i + 1] - result[i] == 1 for i in (0, 1, 2, 3, 5, 6, 7, 8, 10)))
        with self.assertRaises(ContractError):
            moving_block_dates(3, 5, np.random.default_rng(5))

    def test_short_series_reports_unavailable_interval_and_seeded_results_repeat(self):
        comp = [{'id': 'p', 'augmented': 'a', 'control': 'b'}]
        short = paired_bootstrap([0, 1], {'a': [.1, .9], 'b': [.9, .1]}, ['a', 'b'], comp, replicates=20, block_length=5)
        self.assertEqual(short['status'], 'insufficient_distinct_sessions')
        self.assertIsNone(short['comparisons'][0]['delta_auc']['interval'])
        args = ([0, 1] * 10, {'a': [.2, .8] * 10, 'b': [.3, .7] * 10}, [str(i) for i in range(20)], comp)
        self.assertEqual(paired_bootstrap(*args, replicates=20), paired_bootstrap(*args, replicates=20))


@unittest.skipUnless(SKLEARN, 'Modeling dependencies needed for frozen evaluation integration')
class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.features = cls.root / 'features'
        synthetic_features(cls.features)
        cls.config = cls.root / 'config.toml'; small_config(cls.config)
        cls.experiment, cls.predictions, cls.evaluation = cls.root / 'models', cls.root / 'predictions', cls.root / 'evaluation'
        train_models(cls.features, cls.config, cls.experiment)
        predict_holdout(cls.features, cls.experiment, cls.predictions)
        evaluate(cls.features, cls.experiment, cls.predictions, cls.evaluation)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_frozen_holdout_evaluated_on_full_matched_sample_with_data_gate(self):
        summary = read_json(self.evaluation / 'summary.json')
        self.assertEqual(summary['test_rows'], 20)
        self.assertEqual(summary['models_and_baselines'], 17)
        self.assertEqual(summary['research_status'], 'inconclusive_data_gate_unmet')
        self.assertFalse(summary['model_refit'])
        self.assertEqual(len(read_json(self.evaluation / 'uncertainty.json')['results']), 3)
        self.assertEqual(read_json(self.evaluation / 'text_present_subset.json')['role'].split(';')[0], 'exploratory')

    def test_evaluation_replay_is_numerically_identical(self):
        target = self.root / 'evaluation-replayed'
        evaluate(self.features, self.experiment, self.predictions, target)
        for name in ('test_metrics.json', 'uncertainty.json', 'summary.json'):
            self.assertEqual(read_json(self.evaluation / name), read_json(target / name))

    def test_label_tampering_is_rejected_before_results(self):
        path = self.features / 'labels_test.jsonl'; original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ContractError, 'hash mismatch'):
                verify_holdout(self.features, self.experiment, self.predictions)
        finally:
            path.write_bytes(original)

    def test_prediction_receipt_tampering_is_rejected(self):
        path = self.predictions / 'manifest.json'; original = path.read_bytes()
        try:
            receipt = read_json(path); receipt['freeze_id'] = 'wrong'; write_json(path, receipt)
            with self.assertRaisesRegex(ContractError, 'not bound'):
                verify_holdout(self.features, self.experiment, self.predictions)
        finally:
            path.write_bytes(original)

    def test_duplicate_rehashed_predictions_are_rejected_semantically(self):
        path, receipt_path = self.predictions / 'test_predictions.jsonl', self.predictions / 'manifest.json'
        original, original_receipt = path.read_bytes(), receipt_path.read_bytes()
        try:
            path.write_bytes(original + original.splitlines(keepends=True)[0])
            receipt = read_json(receipt_path); receipt['prediction_sha256'] = file_hash(path); write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ContractError, 'duplicate'):
                verify_holdout(self.features, self.experiment, self.predictions)
        finally:
            path.write_bytes(original); receipt_path.write_bytes(original_receipt)

    def test_manifest_hashes_cover_all_evaluation_artifacts(self):
        manifest = read_json(self.evaluation / 'manifest.json')
        for name, expected in manifest['output_sha256'].items():
            self.assertEqual(file_hash(self.evaluation / name), expected)

    @unittest.skipUnless(MPL, 'Matplotlib needed for final report')
    def test_final_report_cards_bundle_and_clean_reproduction(self):
        target = self.root / 'release'
        result = finalize(self.features, self.experiment, self.predictions, self.evaluation, target)
        self.assertEqual(result['reproduction'], 'passed')
        self.assertIn('inconclusive', (target / 'report.md').read_text())
        self.assertIn('data:image/png;base64,', (target / 'report.html').read_text())
        self.assertFalse(read_json(target / 'manifest.json')['empirical_sentiment_study_complete'])
        with ZipFile(target / 'handoff.zip') as bundle:
            self.assertIn('source/inputs/m4/labels_test.jsonl', bundle.namelist())
            self.assertFalse(any('/.env' in name or '/.venv/' in name or '/data/cache/' in name for name in bundle.namelist()))
        for name, expected in read_json(target / 'manifest.json')['files_sha256'].items():
            self.assertEqual(file_hash(target / name), expected)


if __name__ == '__main__':
    unittest.main()
