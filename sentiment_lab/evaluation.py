"""M6: verify the frozen holdout and estimate paired date-block uncertainty."""
from collections import defaultdict
import math
from pathlib import Path
import statistics

from .classifiers import metrics
from .contracts import ContractError, timestamp
from .features import KEYS, read_json, read_jsonl
from .io import file_hash, write_json, write_jsonl
from .modeling import empty_output, verified_freeze
from .modeling_data import load_features, row_key, verify_file
from .acquisition import write_csv
from .store import digest, utc_now


class WeightedScores:
    """Weighted rank-sum AUC, grouping ties exactly, and weighted fixed-cutoff F1."""
    def __init__(self, labels, scores):
        import numpy as np
        self.np = np
        self.y = np.asarray(labels, dtype=float)
        self.scores = np.asarray(scores, dtype=float)
        if len(self.y) != len(self.scores) or not len(self.y) or not np.all(np.isfinite(self.scores)) or np.any((self.scores < 0) | (self.scores > 1)) or not np.all(np.isin(self.y, [0, 1])):
            raise ContractError('Uncertainty needs aligned binary labels and finite probabilities')
        self.order = np.argsort(self.scores, kind='stable')
        ordered = self.scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        self.predicted = self.scores >= .5

    def calculate(self, weights):
        np = self.np
        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.y.shape or np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ContractError('Bootstrap weights must be aligned finite nonnegative numbers')
        pos = weights * self.y
        neg = weights * (1 - self.y)
        p, n = pos.sum(), neg.sum()
        auc = None
        if p > 0 and n > 0:
            pg = np.add.reduceat(pos[self.order], self.starts)
            ng = np.add.reduceat(neg[self.order], self.starts)
            auc = float((pg * (np.cumsum(ng) - .5 * ng)).sum() / (p * n))
        tp = pos[self.predicted].sum()
        denominator = p + weights[self.predicted].sum()
        return auc, float(2 * tp / denominator) if denominator else 0.0


def moving_block_dates(n_dates, length, rng):
    import numpy as np
    if type(n_dates) is not int or type(length) is not int or not 1 <= length <= n_dates:
        raise ContractError('Moving blocks must fit the available distinct sessions')
    starts = rng.integers(0, n_dates - length + 1, size=math.ceil(n_dates / length))
    return np.concatenate([np.arange(start, start + length) for start in starts])[:n_dates]


def paired_bootstrap(labels, model_scores, dates, comparisons, replicates=2000, block_length=5, seed=42, confidence=.95):
    import numpy as np
    if type(replicates) is not int or replicates < 1 or not 0 < confidence < 1:
        raise ContractError('Invalid bootstrap replicate count or confidence level')
    unique_dates = sorted(set(dates))
    if len(dates) != len(labels) or not unique_dates:
        raise ContractError('Bootstrap dates must align with labels')
    if block_length > len(unique_dates):
        return {'status': 'insufficient_distinct_sessions', 'block_sessions': block_length,
                'distinct_sessions': len(unique_dates), 'comparisons': [dict(comp) | {
                    'delta_' + metric: {'estimate': None, 'interval': None, 'valid_replicates': 0, 'invalid_replicates': replicates}
                    for metric in ('auc', 'f1')} for comp in comparisons]}
    date_index = {day: i for i, day in enumerate(unique_dates)}
    row_dates = np.asarray([date_index[day] for day in dates])
    required = {model for comp in comparisons for model in (comp['augmented'], comp['control'])}
    engines = {model: WeightedScores(labels, model_scores[model]) for model in sorted(required)}
    points = {model: engine.calculate(np.ones(len(labels))) for model, engine in engines.items()}
    collected = {comp['id']: {'auc': [], 'f1': []} for comp in comparisons}
    rng = np.random.default_rng(seed + block_length)
    invalid_auc = 0
    for _ in range(replicates):
        indices = moving_block_dates(len(unique_dates), block_length, rng)
        weights = np.bincount(indices, minlength=len(unique_dates))[row_dates]
        values = {model: engine.calculate(weights) for model, engine in engines.items()}
        valid = all(value[0] is not None for value in values.values())
        invalid_auc += int(not valid)
        for comp in comparisons:
            augmented, control = values[comp['augmented']], values[comp['control']]
            if valid:
                collected[comp['id']]['auc'].append(augmented[0] - control[0])
            collected[comp['id']]['f1'].append(augmented[1] - control[1])
    result = []
    tail = (1 - confidence) / 2
    for comp in comparisons:
        item = dict(comp)
        for metric, index in (('auc', 0), ('f1', 1)):
            a, c = points[comp['augmented']][index], points[comp['control']][index]
            values = collected[comp['id']][metric]
            interval = [float(v) for v in np.quantile(values, [tail, 1 - tail], method='linear')] if values else None
            item['delta_' + metric] = {'estimate': a - c if a is not None and c is not None else None,
                 'interval': interval, 'valid_replicates': len(values), 'invalid_replicates': replicates - len(values)}
        result.append(item)
    return {'status': 'computed', 'method': 'paired_non_circular_moving_block_percentile',
            'block_sessions': block_length, 'distinct_sessions': len(unique_dates), 'replicates': replicates,
            'seed': seed + block_length, 'confidence_level': confidence, 'invalid_single_class_auc_replicates': invalid_auc,
            'resampling_unit': 'consecutive session blocks; all ticker rows and model scores jointly resampled',
            'comparisons': result,
            'assumptions': 'Approximate local stationarity and dependence captured by the chosen block length; intervals are not causal or profitability evidence.'}


def verify_holdout(features, experiment, predictions):
    freeze = verified_freeze(experiment)
    bundle = load_features(features, load_development_labels=False)
    receipt = read_json(predictions / 'manifest.json')
    if (receipt.get('freeze_id') != freeze['freeze_id'] or receipt.get('freeze_sha256') != file_hash(experiment / 'freeze.json') or
        receipt.get('feature_manifest_sha256') != bundle['feature_manifest_sha256'] or
        freeze['feature_manifest_sha256'] != bundle['feature_manifest_sha256'] or
        receipt.get('prediction_sha256') != file_hash(predictions / 'test_predictions.jsonl')):
        raise ContractError('Holdout predictions are not bound to the frozen experiment and M4 input')
    rows = bundle['parts']['test']
    verify_file(features, bundle['manifest'], 'labels_test.jsonl')
    labels = read_jsonl(features / 'labels_test.jsonl')
    expected = {row_key(row) for row in rows}
    if len(labels) != len(expected) or {row_key(row) for row in labels} != expected:
        raise ContractError('Test outcome keys are missing, duplicated or mismatched')
    labels_by_key = {}
    for row in labels:
        value = row.get('target_return')
        if row.get('partition') != 'test' or type(row.get('label_up')) is not int or row['label_up'] not in (0, 1) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or row['label_up'] != int(value > 0):
            raise ContractError('Test outcomes must be valid binary directions consistent with finite returns')
        if timestamp(row['label_available_at']) > utc_now():
            raise ContractError('A test outcome is not yet available at evaluation time')
        labels_by_key[row_key(row)] = row
    baseline = read_json(experiment / 'baselines.json')
    expected_models = set(freeze['model_files']) | {'baseline:' + name for name in ('training_majority', 'always_up', 'zero_drift_no_change', 'momentum')} | {f'baseline:coin_seed_{seed}' for seed in baseline['coin_seeds']}
    grouped = defaultdict(dict)
    row_metadata = {row_key(row): row for row in rows}
    for row in read_jsonl(predictions / 'test_predictions.jsonl'):
        try:
            model, key, score = row['model_id'], row_key(row), row['up_probability']
            if model not in expected_models or key not in expected or key in grouped[model]:
                raise ContractError('Test predictions have unknown models, duplicate or unmatched rows')
            if any(row[field] != row_metadata[key][field] for field in KEYS) or row.get('prediction_partition') != 'test':
                raise ContractError('A test prediction has inconsistent partition/cutoff lineage')
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1 or row['predicted_up'] != int(score >= .5):
                raise ContractError('A test prediction score or threshold decision is invalid')
            grouped[model][key] = score
        except KeyError as exc:
            raise ContractError('Test prediction lacks required fields') from exc
    if set(grouped) != expected_models or any(set(grouped[model]) != expected for model in grouped):
        raise ContractError('Every model/baseline must predict the exact complete held-out sample')
    return freeze, bundle, labels_by_key, {model: [values[row_key(row)] for row in rows] for model, values in sorted(grouped.items())}


def evaluate(features: Path, experiment: Path, predictions: Path, output: Path):
    empty_output(output)
    freeze, bundle, labels, scores = verify_holdout(features, experiment, predictions)
    protocol = read_json(experiment / 'protocol.json')
    plan = protocol['m6_evaluation_plan']
    if plan != {'confidence_level': .95, 'paired_moving_block_bootstrap_replicates': 2000,
        'block_sessions': 5, 'sensitivity_block_sessions': [1, 10], 'resample_dates_jointly_across_tickers': True,
        'f1_threshold': .5, 'secondary_comparisons': 'exploratory; no post-test winner redefinition'}:
        raise ContractError('M6 implementation does not match the frozen evaluation plan')
    rows = bundle['parts']['test']
    y = [labels[row_key(row)]['label_up'] for row in rows]
    def score_slice(indices):
        if not indices:
            return []
        return [{'model_id': model, **metrics([y[i] for i in indices], [values[i] for i in indices])} for model, values in scores.items()]
    full = score_slice(list(range(len(rows))))
    ticker_metrics = [{'ticker': symbol, **record} for symbol in sorted({r['ticker'] for r in rows})
                      for record in score_slice([i for i, row in enumerate(rows) if row['ticker'] == symbol])]
    macro = [{'model_id': model, **{metric: statistics.fmean(values) if values else None for metric in ('roc_auc', 'f1', 'accuracy')
              for values in [[r[metric] for r in ticker_metrics if r['model_id'] == model and r[metric] is not None]]}}
             for model in scores]
    dates = sorted({r['target_session'] for r in rows})
    midpoint = len(dates) // 2
    periods = [{'period': name, **record} for name, days in (('first_half', set(dates[:midpoint])), ('second_half', set(dates[midpoint:])))
               for record in score_slice([i for i, row in enumerate(rows) if row['target_session'] in days])]
    present = [i for i, row in enumerate(rows) if not row['newsapi_missing'] or not row['reddit_missing']]
    text_subset = {'status': 'available' if present else 'no_eligible_text', 'rows': len(present),
                   'metrics': score_slice(present) if present else [], 'role': 'exploratory; never replaces the full-sample primary endpoint'}
    primary = protocol['primary_comparison']
    comparisons = [{'id': family + ':' + feature_set, 'augmented': family + ':' + feature_set,
                    'control': family + ':price_only', 'role': 'primary' if family + ':' + feature_set == primary['augmented'] else 'exploratory'}
                   for family in ('logistic_regression', 'random_forest') for feature_set in ('combined', 'price_news', 'price_reddit')]
    uncertainty = [paired_bootstrap(y, scores, [r['target_session'] for r in rows], comparisons,
                    replicates=plan['paired_moving_block_bootstrap_replicates'], block_length=length,
                    seed=protocol['settings']['seed'], confidence=plan['confidence_level'])
                   for length in (plan['block_sessions'], *plan['sensitivity_block_sessions'])]
    main = next(c for c in uncertainty[0]['comparisons'] if c['role'] == 'primary')
    interval = main['delta_auc']['interval']
    evidence = 'positive_increment_interval_excludes_zero' if interval and interval[0] > 0 else 'no_positive_increment_evidence'
    gate = bundle['source_data_gate']
    can_test_sentiment = gate == 'full_study_feasible' and freeze.get('synthetic_data') is not True and len(present) > 0
    conclusion = {'research_status': evidence if can_test_sentiment else 'inconclusive_data_gate_unmet',
                  'statistical_diagnostic': evidence, 'primary_comparison': main,
                  'sentiment_hypothesis_testable': can_test_sentiment,
                  'statement': ('The original sentiment-value hypothesis remains unanswered: eligible news/Reddit history is missing or the study data gate is unmet. Numerical ablations are engineering diagnostics.'
                     if not can_test_sentiment else 'Report the paired primary interval and its effect size; secondary views remain exploratory and do not establish trading profitability.')}
    lookup = {row['model_id']: row for row in full}
    coin = [row for row in full if row['model_id'].startswith('baseline:coin_seed_')]
    coin_variability = {metric: {'mean': statistics.fmean(values), 'std': statistics.pstdev(values)}
                        for metric in ('roc_auc', 'f1') for values in [[r[metric] for r in coin if r[metric] is not None]] if values}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'test_metrics.json', {'partition': 'test', 'models': full})
    write_csv(output / 'test_metrics.csv', full, ['model_id', 'rows', 'positive_prevalence', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall'])
    write_json(output / 'per_ticker.json', {'scope': 'exploratory', 'metrics': ticker_metrics, 'macro_metrics': macro})
    write_json(output / 'period_sensitivity.json', {'scope': 'exploratory', 'metrics': periods})
    write_json(output / 'text_present_subset.json', text_subset)
    write_json(output / 'uncertainty.json', {'primary_block_sessions': 5, 'results': uncertainty})
    write_json(output / 'coin_variability.json', {'seeds': len(coin), 'metrics': coin_variability})
    write_json(output / 'conclusion.json', conclusion)
    write_jsonl(output / 'paired_primary.jsonl', [{key: row[key] for key in KEYS} | {
        'label_up': y[i], 'augmented_probability': scores[primary['augmented']][i], 'control_probability': scores[primary['control']][i]}
        for i, row in enumerate(rows)])
    audit = {'status': 'passed', 'freeze_id': freeze['freeze_id'], 'models_refit': False, 'settings_changed': False,
             'test_label_hash_verified': True, 'exact_model_sample_match': True, 'date_blocks_paired_across_models_and_tickers': True,
             'threshold': .5, 'primary_fixed_before_test': True, 'secondary_results_exploratory': True,
             'source_data_gate': gate, 'test_labels_revealed_at': utc_now().isoformat()}
    write_json(output / 'evaluation_audit.json', audit)
    summary = {'status': 'evaluated_frozen_holdout', 'test_rows': len(rows), 'distinct_test_sessions': len(dates),
               'test_start': dates[0], 'test_end': dates[-1], 'models_and_baselines': len(scores),
               'positive_prevalence': sum(y) / len(y), 'source_data_gate': gate,
               'synthetic_data': freeze.get('synthetic_data'), 'primary_delta_auc': main['delta_auc'],
               'primary_delta_f1': main['delta_f1'], 'research_status': conclusion['research_status'],
               'sentiment_covered_test_rows': len(present), 'model_refit': False,
               'primary_control_metrics': lookup[primary['control']], 'primary_augmented_metrics': lookup[primary['augmented']]}
    write_json(output / 'summary.json', summary)
    identity = {'schema_version': 1, 'freeze_id': freeze['freeze_id'], 'evaluation_plan': plan,
                'input_sha256': {'freeze': file_hash(experiment / 'freeze.json'), 'features': file_hash(features / 'manifest.json'),
                    'prediction_manifest': file_hash(predictions / 'manifest.json'), 'test_labels': file_hash(features / 'labels_test.jsonl')},
                'code_sha256': file_hash(Path(__file__)),
                'output_sha256': {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / 'manifest.json', {**identity, 'run_fingerprint': digest(identity), 'created_at': utc_now().isoformat()})
    return {'output': str(output.resolve()), **summary}
