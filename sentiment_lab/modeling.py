"""M5 development runs and frozen artifacts. Test labels are not parsed here."""
from dataclasses import asdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics

from . import __version__
from .acquisition import write_csv
from .classifiers import (FAMILIES, baseline_predictions, baseline_spec, export_estimator,
    fit_classifier, interpretation, matrix, metrics, prediction_rows, predict_state, probabilities)
from .contracts import ContractError
from .features import FEATURE_SETS, KEYS, project_rows, read_json
from .io import file_hash, write_json, write_jsonl
from .modeling_data import ModelingConfig, expanding_folds, load_features, row_key
from .preprocessing import TrainPreprocessor
from .store import digest, utc_now


def empty_output(output):
    if output.exists() and any(output.iterdir()):
        raise ContractError('Model output directory must be empty; choose a fresh output path')


def y_values(rows, labels):
    return [labels[row_key(row)]['label_up'] for row in rows]


def prepare_development(bundle, columns, settings):
    train = project_rows(bundle['parts']['train'], columns)
    validation = project_rows(bundle['parts']['validation'], columns)
    boundary = min(row['decision_at'] for row in validation)
    processor = TrainPreprocessor.fit(train, columns, boundary)
    outer = {'processor': processor, 'x_train': matrix(processor.transform(train), columns),
             'x_validation': matrix(processor.transform(validation), columns),
             'y_train': y_values(train, bundle['labels']), 'y_validation': y_values(validation, bundle['labels'])}
    folds = []
    for fold in expanding_folds(bundle['parts']['train'], bundle['labels'], settings):
        train_rows, eval_rows = project_rows(fold['training'], columns), project_rows(fold['evaluation'], columns)
        prep = TrainPreprocessor.fit(train_rows, columns, fold['boundary'])
        folds.append({'index': fold['index'], 'processor': prep, 'train_rows': train_rows, 'eval_rows': eval_rows,
                      'x_train': matrix(prep.transform(train_rows), columns), 'x_eval': matrix(prep.transform(eval_rows), columns),
                      'y_train': y_values(train_rows, bundle['labels']), 'y_eval': y_values(eval_rows, bundle['labels']),
                      'purged_rows': fold['purged_rows'], 'boundary': fold['boundary']})
    return outer, folds


def candidate_rank(candidate):
    value = candidate['validation']['roc_auc']
    # If validation has only one class, use the first declared setting untuned.
    if value is None:
        return (0, 0, -candidate['candidate_index'])
    cv = candidate['cv_mean_auc']
    return (round(value, 12), round(cv, 12) if cv is not None else -1, -candidate['candidate_index'])


def fit_group(family, feature_set, bundle, settings):
    columns = FEATURE_SETS[feature_set]
    outer, folds = prepare_development(bundle, columns, settings)
    candidates, best, best_rank = [], None, None
    for candidate_index, parameters in enumerate(settings.candidates(family)):
        fold_metrics, fold_predictions = [], []
        for fold in folds:
            if set(fold['y_train']) != {0, 1}:
                fold_metrics.append({'index': fold['index'], 'status': 'skipped_single_class_training',
                                     'purged_rows': fold['purged_rows']})
                continue
            model = fit_classifier(family, parameters, settings, fold['x_train'], fold['y_train'])
            scores = probabilities(model, fold['x_eval'])
            fold_metrics.append({'index': fold['index'], 'status': 'fit', 'metrics': metrics(fold['y_eval'], scores),
                                 'purged_rows': fold['purged_rows']})
            fold_predictions.extend(prediction_rows(fold['eval_rows'], scores,
                f'{family}:{feature_set}', f'inner_fold_{fold["index"]}'))
        estimator = fit_classifier(family, parameters, settings, outer['x_train'], outer['y_train'])
        scores = probabilities(estimator, outer['x_validation'])
        cv_values = [r['metrics']['roc_auc'] for r in fold_metrics if r['status'] == 'fit' and r['metrics']['roc_auc'] is not None]
        candidate = {'family': family, 'feature_set': feature_set, 'candidate_index': candidate_index,
                     'parameters': parameters, 'validation': metrics(outer['y_validation'], scores),
                     'cv_mean_auc': statistics.fmean(cv_values) if cv_values else None,
                     'folds': fold_metrics}
        candidates.append(candidate)
        rank = candidate_rank(candidate)
        if best is None or rank > best_rank:
            best_rank = rank
            best = {'candidate': candidate, 'estimator': estimator, 'scores': scores, 'fold_predictions': fold_predictions}
    state = export_estimator(best['estimator'], family, columns)
    # Check the JSON representation against the native model before freezing it.
    portable = predict_state(state, outer['x_validation'])
    if any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) for a, b in zip(portable, best['scores'])):
        raise ContractError('Portable model predictions differ from the native fitted estimator')
    result = {'schema_version': 1, 'model_id': f'{family}:{feature_set}', 'family': family, 'feature_set': feature_set,
              'selected_candidate': best['candidate'], 'selection': 'validation_auc_then_cv_auc_then_grid_order'
                   if best['candidate']['validation']['roc_auc'] is not None else 'first_declared_candidate_untuned_single_class_validation',
              'state': state, 'preprocessing': outer['processor'].to_dict(),
              'fold_preprocessing': [{'index': fold['index'], 'boundary': fold['boundary'],
                     'purged_rows': fold['purged_rows'], 'preprocessing': fold['processor'].to_dict()} for fold in folds],
              'interpretation': interpretation(best['estimator'], family, columns, outer['processor'],
                    outer['x_validation'], outer['y_validation'], settings)}
    return result, candidates, prediction_rows(bundle['parts']['validation'], best['scores'], result['model_id'], 'validation'), best['fold_predictions']


def protocol(bundle, settings):
    return {'schema_version': 1, 'stage': 'development_then_frozen_test', 'settings': asdict(settings),
            'feature_manifest_sha256': bundle['feature_manifest_sha256'],
            'source_data_gate': bundle['source_data_gate'], 'synthetic_data': bundle['manifest'].get('synthetic_data'),
            'feature_sets': {name: list(columns) for name, columns in FEATURE_SETS.items()},
            'candidate_grids': {family: settings.candidates(family) for family in FAMILIES},
            'partition_rows': {name: len(rows) for name, rows in bundle['parts'].items()},
            'selection': 'Validation AUC rounded to 12 decimals; tie: mean valid inner-fold AUC then earlier declared grid index. Single-class validation: first declared setting, untuned.',
            'outer_refit': 'Selected estimators fit original train only; no train+validation refit.',
            'preprocessing': 'Fresh train-only median/scaler for every feature set and expanding fold; M4 fitted transforms are not reused.',
            'split_rule': 'Shared target dates; original outer partitions preserved; unresolved labels purged before each fold boundary.',
            'logistic_fixed': {'solver': 'lbfgs', 'class_weight': None, 'max_iter': settings.logistic_max_iter},
            'forest_fixed': {'n_estimators': settings.forest_trees, 'max_features': 1.0, 'bootstrap': True, 'n_jobs': 1, 'class_weight': None},
            'primary_comparison': {'augmented': 'logistic_regression:combined', 'control': 'logistic_regression:price_only', 'endpoint': 'paired held-out delta ROC-AUC'},
            'm6_evaluation_plan': {'confidence_level': .95, 'paired_moving_block_bootstrap_replicates': 2000,
                'block_sessions': 5, 'sensitivity_block_sessions': [1, 10], 'resample_dates_jointly_across_tickers': True,
                'f1_threshold': .5, 'secondary_comparisons': 'exploratory; no post-test winner redefinition'},
            'interpretation': 'Validation metrics and importance are development diagnostics. Missing sentiment cannot establish incremental sentiment value.',
            'holdout_policy': 'No test labels parsed or used during training, tuning, explanations or freezing; holdout prediction is a separate post-freeze command.'}


def train_models(features: Path, settings_path: Path, output: Path):
    empty_output(output)
    settings = ModelingConfig.load(settings_path)
    bundle = load_features(features)
    if set(y_values(bundle['parts']['train'], bundle['labels'])) != {0, 1}:
        raise ContractError('Outer training data requires both target classes for LR/RF')
    # Fail before beginning a partial run if development dates cannot support CV.
    expanding_folds(bundle['parts']['train'], bundle['labels'], settings)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', protocol(bundle, settings))
    write_jsonl(output / 'holdout_keys.jsonl', [{key: row[key] for key in KEYS} for row in bundle['parts']['test']])
    write_jsonl(output / 'development_labels.jsonl', list(bundle['labels'].values()))
    all_candidates, all_predictions, all_oof, selected_metrics, model_files = [], [], [], [], {}
    for family in FAMILIES:
        for feature_set in FEATURE_SETS:
            model, candidates, predicted, oof = fit_group(family, feature_set, bundle, settings)
            filename = f'model_{family}_{feature_set}.json'
            write_json(output / filename, model)
            model_files[model['model_id']] = filename
            all_candidates.extend(candidates)
            all_predictions.extend(predicted)
            all_oof.extend(oof)
            selected_metrics.append({'model_id': model['model_id'], 'family': family, 'feature_set': feature_set,
                                     'parameters': model['selected_candidate']['parameters'], **model['selected_candidate']['validation']})
            # Useful persisted progress; a missing freeze.json means this run is incomplete.
            write_json(output / 'progress.json', {'completed_model_groups': len(model_files), 'total_model_groups': 10})
    baseline = baseline_spec(y_values(bundle['parts']['train'], bundle['labels']), settings)
    write_json(output / 'baselines.json', baseline)
    baseline_rows, baseline_metrics = [], []
    for name, scores in baseline_predictions(bundle['parts']['validation'], baseline).items():
        baseline_rows.extend(prediction_rows(bundle['parts']['validation'], scores, 'baseline:' + name, 'validation'))
        baseline_metrics.append({'model_id': 'baseline:' + name, 'family': 'baseline', 'feature_set': 'matched_sample',
                                 **metrics(y_values(bundle['parts']['validation'], bundle['labels']), scores)})
    coin = [row for row in baseline_metrics if row['model_id'].startswith('baseline:coin_seed_')]
    variability = {metric: {'mean': statistics.fmean(r[metric] for r in coin if r[metric] is not None),
                          'std': statistics.pstdev(r[metric] for r in coin if r[metric] is not None)}
                   for metric in ('f1', 'roc_auc') if any(r[metric] is not None for r in coin)}
    write_json(output / 'coin_variability.json', {'partition': 'validation', 'seeds': settings.coin_seeds, 'metrics': variability})
    write_jsonl(output / 'validation_predictions.jsonl', all_predictions + baseline_rows)
    write_jsonl(output / 'inner_fold_predictions.jsonl', all_oof)
    write_json(output / 'candidate_metrics.json', {'candidates': all_candidates})
    write_json(output / 'validation_metrics.json', {'partition': 'validation', 'models': selected_metrics + baseline_metrics})
    write_csv(output / 'validation_metrics.csv', selected_metrics + baseline_metrics,
              ['model_id', 'family', 'feature_set', 'rows', 'positive_prevalence', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall'])
    audit = {'status': 'passed', 'test_labels_parsed': False, 'test_predictions_generated': False,
             'm4_scaled_files_used': False, 'feature_sets_match': True, 'feature_allowlists_frozen': True,
             'fold_preprocessing_refit': True, 'fold_boundary_labels_purged': True,
             'outer_partitions_preserved': True, 'equal_candidate_budget_within_family': True,
             'candidate_counts': {family: len(settings.candidates(family)) for family in FAMILIES},
             'class_threshold': .5, 'primary_model_family_fixed': 'logistic_regression',
             'source_data_gate': bundle['source_data_gate'], 'portable_models_validated_on_validation': True,
             'limits': 'Temporal/provenance checks rely on declared source availability; no statistical significance or trading-return conclusion is made.'}
    write_json(output / 'training_audit.json', audit)
    summary = {'status': 'frozen', 'trained_models': len(model_files), 'attempted_candidates': len(all_candidates),
               'validation_rows_per_model': len(bundle['parts']['validation']),
               'holdout_rows_reserved': len(bundle['parts']['test']), 'test_labels_parsed': False,
               'source_data_gate': bundle['source_data_gate'], 'synthetic_data': bundle['manifest'].get('synthetic_data'),
               'interpretation': 'Development-stage model/baseline run. Final test evaluation and uncertainty belong to M6.'}
    write_json(output / 'summary.json', summary)
    lock = Path(__file__).parent.parent / 'requirements-modeling.lock'
    freeze = {'schema_version': 1, 'package_version': __version__, 'frozen_at': utc_now().isoformat(),
              'feature_manifest_sha256': bundle['feature_manifest_sha256'], 'model_files': model_files,
              'modeling_config_sha256': file_hash(settings_path), 'source_data_gate': bundle['source_data_gate'],
              'synthetic_data': bundle['manifest'].get('synthetic_data'),
              'dependencies': {name: version(name) for name in ('scikit-learn', 'numpy', 'scipy', 'threadpoolctl')},
              'python_version': platform.python_version(), 'environment_lock_sha256': file_hash(lock) if lock.exists() else None,
              'code_sha256': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'output_sha256': {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / 'freeze.json', {**freeze, 'freeze_id': digest(freeze)})
    return {'output': str(output.resolve()), **summary}


def verified_freeze(experiment: Path):
    freeze = read_json(experiment / 'freeze.json')
    if freeze.get('freeze_id') != digest({key: value for key, value in freeze.items() if key != 'freeze_id'}):
        raise ContractError('Experiment freeze fingerprint is invalid')
    for name, expected in freeze['output_sha256'].items():
        if Path(name).name != name or file_hash(experiment / name) != expected:
            raise ContractError(f'Frozen experiment artifact is missing or changed: {name}')
    return freeze


def predict_holdout(features: Path, experiment: Path, output: Path):
    empty_output(output)
    freeze = verified_freeze(experiment)
    bundle = load_features(features, load_development_labels=False)
    if freeze['feature_manifest_sha256'] != bundle['feature_manifest_sha256']:
        raise ContractError('Frozen experiment is bound to a different M4 input manifest')
    rows, predictions = bundle['parts']['test'], []
    from .features import read_jsonl
    if [{key: row[key] for key in KEYS} for row in rows] != read_jsonl(experiment / 'holdout_keys.jsonl'):
        raise ContractError('Holdout row keys differ from the frozen experiment')
    for model_id, name in freeze['model_files'].items():
        model = read_json(experiment / name)
        columns = FEATURE_SETS[model['feature_set']]
        data = model['preprocessing']
        processor = TrainPreprocessor(tuple(data['feature_names']), data['statistics'], data['fit'])
        transformed = matrix(processor.transform(project_rows(rows, columns)), columns)
        scores = predict_state(model['state'], transformed)
        predictions.extend(prediction_rows(rows, scores, model_id, 'test'))
    for name, scores in baseline_predictions(rows, read_json(experiment / 'baselines.json')).items():
        predictions.extend(prediction_rows(rows, scores, 'baseline:' + name, 'test'))
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'test_predictions.jsonl', predictions)
    report = {'status': 'predicted_without_labels', 'freeze_id': freeze['freeze_id'],
              'freeze_sha256': file_hash(experiment / 'freeze.json'), 'feature_manifest_sha256': bundle['feature_manifest_sha256'],
              'rows_per_model': len(rows), 'prediction_records': len(predictions), 'test_labels_parsed': False,
              'source_data_gate': bundle['source_data_gate'], 'created_at': utc_now().isoformat(),
              'prediction_sha256': file_hash(output / 'test_predictions.jsonl')}
    write_json(output / 'manifest.json', report)
    return {'output': str(output.resolve()), **report}
