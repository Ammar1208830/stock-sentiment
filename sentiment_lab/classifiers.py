"""Deterministic classical models, metrics, baselines and portable JSON state."""
import math
import warnings

from .contracts import ContractError
from .features import KEYS
from .store import digest


FAMILIES = ('logistic_regression', 'random_forest')


def matrix(rows, columns):
    return [[float(row[column]) for column in columns] for row in rows]


def fit_classifier(family, parameters, settings, x, y):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.exceptions import ConvergenceWarning
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ContractError('Install requirements-modeling.lock for model training') from exc
    if set(y) != {0, 1}:
        raise ContractError('A classifier training block requires both target classes')
    if family == 'logistic_regression':
        estimator = LogisticRegression(C=parameters['C'], solver='lbfgs', max_iter=settings.logistic_max_iter,
                                       random_state=settings.seed, class_weight=None)
    elif family == 'random_forest':
        estimator = RandomForestClassifier(**parameters, n_estimators=settings.forest_trees, random_state=settings.seed,
                                          max_features=1.0, bootstrap=True, n_jobs=1, class_weight=None)
    else:
        raise ContractError('Unknown classifier family')
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            estimator.fit(x, y)
        except (ConvergenceWarning, ValueError) as exc:
            raise ContractError(f'{family} fitting failed ({type(exc).__name__}); no experiment was frozen') from exc
    if estimator.classes_.tolist() != [0, 1]:
        raise ContractError('Model classes must be exactly [0, 1]')
    return estimator


def probabilities(estimator, x):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = [float(row[1]) for row in estimator.predict_proba(x)]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in result):
        raise ContractError('Classifier produced an invalid up probability')
    return result


def metrics(labels, scores, threshold=.5):
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    except ImportError as exc:
        raise ContractError('Install requirements-modeling.lock for classification metrics') from exc
    if not labels or len(labels) != len(scores) or any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ContractError('Metrics require aligned binary integer labels and scores')
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ContractError('Metrics require finite probabilities in [0,1]')
    predicted = [int(p >= threshold) for p in scores]
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
    return {'rows': len(labels), 'positive_prevalence': sum(labels) / len(labels), 'threshold': threshold,
            'roc_auc': auc, 'roc_auc_status': 'defined' if auc is not None else 'undefined_single_class',
            'f1': float(f1_score(labels, predicted, zero_division=0)),
            'accuracy': float(accuracy_score(labels, predicted)),
            'precision': float(precision_score(labels, predicted, zero_division=0)),
            'recall': float(recall_score(labels, predicted, zero_division=0)),
            'confusion_matrix': confusion_matrix(labels, predicted, labels=[0, 1]).tolist()}


def baseline_spec(training_labels, settings):
    if not training_labels:
        raise ContractError('Baselines require training labels')
    return {'training_up_prevalence': sum(training_labels) / len(training_labels),
            'coin_seeds': [settings.seed + index for index in range(settings.coin_seeds)],
            'definitions': {'training_majority': 'Constant training up prevalence; threshold ties predict up.',
                'always_up': 'Always-long directional analogue of buy-and-hold; no portfolio return implied.',
                'zero_drift_no_change': 'Previous-close price forecast maps to non-up; constant directional score 0.',
                'momentum': 'Up iff the known prior one-session return is positive.',
                'coin': 'Row-key/seed-hashed uniform score, thresholded at 0.5; score is not a calibrated probability.'}}


def baseline_predictions(rows, spec):
    outputs = {name: [] for name in ('training_majority', 'always_up', 'zero_drift_no_change', 'momentum')}
    outputs.update({f'coin_seed_{seed}': [] for seed in spec['coin_seeds']})
    for row in rows:
        outputs['training_majority'].append(spec['training_up_prevalence'])
        outputs['always_up'].append(1.0)
        outputs['zero_drift_no_change'].append(0.0)
        outputs['momentum'].append(float(row['return_1d'] > 0))
        for seed in spec['coin_seeds']:
            outputs[f'coin_seed_{seed}'].append(int(digest(['coin-v1', seed, row['ticker'], row['target_session']])[:16], 16) / 2**64)
    return outputs


def export_estimator(estimator, family, columns):
    if family == 'logistic_regression':
        return {'family': family, 'feature_names': list(columns), 'classes': [0, 1],
                'coefficients': estimator.coef_[0].tolist(), 'intercept': float(estimator.intercept_[0])}
    trees = []
    for tree in estimator.estimators_:
        nodes = tree.tree_
        values = nodes.value[:, 0, :]
        if values.shape[1] != 2:
            raise ContractError('Unexpected forest tree class shape')
        trees.append({'children_left': nodes.children_left.tolist(), 'children_right': nodes.children_right.tolist(),
                      'feature': nodes.feature.tolist(), 'threshold': nodes.threshold.tolist(),
                      'up_probability': [float(value[1] / sum(value)) for value in values]})
    return {'family': family, 'feature_names': list(columns), 'classes': [0, 1], 'trees': trees,
            'input_cast': 'float32'}


def predict_state(state, x):
    """Predict from local JSON coefficients/trees without pickle deserialization."""
    if state.get('classes') != [0, 1] or any(len(row) != len(state['feature_names']) for row in x):
        raise ContractError('Portable model classes/dimensions differ from inputs')
    if state['family'] == 'logistic_regression':
        result = []
        for row in x:
            value = math.fsum(a * b for a, b in zip(row, state['coefficients'])) + state['intercept']
            result.append(1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value)))
        return result
    if state['family'] != 'random_forest' or not state.get('trees'):
        raise ContractError('Invalid portable classifier family/state')
    import numpy as np
    result = []
    for row in x:
        values = [float(np.float32(value)) for value in row]
        tree_probabilities = []
        for tree in state['trees']:
            node = 0
            for _ in range(len(tree['feature']) + 1):
                if tree['children_left'][node] == -1:
                    tree_probabilities.append(tree['up_probability'][node])
                    break
                feature = tree['feature'][node]
                node = tree['children_left'][node] if values[feature] <= tree['threshold'][node] else tree['children_right'][node]
            else:
                raise ContractError('Portable forest contains a cyclic tree')
        result.append(sum(tree_probabilities) / len(tree_probabilities))
    return result


def prediction_rows(rows, scores, model_id, partition):
    if len(rows) != len(scores):
        raise ContractError('Predictions are not aligned with model input rows')
    return [{key: row[key] for key in KEYS} | {'model_id': model_id, 'up_probability': score,
             'predicted_up': int(score >= .5), 'prediction_partition': partition}
            for row, score in zip(rows, scores)]


def interpretation(estimator, family, columns, processor, x_validation, y_validation, settings):
    if family == 'logistic_regression':
        return {'method': 'signed_logistic_coefficients',
                'coefficients': [{'feature': feature, 'standardized_coefficient': float(value),
                                  'input_scale_coefficient': float(value / processor.statistics[feature]['scale'])}
                                 for feature, value in zip(columns, estimator.coef_[0])],
                'intercept_standardized': float(estimator.intercept_[0]),
                'caveat': 'Associations in fitted/imputed feature scales; not causal effects.'}
    if len(set(y_validation)) != 2:
        return {'method': 'validation_permutation_auc', 'status': 'unavailable_single_class_validation'}
    from sklearn.inspection import permutation_importance
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = permutation_importance(estimator, x_validation, y_validation, scoring='roc_auc',
            n_repeats=settings.permutation_repeats, random_state=settings.seed, n_jobs=1)
    return {'method': 'validation_permutation_auc', 'repeats': settings.permutation_repeats,
            'features': [{'feature': feature, 'mean_auc_drop': float(mean), 'std_auc_drop': float(std)}
                         for feature, mean, std in zip(columns, result.importances_mean, result.importances_std)],
            'caveat': 'Exploratory validation-only importance; correlated columns can mask one another. Not used to select features or settings.'}
