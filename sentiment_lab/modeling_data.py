"""Frozen model settings, validated M4 inputs and date-grouped expanding folds."""
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib

from .contracts import ContractError, session_date, ticker, timestamp
from .features import FEATURE_SETS, KEYS, feature_schema, read_json, read_jsonl
from .indicators import FeatureConfig
from .io import file_hash
from .preprocessing import numeric


@dataclass(frozen=True)
class ModelingConfig:
    schema_version: int = 1
    seed: int = 42
    cv_folds: int = 3
    minimum_fold_train_sessions: int = 60
    logistic_c: tuple[float, ...] = (.01, .1, 1., 10.)
    forest_depths: tuple[int, ...] = (3, 6)
    forest_min_leaf: tuple[int, ...] = (5, 15)
    forest_trees: int = 200
    logistic_max_iter: int = 2000
    coin_seeds: int = 20
    permutation_repeats: int = 5
    threshold: float = .5

    @classmethod
    def load(cls, path: Path):
        with path.open('rb') as stream:
            raw = tomllib.load(stream)
        if set(raw) - set(cls.__dataclass_fields__):
            raise ContractError('Unknown modeling configuration setting')
        for key in ('logistic_c', 'forest_depths', 'forest_min_leaf'):
            if key in raw:
                if not isinstance(raw[key], list):
                    raise ContractError(f'{key} must be an array')
                raw[key] = tuple(raw[key])
        result = cls(**raw)
        result.validate()
        return result

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ContractError('Modeling schema_version must be 1')
        for key, low, high in (('seed', 0, 2**32 - 1), ('cv_folds', 2, 5),
            ('minimum_fold_train_sessions', 3, 10000), ('forest_trees', 10, 1000),
            ('logistic_max_iter', 100, 10000), ('coin_seeds', 1, 100), ('permutation_repeats', 1, 20)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ContractError(f'{key} must be an integer in [{low}, {high}]')
        for key in ('logistic_c', 'forest_depths', 'forest_min_leaf'):
            if not isinstance(getattr(self, key), tuple) or not getattr(self, key):
                raise ContractError(f'{key} must be a nonempty parameter sequence')
        if any(isinstance(c, bool) or not isinstance(c, (float, int)) or not math.isfinite(c) or c <= 0 for c in self.logistic_c):
            raise ContractError('logistic_c values must be finite and positive')
        for key in ('forest_depths', 'forest_min_leaf'):
            if any(type(v) is not int or not 1 <= v <= 100 for v in getattr(self, key)):
                raise ContractError(f'{key} values must be integers in [1,100]')
        for key in ('logistic_c', 'forest_depths', 'forest_min_leaf'):
            if len(set(getattr(self, key))) != len(getattr(self, key)):
                raise ContractError(f'{key} must contain unique values')
        if len(self.logistic_c) > 8 or len(self.forest_depths) * len(self.forest_min_leaf) > 8:
            raise ContractError('Search budget is at most eight candidates per family/feature set')
        if isinstance(self.threshold, bool) or self.threshold != .5:
            raise ContractError('The frozen classification threshold is 0.5')

    def candidates(self, family: str) -> list[dict]:
        if family == 'logistic_regression':
            return [{'C': float(c)} for c in self.logistic_c]
        if family == 'random_forest':
            return [{'max_depth': depth, 'min_samples_leaf': leaf} for depth in self.forest_depths for leaf in self.forest_min_leaf]
        raise ContractError('Unknown model family')


def row_key(row):
    return row['ticker'], row['target_session']


def verify_file(directory, manifest, name):
    try:
        expected = manifest['output_sha256'][name]
    except KeyError as exc:
        raise ContractError(f'M4 output lacks {name}; rebuild M4 features with the current command') from exc
    if file_hash(directory / name) != expected:
        raise ContractError(f'M4 output hash mismatch: {name}')


def load_features(directory: Path, load_development_labels=True) -> dict:
    manifest = read_json(directory / 'manifest.json')
    for name in ('feature_schema.json', 'leakage_audit.json', 'feature_feasibility.json', 'summary.json'):
        verify_file(directory, manifest, name)
    if read_json(directory / 'feature_schema.json') != feature_schema(FeatureConfig()):
        raise ContractError('M4 feature schema differs from the frozen model allowlists')
    if read_json(directory / 'leakage_audit.json').get('status') != 'passed':
        raise ContractError('M4 leakage audit did not pass')
    if read_json(directory / 'summary.json').get('status') != 'ready':
        raise ContractError('M4 has insufficient training-ready history/partitions')
    matrices, reference = {}, None
    partitions_by_date = {}
    for name, columns in FEATURE_SETS.items():
        verify_file(directory, manifest, f'{name}.jsonl')
        rows = read_jsonl(directory / f'{name}.jsonl')
        ids = []
        for row in rows:
            if set(row) != set(KEYS + columns):
                raise ContractError('M4 matrix contains missing or nonallowlisted columns')
            ticker(row['ticker'])
            session_date(row['target_session'])
            timestamp(row['decision_at'])
            if row['partition'] not in {'train', 'validation', 'test'}:
                raise ContractError('Invalid M4 partition')
            old = partitions_by_date.setdefault(row['target_session'], row['partition'])
            if old != row['partition']:
                raise ContractError('The same date cannot occur in different partitions across tickers')
            for column in columns:
                numeric(row[column], column)
            ids.append(tuple(row[key] for key in KEYS))
        if len(set(ids)) != len(ids) or len({row_key(r) for r in rows}) != len(rows):
            raise ContractError('Duplicate ticker/session keys in M4 features')
        if reference is not None and ids != reference:
            raise ContractError('M4 feature sets do not share identical ordered rows')
        reference = ids
        matrices[name] = rows
    combined = matrices['combined']
    for name, columns in FEATURE_SETS.items():
        if any(any(row[c] != other[c] for c in columns) for row, other in zip(matrices[name], combined)):
            raise ContractError('An M4 ablation changes values instead of selecting feature columns')
    parts = {p: [r for r in combined if r['partition'] == p] for p in ('train', 'validation', 'test')}
    if not all(parts.values()):
        raise ContractError('All original train/validation/test partitions must be nonempty')
    if max(r['target_session'] for r in parts['train']) >= min(r['target_session'] for r in parts['validation']) or max(r['target_session'] for r in parts['validation']) >= min(r['target_session'] for r in parts['test']):
        raise ContractError('Model partitions are not strictly chronological')
    if max(timestamp(r['decision_at']) for r in parts['train']) >= min(timestamp(r['decision_at']) for r in parts['validation']) or max(timestamp(r['decision_at']) for r in parts['validation']) >= min(timestamp(r['decision_at']) for r in parts['test']):
        raise ContractError('Model decision instants are not chronological')
    labels = {}
    if load_development_labels:
        for partition, next_partition in (('train', 'validation'), ('validation', 'test')):
            filename = f'labels_{partition}.jsonl'
            verify_file(directory, manifest, filename)
            rows = read_jsonl(directory / filename)
            expected = {row_key(r) for r in parts[partition]}
            if len(rows) != len(expected) or {row_key(r) for r in rows} != expected:
                raise ContractError('Development feature/label keys are mismatched or duplicated')
            boundary = min(timestamp(r['decision_at']) for r in parts[next_partition])
            for row in rows:
                if row['partition'] != partition or type(row['label_up']) is not int or row['label_up'] not in (0, 1):
                    raise ContractError('Development labels must be binary integers in the correct partition')
                if timestamp(row['label_available_at']) >= boundary:
                    raise ContractError('A development label is unresolved at the next partition boundary')
                labels[row_key(row)] = row
        # Test labels and the all-partition label file are deliberately not parsed.
    return {'manifest': manifest, 'matrices': matrices, 'parts': parts, 'labels': labels,
            'source_data_gate': read_json(directory / 'feature_feasibility.json')['decision'],
            'feature_manifest_sha256': file_hash(directory / 'manifest.json')}


def expanding_folds(rows: list[dict], labels: dict, settings: ModelingConfig) -> list[dict]:
    dates = sorted({r['target_session'] for r in rows})
    initial = len(dates) // (settings.cv_folds + 1)
    if initial < settings.minimum_fold_train_sessions:
        raise ContractError('Insufficient outer-training sessions for configured expanding folds')
    result = []
    for index in range(settings.cv_folds):
        left = len(dates) * (index + 1) // (settings.cv_folds + 1)
        right = len(dates) * (index + 2) // (settings.cv_folds + 1)
        evaluation_dates = set(dates[left:right])
        evaluation = [r for r in rows if r['target_session'] in evaluation_dates]
        boundary = min(timestamp(r['decision_at']) for r in evaluation)
        training = [r for r in rows if r['target_session'] < dates[left]]
        eligible = [r for r in training if timestamp(labels[row_key(r)]['label_available_at']) < boundary]
        if len({r['target_session'] for r in eligible}) < settings.minimum_fold_train_sessions:
            raise ContractError('Fold has insufficient eligible training dates after label purging')
        result.append({'index': index, 'training': eligible, 'evaluation': evaluation,
                       'boundary': boundary.isoformat(), 'purged_rows': len(training) - len(eligible)})
    return result
