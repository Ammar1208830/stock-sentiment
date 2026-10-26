"""Train-only imputation/scaling reusable inside future chronological folds."""
from dataclasses import dataclass
import math
import statistics

from .contracts import ContractError, timestamp
from .store import digest


def numeric(value, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f'Feature {name} must be finite numeric data or null')
    if isinstance(value, bool) and not name.endswith('_missing'):
        raise ContractError(f'Boolean value is only valid for an explicit missingness feature: {name}')
    return float(value)


@dataclass(frozen=True)
class TrainPreprocessor:
    feature_names: tuple[str, ...]
    statistics: dict
    fit_metadata: dict

    @classmethod
    def fit(cls, rows: list[dict], feature_names: tuple[str, ...], evaluation_start: str):
        if not rows or not feature_names or len(set(feature_names)) != len(feature_names):
            raise ContractError('Preprocessing requires training rows and a unique explicit feature allowlist')
        boundary = timestamp(evaluation_start)
        ids = set()
        for row in rows:
            identity = (row['ticker'], row['target_session'])
            if identity in ids:
                raise ContractError('Duplicate training row in preprocessor fit')
            ids.add(identity)
            if row.get('partition') != 'train' or timestamp(row['decision_at']) >= boundary:
                raise ContractError('Preprocessing may fit only train rows before the evaluation boundary')
            if set(row) != {'ticker', 'target_session', 'partition', 'decision_at', *feature_names}:
                raise ContractError('Preprocessor fit rows must contain exactly the explicit feature allowlist and keys')
        stats = {}
        for feature in feature_names:
            values = [numeric(row[feature], feature) for row in rows]
            observed = [value for value in values if value is not None]
            fill = statistics.median(observed) if observed else 0.0
            filled = [fill if value is None else value for value in values]
            mean, std = statistics.fmean(filled), statistics.pstdev(filled)
            stats[feature] = {'fill_value': fill, 'mean': mean, 'scale': std if std > 0 else 1.0,
                              'observed_train_values': len(observed), 'all_missing_train': not observed,
                              'constant_after_imputation': std == 0}
        return cls(tuple(feature_names), stats,
                   {'fit_partition': 'train', 'fit_rows': len(rows),
                    'first_session': min(r['target_session'] for r in rows),
                    'last_session': max(r['target_session'] for r in rows),
                    'latest_fit_decision': max(r['decision_at'] for r in rows),
                    'evaluation_start': boundary.isoformat(), 'training_feature_sha256': digest(rows),
                    'method': 'train_median_population_standardize', 'all_missing_fill': 0.0,
                    'fold_requirement': 'Refit a new preprocessor on each chronological training fold; never reuse outer-fit statistics inside CV.'})

    def transform(self, rows: list[dict]) -> list[dict]:
        result = []
        for row in rows:
            if set(row) != {'ticker', 'target_session', 'partition', 'decision_at', *self.feature_names}:
                raise ContractError('Transform input differs from the frozen feature allowlist')
            transformed = {key: row[key] for key in ('ticker', 'target_session', 'partition', 'decision_at')}
            for feature in self.feature_names:
                value = numeric(row[feature], feature)
                stats = self.statistics[feature]
                filled = stats['fill_value'] if value is None else value
                value = (filled - stats['mean']) / stats['scale']
                if not math.isfinite(value):
                    raise ContractError('Preprocessing produced a nonfinite feature')
                transformed[feature] = value
            result.append(transformed)
        return result

    def to_dict(self) -> dict:
        return {'feature_names': list(self.feature_names), 'statistics': self.statistics, 'fit': self.fit_metadata}
