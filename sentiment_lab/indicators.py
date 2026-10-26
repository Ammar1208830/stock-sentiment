"""Past-only technical indicators, with explicit calendar and split barriers."""
from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
import statistics
import tomllib

from .contracts import ContractError, PriceBar, Session


PRICE_FEATURES = ('return_1d', 'return_5d', 'momentum_10d', 'volatility_10d', 'rsi_5d')
MIN_HISTORY = 11  # Ten returns require eleven consecutive closes.


@dataclass(frozen=True)
class FeatureConfig:
    schema_version: int = 1
    split_policy: str = 'reset_at_split_session'
    rsi_method: str = 'wilder_seeded_sma'
    volatility_ddof: int = 1
    preprocessing: str = 'train_median_standardize'

    @classmethod
    def load(cls, path: Path):
        with path.open('rb') as stream:
            raw = tomllib.load(stream)
        if set(raw) - set(cls.__dataclass_fields__):
            raise ContractError('Unknown feature configuration setting')
        result = cls(**raw)
        if type(result.schema_version) is not int or type(result.volatility_ddof) is not int or result != cls():
            raise ContractError('Feature schema v1 has frozen formulas and policies; use the supplied defaults')
        return result


def wilder_rsi(closes: list[float], period: int = 5) -> float:
    if type(period) is not int or period < 1 or len(closes) < period + 1:
        raise ContractError('Wilder RSI needs at least period + 1 closes')
    if any(isinstance(c, bool) or not math.isfinite(c) or c <= 0 for c in closes):
        raise ContractError('RSI closes must be finite and positive')
    differences = [right - left for left, right in zip(closes, closes[1:])]
    gains = statistics.fmean(max(d, 0) for d in differences[:period])
    losses = statistics.fmean(max(-d, 0) for d in differences[:period])
    for delta in differences[period:]:
        gains = ((period - 1) * gains + max(delta, 0)) / period
        losses = ((period - 1) * losses + max(-delta, 0)) / period
    if gains == 0 and losses == 0:
        return 50.0
    if losses == 0:
        return 100.0
    if gains == 0:
        return 0.0
    return 100 - 100 / (1 + gains / losses)


def technical_values(history: list[PriceBar]) -> dict[str, float]:
    if len(history) < MIN_HISTORY:
        raise ContractError('Technical feature schema v1 requires eleven consecutive known closes')
    closes = [bar.close for bar in history]
    returns = [right / left - 1 for left, right in zip(closes[-11:], closes[-10:])]
    result = {'return_1d': closes[-1] / closes[-2] - 1,
              'return_5d': closes[-1] / closes[-6] - 1,
              'momentum_10d': closes[-1] / statistics.fmean(closes[-10:]) - 1,
              'volatility_10d': statistics.stdev(returns),
              'rsi_5d': wilder_rsi(closes, 5)}
    if not all(math.isfinite(value) for value in result.values()):
        raise ContractError('Technical indicator is not finite')
    return result


def known_history(symbol: str, target: date, decision: datetime, sessions: list[Session],
                  bars: dict[tuple[str, date], PriceBar]) -> tuple[list[PriceBar], str]:
    # Search by actual session positions, never by weekdays or by neighboring rows.
    if any(a.date >= b.date for a, b in zip(sessions, sessions[1:])):
        raise ContractError('Technical history requires sorted unique exchange sessions')
    indices = {session.date: i for i, session in enumerate(sessions)}
    if target not in indices:
        raise ContractError('Feature target session is absent from the verified calendar')
    history, barrier = [], 'start_of_dataset'
    for index in range(indices[target] - 1, -1, -1):
        session = sessions[index]
        bar = bars.get((symbol, session.date))
        if bar is None:
            barrier = 'missing_history_bar'
            break
        if bar.available_at >= decision:
            barrier = 'unavailable_history_bar'
            break
        if bar.available_at < session.close_at:
            raise ContractError('A historical close is timestamped before its exchange close')
        history.append(bar)
        if bar.split_ratio != 1:
            # Keep the split-session close as a new anchor. Never apply another adjustment.
            barrier = 'split_reset'
            break
    return list(reversed(history)), barrier
