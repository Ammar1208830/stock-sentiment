"""Strict input contracts shared by offline imports and future adapters."""

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
import re


class ContractError(ValueError):
    """Input would make an experiment ambiguous or temporally unsafe."""


def timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ContractError(f"Invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"Timestamp requires an explicit UTC offset: {value!r}")
    return parsed.astimezone(timezone.utc)


def session_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ContractError(f"Invalid session date: {value!r}") from exc
    if parsed.isoformat() != value:
        raise ContractError("Session dates must use YYYY-MM-DD")
    return parsed


def ticker(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", value):
        raise ContractError(f"Invalid ticker: {value!r}; use an uppercase exchange symbol")
    return value


def nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    return value.strip()


def finite_number(value: object, name: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be numeric")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ContractError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


@dataclass(frozen=True)
class Session:
    date: date
    open_at: datetime
    close_at: datetime

    @classmethod
    def from_dict(cls, row: dict) -> "Session":
        result = cls(session_date(row["session"]), timestamp(row["open_at"]), timestamp(row["close_at"]))
        if result.open_at >= result.close_at:
            raise ContractError("Session open must precede close")
        return result


@dataclass(frozen=True)
class PriceBar:
    ticker: str
    session: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    available_at: datetime
    split_ratio: float

    @classmethod
    def from_dict(cls, row: dict) -> "PriceBar":
        values = [finite_number(row[key], key, positive=True) for key in ("open", "high", "low", "close")]
        o, h, low, c = values
        volume = finite_number(row["volume"], "volume")
        if volume < 0 or not volume.is_integer():
            raise ContractError("volume must be a nonnegative integer")
        if not low <= min(o, c) <= max(o, c) <= h:
            raise ContractError("OHLC prices are inconsistent")
        return cls(ticker(row["ticker"]), session_date(row["session"]), *values,
                   int(volume), timestamp(row["available_at"]),
                   finite_number(row["split_ratio"], "split_ratio", positive=True))


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    source: str
    ticker: str
    text: str
    published_at: datetime
    available_at: datetime
    retrieved_at: datetime
    availability_basis: str
    origin_id: str | None = None

    @classmethod
    def from_dict(cls, row: dict) -> "TextRecord":
        result = cls(nonempty(row["record_id"], "record_id"), row["source"], ticker(row["ticker"]),
                     nonempty(row["text"], "text"), timestamp(row["published_at"]),
                     timestamp(row["available_at"]), timestamp(row["retrieved_at"]),
                     row["availability_basis"],
                     nonempty(row['origin_id'], 'origin_id') if row.get('origin_id') is not None else None)
        if result.source not in {"newsapi", "reddit"}:
            raise ContractError("source must be newsapi or reddit")
        if result.availability_basis not in {"observed", "historical_assumption"}:
            raise ContractError("availability_basis must be observed or historical_assumption")
        if not result.published_at <= result.available_at <= result.retrieved_at:
            raise ContractError("Text requires published_at <= available_at <= retrieved_at")
        return result
