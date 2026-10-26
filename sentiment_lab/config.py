from dataclasses import asdict, dataclass
from pathlib import Path
import math
import tomllib

from .contracts import ContractError, ticker


@dataclass(frozen=True)
class Config:
    schema_version: int = 1
    tickers: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMZN", "TSLA")
    exchange: str = "XNYS"
    timezone: str = "America/New_York"
    decision_minutes_before_open: int = 5
    seed: int = 42
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    allow_historical_assumptions: bool = False

    @classmethod
    def load(cls, path: Path) -> "Config":
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ContractError(f"Unknown configuration keys: {sorted(unknown)}")
        if "tickers" in raw:
            if not isinstance(raw["tickers"], list):
                raise ContractError("tickers must be an array")
            raw["tickers"] = tuple(raw["tickers"])
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ContractError("Only schema_version = 1 is supported")
        if not isinstance(self.tickers, (tuple, list)) or not self.tickers:
            raise ContractError("tickers must be a nonempty sequence")
        for item in self.tickers:
            ticker(item)
        if len(set(self.tickers)) != len(self.tickers):
            raise ContractError("tickers must be unique")
        if self.exchange != "XNYS" or self.timezone != "America/New_York":
            raise ContractError("M1 supports XNYS sessions in America/New_York")
        if type(self.decision_minutes_before_open) is not int or not 1 <= self.decision_minutes_before_open <= 60:
            raise ContractError("decision_minutes_before_open must be an integer in [1, 60]")
        if type(self.seed) is not int or self.seed < 0:
            raise ContractError("seed must be a nonnegative integer")
        for name in ("train_fraction", "validation_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
                raise ContractError(f"{name} must be between zero and one")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ContractError("Split fractions must leave a nonempty test fraction")
        if type(self.allow_historical_assumptions) is not bool:
            raise ContractError("allow_historical_assumptions must be a boolean")

    def to_dict(self) -> dict:
        return asdict(self)
