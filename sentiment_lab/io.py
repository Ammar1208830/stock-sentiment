import csv
import hashlib
import json
from pathlib import Path

from .contracts import ContractError, PriceBar, Session, TextRecord


def read_csv(path: Path, record_type: type) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        records = []
        for line, row in enumerate(reader, 2):
            try:
                records.append(record_type.from_dict(row))
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError(f"{path.name}:{line}: {exc}") from exc
    if not records:
        raise ContractError(f"{path.name} contains no records")
    return records


def read_inputs(sessions: Path, prices: Path, texts: Path) -> tuple[list, list, list]:
    records = []
    with texts.open(encoding="utf-8") as stream:
        for line, value in enumerate(stream, 1):
            if not value.strip():
                continue
            try:
                row = json.loads(value)
                if not isinstance(row, dict):
                    raise ContractError("Text record must be a JSON object")
                records.append(TextRecord.from_dict(row))
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError(f"{texts.name}:{line}: {exc}") from exc
    return read_csv(sessions, Session), read_csv(prices, PriceBar), records


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def file_hash(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()
