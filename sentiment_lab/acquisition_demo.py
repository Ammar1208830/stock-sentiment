"""Small, explicitly synthetic M2 storage/export demonstration."""
from pathlib import Path

from .acquisition import AcquisitionConfig, export_dataset
from .calendar import generate_calendar
from .config import Config
from .contracts import session_date
from .normalization import normalized_version
from .store import Store


def demo(output: Path, cache: Path) -> dict:
    fixture = Path(__file__).parent / 'fixtures'
    config = Config.load(fixture / 'demo.toml')
    settings = AcquisitionConfig(start='2024-11-26', end='2024-12-04', price_availability='observed')
    sessions, metadata = generate_calendar(session_date(settings.start), session_date(settings.end))
    import json
    raw_texts = [json.loads(line) for line in (fixture / 'texts.jsonl').read_text().splitlines()]
    import csv
    with (fixture / 'prices.csv').open(newline='') as stream:
        prices = list(csv.DictReader(stream))
    with Store(cache) as store:
        store.bind(config.to_dict(), synthetic=True)
        snapshot = store.save_snapshot('synthetic_fixture', 'm2-demo-prices-v1', {}, prices, '2024-12-05T12:00:00+00:00')
        for row in prices:
            row.update(availability_basis='observed', retrieved_at='2024-12-05T12:00:00+00:00',
                       dividends=0, snapshot_id=snapshot['id'])
        for row in raw_texts:
            store.put_text(normalized_version(row, row['record_id']))
        return export_dataset(output, config, settings, sessions, metadata, prices, store,
            {'prices': {'status': 'synthetic_fixture'}, 'newsapi': {'status': 'synthetic_fixture'},
             'reddit': {'status': 'synthetic_fixture'}}, synthetic=True)
