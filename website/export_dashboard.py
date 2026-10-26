"""Refresh the local dashboard from saved artifacts; performs no network calls."""
import csv
import json
from pathlib import Path


def export(project: Path):
    def read(path):
        return json.loads((project / path).read_text())
    path = project / 'website/dist/data.json'
    data = json.loads(path.read_text())
    data['evaluation'] = read('artifacts/m6-evaluation/summary.json')
    data['uncertainty'] = read('artifacts/m6-evaluation/uncertainty.json')
    data['conclusion'] = read('artifacts/m6-evaluation/conclusion.json')
    previous = {row['model_id']: row for row in data['metrics']}
    data['testMetrics'] = [{**previous.get(row['model_id'], {}), **row, 'partition': 'test'}
                           for row in read('artifacts/m6-evaluation/test_metrics.json')['models']]
    data['evaluationComplete'] = True
    for milestone in data['milestones']:
        milestone['done'] = True
    data['releaseStatus'] = 'Software complete; original empirical sentiment study data-gated.'
    path.write_text(json.dumps(data, separators=(',', ':')))
    print('Local dashboard refreshed with frozen held-out results.')


if __name__ == '__main__':
    export(Path(__file__).resolve().parents[1])
