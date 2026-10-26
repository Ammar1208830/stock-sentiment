"""M7: generated report, scientific figures and a verified local handoff."""
import base64
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from . import __version__
from .contracts import ContractError
from .features import read_json, read_jsonl, same_structure
from .io import file_hash, write_json
from .modeling import empty_output, verified_freeze
from .store import digest, utc_now


DISPLAY_MODELS = ['logistic_regression:price_only', 'logistic_regression:combined',
                  'random_forest:price_only', 'random_forest:combined',
                  'baseline:always_up', 'baseline:momentum', 'baseline:coin_seed_42']


def verified_evaluation(evaluation: Path, features: Path, experiment: Path, predictions: Path):
    manifest = read_json(evaluation / 'manifest.json')
    freeze = verified_freeze(experiment)
    expected = {'freeze': file_hash(experiment / 'freeze.json'), 'features': file_hash(features / 'manifest.json'),
                'prediction_manifest': file_hash(predictions / 'manifest.json'), 'test_labels': file_hash(features / 'labels_test.jsonl')}
    if manifest.get('freeze_id') != freeze['freeze_id'] or manifest.get('input_sha256') != expected:
        raise ContractError('M7 evaluation provenance differs from the requested frozen inputs')
    for name, value in manifest['output_sha256'].items():
        if Path(name).name != name or file_hash(evaluation / name) != value:
            raise ContractError('Evaluation artifact is missing or changed')
    return manifest, freeze


def figures(evaluation, predictions, output):
    os.environ.setdefault('MPLCONFIGDIR', str((output / '.mplcache').resolve()))
    os.environ.setdefault('XDG_CACHE_HOME', str((output / '.cache').resolve()))
    (Path(os.environ['XDG_CACHE_HOME']) / 'fontconfig').mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError as exc:
        raise ContractError('Install requirements-final.lock to create the report figures') from exc
    figure_dir = output / 'figures'
    figure_dir.mkdir()
    results = {r['model_id']: r for r in read_json(evaluation / 'test_metrics.json')['models']}
    chosen = [name for name in DISPLAY_MODELS if name in results]
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'}):
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        labels = [name.replace('logistic_regression:', 'LR / ').replace('random_forest:', 'RF / ').replace('baseline:', 'Baseline / ').replace('_', ' ') for name in chosen]
        values = [results[name]['roc_auc'] for name in chosen]
        valid = [(label, value) for label, value in zip(labels, values) if value is not None]
        ax.barh([r[0] for r in valid][::-1], [r[1] for r in valid][::-1], color='#208878', height=.6)
        ax.axvline(.5, color='#b68b48', linestyle='--', label='Constant-score AUC = 0.5')
        ax.set(xlim=(0, 1), xlabel='Held-out ROC-AUC', title='Frozen held-out comparisons — engineering run')
        ax.legend(frameon=False, loc='lower right')
        fig.text(.02, .015, 'No sentiment observations are present. These bars do not test sentiment value.', fontsize=9, color='#775f42')
        fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(figure_dir / 'heldout_auc.png', dpi=180); plt.close(fig)

        main = read_json(evaluation / 'uncertainty.json')['results'][0]['comparisons']
        valid = [r for r in main if r['delta_auc']['interval'] is not None]
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        for index, record in enumerate(valid):
            point, (lo, hi) = record['delta_auc']['estimate'], record['delta_auc']['interval']
            ax.plot([lo, hi], [index, index], color='#218777', lw=3)
            ax.scatter([point], [index], color='#218777', zorder=3)
        ax.axvline(0, color='#b68b48', linestyle='--')
        ax.set_yticks(range(len(valid)), [r['id'].replace('logistic_regression:', 'LR / ').replace('random_forest:', 'RF / ') for r in valid])
        extents = [abs(v) for r in valid for v in [r['delta_auc']['estimate'], *r['delta_auc']['interval']]]
        extent = max(.02, max(extents, default=0) * 1.3)
        ax.set(xlim=(-extent, extent), xlabel='Delta ROC-AUC versus matched price-only model', title='Paired 95% intervals — five-session moving blocks')
        fig.text(.02, .015, 'Zero-width intervals arise when predictions are identical; absent source data prevents a sentiment inference.', fontsize=8.5, color='#775f42')
        fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(figure_dir / 'paired_intervals.png', dpi=180); plt.close(fig)

        labels = { (r['ticker'], r['target_session']): r['label_up'] for r in read_jsonl(evaluation / 'paired_primary.jsonl') }
        grouped = {}
        for row in read_jsonl(predictions / 'test_predictions.jsonl'):
            if row['model_id'] in ('logistic_regression:price_only', 'random_forest:price_only', 'baseline:momentum'):
                grouped.setdefault(row['model_id'], []).append(row)
        fig, ax = plt.subplots(figsize=(6.8, 5.5))
        for model, records in grouped.items():
            y = [labels[(r['ticker'], r['target_session'])] for r in records]
            if len(set(y)) == 2:
                fpr, tpr, _ = roc_curve(y, [r['up_probability'] for r in records])
                ax.plot(fpr, tpr, label=model.replace('logistic_regression:', 'LR / ').replace('random_forest:', 'RF / ').replace('baseline:', 'Baseline / '))
        ax.plot([0, 1], [0, 1], '--', color='#9c9c9c')
        ax.set(xlabel='False-positive rate', ylabel='True-positive rate', title='Frozen held-out ROC curves', xlim=(0, 1), ylim=(0, 1))
        ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(figure_dir / 'roc.png', dpi=180); plt.close(fig)


def report_text(summary, conclusion, metrics, protocol, feature_summary, universe, feature_dates):
    def fmt(value):
        return 'N/A' if value is None else f'{value:.4f}'
    table = '\n'.join(f"| {row['model_id']} | {fmt(row['roc_auc'])} | {fmt(row['f1'])} | {fmt(row['accuracy'])} |" for row in metrics if not row['model_id'].startswith('baseline:coin_seed_') or row['model_id'] == 'baseline:coin_seed_42')
    interval = summary['primary_delta_auc']['interval']
    return f'''# Stock Market Sentiment Analysis: Final Project Report

Version {__version__} · 26 October 2026 · Frozen held-out engineering evaluation

## Abstract and outcome

This project implements a reproducible pipeline to test whether news and retail-discussion sentiment improves next-session stock-direction prediction over price-only and naive baselines. The software includes source ingestion/import contracts, FinBERT scoring, timestamp alignment, technical features, chronological development, fixed model selection and paired held-out uncertainty estimates.

**The empirical sentiment hypothesis remains inconclusive.** The completed run has real vendor prices but zero eligible news/Reddit observations. Combined feature sets therefore do not contain actual sentiment information. Their numerical comparisons must not be interpreted as evidence that sentiment works or does not work.

## Research question and design

The primary comparison was fixed before test evaluation: combined Logistic Regression versus matched price-only Logistic Regression, measured by held-out delta ROC-AUC. Random Forest and source ablations are secondary exploratory comparisons. F1 uses a fixed 0.5 threshold. No model or preprocessing was refitted after the freeze, and no winning ticker or model family replaced the primary comparison.

At five minutes before a target exchange session opens, use vendor prices through the previous session and overnight text published since the preceding close and available before the cutoff. Predict whether target close exceeds previous close. Flat changes map to non-up. This close-to-close target includes the overnight gap and does not establish an executable trading return.

## Data and features

The evaluated convenience universe is {', '.join(universe)}. Retained feature target dates span {feature_dates[0]}–{feature_dates[1]}; acquisition includes preceding history and warmup rows. Calendar definitions come from pinned XNYS schedules. FinBERT is pinned to revision `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`; real CPU inference and truncation were verified on authored examples. One sarcasm example was misclassified, and that review is not an independent sentiment benchmark.

Five technical features are 1-/5-session returns, 10-session moving-average distance, sample volatility of ten returns, and Wilder 5-session RSI. Each text source contributes count, missingness, mean class probabilities, mean polarity and polarity dispersion. Split/gap barriers reset histories; warmup requires eleven consecutive known closes. There is no forward/backward fill of source observations.

After technical filtering there are {feature_summary['retained_rows']} rows: {feature_summary['partition_rows']['train']} train, {feature_summary['partition_rows']['validation']} validation and {feature_summary['partition_rows']['test']} test. Price-only and all source ablations use exactly matched keys. Raw sentiment remains null; train-only numerical placeholders are explicitly recorded as all-missing, not neutral observations.

## Models and baselines

Two classifiers across five feature sets produce ten selected models. Four settings per family/set give forty candidate configurations. Expanding folds are grouped by whole date across tickers. Training labels unresolved at the next decision boundary are purged; each fold fits new training-only median/scaling statistics. Validation AUC selects settings with fixed tie rules. Final estimators remain fitted on original train only.

Baselines are training-majority, always-up, literal no-change, prior-return momentum and twenty seeded coin classifiers. Always-up is a directional analogue of buy-and-hold, not a portfolio backtest. Constant-score AUC is 0.5 when both classes exist. Single-class AUC is N/A.

## Held-out results

The untouched test period is **{summary['test_start']} through {summary['test_end']}**, with **{summary['test_rows']} rows across {summary['distinct_test_sessions']} distinct sessions**. Up-class prevalence is {summary['positive_prevalence']:.2%}. All 34 model/baseline variants predict the same complete sample.

| Model / baseline | ROC-AUC | F1 | Accuracy |
| --- | --- | --- | --- |
{table}

![Held-out comparisons](figures/heldout_auc.png)

## Paired uncertainty and ablation

The frozen protocol specifies 2,000 paired non-circular moving-block bootstrap replicates, five consecutive test sessions per block, resampling dates jointly across stocks and model predictions. Percentile 95% intervals use linear quantiles. Sensitivity repeats the procedure for blocks of one and ten sessions. Invalid single-class AUC replicates are counted and omitted from AUC intervals; F1 remains defined under the documented zero-division rule.

Primary delta AUC is **{summary['primary_delta_auc']['estimate']}**, with 95% interval **{interval}**. Primary delta F1 is **{summary['primary_delta_f1']['estimate']}**. When probabilities coincide, the paired delta is identically zero, so an interval can collapse to [0,0]. This is a property of identical predictions, not proof of sentiment equivalence or a powered null result.

![Paired intervals](figures/paired_intervals.png)

![Held-out ROC curves](figures/roc.png)

Per-ticker/macro metrics, first/second-half diagnostics, twenty-seed random-baseline variability, and the text-present subset are exported separately. These are exploratory and never replace the full-sample primary endpoint. The current text-present subset has {summary['sentiment_covered_test_rows']} rows and cannot supply a source-quality evaluation.

## Interpretation and limitations

{conclusion['statement']}

The price-only LR test AUC is {fmt(summary['primary_control_metrics']['roc_auc'])}, close to the no-ranking benchmark; its F1 must be judged alongside class prevalence and always-up predictions. No profitability, causal effect, or future-stability claim follows from these classification metrics.

Material limits include missing textual history, historical price availability assumed at close + 1 minute, retrospective vendor split adjustments/revisions, present-day convenience-universe selection and survivorship bias, correlated stocks, limited test dates, bootstrap stationarity/block-length assumptions, FinBERT domain errors and uncalibrated sentiment scores. The 30 days describe development, not the observation period or a statistical-power guarantee.

## Delivered scope and remaining work

M1–M7 software deliverables are implemented: pipeline, real-model inference, matched features, baselines/ablations, frozen development, held-out evaluation, uncertainty, figures, cards, report and local demonstration. **The original combined-source empirical study is not completed** because appropriately permitted historical news/Reddit data is absent. Obtain those sources, review real text quality and timestamps, amend/freeze a new experiment before opening a fresh test result, and repeat the study. Do not tune against this already revealed test block.

## Reproducibility and references

See REPRODUCE.md, DATA_CARD.md, MODEL_CARD.md and manifest.json. The local handoff contains source/configuration, frozen processed inputs/models and predictions, avoiding API keys, raw Reddit/news content, virtual environments and FinBERT weight downloads. A clean extracted copy is checked against the frozen numerical outputs; source acquisition still requires its original access and permissions.

Primary implementation references: [FinBERT model card](https://huggingface.co/ProsusAI/finbert), [scikit-learn AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [train-only preprocessing](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [NewsAPI access](https://newsapi.org/pricing), [Reddit research access](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data). Exact formulas, code, seeds, manifests and source assumptions are the reproducibility authority for this run.
'''


def html_report(markdown, output):
    # Small controlled renderer for the generated report, avoiding another runtime dependency.
    parts, paragraph, table = [], [], []
    def inline(text):
        import re
        text = escape(text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        return re.sub(r'\[([^\]]+)\]\((https://[^\s)]+)\)', r'<a href="\2">\1</a>', text)
    def flush():
        if paragraph:
            parts.append('<p>' + inline(' '.join(paragraph)) + '</p>'); paragraph.clear()
        if table:
            parts.append('<div class="table"><table>')
            for index, row in enumerate(table):
                if index == 1:
                    continue
                tag = 'th' if index == 0 else 'td'
                parts.append('<tr>' + ''.join(f'<{tag}>{inline(cell.strip())}</{tag}>' for cell in row.strip('|').split('|')) + '</tr>')
            parts.append('</table></div>'); table.clear()
    for line in markdown.splitlines():
        if line.startswith('!['):
            flush(); path = line.split('](')[1][:-1]; payload = base64.b64encode((output / path).read_bytes()).decode()
            parts.append(f'<img alt="{escape(line[2:line.index("]")])}" src="data:image/png;base64,{payload}">')
        elif line.startswith('#'):
            flush(); level = len(line) - len(line.lstrip('#')); parts.append(f'<h{level}>' + inline(line[level:].strip()) + f'</h{level}>')
        elif line.startswith('|'):
            if paragraph: flush()
            table.append(line)
        elif not line:
            flush()
        else:
            if table: flush()
            paragraph.append(line)
    flush()
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stock Sentiment Lab — Final Report</title><style>body{font:16px/1.8 system-ui;color:#203936;background:#f6f8f5;margin:0}main{max-width:980px;margin:30px auto;padding:45px;background:white;border:1px solid #e0e9e2;border-radius:12px}h1{font-size:36px;line-height:1.2}h2{margin-top:42px;font-size:24px}p{color:#48625a}a{color:#087f79}strong{color:#193d31}table{border-collapse:collapse;font-size:12px;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #dfebe1}.table{overflow:auto}img{max-width:100%;margin:20px 0}code{font-size:12px;background:#eff5ef;padding:3px}@media(max-width:650px){main{margin:0;padding:22px}h1{font-size:28px}}</style><main>' + ''.join(parts) + '</main></html>'


def package_handoff(project, output, features, experiment, predictions, evaluation):
    archive = output / 'handoff.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
        for folder in ('sentiment_lab', 'tests', 'configs', 'docs', '.github', 'website'):
            for path in sorted((project / folder).rglob('*')):
                if not path.is_file() or any(part in {'.git', '__pycache__', '.openai'} for part in path.parts) or path.suffix == '.pyc' or path.name in {'dashboard-preview.jpg', 'handoff.zip', 'report.html'}:
                    continue
                bundle.write(path, 'source/' + path.relative_to(project).as_posix())
        bundle.write(output / 'report.html', 'source/website/dist/report.html')
        for path in [project / 'pyproject.toml', project / 'README.md', *sorted(project.glob('requirements*.lock')), *sorted(project.glob('requirements*.in'))]:
            bundle.write(path, 'source/' + path.name)
        for directory, name in ((features, 'm4'), (experiment, 'm5'), (predictions, 'predictions'), (evaluation, 'evaluation')):
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    bundle.write(path, 'source/inputs/' + name + '/' + path.name)
        for name in ('report.md', 'report.html', 'DATA_CARD.md', 'MODEL_CARD.md', 'REPRODUCE.md'):
            bundle.write(output / name, 'report/' + name)
        for path in sorted((output / 'figures').iterdir()):
            bundle.write(path, 'report/figures/' + path.name)
    return archive


def check_clean_handoff(archive, original_evaluation, original_predictions):
    with tempfile.TemporaryDirectory(prefix='sentiment-handoff-') as temporary:
        with ZipFile(archive) as bundle:
            bundle.extractall(temporary)
        root = Path(temporary) / 'source'
        env = os.environ.copy(); env.pop('PYTHONPATH', None)
        commands = [
            [sys.executable, '-m', 'sentiment_lab', 'predict-holdout', '--features', 'inputs/m4', '--experiment', 'inputs/m5', '--out', 'reproduced-predictions'],
            [sys.executable, '-m', 'sentiment_lab', 'evaluate', '--features', 'inputs/m4', '--experiment', 'inputs/m5', '--predictions', 'reproduced-predictions', '--out', 'reproduced-evaluation']]
        for command in commands:
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise ContractError('Clean extracted handoff reproduction failed: ' + result.stderr[-500:])
        if file_hash(root / 'reproduced-predictions/test_predictions.jsonl') != file_hash(original_predictions / 'test_predictions.jsonl'):
            raise ContractError('Clean handoff predictions differ from the frozen original')
        for name in ('test_metrics.json', 'uncertainty.json', 'summary.json'):
            if not same_structure(read_json(root / 'reproduced-evaluation' / name), read_json(original_evaluation / name)):
                raise ContractError('Clean handoff numerical results differ: ' + name)
    return {'status': 'passed', 'clean_extracted_source': True, 'network_required': False,
            'predictions_byte_identical': True, 'metrics_and_bootstrap_reproduced': True,
            'limitation': 'Uses the current installed pinned environment and packaged normalized inputs; not a fresh dependency installation or raw-source acquisition.'}


def finalize(features: Path, experiment: Path, predictions: Path, evaluation: Path, output: Path):
    empty_output(output)
    _, freeze = verified_evaluation(evaluation, features, experiment, predictions)
    summary, conclusion = read_json(evaluation / 'summary.json'), read_json(evaluation / 'conclusion.json')
    results = read_json(evaluation / 'test_metrics.json')['models']
    protocol, feature_summary = read_json(experiment / 'protocol.json'), read_json(features / 'summary.json')
    source_rows = read_jsonl(features / 'combined.jsonl')
    universe = sorted({r['ticker'] for r in source_rows})
    feature_dates = (min(r['target_session'] for r in source_rows), max(r['target_session'] for r in source_rows))
    if 'retained_rows' not in feature_summary:
        feature_summary = {**feature_summary, 'retained_rows': len(source_rows),
            'partition_rows': {p: sum(r['partition'] == p for r in source_rows) for p in ('train', 'validation', 'test')}}
    output.mkdir(parents=True, exist_ok=True)
    figures(evaluation, predictions, output)
    report = report_text(summary, conclusion, results, protocol, feature_summary, universe, feature_dates)
    (output / 'report.md').write_text(report, encoding='utf-8')
    (output / 'report.html').write_text(html_report(report, output), encoding='utf-8')
    (output / 'DATA_CARD.md').write_text(f'''# Data card

Universe: {', '.join(universe)}. Convenience sample; selection/survivorship limitations apply.

Retained feature target range: {feature_dates[0]}–{feature_dates[1]}. Vendor OHLCV, explicit corporate actions and pinned XNYS schedules. Historical price availability is an assumption, not independent point-in-time proof. FinBERT uses a pinned public financial sentiment model.

Retained rows: {feature_summary['retained_rows']}; partitions {feature_summary['partition_rows']}.
Test: {summary['test_start']}–{summary['test_end']}, {summary['test_rows']} rows / {summary['distinct_test_sessions']} dates.
Eligible sentiment-covered test rows: {summary['sentiment_covered_test_rows']}.
Data gate: {summary['source_data_gate']}.

Raw missing sentiment remains null; transformed all-missing fields use disclosed numeric placeholders. No fabricated source observations. Historical news and approved Reddit research data remain absent. No Reddit author/profile attributes are packaged. Raw source redistribution/retention permissions are not granted by this handoff.
''', encoding='utf-8')
    (output / 'MODEL_CARD.md').write_text(f'''# Model card

Intended use: reproducible academic stock-direction research and engineering demonstration. Not live trading or personalized investment advice.

Ten frozen models: LR/RF across five matched feature sets. Seed/grid/preprocessing/threshold are bound to freeze `{freeze['freeze_id']}`. Train-only fitting and expanding-fold label purges precede validation selection; final models are not refitted after holdout revelation.

Held-out primary LR AUC: {summary['primary_control_metrics']['roc_auc']:.4f}; F1: {summary['primary_control_metrics']['f1']:.4f}. Positive prevalence: {summary['positive_prevalence']:.4f}. Class 1 is up; zero change maps to class 0. Threshold 0.5; no calibration or profitability claim.

Research status: {conclusion['research_status']}. {conclusion['statement']}

Known limitations: missing sentiment, vendor revisions/adjustments, convenience universe, regime shift, correlated equities, finite dates and block-bootstrap assumptions. FinBERT sentiment is not a stock-direction probability. Inspect model JSON for signed LR coefficients and exploratory validation permutation importance. Secondary test slices do not select a new winner.
''', encoding='utf-8')
    (output / 'REPRODUCE.md').write_text('''# Reproduce the frozen result

Unzip handoff.zip, enter `source`, create a Python 3.12 environment and install `requirements-modeling-core.lock`. Matplotlib is only needed to regenerate report figures; its full reference environment is `requirements-final.lock`.

```bash
python -m sentiment_lab predict-holdout --features inputs/m4 --experiment inputs/m5 --out reproduced-predictions
python -m sentiment_lab evaluate --features inputs/m4 --experiment inputs/m5 --predictions reproduced-predictions --out reproduced-evaluation
python -m unittest discover -s tests -v
```

The included processed M4 inputs and frozen M5 experiment reproduce the reported predictions and intervals without API calls or FinBERT downloads. Do not modify hashed inputs in place. The holdout has now been revealed: subsequent model changes are exploratory and require a fresh holdout for new confirmatory claims.

Full raw-source reproduction still requires permitted historical news/social data and the original source access. This handoff does not claim those missing sources were collected. External Sites upload was not authorized; the dashboard and package remain local.
''', encoding='utf-8')
    project = Path(__file__).parent.parent
    archive = package_handoff(project, output, features, experiment, predictions, evaluation)
    check = check_clean_handoff(archive, evaluation, predictions)
    write_json(output / 'reproduction_check.json', check)
    manifest = {'schema_version': 1, 'version': __version__, 'created_at': utc_now().isoformat(),
                'freeze_id': freeze['freeze_id'], 'evaluation_manifest_sha256': file_hash(evaluation / 'manifest.json'),
                'research_status': conclusion['research_status'], 'software_milestones_implemented': list(range(1, 8)),
                'empirical_sentiment_study_complete': conclusion['sentiment_hypothesis_testable'],
                'reproduction': check, 'files_sha256': {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob('*'))
                    if p.is_file() and not {'.mplcache', '.cache'}.intersection(p.parts)}}
    write_json(output / 'manifest.json', manifest)
    return {'output': str(output.resolve()), 'report': str((output / 'report.html').resolve()),
            'handoff': str(archive.resolve()), 'reproduction': check['status'], 'research_status': conclusion['research_status']}
