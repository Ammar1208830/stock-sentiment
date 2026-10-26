# Stock Sentiment Lab

Does financial sentiment improve next-session stock direction prediction beyond price-only models? This project tests that question with chronological evaluation, matched baselines and an honest interpretation of negative or inconclusive results.

**Current state: M1–M7 software implemented.** Frozen held-out evaluation, paired uncertainty, final report, cards, scientific figures and a reproducible local handoff are complete. **The empirical sentiment-value study remains inconclusive because eligible news/Reddit history is missing.** No profitability or sentiment-value claim is made.

- [Product requirements](docs/PRD.md)
- [30-day milestones and M1 checklist](docs/MILESTONES.md)
- [Input/output contracts](docs/DATA_CONTRACTS.md)
- [M2 setup and acquisition guide](docs/M2_ACQUISITION.md)
- [M3 FinBERT scoring guide](docs/M3_SENTIMENT.md)
- [Reviewed model smoke examples](docs/M3_SMOKE_REVIEW.md)
- [M4 technical features and leakage audit](docs/M4_FEATURES.md)
- [M5 models, baselines and frozen experiment](docs/M5_MODELS.md)
- [Website dashboard](website/README.md)
- [M6/M7 evaluation and final handoff](docs/M6_M7_EVALUATION_HANDOFF.md)

## Final result and handoff

The frozen 755-row test block spans 151 sessions. Price-only and combined LR both have AUC 0.4897; their delta AUC is 0 with paired 95% interval [0,0]. With no actual sentiment observations, that is an engineering diagnostic, not evidence for or against sentiment's value.

Open `artifacts/m7-release/report.html`, `report.md` or `handoff.zip`. The package contains source, processed frozen inputs/models and evaluation artifacts. A clean extracted copy reproduces predictions and numerical intervals without network calls. The local dashboard includes Final results, partition selection and report/handoff links.

## Frontend

The redesigned React frontend uses shadcn/ui and Lucide. See [frontend setup](website/README.md) for development, builds and local snapshots. Generated data and reports are excluded from Git; a fresh clone displays an empty state until a research snapshot is supplied.

## M5: models and baselines

```bash
.venv/bin/python -m sentiment_lab train-models \
  --features artifacts/m4-model-input --out artifacts/my-model-run
```

The completed run in `artifacts/m5-development` selected **10 LR/RF models from 40 settings** across five matched feature sets. It includes validation predictions, baseline comparisons, coefficients/forest explanations and a frozen protocol. M5 reserved 755 test rows without parsing their labels; M6 has now evaluated that unchanged frozen holdout.

Model selection uses validation AUC with predefined tie rules and expanding training-only diagnostics. Each fold fits its own preprocessing and purges unavailable labels. Sentiment data remains absent, so the source-value study is still data-gated. See the M5 guide for exact grids, baseline definitions, output files and post-freeze prediction behavior.

`artifacts/m4-model-input` is a fresh M4 export with physically separated partition-label files. Rebuild older M4 outputs if training reports those files missing. Full-environment dependencies are pinned in `requirements-modeling.lock`; the smaller classifier-only environment is `requirements-modeling-core.lock`.

## M4: matched features

```bash
.venv/bin/python -m sentiment_lab build-features \
  --dataset artifacts/m2-verified \
  --sentiment artifacts/m3-verified-price-only \
  --out artifacts/my-feature-run
```

M4 produces five matched feature sets: price-only, price + news, price + Reddit, sentiment-only and combined. Indicators use known prior-session prices only, restart at splits/gaps, and retain the original chronological partitions. Raw missing sentiment remains null; numerical filling/scaling fits on training rows only. Labels stay separate.

The verified real-price run retains **3,709 rows** after 60 warmup/reset exclusions. Inspect `artifacts/m4-verified/summary.json`, `leakage_audit.json` and `feature_schema.json`. The combined schema has 19 features, but sentiment observations are still absent, so the original research data gate remains unmet. See the M4 guide for formulas, preprocessing, frozen column allowlists and M5 fold-refitting requirements.

## M3: sentiment scoring

The local model and inference environment are prepared. Run real FinBERT on six clearly synthetic examples:

```bash
.venv/bin/python -m sentiment_lab sentiment-smoke --out artifacts/my-finbert-smoke
```

Score an M2 dataset and aggregate eligible text by ticker, target session and source:

```bash
.venv/bin/python -m sentiment_lab score-sentiment \
  --dataset artifacts/m2-verified --out artifacts/my-sentiment-run
```

The current price dataset has no eligible news/Reddit text, so its 3,769 rows retain missing sentiment; M3 does not fabricate scores. The separate M2 synthetic dataset can demonstrate real-model aggregation using `--dataset artifacts/m2-demo --config sentiment_lab/fixtures/demo.toml --cache data/cache/sentiment-demo.sqlite` and a fresh output path.

To recreate M3, install `requirements-sentiment.lock` under Python 3.12 and run `prepare-finbert` once. Inference defaults to local model files only. See the M3 guide for model revision, truncation, cache behavior and output schemas.

## M2: data acquisition

The local `.venv` is ready. From this project folder:

```bash
.venv/bin/python -m sentiment_lab collect --offline --out artifacts/my-cached-data
.venv/bin/python -m sentiment_lab m2-demo --out artifacts/my-m2-demo
.venv/bin/python -m unittest discover -s tests -v
```

The completed live download contains **3,775 price bars for five stocks**, over October 2023–October 2026. Inspect `artifacts/m2-verified/DATA_CARD.md` and `feasibility.json`. NewsAPI has no configured key, and no approved Reddit export is present, so the full sentiment-study data gate remains unmet. The report says `engineering_demonstration_only`; the prices are real provider data, while the separate M2 demo is synthetic.

For a fresh network collection, omit `--offline`. Successful responses remain in `data/cache/acquisition.sqlite`; new output folders reuse them. `--refresh-prices` and `--refresh-news` explicitly fetch new observations. Use the setup guide for credentials, approved historical imports, date configuration, availability assumptions, request limits and dataset provenance. Recreate the acquisition environment with `python -m pip install -r requirements-acquisition.lock` under Python 3.12.

## Quick start

Python 3.11 or newer. M1 uses only the Python standard library and needs no API keys or downloads. From this folder:

```bash
python3 -m sentiment_lab doctor
python3 -m unittest discover -s tests -v
python3 -m sentiment_lab demo --out artifacts/my-first-demo
```

Use a fresh `--out` path on subsequent runs. The included fixture has **invented prices and text** across actual-format exchange sessions around a holiday and early close. It produces 12 ticker/session rows, with no sentiment scores or trained models.

The output directory contains `windows.jsonl`, `labels.jsonl`, `validation.json` and `manifest.json`. Features and labels are separated, and the manifest records hashes, configuration, code identity, exclusions and synthetic status. These files are for inspecting the pipeline; they are not research results.

Optional isolated environment and command installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
sentiment-lab doctor
```

The source-module commands above work without installation. Packaging uses setuptools. M2 and M3 provide separate pinned acquisition and acquisition-plus-sentiment environments; core validation remains standard-library-only.

## Validate local normalized inputs

```bash
python3 -m sentiment_lab validate \
  --config configs/project.toml \
  --sessions data/raw/sessions.csv \
  --prices data/raw/prices.csv \
  --texts data/raw/texts.jsonl \
  --out artifacts/local-validation-001
```

The files in that example are your future inputs, not files downloaded by M1. Follow the contracts linked above. An empty text JSONL is valid, preserving missingness. A malformed record or unknown ticker fails clearly. At least four supplied sessions and three usable target dates are needed for a three-way split.

## Forecast convention

At five minutes before target-session open, use prices through the preceding session and text published since the preceding close that was actually available before the decision. Predict whether the target close exceeds the previous close. A flat move maps to non-up. Raw price labels exclude target split sessions. This is a research forecast of close-to-close direction; it does not imply a trade could capture the overnight gap.

Actual session times determine holidays, early closes and DST. M1 validates a supplied calendar's structure. M2 generates and verifies completeness against the pinned `exchange_calendars` XNYS schedule. Use `calendar` and `verify-calendar` for real imports.

## Research constraints

Thirty development days do not supply enough daily observations for the proposed statistical study. M2 must obtain sufficiently long aligned text and prices or explicitly report a feasibility study. NewsAPI Developer delays and Reddit's research-access requirements are material source constraints documented with official references in the PRD. Do not infer historical availability from publication time or assume ordinary API credentials provide complete historical research data.

## Layout

```text
configs/                 default five-ticker research configuration
sentiment_lab/           contracts, config, IO, alignment, splits and CLI
sentiment_lab/fixtures/  small synthetic offline dataset
tests/                   temporal integrity and reproducibility checks
docs/                    PRD, milestones and data contracts
data/raw/                local source data (ignored)
data/processed/          derived datasets (ignored)
artifacts/               local run outputs (ignored)
.github/workflows/       Python 3.11–3.13 checks for future repository use
```

The remaining research work is source-dependent: obtain enough permitted, correctly timestamped news/Reddit history, audit it, and reserve a fresh holdout before a new confirmatory study. The original holdout has now been revealed.
