# M5: models, baselines and frozen experiment

M5 trains Logistic Regression and Random Forest across the five matched M4 feature sets, records chronological development diagnostics, selects hyperparameters using validation data, and freezes models and the final evaluation protocol. The real 755-row test partition is reserved for M6. No test-performance or statistical-significance result is claimed in M5.

## Run

The existing `.venv` includes the modeling dependencies. From the project root:

```bash
.venv/bin/python -m sentiment_lab train-models \
  --features artifacts/m4-model-input \
  --out artifacts/my-model-run
```

Settings are in `configs/modeling.toml`. A fresh output folder is required. A run is complete only when `freeze.json` exists and its fingerprint/output hashes verify. `progress.json` records completed model groups during a partial run; an interrupted run is not a frozen experiment.

To recreate the full project environment under Python 3.12, install `requirements-modeling.lock`. For acquisition/features/classifiers without the FinBERT runtime, use `requirements-modeling-core.lock`. The tested classifier environment includes scikit-learn 1.9.1 and SciPy 1.18.1. The full lock was exercised locally on macOS arm64; CI is configured for Python 3.12 pipeline tests without downloading FinBERT.

## M4 partition-label update

M4 now writes `labels_train.jsonl`, `labels_validation.jsonl` and `labels_test.jsonl` in addition to its existing combined label file. M5 parses only the first two files during training. It never parses `labels_test.jsonl` or the all-partition `labels.jsonl` for model selection. This separation is also tested with deliberately unreadable holdout label files.

Older immutable M4 artifacts remain unchanged. `artifacts/m4-model-input` is a fresh export with the same 3,709 rows and the new files. If using an older feature export, rebuild it:

```bash
.venv/bin/python -m sentiment_lab build-features \
  --dataset artifacts/m2-verified \
  --sentiment artifacts/m3-verified-price-only \
  --out artifacts/my-model-input
```

The training loader validates hashes, frozen feature-schema equality, ordered sample matching across all five sets, matching column values, date-level partition consistency, chronological decision times, binary development labels and development-label availability at the next partition boundary. It uses raw allowlisted M4 matrices, not the previously fitted `scaled_*` files.

## Development protocol

Both classifier families run on each of:

- Price-only: 5 features.
- Price + news: 12 features.
- Price + Reddit: 12 features.
- Sentiment-only: 14 features.
- Combined: 19 features.

All comparisons have identical target rows and original outer date partitions. Default search grids contain four settings per family/feature set, for 40 candidates total:

| Family | Search grid | Fixed settings |
| --- | --- | --- |
| Logistic Regression | C = 0.01, 0.1, 1, 10 | lbfgs, max_iter 2000, no class weighting, seed 42 |
| Random Forest | max_depth = 3 or 6; min_samples_leaf = 5 or 15 | 200 trees, bootstrap, max_features 1.0, one worker, no class weighting, seed 42 |

The grids are recorded before fitting, and validation/test results do not expand them. Configuration rejects more than eight candidates per family/feature set. Thread pools are limited to one during fitting/prediction. The [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) implementations come from the pinned scikit-learn environment.

For each setting, three expanding folds use **only the outer training partition**. Folds split distinct target-session dates, keeping all stocks on a date together. Each fold purges training rows whose labels are not available strictly before its first evaluation decision. Default initial fold training requires at least 60 eligible distinct sessions.

Every feature-set/fold fits its own median fill and population scaler on that fold's eligible training rows. Outer training also fits a fresh transform. M4 fitted statistics are not reused. Prior evaluation-session prices may appear in later historical features once known, but evaluation labels never enter fitting. This follows [train-only fitting guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Candidates rank by outer validation ROC-AUC rounded to 12 decimals; ties use mean defined inner-fold AUC, then declared grid order. Inner folds provide chronological diagnostics and the predeclared tie-break, rather than replacing the validation criterion. If validation has one class, AUC is null and the first declared candidate is selected **untuned**, explicitly recorded. A single-class inner training fold is skipped and logged. Single-class outer training fails clearly because LR/RF cannot provide the intended binary comparison.

Selected estimators remain fitted on the original training block. There is no hidden refit on train + validation. The primary comparison stays combined Logistic Regression versus price-only Logistic Regression; Random Forest does not replace it after looking at development performance.

## Baselines

Each baseline uses the same validation sample as the models:

| Baseline | Score/decision |
| --- | --- |
| training_majority | Constant up-class prevalence learned from training only; threshold 0.5, ties predict up |
| always_up | Constant score 1; directional analogue of an always-long/buy-and-hold policy |
| zero_drift_no_change | Previous-close point forecast maps to non-up under the target's tie rule; constant directional score 0 |
| momentum | Score 1 when the known prior one-session return is positive, otherwise 0 |
| coin_seed_* | Uniform score derived from seed + ticker + target date; threshold 0.5 |

The coin baseline has 20 default seeds, 42–61. Scores are stable when row order changes and independent of labels. `coin_variability.json` reports development F1/AUC variation across seeds. The continuous coin score supports an AUC comparison; it is not a calibrated probability. Constant baseline scores have AUC 0.5 when both classes exist. A one-class evaluation slice has undefined AUC, saved as null with a status, never a fabricated number.

The no-change price point forecast and the random coin classifier are distinct baselines. Always-up predictions are not an executable buy-and-hold return series. M5 reports no portfolio profits, transaction-cost-adjusted returns or investment recommendation.

## Metrics and interpretation

Validation diagnostics include positive-class F1 at fixed threshold 0.5, ROC-AUC, accuracy, precision, recall, class prevalence and confusion matrix. Per-candidate/fold diagnostics are saved. These validation numbers support model development; they are not final out-of-sample test evidence.

Logistic artifacts include signed coefficients in standardized and original imputed-input scales. Forest artifacts include validation-only permutation importance measured by AUC drop across five seeded repeats. Correlated columns may mask one another's importance. Explanations are exploratory associations, not causal effects, and do not trigger post-hoc feature selection or another tuning search.

Model states are portable JSON: LR coefficients/intercept and RF tree nodes/leaf probabilities. Preprocessing receipts and ordered column names are saved alongside each state. The exporter verifies portable predictions against native scikit-learn validation probabilities before freezing. Forest prediction reproduces scikit-learn's float32 input casting. No model pickle needs to be loaded.

## Freezing and later holdout prediction

`protocol.json` records settings, grids, splits, preprocessing rules and the M6 plan before candidate fitting. At completion `freeze.json` binds the selected model files, protocol, diagnostics, baseline parameters, code/environment identity and feature-manifest hash. It includes a fingerprint and SHA-256 hashes of every experiment artifact. These checks detect changes; they are not independent authentication of source data or a signed preregistration service.

The frozen M6 plan names paired held-out delta AUC for combined LR minus price-only LR, 2,000 paired date-block bootstrap replicates, block length 5 sessions with 1/10-session sensitivity, 95% intervals and fixed F1 threshold 0.5. Secondary model/source comparisons remain exploratory. M6 will implement the evaluation; M5 only records this previously specified plan.

A separate command can predict **after** freezing:

```bash
.venv/bin/python -m sentiment_lab predict-holdout \
  --features artifacts/m4-model-input \
  --experiment artifacts/m5-development \
  --out artifacts/my-frozen-holdout-predictions
```

It validates the freeze and bound M4 manifest, reloads JSON models, uses frozen preprocessing/baselines and emits test predictions without parsing any label file or calculating test metrics. Its manifest references the exact freeze ID. A changed model or source manifest blocks prediction. This command was tested on synthetic fixtures; it has **not** been run on the project's real held-out data in M5.

Before M6 computes test metrics, it must separately verify `labels_test.jsonl` against the original M4 manifest, join by ticker/session and respect the frozen plan. Predictions alone cannot validate outcome data.

## Outputs

| File | Contents |
| --- | --- |
| protocol.json | Declared development/holdout protocol and candidate grids |
| model_<family>_<set>.json | Selected native-equivalent JSON state, preprocessing, parameters, fold receipts and interpretation |
| candidate_metrics.json | All candidate validation and inner-fold diagnostics |
| validation_predictions.jsonl | Selected model and baseline probabilities/decisions, with keys and no outcomes |
| inner_fold_predictions.jsonl | Selected candidates' expanding-fold out-of-fold predictions |
| validation_metrics.json / .csv | Development metrics for ten selected models and baseline variants |
| baselines.json / coin_variability.json | Frozen baseline parameters, definitions and repeated-seed diagnostics |
| development_labels.jsonl | Training/validation outcomes only, physically separate from predictions |
| holdout_keys.jsonl | Reserved test row keys, dates and cutoff metadata, without outcomes |
| training_audit.json | Train-only/fold/holdout/allowlist/budget checks and data-gate status |
| progress.json / summary.json | Run progress and completion summary |
| freeze.json | Completion receipt, artifact hashes, environment and immutable experiment identity |

## Verified real-price run

`artifacts/m5-development` contains ten selected LR/RF models from 40 candidate settings, each evaluated on the same 755 validation rows. Training uses 2,199 rows; 755 test rows remain reserved. The validation comparison table contains ten model entries and 24 baseline variants, including the 20 coin seeds.

All source-sentiment means/dispersion remain missing. Consequently these fits are an engineering demonstration on real vendor price data, not evidence for sentiment's incremental value. Some added columns are constant or numerical placeholders; a difference between fitted ablations cannot be attributed to news/Reddit signals that do not exist in the data. The unmet source-data gate is preserved in the protocol, audit, summary and freeze.

No actual holdout labels were parsed during the modeling run. No final test prediction/metric artifact was produced. Actual held-out evaluation, uncertainty and conclusions belong to M6, with source-data limitations carried forward.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p test_modeling.py -v
RUN_FINBERT_INTEGRATION=1 .venv/bin/python -m unittest discover -s tests -v
```

Tests cover whole-date folds, strict label-boundary purging, independent fold transforms, model/grid budgets, seeded baselines, undefined AUC, portable LR/RF fidelity, frozen-artifact tampering, deterministic replay, malformed/unreadable test labels, one-class development data and preserved development outputs after held-out feature changes. The label-free holdout predictor is exercised on synthetic inputs only. Core contract tests remain available without scikit-learn; estimator integration tests skip when the dependency is absent.

Verified locally: all 131 tests pass with actual FinBERT integration enabled, including 20 M5 checks. `pip check` reports no broken requirements. The real M5 freeze and all its output hashes verify, and no real holdout prediction file is present.
