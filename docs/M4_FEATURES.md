# M4: features and leakage audit

M4 builds fixed technical indicators and matched price/sentiment feature sets from M2 acquisition data and M3 scored sentiment. It verifies source bindings, reconciles text-level scores with their aggregates, checks time availability, and fits preprocessing statistics on training rows only. It does not fit a stock-direction classifier or report prediction performance.

## Run

From the project root, using the existing environment:

```bash
.venv/bin/python -m sentiment_lab build-features \
  --dataset artifacts/m2-verified \
  --sentiment artifacts/m3-verified-price-only \
  --out artifacts/my-feature-run
```

Project settings default to `configs/project.toml`. Frozen feature policies are in `configs/features.toml`. M4 adds no runtime dependency: numerical transforms use Python's standard library, and schedule verification uses the existing pinned acquisition environment. Use a fresh output folder on each run.

The acquisition dataset and M3 output must refer to the same input bytes, configuration, synthetic status and source-data gate. The command checks M2 file hashes and M3's input/output hashes. It regenerates M1/M2 windows, matches every included M3 score to its exact eligible text version, validates probabilities and polarity, recomputes source aggregates, and compares outcome files by row keys. An internally inconsistent but rehashed M3 output still fails reconciliation.

The full XNYS calendar is verified against the library version used during acquisition. Incoming calendar order is normalized before feature generation. A mismatch in dates, holidays, session instants or library version requires consistent upstream data/environment; the command does not approximate a weekday calendar.

## Frozen technical definitions

Let `s` denote the exchange session immediately preceding the target session, and `C` denote its vendor closing price. All feature inputs must have `available_at < decision_at`, and no target-session price enters an indicator.

| Feature | Definition |
| --- | --- |
| return_1d | `C[s] / C[s-1] - 1` |
| return_5d | `C[s] / C[s-5] - 1` |
| momentum_10d | `C[s] / mean(C[s-9], ..., C[s]) - 1` |
| volatility_10d | Sample standard deviation, ddof 1, of the ten most recent simple daily returns; not annualized |
| rsi_5d | Wilder RSI with period 5, an initial SMA of five gains/losses, then recursive smoothing with weight 1/5 |

RSI returns 50 when smoothed gains and losses are both zero, 100 with positive gains and zero losses, and 0 with zero gains and positive losses. Its recursive state depends on every close since the latest history reset, not just the most recent five changes. The window lineage records that full dependency range. See [Fidelity's RSI definition](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI); this project explicitly fixes its five-period and flat-series conventions.

All five indicators require at least eleven consecutive known closes, because ten returns require eleven closes. Target rows without that history are excluded with a reason. The formulas and column names are versioned together; schema v1 rejects policy changes that would silently alter their meaning.

## Splits, missing bars and availability

History is isolated by ticker and actual exchange-session position. A missing bar breaks the chain; the next known close starts a fresh history. There is no forward/backward fill or rolling across a missing session. A historical bar not known before the current decision also forms a barrier. If it later becomes known, a later forecast may legitimately use it, while earlier feature values remain unchanged.

Target split sessions were already excluded by M1/M2. For technical features, M4 starts a new history at the most recent prior split session and retains that day's close as the new anchor. It excludes the next ten target rows for that ticker until eleven closes accumulate. This works for both an observed raw split discontinuity and already vendor-adjusted history without applying a second split adjustment.

Wilder RSI also restarts at each barrier. This conservative policy sacrifices some observations rather than mixing price scales. It cannot independently verify historical source revisions or identify an unrecorded corporate action. Yahoo history and close-availability assumptions remain those disclosed by M2.

Features for a later validation/test session may use prices from earlier validation/test sessions once those prices are known. That is normal chronological forecasting. It does not permit training transformations or model parameters on validation/test data.

## Matched feature sets

| Set | Features | Intended later comparison |
| --- | --- | --- |
| price_only | 5 | Technical baseline |
| price_news | 12 | Price plus news ablation |
| price_reddit | 12 | Price plus social ablation |
| sentiment_only | 14 | Sentiment-only ablation |
| combined | 19 | Price plus both sentiment sources |

Each sentiment source contributes count, missingness, mean positive/negative/neutral probabilities, mean polarity and polarity dispersion. No source-presence filter changes the sample. All five sets have the same ticker/session ordering, outcome keys and chronological partitions. There is no selective removal of missing-text rows.

Raw missing sentiment means and dispersion stay null; counts remain zero and missingness remains true. The current real dataset has no eligible source text, so a 19-column combined schema does not imply actual sentiment information exists.

`feature_schema.json` is the model-column allowlist. Keys and metadata such as ticker, target date, partition and decision time are not model inputs. Target direction, target return, label availability and future close fields are forbidden. Model code must consume a named allowlist rather than automatically select numeric columns.

## Train-only preprocessing

M4 emits raw matrices and, when every partition has retained rows, scaled matrices. `TrainPreprocessor.fit` accepts only training rows strictly before the original validation boundary and requires the exact explicit allowlist. It never sees outcome fields.

For each feature, it computes the observed training median, fills missing values with that median, and computes the mean and population standard deviation of the filled training values. Transform uses those frozen statistics on train/validation/test. A constant column uses scale 1. A wholly missing training column uses an explicit fill value 0 and scale 1, with an `all_missing_train` flag and observed count zero.

That all-missing fill is a numerical placeholder in the transformed model matrix, not a neutral sentiment observation. Raw values and missingness remain available, and the summary lists every affected column. No performance claim can follow from missing sentiment history.

The saved statistics fit the outer training block only. M5 must refit a new `TrainPreprocessor` on each expanding training fold; reusing outer-training medians/scales inside earlier folds would leak later training observations. The fit receipt records input-row hashes, dates, row counts, boundary and feature ordering. This follows the train-only fitting principle in [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Warmup filtering preserves the original M3 date partitions and boundary-label purges. It never recomputes 60/20/20 percentages on the smaller dataset. Labels must still be available before the next **original** partition boundary. If filtering empties a partition, M4 exports the audited raw rows but records `insufficient_history_or_partitions` and does not fit preprocessing. A zero-row file is not training-ready.

## Outputs

| File | Contents |
| --- | --- |
| price_only / price_news / price_reddit / sentiment_only / combined, each CSV and JSONL | Raw matched feature matrices with four key/metadata columns |
| scaled_*.jsonl | Corresponding finite numerical matrices using train-only statistics, when fitting is possible |
| labels.jsonl and labels_train/validation/test.jsonl | Separate binary outcomes, with partition-specific files so M5 need not parse test labels |
| feature_schema.json | Frozen column ordering, formulas, policies, forbidden model inputs and dependency descriptions |
| preprocessing.json | Per-column fill/scale statistics and training fit receipt, or explicit not-fitted status |
| lineage.jsonl | Input history start/end, bar count, maximum availability, split/gap barrier and latest source-text availability |
| excluded_rows.jsonl | Every technical warmup/history exclusion with original keys and reason |
| leakage_audit.json | Reconciliation, cutoff, calendar, partition, matched-sample and boundary-label checks |
| feature_feasibility.json | Source/history coverage after feature filtering, with the inherited data gate |
| summary.json | Row counts, feature counts, partition sizes, missing-source coverage and status |
| manifest.json | Source-manifest/config/code/output hashes, model provenance, environment-lock hash and fingerprint |

Price lineage covers the full recursive RSI history even when other indicators use shorter trailing windows. Maximum price and source-text availability must precede the decision. The audit checks declared temporal validity and internal consistency; it does not independently certify provider point-in-time truth.

Feature feasibility is checked again after filtering. The gate may downgrade when warmup or split exclusions reduce coverage, and can never upgrade an unmet upstream study gate. Synthetic inputs always remain engineering demonstrations.

## Verified real-price run

The M4 run in `artifacts/m4-verified` starts from 3,769 M3 ticker/session rows and retains **3,709**:

- Training: 2,199 rows.
- Validation: 755 rows.
- Test: 755 rows.

Fifty rows are excluded for the initial eleven-close warmup across five tickers, and ten more follow the recorded NVDA split. That leaves 733 shared usable sessions and 151 shared test sessions. All five feature sets use identical retained keys. Ten training columns containing sentiment means/dispersion are wholly missing and are explicitly disclosed.

The source-data gate remains `engineering_demonstration_only`: no actual news/Reddit text history is present. Price features and software audits are ready; the original sentiment-value research question is still unanswered.

The existing seven-session M2/M3 synthetic demo is intentionally too short for the eleven-close technical window. Running M4 on it exports zero retained rows with an explicit insufficient-history status. Integration tests use a longer synthetic dataset to exercise nonmissing sentiment, split resets, all five matched sets and preprocessing without making a research claim.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p test_features.py -v
RUN_FINBERT_INTEGRATION=1 .venv/bin/python -m unittest discover -s tests -v
```

M4 checks include known indicator arithmetic, Wilder seed/recursion/flat behavior, sample-volatility convention, future-data prefix invariance, ticker isolation, missing/delayed bar barriers, raw/already-adjusted split resets, complete-calendar normalization, training-only statistics, independent fold refitting, null preservation, matched samples, deterministic replay, input tampering and semantic reconciliation after a changed file is rehashed.

Verified locally: all 111 tests pass with real FinBERT integration enabled, including 29 M4 tests. The real-price feature outputs pass the leakage audit and preserve matching row keys across all five feature sets.

No new source collection, hosted service, model download, classifier fitting or test-performance calculation is required for M4.
