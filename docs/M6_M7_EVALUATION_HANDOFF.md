# M6 and M7: final evaluation and handoff

M6 evaluates the original frozen M5 experiment without refitting. M7 produces the report, scientific figures, data/model cards and a local reproducible handoff. All seven software milestones are implemented. The **empirical sentiment-value study remains incomplete** because eligible news/Reddit history is absent.

## Frozen holdout workflow

```bash
.venv/bin/python -m sentiment_lab predict-holdout \
  --features artifacts/m4-model-input --experiment artifacts/m5-development \
  --out artifacts/m6-holdout-predictions

.venv/bin/python -m sentiment_lab evaluate \
  --features artifacts/m4-model-input --experiment artifacts/m5-development \
  --predictions artifacts/m6-holdout-predictions --out artifacts/m6-evaluation

.venv/bin/python -m sentiment_lab finalize \
  --features artifacts/m4-model-input --experiment artifacts/m5-development \
  --predictions artifacts/m6-holdout-predictions --evaluation artifacts/m6-evaluation \
  --out artifacts/m7-release
```

These commands use fresh output directories. The shown M6 artifacts already exist; choose new names for a replay. The M5 freeze is unchanged. Test predictions are made before outcomes are joined; evaluation verifies prediction/feature/freeze hashes and the exact held-out outcome-file hash before parsing labels. Every model and baseline must have the same complete row keys, partition and decision metadata. Binary labels must agree with finite target-return signs and be available at evaluation time.

## Statistical implementation

The frozen primary endpoint is combined LR minus matched price-only LR delta AUC. M6 implements 2,000 paired non-circular moving-block bootstrap replicates over five consecutive distinct test-session dates, with one- and ten-session sensitivity. Dates are resampled with replacement in contiguous blocks; blocks are concatenated and trimmed to the original number of dates. All ticker rows and model scores on a sampled date receive the same multiplicity.

A weighted rank-sum calculation groups equal scores exactly. This computes the same AUC as explicitly repeating bootstrap rows, while avoiding repeated sorting. F1 uses the same paired weights and fixed 0.5 threshold. Tests compare tied/weighted AUC with scikit-learn's implementation. Percentile 95% intervals use linear quantiles. Single-class AUC replicates are counted and excluded from AUC intervals; an entirely undefined AUC interval stays null. Insufficient dates for a configured block length produce an unavailable result, not a silently shortened block.

Primary and secondary comparisons remain fixed. Source/RF ablations, per-ticker/macro metrics, early/late test halves and text-present diagnostics are exploratory. No secondary winner replaces the primary comparison. The text-present slice is empty in this run. Twenty random-baseline seeds are summarized separately.

Intervals depend on approximate stationarity and how well block length captures serial dependence; clustering by date preserves within-date stock correlation. They do not establish causality, profits or future stability. A collapsed zero paired interval is possible when two models make identical predictions; it is not a powered demonstration of sentiment equivalence.

## Observed result and honest conclusion

The held-out period is **20 March–23 October 2026**: **755 rows across 151 sessions**. Price-only LR and combined LR both have AUC **0.489703** and F1 **0.676056**. The primary delta AUC and delta F1 are zero; the five-session paired 95% intervals are [0, 0].

This cannot answer whether sentiment helps: **zero test rows contain eligible sentiment**. The `engineering_demonstration_only` gate is preserved and `research_status` is `inconclusive_data_gate_unmet`. Real vendor prices, successful inference software, and a complete evaluation implementation do not substitute for aligned source data. The final report carries that limitation prominently.

The holdout has now been revealed. Any subsequent test-driven model/feature revision is exploratory and needs a fresh holdout for a new confirmatory claim. Do not retrain or change the original freeze in place.

## M6 artifacts

`artifacts/m6-evaluation` contains complete test metrics (JSON/CSV), paired uncertainty and sensitivity, per-ticker/macro and period diagnostics, empty-text-subset status, coin variability, paired primary records, conclusion, evaluation audit and hashed manifest. `artifacts/m6-holdout-predictions` contains all 25,670 predictions for ten selected models and 24 baseline variants.

## M7 handoff

`artifacts/m7-release` contains:

- `report.html`: standalone report with embedded scientific figures.
- `report.md`: editable report source.
- `figures/`: held-out AUC comparisons, paired intervals and ROC curves.
- `DATA_CARD.md`, `MODEL_CARD.md`, `REPRODUCE.md`.
- `handoff.zip`: source/configuration/tests, frozen processed inputs/models, predictions, evaluation and report.
- `reproduction_check.json`: clean-extracted-source verification.
- `manifest.json`: output hashes, freeze binding and explicit scope status.

The ZIP excludes virtual environments, cache directories, API credentials, raw social/news data and FinBERT weight files. It is a **local handoff**, not an external publication. Reproduction extracts a new source tree, removes inherited PYTHONPATH, runs label-free prediction and evaluation, and compares prediction bytes plus numerical metrics/bootstrap results. It uses the installed reference environment and packaged processed inputs; it does not claim a clean dependency installation or fresh raw-source acquisition.

The full reference environment is `requirements-final.lock`; classifier-only reruns can use `requirements-modeling-core.lock`. Matplotlib generates the standard scientific figures. No source-data purchase, cloud upload or private Sites deployment is performed.

## Local dashboard

The existing dashboard now has held-out/validation selection, a Final results view, report access and handoff download. It distinguishes software completion from the data-gated research conclusion. Regenerate its data with `python3 website/export_dashboard.py`; copy the current report and ZIP into `website/dist/` for local serving. Publishing remains subject to the previously requested explicit authorization; this milestone does not bypass that restriction.

## Remaining source-dependent work

Obtain suitable permitted historical news and approved Reddit data; preserve observed availability/content versions; manually review ticker/sentiment errors on actual sources; establish sufficient aligned coverage; freeze a new protocol and use a fresh holdout. These empirical requirements remain open in the final scope statement.
