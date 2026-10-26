# M3: FinBERT sentiment scoring

M3 implements real FinBERT inference, a persistent score cache, timestamp-safe ticker/session/source aggregation, truncation audits, and reproducible manifests. It consumes M2 datasets. Technical price indicators and stock-direction classifiers remain M4 and M5.

## Setup and pinned artifacts

The existing `.venv` is ready. To recreate the full acquisition + sentiment environment under Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-sentiment.lock
.venv/bin/python -m sentiment_lab prepare-finbert
```

The model is `ProsusAI/finbert`, pinned to commit `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`. `configs/sentiment.toml` defines batch size, maximum token length, seed and CPU thread count. A moving branch name such as `main` is rejected. The full tested environment is in `requirements-sentiment.lock`; top-level constraints are in `requirements-sentiment.in`.

The downloader fetches the exact model config, safetensors weights, vocabulary, tokenizer config, special token map and model README. It records a SHA-256 receipt in `data/cache/models/finbert/<revision>/model_manifest.json`. Every model load rehashes those files and rejects missing/modified artifacts. Weights are approximately 438 MB. Only safetensors weights are loaded; remote model code is disabled. Downloads can resume after interruption.

Use `prepare-finbert --offline` to verify an existing local copy. Scoring defaults to local files only, so inference does not unexpectedly download a model. `--download-model` is an explicit first-time download option for scoring/smoke commands. No hosted inference API or key is required.

The [official model card](https://huggingface.co/ProsusAI/finbert) describes positive, negative and neutral classes and the model's noncommercial usage terms. The pinned README is retained with the downloaded artifacts. Attribution: Dogu Araci, *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* (2019), [paper](https://arxiv.org/abs/1908.10063).

## Score an M2 dataset

```bash
.venv/bin/python -m sentiment_lab score-sentiment \
  --dataset artifacts/m2-verified \
  --out artifacts/my-sentiment-run
```

Before inference the command validates the project configuration against the acquisition manifest and checks the hashes of sessions, prices, text, calendar metadata and feasibility report. It rebuilds the existing M1/M2 alignment and chronological partitions. Delayed, duplicate, superseded, outside-window and disallowed historical-assumption text is excluded **before scoring**.

All tickers on a date retain the existing partition. The latest eligible source-origin version before the cutoff is used, and no newly observed edit can replace an older historical version. A model computation performed later is part of a retrospective experiment; it does not establish that a live system had completed inference at the historical decision instant.

If the dataset has no eligible text, the command writes missing sentiment features for every valid price-based row, produces an empty score file, and records `no_eligible_text`. It does not load the model, invent neutral scores, or upgrade the acquisition feasibility decision.

The current real-price dataset has 3,769 ticker/session rows and zero eligible text records. `artifacts/m3-price-only` records that condition. Once sufficient approved text is imported through M2, export a fresh M2 dataset and run the same command against it.

## Inference behavior

The reference backend uses CPU float32, eager attention, deterministic Torch algorithms, a fixed seed, two CPU threads and batches of eight. The model runs in evaluation mode under `torch.inference_mode()`; M3 never fine-tunes its weights. Backend/runtime versions and architecture are recorded with the inference settings.

Whitespace normalization preserves case, words, punctuation and financial numbers. Empty text and inputs longer than 100,000 characters fail validation. The tokenizer uses dynamic right padding and right truncation with a default maximum of 512 tokens, including special tokens. Original token count and a truncation flag are retained. Long text loses its tail; the audit counts that loss rather than hiding it.

The code reads `id2label` from the actual model config, validates that it maps exactly three indices to the expected classes, and maps logits accordingly. It does not assume indices alphabetically. Softmax outputs must be finite, within [0, 1], and sum to one within numerical tolerance. Invalid outputs are errors, with no silent renormalization or substitute lexicon scorer.

For every included record:

`polarity = P(positive) - P(negative)`

These are model sentiment outputs. They are not calibrated sentiment probabilities, stock-up probabilities, trading returns or evidence of incremental predictive value.

## Score cache and reproducibility

The cache is a separate SQLite database, normally `data/cache/sentiment.sqlite`. It stores normalized text hashes and validated scores, not raw text. Cache identity includes model/tokenizer file hashes, revision, inference implementation hash, preprocessing settings, batch size, seed, CPU configuration, Python/runtime versions and architecture.

Identical normalized text is inferred once even if mapped to several tickers or referenced by several records. A changed content version gets a different text hash. A model, tokenizer, runtime or scoring-configuration change invalidates cache reuse. Batches commit atomically; completed batches survive an interruption. Invalid cached probability data fails clearly instead of silently entering features.

Synthetic runs must use a separate cache. A database bound to synthetic data cannot be reused for a non-synthetic dataset. Use a fresh output directory on every run; outputs are never overwritten. Stable scored/daily feature files are deterministic on the tested runtime. Summaries differ between an initial run and a replay because cache-hit counts change, and manifest creation timestamps also change.

## Daily aggregates

For each eligible ticker/target-session row, news and Reddit have separate equal-weight summaries:

| Field suffix | Definition |
| --- | --- |
| count | Number of included deduplicated text records |
| missing | True when no text exists for that source/window |
| mean_positive / mean_negative / mean_neutral | Arithmetic means of text class probabilities |
| mean_polarity | Mean of positive minus negative probability |
| polarity_dispersion | Population standard deviation of record polarities |
| truncated_count | Number of included truncated texts |
| latest_available_at | Latest included text availability, for leakage auditing |

Columns are prefixed `newsapi_` or `reddit_`. One record has dispersion zero. An empty source window has count zero and missing true, with means, dispersion and latest availability set to JSON null / empty CSV cells. There is no forward fill, vote weighting, calibration, future-dependent weighting or imputation in M3.

## Outputs

| File | Purpose |
| --- | --- |
| daily_sentiment.jsonl / daily_sentiment.csv | Source-specific numeric summaries, missingness and row keys |
| scored_texts.jsonl | Record/source/version/time lineage, model probabilities, polarity, sentiment label and truncation |
| labels.jsonl | Stock outcomes stored separately, never part of sentiment features |
| review_sample.jsonl | Up to 25 included scored records with text for manual source/error review |
| summary.json | Eligible counts, exclusions, missingness, cache statistics and original data-gate status |
| manifest.json | Input/code/output hashes, pinned model artifacts, runtime settings and environment-lock hash |

`partition`, timestamps and source lineage are audit metadata. M4 must choose a numeric feature allowlist explicitly and join by ticker/session; it must not infer model inputs from every numeric column. Sentiment `sentiment_label` is distinct from the stock target `label_up`.

## Smoke test and aggregation demonstration

```bash
.venv/bin/python -m sentiment_lab sentiment-smoke --out artifacts/my-finbert-smoke

.venv/bin/python -m sentiment_lab score-sentiment \
  --dataset artifacts/m2-demo \
  --config sentiment_lab/fixtures/demo.toml \
  --cache data/cache/sentiment-demo.sqlite \
  --out artifacts/my-sentiment-demo
```

The smoke test runs the real model on six authored financial examples. Both inputs and outputs are explicitly marked as a synthetic software demonstration. The dataset demonstration scores four eligible synthetic records across 12 ticker/session rows. If `artifacts/m2-demo` is absent, first run `m2-demo --out artifacts/m2-demo`.

The real CPU smoke test matched five straightforward expected labels and labeled the sarcastic loss example neutral. That six-example agreement is not a benchmark accuracy estimate. See [the reviewed examples](M3_SMOKE_REVIEW.md). A second run reused six cached smoke scores and four cached demonstration scores, without new inference; the corresponding scored/aggregate data bytes matched.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
RUN_FINBERT_INTEGRATION=1 .venv/bin/python -m unittest discover -s tests -v
```

The opt-in integration test requires the prepared local model. It checks real CPU positive/negative outputs, probability sums, identical repeated batch outputs, and truncation of text longer than 512 tokens. The full local suite also covers M1/M2 regressions, model-label validation, artifact integrity, cache invalidation/resume/corruption, cutoff enforcement, aggregation arithmetic, missingness, input-manifest tampering and separate feature/label output. Without acquisition dependencies or prepared model, the corresponding integration checks skip; core unit tests remain available without Torch.

Verified locally: all 82 tests pass with real-model integration enabled; `pip check` reports no broken requirements. The standard-library environment also passes its available checks, with four dependency/model integration checks skipped. Cached real-model smoke scores and aggregation data files match their initial-run bytes.

## Retention and pending research work

`purge-text` now also clears the selected derived sentiment cache. The default is `data/cache/sentiment.sqlite`; specify `--sentiment-cache` when using another path. Source-origin IDs are not stored in the content-addressed score database, so the command conservatively clears all its cached scores. Remove affected old scored/review/aggregate exports and regenerate them as required by the source's actual retention conditions.

M3's implementation and authored-example review are complete. Reviewing real news/social samples and evaluating sentiment quality on a suitable independent labeled dataset remain data-dependent work. The existing news/Reddit historical-data gaps still prevent the original combined-source stock-prediction study. M3 does not remove those gaps or claim price-prediction performance.
