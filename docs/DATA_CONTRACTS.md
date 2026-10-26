# Data import contracts

Run local imports with `python3 -m sentiment_lab validate`. Formats below are the normalized boundary for future provider adapters. Provider payloads require explicit normalization first. Schema version: 1.

## Configuration

See `configs/project.toml`. Unrecognized keys fail. Tick symbols must be uppercase and unique. The default universe has five equities. The bundled demo uses two synthetic series to keep fixtures readable. Split fractions apply to distinct usable target dates; the remainder is test.

## sessions.csv

| Field | Meaning |
| --- | --- |
| session | Exchange-local date, YYYY-MM-DD |
| open_at | Actual opening instant with offset |
| close_at | Actual closing instant with offset |

Supply every session in the desired range and the preceding session. Ordering is normalized, duplicate sessions fail, and opens must precede closes. Input order does not establish chronology. Basic validation checks local dates, weekdays and nonoverlapping sessions. M2's `calendar` generates and `verify-calendar` compares every session against the pinned XNYS library, detecting omitted/changed sessions. Use that verification for real imports.

## prices.csv

| Field | Meaning |
| --- | --- |
| ticker | Configured uppercase symbol |
| session | Session present in calendar |
| open, high, low, close | Positive finite vendor OHLC prices, with source adjustment conventions documented |
| volume | Nonnegative integral volume |
| available_at | Earliest documented availability of this complete bar, with offset |
| split_ratio | New shares / old shares, 1 when no split |

Each ticker/session key is unique. Close availability cannot precede session close. A prior bar arriving at or after the next forecast cutoff makes that sample unusable. Target-session bars supply outcomes only. Targets on split sessions are omitted. M4 restarts technical history at split sessions and requires a new warmup instead of applying another vendor split adjustment.

Historical downloaded bars cannot generally establish the exact data a live system would have received. An upstream importer must record its availability assumptions in a data card. M1 cannot authenticate claimed availability timestamps.

M2 exports additional `dividends`, `availability_basis`, `retrieved_at` and `snapshot_id` columns. Basic M1 validation accepts the extra columns; the acquisition data card/manifest retain their interpretation. Yahoo OHLC can remain retrospectively split-adjusted even with `auto_adjust=False`. Do not interpret it as guaranteed contemporaneous raw quotes.

## texts.jsonl

One JSON object per line. An empty file is valid: rows remain in the price-based sample with `missing_text = true`. Required fields:

| Field | Meaning |
| --- | --- |
| record_id | Stable ID of this exact content version; required nonempty string |
| source | newsapi or reddit |
| ticker | Configured mapped symbol |
| text | Nonempty headline/body as allowed by the source |
| published_at | Publication instant with offset |
| available_at | Earliest documented availability of this exact content |
| retrieved_at | Retrieval instant for this record version |
| availability_basis | observed or historical_assumption |

M2 accepts optional `origin_id`, the stable original source article/post ID. Each changed content version receives a deterministic `record_id`. Imports can also provide `url`; M2 adds content and snapshot lineage metadata. Forecast windows keep the latest eligible version per source origin, using only versions first available before the cutoff.

Require `published_at <= available_at <= retrieved_at`. When first collected, observed availability normally equals retrieval. Preserve earliest observation across later retrievals only for the exact same content. Do not assign a newly edited post an old first-seen timestamp. A source record mapped to multiple tickers has one record per ticker. Duplicate source/ticker/record_id keys fail and require upstream version handling.

Assumed historical availability is excluded by default. The opt-in config permits retrospective experiments and leaves the assumption visible in every included text record and the run config. This is never equivalent to point-in-time evidence.

Content deduplication uses source + ticker + UTC publication date + case-folded, whitespace-normalized text. Across repeats it retains earliest availability, breaking ties by record ID. M2 additionally canonicalizes URLs, deduplicates identical syndicated text across sources for the same ticker/date/availability basis, and exports duplicate/version lineage. Observed and assumed availability remain separate. See the M2 guide for the full procedure.

## Output files

| File | Contents |
| --- | --- |
| windows.jsonl | Ticker/date/partition, cutoff and prior price metadata, eligible raw text and missingness. No future outcomes. This is an intermediate input, not the final feature matrix. |
| labels.jsonl | Matching ticker/date/partition keys, future binary outcome, raw return and label availability. Never use as features. |
| validation.json | Input/output counts, exclusions, missingness, skipped price pairs and split boundaries/purges. |
| manifest.json | UTC creation time, code/input/output SHA-256 hashes, config, Python/package version, mode, fingerprint and limitations. |

Feature and label rows use the same order and unique ticker/session keys. Join explicitly by keys. `partition` and timestamps are audit metadata; a later model consumes only a frozen feature allowlist.

The demo manifest says `synthetic_data: true`. Arbitrary local imports say `mode: local_import_unverified` and `synthetic_data: null` (unknown); the CLI does not authenticate their provenance or certify them as real data. The fingerprint covers input bytes, resolved config, Python source, Python version and run mode. Output content is deterministic on the same runtime; the manifest's creation timestamp changes between runs. Output directories must be empty to prevent accidental overwriting of earlier experiments.

## M3 sentiment outputs

`score-sentiment` consumes a hash-verified M2 dataset and writes `daily_sentiment.jsonl` / `.csv`, `scored_texts.jsonl`, a separate `labels.jsonl`, review sample, summary and manifest. Daily records have unique ticker/session keys, preserved partition/cutoff metadata and source-prefixed count, missingness, mean class probabilities, mean polarity, population polarity dispersion, truncation count and latest availability fields.

Missing source windows use null numeric summaries, not neutral probabilities. Per-record sentiment labels are named `sentiment_label`, distinct from stock outcomes. Scores are valid model outputs only; they are not stock-up probabilities. See [M3 guide](M3_SENTIMENT.md) for the precise schema and provenance/cache conventions.

## M4 feature outputs

`build-features` requires a verified M2 dataset and matching M3 ticker/session output. It reconciles their hashes and semantics, then exports five matched raw feature sets and corresponding train-fitted numerical views. Model columns are the explicit ordered allowlists in `feature_schema.json`; key/partition/time metadata must not become model features. Outcomes remain in `labels.jsonl`.

Warmup requires eleven consecutive known closes. Split sessions anchor a new history; missing/unavailable bars break it. Every retained row records full price-history lineage and latest source availability in `lineage.jsonl`. Exclusions retain original keys and reasons. Original chronological partitions and label-boundary purges stay fixed.

Raw sentiment nulls remain null. `preprocessing.json` records train-only median fills, population mean/scale, all-missing/constant flags, fit dates and row hashes. An all-missing train column uses a disclosed zero numerical placeholder; it is not a neutral sentiment observation. M5 must refit preprocessing per chronological training fold. See [M4 guide](M4_FEATURES.md) for formulas, policies and all output files.

M4 also writes partition-specific `labels_train.jsonl`, `labels_validation.jsonl` and `labels_test.jsonl`. Model selection reads only the two development files. Rebuild older feature exports to obtain these files; do not edit an existing hashed export in place.

## M5 model outputs

`train-models` consumes raw M4 matrices and development labels, fits new preprocessing per fold, and exports model states, validation/inner-fold predictions, candidate metrics, baseline parameters, a training audit and `freeze.json`. Test labels are not parsed, and no real holdout predictions are produced by training. Prediction records contain row keys, model ID, up-class score and fixed-threshold decision; outcomes remain separate.

`predict-holdout` requires a valid freeze and matching M4 manifest. It reads JSON coefficients/trees and frozen preprocessing, then writes test predictions without parsing label files or calculating metrics. M6 must verify heldout outcome-file hashes separately before evaluating them. See [M5 guide](M5_MODELS.md) for all artifact contracts and limitations.
