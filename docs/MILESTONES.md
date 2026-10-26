# 30-day milestones

Day numbers are relative to project kickoff. The development schedule does not restrict the historical observation period. Each milestone depends on the previous milestone's exit criteria. Do not spend on API plans automatically.

| Milestone | Days | Deliverable | Exit criteria | Status |
| --- | --- | --- | --- | --- |
| M1: Foundation and research specification | 1–3 | PRD, protocol decisions, Python package, config, normalized contracts, session windows, chronological split, synthetic CLI demo, tests | Offline run succeeds; temporal boundary tests pass; hashes and exclusions saved; labels separate from inputs | Implemented |
| M2: Acquisition and feasibility | 4–9 | yfinance/news adapters; approved social adapter or documented gap; complete exchange calendar; cache; coverage and provenance report | Request quotas/retries/checkpoints tested; availability evidence recorded; duplicate/version audit; data feasibility decision documented | Implemented; text data gate unmet |
| M3: Financial sentiment | 10–14 | Pinned FinBERT batch scorer, cached text scores, ticker/source/window aggregates | Correct model label mapping and probability sums; reproducible cache; reviewed text sample; CPU smoke run; no fabricated sentiment | Implemented; authored smoke reviewed, real-source review awaits data |
| M4: Features and leakage audit | 15–18 | RSI, momentum, volatility, missingness features, price-only/combined matrices, frozen feature allowlist | Trailing windows per ticker; past-only transformations; split handling; feature/label availability audit; matched evaluation rows | Implemented; original sentiment data gate remains unmet |
| M5: Models and baselines | 19–24 | LR and RF pipelines, naive baselines, source ablations, frozen experiment protocol | Shared chronological boundaries; preprocessing fits train only; equal tuning budgets; predictions saved; final test untouched until freeze | Implemented; real test reserved for M6, sentiment study remains data-gated |
| M6: Evaluation and uncertainty | 25–28 | F1/AUC tables, paired delta intervals, per-ticker diagnostics, sensitivity results | Date-block bootstrap; primary/secondary comparisons distinguished; coverage and one-class cases reported; conclusions follow evidence | Implemented; frozen holdout evaluated, sentiment hypothesis inconclusive |
| M7: Report and demonstration | 29–30 | Reproducibility guide, data/model cards, final report, review demo | Clean rerun produces results from permitted inputs; sample limitations stated; original scope mapped to delivered work | Implemented for available data; original empirical sentiment study remains incomplete |

## M1 implemented acceptance checklist

- [x] Establish project scope and hypothesis without promising predictive success.
- [x] Record five starting tickers and pre-open cutoff convention.
- [x] Define publication, availability and retrieval timestamps separately.
- [x] Validate TOML settings, source records, OHLCV and supplied exchange sessions.
- [x] Exclude delayed text, exact-cutoff text and unapproved historical availability assumptions by default.
- [x] Deduplicate normalized identical text, retaining earliest availability.
- [x] Align holiday/weekend/early-close windows and handle supplied DST timestamps.
- [x] Construct close-to-close labels, retain missing-text rows and exclude split target sessions.
- [x] Split whole target dates and purge labels unavailable at subsequent partition boundaries.
- [x] Separate input windows from labels on disk.
- [x] Write deterministic data outputs, SHA-256 manifests and validation counts.
- [x] Run a synthetic end-to-end CLI demo and automated tests without network or credentials.
- [x] Provide CI configuration; remote execution has not been claimed.

## M2 work breakdown

1. Establish which historical news/social datasets are actually available, their date range and permission for the intended research. If Reddit research access is unavailable, document the scope gap and select a declared reduced study or engineering demonstration.
2. Generate a complete, version-pinned XNYS calendar for the study period, including one preceding session. Verify holidays, early closes and timezone handling.
3. Add price acquisition with explicit adjustment flags and corporate-action metadata. Preserve download metadata and explain historical availability assumptions.
4. Add NewsAPI collection/import with bounded requests, pagination, retry handling, incremental checkpoints, latency and provenance fields.
5. Implement the approved Reddit research interface if access exists. PRAW is an option only when appropriate for that access and approved use; it is not the default assumption.
6. Add stable origin IDs, URL/content deduplication and content-version tracking. Audit ticker matching on a manually inspected sample.
7. Export per-source/per-ticker/date coverage, exclusion counts and data cards. Freeze the available evaluation period before modeling.
8. Record the feasibility gate decision. If minimum history cannot be obtained, continue engineering with clearly labeled fixtures and report an inconclusive research outcome.

## M1 intentionally stops here

M1 does not fetch real data, download FinBERT, compute technical features, fit classifiers or report predictive performance. Its tiny synthetic fixture validates the infrastructure. A successful M1 run is not evidence for the sentiment hypothesis.

## M2 implementation and current source status

- [x] Generate a complete, version-pinned XNYS calendar with the preceding price session; verify imported calendars against it.
- [x] Download historical vendor OHLCV with explicit adjustment flags, splits, dividends, retrieval/snapshot IDs and documented availability assumptions.
- [x] Implement NewsAPI queries, pagination, bounded transient retries, persistent request accounting and page checkpoints.
- [x] Implement licensed-news / approved-Reddit normalized export imports with provenance hashes and record validation.
- [x] Record the Reddit live research-interface gap; do not substitute an undocumented generic scraper.
- [x] Add SQLite raw response snapshots, incremental resume, exact content versions, canonical URLs, daily content/syndication deduplication and first-observation lineage.
- [x] Use the latest content version available before each forecast cutoff.
- [x] Export ticker/date/source coverage, duplicate lineage, manual review samples, data cards and run manifests.
- [x] Record an engineering-only feasibility decision when text history is absent.
- [x] Exercise the real five-ticker price adapter and verify offline cache reuse.
- [x] Pass 56 local tests, including all M1 regressions and three installed-calendar integration checks.
- [ ] Obtain NewsAPI credentials or an appropriate licensed historical export for the study.
- [ ] Obtain approved Reddit research data and its actual interface/export.
- [ ] Review real-text ticker matches and pass the combined-source historical coverage gate.

Live evidence: 3,775 price bars over 755 price sessions, 753 shared usable target sessions, 151 shared test sessions and zero text records. All five price downloads succeeded. News transport was tested with mocked responses because no key is set; the social importer was exercised with fixtures. See [M2 acquisition guide](M2_ACQUISITION.md). No sentiment hypothesis or predictive claim has been evaluated.

## M3 implementation and verification

- [x] Pin model/tokenizer artifacts to an exact FinBERT repository commit and verify local SHA-256 hashes.
- [x] Run real CPU inference with safetensors weights, explicit label mapping, evaluation mode and deterministic settings.
- [x] Validate all three probabilities and derive polarity without a substitute scoring heuristic.
- [x] Record original token counts, truncation settings and affected records.
- [x] Persist scores by normalized content and model/runtime identity; resume completed batches and reject corrupted entries.
- [x] Preserve latest-before-cutoff content versions and existing shared chronological partitions.
- [x] Aggregate equal-weight probability means, polarity, population dispersion, counts and missingness separately by source/ticker/window.
- [x] Preserve absent text as missing values, with no invented neutral probabilities.
- [x] Keep future stock labels separate from daily sentiment features.
- [x] Review six authored financial examples using actual FinBERT outputs and document the sarcasm failure.
- [x] Exercise the real-model synthetic aggregation demo and cached replay.
- [x] Pass all 82 local tests with real-model integration enabled, including repeatability, probability sums and long-input truncation.
- [x] Export missing sentiment for all 3,769 current real-price ticker/session rows because there is no eligible source text.
- [ ] Review real news/social sentiment errors once approved text data is available.

See [M3 sentiment guide](M3_SENTIMENT.md) for commands, outputs, cache identity and the model/data limitations. The statistical study remains data-gated; no price-prediction result is claimed.

## M4 implementation and verification

- [x] Implement fixed 1-/5-session returns, 10-session moving-average momentum, sample volatility and Wilder 5-session RSI.
- [x] Use only consecutive prior-session prices known before each forecast cutoff; isolate tickers and record full recursive RSI lineage.
- [x] Reset at stock splits, missing bars and unavailable-history barriers without a second vendor split adjustment.
- [x] Drop insufficient technical history consistently across all ablations; preserve M3 chronological partitions and original boundary purges.
- [x] Reconcile M3 per-record scores, text versions, aggregates and labels with hash-verified acquisition data.
- [x] Export identical-key price-only, price + news, price + Reddit, sentiment-only and combined matrices.
- [x] Freeze feature names, formulas, ordering, missing-value policies and forbidden model inputs.
- [x] Fit median/scaling statistics on training rows only, record all-missing columns, and provide a reusable fold-specific preprocessor.
- [x] Keep raw missing sentiment and separate labels; report post-warmup source feasibility without upgrading an unmet data gate.
- [x] Export temporal lineage, excluded rows, a leakage audit, schema, preprocessing receipt and reproducibility manifest.
- [x] Exercise the real-price pipeline: 3,709 retained rows, split 2,199 / 755 / 755, with 60 warmup/reset exclusions.
- [x] Test arithmetic, future-data invariance, split/gap behavior, train-only fitting, semantic tampering checks and deterministic replay.
- [x] Pass all 111 local tests with real FinBERT integration enabled, including 29 M4 checks.

See [M4 features guide](M4_FEATURES.md). M4 itself fits no directional model. M5 refits preprocessing inside each expanding chronological fold rather than reusing the outer-training transform.

## M5 implementation and verification

- [x] Train LR and RF on all five frozen matched feature sets with bounded, declared grids.
- [x] Preserve outer partitions and group expanding development folds by whole target-session dates.
- [x] Purge unresolved fold labels and fit fresh preprocessing inside each fold and outer training block.
- [x] Select settings by validation AUC with fixed tie and single-class rules, without changing the primary LR comparison.
- [x] Implement training-majority, always-up, literal no-change, momentum and 20 seeded coin baselines.
- [x] Save candidate/fold diagnostics and selected-model/baseline validation predictions on identical samples.
- [x] Save LR coefficient explanations and validation-only RF permutation importance with correlation caveats.
- [x] Export native-equivalent portable JSON model states and freeze the models/protocol/artifact hashes.
- [x] Separate M4 partition-label files so training never parses actual holdout labels.
- [x] Implement a separate post-freeze, label-free holdout predictor and test it on synthetic fixtures.
- [x] Exercise real-price training: 10 selected models, 40 candidates, 755 validation rows per model and 755 held-out rows still reserved.
- [x] Preserve the unmet news/Reddit data gate and distinguish development diagnostics from research findings.
- [x] Pass all 131 local tests with real FinBERT integration enabled, including 20 M5 checks.

See [M5 models guide](M5_MODELS.md). Actual final-test predictions, paired uncertainty estimates and result interpretation remain M6 work. The M5 run has not been used to claim that sentiment improves stock-direction prediction.

## M6 / M7 completion

- [x] Verify and evaluate all 755 frozen test rows across 34 model/baseline variants without refitting.
- [x] Implement 2,000 paired date-block bootstrap replicates and one-/ten-session sensitivity.
- [x] Handle tied scores, undefined AUC, invalid bootstrap replicates and insufficient block histories honestly.
- [x] Save per-ticker/macro, period, random-seed and text-present diagnostics as exploratory views.
- [x] Preserve the data gate and state that missing sentiment prevents answering the original hypothesis.
- [x] Generate final Markdown/HTML report, scientific figures, data/model cards and reproduction guide.
- [x] Package local source and frozen processed artifacts; verify a clean extracted prediction/evaluation rerun.
- [x] Update the local dashboard with final results and report/handoff access.
- [ ] Collect and audit sufficient actual news/Reddit history and run a new confirmatory sentiment study on a fresh holdout.

See [M6/M7 handoff guide](M6_M7_EVALUATION_HANDOFF.md). Software delivery is complete for the available inputs. The original combined-source empirical requirement is explicitly unfulfilled, rather than replaced by synthetic evidence.
