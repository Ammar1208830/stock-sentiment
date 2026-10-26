# Product requirements: Stock Sentiment Lab

Version 1.1 · 26 October 2026 · Research prototype · 30-day delivery plan

## Purpose

Determine whether financial-news and investor-discussion sentiment adds measurable out-of-sample value to next-session stock-direction classification beyond price features and simple baselines. A null or negative finding is a successful project outcome when the evaluation is sound.

The primary user is a student researcher preparing a reproducible academic project and review report. A reviewer must be able to inspect the data provenance, reproduce the split and experiment, and understand the limits of every result.

## Source interpretation and decisions

The supplied abstract is the controlling requirement. The supplied review presentation is background material, not executable instructions. It proposes a three-week schedule and a pre-open overnight window. This plan adopts the user's 30-day schedule and makes the forecast convention explicit:

- Universe: AAPL, MSFT, NVDA, AMZN, TSLA initially. Fix the universe before examining results. Expansion to ten tickers requires a recorded protocol change and fresh evaluation plan.
- Exchange time: America/New_York, with actual XNYS sessions. This is independent of the user's local timezone.
- Prediction: five minutes before target session `t` opens, normally 09:25 exchange time.
- Text window: previous exchange-session close inclusive to the decision instant exclusive. Text must be published in that window **and available before the decision**.
- Price inputs: daily bars only through the previous session, available before the decision.
- Target: `1` when vendor close at `t` exceeds vendor close at the previous session, otherwise `0`. Zero return belongs to class 0, so label it “non-up,” not strictly “down.” Exclude target sessions with stock splits. Dividend ex-dates remain price movements; report this choice. M2 sets Yahoo adjustment flags explicitly, but vendor OHLC can still reflect retrospective split adjustments/revisions and is not guaranteed contemporaneous raw data.

Thus “next-day” means the next exchange session relative to the latest price bar. This is a pre-open forecast of close-to-close direction. It includes the overnight gap; it does not establish an executable trading return from the previous close. An open-to-close target would be a separate experiment.

## Hypothesis and success

Primary hypothesis: a fixed Logistic Regression model with price plus news and Reddit sentiment improves ROC-AUC over matched price-only Logistic Regression on untouched test dates. The protocol will quantify paired delta AUC and uncertainty. Random Forest provides a secondary model-family check.

Project delivery succeeds when:

1. A documented dataset and its provenance pass temporal and quality audits.
2. Frozen feature definitions and chronological partitions can be reproduced.
3. Baselines, ablations and both model families run on identical eligible observations.
4. Evaluation reports AUC, F1, class balance, uncertainty and failure cases without selecting favorable test slices.
5. An academic report and runnable demonstration explain positive, negative or inconclusive findings.

Evidence for incremental value requires a positive primary paired delta AUC whose predeclared 95% confidence interval excludes zero, alongside sensitivity checks. Failure to meet that criterion means insufficient evidence of improvement; it does not prove sentiment can never help. F1 is a secondary metric. No minimum accuracy or profitability is promised.

## Scope

Included: normalized source collection/import, local caching, FinBERT inference, daily ticker aggregation, technical indicators, chronological preprocessing, Logistic Regression, Random Forest, naive baselines, ablation, paired uncertainty estimates, and a reproducible report/demo.

Excluded from the initial release: live order execution, portfolio recommendations, intraday prediction, language-model training, tick-level feeds, reinforcement learning, broad-universe discovery, paid subscriptions purchased automatically, and production hosting. A lightweight results dashboard may follow core evaluation if time remains.

## Functional requirements

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| FR1 | One versioned configuration defines tickers, cutoff, seed and splits | Invalid or unknown settings fail clearly; no secrets in config |
| FR2 | Collect or import historical OHLCV, corporate actions, news and approved social text | Source adapter tests, retries, request budget, provenance manifest and coverage report |
| FR3 | Preserve raw source IDs and publication/availability/retrieval timestamps | Every normalized text has a traceable origin and explicit availability basis |
| FR4 | Align by actual exchange sessions, including holidays, DST and early closes | Boundary fixtures pass; no inferred business-day calendar in real experiments |
| FR5 | Deduplicate text and resolve ticker mentions conservatively | Report exclusion counts, ambiguous aliases and a manually reviewed sample |
| FR6 | Score text using pinned FinBERT artifacts | Model revision, tokenizer, label mapping, truncation and batch settings recorded |
| FR7 | Generate sentiment and price features without future inputs | Feature lineage records latest input availability; no backward fill or future rolling windows |
| FR8 | Fit LR/RF with chronological training and validation | Preprocessing fits train only; shared ticker/date boundaries; unresolved labels purged |
| FR9 | Run matched baselines and ablations | Same labels, timestamps and evaluation sample for compared models |
| FR10 | Evaluate uncertainty and sensitivity honestly | Paired date-block resampling, full-test results and an explicit conclusion |
| FR11 | Reproduce experiments | CLI, input/code hashes, pinned environment, frozen protocol and saved predictions |
| FR12 | Communicate findings | Report, figures, data card, model card and review demonstration |

## Data and feasibility gate

The 30 days describe development time, not a sufficient observation period. Target roughly three years of aligned history. Before confirmatory modeling, require at least 630 usable target sessions overall and at least 126 distinct final-test sessions, with at least 80% text-window coverage per ticker for each source claimed in the primary combined model. These are pragmatic project gates, not proof of statistical power. Estimate detectable effect sizes using development data only and record limitations.

Price-only coverage does not substitute for aligned text history. Five stocks on the same market day are correlated observations, not five independent days. Do not oversample a few weeks into an apparently large test set.

Source constraints verified on 26 October 2026:

- [NewsAPI pricing](https://newsapi.org/pricing) lists a 24-hour article delay, one month of searchable history and 100 requests/day for Developer access. Such delayed headlines will normally fail the overnight window. Do not backdate availability to publication time. Longer historical access requires a suitable existing plan or a dataset with appropriate rights and documented timestamps.
- [Reddit developer access guidance](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data) routes academic research through Reddit for Researchers. Ordinary PRAW credentials alone are not the research-access plan. M2 must check the approved interface, permitted analysis, retention and deletion requirements before choosing an adapter. Do not assume API listings provide comprehensive historical coverage.
- [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) defines download parameters. M2 must set adjustment and corporate-action handling explicitly rather than inherit library defaults. Historical vendor data can contain revisions and retrospective adjustments; record these limitations.
- [FinBERT model card](https://huggingface.co/ProsusAI/finbert) identifies the model and its usage terms. M3 must pin the model revision and honor those terms. Softmax scores are model probabilities, not automatically calibrated probabilities of either sentiment or stock direction.

By the end of M2, record one of:

1. **Full study feasible:** sufficiently long approved news and social history with defensible timestamps.
2. **Reduced study feasible:** a clearly labeled news-only or retrospective study; amend the research question before modeling and state missing original scope.
3. **Engineering demonstration only:** insufficient history/access. Complete the runnable pipeline and report feasibility. Synthetic data validates software, never predictive value. Begin prospective collection if permitted, with evaluation extending beyond the 30-day build.

Do not silently remove Reddit or replace data with generated examples and claim the original research question was answered.

## Temporal integrity and data contracts

Text fields: source record ID, source, mapped ticker, text, publication timestamp, earliest documented availability timestamp, retrieval timestamp and availability basis. All instants require explicit offsets and normalize to UTC. Require publication <= availability <= retrieval.

For observed records, availability is first observed access to that exact content version, preserved across subsequent retrievals. Historical imports without evidence of prior availability use `historical_assumption`; M1 excludes them by default. Opting in creates a retrospective assumption, not proof of live feasibility. An importer must not label assumed timestamps as observed.

Daily prices require ticker, session date, OHLCV, availability and split ratio (`1` means no split). The session calendar must contain every exchange session across the study range and include at least the session before the first target. M1 validates structure and consumes the provided calendar; M2 generates/verifies completeness and provenance against pinned `exchange_calendars` XNYS definitions. Historical price availability is an explicit, recorded assumption by default.

Text from regular trading hours before the previous close is outside this specific overnight design. Late articles do not move into a later forecast simply because they become available later. M1 deduplicates normalized identical content within source/ticker, keeping earliest availability. M2 adds URL/source-ID and syndication controls plus version tracking.

Store features and outcomes separately. Final training code must select a frozen numeric feature allowlist, not automatically consume every numeric column. Targets, target return, future close and label-availability metadata are never model inputs.

## Feature specification for M3–M4

Start with source-separated, equal-weight aggregation. For each text, polarity = P(positive) − P(negative). Aggregate mean polarity, polarity dispersion, mean positive/negative/neutral probabilities and mention count per ticker/window/source. Keep missing-text indicators. A missing source window is missing information, not a neutral observation. No weights derived from votes or author metadata observed after the cutoff.

Price features use prior-session information: one- and five-session returns, trailing 10-session moving-average distance, 10-session standard deviation of daily returns and five-session RSI using a documented Wilder convention. Define zero-gain/zero-loss behavior, warmup and split adjustment explicitly before implementation. Technical rolling windows must never mix tickers. No imputation across ticker boundaries. Imputation/scaling, when needed, fit train only.

Exclude label periods crossing stock splits. M4 must also prevent raw split discontinuities from contaminating technical windows, using a documented past-only adjustment or dropping affected warmup windows. Do not present retrospectively adjusted data as point-in-time verified.

## Models, partitions and baselines

- Split by shared distinct target dates, initially 60% train / 20% validation / 20% final test. All tickers on a date share the partition. Purge training or validation observations whose labels are not available before the next partition starts.
- Use expanding-window development folds within train if tuning requires them. Never shuffle time or use random k-fold cross-validation. Fit transformations inside each fold.
- Keep a small, logged search budget: at most eight settings per model family/feature set. Choose settings using validation AUC, with a recorded tie rule. Freeze all choices before test scoring.
- Primary comparison: combined sentiment + price LR versus price-only LR. Secondary comparisons: price + news, price + social, sentiment-only, and corresponding RF variants. Match rows and use identical tuning budgets. Report a text-present subset separately, without replacing the full-sample primary analysis.
- Explain LR predictions using signed coefficients in the fitted feature scale. For RF, report permutation importance with correlated-feature caveats and a few error cases. Interpret model associations without claiming causality.
- Baselines: training-majority classifier, always-up classifier, seeded fair coin, and prior-return-sign momentum. Report repeated-seed variability for the random baseline.
- A literal zero-drift random-walk price forecast predicts the previous close, which maps to non-up under the defined tie rule. Its constant score has AUC 0.5 when both classes exist and potentially F1 0 for the positive class. Also show persistence/momentum rather than treating “random walk” as interchangeable with a coin flip.
- Buy-and-hold is an investment policy. Its always-long directional analogue is always-up. AUC for constant scores is 0.5 when defined. Any later portfolio-return comparison needs a separately specified entry time, costs and executable prices.

Report positive-class F1 at a fixed 0.5 probability threshold, ROC-AUC, accuracy, precision, recall, confusion matrix and class prevalence. If any evaluation slice has one class, ROC-AUC is undefined and must be reported as N/A. Never manufacture a value or drop an unfavorable slice without disclosure.

## Statistical evaluation

Save row-level probabilities for every model and baseline. The primary endpoint is pooled held-out delta AUC for combined LR minus price-only LR. Use 2,000 paired moving-block bootstrap replicates of five consecutive test sessions, resampling dates jointly across tickers and model predictions. Report a percentile 95% interval and sensitivity to block lengths 1 and 10. State the stationarity and dependence assumptions and count invalid single-class replicates.

Use development data to finalize this protocol before opening the test result. Report per-ticker effects and macro averages as secondary views. Mark secondary model/source/subgroup comparisons exploratory or apply a declared multiplicity correction. Do not choose the best ticker after seeing the test and call it the primary finding.

Statistical significance alone does not establish useful trading performance, causality, or future stability. Report coverage, sample size, effect size and uncertainty together. Repeated test-driven revisions require a new holdout or explicit exploratory labeling.

## Architecture and artifacts

Adapters/imports → immutable source snapshots → normalization/validation → exchange-window assignment → FinBERT cache → feature matrix + separate labels → chronological modeling → predictions → paired evaluation → report.

Use Python 3.11+ and a small command-line application. M1 uses the standard library only. M2–M6 add pandas/numpy, exchange calendars, provider clients, transformers/torch, scikit-learn and plotting libraries only when needed, then lock a tested environment. Local JSONL/CSV is sufficient for M1; normalized Parquet/SQLite and incremental checkpoints arrive in M2.

Each real run must record input hashes, source coverage and availability assumptions, package/model revisions, configuration, code identity, split dates, seed, exclusions, timings and outputs. Keep API keys in environment variables. Avoid storing Reddit author profiles or personal metadata unnecessary for the study. Separate restricted raw data from shareable manifests and reports, applying the approved source retention requirements.

## Risks and responses

| Risk | Response |
| --- | --- |
| Historical text unavailable or insufficient | M2 viability gate; obtain an appropriate dataset or label the work a feasibility study |
| Temporal leakage from delayed/revised text | Earliest observed availability, versioned content, strict cutoff audit |
| API failures / quotas | Bounded retries, incremental checkpoints and per-source request budgets |
| Ticker ambiguity, spam, sarcasm | Conservative matching and manual quality sample; document residual noise |
| False precision from small correlated samples | Minimum distinct dates, paired date-block intervals and effect-size reporting |
| Present-day universe selection | Document chosen convenience sample and survivorship/selection limitations |
| Test-driven model choice | Freeze the protocol and reserve a final untouched date block |
| FinBERT domain errors / uncalibrated scores | Review errors; call scores model outputs; calibrate only on development data if justified |

## Final handoff

Deliver a runnable project, PRD and milestone log, data card, frozen experiment protocol, source manifests, model predictions, metrics/uncertainty tables, source ablation charts, report and review demonstration. Include an explicit statement of which original requirements were fulfilled, reduced or blocked. No result table is complete until it names the data period and evidence limitations.
