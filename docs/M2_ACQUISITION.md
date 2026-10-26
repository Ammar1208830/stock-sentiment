# M2 acquisition and feasibility

M2 implements a real yfinance adapter, a NewsAPI client, approved historical text imports, pinned XNYS calendar generation/verification, SQLite checkpoints, content versions, deduplication, source coverage and a feasibility gate. It does not implement FinBERT or classifiers.

## Installed environment

The project virtual environment is `.venv`, tested with Python 3.12.6. Recreate it with:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-acquisition.lock
.venv/bin/python -m unittest discover -s tests -v
```

The full resolved dependency set is pinned in `requirements-acquisition.lock`; the top-level constraints are in `requirements-acquisition.in`. The lock was exercised on macOS arm64 / Python 3.12. CI defines a Linux 3.12 acquisition job but remote execution is not claimed. Core M1 and most unit tests remain usable without acquisition dependencies. Calendar integration tests skip in that environment.

## Collect data

From the project root:

```bash
.venv/bin/python -m sentiment_lab collect --out artifacts/acquisition-001
```

Default config is `configs/project.toml`; collection settings are `configs/acquisition.toml`. The initial target date range is 2023-10-23 to 2026-10-23 inclusive. The calendar and prices include one preceding session. The default five equities are AAPL, MSFT, NVDA, AMZN and TSLA.

Each ticker's successful price response and every successful NewsAPI page are committed to `data/cache/acquisition.sqlite`. Resume with the same cache and a fresh output folder. The client reuses completed requests and normalizes their original observation time. Refresh flags explicitly request new provider snapshots:

```bash
.venv/bin/python -m sentiment_lab collect --out artifacts/acquisition-002
.venv/bin/python -m sentiment_lab collect --refresh-news --out artifacts/acquisition-refresh
.venv/bin/python -m sentiment_lab collect --offline --out artifacts/acquisition-offline
```

Offline mode makes no provider requests; missing cache entries become source gaps. `--refresh-prices` also exists. Offline mode rejects refresh flags. Keep separate caches when changing the universe or using synthetic fixtures. A nonempty output directory is never overwritten.

The command completes even if external sources are missing, producing explicit collection statuses and an engineering-only decision. For a strict automation check, add `--require-feasible`: exit code 3 means the full-study data gate failed. Exit code 2 means invalid input/setup; zero otherwise means the command ran successfully, not that predictive value exists.

## Price conventions

The adapter calls `Ticker.history` with daily interval, `auto_adjust=False`, `back_adjust=False`, `repair=False`, `actions=True`, and an explicit exclusive end date. It preserves dividends, split ratios and retrieval/snapshot metadata. Finished sessions only enter the export. Invalid OHLCV rows are counted as rejected; missing bars remain missing.

Yahoo vendor OHLC can still contain retrospective split adjustments and corrections. “Unadjusted” API flags do not establish contemporaneous raw exchange quotes. The study excludes split target sessions; M4 must handle split effects in technical windows. Do not silently apply the split ratio again to already vendor-adjusted closes.

Default `price_availability = "historical_assumption"` assigns close + 1 minute, clearly recorded as a retrospective assumption. This makes historical price experiments possible without pretending to have observed historical API latency. Setting `price_availability = "observed"` uses actual retrieval time; a newly downloaded three-year series then cannot be used as live-known features at past cutoffs. Feasibility reports expose that distinction.

## NewsAPI

Set `NEWSAPI_API_KEY` in your local shell environment, using your existing key. The application reads only the environment, does not automatically read `.env`, and never prints or persists the key. Do not paste credentials into chat or committed config.

The collector searches a configurable recent lookback, by UTC date and ticker/company query, with pagination and bounded retries. Defaults: 28-day lookback, 90 HTTP attempts per UTC day across all reruns, 100 articles/page, at most 3 pages per ticker/day, at most 3 attempts per request and 20-second timeout. Cached reads do not consume budget; retries and failures do. HTTP 429/temporary server errors/network failures retry with capped backoff; authentication errors do not retry. A truncated query or exhausted budget produces `partial`, preserving progress.

`price_daily_call_budget` limits top-level yfinance SDK calls including retries, not its hidden internal HTTP request count. Provider/account quotas may also include requests from other applications; the local ledger only covers this cache. The code does not buy plans or increase configured budgets automatically.

NewsAPI Developer currently delays articles by 24 hours and limits searchable history to one month. A bounded recent collector cannot furnish the multi-year historical study. See [pricing](https://newsapi.org/pricing) and [Everything endpoint](https://newsapi.org/docs/endpoints/everything). Provider plans determine which requested dates are available; access errors are reported, not worked around.

Only headlines are scored in the initial news design. Publication timestamps and first observed access are different fields. API collection always uses first observed retrieval for availability; it never backdates availability to publication. Re-poll with `--refresh-news` when you want new observations; a stored old page is a checkpoint, not a live subscription.

## Historical news and Reddit imports

Reddit's academic research route requires approved research access. An ordinary PRAW client is not assumed to provide that access or full history. See [Reddit's access guidance](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data). There is no undocumented research-endpoint scraper. M2 provides an import adapter for an approved research export; adapting a live research API requires its actual approved interface and permissions.

Normalize permitted exports to the text JSONL contract, adding `origin_id` for the stable original article/post and optionally `url`:

```json
{"record_id":"source-content-version","origin_id":"source-origin-id","source":"reddit","ticker":"AAPL","text":"Actual permitted text","published_at":"2024-11-25T22:00:00Z","available_at":"2024-11-25T22:01:00Z","retrieved_at":"2024-11-25T22:01:00Z","availability_basis":"observed"}
```

The example demonstrates the schema, not a real research record. For genuine historical text without evidence of prior access, use `historical_assumption`; it is excluded by default. An opt-in requires `allow_historical_assumptions = true` in project config and remains visible in the manifest. Do not mark estimated timestamps as observed.

Copy and complete the relevant provenance example in `configs/`. `reference` should identify your existing license/approved research export and permitted study. The importer validates the declaration but cannot independently authenticate it.

```bash
.venv/bin/python -m sentiment_lab import-text \
  --input data/raw/approved-reddit.jsonl \
  --provenance data/raw/reddit-provenance.json

.venv/bin/python -m sentiment_lab import-text \
  --input data/raw/licensed-news.jsonl \
  --provenance data/raw/news-provenance.json

.venv/bin/python -m sentiment_lab collect --offline --out artifacts/with-imports
```

The importer rejects inconsistent sources, invalid ticker/timestamp contracts and future retrieval timestamps. It validates the complete batch before writes. It retains only normalized study fields, never Reddit author/profile attributes. It preserves input/provenance hashes and the declared reference.

## Versions, deduplication and matching

- Canonicalize URLs, removing fragments and common tracking parameters while preserving other query parameters.
- News tickers match explicit uppercase symbols/cashtags or company names accompanied by financial context. Aliases are defined for the starting five stocks; add reviewed aliases before expanding the universe. Ambiguous/unmatched records are counted.
- Exact content versions have stable hashes. Repeated retrievals preserve earliest availability and its first snapshot ID, while retaining latest retrieval/snapshot metadata. Changed text/publication metadata produces a new version.
- Export deduplication groups identical normalized text by ticker, UTC publication date and availability basis, including copies across sources. `text_audit.jsonl` retains every version's lineage and duplicate links. Cross-source duplicate removal can reduce a source's measured coverage and is disclosed.
- Window selection uses the latest available version of each source origin before the decision cutoff. An edit first observed after the cutoff cannot replace an earlier version in a historical forecast.
- Daily content deduplication permits the same boilerplate to appear again on another publication date. M1's global identical-text deduplication was refined accordingly.

Review `matching_review.jsonl` manually before modeling. Exporting this sample does not certify its entity mapping. No manual audit of real news/social text is claimed when those sources have no data.

## Calendar generation and verification

```bash
.venv/bin/python -m sentiment_lab calendar \
  --start 2023-10-23 --end 2026-10-23 --out artifacts/calendar-001

.venv/bin/python -m sentiment_lab verify-calendar \
  --start 2023-10-23 --end 2026-10-23 \
  --sessions artifacts/calendar-001/sessions.csv
```

Completeness, holidays, early closes and DST follow pinned `exchange_calendars` XNYS definitions. Verification requires every expected session, including the preceding session. It does not replace checking extraordinary exchange announcements when needed.

## Outputs and feasibility

Every collection/export produces:

| Artifact | Meaning |
| --- | --- |
| sessions.csv / calendar.json | Complete versioned schedule with source metadata |
| prices.csv | Normalized vendor OHLCV, corporate actions and availability assumptions |
| texts.jsonl / text_audit.jsonl | Exported versions and duplicate/provenance lineage |
| matching_review.jsonl | Up to 25 records for manual ticker-matching review |
| coverage.csv | Every closed target date and ticker, usable price pair and eligible source counts |
| feasibility.json | Date splits, coverage, data thresholds, source gaps and study decision |
| DATA_CARD.md | Human-readable scope and provenance limitations |
| manifest.json | Output/code hashes, environment/dependency versions, lock hash, snapshots, import provenance and request ledger |

The default gate requires 630 shared usable sessions, 126 shared test sessions, and 80% coverage for each source on each ticker overall and within every partition. Shared means all configured tickers have usable price pairs on that session. Text coverage only counts actual cutoff-eligible records, not the number downloaded. Split-target rows and labels not known at later split boundaries are excluded. These gates assess data availability, not effect size or statistical power.

Decisions: `full_study_feasible`, `reduced_news_only_study_feasible`, or `engineering_demonstration_only`. Reduced scope needs an explicit protocol amendment before claiming the original combined-source study is answered. Synthetic data always produces engineering-only status. Existing price/timestamp assumptions remain visible even when a numerical coverage gate passes.

## Retention and deletion

For deleted/revoked source records:

```bash
.venv/bin/python -m sentiment_lab purge-text --source reddit --origin-id source-origin-id
```

This removes all normalized cached versions of that source origin and stores a content-free deletion audit. It does not delete your original import files or existing exported artifacts. Remove affected files and regenerate exports; follow the retention/deletion conditions attached to your actual approved source. The cache and local data/artifacts are excluded from version control.

## Verified run on 26 October 2026

The live run in `artifacts/m2-live` downloaded 3,775 daily bars across five stocks over 755 price sessions, including the preceding session. The feasibility report has 753 shared usable target sessions and 151 shared test sessions. NewsAPI was unavailable because no key was set. Reddit had no approved export configured. There were zero text records and the study correctly remained `engineering_demonstration_only`.

Price calls succeeded for all five tickers. News transport behavior was verified with deterministic mocked responses, not a live authenticated NewsAPI request. The social importer was tested with fixtures, not a real approved dataset. A separate synthetic `m2-demo` verifies cache/export behavior without network access.
