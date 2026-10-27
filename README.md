# Stock Sentiment Lab

A small local AI project for **Apple (AAPL), NVIDIA (NVDA), and Amazon (AMZN)**.

**Price history + financial news → FinBERT sentiment → Ridge regression → predicted closing price → comparison with the actual close.**

## What happens

1. **yfinance** downloads daily opening, high, low, closing prices and trading volume from Yahoo Finance. This is data collection, not AI.
2. **NewsAPI** supplies financial headlines. Its free Developer plan has a 24-hour delay and is for local development. There is no active Reddit integration.
3. **FinBERT**, the pinned `ProsusAI/finbert` transformer, scores each headline as positive, neutral and negative. Eligible news from a three-day window is averaged by stock. Sentiment is not the probability of a price rise.
4. **Ridge regression** combines five price indicators (one-day return, five-day return, momentum, volatility and RSI) with news polarity and headline count. It learns daily returns from earlier stock-days; predicted close = preceding close × (1 + predicted return). The scaler is fitted on training examples only; alpha is fixed at 10.
5. The app shows **predicted close, actual close, dollar error and percentage error**. Lower error is better. A separate price-only Ridge model and predicting the preceding close are the baselines.

The website has Overview, News & sentiment, Price comparison and Market data. The old experiment screens, snapshot import/export and downloadable research archive have been removed from the dashboard.

## Run locally

Python 3.12 and Node.js 22.19+ are used by the current environment.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-modeling.lock
.venv/bin/python -m sentiment_lab prepare-finbert
cd website
npm ci
npm run build
cd ..
```

Copy `.env.example` to `.env` and enter `NEWSAPI_API_KEY` locally. Never commit the key. Then:

```bash
.venv/bin/python -m sentiment_lab.live refresh
.venv/bin/python -m sentiment_lab.live compare
.venv/bin/python -m sentiment_lab.live serve --port 8765
```

Open [the local dashboard](http://127.0.0.1:8765). The API key is read only by Python; the browser receives results through `/api/live`. SQLite caches, provider articles, model weights and results stay in ignored local files.

For the evaluator's example:

```bash
.venv/bin/python -m sentiment_lab.live compare --target 2026-10-26
```

## How the comparison works

The comparison is run today using historical data. For 26 October 2026, the previous trading close is Friday, 23 October; the 25th was Sunday. Features are cut off five minutes before the target session opens. News is assumed available at publication + 24 hours, and daily prices at market close + one minute. Later Sunday news is excluded if its delay has not ended by Monday's cutoff.

Training uses only earlier days whose closing labels were available before the target cutoff. The target close is withheld from fitting and scaling, then revealed for evaluation. Historical delivery times, article revisions and vendor price revisions cannot be verified. This explanation is in the website's **How this works** dialog; the main page remains **Price comparison**.

A bounded recent news history is collected and scored once per completed target day. Results and fitted model parameters are saved by date. Repeating that date reuses the saved result. A short training window and one comparison day demonstrate the method; they cannot establish forecasting accuracy or prove sentiment's value.

## Refreshing

The news/price collector has a six-hour due check and an overlapping-run lease. Run `refresh` and then `compare` every six hours. The configured local Codex automation does this while the Mac is awake and Codex is available; it is not a cloud service. The browser checks saved results once a minute. A page reload does not spend NewsAPI requests.

`compare` uses the latest completed exchange session by default, so weekends and holidays are handled by the exchange calendar. Headline requests and FinBERT scores are cached; capped searches are disclosed. Missing news is recorded as missing, not given invented neutral scores.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests -q
cd website
npm test
npm run build
```

The regression checks ensure changing target actuals cannot affect the fitted model or predictions, unavailable labels cannot enter training, and delayed Sunday news is excluded at Monday's cutoff. The earlier pipeline modules remain available as implementation utilities; they do not power extra archive pages or old headline claims in the current app.
