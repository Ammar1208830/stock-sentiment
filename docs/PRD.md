# Stock Sentiment Lab — current requirements

## Goal

Demonstrate whether financial-headline sentiment improves a simple closing-price estimate for Apple (AAPL), NVIDIA (NVDA) and Amazon (AMZN). Report errors honestly rather than assume the sentiment model is better.

## Scope

- Daily price history from yfinance.
- Delayed financial headlines from NewsAPI; no active Reddit collection.
- Real local inference using pinned ProsusAI/finbert weights.
- Price indicators plus three-day eligible news polarity and headline count.
- StandardScaler and Ridge regression with fixed alpha 10, predicting daily return and converting it to closing price.
- Price-only Ridge and unchanged preceding close as baselines.
- A simple local dashboard: Overview, News & sentiment, Price comparison, Market data.
- Six-hour collection, cached historical comparisons and a one-minute browser results check.
- Timing assumptions explained in a How this works dialog.

## Evaluation

For a completed target session, train on earlier stock-days whose labels were available before the target opening cutoff. Fit scaling on training rows only. Hide the target close from fitting and then compare predicted and actual closes. Show absolute dollar error, absolute percentage error, mean absolute error and both baselines.

The evaluator's example is 26 October 2026: prior closing prices are from Friday the 23rd, as the 25th is Sunday. The cutoff is 09:25 America/New_York on the 26th. Headlines are usable only after their assumed 24-hour delivery delay; later Sunday headlines are excluded.

These comparisons are run now with historical data, not forecasts actually issued at those past times. Historical article versions, delivery timestamps and price revisions are not verified. A recent small news sample and one comparison day do not establish forecasting accuracy.

## Acceptance

1. Only AAPL, NVDA and AMZN appear in active collection and the website.
2. The combined model actually receives nonempty scored news and price features.
3. Target-day closing labels cannot influence training, scaling or predictions.
4. Delayed headlines and unavailable training labels are excluded at the cutoff.
5. Price errors and both baselines are visible, including worse-than-baseline results.
6. Missing news stays missing; unrelated video tags, shopping titles and Apple Hospitality stories are excluded.
7. API keys stay in the ignored local .env, never in browser bundles or Git.
8. Repeated collection before six hours and repeated saved comparison days reuse existing results.
9. The site works in both themes and responds to reduced-motion preferences.

## Out of scope

Reddit, trading execution, live market quotes, guaranteed returns, large model grids and the previous archive screens/downloads.
