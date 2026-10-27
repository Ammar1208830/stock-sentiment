"""Small, delayed-news price regression with a chronological comparison day.

Historical replay assumes headline availability at publication + 24 hours and
vendor closes at exchange close + one minute. These are explicit assumptions,
not observed historical delivery timestamps or forecasts issued in the past.
"""
from dataclasses import replace
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import statistics
import tomllib

from .acquisition import AcquisitionConfig
from .calendar import generate_calendar
from .config import Config
from .contracts import ContractError, PriceBar, timestamp
from .finbert import FinBERTScorer, SentimentConfig, normalize_text
from .indicators import PRICE_FEATURES, known_history, technical_values
from .normalization import ALIASES, deduplicate_export, normalize_news
from .news_filter import financial_headline, QUERY_CONTEXT, VERSION
from .providers import NewsClient, yahoo_history
from .sentiment_cache import SentimentCache, score_unique
from .store import Store, digest, utc_now


FEATURES = (*PRICE_FEATURES, 'news_polarity', 'news_count_log')


def eligible_news(articles, ticker, decision):
    """A three-day window of news released by the cutoff, including the delay."""
    lower = decision - timedelta(days=3)
    return [row for row in articles if row['ticker'] == ticker
            and lower <= timestamp(row['published_at']) + timedelta(hours=24) < decision]


def make_examples(tickers, sessions, bars, articles, first_day, last_day):
    """Construct predictors separately from labels; never use target-day prices."""
    converted = {(ticker, datetime.fromisoformat(day).date()): PriceBar.from_dict(row)
                 for (ticker, day), row in bars.items()}
    result = []
    for index, session in enumerate(sessions):
        if index == 0 or not first_day <= session.date <= last_day:
            continue
        decision = session.open_at - timedelta(minutes=5)
        for ticker in tickers:
            history, _ = known_history(ticker, session.date, decision, sessions, converted)
            if not history or history[-1].session != sessions[index - 1].date:
                continue
            try:
                values = technical_values(history)
            except ContractError:
                continue
            news = eligible_news(articles, ticker, decision)
            values.update(news_polarity=statistics.fmean(row['sentiment']['polarity'] for row in news) if news else 0.,
                          news_count_log=math.log1p(len(news)))
            actual = bars.get((ticker, str(session.date)))
            if actual and actual['split_ratio'] != 1:
                continue
            result.append({'ticker': ticker, 'session': str(session.date), 'decision_at': decision.isoformat(),
                           'reference_session': str(history[-1].session), 'reference_close': history[-1].close,
                           'values': values, 'news_count': len(news),
                           'news_latest_published_at': max((row['published_at'] for row in news), default=None),
                           'label_available_at': (session.close_at + timedelta(minutes=1)).isoformat(),
                           'actual_close': actual['close'] if actual else None})
    return result


def fit_comparison(examples, target_day, evaluated_at):
    """Train two fixed Ridge models; target outcomes never reach fit or scaling."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    targets = [row for row in examples if row['session'] == target_day]
    if not targets:
        raise ContractError('No complete price histories for the comparison day')
    cutoff = min(timestamp(row['decision_at']) for row in targets)
    training = [row for row in examples if row['session'] < target_day and row['actual_close'] is not None
                and timestamp(row['label_available_at']) < cutoff]
    if len(training) < 15 or len({row['session'] for row in training}) < 5:
        raise ContractError('Need at least five earlier trading days and fifteen training examples')
    if not any(row['news_count'] for row in training):
        raise ContractError('Combined training requires actual scored news in earlier examples')
    y = [row['actual_close'] / row['reference_close'] - 1 for row in training]
    predictions = {}
    states = {}
    for name, features in (('price_only', PRICE_FEATURES), ('combined', FEATURES)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit([[row['values'][key] for key in features] for row in training], y)
        predictions[name] = model.predict([[row['values'][key] for key in features] for row in targets]).tolist()
        scaler, ridge = model.steps[0][1], model.steps[1][1]
        states[name] = {'features': list(features), 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                        'coefficients': ridge.coef_.tolist(), 'intercept': float(ridge.intercept_), 'alpha': 10.0}
    rows = []
    for index, target in enumerate(targets):
        predicted = target['reference_close'] * (1 + predictions['combined'][index])
        price_only = target['reference_close'] * (1 + predictions['price_only'][index])
        if not math.isfinite(predicted) or predicted <= 0 or not math.isfinite(price_only) or price_only <= 0:
            raise ContractError('Regression produced an invalid price; comparison was not saved')
        actual = target['actual_close'] if timestamp(target['label_available_at']) <= evaluated_at else None
        error = abs(predicted - actual) if actual is not None else None
        rows.append({key: value for key, value in target.items() if key != 'values'} | {
            'actual_close': actual, 'predicted_close': predicted, 'price_only_close': price_only,
            'baseline_close': target['reference_close'], 'absolute_error': error,
            'percentage_error': error / actual * 100 if actual is not None else None,
            'price_only_error': abs(price_only - actual) if actual is not None else None,
            'baseline_error': abs(target['reference_close'] - actual) if actual is not None else None,
            'news_polarity': target['values']['news_polarity'] if target['news_count'] else None})
    resolved = [row for row in rows if row['actual_close'] is not None]
    return {'target_session': target_day, 'evaluated_at': evaluated_at.isoformat(), 'rows': rows,
            'training_rows': len(training), 'training_sessions': len({row['session'] for row in training}),
            'training_start': min(row['session'] for row in training), 'training_end': max(row['session'] for row in training),
            'training_news_rows': sum(bool(row['news_count']) for row in training),
            'mae': statistics.fmean(row['absolute_error'] for row in resolved) if resolved else None,
            'mape': statistics.fmean(row['percentage_error'] for row in resolved) if resolved else None,
            'price_only_mae': statistics.fmean(row['price_only_error'] for row in resolved) if resolved else None,
            'baseline_mae': statistics.fmean(row['baseline_error'] for row in resolved) if resolved else None,
            'model': 'StandardScaler + Ridge regression (alpha=10), predicting daily return then closing price',
            'models': states, 'input_sha256': digest(examples), 'evaluation_kind': 'historical_replay',
            'availability_assumption': 'News publication + 24 hours; daily close + one minute. Historical data fetched now; this is not a forecast issued on that historical date.',
            'limitation': 'One comparison day and a short news history cannot establish forecasting accuracy or that sentiment improves predictions.'}


def run(root, target_day=None, *, now_fn=utc_now):
    """Collect a bounded news history, score once, fit and save a comparison."""
    from .live import LiveStore, load_local_env
    root = Path(root)
    environment = load_local_env(root)
    config = Config.load(root / 'configs/project.toml')
    limits = tomllib.loads((root / 'configs/live.toml').read_text())
    if not environment['NEWSAPI_API_KEY']:
        raise ContractError('Configure NEWSAPI_API_KEY in the local .env file first')
    if environment['NEWSAPI_PLAN'] not in {'', 'developer'}:
        raise ContractError('This comparison is configured for the delayed Developer plan')
    now = now_fn()
    sessions, _ = generate_calendar(datetime(2023, 10, 23).date(), now.date() + timedelta(days=1))
    closed = [session for session in sessions if session.close_at + timedelta(minutes=1) < now]
    target = next((s for s in closed if str(s.date) == target_day), None) if target_day else closed[-1]
    if target is None:
        raise ContractError('Choose a completed exchange session')
    decision = target.open_at - timedelta(minutes=5)
    end = min(decision - timedelta(hours=24), now - timedelta(hours=24, minutes=5))
    start = max(end - timedelta(days=28), now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=28))
    if (end - start).days < 14:
        raise ContractError('The free news history is too short for that comparison date')
    live = LiveStore(root / 'data/cache/live.sqlite')
    try:
        saved = live.get(f'price_evaluation:{target.date}')
        if saved and saved.get('news_filter_version') == VERSION and [row['ticker'] for row in saved['rows']] == list(config.tickers):
            live.put('price_evaluation', saved)
            live.put('price_evaluation_status', 'ready')
            live.put('price_evaluation_error', None)
            return {'status': 'cached', 'target_session': str(target.date), 'training_rows': saved['training_rows'], 'mae': saved['mae']}
        live.put('price_evaluation_status', 'running')
        articles = []
        truncated = 0
        with Store(root / 'data/cache/live_acquisition.sqlite') as cache:
            cache.bind(config.to_dict())
            client = NewsClient(cache, environment['NEWSAPI_API_KEY'], replace(AcquisitionConfig(), http_attempts=2), now=now_fn)
            # Weekly windows preserve earlier training coverage; cached pages cost no new requests.
            cursor = start
            while cursor < end:
                until = min(cursor + timedelta(days=7), end)
                for ticker in config.tickers:
                    query = f'("{ticker}" OR "{ALIASES[ticker][0]}") AND {QUERY_CONTEXT}'
                    for page in (1,):
                        params = {'q': query, 'from': cursor.isoformat(), 'to': until.isoformat(), 'language': 'en',
                                  'sortBy': 'publishedAt', 'searchIn': 'title', 'pageSize': 100, 'page': page}
                        snapshot = client.page(params)
                        payload = json.loads(snapshot['payload'])
                        normalized, _ = normalize_news(payload, snapshot['retrieved_at'], config.tickers, snapshot['id'])
                        articles.extend(row for row in normalized if start <= timestamp(row['published_at']) <= end
                                        and financial_headline(row['text'], row['ticker']))
                        if page * 100 >= payload['totalResults']:
                            break
                        if page == 1:
                            truncated += 1
                cursor = until
            articles, _ = deduplicate_export(articles)
            settings = SentimentConfig.load(root / 'configs/sentiment.toml')
            scorer = FinBERTScorer(settings, root / 'data/cache/models/finbert', offline=True)
            print(f'Scoring {len(articles)} historical ticker-headline records with FinBERT.', flush=True)
            with SentimentCache(root / 'data/cache/live_sentiment.sqlite', synthetic=False) as sentiment:
                scores, score_stats = score_unique([row['text'] for row in articles], scorer, sentiment, settings.batch_size)
            scored = [{'ticker': row['ticker'], 'published_at': row['published_at'], 'title': row['text'],
                       'url': row['url'], 'sentiment': scores[normalize_text(row['text'])]} for row in articles]
            live.put('price_training_news', scored)
            bars = {}
            by_day = {str(session.date): session for session in closed}
            for ticker in config.tickers:
                attempt = cache.reserve('yfinance', f'comparison:{ticker}', limits['price_daily_call_budget'], now_fn())
                try:
                    raw = yahoo_history(ticker, str(sessions[0].date), (now.date() + timedelta(days=1)).isoformat(), cache_dir=root / 'data/cache/yfinance')
                    cache.finish_attempt(attempt, 'ok')
                except Exception:
                    cache.finish_attempt(attempt, 'failed')
                    raise
                for row in raw:
                    session = by_day.get(row['session'])
                    if session:
                        bars[(ticker, row['session'])] = {'ticker': ticker, 'session': row['session'],
                            **{key.lower(): row[key] for key in ('Open', 'High', 'Low', 'Close', 'Volume')},
                            'available_at': (session.close_at + timedelta(minutes=1)).isoformat(), 'split_ratio': row.get('Stock Splits', 0) or 1}
        examples = make_examples(config.tickers, sessions, bars, scored, (start + timedelta(days=4)).date(), target.date)
        result = fit_comparison(examples, str(target.date), now_fn())
        result.update(news_start=start.isoformat(), news_end=end.isoformat(), news_records=len(scored),
                      truncated_queries=truncated, finbert_signature=scorer.signature, scoring=score_stats, news_filter_version=VERSION)
        live.put('price_evaluation', result)
        live.put('price_evaluation_status', 'ready')
        live.put('price_evaluation_error', None)
        # Keep a date-specific result so the next comparison does not erase this one.
        live.put(f'price_evaluation:{target.date}', result)
        return {'status': 'ready', 'target_session': str(target.date), 'training_rows': result['training_rows'],
                'news_records': len(scored), 'mae': result['mae'], 'baseline_mae': result['baseline_mae']}
    except Exception as error:
        live.put('price_evaluation_status', 'error')
        live.put('price_evaluation_error', f'Price comparison unavailable ({type(error).__name__}). Previous results retained.')
        raise
    finally:
        live.close()
