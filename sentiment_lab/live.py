"""Local three-stock news feed and combined price-comparison dashboard.

Provider credentials and collected data stay in ignored local files.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import statistics
import tomllib
from urllib.parse import urlsplit

from .acquisition import AcquisitionConfig
from .config import Config
from .contracts import ContractError, PriceBar, timestamp
from .finbert import FinBERTScorer, SentimentConfig, normalize_text
from .normalization import ALIASES, canonical_url, deduplicate_export, normalize_news
from .news_filter import financial_headline, QUERY_CONTEXT
from .providers import NewsClient, ProviderError, yahoo_history
from .sentiment_cache import SentimentCache, score_unique
from .store import Store, digest, utc_now


ROOT = Path(__file__).resolve().parents[1]
INTERVAL = timedelta(hours=6)


def load_local_env(root):
    """Read a deliberately simple ignored .env; shell variables take precedence."""
    values = {}
    path = root / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip().startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            if key.strip() in {'NEWSAPI_API_KEY', 'NEWSAPI_PLAN'}:
                values[key.strip()] = value.strip().strip('\"\'')
    # Use the installed, verified CA bundle without weakening TLS verification.
    try:
        import certifi
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    except ImportError:
        pass
    return {key: os.environ.get(key, values.get(key, '')) for key in ('NEWSAPI_API_KEY', 'NEWSAPI_PLAN')}


class LiveStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS live_state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS live_articles(id TEXT,ticker TEXT,payload TEXT NOT NULL,PRIMARY KEY(id,ticker));
        ''')
        self.db.commit()

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM live_state WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO live_state VALUES (?,?)', (key, json.dumps(value, allow_nan=False)))

    def claim(self, now, force=False):
        self.db.execute('BEGIN IMMEDIATE')
        if timestamp(self.get('lease_until', '2000-01-01T00:00:00Z')) > now:
            self.db.rollback()
            return False
        last = self.get('last_attempt_at')
        if not force and last and timestamp(last) + INTERVAL > now:
            self.db.rollback()
            return False
        self.db.execute('INSERT OR REPLACE INTO live_state VALUES (?,?)', ('lease_until', json.dumps((now + timedelta(minutes=30)).isoformat())))
        self.db.execute('INSERT OR REPLACE INTO live_state VALUES (?,?)', ('last_attempt_at', json.dumps(now.isoformat())))
        self.db.commit()
        return True

    def close(self):
        self.db.close()


def news_summary(articles, tickers):
    result = []
    for ticker in tickers:
        rows = [row for row in articles if row['ticker'] == ticker and row.get('sentiment')]
        means = {label: statistics.fmean(row['sentiment'][label] for row in rows) if rows else None
                 for label in ('positive', 'negative', 'neutral', 'polarity')}
        result.append({'ticker': ticker, 'articles': len(rows), **means,
                       'label': max(('positive', 'negative', 'neutral'), key=lambda label: means[label]) if rows else 'unavailable'})
    return result


def refresh(root=ROOT, force=False, *, news_transport=None, price_fetch=yahoo_history,
            scorer_factory=FinBERTScorer, now_fn=utc_now, sessions_factory=None):
    root = Path(root)
    environment = load_local_env(root)
    config = Config.load(root / 'configs/project.toml')
    live = tomllib.loads((root / 'configs/live.toml').read_text())
    store = LiveStore(root / 'data/cache/live.sqlite')
    now = now_fn()
    if not store.claim(now, force):
        store.close()
        return {'status': 'not_due_or_already_running'}
    issues = []
    try:
        if environment['NEWSAPI_PLAN'] not in {'', 'developer'}:
            raise ContractError('This local feed is configured explicitly for the delayed Developer plan')
        settings = replace(AcquisitionConfig(), news_daily_request_budget=live['news_daily_request_budget'], http_attempts=2)
        if not environment['NEWSAPI_API_KEY']:
            issues.append('NewsAPI setup required: add NEWSAPI_API_KEY to the ignored local .env file.')
        else:
            end = now - timedelta(hours=live['news_delay_hours'], minutes=5)
            start = end - timedelta(hours=live['news_window_hours'])
            records = []
            publisher_names = {}
            truncated = 0
            successful_queries = 0
            with Store(root / 'data/cache/live_acquisition.sqlite') as acquisition:
                acquisition.bind(config.to_dict())
                kwargs = {'transport': news_transport} if news_transport else {}
                client = NewsClient(acquisition, environment['NEWSAPI_API_KEY'], settings, now=now_fn, **kwargs)
                for ticker in config.tickers:
                    query = f'("{ticker}" OR "{ALIASES.get(ticker, (ticker,))[0]}") AND {QUERY_CONTEXT}'
                    parameters = {'q': query, 'from': start.isoformat(), 'to': end.isoformat(), 'language': 'en',
                                  'sortBy': 'publishedAt', 'searchIn': 'title', 'pageSize': live['news_page_size'], 'page': 1}
                    try:
                        snapshot = client.page(parameters, refresh=True)
                        payload = json.loads(snapshot['payload'])
                        successful_queries += 1
                        normalized, _ = normalize_news(payload, snapshot['retrieved_at'], config.tickers, snapshot['id'])
                        for record in normalized:
                            if timestamp(record['published_at']) <= timestamp(record['available_at']):
                                acquisition.put_text(record)
                        for article in payload['articles']:
                            try:
                                publisher_names[canonical_url(article['url'])] = str(article.get('source', {}).get('name') or 'News publisher')
                            except (ValueError, KeyError, TypeError):
                                continue
                        truncated += int(payload['totalResults'] > live['news_page_size'])
                    except (ContractError, ProviderError) as error:
                        issues.append(f'{ticker}: {error}')
                cutoff = now - timedelta(days=live['retention_days'])
                collected_at = now_fn()
                records = [row for row in acquisition.texts() if row['source'] == 'newsapi'
                           and row['ticker'] in config.tickers and cutoff <= timestamp(row['published_at']) <= collected_at
                           and timestamp(row['available_at']) <= collected_at and financial_headline(row['text'], row['ticker'])]
                records, _ = deduplicate_export(records)
            if records:
                settings_sentiment = SentimentConfig.load(root / 'configs/sentiment.toml')
                scorer = scorer_factory(settings_sentiment, root / 'data/cache/models/finbert', offline=True)
                with SentimentCache(root / 'data/cache/live_sentiment.sqlite', synthetic=False) as cache:
                    scores, _ = score_unique([row['text'] for row in records], scorer, cache, settings_sentiment.batch_size)
                with store.db:
                    for row in records:
                        article = {'id': row['record_id'], 'ticker': row['ticker'], 'title': row['text'], 'url': row['url'],
                                   'publisher': publisher_names.get(row['url'], urlsplit(row['url']).hostname),
                                   'published_at': row['published_at'], 'first_observed_at': row['available_at'],
                                   'sentiment': scores[normalize_text(row['text'])]}
                        store.db.execute('INSERT OR REPLACE INTO live_articles VALUES (?,?,?)',
                                         (article['id'], article['ticker'], json.dumps(article)))
            with store.db:
                for saved in store.db.execute('SELECT id,ticker,payload FROM live_articles').fetchall():
                    if not financial_headline(json.loads(saved['payload'])['title'], saved['ticker']):
                        store.db.execute('DELETE FROM live_articles WHERE id=? AND ticker=?', (saved['id'], saved['ticker']))
            with store.db:
                store.db.execute('DELETE FROM live_articles WHERE json_extract(payload,\'$.published_at\') < ?', (cutoff.isoformat(),))
            store.put('news_truncated_queries', truncated)
            if successful_queries:
                store.put('last_news_success_at', now_fn().isoformat())
        # Closed daily market prices refresh independently of news availability.
        from .calendar import generate_calendar
        factory = sessions_factory or generate_calendar
        observed = now_fn()
        sessions, _ = factory(datetime(2023, 10, 23).date(), (observed + timedelta(days=14)).date())
        calendar = {str(session.date): session for session in sessions}
        bars = {}
        with Store(root / 'data/cache/live_acquisition.sqlite') as acquisition:
            for ticker in config.tickers:
                attempt = None
                try:
                    attempt = acquisition.reserve('yfinance', f'live:{ticker}', live['price_daily_call_budget'], observed)
                    raw = price_fetch(ticker, str(sessions[0].date), (observed.date() + timedelta(days=1)).isoformat(), cache_dir=root / 'data/cache/yfinance')
                    acquisition.finish_attempt(attempt, 'ok')
                    available = now_fn()
                    for value in raw:
                        session = calendar.get(value['session'])
                        if not session or session.close_at + timedelta(minutes=1) > available:
                            continue
                        row = {'ticker': ticker, 'session': value['session'], **{key.lower(): value[key] for key in ('Open', 'High', 'Low', 'Close', 'Volume')},
                               'available_at': available.isoformat(), 'split_ratio': value.get('Stock Splits', 0) or 1}
                        PriceBar.from_dict(row)
                        bars[(ticker, value['session'])] = row
                except Exception as error:
                    if attempt is not None:
                        acquisition.finish_attempt(attempt, 'failed')
                    issues.append(f'{ticker} daily prices unavailable ({type(error).__name__}).')
        if bars:
            previous_prices = (store.get('market') or {}).get('prices', {})
            prices = {ticker: [{'date': day, 'close': row['close'], 'volume': row['volume']}
                                                   for (symbol, day), row in sorted(bars.items()) if symbol == ticker]
                      or previous_prices.get(ticker, []) for ticker in config.tickers}
            store.put('market', {'as_of': max(day for _, day in bars), 'observed_at': now_fn().isoformat(), 'prices': prices})
        store.put('issues', issues)
        status = 'setup_required' if not environment['NEWSAPI_API_KEY'] else 'partial' if issues else 'ready'
        store.put('status', status)
        store.put('last_completed_at', now_fn().isoformat())
        return {'status': status, 'articles': sum(1 for row in store.db.execute('SELECT ticker FROM live_articles') if row[0] in config.tickers),
                'issues': issues}
    except Exception as error:
        message = f'Refresh failed ({type(error).__name__}); previous successful results are retained.'
        store.put('issues', [message])
        store.put('status', 'error')
        return {'status': 'error', 'issues': [message]}
    finally:
        store.put('lease_until', '2000-01-01T00:00:00Z')
        store.close()


def snapshot(root=ROOT):
    root = Path(root)
    config = Config.load(root / 'configs/project.toml')
    store = LiveStore(root / 'data/cache/live.sqlite')
    try:
        cutoff = utc_now() - timedelta(days=7)
        articles = [article for row in store.db.execute('SELECT payload FROM live_articles ORDER BY json_extract(payload,\'$.published_at\') DESC')
                    if (article := json.loads(row[0]))['ticker'] in config.tickers and timestamp(article['published_at']) >= cutoff
                    and financial_headline(article['title'], article['ticker'])]
        market = store.get('market')
        if market:
            market['prices'] = {ticker: rows for ticker, rows in market['prices'].items() if ticker in config.tickers and rows}
        last = store.get('last_attempt_at')
        evaluation = store.get('price_evaluation')
        walkthrough = None
        if evaluation and evaluation.get('rows'):
            from .price_replay import eligible_news
            first = evaluation['rows'][0]
            eligible = eligible_news(store.get('price_training_news', []), first['ticker'], timestamp(first['decision_at']))
            walkthrough = max(eligible, key=lambda row: row['published_at'], default=None)
        return {'scope': 'news_only', 'tickers': list(config.tickers), 'provider': 'NewsAPI Developer', 'delay_hours': 24, 'refresh_hours': 6,
                'status': store.get('status', 'setup_required'), 'last_attempt_at': last,
                'last_completed_at': store.get('last_completed_at'), 'last_news_success_at': store.get('last_news_success_at'),
                'next_refresh_at': (timestamp(last) + INTERVAL).isoformat() if last else None,
                'issues': store.get('issues', []), 'truncated_queries': store.get('news_truncated_queries', 0),
                'articles': articles[:250], 'total_article_records': len(articles), 'sentiment': news_summary(articles, config.tickers),
                'market': market,
                'price_evaluation': evaluation, 'walkthrough_article': walkthrough,
                'price_evaluation_status': store.get('price_evaluation_status', 'not_run'),
                'price_evaluation_error': store.get('price_evaluation_error')}
    finally:
        store.close()


def serve(root=ROOT, port=8765):
    root = Path(root)
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path == '/api/live':
                try:
                    body = json.dumps(snapshot(root), allow_nan=False).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(body)
                except Exception:
                    self.send_error(503, 'Research feed temporarily unavailable')
                return
            super().do_GET()
    server = ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=str(root / 'website/dist')))
    print(f'News-only dashboard: http://127.0.0.1:{port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('refresh', 'compare', 'serve', 'status'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--target', help='Completed trading date for the price comparison, YYYY-MM-DD')
    args = parser.parse_args()
    if args.command == 'serve':
        serve(args.root, args.port)
    else:
        if args.command == 'compare':
            from .price_replay import run
            try:
                result = run(args.root, args.target)
            except Exception as error:
                result = {'status': 'error', 'issues': [f'Price comparison failed ({type(error).__name__}); previous results retained.']}
        else:
            result = refresh(args.root, args.force) if args.command == 'refresh' else snapshot(args.root)
        print(json.dumps(result, allow_nan=False))
        if result.get('status') == 'error':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
