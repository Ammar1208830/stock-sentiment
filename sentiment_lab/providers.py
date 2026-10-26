"""Bounded provider clients with persistent, credential-free checkpoints."""
from dataclasses import dataclass
from datetime import timedelta
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .contracts import ContractError, PriceBar, finite_number, timestamp
from .normalization import ALIASES, normalize_news
from .store import BudgetExceeded, Store, digest, utc_now


class ProviderError(ContractError):
    pass


class HTTPFailure(Exception):
    def __init__(self, status: int, retry_after: float = 0):
        self.status, self.retry_after = status, retry_after


def news_transport(parameters: dict, api_key: str, timeout: int) -> dict:
    request = Request('https://newsapi.org/v2/everything?' + urlencode(parameters),
                      headers={'X-Api-Key': api_key, 'User-Agent': 'stock-sentiment-lab/0.2'})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        try:
            retry_after = float(exc.headers.get('Retry-After', '0'))
        except ValueError:
            retry_after = 0
        raise HTTPFailure(exc.code, retry_after) from None
    except (URLError, TimeoutError, OSError):
        raise HTTPFailure(0) from None
    except (ValueError, UnicodeError):
        raise ProviderError('NewsAPI returned invalid JSON') from None


@dataclass
class NewsClient:
    store: Store
    api_key: str
    settings: object
    transport: object = news_transport
    now: object = utc_now
    sleep: object = time.sleep

    def page(self, parameters: dict, refresh: bool = False) -> dict:
        key = digest(['newsapi-v1', parameters])
        cached = self.store.latest(key)
        if cached and not refresh:
            return cached
        for attempt_number in range(self.settings.http_attempts):
            attempt = self.store.reserve('newsapi', key, self.settings.news_daily_request_budget, self.now())
            try:
                payload = self.transport(parameters, self.api_key, self.settings.http_timeout_seconds)
                if not isinstance(payload, dict) or payload.get('status') != 'ok':
                    raise ProviderError('NewsAPI rejected the request; check access, plan and date range')
                if not isinstance(payload.get('articles'), list) or type(payload.get('totalResults')) is not int or payload['totalResults'] < 0:
                    raise ProviderError('NewsAPI response has an invalid article listing')
                snapshot = self.store.save_snapshot('newsapi', key, parameters, payload, self.now().isoformat())
                self.store.finish_attempt(attempt, 'ok')
                return snapshot
            except HTTPFailure as exc:
                self.store.finish_attempt(attempt, f'http_{exc.status}')
                if exc.status not in {0, 429, 500, 502, 503, 504} or attempt_number + 1 == self.settings.http_attempts:
                    raise ProviderError(f'NewsAPI request failed (HTTP {exc.status}); checkpoint retained') from None
                self.sleep(min(30, max(exc.retry_after, 2 ** attempt_number)))
            except ProviderError:
                self.store.finish_attempt(attempt, 'invalid_response')
                raise
        raise ProviderError('NewsAPI exhausted attempts')


def collect_news(client: NewsClient, config, settings, sessions, refresh: bool = False) -> dict:
    if not client.api_key:
        return {'status': 'unavailable', 'reason': 'NEWSAPI_API_KEY is not set', 'pages': 0}
    now = client.now()
    first_day = max(sessions[0].date, now.date() - timedelta(days=settings.news_lookback_days))
    last_day = min(sessions[-1].date, now.date() - timedelta(days=1))
    if first_day > last_day:
        return {'status': 'unavailable', 'reason': 'Study dates are outside configured news lookback', 'pages': 0}
    result = {'status': 'complete', 'pages': 0, 'normalized_records': 0, 'exclusions': {},
              'truncated_queries': 0, 'start': first_day.isoformat(), 'end': last_day.isoformat(),
              'scope': 'bounded lookback only; not a complete multi-year news archive'}
    day = first_day
    try:
        while day <= last_day:
            for symbol in config.tickers:
                alias = ALIASES.get(symbol, (symbol,))[0]
                base = {'q': f'"{symbol}" OR "{alias}"', 'from': day.isoformat() + 'T00:00:00Z',
                        'to': day.isoformat() + 'T23:59:59Z', 'language': 'en',
                        'sortBy': 'publishedAt', 'pageSize': settings.news_page_size}
                for page in range(1, settings.news_max_pages_per_day + 1):
                    snapshot = client.page({**base, 'page': page}, refresh=refresh)
                    payload = json.loads(snapshot['payload'])
                    records, excluded = normalize_news(payload, snapshot['retrieved_at'], config.tickers, snapshot['id'])
                    for row in records:
                        client.store.put_text(row)
                    result['pages'] += 1
                    result['normalized_records'] += len(records)
                    for key, value in excluded.items():
                        result['exclusions'][key] = result['exclusions'].get(key, 0) + value
                    if page * settings.news_page_size >= payload['totalResults']:
                        break
                    if not payload['articles'] or page == settings.news_max_pages_per_day:
                        result['truncated_queries'] += 1
                        break
            day += timedelta(days=1)
    except (BudgetExceeded, ProviderError) as exc:
        result.update(status='partial', reason=str(exc))
    if result['truncated_queries']:
        result['status'] = 'partial'
    return result


def yahoo_history(symbol: str, start: str, end_exclusive: str, cache_dir=None) -> list[dict]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ProviderError('Install requirements-acquisition.lock for yfinance support') from exc
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
    frame = yf.Ticker(symbol).history(start=start, end=end_exclusive, interval='1d',
        auto_adjust=False, back_adjust=False, actions=True, repair=False, raise_errors=True,
        timeout=20)
    if frame.empty:
        raise ProviderError('Yahoo returned an empty price history')
    rows = []
    for index, value in frame.iterrows():
        rows.append({'session': index.date().isoformat(), **{name: float(value[name]) for name in ('Open', 'High', 'Low', 'Close', 'Volume')},
                     'Stock Splits': float(value.get('Stock Splits', 0)), 'Dividends': float(value.get('Dividends', 0))})
    return rows


def collect_prices(store: Store, config, settings, sessions, refresh=False, fetch=yahoo_history,
                   now=utc_now, sleep=time.sleep, cache_dir=None, offline=False) -> tuple[list[dict], dict]:
    prices, report = [], {'status': 'complete', 'tickers': {}, 'availability_basis': settings.price_availability,
                         'price_basis': 'Yahoo vendor OHLC; auto_adjust=false, back_adjust=false, repair=false',
                         'caveat': 'Vendor history can contain retrospective split adjustments/revisions; not point-in-time verified.'}
    calendar = {s.date.isoformat(): s for s in sessions}
    for symbol in config.tickers:
        params = {'ticker': symbol, 'start': sessions[0].date.isoformat(),
                  'end_exclusive': (sessions[-1].date + timedelta(days=1)).isoformat(),
                  'auto_adjust': False, 'back_adjust': False, 'repair': False, 'actions': True}
        key = digest(['yfinance-v1', params])
        snapshot = None if refresh else store.latest(key)
        error = None
        if snapshot is None and not offline:
            for retry in range(settings.http_attempts):
                try:
                    attempt = store.reserve('yfinance', key, settings.price_daily_call_budget, now())
                except BudgetExceeded as exc:
                    error = str(exc)
                    break
                try:
                    raw = fetch(symbol, params['start'], params['end_exclusive'], cache_dir=cache_dir)
                    if not raw:
                        raise ProviderError('Empty price history')
                    snapshot = store.save_snapshot('yfinance', key, params, raw, now().isoformat())
                    store.finish_attempt(attempt, 'ok')
                    break
                except Exception as exc:
                    store.finish_attempt(attempt, 'provider_error')
                    # Never print provider exception text: URLs/messages may contain sensitive data.
                    error = f'yfinance failed ({type(exc).__name__}); inspect network/provider access'
                    if retry + 1 < settings.http_attempts:
                        sleep(min(30, 2 ** retry))
        if snapshot is None:
            report['tickers'][symbol] = {'status': 'unavailable', 'reason': error or 'No cached snapshot; offline mode performs no requests'}
            report['status'] = 'partial'
            continue
        count, rejected, open_sessions = 0, 0, 0
        for raw in json.loads(snapshot['payload']):
            try:
                session = calendar.get(raw['session'])
                if session is None:
                    raise ContractError('Price outside generated calendar')
                retrieved = timestamp(snapshot['retrieved_at'])
                if session.close_at > retrieved:
                    open_sessions += 1
                    continue
                available = retrieved if settings.price_availability == 'observed' else session.close_at + timedelta(minutes=settings.price_latency_minutes)
                row = {'ticker': symbol, 'session': raw['session'],
                       **{k.lower(): raw[k] for k in ('Open', 'High', 'Low', 'Close', 'Volume')},
                       'split_ratio': raw.get('Stock Splits', 0) or 1,
                       'available_at': available.isoformat(), 'retrieved_at': retrieved.isoformat(),
                       'availability_basis': settings.price_availability,
                       'dividends': finite_number(raw.get('Dividends', 0), 'dividends'),
                       'snapshot_id': snapshot['id']}
                PriceBar.from_dict(row)
                prices.append(row)
                count += 1
            except (ValueError, TypeError, KeyError):
                rejected += 1
        report['tickers'][symbol] = {'status': 'complete' if count and not rejected else 'partial',
                                     'bars': count, 'rejected': rejected, 'unfinished_sessions': open_sessions,
                                     'snapshot_id': snapshot['id']}
        if rejected or not count:
            report['status'] = 'partial'
    return sorted(prices, key=lambda r: (r['session'], r['ticker'])), report
