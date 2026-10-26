from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from io import StringIO
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from sentiment_lab.acquisition import AcquisitionConfig, collect, export_dataset, feasibility
from sentiment_lab.calendar import generate_calendar, verify_calendar
from sentiment_lab.cli import main
from sentiment_lab.config import Config
from sentiment_lab.contracts import ContractError, TextRecord, timestamp
from sentiment_lab.io import file_hash, read_inputs
from sentiment_lab.normalization import (canonical_url, deduplicate_export, import_text,
                                         match_tickers, normalize_news, normalized_version)
from sentiment_lab.pipeline import assemble, text_windows
from sentiment_lab.providers import HTTPFailure, NewsClient, ProviderError, collect_news, collect_prices
from sentiment_lab.store import BudgetExceeded, Store, digest


FIXTURES = Path(__file__).resolve().parents[1] / 'sentiment_lab' / 'fixtures'
NOW = timestamp('2024-12-05T12:00:00Z')
CALENDAR_AVAILABLE = bool(importlib.util.find_spec('exchange_calendars'))


def payload(total=1, title='AAPL shares climb on earnings', url='https://example.org/a', published='2024-11-25T22:00:00Z'):
    return {'status': 'ok', 'totalResults': total,
            'articles': [{'title': title, 'url': url, 'publishedAt': published}] if total else []}


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.store = Store(self.path / 'cache.sqlite')
        self.addCleanup(self.store.__exit__, None, None, None)
        self.config = Config.load(FIXTURES / 'demo.toml')
        self.settings = AcquisitionConfig(start='2024-11-26', end='2024-12-04')
        self.sessions, self.prices, self.texts = read_inputs(FIXTURES / 'sessions.csv', FIXTURES / 'prices.csv', FIXTURES / 'texts.jsonl')

    def client(self, transport, **kwargs):
        return NewsClient(self.store, 'secret-never-save', kwargs.pop('settings', self.settings),
                          transport=transport, now=lambda: NOW, sleep=lambda _: None, **kwargs)

    def price_rows(self):
        rows = []
        for bar in self.prices:
            row = asdict(bar)
            row['session'] = str(bar.session)
            row['available_at'] = bar.available_at.isoformat()
            rows.append(row)
        return rows

    def test_acquisition_config_rejects_bad_dates_unknown_keys_and_ranges(self):
        for settings in (replace(self.settings, end='2020-01-01'), replace(self.settings, news_page_size=101),
                         replace(self.settings, http_attempts=True), replace(self.settings, minimum_source_coverage=float('nan'))):
            with self.subTest(settings=settings), self.assertRaises(ContractError):
                settings.validate()
        path = self.path / 'bad.toml'
        path.write_text('unknown_key = 1\n')
        with self.assertRaisesRegex(ContractError, 'Unknown acquisition'):
            AcquisitionConfig.load(path)

    def test_sqlite_budget_survives_restart_and_counts_failed_attempts(self):
        path = self.path / 'budget.sqlite'
        with Store(path) as store:
            attempt = store.reserve('newsapi', 'request', 1, NOW)
            store.finish_attempt(attempt, 'http_429')
        with Store(path) as store:
            with self.assertRaises(BudgetExceeded):
                store.reserve('newsapi', 'request2', 1, NOW)
            self.assertEqual(store.usage()[0]['attempts'], 1)

    def test_cache_rejects_mixed_universe_or_synthetic_mode(self):
        self.store.bind(self.config.to_dict())
        with self.assertRaises(ContractError):
            self.store.bind(self.config.to_dict(), synthetic=True)

    def test_news_retry_backoff_and_checkpoint_no_credential_persistence(self):
        calls, delays = [], []
        def transport(params, key, timeout):
            calls.append(params)
            if len(calls) == 1:
                raise HTTPFailure(429, 4)
            return payload()
        client = self.client(transport)
        client.sleep = delays.append
        first = client.page({'page': 1})
        second = client.page({'page': 1})
        self.assertEqual(first['id'], second['id'])
        self.assertEqual(len(calls), 2)
        self.assertEqual(delays, [4])
        dump = '\n'.join(self.store.db.iterdump())
        self.assertNotIn('secret-never-save', dump)
        self.assertEqual(sum(r['attempts'] for r in self.store.usage()), 2)

    def test_news_auth_failure_does_not_retry(self):
        calls = []
        def transport(*args):
            calls.append(1)
            raise HTTPFailure(401)
        with self.assertRaisesRegex(ProviderError, '401'):
            self.client(transport).page({'page': 1})
        self.assertEqual(len(calls), 1)

    def test_news_network_failure_has_bounded_attempts(self):
        calls = []
        def transport(*args):
            calls.append(1)
            raise HTTPFailure(0)
        with self.assertRaises(ProviderError):
            self.client(transport).page({'page': 1})
        self.assertEqual(len(calls), 3)

    def test_news_failed_response_is_not_cached(self):
        with self.assertRaises(ProviderError):
            self.client(lambda *args: {'status': 'error', 'message': 'private'}).page({'page': 1})
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM snapshots').fetchone()[0], 0)

    def test_news_refresh_creates_new_observation_without_erasing_prior_snapshot(self):
        client = self.client(lambda *args: payload())
        one = client.page({'page': 1})
        client.now = lambda: timestamp('2024-12-05T12:01:00Z')
        two = client.page({'page': 1}, refresh=True)
        self.assertNotEqual(one['id'], two['id'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM snapshots').fetchone()[0], 2)

    def test_news_pagination_truncation_and_resume_use_original_observation(self):
        settings = replace(self.settings, news_lookback_days=1, news_page_size=1, news_max_pages_per_day=2)
        sessions = [replace(self.sessions[0], date=date(2024, 12, 4)), replace(self.sessions[-1], date=date(2024, 12, 4))]
        calls = []
        def transport(parameters, *args):
            calls.append(parameters['page'])
            return payload(total=3, url=f"https://example.org/{parameters['page']}")
        client = self.client(transport, settings=settings)
        config = replace(self.config, tickers=('AAPL',))
        first = collect_news(client, config, settings, sessions)
        second = collect_news(client, config, settings, sessions)
        self.assertEqual(first['status'], 'partial')
        self.assertEqual(first['truncated_queries'], 1)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(second['pages'], 2)
        self.assertTrue(all(r['available_at'] == NOW.isoformat() for r in self.store.texts()))

    def test_news_budget_failure_keeps_partial_page_progress(self):
        settings = replace(self.settings, news_daily_request_budget=1, news_page_size=1, news_lookback_days=1)
        client = self.client(lambda *args: payload(total=3), settings=settings)
        sessions = [replace(self.sessions[0], date=date(2024, 12, 4)), replace(self.sessions[-1], date=date(2024, 12, 4))]
        result = collect_news(client, replace(self.config, tickers=('AAPL',)), settings, sessions)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['pages'], 1)
        self.assertEqual(len(self.store.texts()), 1)

    def test_missing_key_is_reported_without_transport_calls(self):
        client = self.client(lambda *args: self.fail('Network should not be called'))
        client.api_key = ''
        self.assertEqual(collect_news(client, self.config, self.settings, self.sessions)['status'], 'unavailable')

    def test_canonical_url_and_conservative_ticker_matching(self):
        self.assertEqual(canonical_url('https://EXAMPLE.org/a?utm_source=x&b=2#part'), 'https://example.org/a?b=2')
        self.assertEqual(match_tickers('apple pie recipes from the amazon rainforest', self.config.tickers), [])
        self.assertEqual(match_tickers('Apple shares rise with Microsoft earnings', self.config.tickers), ['AAPL', 'MSFT'])
        self.assertEqual(match_tickers('$AAPL and MSFT', self.config.tickers), ['AAPL', 'MSFT'])
        with self.assertRaises(ContractError):
            canonical_url('https://user:password@example.org/')

    def test_news_normalization_rejects_removed_invalid_and_ambiguous_articles(self):
        data = payload()
        data['articles'] += [{'title': '[Removed]'}, {'title': 'apple pie', 'url': 'https://example.org/b'},
                             {'title': 'AAPL earnings', 'url': 'invalid'}]
        rows, excluded = normalize_news(data, NOW.isoformat(), self.config.tickers, 'snapshot')
        self.assertEqual(len(rows), 1)
        self.assertEqual(excluded, {'removed': 1, 'unmatched_or_ambiguous': 1, 'invalid_article': 1})
        self.assertEqual(rows[0]['availability_basis'], 'observed')
        self.assertEqual(rows[0]['available_at'], NOW.isoformat())

    def test_first_seen_is_preserved_but_changed_content_gets_new_version(self):
        rows, _ = normalize_news(payload(), '2024-11-25T22:01:00Z', self.config.tickers, 'one')
        self.store.put_text(rows[0])
        repeated, _ = normalize_news(payload(), '2024-11-25T23:00:00Z', self.config.tickers, 'two')
        self.store.put_text(repeated[0])
        changed, _ = normalize_news(payload(title='AAPL shares fall after earnings update'), '2024-11-26T14:26:00Z', self.config.tickers, 'three')
        self.store.put_text(changed[0])
        stored = self.store.texts()
        self.assertEqual(len(stored), 2)
        first = next(r for r in stored if r['record_id'] == rows[0]['record_id'])
        self.assertEqual(first['available_at'], '2024-11-25T22:01:00+00:00')
        self.assertEqual(first['retrieved_at'], '2024-11-25T23:00:00+00:00')
        self.assertEqual(first['first_snapshot_id'], 'one')
        self.assertEqual(first['snapshot_id'], 'two')
        buckets, _ = text_windows(self.config, self.sessions, [TextRecord.from_dict(r) for r in stored])
        self.assertEqual([r.record_id for r in buckets[('AAPL', date(2024, 11, 26))]], [rows[0]['record_id']])

    def test_latest_version_before_cutoff_counts_once(self):
        first = replace(self.texts[0], origin_id='same-story')
        second = replace(first, record_id='updated', text='Updated article before cutoff',
                         available_at=timestamp('2024-11-26T14:24:00Z'), retrieved_at=timestamp('2024-11-26T14:24:00Z'))
        rows, report = assemble(self.config, self.sessions, self.prices, [first, second])
        target = next(r for r in rows if r['ticker'] == 'AAPL' and r['target_session'] == '2024-11-26')
        self.assertEqual(target['text_count'], 1)
        self.assertEqual(target['texts'][0]['record_id'], 'updated')
        self.assertEqual(report['text_exclusions']['superseded_content_version'], 1)

    def test_cross_source_syndication_dedup_retains_lineage(self):
        template = json.loads((FIXTURES / 'texts.jsonl').read_text().splitlines()[0])
        first = normalized_version(template, 'news-story')
        other = normalized_version({**template, 'source': 'reddit'}, 'post')
        selected, audit = deduplicate_export([first, other])
        self.assertEqual(len(selected), 1)
        self.assertEqual(sum(r['duplicate_of'] is not None for r in audit), 1)

    def test_import_requires_provenance_and_does_not_store_author_fields(self):
        original = json.loads((FIXTURES / 'texts.jsonl').read_text().splitlines()[0])
        path, provenance = self.path / 'input.jsonl', self.path / 'provenance.json'
        path.write_text(json.dumps({**original, 'source': 'reddit', 'author': 'private-author', 'origin_id': 'post-123'}) + '\n')
        provenance.write_text(json.dumps({'source': 'reddit', 'access_basis': 'generic_api', 'reference': 'example', 'synthetic_data': False}))
        with self.assertRaises(ContractError):
            import_text(path, provenance, self.config, self.store)
        provenance.write_text(json.dumps({'source': 'reddit', 'access_basis': 'approved_research_export', 'reference': 'local-test-attestation', 'synthetic_data': False}))
        result = import_text(path, provenance, self.config, self.store)
        self.assertEqual(result['records'], 1)
        self.assertNotIn('private-author', '\n'.join(self.store.db.iterdump()))
        self.assertEqual(self.store.purge('reddit', 'post-123'), 1)
        self.assertEqual(self.store.texts(), [])

    def test_import_validates_whole_batch_before_writing(self):
        original = json.loads((FIXTURES / 'texts.jsonl').read_text().splitlines()[0])
        path, provenance = self.path / 'input.jsonl', self.path / 'provenance.json'
        path.write_text(json.dumps(original) + '\n' + json.dumps({**original, 'text': ''}) + '\n')
        provenance.write_text(json.dumps({'source': 'newsapi', 'access_basis': 'licensed_export', 'reference': 'test', 'synthetic_data': False}))
        with self.assertRaises(ContractError):
            import_text(path, provenance, self.config, self.store)
        self.assertEqual(self.store.texts(), [])

    def test_price_checkpoint_explicit_availability_and_actions(self):
        raw = [{'session': str(s.date), 'Open': 99, 'High': 102, 'Low': 98, 'Close': 100,
                'Volume': 10, 'Stock Splits': 2 if i == 2 else 0, 'Dividends': .2}
               for i, s in enumerate(self.sessions)]
        calls = []
        def fetch(symbol, start, end, **kwargs):
            calls.append((symbol, start, end))
            return raw
        config = replace(self.config, tickers=('AAPL',))
        prices, report = collect_prices(self.store, config, self.settings, self.sessions, fetch=fetch, now=lambda: NOW, sleep=lambda _: None)
        cached, _ = collect_prices(self.store, config, self.settings, self.sessions, fetch=fetch, now=lambda: NOW)
        self.assertEqual(prices, cached)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], '2024-12-05')
        self.assertEqual(prices[0]['available_at'], '2024-11-25T21:01:00+00:00')
        self.assertEqual(prices[2]['split_ratio'], 2)
        self.assertEqual(prices[0]['dividends'], .2)
        self.assertEqual(report['status'], 'complete')

    def test_price_retry_is_bounded_and_failure_messages_do_not_expose_provider_text(self):
        calls = []
        def fetch(*args, **kwargs):
            calls.append(1)
            raise RuntimeError('secret-provider-message')
        prices, report = collect_prices(self.store, replace(self.config, tickers=('AAPL',)), self.settings, self.sessions,
                                        fetch=fetch, now=lambda: NOW, sleep=lambda _: None)
        self.assertEqual(prices, [])
        self.assertEqual(len(calls), 3)
        self.assertNotIn('secret-provider-message', json.dumps(report))

    def test_price_observed_mode_does_not_backdate_history(self):
        raw = [{'session': str(s.date), 'Open': 99, 'High': 102, 'Low': 98, 'Close': 100, 'Volume': 10} for s in self.sessions]
        settings = replace(self.settings, price_availability='observed')
        prices, _ = collect_prices(self.store, self.config, settings, self.sessions, fetch=lambda *a, **k: raw, now=lambda: NOW)
        self.assertTrue(all(r['available_at'] == NOW.isoformat() for r in prices))
        gate, _ = feasibility(self.config, settings, self.sessions, prices, [], {})
        self.assertEqual(gate['shared_usable_sessions'], 0)

    def test_price_offline_miss_does_not_call_provider(self):
        prices, report = collect_prices(self.store, self.config, self.settings, self.sessions, offline=True,
                                        fetch=lambda *a, **k: self.fail('Provider should not be called'))
        self.assertEqual(prices, [])
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(self.store.usage(), [])

    def test_gate_counts_eligible_windows_and_both_sources(self):
        rows = [json.loads(line) for line in (FIXTURES / 'texts.jsonl').read_text().splitlines()]
        gate, coverage = feasibility(self.config, self.settings, self.sessions, self.price_rows(), rows, {})
        self.assertEqual(gate['decision'], 'engineering_demonstration_only')
        self.assertEqual(gate['shared_usable_sessions'], 6)
        self.assertEqual(sum(r['newsapi_count'] for r in coverage), 2)
        self.assertEqual(sum(r['reddit_count'] for r in coverage), 2)
        self.assertEqual(gate['text_exclusions']['not_available_in_overnight_window'], 2)

    def test_gate_full_reduced_and_synthetic_decisions(self):
        settings = replace(self.settings, minimum_usable_sessions=3, minimum_test_sessions=1, minimum_source_coverage=1)
        texts = []
        for previous, target in zip(self.sessions, self.sessions[1:]):
            at = previous.close_at.isoformat()
            for symbol in self.config.tickers:
                for source in ('newsapi', 'reddit'):
                    texts.append({'record_id': f'{source}-{symbol}-{target.date}', 'source': source, 'ticker': symbol,
                                  'text': f'{symbol} {source} {target.date}', 'published_at': at, 'available_at': at,
                                  'retrieved_at': at, 'availability_basis': 'observed'})
        full, _ = feasibility(self.config, settings, self.sessions, self.price_rows(), texts, {})
        reduced, _ = feasibility(self.config, settings, self.sessions, self.price_rows(), [r for r in texts if r['source'] == 'newsapi'], {})
        synthetic, _ = feasibility(self.config, settings, self.sessions, self.price_rows(), texts, {}, synthetic=True)
        self.assertEqual(full['decision'], 'full_study_feasible')
        self.assertEqual(reduced['decision'], 'reduced_news_only_study_feasible')
        self.assertEqual(synthetic['decision'], 'engineering_demonstration_only')

    def test_no_prices_still_produces_honest_feasibility_artifacts(self):
        result = export_dataset(self.path / 'out', self.config, self.settings, self.sessions,
                                {'version': 'test'}, [], self.store, {'prices': {'status': 'unavailable'}})
        self.assertEqual(result['price_bars'], 0)
        self.assertEqual(result['feasibility'], 'engineering_demonstration_only')
        manifest = json.loads((self.path / 'out' / 'manifest.json').read_text())
        self.assertIsNone(manifest['synthetic_data'])
        for name, expected in manifest['outputs'].items():
            self.assertEqual(file_hash(self.path / 'out' / name), expected)

    @unittest.skipUnless(CALENDAR_AVAILABLE, 'Install acquisition dependencies for calendar integration')
    def test_calendar_holidays_halfdays_and_complete_verification(self):
        sessions, metadata = generate_calendar(date(2024, 11, 26), date(2024, 12, 4))
        self.assertEqual(sessions, self.sessions)
        self.assertEqual(verify_calendar(sessions, date(2024, 11, 26), date(2024, 12, 4)), metadata)
        with self.assertRaisesRegex(ContractError, 'incomplete'):
            verify_calendar(sessions[1:], date(2024, 11, 26), date(2024, 12, 4))

    @unittest.skipUnless(CALENDAR_AVAILABLE, 'Install acquisition dependencies for calendar integration')
    def test_calendar_dst_and_weekend_start(self):
        sessions, _ = generate_calendar(date(2024, 3, 9), date(2024, 3, 12))
        self.assertEqual(sessions[0].date, date(2024, 3, 8))
        self.assertEqual(sessions[0].open_at.hour, 14)
        self.assertEqual(sessions[1].open_at.hour, 13)
        with self.assertRaisesRegex(ContractError, 'No exchange sessions'):
            generate_calendar(date(2024, 3, 9), date(2024, 3, 10))

    @unittest.skipUnless(CALENDAR_AVAILABLE, 'Install acquisition dependencies for CLI integration')
    def test_m2_cli_calendar_demo_and_offline_collection(self):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(main(['calendar', '--start', '2024-11-26', '--end', '2024-12-04', '--out', str(self.path / 'calendar')]), 0)
            self.assertEqual(main(['verify-calendar', '--start', '2024-11-26', '--end', '2024-12-04', '--sessions', str(self.path / 'calendar' / 'sessions.csv')]), 0)
            self.assertEqual(main(['m2-demo', '--cache', str(self.path / 'demo.sqlite'), '--out', str(self.path / 'demo')]), 0)
            settings = self.path / 'acquisition.toml'
            settings.write_text('start = "2024-11-26"\nend = "2024-12-04"\n')
            self.assertEqual(main(['collect', '--offline', '--require-feasible', '--config', str(FIXTURES / 'demo.toml'),
                '--acquisition', str(settings), '--cache', str(self.path / 'offline.sqlite'), '--out', str(self.path / 'offline')]), 3)
        self.assertTrue(json.loads((self.path / 'demo' / 'manifest.json').read_text())['synthetic_data'])


if __name__ == '__main__':
    unittest.main()
