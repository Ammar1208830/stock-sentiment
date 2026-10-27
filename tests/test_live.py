from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import tempfile
import unittest

from sentiment_lab.live import LiveStore, news_summary, snapshot

UTC = timezone.utc


class LiveFeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        (self.root / 'configs').mkdir()
        (self.root / 'configs/project.toml').write_text('tickers = ["AAPL", "NVDA", "AMZN"]\n')
        self.store = LiveStore(self.root / 'data/cache/live.sqlite')
        self.addCleanup(self.store.close)

    def test_six_hour_due_check_and_overlapping_run_lease(self):
        now = datetime(2026, 10, 27, 1, tzinfo=UTC)
        self.assertTrue(self.store.claim(now))
        self.assertFalse(self.store.claim(now, force=True))
        self.store.put('lease_until', '2000-01-01T00:00:00Z')
        self.assertFalse(self.store.claim(now + timedelta(hours=5)))
        self.assertTrue(self.store.claim(now + timedelta(hours=6)))

    def test_missing_news_stays_missing_instead_of_neutral(self):
        row = news_summary([], ['AAPL'])[0]
        self.assertEqual(row['label'], 'unavailable')
        self.assertIsNone(row['polarity'])
        self.assertIsNone(row['neutral'])

    def test_public_feed_filters_removed_stocks_and_does_not_expose_local_key(self):
        secret = 'fake-test-key-not-for-provider-use'
        (self.root / '.env').write_text('NEWSAPI_API_KEY=' + secret)
        now = datetime.now(UTC).isoformat()
        for ticker in ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'TSLA'):
            article = {'ticker': ticker, 'title': ticker + ' shares rise after earnings', 'published_at': now, 'sentiment': {'positive': .6, 'neutral': .3, 'negative': .1, 'polarity': .5}}
            self.store.db.execute('INSERT INTO live_articles VALUES (?,?,?)', ('headline', ticker, json.dumps(article)))
        self.store.db.commit()
        self.store.put('market', {'prices': {ticker: [{'date': '2026-10-26', 'close': 100}] for ticker in ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'TSLA')}})
        result = snapshot(self.root)
        expected = {'AAPL', 'NVDA', 'AMZN'}
        for key in ('articles', 'sentiment'):
            self.assertEqual({row['ticker'] for row in result[key]}, expected)
        self.assertEqual(set(result['market']['prices']), expected)
        self.assertEqual(result['total_article_records'], 3)
        self.assertNotIn(secret, json.dumps(result))
        self.assertNotIn('backtest', result)
        self.assertNotIn('forecasts', result)


if __name__ == '__main__':
    unittest.main()
