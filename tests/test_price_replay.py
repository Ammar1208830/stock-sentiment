from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import unittest

from sentiment_lab.contracts import ContractError
from sentiment_lab.indicators import PRICE_FEATURES
from sentiment_lab.price_replay import eligible_news, fit_comparison
from sentiment_lab.news_filter import financial_headline

UTC = timezone.utc
MODELING_AVAILABLE = importlib.util.find_spec('sklearn') is not None


class DelayedNewsTests(unittest.TestCase):
    def test_financial_relevance_excludes_video_tags_and_wrong_apple_company(self):
        self.assertFalse(financial_headline('The Distinguished Gentleman 1992 1080p AMZN WEB-DL H264', 'AMZN'))
        self.assertFalse(financial_headline('Apple Hospitality REIT shares rise after dividend', 'AAPL'))
        self.assertFalse(financial_headline('Apple packed the iPhone with repair tricks', 'AAPL'))
        self.assertFalse(financial_headline('Tramontina stainless steel stock pot at Amazon', 'AMZN'))
        self.assertFalse(financial_headline('In-Stock Alert: G.I. Joe figure at Amazon', 'AMZN'))
        self.assertTrue(financial_headline('Apple shares rise after stronger earnings', 'AAPL'))
        self.assertTrue(financial_headline('Amazon quarterly revenue beats expectations', 'AMZN'))

    def test_twenty_four_hour_delay_excludes_late_sunday_news_from_monday_cutoff(self):
        cutoff = datetime(2026, 10, 26, 13, 25, tzinfo=UTC)
        articles = [{'ticker': ticker, 'published_at': time} for ticker, time in [
            ('AAPL', '2026-10-25T13:24:59Z'), ('AAPL', '2026-10-25T13:25:00Z'),
            ('AAPL', '2026-10-25T22:00:00Z'), ('NVDA', '2026-10-25T12:00:00Z'),
            ('AAPL', '2026-10-21T12:00:00Z')]]
        rows = eligible_news(articles, 'AAPL', cutoff)
        self.assertEqual([row['published_at'] for row in rows], ['2026-10-25T13:24:59Z'])


@unittest.skipUnless(MODELING_AVAILABLE, 'requires the modeling environment')
class PriceComparisonTests(unittest.TestCase):
    def examples(self):
        result = []
        for i in range(7):
            day = datetime(2026, 10, 12, tzinfo=UTC) + timedelta(days=i)
            for stock in ('AAPL', 'NVDA', 'AMZN'):
                result.append({'ticker': stock, 'session': str(day.date()), 'decision_at': (day + timedelta(hours=13)).isoformat(),
                    'reference_session': str((day - timedelta(days=1)).date()), 'reference_close': 100.,
                    'actual_close': 101. + i / 10, 'news_count': 2, 'news_latest_published_at': None,
                    'label_available_at': (day + timedelta(hours=20)).isoformat(),
                    'values': {**{key: i / 100 for key in PRICE_FEATURES}, 'news_polarity': i / 10, 'news_count_log': 1.}})
        return result

    def test_target_actual_prices_cannot_influence_fit_or_prediction(self):
        examples = self.examples()
        now = datetime(2026, 10, 20, tzinfo=UTC)
        first = fit_comparison(examples, '2026-10-18', now)
        changed = deepcopy(examples)
        for row in changed:
            if row['session'] == '2026-10-18':
                row['actual_close'] = 9000.
        second = fit_comparison(changed, '2026-10-18', now)
        self.assertEqual(first['models'], second['models'])
        self.assertEqual([r['predicted_close'] for r in first['rows']], [r['predicted_close'] for r in second['rows']])
        self.assertNotEqual(first['mae'], second['mae'])
        for row in first['rows']:
            self.assertAlmostEqual(row['absolute_error'], abs(row['predicted_close'] - row['actual_close']))
            self.assertAlmostEqual(row['percentage_error'], row['absolute_error'] / row['actual_close'] * 100)
            self.assertEqual(row['baseline_close'], row['reference_close'])

    def test_training_labels_unavailable_at_cutoff_are_excluded(self):
        rows = self.examples()
        for row in rows:
            if row['session'] == '2026-10-17':
                row['label_available_at'] = '2026-10-18T20:00:00Z'
                row['actual_close'] = 9000.
        result = fit_comparison(rows, '2026-10-18', datetime(2026, 10, 20, tzinfo=UTC))
        self.assertEqual(result['training_end'], '2026-10-16')
        self.assertEqual(result['training_rows'], 15)

    def test_no_combined_model_claim_when_training_news_is_absent(self):
        rows = self.examples()
        for row in rows:
            row['news_count'] = 0
        with self.assertRaises(ContractError):
            fit_comparison(rows, '2026-10-18', datetime(2026, 10, 20, tzinfo=UTC))


if __name__ == '__main__':
    unittest.main()
