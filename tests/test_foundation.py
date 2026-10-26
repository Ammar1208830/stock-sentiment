from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from sentiment_lab.cli import main, run
from sentiment_lab.config import Config
from sentiment_lab.contracts import ContractError, PriceBar, Session, TextRecord, timestamp
from sentiment_lab.io import read_inputs
from sentiment_lab.pipeline import assemble, chronological_split


FIXTURES = Path(__file__).resolve().parents[1] / 'sentiment_lab' / 'fixtures'


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.config = Config.load(FIXTURES / 'demo.toml')
        self.sessions, self.prices, self.texts = read_inputs(
            FIXTURES / 'sessions.csv', FIXTURES / 'prices.csv', FIXTURES / 'texts.jsonl')

    def assemble(self, **kwargs):
        return assemble(kwargs.get('config', self.config), kwargs.get('sessions', self.sessions),
                        kwargs.get('prices', self.prices), kwargs.get('texts', self.texts))

    def row(self, rows, ticker, day):
        return next(r for r in rows if r['ticker'] == ticker and r['target_session'] == day)

    def test_timestamp_rejects_naive_values(self):
        with self.assertRaisesRegex(ContractError, 'explicit UTC offset'):
            timestamp('2024-11-25T16:00:00')

    def test_timestamp_normalizes_offsets(self):
        self.assertEqual(timestamp('2024-11-25T16:00:00-05:00'), timestamp('2024-11-25T21:00:00Z'))

    def test_complete_demo_has_twelve_rows(self):
        rows, report = self.assemble()
        self.assertEqual(len(rows), 12)
        self.assertEqual(report['unique_eligible_texts_in_output'], 4)

    def test_cutoff_is_exclusive_and_delayed_text_is_excluded(self):
        rows, report = self.assemble()
        ids = {t['record_id'] for r in rows for t in r['texts']}
        self.assertNotIn('synthetic-cutoff', ids)
        self.assertNotIn('synthetic-delayed', ids)
        self.assertEqual(report['text_exclusions']['not_available_in_overnight_window'], 2)

    def test_dedup_keeps_earliest_available_record_regardless_of_input_order(self):
        rows, report = self.assemble(texts=list(reversed(self.texts)))
        row = self.row(rows, 'AAPL', '2024-11-26')
        self.assertEqual([r['record_id'] for r in row['texts']], ['synthetic-1'])
        self.assertEqual(report['text_exclusions']['duplicate_content'], 1)

    def test_previous_close_is_inclusive_and_intraday_text_is_excluded(self):
        at_close = replace(self.texts[0], record_id='at-close', text='At close',
                           published_at=self.sessions[0].close_at, available_at=self.sessions[0].close_at)
        intraday = replace(self.texts[0], record_id='intraday', text='During prior session',
                           published_at=timestamp('2024-11-25T20:59:00Z'))
        rows, _ = self.assemble(texts=[at_close, intraday])
        self.assertEqual([r['record_id'] for r in self.row(rows, 'AAPL', '2024-11-26')['texts']], ['at-close'])

    def test_report_accounts_for_every_input_text(self):
        _, report = self.assemble()
        self.assertEqual(report['input_texts'], sum(report['text_exclusions'].values())
                         + report['unique_eligible_texts_in_output'] + report['text_records_without_output_row'])

    def test_holiday_routes_to_next_actual_session(self):
        rows, _ = self.assemble()
        row = self.row(rows, 'MSFT', '2024-11-29')
        self.assertEqual(row['previous_session'], '2024-11-27')
        self.assertEqual(row['texts'][0]['record_id'], 'synthetic-holiday')

    def test_early_close_and_weekend_are_respected(self):
        rows, _ = self.assemble()
        row = self.row(rows, 'AAPL', '2024-12-02')
        self.assertEqual(row['window_start'], '2024-11-29T18:00:00+00:00')
        self.assertEqual(row['texts'][0]['record_id'], 'synthetic-halfday')
        self.assertEqual(self.row(rows, 'MSFT', '2024-12-02')['texts'][0]['record_id'], 'synthetic-weekend')

    def test_dst_uses_calendar_instants_instead_of_fixed_utc_open(self):
        sessions = [Session.from_dict(dict(session=day, open_at=opened, close_at=closed)) for day, opened, closed in [
            ('2024-03-07', '2024-03-07T14:30:00Z', '2024-03-07T21:00:00Z'),
            ('2024-03-08', '2024-03-08T14:30:00Z', '2024-03-08T21:00:00Z'),
            ('2024-03-11', '2024-03-11T13:30:00Z', '2024-03-11T20:00:00Z'),
            ('2024-03-12', '2024-03-12T13:30:00Z', '2024-03-12T20:00:00Z')]]
        prices = [replace(self.prices[0], session=s.date, available_at=s.close_at) for s in sessions]
        rows, _ = assemble(replace(self.config, tickers=('AAPL',)), sessions, prices, [])
        self.assertEqual(self.row(rows, 'AAPL', '2024-03-08')['decision_at'], '2024-03-08T14:25:00+00:00')
        self.assertEqual(self.row(rows, 'AAPL', '2024-03-11')['decision_at'], '2024-03-11T13:25:00+00:00')

    def test_missing_text_remains_missing_without_forward_fill(self):
        rows, _ = self.assemble()
        row = self.row(rows, 'AAPL', '2024-12-03')
        self.assertTrue(row['missing_text'])
        self.assertEqual(row['texts'], [])

    def test_assumed_historical_availability_requires_explicit_opt_in(self):
        rows, report = self.assemble()
        self.assertEqual(report['text_exclusions']['historical_assumption_disabled'], 1)
        rows, _ = self.assemble(config=replace(self.config, allow_historical_assumptions=True))
        self.assertEqual(self.row(rows, 'AAPL', '2024-12-03')['texts'][0]['availability_basis'], 'historical_assumption')

    def test_previous_close_is_feature_and_next_close_is_label(self):
        rows, _ = self.assemble()
        row = self.row(rows, 'AAPL', '2024-11-26')
        self.assertEqual(row['prior_close'], 100)
        self.assertEqual(row['label_up'], 1)
        self.assertAlmostEqual(row['target_return'], .02)

    def test_flat_return_is_class_zero(self):
        prices = [replace(p, close=100, low=98, high=106) if p.ticker == 'AAPL' else p for p in self.prices]
        rows, _ = self.assemble(prices=prices)
        self.assertEqual(self.row(rows, 'AAPL', '2024-11-26')['label_up'], 0)

    def test_same_date_all_tickers_in_same_partition(self):
        rows, report = self.assemble()
        dates = {}
        for row in rows:
            dates.setdefault(row['target_session'], set()).add(row['partition'])
        self.assertTrue(all(len(v) == 1 for v in dates.values()))
        self.assertLess(report['split']['train']['last_session'], report['split']['validation']['first_session'])
        self.assertLess(report['split']['validation']['last_session'], report['split']['test']['first_session'])

    def test_boundary_label_purging(self):
        rows, _ = self.assemble()
        # Nov 29 is the final training date, followed by Dec 2 validation.
        rows = [{k: v for k, v in r.items() if k != 'partition'} for r in rows]
        for row in rows:
            if row['target_session'] == '2024-11-29' and row['ticker'] == 'AAPL':
                row['label_available_at'] = '2024-12-02T14:25:00+00:00'
        result, report = chronological_split(rows, self.config)
        self.assertEqual(report['train']['purged_rows'], 1)
        self.assertFalse(any(r['ticker'] == 'AAPL' and r['target_session'] == '2024-11-29' for r in result))

    def test_insufficient_sessions_fail_instead_of_random_split(self):
        with self.assertRaisesRegex(ContractError, 'four unique'):
            self.assemble(sessions=self.sessions[:3])

    def test_price_before_close_rejected(self):
        prices = [replace(self.prices[0], available_at=self.sessions[0].open_at), *self.prices[1:]]
        with self.assertRaisesRegex(ContractError, 'before session close'):
            self.assemble(prices=prices)

    def test_unavailable_prior_bar_drops_sample(self):
        prices = [replace(self.prices[0], available_at=timestamp('2024-11-26T14:25:00Z')), *self.prices[1:]]
        rows, report = self.assemble(prices=prices)
        self.assertEqual(report['skipped_price_pairs']['prior_price_unavailable'], 1)
        self.assertFalse(any(r['ticker'] == 'AAPL' and r['target_session'] == '2024-11-26' for r in rows))

    def test_split_target_session_is_excluded(self):
        prices = [replace(p, split_ratio=2) if p.session == date(2024, 11, 26) and p.ticker == 'AAPL' else p for p in self.prices]
        rows, report = self.assemble(prices=prices)
        self.assertEqual(report['skipped_price_pairs']['target_split_session'], 1)
        self.assertFalse(any(r['ticker'] == 'AAPL' and r['target_session'] == '2024-11-26' for r in rows))

    def test_duplicate_price_or_record_id_rejected(self):
        with self.assertRaisesRegex(ContractError, 'Duplicate price'):
            self.assemble(prices=self.prices + self.prices[:1])
        with self.assertRaisesRegex(ContractError, 'Duplicate source'):
            self.assemble(texts=self.texts + self.texts[:1])

    def test_unknown_symbol_rejected(self):
        with self.assertRaisesRegex(ContractError, 'outside configured'):
            self.assemble(texts=[replace(self.texts[0], ticker='GOOG')])

    def test_calendar_duplicate_and_local_date_mismatch_rejected(self):
        with self.assertRaisesRegex(ContractError, 'unique exchange sessions'):
            self.assemble(sessions=self.sessions + self.sessions[:1])
        with self.assertRaisesRegex(ContractError, 'exchange-local session date'):
            self.assemble(sessions=[replace(self.sessions[0], date=date(2024, 11, 24)), *self.sessions[1:]])

    def test_nonfinite_or_inconsistent_price_rejected(self):
        template = dict(ticker='AAPL', session='2024-11-25', open='100', high='101', low='99',
                        close='100', volume='10', available_at='2024-11-25T21:01:00Z', split_ratio='1')
        for update in ({'close': 'NaN'}, {'close': '102'}, {'volume': '1.5'}, {'split_ratio': '0'}):
            with self.subTest(update=update), self.assertRaises(ContractError):
                PriceBar.from_dict({**template, **update})

    def test_text_contract_rejects_future_publication_and_empty_text(self):
        template = json.loads((FIXTURES / 'texts.jsonl').read_text().splitlines()[0])
        for update in ({'published_at': '2024-11-27T22:00:00Z'}, {'text': ' '}, {'source': 'unknown'},
                       {'availability_basis': 'unknown'}, {'retrieved_at': '2024-11-25T21:00:00Z'}):
            with self.subTest(update=update), self.assertRaises(ContractError):
                TextRecord.from_dict({**template, **update})

    def test_config_rejects_invalid_splits_and_misspelled_keys(self):
        for config in (replace(self.config, train_fraction=.9, validation_fraction=.2),
                       replace(self.config, decision_minutes_before_open=True),
                       replace(self.config, tickers=('AAPL', 'AAPL'))):
            with self.subTest(config=config), self.assertRaises(ContractError):
                config.validate()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.toml'
            path.write_text('sead = 42\n')
            with self.assertRaisesRegex(ContractError, 'Unknown configuration'):
                Config.load(path)
            path.write_text('tickers = [["AAPL"]]\n')
            with self.assertRaisesRegex(ContractError, 'Invalid ticker'):
                Config.load(path)

    def test_run_reproducible_and_features_separate_from_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / 'a', Path(folder) / 'b'
            paths = [FIXTURES / name for name in ('demo.toml', 'sessions.csv', 'prices.csv', 'texts.jsonl')]
            first, second = run(*paths, a, 'synthetic_demo'), run(*paths, b, 'synthetic_demo')
            self.assertEqual(first['run_fingerprint'], second['run_fingerprint'])
            self.assertEqual((a / 'windows.jsonl').read_bytes(), (b / 'windows.jsonl').read_bytes())
            features = json.loads((a / 'windows.jsonl').read_text().splitlines()[0])
            self.assertFalse({'label_up', 'target_return', 'label_available_at'} & features.keys())
            self.assertTrue(json.loads((a / 'manifest.json').read_text())['synthetic_data'])
            self.assertEqual(json.loads((a / 'labels.jsonl').read_text().splitlines()[0])['label_up'], 1)
            with self.assertRaisesRegex(ContractError, 'must be empty'):
                run(*paths, a, 'synthetic_demo')
            imported = Path(folder) / 'imported'
            run(*paths, imported, 'local_import_unverified')
            self.assertIsNone(json.loads((imported / 'manifest.json').read_text())['synthetic_data'])

    def test_cli_help_doctor_demo_and_validation_error(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            self.assertEqual(main(['doctor', '--config', str(FIXTURES / 'demo.toml')]), 0)
            self.assertEqual(main(['demo', '--out', str(Path(folder) / 'demo')]), 0)
            self.assertEqual(main(['doctor', '--config', str(Path(folder) / 'missing.toml')]), 2)
            self.assertIn('error:', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
