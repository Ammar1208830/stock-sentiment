"""Assemble audited prediction windows and labels, never fit a model."""

from bisect import bisect_right
from collections import Counter
from datetime import timedelta
import hashlib
from zoneinfo import ZoneInfo

from .config import Config
from .contracts import ContractError, PriceBar, Session, TextRecord


def content_key(record: TextRecord) -> tuple[str, str, str]:
    normalized = " ".join(record.text.casefold().split())
    return record.source, record.ticker, hashlib.sha256((record.published_at.date().isoformat() + normalized).encode()).hexdigest()


def chronological_split(rows: list[dict], config: Config) -> tuple[list[dict], dict]:
    """Split whole dates across all tickers and purge unresolved boundary labels."""
    dates = sorted({row["target_session"] for row in rows})
    if len(dates) < 3:
        raise ContractError("At least three usable target sessions are required for train/validation/test")
    train_end = max(1, int(len(dates) * config.train_fraction))
    val_end = max(train_end + 1, int(len(dates) * (config.train_fraction + config.validation_fraction)))
    val_end = min(val_end, len(dates) - 1)
    groups = {"train": dates[:train_end], "validation": dates[train_end:val_end], "test": dates[val_end:]}
    assignment = {day: name for name, days in groups.items() for day in days}
    cutoffs = {name: min(row["decision_at"] for row in rows if row["target_session"] in days)
               for name, days in groups.items()}
    next_cutoff = {"train": cutoffs["validation"], "validation": cutoffs["test"]}
    result, purged = [], Counter()
    for row in rows:
        partition = assignment[row["target_session"]]
        if partition in next_cutoff and row["label_available_at"] >= next_cutoff[partition]:
            purged[partition] += 1
            continue
        result.append({**row, "partition": partition})
    if any(not any(row["partition"] == name for row in result) for name in groups):
        raise ContractError("A partition is empty after purging labels unavailable at the next boundary")
    report = {name: {"first_session": days[0], "last_session": days[-1],
                     "rows": sum(row["partition"] == name for row in result),
                     "purged_rows": purged[name]} for name, days in groups.items()}
    return result, report


def text_windows(config: Config, sessions: list[Session], texts: list[TextRecord]) -> tuple[dict, Counter]:
    excluded = Counter()
    deduped, ids = {}, set()
    for record in texts:
        if record.ticker not in config.tickers:
            raise ContractError(f"Text ticker outside configured universe: {record.ticker}")
        key = (record.source, record.ticker, record.record_id)
        if key in ids:
            raise ContractError("Duplicate source/ticker/record_id; version records explicitly")
        ids.add(key)
        if record.availability_basis == "historical_assumption" and not config.allow_historical_assumptions:
            excluded["historical_assumption_disabled"] += 1
            continue
        key = content_key(record)
        if key not in deduped or (record.available_at, record.record_id) < (deduped[key].available_at, deduped[key].record_id):
            deduped[key] = record
    excluded["duplicate_content"] = len(texts) - excluded["historical_assumption_disabled"] - len(deduped)
    candidates = list(deduped.values())
    # Assign each text once using actual close windows, including holidays/half-days.
    closes = [session.close_at for session in sessions]
    buckets = {}
    for record in candidates:
        target_index = bisect_right(closes, record.published_at)
        if not 1 <= target_index < len(sessions):
            excluded["outside_session_range"] += 1
            continue
        target = sessions[target_index]
        decision = target.open_at - timedelta(minutes=config.decision_minutes_before_open)
        if record.published_at >= decision or record.available_at >= decision:
            excluded["not_available_in_overnight_window"] += 1
            continue
        buckets.setdefault((record.ticker, target.date), []).append(record)
    for key, records in buckets.items():
        latest = {}
        for record in sorted(records, key=lambda r: (r.available_at, r.record_id)):
            origin = (record.source, record.origin_id or record.record_id)
            if origin in latest:
                excluded['superseded_content_version'] += 1
            latest[origin] = record
        buckets[key] = list(latest.values())
    return buckets, excluded


def assemble(config: Config, sessions: list[Session], prices: list[PriceBar], texts: list[TextRecord]) -> tuple[list[dict], dict]:
    config.validate()
    sessions = sorted(sessions, key=lambda s: s.date)
    if len(sessions) < 4 or len({s.date for s in sessions}) != len(sessions):
        raise ContractError("Provide at least four unique exchange sessions")
    local = ZoneInfo(config.timezone)
    for session in sessions:
        if session.open_at.astimezone(local).date() != session.date or session.close_at.astimezone(local).date() != session.date:
            raise ContractError("Session timestamps must match the exchange-local session date")
        if session.date.weekday() >= 5:
            raise ContractError("XNYS sessions cannot fall on weekends")
    if any(a.close_at >= b.open_at for a, b in zip(sessions, sessions[1:])):
        raise ContractError("Exchange sessions overlap")
    calendar = {s.date: s for s in sessions}
    bars = {}
    for bar in prices:
        if bar.ticker not in config.tickers:
            raise ContractError(f"Price ticker outside configured universe: {bar.ticker}")
        if bar.session not in calendar:
            raise ContractError(f"Price session missing from calendar: {bar.session}")
        if bar.available_at < calendar[bar.session].close_at:
            raise ContractError("Daily closing prices cannot be available before session close")
        key = (bar.ticker, bar.session)
        if key in bars:
            raise ContractError(f"Duplicate price bar: {key}")
        bars[key] = bar
    buckets, excluded = text_windows(config, sessions, texts)
    rows = []
    skipped = Counter()
    for previous, target in zip(sessions, sessions[1:]):
        decision = target.open_at - timedelta(minutes=config.decision_minutes_before_open)
        if decision <= previous.close_at:
            raise ContractError("Decision must follow the previous session close")
        for symbol in config.tickers:
            prior, outcome = bars.get((symbol, previous.date)), bars.get((symbol, target.date))
            if prior is None or outcome is None:
                skipped["missing_price_pair"] += 1
                continue
            if prior.available_at >= decision:
                skipped["prior_price_unavailable"] += 1
                continue
            if outcome.split_ratio != 1:
                skipped["target_split_session"] += 1
                continue
            eligible = sorted(buckets.get((symbol, target.date), []), key=lambda r: (r.available_at, r.record_id))
            daily_return = outcome.close / prior.close - 1
            rows.append({
                "ticker": symbol, "target_session": target.date.isoformat(),
                "previous_session": previous.date.isoformat(), "window_start": previous.close_at.isoformat(),
                "decision_at": decision.isoformat(), "target_close_at": target.close_at.isoformat(),
                "prior_close": prior.close, "price_available_at": prior.available_at.isoformat(),
                "texts": [{"record_id": r.record_id, "source": r.source, "text": r.text,
                           "published_at": r.published_at.isoformat(), "available_at": r.available_at.isoformat(),
                           "availability_basis": r.availability_basis, "origin_id": r.origin_id} for r in eligible],
                "text_count": len(eligible), "missing_text": not eligible,
                "label_up": int(daily_return > 0), "target_return": daily_return,
                "label_available_at": outcome.available_at.isoformat(),
            })
    rows, split_report = chronological_split(rows, config)
    final_used = {(t["source"], row["ticker"], t["record_id"]) for row in rows for t in row["texts"]}
    return rows, {"input_sessions": len(sessions), "input_prices": len(prices), "input_texts": len(texts),
                  "unique_eligible_texts_in_output": len(final_used), "text_exclusions": dict(excluded),
                  "text_records_without_output_row": len(texts) - sum(excluded.values()) - len(final_used),
                  "skipped_price_pairs": dict(skipped), "output_rows": len(rows),
                  "missing_text_rows": sum(row["missing_text"] for row in rows), "split": split_report}
