"""Versioned exchange schedules, never weekday approximations."""
from datetime import date, timedelta
from importlib.metadata import version

from .contracts import ContractError, Session


def generate_calendar(start: date, end: date) -> tuple[list[Session], dict]:
    if end < start:
        raise ContractError("Calendar end must be on or after start")
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise ContractError("M2 requires: python -m pip install -r requirements-acquisition.lock") from exc
    calendar = xcals.get_calendar("XNYS", start=start - timedelta(days=40), end=end + timedelta(days=7))
    targets = calendar.sessions_in_range(start.isoformat(), end.isoformat())
    if targets.empty:
        raise ContractError("No exchange sessions in the requested target range")
    indices = [calendar.previous_session(targets[0]), *targets]
    sessions = [Session.from_dict({"session": day.date().isoformat(),
                "open_at": calendar.session_open(day).isoformat(),
                "close_at": calendar.session_close(day).isoformat()}) for day in indices]
    return sessions, {"provider": "exchange_calendars", "version": version("exchange-calendars"),
                      "exchange": "XNYS", "requested_target_start": start.isoformat(),
                      "requested_target_end": end.isoformat(), "session_count": len(sessions),
                      "includes_preceding_session": True, "verification": "generated_from_pinned_library"}


def verify_calendar(sessions: list[Session], start: date, end: date) -> dict:
    expected, metadata = generate_calendar(start, end)
    if sorted(sessions, key=lambda s: s.date) != expected:
        raise ContractError("Calendar is incomplete or differs from the pinned XNYS schedule")
    return metadata
