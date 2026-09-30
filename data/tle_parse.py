"""Shared TLE text parsing: raw 2-line/3-line element sets → TleRecord."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data.store import TleRecord


def epoch_from_line1(line1: str) -> str:
    """Decode the TLE epoch field (cols 19-32) to ISO-8601 UTC.

    Format: YYDDD.DDDDDDDD (two-digit year + fractional day-of-year).
    Years 57-99 → 1957-1999, 00-56 → 2000-2056 (standard convention).
    """
    yy = int(line1[18:20])
    year = 1900 + yy if yy >= 57 else 2000 + yy
    frac_day = float(line1[20:32])
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=frac_day - 1.0)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def norad_from_line1(line1: str) -> int:
    return int(line1[2:7])


def parse_tle_text(text: str, source: str) -> list[TleRecord]:
    """Parse a blob of TLEs (with or without name lines) into records.

    Tolerates blank lines and interleaved name lines; skips malformed
    pairs rather than raising, so one bad element set can't kill an
    ingestion run.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    records: list[TleRecord] = []
    i = 0
    while i < len(lines) - 1:
        l1, l2 = lines[i], lines[i + 1]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                records.append(
                    TleRecord(
                        norad_id=norad_from_line1(l1),
                        epoch=epoch_from_line1(l1),
                        line1=l1,
                        line2=l2,
                        source=source,
                    )
                )
            except (ValueError, IndexError):
                pass  # malformed set — skip, don't crash the run
            i += 2
        else:
            i += 1  # name line or junk
    return records
