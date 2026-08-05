"""Propagation engine (F0.2): TLE → position/velocity over a time window.

Built directly on `sgp4` (C-accelerated; full-catalog propagation is fast
even on an M1). Output states are in the TEME frame, km and km/s — the
native SGP4 frame. Conversions live in core/frames.py.

Precision honesty (§9): SGP4 + public TLEs gives km-level position error
that grows with time from epoch. Nothing downstream may claim better.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
from sgp4.api import SGP4_ERRORS, Satrec, jday

from data.store import TleRecord


@dataclass
class Ephemeris:
    """Propagated states in TEME. Positions km, velocities km/s."""
    norad_id: int
    times: list[datetime]          # UTC
    r_teme: np.ndarray             # (N, 3) km
    v_teme: np.ndarray             # (N, 3) km/s

    def state_at(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.r_teme[i], self.v_teme[i]


def satrec_from_tle(line1: str, line2: str) -> Satrec:
    return Satrec.twoline2rv(line1, line2)


def tle_epoch_datetime(sat: Satrec) -> datetime:
    """Epoch of a parsed TLE as a timezone-aware UTC datetime."""
    # sgp4 exposes jdsatepoch (+ fraction) as Julian date.
    jd = sat.jdsatepoch + sat.jdsatepochF
    # JD → Unix epoch: JD 2440587.5 == 1970-01-01T00:00Z
    unix = (jd - 2440587.5) * 86400.0
    return datetime.fromtimestamp(unix, tz=timezone.utc)


def propagate_at(sat: Satrec, t: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Single-epoch propagation → (r_teme km, v_teme km/s). Raises on SGP4 error."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                  t.second + t.microsecond / 1e6)
    err, r, v = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error {err}: {SGP4_ERRORS[err]}")
    return np.array(r), np.array(v)


def propagate_window(
    record: TleRecord,
    start: datetime,
    end: datetime,
    step_s: float = 60.0,
) -> Ephemeris:
    """Propagate one object across [start, end] at fixed step."""
    sat = satrec_from_tle(record.line1, record.line2)
    n = max(2, int((end - start).total_seconds() // step_s) + 1)
    times = [start + timedelta(seconds=i * step_s) for i in range(n)]
    r = np.empty((n, 3))
    v = np.empty((n, 3))
    for i, t in enumerate(times):
        r[i], v[i] = propagate_at(sat, t)
    return Ephemeris(record.norad_id, times, r, v)


def propagate_aligned(
    records_a: Sequence[TleRecord],
    records_b: Sequence[TleRecord],
    step_hours: float = 6.0,
    max_epoch_gap_days: float = 3.0,
) -> tuple[list[datetime], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned ephemerides for TWO objects over their OVERLAPPING TLE
    coverage window — the input relative motion (F2.1/RPO) needs.

    At each grid time, each object is propagated from its OWN nearest-
    epoch TLE. A grid point is INCLUDED only if both objects have a TLE
    within max_epoch_gap_days of that time — extrapolating either object
    far past its nearest real element set would fabricate relative motion
    that isn't backed by data, so those points are skipped, not guessed.

    Returns (times, r_a (N,3), v_a (N,3), r_b (N,3), v_b (N,3)), all
    trimmed to only the grid points that passed the gap check.
    """
    if not records_a or not records_b:
        return [], np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))

    sats_a = [(tle_epoch_datetime(satrec_from_tle(r.line1, r.line2)), r) for r in records_a]
    sats_b = [(tle_epoch_datetime(satrec_from_tle(r.line1, r.line2)), r) for r in records_b]

    start = max(sats_a[0][0], sats_b[0][0])
    end = min(sats_a[-1][0], sats_b[-1][0])
    if end <= start:
        return [], np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))

    n_steps = max(2, int((end - start).total_seconds() / (step_hours * 3600.0)) + 1)
    grid = [start + timedelta(hours=i * step_hours) for i in range(n_steps)]
    gap = timedelta(days=max_epoch_gap_days)

    def nearest(sats: list[tuple[datetime, TleRecord]], t: datetime) -> TleRecord | None:
        best = min(sats, key=lambda s: abs(s[0] - t))
        return best[1] if abs(best[0] - t) <= gap else None

    times: list[datetime] = []
    r_a_list, v_a_list, r_b_list, v_b_list = [], [], [], []
    for t in grid:
        rec_a = nearest(sats_a, t)
        rec_b = nearest(sats_b, t)
        if rec_a is None or rec_b is None:
            continue
        try:
            r_a, v_a = propagate_at(satrec_from_tle(rec_a.line1, rec_a.line2), t)
            r_b, v_b = propagate_at(satrec_from_tle(rec_b.line1, rec_b.line2), t)
        except RuntimeError:
            continue
        times.append(t)
        r_a_list.append(r_a)
        v_a_list.append(v_a)
        r_b_list.append(r_b)
        v_b_list.append(v_b)

    return (
        times,
        np.array(r_a_list) if r_a_list else np.empty((0, 3)),
        np.array(v_a_list) if v_a_list else np.empty((0, 3)),
        np.array(r_b_list) if r_b_list else np.empty((0, 3)),
        np.array(v_b_list) if v_b_list else np.empty((0, 3)),
    )


def propagate_many_at(
    records: Sequence[TleRecord], t: datetime
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Propagate many objects to one instant (catalog screening workhorse).

    Objects whose propagation fails (decayed, bad elements) are skipped.
    Returns (r (M,3), v (M,3), norad_ids kept).
    """
    rs, vs, ids = [], [], []
    for rec in records:
        try:
            r, v = propagate_at(satrec_from_tle(rec.line1, rec.line2), t)
        except RuntimeError:
            continue
        rs.append(r)
        vs.append(v)
        ids.append(rec.norad_id)
    return np.array(rs), np.array(vs), ids
