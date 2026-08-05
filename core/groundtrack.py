"""Ground track (lat/lon/alt) via skyfield, which handles TEME→ITRF
internally and correctly. Used by the dashboard (F0.4) and later by
pass-prediction bonus features.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

from data.store import TleRecord

_ts = load.timescale()


def ground_track(
    rec: TleRecord, start: datetime, end: datetime, step_s: float = 60.0
) -> dict[str, np.ndarray]:
    """Sub-satellite latitude/longitude (deg) and altitude (km) over a window."""
    sat = EarthSatellite(rec.line1, rec.line2, str(rec.norad_id), _ts)
    n = max(2, int((end - start).total_seconds() // step_s) + 1)
    times = _ts.utc(
        start.year, start.month, start.day,
        start.hour, start.minute,
        [start.second + i * step_s for i in range(n)],
    )
    geocentric = sat.at(times)
    sub = wgs84.subpoint_of(geocentric)
    alt = wgs84.height_of(geocentric).km
    return {
        "lat_deg": sub.latitude.degrees,
        "lon_deg": sub.longitude.degrees,
        "alt_km": alt,
        "unix_s": np.array(
            [(start + timedelta(seconds=i * step_s)).timestamp() for i in range(n)]
        ),
    }
