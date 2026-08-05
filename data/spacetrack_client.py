"""Space-Track historical archive client (F0.1, the validation enabler).

The historical TLE archive is what makes "detect Cosmos 2542 in 2020"
possible. Requires a free account; credentials come from .env via
config.py (SPACETRACK_USER / SPACETRACK_PASS) and are never hardcoded.

Space-Track etiquette: they rate-limit and will suspend accounts that
hammer the API. The `spacetrack` library throttles automatically;
still, batch requests by date window rather than per-TLE.
"""
from __future__ import annotations

from datetime import date

from spacetrack import SpaceTrackClient
import spacetrack.operators as op

from config import SPACETRACK_PASS, SPACETRACK_USER
from data.store import TleRecord, insert_tles
from data.tle_parse import epoch_from_line1, norad_from_line1


def _client() -> SpaceTrackClient:
    if not SPACETRACK_USER or not SPACETRACK_PASS:
        raise RuntimeError(
            "Space-Track credentials missing. Copy .env.example to .env and "
            "fill SPACETRACK_USER / SPACETRACK_PASS (free account: "
            "https://www.space-track.org/auth/createAccount)."
        )
    return SpaceTrackClient(identity=SPACETRACK_USER, password=SPACETRACK_PASS)


def fetch_historical(
    norad_ids: list[int],
    start: date,
    end: date,
) -> list[TleRecord]:
    """Fetch all archived TLEs for the given objects in [start, end].

    Uses the `gp_history` class, not the legacy `tle`/`tle_latest` classes —
    Space-Track has retired those (confirmed live 2026-07-15: `st.tle(...)`
    fails with "Your Class Does Not Exist"). `gp_history` is the current
    historical-archive endpoint and still exposes TLE_LINE1/TLE_LINE2 via
    format="tle".
    """
    st = _client()
    raw = st.gp_history(
        norad_cat_id=norad_ids,
        epoch=op.inclusive_range(start, end),
        orderby="epoch asc",
        format="tle",
    )
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    records: list[TleRecord] = []
    for i in range(0, len(lines) - 1, 2):
        l1, l2 = lines[i], lines[i + 1]
        if l1.startswith("1 ") and l2.startswith("2 "):
            records.append(
                TleRecord(
                    norad_id=norad_from_line1(l1),
                    epoch=epoch_from_line1(l1),
                    line1=l1,
                    line2=l2,
                    source="spacetrack",
                )
            )
    return records


def ingest_historical(norad_ids: list[int], start: date, end: date) -> dict:
    recs = fetch_historical(norad_ids, start, end)
    new = insert_tles(recs)
    return {"fetched": len(recs), "new": new}


def check_archive_coverage(norad_ids: list[int], start: date, end: date) -> dict[int, int]:
    """Week-1 sanity check (§4): how many archived TLEs exist per object
    in the window? Zero for any object means the validation event can't
    be built on — stop and flag it, don't fake it.
    """
    recs = fetch_historical(norad_ids, start, end)
    counts = {nid: 0 for nid in norad_ids}
    for r in recs:
        counts[r.norad_id] = counts.get(r.norad_id, 0) + 1
    return counts
