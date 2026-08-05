"""Celestrak ingestion (no auth): current TLE groups + SATCAT metadata."""
from __future__ import annotations

import csv
import io

import requests

from config import CELESTRAK_GP_URL, CELESTRAK_SATCAT_URL, DEFAULT_CELESTRAK_GROUPS
from data.store import TleRecord, insert_tles, record_catalog_snapshot, upsert_satcat
from data.tle_parse import parse_tle_text

TIMEOUT = 60


def fetch_group(group: str) -> list[TleRecord]:
    """Fetch a Celestrak GP group (e.g. 'active', 'stations', 'geo') as TLEs."""
    resp = requests.get(
        CELESTRAK_GP_URL,
        params={"GROUP": group, "FORMAT": "tle"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return parse_tle_text(resp.text, source="celestrak")


def fetch_object(norad_id: int) -> list[TleRecord]:
    """Fetch ONE object's current element set from Celestrak by catalog
    number (no auth) — the on-demand 'refresh from Celestrak' path."""
    resp = requests.get(
        CELESTRAK_GP_URL,
        params={"CATNR": norad_id, "FORMAT": "tle"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    if "No GP data found" in resp.text:
        return []
    return parse_tle_text(resp.text, source="celestrak")


def ingest_object(norad_id: int) -> int:
    """Fetch one object's current TLE and store it. Returns rows inserted."""
    return insert_tles(fetch_object(norad_id))


def ingest_groups(groups: list[str] | None = None, snapshot: bool = True) -> dict:
    """Pull groups, store them, and optionally record a catalog snapshot.

    Returns per-group counts: {'active': {'fetched': N, 'new': M}, ...}
    """
    groups = groups or DEFAULT_CELESTRAK_GROUPS
    summary: dict = {}
    all_ids: set[int] = set()
    for g in groups:
        recs = fetch_group(g)
        new = insert_tles(recs)
        all_ids.update(r.norad_id for r in recs)
        summary[g] = {"fetched": len(recs), "new": new}
    if snapshot and all_ids:
        summary["snapshot_at"] = record_catalog_snapshot(sorted(all_ids))
    return summary


# ---------------------------------------------------------------------------
# SATCAT
# ---------------------------------------------------------------------------

def _f(val: str) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def ingest_satcat() -> int:
    """Download the full Celestrak SATCAT CSV into the local store."""
    resp = requests.get(CELESTRAK_SATCAT_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = []
    for r in reader:
        try:
            norad = int(r["NORAD_CAT_ID"])
        except (KeyError, ValueError):
            continue
        rows.append(
            {
                "norad_id": norad,
                "intl_desig": r.get("OBJECT_ID"),
                "name": r.get("OBJECT_NAME"),
                "country": r.get("OWNER"),
                "launch_date": r.get("LAUNCH_DATE"),
                "decay_date": r.get("DECAY_DATE") or None,
                "object_type": r.get("OBJECT_TYPE"),
                "rcs_m2": _f(r.get("RCS")),
                "period_min": _f(r.get("PERIOD")),
                "inclination": _f(r.get("INCLINATION")),
                "apogee_km": _f(r.get("APOGEE")),
                "perigee_km": _f(r.get("PERIGEE")),
            }
        )
    return upsert_satcat(rows)
