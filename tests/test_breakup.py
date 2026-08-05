"""Tests for F1.3 breakup detection (detect/breakup.py::cluster_from_rows).

Offline, synthetic SATCAT-shaped rows — no DB. Thresholds (MIN_FRAGMENTS,
REGIME_INCL_SPREAD_DEG) are calibrated against real SATCAT data (see
detect/breakup.py docstring); the real-event validation run against actual
Cosmos 1408 / Fengyun-1C / Cosmos 2542-2543-launch data lives in
validate/run_benchmark.py, which needs the local store already populated
from Space-Track (offline tests here don't depend on that).
"""
from __future__ import annotations

from detect.breakup import MIN_FRAGMENTS, cluster_from_rows


def _row(norad_id, intl_desig, object_type, incl=82.5, perigee=150.0, apogee=300.0, name=None):
    return {
        "norad_id": norad_id,
        "intl_desig": intl_desig,
        "name": name or f"OBJ {norad_id}",
        "object_type": object_type,
        "inclination": incl,
        "perigee_km": perigee,
        "apogee_km": apogee,
    }


def test_ordinary_launch_with_no_debris_is_not_a_breakup():
    # A payload + rocket body + a couple of sub-satellites — the normal
    # shape of a launch, no fragmentation signature (mirrors the real
    # Cosmos 2542/2543 launch: 5 objects, zero DEB).
    rows = [
        _row(1, "2019-079A", "PAY"),
        _row(2, "2019-079B", "R/B"),
        _row(3, "2019-079C", "R/B"),
        _row(4, "2019-079D", "PAY"),
    ]
    assert cluster_from_rows("2019-079", rows) is None


def test_few_debris_below_threshold_is_not_a_breakup():
    rows = [_row(1, "1999-001A", "PAY"), _row(2, "1999-001B", "R/B")]
    rows += [_row(100 + i, f"1999-001{chr(65 + i)}", "DEB") for i in range(MIN_FRAGMENTS - 1)]
    assert len(rows) - 2 == MIN_FRAGMENTS - 1
    assert cluster_from_rows("1999-001", rows) is None


def test_large_tight_cluster_is_high_confidence_breakup():
    # Mirrors the real Cosmos 1408 shape: parent + tight cloud of >=100
    # DEB objects clustered within a few degrees of inclination.
    rows = [_row(1, "1982-092A", "PAY", name="COSMOS 1408")]
    rows += [
        _row(1000 + i, f"1982-092{chr(65 + i % 26)}{chr(65 + i // 26)}", "DEB",
             incl=82.5 + (i % 5) * 0.1, perigee=140 + i, apogee=300 - i)
        for i in range(150)
    ]
    cand = cluster_from_rows("1982-092", rows)
    assert cand is not None
    assert cand.n_fragments == 150
    assert cand.parent_norad_id == 1
    assert cand.parent_name == "COSMOS 1408"
    assert cand.confidence == "high"


def test_scattered_debris_not_tightly_clustered_is_low_confidence():
    # Enough DEB rows to clear MIN_FRAGMENTS, but scattered across
    # inclinations that don't cohere into one physical cloud — still
    # worth surfacing, but not with unwarranted confidence.
    rows = [_row(1, "2001-050A", "PAY")]
    rows += [
        _row(2000 + i, f"2001-050{chr(65 + i)}", "DEB", incl=20.0 + i * 8.0)
        for i in range(MIN_FRAGMENTS)
    ]
    cand = cluster_from_rows("2001-050", rows)
    assert cand is not None
    assert cand.confidence == "low"


def test_debris_scattered_across_decades_of_catalog_growth_is_not_a_breakup():
    # Mirrors the real ISS case: individually-lost EVA items (tools, a
    # camera) cataloged years apart share the parent's orbit plane but
    # were never one physical event. Catalog-number dispersion — not
    # inclination — is what tells this apart from a genuine fragmentation
    # cloud, whose fragments all get catalogued together in a burst.
    rows = [_row(1, "1998-067A", "PAY", name="ISS")]
    rows += [
        _row(20000 + i * 4000, f"1998-067{chr(65 + i)}", "DEB")
        for i in range(MIN_FRAGMENTS)
    ]
    assert cluster_from_rows("1998-067", rows) is None


def test_missing_orbital_data_excludes_row_from_count():
    # A DEB row with no inclination/perigee/apogee (not yet fully
    # characterized by SATCAT) must not be silently counted as a
    # confirmed cluster member.
    rows = [_row(1, "2005-010A", "PAY")]
    rows += [_row(200 + i, f"2005-010{chr(65+i)}", "DEB") for i in range(MIN_FRAGMENTS)]
    rows.append({
        "norad_id": 999, "intl_desig": "2005-010Z", "name": "UNCATALOGUED",
        "object_type": "DEB", "inclination": None, "perigee_km": None, "apogee_km": None,
    })
    cand = cluster_from_rows("2005-010", rows)
    assert cand is not None
    assert 999 not in cand.fragment_norad_ids
    assert cand.n_fragments == MIN_FRAGMENTS
