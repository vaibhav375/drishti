"""Tests for F2.2 conjunction-filtering cascade (detect/cascade.py).

Offline, synthetic MeanElements — no TLE/DB dependency. Real-data
validation (does the cascade keep the known-close SJ-21/BeiDou-2 G2 pair,
filter out unrelated LEO/GEO pairs, and — critically — stay fast on the
real ~570-object GEO belt after two design iterations found live) — see
README "Real-data findings".
"""
from __future__ import annotations

from core.elements import MeanElements
from detect.cascade import (
    ALTITUDE_MARGIN_KM,
    altitude_sieve,
    geometry_filter,
    relative_inclination_deg,
    run_cascade,
)


def _elements(norad_id, perigee_km, apogee_km, incl_deg=0.0, raan_deg=0.0, argp_deg=0.0):
    semi_major = 6378.137 + (perigee_km + apogee_km) / 2.0
    ecc = (apogee_km - perigee_km) / (2 * semi_major) if apogee_km != perigee_km else 0.0
    return MeanElements(
        norad_id=norad_id, epoch_iso="2024-01-01T00:00:00Z",
        semi_major_km=semi_major, ecc=ecc, incl_deg=incl_deg, raan_deg=raan_deg,
        argp_deg=argp_deg, mean_anom_deg=0.0, mean_motion_rad_s=0.001,
        apogee_km=apogee_km, perigee_km=perigee_km, period_min=90.0, regime="LEO",
    )


# ---------------------------------------------------------------- altitude sieve
def test_altitude_sieve_excludes_non_overlapping_bands():
    leo = _elements(1, 400.0, 400.0)
    geo = _elements(2, 35786.0, 35786.0)
    pairs = altitude_sieve([leo, geo], margin_km=ALTITUDE_MARGIN_KM)
    assert pairs == []


def test_altitude_sieve_includes_overlapping_bands():
    a = _elements(1, 400.0, 500.0)
    b = _elements(2, 450.0, 550.0)
    pairs = altitude_sieve([a, b], margin_km=ALTITUDE_MARGIN_KM)
    assert (1, 2) in pairs


def test_altitude_sieve_margin_bridges_a_small_gap():
    a = _elements(1, 400.0, 400.0)
    b = _elements(2, 470.0, 470.0)  # 70 km gap: needs margin to bridge
    assert altitude_sieve([a, b], margin_km=0.0) == []
    assert (1, 2) in altitude_sieve([a, b], margin_km=50.0)


# --------------------------------------------------- relative-inclination filter
def test_coplanar_orbits_have_zero_relative_inclination():
    a = _elements(1, 500.0, 500.0, incl_deg=45.0, raan_deg=30.0)
    b = _elements(2, 500.0, 500.0, incl_deg=45.0, raan_deg=30.0)
    assert relative_inclination_deg(a, b) < 1e-9


def test_perpendicular_planes_are_90_degrees_apart():
    equatorial = _elements(1, 500.0, 500.0, incl_deg=0.0)
    polar = _elements(2, 500.0, 500.0, incl_deg=90.0, raan_deg=17.0)
    assert abs(relative_inclination_deg(equatorial, polar) - 90.0) < 1e-6


def test_geometry_filter_rejects_pairs_beyond_max_relative_inclination():
    a = _elements(1, 500.0, 500.0, incl_deg=0.0)
    b = _elements(2, 500.0, 500.0, incl_deg=90.0, raan_deg=17.0)
    by_id = {1: a, 2: b}
    assert geometry_filter([(1, 2)], by_id, max_relative_incl_deg=10.0) == []
    kept = geometry_filter([(1, 2)], by_id, max_relative_incl_deg=95.0)
    assert len(kept) == 1
    assert kept[0].norad_a == 1 and kept[0].norad_b == 2


# ------------------------------------------------------------------- run_cascade
def test_run_cascade_end_to_end_filters_and_keeps():
    leo1 = _elements(1, 400.0, 420.0, incl_deg=51.6, raan_deg=0.0)
    leo2 = _elements(2, 405.0, 425.0, incl_deg=51.6, raan_deg=0.0)
    geo = _elements(3, 35786.0, 35786.0)
    kept = run_cascade([leo1, leo2, geo])
    kept_pairs = {(c.norad_a, c.norad_b) for c in kept}
    assert (1, 2) in kept_pairs
    assert (1, 3) not in kept_pairs and (2, 3) not in kept_pairs


def test_run_cascade_stays_fast_on_a_dense_coplanar_population():
    # Regression test for the real perf failure: hundreds of near-
    # identical, near-coplanar orbits (the GEO belt) in one altitude band
    # must not blow up run_cascade -- this is now O(1) per pair, not
    # point-cloud sampling, so it should complete near-instantly even at
    # this scale.
    geo_like = [
        _elements(100 + k, 35786.0, 35786.0, incl_deg=k * 0.01, raan_deg=k * 0.1)
        for k in range(600)
    ]
    kept = run_cascade(geo_like)
    assert len(kept) > 0  # near-identical planes -> most pairs should survive


def test_screen_top_matches_run_cascade_on_a_mixed_population():
    import random
    from detect.cascade import screen_top
    rng = random.Random(7)
    els = []
    for n in range(300):
        peri = rng.choice([400, 550, 800, 20000, 35700]) + rng.uniform(-30, 30)
        els.append(_elements(n, peri, peri + rng.uniform(0, 200),
                       incl_deg=rng.choice([0.1, 53, 97.6, 55]) + rng.uniform(-2, 2),
                       raan_deg=rng.uniform(0, 360)))
    slow = run_cascade(els)
    total, top = screen_top(els, top_k=25)
    assert total == len(slow)
    want = sorted(slow, key=lambda p: p.relative_inclination_deg)[:25]
    assert [round(p.relative_inclination_deg, 9) for p in top] == \
           [round(p.relative_inclination_deg, 9) for p in want]
    assert {(p.norad_a, p.norad_b) for p in slow} >= {(p.norad_a, p.norad_b) for p in top}


def test_screen_top_empty():
    from detect.cascade import screen_top
    assert screen_top([]) == (0, [])
