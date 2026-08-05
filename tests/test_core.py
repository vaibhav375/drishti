"""Tier-0/1 unit tests. Run: python -m pytest tests/ -v

Real-event validation tests (Cosmos 2542, Cosmos 1408, controls) live in
validate/ and need the Space-Track archive; these tests are offline.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from core.elements import mean_elements
from core.frames import cw_propagate, eci_to_ric, ric_basis
from core.propagation import (
    propagate_at,
    propagate_window,
    satrec_from_tle,
    tle_epoch_datetime,
)
from data.store import TleRecord
from data.tle_parse import epoch_from_line1, parse_tle_text
from detect.maneuver import detect_maneuvers

# A real ISS (ZARYA) TLE — fixed test fixture, epoch 2024-01-01ish.
ISS_L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30777-3 0  9990"
ISS_L2 = "2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.49384367430123"
ISS = TleRecord(25544, epoch_from_line1(ISS_L1), ISS_L1, ISS_L2, "test")


# ---------------------------------------------------------------- frames
def test_ric_basis_orthonormal():
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    M = ric_basis(r, v)
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
    # R axis must point along r, C along r×v
    assert np.allclose(M[0], [1, 0, 0])
    assert np.allclose(M[2], [0, 0, 1])
    assert np.isclose(np.linalg.det(M), 1.0)  # right-handed


def test_eci_to_ric_known_offset():
    r_ref = np.array([7000.0, 0.0, 0.0])
    v_ref = np.array([0.0, 7.5, 0.0])
    # Other object 10 km directly above (radial) and 5 km ahead (in-track).
    rho, _ = eci_to_ric(r_ref, v_ref, r_ref + [10.0, 5.0, 0.0], v_ref)
    assert np.allclose(rho, [10.0, 5.0, 0.0], atol=1e-9)


def test_ric_corotating_object_has_zero_relative_velocity():
    """A point rigidly co-rotating with the frame must show ~0 RIC velocity —
    validates the transport term (−ω×ρ)."""
    r_ref = np.array([7000.0, 0.0, 0.0])
    v_ref = np.array([0.0, 7.5, 0.0])
    omega = np.cross(r_ref, v_ref) / (7000.0**2)
    rho = np.array([10.0, 0.0, 0.0])                # 10 km radially out
    v_other = v_ref + np.cross(omega, rho)          # co-rotating velocity
    _, vel_ric = eci_to_ric(r_ref, v_ref, r_ref + rho, v_other)
    assert np.linalg.norm(vel_ric) < 1e-12


def test_cw_stationary_at_origin_stays_put():
    n = 0.0011  # ~ISS mean motion rad/s
    state = cw_propagate(np.zeros(6), n, 3600.0)
    assert np.allclose(state, 0.0)


# ----------------------------------------------------------- propagation
def test_propagate_iss_altitude_reasonable():
    sat = satrec_from_tle(ISS.line1, ISS.line2)
    t0 = tle_epoch_datetime(sat)
    r, v = propagate_at(sat, t0)
    alt = np.linalg.norm(r) - 6378.137
    assert 350 < alt < 500          # ISS altitude band, km
    assert 7.0 < np.linalg.norm(v) < 8.0   # LEO circular speed, km/s


def test_propagate_window_shapes():
    sat = satrec_from_tle(ISS.line1, ISS.line2)
    t0 = tle_epoch_datetime(sat)
    eph = propagate_window(ISS, t0, t0 + timedelta(minutes=90), step_s=60)
    assert eph.r_teme.shape == eph.v_teme.shape == (91, 3)


def test_mean_elements_iss():
    el = mean_elements(ISS)
    assert el.regime == "LEO"
    assert 88 < el.period_min < 96
    assert 50 < el.incl_deg < 53


# ------------------------------------------------------------- ingestion
def test_parse_tle_text_with_name_lines():
    blob = f"ISS (ZARYA)\n{ISS_L1}\n{ISS_L2}\n\nGARBAGE LINE\n"
    recs = parse_tle_text(blob, "test")
    assert len(recs) == 1
    assert recs[0].norad_id == 25544
    assert recs[0].epoch.startswith("2024-01-01")


# ------------------------------------------------------ maneuver (F1.1)
def _tle_checksum(line: str) -> str:
    s = sum(int(c) for c in line[:68] if c.isdigit()) + line[:68].count("-")
    return line[:68] + str(s % 10)


def _synthetic_history(n_pairs: int = 14, burn_at: int | None = None,
                       burn_dv_kms: float = 0.010,
                       burns: dict[int, float] | None = None) -> list[TleRecord]:
    """Build a SELF-CONSISTENT daily TLE history by string surgery.

    Between epochs, M / RAAN / argp are advanced using SGP4's own secular
    rates (mdot, nodedot, argpdot) so consecutive element sets describe
    the same orbit — quiet pairs then differ only by injected mean-motion
    jitter (TLE noise proxy) and small drag terms.

    At each index in `burns` (or at `burn_at` for the single-burn case),
    mean motion steps AS IF a tangential burn of that size changed the
    semi-major axis (dn/n ≈ −3·dv/v), applied on top of the CURRENT mean
    motion so successive burns stack realistically — the classic on-orbit
    maneuver signature detect_maneuvers must flag, repeatable for a
    pattern-of-life history with several burns of varying size.
    """
    from sgp4.api import Satrec

    burns = dict(burns) if burns else {}
    if burn_at is not None:
        burns[burn_at] = burn_dv_kms

    rng = np.random.default_rng(42)
    base_doy = float(ISS_L1[20:32])          # fractional day-of-year
    base_mm = float(ISS_L2[52:63])           # rev/day
    raan = float(ISS_L2[17:25])
    argp = float(ISS_L2[34:42])
    manom = float(ISS_L2[43:51])
    mm = base_mm

    def make(k: int, mm_k: float) -> tuple[str, str]:
        l1 = ISS_L1[:20] + f"{base_doy + k:012.8f}" + ISS_L1[32:]
        l2 = (ISS_L2[:17] + f"{raan % 360:8.4f}" + ISS_L2[25:34]
              + f"{argp % 360:8.4f} {manom % 360:8.4f} {mm_k:11.8f}"
              + ISS_L2[63:])
        return _tle_checksum(l1), _tle_checksum(l2)

    recs: list[TleRecord] = []
    for k in range(n_pairs + 1):
        if k in burns:
            mm = mm - 3.0 * (burns[k] / 7.66) * base_mm
        mm_k = mm + rng.normal(0, 1e-6) * base_mm   # TLE noise proxy
        l1, l2 = make(k, mm_k)
        recs.append(TleRecord(25544, epoch_from_line1(l1), l1, l2, "synthetic"))
        # advance angles one day using THIS element set's secular rates
        sat = Satrec.twoline2rv(l1, l2)
        rad_min_to_deg_day = 1440.0 * 180.0 / np.pi
        manom += sat.mdot * rad_min_to_deg_day
        raan += sat.nodedot * rad_min_to_deg_day
        argp += sat.argpdot * rad_min_to_deg_day
    return recs


def test_no_false_positives_on_quiet_history():
    events, floor = detect_maneuvers(_synthetic_history(burn_at=None))
    assert floor is not None
    assert events == []


def test_detects_injected_burn():
    burn_idx = 8
    hist = _synthetic_history(burn_at=burn_idx, burn_dv_kms=0.010)  # 10 m/s
    events, floor = detect_maneuvers(hist)
    assert floor is not None
    assert len(events) >= 1
    # the flagged epoch must be the pair spanning the injected burn
    assert any(e.epoch_iso == hist[burn_idx].epoch for e in events)
    top = max(events, key=lambda e: e.dv_ms)
    assert top.direction in ("prograde", "retrograde", "mixed")
    assert top.zscore > 5


def test_insufficient_history_returns_nothing():
    events, floor = detect_maneuvers(_synthetic_history(n_pairs=3))
    assert events == [] and floor is None
