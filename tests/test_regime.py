"""Tests for orbital regime auto-classification (core/regime.py).

Offline, synthetic MeanElements. The physics (SSO inclination, Molniya
vs Tundra period split) is validated against REAL objects too — see
README "Real-data findings": Sentinel-2A/3A classify as LEO[SSO], real
Molniya sats as HEO[MOLNIYA], and BeiDou-2 G2 (the one SJ-21 towed) as
GRAVEYARD, independently confirming the F2.1/F3.1 tow result.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import GEO_ALTITUDE_KM, MU_EARTH_KM3_S2, R_EARTH_KM, SIDEREAL_DAY_S
from core.elements import MeanElements
from core.regime import classify_regime, required_sso_inclination_deg


def _el(norad_id, apogee_km, perigee_km, incl_deg, argp_deg=270.0):
    a = R_EARTH_KM + (apogee_km + perigee_km) / 2.0
    ecc = (apogee_km - perigee_km) / (2 * a) if apogee_km != perigee_km else 0.0
    n = np.sqrt(MU_EARTH_KM3_S2 / a ** 3)
    period_min = 2 * np.pi / n / 60.0
    return MeanElements(
        norad_id=norad_id, epoch_iso="2024-01-01T00:00:00Z", semi_major_km=a, ecc=ecc,
        incl_deg=incl_deg, raan_deg=0.0, argp_deg=argp_deg, mean_anom_deg=0.0,
        mean_motion_rad_s=n, apogee_km=apogee_km, perigee_km=perigee_km,
        period_min=period_min, regime="",
    )


def test_leo_classifies_as_leo():
    assert classify_regime(_el(1, 420, 415, 51.6)).primary == "LEO"


def test_meo_classifies_as_meo():
    assert classify_regime(_el(2, 20744, 19619, 54.5)).primary == "MEO"


def test_geo_classifies_as_geo():
    c = classify_regime(_el(3, GEO_ALTITUDE_KM, GEO_ALTITUDE_KM, 0.05))
    assert c.primary == "GEO"


def test_graveyard_orbit_detected_above_geo():
    c = classify_regime(_el(4, GEO_ALTITUDE_KM + 300, GEO_ALTITUDE_KM + 300, 0.1))
    assert c.primary == "GRAVEYARD"


def test_decaying_orbit_detected():
    c = classify_regime(_el(5, 300, 120, 51.6))
    assert c.primary == "DECAYING"


def test_sso_inclination_is_retrograde_and_near_98_for_leo():
    # A ~790 km orbit should require an inclination near 98.5-98.7 deg.
    a = R_EARTH_KM + 790
    i_sso = required_sso_inclination_deg(a, 0.0)
    assert i_sso is not None
    assert 97.5 < i_sso < 99.5


def test_sso_tag_fires_at_computed_inclination():
    a = R_EARTH_KM + 790
    i_sso = required_sso_inclination_deg(a, 0.0)
    c = classify_regime(_el(6, 790, 788, i_sso))
    assert "SSO" in c.special_tags
    assert c.primary == "LEO"


def test_same_altitude_wrong_inclination_is_not_sso():
    c = classify_regime(_el(7, 790, 788, 45.0))
    assert "SSO" not in c.special_tags


def test_molniya_is_heo_with_molniya_tag_not_gto():
    # Critical inclination, ~12h period, high ecc -> HEO [MOLNIYA],
    # NOT GTO (the real-data bug this locks in).
    c = classify_regime(_el(8, 38565, 1788, 64.1))
    assert c.primary == "HEO"
    assert "MOLNIYA" in c.special_tags


def test_tundra_is_heo_with_tundra_tag():
    # Critical inclination, ~24h period, moderate ecc.
    apo, per = 46000, 25000  # ecc ~0.28, period ~sidereal day at this a
    c = classify_regime(_el(9, apo, per, 63.4))
    assert "TUNDRA" in c.special_tags


def test_genuine_gto_low_inclination_classifies_as_gto():
    # LEO perigee, near-GEO apogee, LOW inclination -> real transfer orbit.
    c = classify_regime(_el(10, GEO_ALTITUDE_KM, 300, 6.0))
    assert c.primary == "GTO"


def test_equatorial_and_retrograde_orientation_tags():
    assert "EQUATORIAL" in classify_regime(_el(11, 35786, 35786, 0.05)).special_tags
    assert "RETROGRADE" in classify_regime(_el(12, 800, 800, 100.5)).special_tags


def test_label_combines_primary_and_tags():
    c = classify_regime(_el(13, 800, 798, 98.6))
    assert c.label.startswith("LEO [")
    assert "SSO" in c.label
