"""Tests for graveyard disposal-compliance monitoring (detect/graveyard.py).

Offline, synthetic MeanElements. Validated live against the real GEO belt
too — see README "Real-data findings": BeiDou-2 G2 (the object SJ-21
towed) comes out COMPLIANT_GRAVEYARD, and a live-catalog scan finds real
uncontrolled drifters and graveyard objects.
"""
from __future__ import annotations

import numpy as np

from config import GEO_ALTITUDE_KM, MU_EARTH_KM3_S2, R_EARTH_KM
from core.elements import MeanElements
from detect.graveyard import IADC_MIN_REORBIT_KM, assess_disposal


def _el(norad_id, apogee_km, perigee_km, incl_deg):
    a = R_EARTH_KM + (apogee_km + perigee_km) / 2.0
    ecc = (apogee_km - perigee_km) / (2 * a) if apogee_km != perigee_km else 0.0
    n = np.sqrt(MU_EARTH_KM3_S2 / a ** 3)
    return MeanElements(
        norad_id=norad_id, epoch_iso="2024-01-01T00:00:00Z", semi_major_km=a, ecc=ecc,
        incl_deg=incl_deg, raan_deg=0.0, argp_deg=0.0, mean_anom_deg=0.0,
        mean_motion_rad_s=n, apogee_km=apogee_km, perigee_km=perigee_km,
        period_min=2 * np.pi / n / 60.0, regime="",
    )


def test_properly_reorbited_is_compliant():
    alt = GEO_ALTITUDE_KM + IADC_MIN_REORBIT_KM + 60
    a = assess_disposal(_el(1, alt, alt, 0.5))
    assert a.status == "COMPLIANT_GRAVEYARD"
    assert a.compliant is True


def test_operational_box_is_not_scored():
    a = assess_disposal(_el(2, GEO_ALTITUDE_KM, GEO_ALTITUDE_KM, 0.1))
    assert a.status == "OPERATIONAL_OR_ABANDONED_IN_PLACE"
    assert a.compliant is None


def test_shallow_reorbit_is_non_compliant():
    # Raised above GEO but only ~120 km — inside the 235 km buffer.
    alt = GEO_ALTITUDE_KM + 120
    a = assess_disposal(_el(3, alt, alt, 0.5))
    assert a.status == "NON_COMPLIANT_SHALLOW"
    assert a.compliant is False


def test_inclined_geosynchronous_is_operational_not_drift():
    # A geosynchronous object near GEO altitude but inclined (IGSO: QZSS,
    # BeiDou-IGSO, IRNSS) is a legitimate operational orbit type, NOT an
    # uncontrolled drifter. Regression for a real mislabel found live.
    a = assess_disposal(_el(4, GEO_ALTITUDE_KM, GEO_ALTITUDE_KM, 45.0))
    assert a.status == "INCLINED_GEOSYNCHRONOUS"
    assert a.compliant is None


def test_non_geo_object_is_excluded():
    a = assess_disposal(_el(5, 420, 415, 51.6))  # LEO
    assert a.status == "NOT_GEO_BELT"
    assert a.compliant is None


def test_km_above_geo_sign_and_magnitude():
    alt = GEO_ALTITUDE_KM + 300
    a = assess_disposal(_el(6, alt, alt, 1.0))
    assert abs(a.km_above_geo - 300) < 1.0
