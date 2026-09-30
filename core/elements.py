"""Orbital-element utilities: mean elements straight off the parsed TLE,
apogee/perigee, and coarse regime tagging. Feeds the conjunction cascade
(F2.2) and threat-score context (F1.4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sgp4.api import Satrec

from config import MU_EARTH_KM3_S2, R_EARTH_KM
from core.propagation import satrec_from_tle
from data.store import TleRecord


@dataclass(frozen=True)
class MeanElements:
    norad_id: int
    epoch_iso: str
    semi_major_km: float
    ecc: float
    incl_deg: float
    raan_deg: float
    argp_deg: float
    mean_anom_deg: float
    mean_motion_rad_s: float
    apogee_km: float      # altitude above equatorial radius
    perigee_km: float
    period_min: float
    regime: str


def _regime(apogee_km: float, perigee_km: float, incl_deg: float, ecc: float) -> str:
    """Coarse regime tag. Deliberately simple; the bonus-feature
    auto-classifier (§10) can refine this later."""
    if apogee_km < 2000:
        return "LEO"
    if 34000 < perigee_km and apogee_km < 38000 and ecc < 0.05:
        return "GEO"
    if perigee_km < 2000 and apogee_km > 30000:
        return "GTO/HEO"
    if ecc > 0.5 and 60 < incl_deg < 70:
        return "MOLNIYA"
    if 2000 <= perigee_km and apogee_km <= 34000:
        return "MEO"
    return "OTHER"


def mean_elements(rec: TleRecord) -> MeanElements:
    sat: Satrec = satrec_from_tle(rec.line1, rec.line2)
    n = sat.no_kozai / 60.0                     # rad/min → rad/s
    a = (MU_EARTH_KM3_S2 / n**2) ** (1.0 / 3.0)  # km
    e = sat.ecco
    apo = a * (1 + e) - R_EARTH_KM
    per = a * (1 - e) - R_EARTH_KM
    return MeanElements(
        norad_id=rec.norad_id,
        epoch_iso=rec.epoch,
        semi_major_km=a,
        ecc=e,
        incl_deg=np.degrees(sat.inclo),
        raan_deg=np.degrees(sat.nodeo),
        argp_deg=np.degrees(sat.argpo),
        mean_anom_deg=np.degrees(sat.mo),
        mean_motion_rad_s=n,
        apogee_km=apo,
        perigee_km=per,
        period_min=2 * np.pi / n / 60.0,
        regime=_regime(apo, per, np.degrees(sat.inclo), e),
    )
