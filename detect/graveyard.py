"""Graveyard-orbit disposal-compliance monitoring (bonus §10).

A policy-relevant SDA feature: are objects in the GEO belt disposed of per
the IADC end-of-life guidelines? The IADC "protected region" for GEO is a
band GEO ± ~200 km in altitude and ± 15° in inclination; a decommissioned
GEO satellite is supposed to be re-orbited so its PERIGEE clears that
region — the classic minimum re-orbit altitude is

    ΔH = 235 km + 1000·Cr·(A/m)      (km, A/m in m²/kg)

We use the fixed 235 km IADC floor as the compliance threshold and state
plainly that the A/m-dependent term is NOT added: it needs each object's
mass, which public data doesn't give (SATCAT publishes radar cross-section,
an area proxy, but no mass). So this is a MINIMUM-standard check — an
object below the 235 km floor is unambiguously non-compliant; one just
above it is compliant against the floor but might still fall short of its
own A/m-adjusted requirement. Disclosed, not hidden (§9).

Statuses (per object, from mean elements):
* COMPLIANT_GRAVEYARD — perigee ≥ GEO + 235 km, near-circular: properly
  re-orbited above the protected region.
* NON_COMPLIANT_SHALLOW — raised above GEO but perigee still inside the
  0–235 km disposal buffer: a partial/insufficient re-orbit.
* OPERATIONAL_OR_ABANDONED_IN_PLACE — sitting in the operational GEO box:
  either an active station-kept satellite or one abandoned there (public
  elements alone can't tell active from abandoned — stated, not guessed).
* BELOW_GEO_DRIFT — perigee below GEO and/or high inclination: an
  uncontrolled object librating/drifting through the belt.
* NOT_GEO_BELT — not a GEO-regime object; excluded from the assessment.

Ties back into the rest of the system: BeiDou-2 G2 — the object SJ-21
towed in the F2.1/F3.1 work — comes out COMPLIANT_GRAVEYARD here,
independently confirming it was moved into the disposal region.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import GEO_ALTITUDE_KM
from core.elements import MeanElements, mean_elements
from core.regime import classify_regime

IADC_MIN_REORBIT_KM = 235.0     # IADC minimum re-orbit altitude above GEO (floor)
GEO_PROTECTED_INCL_DEG = 15.0   # protected region inclination half-width
# Operational station-keeping box (where ACTIVE satellites actually sit) is
# much tighter than the ±200 km protected region — real GEO station-keeping
# holds altitude to ~±75 km. An object outside this tight box but still near
# GEO is NOT operational: it's a disposal attempt (if above) or a drifter
# (if below). Found live 2026-07-15: a +120 km object was wrongly read as
# operational when this box was set to the full 200 km protected width.
GEO_OPERATIONAL_BOX_KM = 100.0
NEAR_CIRCULAR_ECC = 0.02
# Inclined geosynchronous orbits (IGSO: QZSS, BeiDou-IGSO, IRNSS, some SDX
# science sats) are a legitimate OPERATIONAL orbit type — geosynchronous
# period but deliberately inclined, so perigee dips below GEO. Found live
# 2026-07-15: without this, ~20 active IGSO satellites were mislabeled
# "uncontrolled drift". A geosynchronous object (period ≈ 1 sidereal day)
# near GEO altitude is operational, not a drifter — its inclination is by
# design, not decay.
from config import SIDEREAL_DAY_S  # noqa: E402
GEO_SYNC_PERIOD_MIN = SIDEREAL_DAY_S / 60.0   # ~1436 min
GEO_SYNC_PERIOD_TOL = 0.03                     # within 3% of a sidereal day
NEAR_GEO_ALT_KM = 1500.0                        # mean altitude this close to GEO


@dataclass(frozen=True)
class DisposalAssessment:
    norad_id: int
    status: str
    perigee_altitude_km: float          # perigee above equatorial radius
    km_above_geo: float                 # perigee minus GEO altitude
    inclination_deg: float
    compliant: Optional[bool]           # True/False, or None when N/A (operational/not-GEO)
    note: str

    def summary(self) -> str:
        return (f"NORAD {self.norad_id}: {self.status} — perigee "
                f"{self.km_above_geo:+.0f} km vs GEO, incl {self.inclination_deg:.1f}°. {self.note}")


def assess_disposal(el: MeanElements) -> DisposalAssessment:
    perigee_alt = el.perigee_km
    apogee_alt = el.apogee_km
    mean_alt = (perigee_alt + apogee_alt) / 2.0
    km_above_geo = perigee_alt - GEO_ALTITUDE_KM
    incl = el.incl_deg

    regime = classify_regime(el)
    is_geo_family = regime.primary in ("GEO", "GRAVEYARD") or (
        abs(mean_alt - GEO_ALTITUDE_KM) < 2000 and el.ecc < 0.1
    )
    if not is_geo_family:
        return DisposalAssessment(
            norad_id=el.norad_id, status="NOT_GEO_BELT", perigee_altitude_km=perigee_alt,
            km_above_geo=km_above_geo, inclination_deg=incl, compliant=None,
            note=f"{regime.label} — not a GEO-belt object, disposal compliance N/A",
        )

    # Properly re-orbited above the protected region.
    if perigee_alt >= GEO_ALTITUDE_KM + IADC_MIN_REORBIT_KM and el.ecc < 0.05:
        return DisposalAssessment(
            norad_id=el.norad_id, status="COMPLIANT_GRAVEYARD", perigee_altitude_km=perigee_alt,
            km_above_geo=km_above_geo, inclination_deg=incl, compliant=True,
            note=f"perigee clears the IADC {IADC_MIN_REORBIT_KM:.0f} km re-orbit floor "
                 "(A/m term not added — see module docstring)",
        )

    # In the tight operational station-keeping box: active or abandoned-in-place.
    in_altitude_box = abs(mean_alt - GEO_ALTITUDE_KM) <= GEO_OPERATIONAL_BOX_KM
    if in_altitude_box and incl <= GEO_PROTECTED_INCL_DEG and el.ecc < NEAR_CIRCULAR_ECC:
        return DisposalAssessment(
            norad_id=el.norad_id, status="OPERATIONAL_OR_ABANDONED_IN_PLACE",
            perigee_altitude_km=perigee_alt, km_above_geo=km_above_geo, inclination_deg=incl,
            compliant=None,
            note="in the operational GEO box — public elements can't tell active from "
                 "abandoned-in-place; not scored",
        )

    # Inclined geosynchronous (IGSO): geosynchronous period, near-GEO
    # altitude, but inclined by design — an operational orbit type, not a
    # disposal case. Must be checked BEFORE the below-GEO-drift fallback,
    # which would otherwise mislabel active QZSS/BeiDou-IGSO/IRNSS sats.
    is_geosynchronous = abs(el.period_min - GEO_SYNC_PERIOD_MIN) / GEO_SYNC_PERIOD_MIN < GEO_SYNC_PERIOD_TOL
    if (is_geosynchronous and abs(mean_alt - GEO_ALTITUDE_KM) < NEAR_GEO_ALT_KM
            and incl > GEO_PROTECTED_INCL_DEG):
        return DisposalAssessment(
            norad_id=el.norad_id, status="INCLINED_GEOSYNCHRONOUS", perigee_altitude_km=perigee_alt,
            km_above_geo=km_above_geo, inclination_deg=incl, compliant=None,
            note=f"inclined geosynchronous ({incl:.0f}°, period ~{el.period_min:.0f} min) — an "
                 "operational orbit type (IGSO/QZSS-class), inclined by design; not a disposal case",
        )

    # Raised above GEO but not far enough — partial/insufficient disposal.
    if 0 < km_above_geo < IADC_MIN_REORBIT_KM:
        return DisposalAssessment(
            norad_id=el.norad_id, status="NON_COMPLIANT_SHALLOW", perigee_altitude_km=perigee_alt,
            km_above_geo=km_above_geo, inclination_deg=incl, compliant=False,
            note=f"re-orbited only {km_above_geo:.0f} km above GEO — inside the "
                 f"{IADC_MIN_REORBIT_KM:.0f} km disposal buffer (insufficient)",
        )

    # Perigee below GEO and/or high inclination: uncontrolled drift.
    return DisposalAssessment(
        norad_id=el.norad_id, status="BELOW_GEO_DRIFT", perigee_altitude_km=perigee_alt,
        km_above_geo=km_above_geo, inclination_deg=incl, compliant=False,
        note="perigee below GEO and/or high inclination — uncontrolled object "
             "librating/drifting through the protected region",
    )


def scan_geo_belt() -> dict[str, list[DisposalAssessment]]:
    """Assess disposal compliance for every live-tracked object, grouped by
    status. The belt-wide policy view: how much of the GEO region is
    properly disposed vs. uncontrolled. Uses only objects we hold a
    current TLE for (whatever's been ingest-celestrak'd)."""
    from data.store import get_latest_tles_many, list_current_objects

    grouped: dict[str, list[DisposalAssessment]] = {}
    current = list_current_objects()
    tles = get_latest_tles_many(c["norad_id"] for c in current)
    for c in current:
        tle = tles.get(c["norad_id"])
        if tle is None:
            continue
        try:
            a = assess_disposal(mean_elements(tle))
        except Exception:
            continue
        grouped.setdefault(a.status, []).append(a)
    return grouped
