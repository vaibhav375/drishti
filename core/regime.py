"""Orbital regime auto-classification (bonus §10).

Refines the coarse `core/elements.py::_regime` tag into a physics-based
classification: a primary regime (LEO/MEO/GEO/HEO/GTO) plus special-orbit
tags (SSO, MOLNIYA, TUNDRA, GRAVEYARD, POLAR, EQUATORIAL, RETROGRADE),
each derived from real orbital mechanics rather than hard-coded altitude
boxes. "Cheap, improves every other feature's context" (§10) — regime
feeds the threat score (F1.4), the conjunction cascade (F2.2), and
graveyard-compliance monitoring.

The physics that matters here, and why it's not just altitude bins:

* Sun-synchronous (SSO): NOT an altitude — it's a J2-driven nodal
  precession that keeps the orbit plane fixed relative to the Sun. The
  required inclination is a specific function of a and e (retrograde,
  ~96-102° for typical LEO altitudes). We compute the exact SSO
  inclination for THIS orbit's a,e and check the object is within
  tolerance — an 800 km orbit at 98.6° is SSO; the same altitude at 45°
  is not.
* Molniya vs Tundra: both sit at the ~63.4° critical inclination (where
  J2 stops rotating the argument of perigee, so apogee stays parked over
  one hemisphere), but Molniya is a half-sidereal-day period (~12h) and
  Tundra a full-sidereal-day period (~24h). Period, not altitude,
  separates them.
* Graveyard: a near-circular orbit a few hundred km ABOVE GEO — the IADC
  disposal region. Distinguished from operational GEO by altitude margin,
  which is exactly what graveyard-compliance monitoring keys on.

Honesty note (§9): this is deterministic classification off mean
elements, not a probabilistic label — every tag traces to a stated
numeric criterion, surfaced in `explanation`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    EARTH_ORBIT_RAD_PER_S,
    GEO_ALTITUDE_KM,
    J2_EARTH,
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    SIDEREAL_DAY_S,
)
from core.elements import MeanElements

# --- classification tolerances (documented, not magic) ------------------
SSO_INCL_TOLERANCE_DEG = 1.5    # how close to the computed SSO inclination counts
CRITICAL_INCL_DEG = 63.4        # Molniya/Tundra critical inclination
CRITICAL_INCL_TOLERANCE_DEG = 2.5
PERIOD_MATCH_TOLERANCE = 0.05   # ±5% of a target period counts as a match
GEO_INCL_MAX_DEG = 15.0         # station-kept GEO stays low-inclination
GEO_ECC_MAX = 0.02              # GEO is near-circular
GEO_ALTITUDE_TOLERANCE_KM = 300.0
GRAVEYARD_MIN_MARGIN_KM = 150.0  # disposal orbits sit at least this far above GEO
HEO_ECC_MIN = 0.25              # "highly elliptical" threshold
POLAR_INCL_DEG = (80.0, 100.0)  # near-polar band
EQUATORIAL_INCL_DEG = 5.0       # near-equatorial band


@dataclass(frozen=True)
class RegimeClassification:
    norad_id: int
    primary: str                 # LEO | MEO | GEO | HEO | GTO | GRAVEYARD | DECAYING
    special_tags: tuple[str, ...]
    period_min: float
    explanation: str

    @property
    def label(self) -> str:
        if self.special_tags:
            return f"{self.primary} [{', '.join(self.special_tags)}]"
        return self.primary

    def summary(self) -> str:
        return f"NORAD {self.norad_id}: {self.label} — {self.explanation}"


def required_sso_inclination_deg(semi_major_km: float, ecc: float) -> float | None:
    """The inclination that makes THIS orbit sun-synchronous, from J2
    nodal precession Ω̇ = −1.5·n·J2·(Re/p)²·cos(i) set equal to Earth's
    mean orbital rate. Returns None if no real inclination satisfies it
    (orbit too high/eccentric for J2 to precess fast enough — SSO simply
    isn't achievable there)."""
    n = np.sqrt(MU_EARTH_KM3_S2 / semi_major_km ** 3)   # rad/s
    p = semi_major_km * (1 - ecc ** 2)
    denom = 1.5 * n * J2_EARTH * (R_EARTH_KM / p) ** 2
    cos_i = -EARTH_ORBIT_RAD_PER_S / denom
    if not -1.0 <= cos_i <= 1.0:
        return None
    return float(np.degrees(np.arccos(cos_i)))


def _special_tags(el: MeanElements, period_min: float) -> tuple[list[str], list[str]]:
    """Returns (tags, reasons)."""
    tags: list[str] = []
    reasons: list[str] = []
    incl = el.incl_deg

    # Sun-synchronous — compare to the computed SSO inclination for this a,e.
    i_sso = required_sso_inclination_deg(el.semi_major_km, el.ecc)
    if i_sso is not None and abs(incl - i_sso) <= SSO_INCL_TOLERANCE_DEG and el.apogee_km < 2000:
        tags.append("SSO")
        reasons.append(f"inclination {incl:.1f}° matches sun-synchronous {i_sso:.1f}° for this altitude")

    # Critical-inclination frozen-apogee orbits: Molniya (12h) vs Tundra (24h).
    at_critical = abs(incl - CRITICAL_INCL_DEG) <= CRITICAL_INCL_TOLERANCE_DEG
    if at_critical and el.ecc > 0.15:
        half_day_min = SIDEREAL_DAY_S / 2 / 60.0
        full_day_min = SIDEREAL_DAY_S / 60.0
        if abs(period_min - half_day_min) / half_day_min <= PERIOD_MATCH_TOLERANCE:
            tags.append("MOLNIYA")
            reasons.append(f"critical incl {incl:.1f}°, ~12h period, ecc {el.ecc:.2f}")
        elif abs(period_min - full_day_min) / full_day_min <= PERIOD_MATCH_TOLERANCE:
            tags.append("TUNDRA")
            reasons.append(f"critical incl {incl:.1f}°, ~24h period, ecc {el.ecc:.2f}")

    # Orientation tags.
    if incl > 90.0 and "SSO" not in tags:
        tags.append("RETROGRADE")
        reasons.append(f"inclination {incl:.1f}° > 90° (retrograde)")
    if POLAR_INCL_DEG[0] <= incl <= POLAR_INCL_DEG[1]:
        tags.append("POLAR")
        reasons.append(f"near-polar inclination {incl:.1f}°")
    elif incl <= EQUATORIAL_INCL_DEG:
        tags.append("EQUATORIAL")
        reasons.append(f"near-equatorial inclination {incl:.1f}°")

    return tags, reasons


def classify_regime(el: MeanElements) -> RegimeClassification:
    apo, per, ecc = el.apogee_km, el.perigee_km, el.ecc
    period_min = el.period_min
    geo_alt = GEO_ALTITUDE_KM

    tags, reasons = _special_tags(el, period_min)

    # --- primary regime -------------------------------------------------
    if per < 150:
        primary = "DECAYING"
        reasons.insert(0, f"perigee {per:.0f} km is in the drag-dominated decay zone")
    elif per > geo_alt + GRAVEYARD_MIN_MARGIN_KM and ecc < 0.05:
        primary = "GRAVEYARD"
        reasons.insert(0, f"near-circular, ~{per - geo_alt:.0f} km above GEO (disposal region)")
    elif (abs((apo + per) / 2 - geo_alt) < GEO_ALTITUDE_TOLERANCE_KM
          and ecc < GEO_ECC_MAX and el.incl_deg < GEO_INCL_MAX_DEG):
        primary = "GEO"
        reasons.insert(0, f"near-circular at GEO altitude, inclination {el.incl_deg:.1f}°")
    elif ecc >= HEO_ECC_MIN and apo > 20000:
        # Highly elliptical. Molniya/Tundra are operational HEO at the
        # critical inclination — NOT transfer orbits — even though they
        # share the LEO-perigee/GEO-apogee band with GTO. Inclination is
        # the real discriminator: a genuine GTO is a low-inclination
        # temporary transfer (launched near-equatorial); a high-incl,
        # high-ecc orbit is Molniya-class HEO. Found live 2026-07-15:
        # real Molniya 1-29 (64° incl) was mislabeled GTO without this.
        is_critical_incl = ("MOLNIYA" in tags) or ("TUNDRA" in tags)
        if per < 2000 and abs(apo - geo_alt) < 5000 and el.incl_deg < 30 and not is_critical_incl:
            primary = "GTO"
            reasons.insert(0, f"transfer orbit: LEO perigee {per:.0f} km, near-GEO apogee {apo:.0f} km, low inclination")
        else:
            primary = "HEO"
            reasons.insert(0, f"highly elliptical (ecc {ecc:.2f}), apogee {apo:.0f} km")
    elif apo < 2000:
        primary = "LEO"
        reasons.insert(0, f"apogee {apo:.0f} km (low Earth orbit)")
    elif per >= 2000 and apo <= geo_alt + GEO_ALTITUDE_TOLERANCE_KM:
        primary = "MEO"
        reasons.insert(0, f"medium Earth orbit (perigee {per:.0f} km, apogee {apo:.0f} km)")
    else:
        primary = "OTHER"
        reasons.insert(0, f"perigee {per:.0f} km, apogee {apo:.0f} km, ecc {ecc:.2f}")

    return RegimeClassification(
        norad_id=el.norad_id,
        primary=primary,
        special_tags=tuple(tags),
        period_min=period_min,
        explanation="; ".join(reasons),
    )
