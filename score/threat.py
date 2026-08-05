"""Composite threat score (F1.4).

Per handoff §5: one explainable number per object from maneuver tempo,
proximity to high-value assets, estimated maneuver capability, regime, and
ownership — with a MANDATORY per-factor breakdown. The score must explain
itself, never just assert a number (§9).

Five independently-computable factors:

* maneuver_activity  — from F1.1: has this object demonstrated propulsive
                        capability, and how large was its biggest burn?
* anomalous_rate      — from F1.2: what fraction of its OWN maneuver
                        campaigns deviate from its OWN established pattern?
* regime_sensitivity  — coarse tag: GEO/MEO (the GPS/comsat belts) are
                        more strategically sensitive than other regimes.
* proximity_to_hva    — coarse CURRENT-EPOCH range to the nearest
                        high-value asset we hold a live TLE for (ISS, GPS,
                        GEO comsats). Explicitly approximate — instantaneous
                        SGP4 propagation to "now", not a conjunction-grade
                        screening with uncertainty (that's F3.1's job).
* attribution         — SATCAT completeness (known country/object_type vs
                        UNK/blank) as a NEUTRAL transparency signal.
                        Deliberately not a per-nation weighting — §1 rules
                        out anything targeting-flavored, and "we know less
                        about this object" is a defensible, explainable
                        factor where "which country" as a weight is not.

Honesty rule (§9): a factor that can't be computed (thin history, no
stored TLE, no SATCAT row, no high-value-asset pool) is OMITTED, never
defaulted to zero. The composite is a weighted average over only the
AVAILABLE factors, and reports how many of the 5 were actually used, so a
score built on 1/5 factors never reads as confidently as one built on 5/5.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.elements import mean_elements
from core.propagation import propagate_at, satrec_from_tle
from data.store import (
    TleRecord,
    get_latest_tle,
    get_satcat_many,
    get_tles,
    list_high_value_asset_ids,
)
from detect.maneuver import detect_maneuvers
from detect.pattern_of_life import group_into_campaigns, pattern_of_life

# Weight of each factor in the composite (renormalized over whichever are
# actually available for a given object).
WEIGHTS = {
    "maneuver_activity": 0.20,
    "anomalous_rate": 0.30,
    "regime_sensitivity": 0.15,
    "proximity_to_hva": 0.25,
    "attribution": 0.10,
}

MANEUVER_ACTIVITY_SATURATION_MS = 20.0  # campaign Δv at/above this saturates the factor at 1.0
PROXIMITY_NEAR_KM = 50.0    # range at/below this saturates the factor at 1.0
PROXIMITY_FAR_KM = 5000.0   # range at/above this floors the factor at 0.0
PROXIMITY_MAX_TLE_AGE_DAYS = 30.0  # above this, propagating the object's element set to
                                     # "now" is too inaccurate to screen proximity — a stale
                                     # TLE (e.g. an object last tracked years ago in an
                                     # archive window) would produce a meaningless distance,
                                     # so the factor is OMITTED rather than fabricated.
MIN_PHYSICAL_SEPARATION_KM = 2.0  # closer than this = same physical structure, not a
                                    # distinct object (see _proximity_factor docstring)

# Primary-regime base sensitivity, now keyed on the rich classifier
# (core/regime.py) rather than the coarse tag.
REGIME_WEIGHT = {
    "GEO": 1.0,        # congested, high-value comsat belt
    "GRAVEYARD": 0.9,   # disposal region — a MANEUVERING object here is anomalous
    "MEO": 0.6,         # GPS/GNSS belt
    "LEO": 0.5,         # human-spaceflight-adjacent, most-populated regime
    "HEO": 0.5,
    "GTO": 0.4,         # usually a transient transfer orbit
    "DECAYING": 0.3,
    "OTHER": 0.3,
}

# Special-orbit bumps (added on top of the primary weight, capped at 1.0):
# these orbit types carry extra strategic/observational relevance.
REGIME_TAG_BUMP = {
    "MOLNIYA": 0.2,     # classic early-warning / high-latitude comms orbit
    "TUNDRA": 0.2,      # same strategic family, 24h period
    "SSO": 0.15,        # sun-synchronous — the reconnaissance/EO orbit
}


@dataclass(frozen=True)
class ThreatFactor:
    name: str
    value: Optional[float]     # 0..1, or None if not computable
    weight: float
    explanation: str


@dataclass(frozen=True)
class ThreatScore:
    norad_id: int
    score: Optional[float]     # 0..100, or None if NO factors were available
    factors: tuple[ThreatFactor, ...]
    n_factors_available: int
    n_factors_total: int

    def summary(self) -> str:
        head = "n/a" if self.score is None else f"{self.score:.0f}/100"
        lines = [f"NORAD {self.norad_id}: threat score {head} "
                 f"({self.n_factors_available}/{self.n_factors_total} factors available)"]
        for f in self.factors:
            v = "n/a" if f.value is None else f"{f.value:.2f}"
            lines.append(f"  - {f.name} (w={f.weight:.2f}): {v} — {f.explanation}")
        return "\n".join(lines)


def _maneuver_activity_factor(events, noise_floor: Optional[float]) -> ThreatFactor:
    if noise_floor is None:
        return ThreatFactor("maneuver_activity", None, WEIGHTS["maneuver_activity"],
                             "insufficient TLE history to establish a maneuver baseline")
    if not events:
        return ThreatFactor("maneuver_activity", 0.0, WEIGHTS["maneuver_activity"],
                             "no anomalous maneuvers detected against its own noise floor")
    campaigns = group_into_campaigns(events)
    peak = max(c.peak_dv_ms for c in campaigns)
    value = min(1.0, peak / MANEUVER_ACTIVITY_SATURATION_MS)
    return ThreatFactor("maneuver_activity", value, WEIGHTS["maneuver_activity"],
                         f"{len(campaigns)} maneuver campaign(s) detected, largest {peak:.2f} m/s")


def _anomalous_rate_factor(classifications, baseline) -> ThreatFactor:
    if baseline is None:
        return ThreatFactor("anomalous_rate", None, WEIGHTS["anomalous_rate"],
                             "fewer than 3 maneuver campaigns — no pattern-of-life baseline yet")
    n_anom = sum(1 for c in classifications if c.label == "anomalous")
    value = n_anom / len(classifications)
    return ThreatFactor("anomalous_rate", value, WEIGHTS["anomalous_rate"],
                         f"{n_anom}/{len(classifications)} maneuver campaigns deviate from "
                         "this object's own established pattern")


def _tle_epoch(rec: TleRecord) -> datetime:
    return datetime.fromisoformat(rec.epoch.replace("Z", "+00:00"))


def _regime_factor(latest_tle: Optional[TleRecord]) -> ThreatFactor:
    if latest_tle is None:
        return ThreatFactor("regime_sensitivity", None, WEIGHTS["regime_sensitivity"],
                             "no stored TLE — orbital regime unknown")
    from core.regime import classify_regime
    c = classify_regime(mean_elements(latest_tle))
    value = REGIME_WEIGHT.get(c.primary, 0.3)
    bump = max((REGIME_TAG_BUMP.get(t, 0.0) for t in c.special_tags), default=0.0)
    value = min(1.0, value + bump)
    return ThreatFactor("regime_sensitivity", value, WEIGHTS["regime_sensitivity"],
                         f"regime {c.label}")


def _proximity_factor(norad_id: int, latest_tle: Optional[TleRecord]) -> ThreatFactor:
    if latest_tle is None:
        return ThreatFactor("proximity_to_hva", None, WEIGHTS["proximity_to_hva"],
                             "no stored TLE — cannot screen against high-value assets")

    now = datetime.now(timezone.utc)
    age_days = abs((now - _tle_epoch(latest_tle)).total_seconds()) / 86400.0
    if age_days > PROXIMITY_MAX_TLE_AGE_DAYS:
        return ThreatFactor("proximity_to_hva", None, WEIGHTS["proximity_to_hva"],
                             f"latest element set is {age_days:.0f} days old — the object's "
                             "current position is unknown, so proximity is not screened "
                             "(propagating a stale TLE to now would be meaningless)")
    # Evaluate at the object's OWN element epoch, not wall-clock "now": zero
    # propagation error, and DETERMINISTIC — evaluating at "now" made the
    # score jump run-to-run for fast/eccentric orbits (found live: Vanguard 1
    # swung 10→52 as it moved along its orbit between evaluations).
    eval_epoch = _tle_epoch(latest_tle)
    try:
        r_own, _ = propagate_at(satrec_from_tle(latest_tle.line1, latest_tle.line2), eval_epoch)
    except RuntimeError:
        return ThreatFactor("proximity_to_hva", None, WEIGHTS["proximity_to_hva"],
                             "SGP4 propagation failed for this object's latest TLE")

    hva_ids = list_high_value_asset_ids(exclude_norad_id=norad_id)
    if not hva_ids:
        return ThreatFactor("proximity_to_hva", None, WEIGHTS["proximity_to_hva"],
                             "no high-value-asset pool available locally "
                             "(ingest-celestrak --group geo --group gps-ops --group stations)")
    satcat = get_satcat_many(hva_ids)

    min_range: Optional[float] = None
    nearest_name = None
    for hid in hva_ids:
        hva_tle = get_latest_tle(hid)
        if hva_tle is None:
            continue
        try:
            r_hva, _ = propagate_at(satrec_from_tle(hva_tle.line1, hva_tle.line2), eval_epoch)
        except RuntimeError:
            continue
        rng = float(np.linalg.norm(r_own - r_hva))
        # Sub-MIN_PHYSICAL_SEPARATION_KM at single-epoch granularity is,
        # for real catalogued objects, essentially always the SAME docked
        # structure rather than a distinct free-flying object — ISS itself
        # only spans ~100 m end to end. Found live 2026-07-15: scoring ISS
        # itself "found" ~0 km to its own module (first POISK, then NAUKA
        # once POISK was excluded by a country-code heuristic that turned
        # out inconsistent across modules — CIS vs ISS vs US vs JPN). A
        # distance floor is the general, robust fix; genuine close-approach
        # behavior over multiple passes is F2.1's job, not this coarse
        # single-snapshot screen's.
        if rng < MIN_PHYSICAL_SEPARATION_KM:
            continue
        if min_range is None or rng < min_range:
            min_range = rng
            nearest_name = satcat.get(hid, {}).get("name", str(hid))

    if min_range is None:
        return ThreatFactor("proximity_to_hva", None, WEIGHTS["proximity_to_hva"],
                             "no high-value asset beyond same-structure range "
                             f"({MIN_PHYSICAL_SEPARATION_KM:.0f} km) had a propagatable TLE")

    if min_range <= PROXIMITY_NEAR_KM:
        value = 1.0
    elif min_range >= PROXIMITY_FAR_KM:
        value = 0.0
    else:
        value = 1.0 - (min_range - PROXIMITY_NEAR_KM) / (PROXIMITY_FAR_KM - PROXIMITY_NEAR_KM)

    return ThreatFactor("proximity_to_hva", value, WEIGHTS["proximity_to_hva"],
                         f"~{min_range:.0f} km from nearest high-value asset ({nearest_name}) "
                         "at evaluation epoch — coarse instantaneous screening, "
                         "NOT conjunction-grade (see F3.1)")


def _attribution_factor(norad_id: int) -> ThreatFactor:
    row = get_satcat_many([norad_id]).get(norad_id)
    if row is None:
        return ThreatFactor("attribution", None, WEIGHTS["attribution"],
                             "no SATCAT record for this object")
    known_country = bool(row.get("country")) and row["country"] not in ("", "UNK")
    known_type = row.get("object_type") not in (None, "", "UNK")
    if known_country and known_type:
        value, note = 0.0, f"fully attributed: {row['country']}, type {row['object_type']}"
    elif known_country or known_type:
        value, note = 0.5, "partially attributed catalog record"
    else:
        value, note = 1.0, "unattributed: country and/or object type unknown in SATCAT"
    return ThreatFactor("attribution", value, WEIGHTS["attribution"], note)


def combine_factors(factors: tuple[ThreatFactor, ...]) -> tuple[Optional[float], int]:
    """Weighted average over only the AVAILABLE factors, renormalized —
    the honesty-rule core of F1.4 (§9), pulled out as a pure function so
    the combination logic is testable without touching the DB. Returns
    (score_0_to_100_or_None, n_available).
    """
    available = [f for f in factors if f.value is not None]
    if not available:
        return None, 0
    total_w = sum(f.weight for f in available)
    score = 100.0 * sum(f.value * f.weight for f in available) / total_w
    return score, len(available)


def compute_threat_score(norad_id: int) -> ThreatScore:
    tles = get_tles(norad_id)
    latest_tle = tles[-1] if tles else None

    events, noise_floor = detect_maneuvers(tles)
    classifications, baseline = pattern_of_life(events)

    factors = (
        _maneuver_activity_factor(events, noise_floor),
        _anomalous_rate_factor(classifications, baseline),
        _regime_factor(latest_tle),
        _proximity_factor(norad_id, latest_tle),
        _attribution_factor(norad_id),
    )

    score, n_available = combine_factors(factors)

    return ThreatScore(
        norad_id=norad_id,
        score=score,
        factors=factors,
        n_factors_available=n_available,
        n_factors_total=len(factors),
    )
