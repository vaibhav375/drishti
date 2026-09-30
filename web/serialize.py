"""JSON serialization for the DRISHTI API (roadmap feature 4).

Pure functions that turn the domain objects into JSON-safe dicts, so the
whole analytic core is consumable programmatically (the "APIs / workflows"
another tool can build on). Every field is a computed value — the API
exposes exactly what the UI shows, nothing invented. Kept as pure
functions (no Flask) so they're unit-testable offline.
"""
from __future__ import annotations

from typing import Optional


def factor_json(f) -> dict:
    return {"name": f.name, "value": f.value, "weight": f.weight, "explanation": f.explanation}


def threat_json(t) -> Optional[dict]:
    if t is None:
        return None
    return {
        "score": t.score,
        "factors_available": t.n_factors_available,
        "factors_total": t.n_factors_total,
        "factors": [factor_json(f) for f in t.factors],
    }


def object_json(d: dict) -> dict:
    """From web.characterize.characterize(...) → JSON-safe dict."""
    el = d.get("elements")
    regime = d.get("regime")
    rpo = d.get("rpo")
    decay = d.get("decay")
    disposal = d.get("disposal")
    cp = d.get("changepoint")
    return {
        "norad_id": d["norad_id"],
        "name": d["name"],
        "decayed": d.get("decayed", False),
        "decay_date": d.get("decay_date"),
        "has_tle": d.get("has_tle", False),
        "regime": ({"primary": regime.primary, "special_tags": list(regime.special_tags),
                    "label": regime.label, "period_min": regime.period_min}
                   if regime is not None else None),
        "elements": ({"perigee_km": el.perigee_km, "apogee_km": el.apogee_km,
                      "inclination_deg": el.incl_deg, "eccentricity": el.ecc,
                      "period_min": el.period_min} if el is not None else None),
        "threat": threat_json(d.get("threat")),
        "n_maneuvers": len(d.get("maneuvers", [])),
        "rpo": ({"label": rpo.label, "min_range_km": rpo.min_range_km,
                 "n_episodes": rpo.n_episodes, "total_close_days": rpo.total_close_days,
                 "target_norad": rpo.target_norad, "same_launch_lineage": rpo.same_launch_lineage}
                if rpo is not None else None),
        "decay": ({"perigee_km": decay.latest_perigee_km,
                   "decay_rate_km_per_day": decay.decay_rate_km_per_day,
                   "predicted_reentry": decay.predicted_reentry_iso,
                   "days_to_reentry": decay.days_to_reentry, "model": decay.model}
                  if decay is not None else None),
        "disposal": ({"status": disposal.status, "km_above_geo": disposal.km_above_geo,
                      "compliant": disposal.compliant} if disposal is not None else None),
        "changepoint": ({"metric": cp.metric, "direction": cp.direction,
                         "before": cp.before_mean, "after": cp.after_mean,
                         "unit": cp.unit.strip(), "epoch": cp.epoch_iso, "t_stat": cp.t_stat}
                        if cp is not None else None),
        "mitigations": [{"title": m.title, "category": m.category, "urgency": m.urgency,
                         "rationale": m.rationale} for m in d.get("mitigations", [])],
    }


def conjunction_cdm(assessment, name_a: str, name_b: str) -> str:
    """A CCSDS-CDM-STYLE conjunction data message from a Pc assessment.

    Honest framing: this mirrors the CCSDS Conjunction Data Message layout
    for interoperability, but the covariance is MODELED (grows with TLE age)
    and the hard-body radius is RCS-derived — it is NOT an operational CDM
    from tracked covariance. Stated in the header comment so a downstream
    consumer can't mistake it for one."""
    a = assessment
    return "\n".join([
        "CCSDS_CDM_VERS = 1.0",
        "COMMENT DRISHTI screening product — MODELED covariance (grows with TLE age),",
        "COMMENT RCS-derived hard-body radius. NOT an operational CDM from tracked covariance.",
        f"CREATION_DATE = {_now()}",
        "ORIGINATOR = DRISHTI",
        f"TCA = {a.tca_iso}",
        f"MISS_DISTANCE = {a.min_range_km * 1000:.1f} [m]",
        f"COLLISION_PROBABILITY = {a.pc_estimate:.3e}",
        f"HARD_BODY_RADIUS = {a.hard_body_radius_km * 1000:.1f} [m]",
        "OBJECT = OBJECT1",
        f"OBJECT_DESIGNATOR = {a.actor_norad}",
        f"OBJECT_NAME = {name_a}",
        "OBJECT = OBJECT2",
        f"OBJECT_DESIGNATOR = {a.target_norad}",
        f"OBJECT_NAME = {name_b}",
    ])


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
