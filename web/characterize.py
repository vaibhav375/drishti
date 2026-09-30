"""Unified per-object characterization for the web console.

Runs every applicable detector for one object and assembles a single
structured result — the "detailed threat report" the dossier page renders.
Honest by construction: sections that need data the object doesn't have
(no TLE history → no maneuvers/decay; not GEO → no disposal) are simply
absent, never faked.
"""
from __future__ import annotations

from typing import Optional

from core.elements import mean_elements
from core.regime import classify_regime
from data.store import get_latest_tle, get_satcat_many, get_tles
from detect.decay import predict_decay
from detect.graveyard import assess_disposal
from detect.maneuver import detect_maneuvers
from detect.pattern_of_life import pattern_of_life
from detect.rpo import assess_rpo
from score.mitigations import recommend_mitigations
from score.threat import compute_threat_score
from validate.events import EVENTS


def _known_rpo_target(norad_id: int) -> Optional[int]:
    """If this object is a listed actor in a validation RPO event, return
    its first target that we actually hold TLE history for."""
    for e in EVENTS:
        if e.event_type == "rpo" and norad_id in e.actor_norad:
            for t in e.target_norad:
                if get_tles(t):
                    return t
    return None


def characterize(norad_id: int) -> dict:
    satcat = get_satcat_many([norad_id]).get(norad_id)
    latest = get_latest_tle(norad_id)
    tles = get_tles(norad_id)

    decay_date = (satcat or {}).get("decay_date")
    result: dict = {
        "norad_id": norad_id,
        "satcat": satcat,
        "name": (satcat or {}).get("name") or f"NORAD {norad_id}",
        "decayed": bool(decay_date),
        "decay_date": decay_date,
        "has_tle": latest is not None,
        "n_tles": len(tles),
        "elements": None,
        "regime": None,
        "threat": None,
        "maneuvers": [],
        "pattern": None,
        "changepoint": None,
        "decay": None,
        "disposal": None,
        "rpo": None,
        "mitigations": [],
    }

    if latest is None:
        return result

    el = mean_elements(latest)
    regime = classify_regime(el)
    result["elements"] = el
    result["regime"] = regime

    threat = compute_threat_score(norad_id)
    result["threat"] = threat

    events, floor = detect_maneuvers(tles)
    result["maneuvers"] = events
    result["noise_floor"] = floor
    result["changepoint"] = None
    if events:
        classifications, baseline = pattern_of_life(events)
        if baseline is not None:
            result["pattern"] = {"baseline": baseline, "classifications": classifications}
        from detect.changepoint import detect_changepoint
        from detect.pattern_of_life import group_into_campaigns
        result["changepoint"] = detect_changepoint(group_into_campaigns(events))

    decay = predict_decay(tles)
    if decay is not None and decay.predicted_reentry_iso is not None:
        result["decay"] = decay

    if regime.primary in ("GEO", "GRAVEYARD"):
        result["disposal"] = assess_disposal(el)

    target = _known_rpo_target(norad_id)
    if target is not None:
        target_tles = get_tles(target)
        rpo = assess_rpo(norad_id, target, tles, target_tles)
        if rpo is not None:
            result["rpo"] = rpo
            result["rpo_target_name"] = (get_satcat_many([target]).get(target, {}) or {}).get("name")
            # relative-range series for the 'how close' plot
            from core.frames import relative_range_series
            from core.propagation import propagate_aligned
            # 6 h step matches assess_rpo's default, so the plotted minimum
            # agrees with the reported min_range_km (no header/plot mismatch)
            times, r_a, v_a, r_b, v_b = propagate_aligned(tles, target_tles, step_hours=6.0)
            if len(times) >= 2:
                rel = relative_range_series(r_a, v_a, r_b, v_b)
                t0 = times[0]
                result["rpo_series"] = {
                    "days": [(t - t0).total_seconds() / 86400.0 for t in times],
                    "ranges_km": [float(x) for x in rel["range_km"]],
                }

    result["mitigations"] = recommend_mitigations(
        threat, regime=regime, rpo=result["rpo"], decay=result["decay"],
        disposal=result["disposal"],
    )
    return result
