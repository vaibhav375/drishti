"""Decision-support mitigation recommendations (bonus).

Given an object's full characterization — threat score + factors, regime,
RPO status, decay, disposal — produce a ranked list of RECOMMENDED
RESPONSES. These are the standard defensive/space-safety playbook of a
space-situational-awareness cell: task more tracking, characterize an
unknown, plan a collision-avoidance window, notify an operator, follow up
on a disposal violation, monitor a re-entry.

Framing discipline (§1, non-negotiable): every action here is
DECISION-SUPPORT and SPACE-SAFETY — increasing awareness, protecting
assets, informing operators and registries. Nothing here is an
engagement, an intercept, or any action against another object. That
boundary is the project's ethical core and it is enforced by the fixed
action vocabulary below — there is no rule that emits an offensive action
because no such action exists in the catalogue.

Grounded (§9): each recommendation is rule-triggered off the actual
computed factor values and states, and carries a rationale that cites
them. Nothing is invented; a low-threat, well-attributed, stable object
yields only routine custody.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# urgency ordering for sorting (higher = more urgent)
_URGENCY_RANK = {"priority": 3, "elevated": 2, "routine": 1}


def _fmt_range(km: float) -> str:
    """Sub-kilometre ranges read better in metres — a 36 m closest approach
    should not display as '0.0 km'."""
    return f"{km * 1000:.0f} m" if km < 1.0 else f"{km:.1f} km"


@dataclass(frozen=True)
class MitigationAction:
    action_id: str
    title: str
    category: str        # tracking | characterization | collision-avoidance | policy | monitoring
    urgency: str         # priority | elevated | routine
    rationale: str

    def summary(self) -> str:
        return f"[{self.urgency.upper()}] {self.title} — {self.rationale}"


def _factor(threat_score, name: str) -> Optional[float]:
    for f in threat_score.factors:
        if f.name == name:
            return f.value
    return None


def recommend_mitigations(
    threat_score,
    regime=None,
    rpo=None,
    decay=None,
    disposal=None,
) -> list[MitigationAction]:
    """Assemble decision-support actions from whatever characterization is
    available. All inputs except `threat_score` are optional — an action
    only fires when the data that would justify it is present."""
    actions: list[MitigationAction] = []

    maneuver = _factor(threat_score, "maneuver_activity")
    anomalous = _factor(threat_score, "anomalous_rate")
    proximity = _factor(threat_score, "proximity_to_hva")
    attribution = _factor(threat_score, "attribution")

    # --- proximity + capability → persistent custody -------------------
    if maneuver is not None and proximity is not None and maneuver >= 0.5 and proximity >= 0.4:
        actions.append(MitigationAction(
            "persistent_custody",
            "Place object under persistent tracking custody",
            "tracking", "priority",
            f"demonstrated maneuver capability (factor {maneuver:.2f}) combined with proximity "
            f"to a high-value asset (factor {proximity:.2f}) — sustained custody keeps its "
            "orbit solution fresh enough to see the next maneuver early",
        ))
    elif proximity is not None and proximity >= 0.6:
        actions.append(MitigationAction(
            "elevate_cadence",
            "Increase observation cadence on this object",
            "tracking", "elevated",
            f"currently close to a high-value asset (proximity factor {proximity:.2f}); denser "
            "observations tighten the covariance for any conjunction assessment",
        ))

    # --- RPO → collision-avoidance + operator notification -------------
    if rpo is not None and rpo.label in ("likely", "possible"):
        if getattr(rpo, "same_launch_lineage", False):
            actions.append(MitigationAction(
                "confirm_deployment",
                "Confirm deployment vs. inspection before escalating",
                "characterization", "elevated",
                f"sustained close approach (min {_fmt_range(rpo.min_range_km)}) but actor and target "
                "share a launch designator — likely a sub-satellite deployment; verify against "
                "launch records before treating as an inspection",
            ))
        else:
            actions.append(MitigationAction(
                "notify_operator",
                "Notify the target operator and prepare a collision-avoidance window",
                "collision-avoidance", "priority",
                f"sustained proximity operations detected (min range {_fmt_range(rpo.min_range_km)}, "
                f"{rpo.total_close_days:.0f} days close) — the target operator should have "
                "custody of this relationship and a screened avoidance option ready",
            ))

    # --- poor attribution → characterization tasking -------------------
    if attribution is not None and attribution >= 0.5:
        actions.append(MitigationAction(
            "characterize_unknown",
            "Task optical/radar characterization of the object",
            "characterization", "elevated",
            "object is only partially attributed in the public catalogue; dedicated "
            "characterization would establish type, size, and likely operator",
        ))

    # --- anomalous maneuver tempo → watch-list -------------------------
    if anomalous is not None and anomalous >= 0.3:
        actions.append(MitigationAction(
            "maneuver_watchlist",
            "Add to the anomalous-maneuver watch-list",
            "monitoring", "elevated",
            f"{anomalous:.0%} of this object's own maneuver campaigns deviate from its "
            "established pattern of life — worth a standing alert on the next deviation",
        ))

    # --- disposal non-compliance → policy follow-up --------------------
    if disposal is not None and disposal.compliant is False:
        actions.append(MitigationAction(
            "disposal_followup",
            "Flag for disposal-compliance follow-up with operator/registry",
            "policy", "elevated",
            f"GEO disposal status is {disposal.status} ({disposal.km_above_geo:+.0f} km vs GEO) — "
            "outside the IADC re-orbit guideline; a registry/operator note is the appropriate "
            "space-safety response",
        ))

    # --- decay → re-entry monitoring -----------------------------------
    if decay is not None and decay.days_to_reentry is not None:
        days = decay.days_to_reentry
        urgency = "priority" if days < 30 else ("elevated" if days < 120 else "routine")
        actions.append(MitigationAction(
            "reentry_monitoring",
            "Initiate re-entry monitoring and issue a re-entry advisory",
            "monitoring", urgency,
            f"perigee decaying ~{decay.decay_rate_km_per_day:.2f} km/day; predicted re-entry in "
            f"~{days:.0f} days — track the plunge and advise on the debris-footprint window",
        ))

    # --- floor: always something -------------------------------------
    if not actions:
        actions.append(MitigationAction(
            "routine_custody",
            "Maintain routine catalogue custody",
            "tracking", "routine",
            "no elevated indicators — nominal object; standard catalogue tracking is sufficient",
        ))

    actions.sort(key=lambda a: _URGENCY_RANK[a.urgency], reverse=True)
    return actions


# Real, public coordination frameworks a case flows through, keyed by the
# mitigation categories present. These are FACTUAL references to how space
# situational-awareness is coordinated in the open — NOT claims that any
# organization has taken a specific action (that would be fabrication, §9).
_FRAMEWORKS = {
    "collision-avoidance": (
        "Conjunction data coordination",
        "Operator collision-avoidance is coordinated through CDM-style conjunction "
        "messages; the U.S. Space Force publishes screenings and CDMs via Space-Track, "
        "and satellite operators run their own avoidance planning."),
    "policy": (
        "Debris-mitigation & registration",
        "Disposal is governed by the IADC space-debris mitigation guidelines; launching "
        "states register objects with the UN Register of Objects Launched into Outer "
        "Space (UNOOSA)."),
    "characterization": (
        "Sensor tasking",
        "Uncharacterized objects are handed to an SSA sensor network (radar / optical) "
        "for dedicated tracking and characterization."),
    "monitoring": (
        "Standing watch & advisories",
        "Persistent watch-items and re-entry advisories are issued through national SSA "
        "cells and IADC re-entry campaigns."),
    "tracking": (
        "Catalogue custody",
        "Routine and elevated tracking custody is maintained by the cataloguing "
        "authority (e.g. the U.S. Space Surveillance Network, feeding Space-Track)."),
}


@dataclass(frozen=True)
class ResponsePosture:
    level: str                       # priority | elevated | routine
    frameworks: tuple[tuple[str, str], ...]   # (name, description) real channels


def response_posture(actions: list[MitigationAction]) -> ResponsePosture:
    """Aggregate a set of recommended mitigations into an overall response
    posture and the REAL public coordination channels a case like this is
    handled through — context, not claimed actions."""
    if not actions:
        level = "routine"
    else:
        level = max(actions, key=lambda a: _URGENCY_RANK[a.urgency]).urgency
    cats: list[str] = []
    for a in actions:
        if a.category not in cats:
            cats.append(a.category)
    frameworks = tuple(_FRAMEWORKS[c] for c in cats if c in _FRAMEWORKS)
    return ResponsePosture(level=level, frameworks=frameworks)
