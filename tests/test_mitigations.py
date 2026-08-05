"""Tests for decision-support mitigation recommendations (score/mitigations.py).

Offline, synthetic threat profiles. Uses lightweight stand-ins for the
characterization objects (only the attributes the rules read), so the
recommender is tested independent of the detectors that produce them.
"""
from __future__ import annotations

from dataclasses import dataclass

from score.mitigations import recommend_mitigations, response_posture
from score.threat import ThreatFactor, ThreatScore


def _score(**factor_values):
    factors = tuple(
        ThreatFactor(name, val, 0.2, "") for name, val in factor_values.items()
    )
    return ThreatScore(norad_id=1, score=50.0, factors=factors,
                       n_factors_available=len(factors), n_factors_total=5)


@dataclass
class _Rpo:
    label: str
    min_range_km: float
    total_close_days: float
    same_launch_lineage: bool = False


@dataclass
class _Decay:
    days_to_reentry: float
    decay_rate_km_per_day: float


@dataclass
class _Disposal:
    status: str
    km_above_geo: float
    compliant: bool


def test_low_threat_object_gets_only_routine_custody():
    actions = recommend_mitigations(_score(maneuver_activity=0.0, proximity_to_hva=0.0,
                                            attribution=0.0, anomalous_rate=0.0))
    assert len(actions) == 1
    assert actions[0].action_id == "routine_custody"
    assert actions[0].urgency == "routine"


def test_capability_plus_proximity_triggers_priority_custody():
    actions = recommend_mitigations(_score(maneuver_activity=1.0, proximity_to_hva=0.6))
    ids = [a.action_id for a in actions]
    assert "persistent_custody" in ids
    assert actions[0].urgency == "priority"  # sorted, priority first


def test_likely_rpo_recommends_operator_notification():
    rpo = _Rpo(label="likely", min_range_km=0.04, total_close_days=19)
    actions = recommend_mitigations(_score(maneuver_activity=1.0), rpo=rpo)
    ids = [a.action_id for a in actions]
    assert "notify_operator" in ids
    notify = next(a for a in actions if a.action_id == "notify_operator")
    assert notify.category == "collision-avoidance"
    assert notify.urgency == "priority"


def test_shared_launch_rpo_is_deployment_check_not_escalation():
    rpo = _Rpo(label="likely", min_range_km=0.8, total_close_days=3, same_launch_lineage=True)
    actions = recommend_mitigations(_score(maneuver_activity=1.0), rpo=rpo)
    ids = [a.action_id for a in actions]
    assert "confirm_deployment" in ids
    assert "notify_operator" not in ids  # must not escalate a sibling deployment


def test_poor_attribution_triggers_characterization():
    actions = recommend_mitigations(_score(attribution=1.0))
    assert any(a.action_id == "characterize_unknown" for a in actions)


def test_disposal_noncompliance_triggers_policy_followup():
    disp = _Disposal(status="BELOW_GEO_DRIFT", km_above_geo=-206, compliant=False)
    actions = recommend_mitigations(_score(), disposal=disp)
    assert any(a.action_id == "disposal_followup" and a.category == "policy" for a in actions)


def test_imminent_reentry_is_priority():
    decay = _Decay(days_to_reentry=20, decay_rate_km_per_day=0.7)
    actions = recommend_mitigations(_score(), decay=decay)
    reentry = next(a for a in actions if a.action_id == "reentry_monitoring")
    assert reentry.urgency == "priority"


def test_distant_reentry_is_routine():
    decay = _Decay(days_to_reentry=200, decay_rate_km_per_day=0.05)
    actions = recommend_mitigations(_score(), decay=decay)
    reentry = next(a for a in actions if a.action_id == "reentry_monitoring")
    assert reentry.urgency == "routine"


def test_response_posture_takes_highest_urgency_and_real_frameworks():
    rpo = _Rpo(label="likely", min_range_km=0.04, total_close_days=19)
    actions = recommend_mitigations(_score(maneuver_activity=1.0, proximity_to_hva=0.6), rpo=rpo)
    posture = response_posture(actions)
    assert posture.level == "priority"          # highest urgency present
    assert len(posture.frameworks) >= 1
    # frameworks are (name, description) real public channels, not fabricated actions
    names = " ".join(n for n, _ in posture.frameworks).lower()
    assert "conjunction" in names or "custody" in names


def test_response_posture_routine_when_no_actions():
    assert response_posture([]).level == "routine"


def test_actions_are_sorted_by_urgency():
    rpo = _Rpo(label="likely", min_range_km=0.04, total_close_days=19)
    actions = recommend_mitigations(
        _score(maneuver_activity=1.0, proximity_to_hva=0.6, attribution=1.0, anomalous_rate=0.5),
        rpo=rpo,
    )
    ranks = ["priority", "elevated", "routine"]
    seen = [ranks.index(a.urgency) for a in actions]
    assert seen == sorted(seen)  # non-decreasing urgency rank = priority-first
