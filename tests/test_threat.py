"""Tests for F1.4 composite threat score (score/threat.py).

Offline, synthetic — covers the factor functions and combination logic
that don't touch the DB (maneuver_activity, anomalous_rate, regime,
combine_factors). proximity_to_hva and attribution need SATCAT/current-
catalog lookups and were exercised live against real data instead (ISS,
Cosmos 2542/2543 — see README "Real-data findings"), the same offline/
live split used for F1.1-F1.3.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data.store import TleRecord
from detect.maneuver import detect_maneuvers
from detect.pattern_of_life import pattern_of_life
from score.threat import (
    ThreatFactor,
    _anomalous_rate_factor,
    _maneuver_activity_factor,
    _proximity_factor,
    _regime_factor,
    combine_factors,
)
from tests.test_core import ISS, ISS_L1, ISS_L2, _synthetic_history


def test_proximity_omitted_for_stale_tle():
    # A stale element set must NOT be propagated to "now" to fabricate a
    # distance — the factor is omitted. Regression for a real bug: objects
    # last tracked years ago (Luch +3484d, BeiDou-2 G2 +1612d) were given
    # meaningless proximity values. Old epoch => returns before any DB use.
    old_epoch = (datetime.now(timezone.utc) - timedelta(days=400)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = TleRecord(25544, old_epoch, ISS_L1, ISS_L2, "test")
    f = _proximity_factor(25544, stale)
    assert f.value is None
    assert "days old" in f.explanation


def test_maneuver_activity_unavailable_below_min_history():
    events, floor = detect_maneuvers(_synthetic_history(n_pairs=3))
    f = _maneuver_activity_factor(events, floor)
    assert f.value is None


def test_maneuver_activity_zero_on_quiet_history():
    events, floor = detect_maneuvers(_synthetic_history(burn_at=None))
    f = _maneuver_activity_factor(events, floor)
    assert f.value == 0.0


def test_maneuver_activity_scales_with_peak_burn():
    # Small injected burns, well under MANEUVER_ACTIVITY_SATURATION_MS —
    # the synthetic fixture's injected-vs-measured Δv doesn't calibrate
    # 1:1 (documented in PROGRESS_HANDOFF.md), so this stays comfortably
    # unsaturated rather than assuming a specific measured magnitude.
    burns = {5: 0.002, 10: 0.0022, 15: 0.0019, 20: 0.0021}
    events, floor = detect_maneuvers(_synthetic_history(n_pairs=24, burns=burns))
    f = _maneuver_activity_factor(events, floor)
    assert 0.0 < f.value < 1.0


def test_anomalous_rate_unavailable_below_min_campaigns():
    burns = {5: 0.010, 10: 0.011}  # only 2 campaigns, below MIN_MANEUVERS (3)
    events, _ = detect_maneuvers(_synthetic_history(n_pairs=14, burns=burns))
    classifications, baseline = pattern_of_life(events)
    f = _anomalous_rate_factor(classifications, baseline)
    assert f.value is None


def test_anomalous_rate_reflects_fraction_flagged():
    burns = {5: 0.010, 10: 0.011, 15: 0.080, 20: 0.0105}  # one outsized burn / 4
    events, _ = detect_maneuvers(_synthetic_history(n_pairs=24, burns=burns))
    classifications, baseline = pattern_of_life(events)
    f = _anomalous_rate_factor(classifications, baseline)
    assert f.value == 0.25


def test_regime_factor_unavailable_with_no_tle():
    f = _regime_factor(None)
    assert f.value is None


def test_regime_factor_leo_for_iss():
    f = _regime_factor(ISS)
    assert f.value is not None
    assert "LEO" in f.explanation


def test_combine_factors_no_factors_available_is_none():
    factors = (
        ThreatFactor("a", None, 0.5, "n/a"),
        ThreatFactor("b", None, 0.5, "n/a"),
    )
    score, n = combine_factors(factors)
    assert score is None and n == 0


def test_combine_factors_renormalizes_over_available_only():
    # Two available factors with equal weight but different values must
    # average to the midpoint regardless of the unavailable third factor's
    # weight share -- the whole point of the honesty rule (§9): partial
    # coverage never silently drags the score toward zero.
    factors = (
        ThreatFactor("a", 1.0, 0.5, "high"),
        ThreatFactor("b", 0.0, 0.5, "low"),
        ThreatFactor("c", None, 10.0, "unavailable, huge nominal weight"),
    )
    score, n = combine_factors(factors)
    assert n == 2
    assert score == 50.0
