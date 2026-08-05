"""Tests for re-entry / decay prediction (detect/decay.py).

Offline, synthetic perigee-vs-time series fed to `predict_decay_from_series`
(the pure core, split out so it's testable without TLE parsing). The
exponential-atmosphere model was validated against REAL decayed Cosmos
1408 fragments — see README "Real-data findings": known re-entry dates
predicted within a few days, vs. hundreds of days off for a naive line
fit.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from detect.decay import REENTRY_ALTITUDE_KM, predict_decay_from_series


def _series(perigees, step_days=5.0):
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    epochs = [t0 + timedelta(days=i * step_days) for i in range(len(perigees))]
    return epochs, list(perigees)


def test_too_little_history_returns_none():
    epochs, per = _series([400, 399, 398])
    assert predict_decay_from_series(1, epochs, per) is None


def test_high_perigee_object_gets_no_reentry_prediction():
    # A GEO-altitude object with a tiny perigee "trend" (TLE noise) must
    # NOT get a re-entry date — drag doesn't act up there. Regression for
    # a real bug: SJ-21 (perigee ~35,500 km) was predicted to re-enter
    # ~1.7 million days out.
    epochs, per = _series([35567 - 0.1 * i for i in range(12)])  # trivial downward noise
    p = predict_decay_from_series(1, epochs, per)
    assert p is not None
    assert p.predicted_reentry_iso is None
    assert "drag-dominated" in p.note


def test_stable_orbit_gives_no_reentry_prediction():
    # Flat perigee (station-kept / high enough to be stable).
    epochs, per = _series([800] * 10)
    p = predict_decay_from_series(1, epochs, per)
    assert p is not None
    assert p.predicted_reentry_iso is None
    assert "not measurably decaying" in p.note


def test_already_below_reentry_altitude_is_imminent():
    epochs, per = _series([120, 110, 100, 95, 92, 90])
    p = predict_decay_from_series(1, epochs, per)
    assert p is not None
    assert p.days_to_reentry == 0.0
    assert "imminent" in p.note


def test_constant_rate_decay_uses_linear_model():
    # Perfectly constant decay rate -> no acceleration -> linear fallback.
    epochs, per = _series([500 - 2 * i for i in range(12)])  # 2 km per step
    p = predict_decay_from_series(1, epochs, per)
    assert p is not None
    assert p.model == "linear"
    assert p.predicted_reentry_iso is not None
    assert p.days_to_reentry > 0


def test_accelerating_decay_uses_exponential_model_and_beats_linear():
    # Quadratically-accelerating perigee loss (like a real terminal decay):
    # the exponential model should predict re-entry SOONER than the naive
    # linear current-rate figure.
    perigees = [400 - 0.5 * i - 0.08 * i * i for i in range(16)]
    epochs, per = _series(perigees)
    p = predict_decay_from_series(1, epochs, per)
    assert p is not None
    assert p.model == "exponential-atmosphere"
    assert p.accelerating
    assert p.scale_height_km is not None
    assert p.days_to_reentry < p.linear_days_to_reentry


def test_prediction_target_is_the_reentry_altitude():
    # A clean constant-rate decay to exactly the threshold: re-entry date
    # should land where perigee hits REENTRY_ALTITUDE_KM.
    start = 300.0
    rate = 2.0  # km per step, 5-day steps -> 0.4 km/day
    epochs, per = _series([start - rate * i for i in range(12)])
    p = predict_decay_from_series(1, epochs, per)
    assert p is not None
    # latest perigee is start - rate*11; remaining drop / rate_per_day = days
    remaining_km = per[-1] - REENTRY_ALTITUDE_KM
    assert p.days_to_reentry > 0
    # linear model: days ~= remaining_km / (rate/step_days)
    expected_days = remaining_km / (rate / 5.0)
    assert abs(p.linear_days_to_reentry - expected_days) < 1.0
