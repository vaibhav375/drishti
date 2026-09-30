"""Tests for behavioral change-point detection (detect/changepoint.py).

Offline, synthetic campaign series. Validated live against real maneuver
histories too (see README findings).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from detect.changepoint import detect_changepoint


@dataclass
class _Campaign:
    norad_id: int
    start_epoch: str
    peak_dv_ms: float


def _series(dvs, gaps_days):
    """Build campaigns from a list of burn sizes and inter-campaign gaps."""
    t = datetime(2023, 1, 1)
    camps = []
    for i, dv in enumerate(dvs):
        if i > 0:
            t = t + timedelta(days=gaps_days[i - 1])
        camps.append(_Campaign(1, t.strftime("%Y-%m-%dT%H:%M:%SZ"), dv))
    return camps


def test_too_few_campaigns_returns_none():
    camps = _series([1.0, 1.1, 2.0], [10, 10])
    assert detect_changepoint(camps) is None


def test_no_shift_returns_none():
    # steady burns, steady tempo
    camps = _series([1.0, 1.05, 0.98, 1.02, 1.0, 1.03, 0.99, 1.01],
                    [20] * 7)
    assert detect_changepoint(camps) is None


def test_burn_size_step_up_detected():
    # first 4 campaigns ~1 m/s, next 4 ~10 m/s at steady tempo
    camps = _series([1.0, 1.1, 0.9, 1.0, 10.0, 10.2, 9.8, 10.1], [20] * 7)
    cp = detect_changepoint(camps)
    assert cp is not None
    assert cp.metric == "burn size"
    assert cp.direction == "increased"
    assert cp.before_mean < 2 and cp.after_mean > 8


def test_tempo_tightening_detected():
    # steady burns, but cadence tightens from ~30 days to ~5 days
    camps = _series([1.0] * 8, [30, 30, 30, 5, 5, 5, 5])
    cp = detect_changepoint(camps)
    assert cp is not None
    assert cp.metric == "tempo"
    assert cp.direction == "decreased"


def test_stronger_metric_wins():
    # a small burn drift but a huge tempo change -> tempo should win
    camps = _series([1.0, 1.1, 1.0, 1.05, 1.0, 1.1, 1.0, 1.05],
                    [40, 40, 40, 2, 2, 2, 2])
    cp = detect_changepoint(camps)
    assert cp is not None
    assert cp.metric == "tempo"
