"""Tests for unsupervised anomaly detection (detect/anomaly.py).

Offline, synthetic population. A tight cluster of normal orbits plus a
deliberate outlier — the model must score the outlier higher and name the
feature that makes it unusual.
"""
from __future__ import annotations

import pytest

from detect.anomaly import AnomalyModel


def _row(nid, apo, per, inc, period=100.0):
    return {"norad_id": nid, "apogee_km": apo, "perigee_km": per,
            "inclination": inc, "period_min": period}


def _population():
    # 60 near-identical LEO orbits (the "normal" cluster)
    rows = [_row(i, 800 + (i % 5), 780 + (i % 5), 98.0 + (i % 3) * 0.1) for i in range(60)]
    return rows


def test_fit_requires_minimum_population():
    m = AnomalyModel()
    with pytest.raises(ValueError):
        m.fit([_row(i, 800, 780, 98) for i in range(5)])


def test_outlier_scores_higher_than_the_cluster():
    rows = _population()
    outlier = _row(999, 35786, 35786, 5.0, period=1436.0)  # a GEO object among LEO SSO
    m = AnomalyModel().fit(rows + [outlier])

    out = m.score_row(outlier)
    typical = m.score_row(rows[0])
    assert out is not None and typical is not None
    assert out.score > typical.score
    assert out.is_outlier


def test_top_feature_is_named():
    rows = _population()
    # an orbit unusual specifically in inclination (equatorial among polar SSO)
    outlier = _row(998, 800, 780, 2.0)
    m = AnomalyModel().fit(rows + [outlier])
    s = m.score_row(outlier)
    assert s is not None
    assert s.top_feature == "inclination"
    assert abs(s.top_feature_z) > 2  # a real multi-sigma deviation


def test_top_anomalies_ranks_the_planted_outlier_first():
    rows = _population()
    outlier = _row(997, 42000, 400, 63.0, period=720.0)  # wild HEO among tight LEO
    m = AnomalyModel().fit(rows + [outlier])
    top = m.top_anomalies(rows + [outlier], n=3)
    assert top[0].norad_id == 997


def test_missing_orbital_data_scores_none():
    rows = _population()
    m = AnomalyModel().fit(rows)
    assert m.score_row({"norad_id": 1, "apogee_km": None, "perigee_km": None,
                        "inclination": None, "period_min": None}) is None
