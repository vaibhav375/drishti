"""Tests for the API serializers (web/serialize.py). Offline, pure.

Also asserts JSON-safety (json.dumps round-trips) and that the CDM export
carries its honesty caveat.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from web.serialize import conjunction_cdm, object_json


def test_object_json_minimal_no_tle():
    d = {"norad_id": 5, "name": "VANGUARD 1", "satcat": {}, "has_tle": False,
         "decayed": False, "decay_date": None, "elements": None, "regime": None,
         "threat": None, "maneuvers": [], "rpo": None, "decay": None, "disposal": None,
         "changepoint": None, "mitigations": []}
    j = object_json(d)
    assert j["norad_id"] == 5 and j["name"] == "VANGUARD 1"
    assert j["threat"] is None and j["regime"] is None
    json.dumps(j)  # must be JSON-safe


def test_object_json_with_regime_and_threat():
    @dataclass
    class _Regime:
        primary = "GEO"; special_tags = ("EQUATORIAL",); label = "GEO [EQUATORIAL]"; period_min = 1436.0

    @dataclass
    class _F:
        name: str
        value: float
        weight: float
        explanation: str

    @dataclass
    class _T:
        score = 66.0
        n_factors_available = 5
        n_factors_total = 5
        factors = (_F("maneuver_activity", 1.0, 0.2, "x"),)

    d = {"norad_id": 49330, "name": "SJ-21", "satcat": {}, "has_tle": True,
         "decayed": False, "decay_date": None, "elements": None, "regime": _Regime(),
         "threat": _T(), "maneuvers": [1, 2, 3], "rpo": None, "decay": None,
         "disposal": None, "changepoint": None, "mitigations": []}
    j = object_json(d)
    assert j["regime"]["primary"] == "GEO"
    assert j["threat"]["score"] == 66.0
    assert j["n_maneuvers"] == 3
    assert j["threat"]["factors"][0]["name"] == "maneuver_activity"
    json.dumps(j)


def test_conjunction_cdm_has_fields_and_caveat():
    @dataclass
    class _A:
        tca_iso = "2021-12-31T15:44:31Z"; min_range_km = 0.036; pc_estimate = 1e-5
        hard_body_radius_km = 0.006; actor_norad = 49330; target_norad = 34779

    cdm = conjunction_cdm(_A(), "SJ-21", "BEIDOU-2 G2")
    assert "CCSDS_CDM_VERS" in cdm
    assert "COLLISION_PROBABILITY = 1.000e-05" in cdm
    assert "MISS_DISTANCE = 36.0 [m]" in cdm
    assert "49330" in cdm and "34779" in cdm
    # honesty caveat present — not an operational CDM
    assert "MODELED covariance" in cdm and "NOT an operational CDM" in cdm
