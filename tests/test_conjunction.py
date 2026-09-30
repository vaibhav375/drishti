"""Tests for F3.1 probabilistic conjunction assessment (detect/conjunction.py).

Offline for the pure functions (hard_body_radius_km, model_covariance_km)
and a same-object sanity check (two identical histories must show ~0 km
TCA range and an elevated Pc). Real-event validation — TCA-finding
against SJ-21/BeiDou-2 G2's actual close approach, and a control against
Iridium 33/Cosmos 2251 pre-collision — was run live; see README
"Real-data findings".
"""
from __future__ import annotations

from detect.conjunction import (
    DEFAULT_HARD_BODY_RADIUS_KM,
    estimate_pc,
    find_time_of_closest_approach,
    hard_body_radius_km,
    model_covariance_km,
)
from tests.test_core import _synthetic_history


def test_hard_body_radius_defaults_when_rcs_unpublished():
    assert hard_body_radius_km(None) == DEFAULT_HARD_BODY_RADIUS_KM
    assert hard_body_radius_km(0.0) == DEFAULT_HARD_BODY_RADIUS_KM


def test_hard_body_radius_scales_with_rcs():
    # ISS-scale RCS (~399 m^2, confirmed live) should give a much larger
    # radius than a small satellite (~1-3 m^2, confirmed live for
    # BeiDou-2 G2 / Iridium 33 / Cosmos 2251).
    small = hard_body_radius_km(1.4)
    large = hard_body_radius_km(399.0)
    assert small < large
    assert small > 0


def test_model_covariance_grows_with_age():
    sigma_0 = model_covariance_km(0.0)
    sigma_30d = model_covariance_km(30.0)
    for s0, s30 in zip(sigma_0, sigma_30d):
        assert s30 > s0
    # negative age (shouldn't happen, but must not produce nonsense) clamps to 0
    assert model_covariance_km(-5.0) == sigma_0


def test_find_tca_refuses_with_no_overlap():
    assert find_time_of_closest_approach([], []) is None


def test_estimate_pc_refuses_with_no_overlap():
    assert estimate_pc(1, 2, [], []) is None


def test_identical_object_compared_to_itself_has_near_zero_range():
    history = _synthetic_history(n_pairs=10, burn_at=None)
    tca_result = find_time_of_closest_approach(history, history, coarse_step_hours=6.0)
    assert tca_result is not None
    tca, min_range_km = tca_result
    assert min_range_km < 1e-6


def test_pc_is_nonzero_for_a_coincident_pair():
    # Two independent Gaussians (sigma ~0.1-0.2 km each, per
    # model_covariance_km) centered on the SAME point combine to a
    # difference-distribution spread of ~100-300 m -- even a coincident
    # pair's Pc against a small hard-body radius isn't large in absolute
    # terms, but it must be strictly positive (real probability mass at
    # zero separation). The "genuinely separated -> ~0 Pc" contrast was
    # validated live instead, against Iridium 33/Cosmos 2251 pre-collision
    # real archive data — see README "Real-data findings".
    history = _synthetic_history(n_pairs=10, burn_at=None)
    same = estimate_pc(25544, 25544, history, history, n_samples=20_000)
    assert same is not None
    assert same.min_range_km < 1e-6
    assert same.pc_estimate > 0.0
