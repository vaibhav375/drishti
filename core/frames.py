"""Frames & relative motion (F0.3). Implemented directly in numpy — we
deliberately do NOT depend on poliastro (archived Oct 2023).

RIC (Radial / In-track / Cross-track), also called RSW or LVLH:
    R = r̂                (radial, away from Earth's center)
    C = (r × v)̂          (cross-track, along orbital angular momentum)
    I = C × R             (in-track, ~velocity direction for near-circular)

RIC is the backbone of RPO detection (F2.1) and how Δv direction is
reported (F1.1): a prograde burn shows up as an in-track velocity jump.

Note on frames: both objects' SGP4 states are in TEME. Because RIC is a
*relative* frame built from the reference object's own state, computing
it directly on consistent TEME vectors is valid — no inertial→inertial
conversion needed for relative geometry. TEME→geodetic conversion for
ground tracks is delegated to skyfield in core/groundtrack.py.
"""
from __future__ import annotations

import numpy as np


def ric_basis(r_ref: np.ndarray, v_ref: np.ndarray) -> np.ndarray:
    """Rotation matrix M (3x3) whose ROWS are the R, I, C unit vectors.

    Transform any inertial vector u into RIC with:  u_ric = M @ u
    """
    r_ref = np.asarray(r_ref, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)
    R = r_ref / np.linalg.norm(r_ref)
    h = np.cross(r_ref, v_ref)
    C = h / np.linalg.norm(h)
    I = np.cross(C, R)
    return np.vstack([R, I, C])


def eci_to_ric(
    r_ref: np.ndarray,
    v_ref: np.ndarray,
    r_other: np.ndarray,
    v_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative position & velocity of `other` w.r.t. `ref`, in RIC.

    The relative velocity includes the transport term (−ω × ρ) so that
    it is the velocity as seen in the rotating RIC frame — this is what
    makes "low relative velocity loitering" detectable for RPO.
    """
    M = ric_basis(r_ref, v_ref)
    rho = np.asarray(r_other, float) - np.asarray(r_ref, float)
    dv = np.asarray(v_other, float) - np.asarray(v_ref, float)

    # Angular velocity of the RIC frame: ω = h / |r|²
    h_vec = np.cross(r_ref, v_ref)
    omega = h_vec / np.dot(r_ref, r_ref)

    rho_ric = M @ rho
    vel_ric = M @ (dv - np.cross(omega, rho))
    return rho_ric, vel_ric


def relative_range_series(
    r_ref: np.ndarray, v_ref: np.ndarray, r_other: np.ndarray, v_other: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorized RIC relative motion over aligned ephemerides.

    Inputs are (N,3) arrays sampled at identical times. Returns dict of
    (N,3) rho_ric / vel_ric and (N,) range / speed — the raw material
    the RPO classifier (F2.1) consumes.
    """
    n = len(r_ref)
    rho = np.empty((n, 3))
    vel = np.empty((n, 3))
    for i in range(n):
        rho[i], vel[i] = eci_to_ric(r_ref[i], v_ref[i], r_other[i], v_other[i])
    return {
        "rho_ric": rho,
        "vel_ric": vel,
        "range_km": np.linalg.norm(rho, axis=1),
        "rel_speed_kms": np.linalg.norm(vel, axis=1),
    }


# ---------------------------------------------------------------------------
# Clohessy–Wiltshire (Hill) equations — linearized relative motion about a
# circular reference orbit. Used for short-horizon RPO prediction and as a
# sanity model; ~50 lines, and we own it (per handoff §3).
# ---------------------------------------------------------------------------

def cw_state_transition(n: float, t: float) -> np.ndarray:
    """CW state-transition matrix Φ(t) for mean motion n (rad/s), time t (s).

    State ordering: [x, y, z, vx, vy, vz] in RIC
    (x radial, y in-track, z cross-track).
    """
    s, c = np.sin(n * t), np.cos(n * t)
    return np.array([
        [4 - 3 * c,        0, 0,        s / n,      2 * (1 - c) / n, 0],
        [6 * (s - n * t),  1, 0, 2 * (c - 1) / n, (4 * s - 3 * n * t) / n, 0],
        [0,                0, c,            0,               0, s / n],
        [3 * n * s,        0, 0,            c,           2 * s, 0],
        [6 * n * (c - 1),  0, 0,       -2 * s,       4 * c - 3, 0],
        [0,                0, -n * s,       0,               0, c],
    ])


def cw_propagate(state0: np.ndarray, n: float, t: float) -> np.ndarray:
    """Propagate a RIC relative state forward t seconds under CW dynamics."""
    return cw_state_transition(n, t) @ np.asarray(state0, float)
