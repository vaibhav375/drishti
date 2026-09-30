"""Probabilistic conjunction assessment / Pc (F3.1).

Per handoff §5: assign MODELED covariance to TLEs (public TLEs carry
none — real Pc uses non-public CDM covariance from the owning agency;
this is an approximation and says so), grow it with propagation time from
each TLE's own epoch, compute probability-of-collision via Monte Carlo,
on cascade-screened pairs only (detect/cascade.py — never run this on an
unscreened all-pairs set).

No TEME->GCRS conversion needed: like RIC elsewhere in this codebase,
both objects are propagated to the SAME instant (time of closest
approach) and compared in the same TEME frame — valid for this relative
comparison without an absolute-frame conversion.

Method:
1. `find_time_of_closest_approach` — coarse grid (propagate_aligned) then
   successively finer local regrids down to ~1s resolution. F2.1's 6h RPO
   grid is too coarse for this: at ~7.5 km/s relative velocity, 6h covers
   ~160,000 km, easily skipping a genuine sub-km encounter (the real
   Iridium 33/Cosmos 2251 collision would be invisible to it).
2. At TCA, build each object's own RIC frame (core/frames.py::ric_basis)
   and sample Gaussian position noise in it — radial/in-track/cross-track,
   scaled by `model_covariance_km` (grows with age since that object's
   OWN nearest TLE epoch; in-track grows fastest, since drag mismodeling
   is the dominant real TLE error source).
3. Pc = fraction of Monte Carlo sample pairs whose separation is under
   the combined hard-body radius, derived from SATCAT's real RCS_m2
   field via `hard_body_radius_km` (a radar cross-section, not a measured
   physical size — another disclosed approximation) when published
   (~47% of objects, confirmed live 2026-07-15), else a documented
   default.

Honesty rule (§9): every result carries the disclaimer that this uses a
MODELED covariance and RCS-derived size, not real conjunction-grade
inputs. No overlapping TLE coverage -> None, never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from core.frames import ric_basis
from core.propagation import propagate_aligned, propagate_at, satrec_from_tle, tle_epoch_datetime
from data.store import TleRecord

# Covariance growth model (RIC, km) — MODELED, not measured. In-track
# grows fastest: drag mismodeling accumulates along-track position error
# over time far faster than radial/cross-track for a typical LEO/MEO/GEO
# object. Values are a defensible starting point (km-scale after weeks),
# not a fit to any real covariance data (none is public).
SIGMA_R0_KM, SIGMA_R_GROWTH_KM_PER_DAY = 0.10, 0.02
SIGMA_I0_KM, SIGMA_I_GROWTH_KM_PER_DAY = 0.20, 0.15
SIGMA_C0_KM, SIGMA_C_GROWTH_KM_PER_DAY = 0.10, 0.01

DEFAULT_HARD_BODY_RADIUS_KM = 0.005  # ~5 m, used when RCS is unpublished
MIN_HARD_BODY_RADIUS_KM = 0.001

DEFAULT_N_SAMPLES = 200_000


@dataclass(frozen=True)
class ConjunctionAssessment:
    actor_norad: int
    target_norad: int
    tca_iso: str
    min_range_km: float           # deterministic range at TCA, no noise
    pc_estimate: float
    hard_body_radius_km: float
    sigma_actor_ric_km: tuple[float, float, float]
    sigma_target_ric_km: tuple[float, float, float]
    n_samples: int

    def summary(self) -> str:
        return (
            f"NORAD {self.actor_norad} / {self.target_norad}: Pc ~ {self.pc_estimate:.2e} "
            f"at TCA {self.tca_iso} (deterministic min range {self.min_range_km:.3f} km, "
            f"combined hard-body radius {self.hard_body_radius_km * 1000:.1f} m, "
            f"{self.n_samples} Monte Carlo samples)\n"
            "  DISCLAIMER: covariance is MODELED (grows with TLE age), not measured — "
            "real Pc requires non-public CDM covariance. Hard-body radius is derived "
            "from radar cross-section (SATCAT RCS), not a measured physical size. "
            "Treat as an order-of-magnitude screening estimate, not an operational Pc."
        )


def model_covariance_km(age_days: float) -> tuple[float, float, float]:
    """(sigma_r, sigma_i, sigma_c) in km, growing with time since the
    object's own nearest TLE epoch. See module docstring for rationale."""
    age_days = max(age_days, 0.0)
    return (
        SIGMA_R0_KM + SIGMA_R_GROWTH_KM_PER_DAY * age_days,
        SIGMA_I0_KM + SIGMA_I_GROWTH_KM_PER_DAY * age_days,
        SIGMA_C0_KM + SIGMA_C_GROWTH_KM_PER_DAY * age_days,
    )


def hard_body_radius_km(rcs_m2: Optional[float]) -> float:
    """Radius from radar cross-section (treated as an equivalent circular
    area: r = sqrt(RCS/pi)) — a coarse proxy for physical size, not a
    measurement. Falls back to DEFAULT_HARD_BODY_RADIUS_KM when RCS is
    unpublished (SATCAT rcs_m2 is NULL for ~53% of objects, confirmed
    live 2026-07-15 — including SJ-21 itself)."""
    if rcs_m2 is None or rcs_m2 <= 0:
        return DEFAULT_HARD_BODY_RADIUS_KM
    radius_km = float(np.sqrt(rcs_m2 / np.pi)) / 1000.0
    return max(radius_km, MIN_HARD_BODY_RADIUS_KM)


def _nearest_tle(records: list[TleRecord], t: datetime) -> TleRecord:
    sats = [(tle_epoch_datetime(satrec_from_tle(r.line1, r.line2)), r) for r in records]
    return min(sats, key=lambda s: abs(s[0] - t))[1]


def find_time_of_closest_approach(
    actor_tles: list[TleRecord],
    target_tles: list[TleRecord],
    coarse_step_hours: float = 1.0,
) -> Optional[tuple[datetime, float]]:
    """Coarse grid (propagate_aligned) then successively finer local
    regrids (10 min -> 30 s -> 1 s) around the coarse minimum. Returns
    (tca, min_range_km), or None with insufficient overlapping coverage.
    """
    times, r_a, _, r_b, _ = propagate_aligned(actor_tles, target_tles, step_hours=coarse_step_hours)
    if len(times) < 2:
        return None

    range_km = np.linalg.norm(r_a - r_b, axis=1)
    idx = int(np.argmin(range_km))
    best_t = times[idx]
    best_range = float(range_km[idx])

    rec_a = _nearest_tle(actor_tles, best_t)
    rec_b = _nearest_tle(target_tles, best_t)
    sat_a = satrec_from_tle(rec_a.line1, rec_a.line2)
    sat_b = satrec_from_tle(rec_b.line1, rec_b.line2)

    span_s = coarse_step_hours * 3600.0
    for resolution_s in (600.0, 30.0, 1.0):
        candidates = [best_t + timedelta(seconds=k * resolution_s)
                      for k in range(-40, 41)]
        for t in candidates:
            if abs((t - best_t).total_seconds()) > span_s:
                continue
            try:
                r1, _ = propagate_at(sat_a, t)
                r2, _ = propagate_at(sat_b, t)
            except RuntimeError:
                continue
            d = float(np.linalg.norm(r1 - r2))
            if d < best_range:
                best_range = d
                best_t = t
        span_s = resolution_s * 40.0

    return best_t, best_range


def estimate_pc(
    actor_id: int,
    target_id: int,
    actor_tles: list[TleRecord],
    target_tles: list[TleRecord],
    actor_rcs_m2: Optional[float] = None,
    target_rcs_m2: Optional[float] = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    coarse_step_hours: float = 1.0,
    rng_seed: int = 42,
) -> Optional[ConjunctionAssessment]:
    tca_result = find_time_of_closest_approach(actor_tles, target_tles, coarse_step_hours)
    if tca_result is None:
        return None
    tca, min_range_km = tca_result

    rec_a = _nearest_tle(actor_tles, tca)
    rec_b = _nearest_tle(target_tles, tca)
    sat_a = satrec_from_tle(rec_a.line1, rec_a.line2)
    sat_b = satrec_from_tle(rec_b.line1, rec_b.line2)
    r_a, v_a = propagate_at(sat_a, tca)
    r_b, v_b = propagate_at(sat_b, tca)

    age_a_days = abs((tca - tle_epoch_datetime(sat_a)).total_seconds()) / 86400.0
    age_b_days = abs((tca - tle_epoch_datetime(sat_b)).total_seconds()) / 86400.0
    sigma_a = model_covariance_km(age_a_days)
    sigma_b = model_covariance_km(age_b_days)

    m_a = ric_basis(r_a, v_a)  # rows = R, I, C unit vectors
    m_b = ric_basis(r_b, v_b)

    rng = np.random.default_rng(rng_seed)
    noise_a_ric = rng.normal(size=(n_samples, 3)) * np.array(sigma_a)
    noise_b_ric = rng.normal(size=(n_samples, 3)) * np.array(sigma_b)
    # RIC -> TEME for a batch of row-vectors: teme = ric @ M (M's rows are
    # the RIC basis vectors expressed in TEME, so ric_vec @ M == M.T @ ric_vec
    # for a single column vector).
    samples_a = r_a + noise_a_ric @ m_a
    samples_b = r_b + noise_b_ric @ m_b

    dist_km = np.linalg.norm(samples_a - samples_b, axis=1)
    hbr = hard_body_radius_km(actor_rcs_m2) + hard_body_radius_km(target_rcs_m2)
    pc = float(np.mean(dist_km < hbr))

    return ConjunctionAssessment(
        actor_norad=actor_id,
        target_norad=target_id,
        tca_iso=tca.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        min_range_km=min_range_km,
        pc_estimate=pc,
        hard_body_radius_km=hbr,
        sigma_actor_ric_km=sigma_a,
        sigma_target_ric_km=sigma_b,
        n_samples=n_samples,
    )
