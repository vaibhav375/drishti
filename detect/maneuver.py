"""Maneuver detection + Δv estimation (F1.1).

Method (per handoff §5): for each consecutive TLE pair of one object,
propagate the EARLIER element set forward to the LATER TLE's epoch and
difference against the later TLE's own state at that epoch. If the
object didn't maneuver, the residual is TLE/SGP4 noise; if it did, the
velocity residual jumps — and its RIC decomposition gives the burn
direction (in-track ⇒ prograde/retrograde, radial, cross-track ⇒ plane
change).

Honesty rules (§9):
* TLE noise is real. Every event carries a z-score against the object's
  own residual noise floor and a coarse confidence label — never a bare
  number pretending to be precise.
* Nothing is fabricated: if fewer than `min_history` TLE pairs exist,
  we return no detections rather than guessing a baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.frames import eci_to_ric
from core.propagation import propagate_at, satrec_from_tle, tle_epoch_datetime
from data.store import TleRecord

MIN_HISTORY_PAIRS = 6      # need this many residuals before flagging
MAX_GAP_DAYS = 10.0        # pairs farther apart than this are unreliable
Z_FLAG = 5.0               # z-score above noise floor to call a maneuver
Z_STRONG = 12.0            # z-score for high confidence


@dataclass
class Residual:
    """One consecutive-TLE-pair comparison."""
    epoch_iso: str                 # epoch of the later TLE (detection time)
    gap_days: float
    dv_ms: float                   # |velocity residual|, m/s
    dv_ric_ms: np.ndarray          # (3,) radial / in-track / cross-track, m/s
    pos_err_km: float              # position residual magnitude


@dataclass
class ManeuverEvent:
    norad_id: int
    epoch_iso: str
    dv_ms: float
    dv_ric_ms: tuple[float, float, float]
    direction: str                 # 'prograde' | 'retrograde' | 'plane-change' | 'radial' | 'mixed'
    zscore: float
    confidence: str                # 'low' | 'medium' | 'high'
    noise_floor_ms: float          # the baseline this was judged against

    def summary(self) -> str:
        r, i, c = self.dv_ric_ms
        return (
            f"NORAD {self.norad_id}: est. Δv {self.dv_ms:.2f} m/s "
            f"({self.direction}; RIC [{r:+.2f}, {i:+.2f}, {c:+.2f}] m/s) "
            f"at {self.epoch_iso} — z={self.zscore:.1f}, "
            f"confidence {self.confidence} (noise floor {self.noise_floor_ms:.2f} m/s)"
        )


def _classify_direction(dv_ric: np.ndarray) -> str:
    r, i, c = np.abs(dv_ric)
    total = r + i + c
    if total == 0:
        return "mixed"
    if i / total > 0.7:
        return "prograde" if dv_ric[1] > 0 else "retrograde"
    if c / total > 0.7:
        return "plane-change"
    if r / total > 0.7:
        return "radial"
    return "mixed"


def pairwise_residuals(tles: list[TleRecord]) -> list[Residual]:
    """Residuals for every valid consecutive TLE pair (chronological input)."""
    out: list[Residual] = []
    for a, b in zip(tles, tles[1:]):
        sat_a = satrec_from_tle(a.line1, a.line2)
        sat_b = satrec_from_tle(b.line1, b.line2)
        t_b = tle_epoch_datetime(sat_b)
        t_a = tle_epoch_datetime(sat_a)
        gap_days = (t_b - t_a).total_seconds() / 86400.0
        if gap_days <= 0 or gap_days > MAX_GAP_DAYS:
            continue
        try:
            r_pred, v_pred = propagate_at(sat_a, t_b)   # old elements, new epoch
            r_act, v_act = propagate_at(sat_b, t_b)     # new elements, own epoch
        except RuntimeError:
            continue
        # Express the residual in the ACTUAL state's RIC frame.
        _, dv_ric_kms = eci_to_ric(r_act, v_act, r_pred, v_pred)
        dv_ric_ms = -dv_ric_kms * 1000.0  # sign: actual minus predicted
        out.append(
            Residual(
                epoch_iso=b.epoch,
                gap_days=gap_days,
                dv_ms=float(np.linalg.norm(dv_ric_ms)),
                dv_ric_ms=dv_ric_ms,
                pos_err_km=float(np.linalg.norm(r_act - r_pred)),
            )
        )
    return out


def detect_maneuvers(
    tles: list[TleRecord],
    z_flag: float = Z_FLAG,
    min_history: int = MIN_HISTORY_PAIRS,
) -> tuple[list[ManeuverEvent], Optional[float]]:
    """Detect maneuvers from an object's chronological TLE history.

    Returns (events, noise_floor_ms). noise_floor_ms is None when there
    is not enough history to establish a baseline (and events is empty —
    we do not flag against an unfounded baseline).

    Baseline = median + MAD of Δv residuals, which is robust: the
    maneuvers themselves are the outliers we must not let contaminate
    the noise estimate.
    """
    residuals = pairwise_residuals(tles)
    if len(residuals) < min_history:
        return [], None

    dvs = np.array([r.dv_ms for r in residuals])
    med = float(np.median(dvs))
    mad = float(np.median(np.abs(dvs - med)))
    sigma = max(1.4826 * mad, 1e-3)  # MAD→σ; floor avoids div-by-zero
    noise_floor = med

    events: list[ManeuverEvent] = []
    for res in residuals:
        z = (res.dv_ms - med) / sigma
        if z < z_flag:
            continue
        confidence = "high" if z >= Z_STRONG else ("medium" if z >= 2 * z_flag else "low")
        events.append(
            ManeuverEvent(
                norad_id=tles[0].norad_id,
                epoch_iso=res.epoch_iso,
                dv_ms=res.dv_ms,
                dv_ric_ms=tuple(float(x) for x in res.dv_ric_ms),
                direction=_classify_direction(res.dv_ric_ms),
                zscore=float(z),
                confidence=confidence,
                noise_floor_ms=noise_floor,
            )
        )
    return events, noise_floor
