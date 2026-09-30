"""Behavioral change-point detection (roadmap feature 3).

Pattern-of-life (F1.2) flags an individual maneuver campaign that deviates
from an object's OWN history. This complements it at the other time scale:
it detects a SHIFT in the whole behavioral regime — a satellite whose
station-keeping tempo suddenly tightens, or whose burn sizes step up. That
"it started behaving differently" signal is exactly what an analyst watches
for, and no single-campaign anomaly test captures it.

Method (standard, interpretable, no heavy deps): over the campaign series
(peak Δv per campaign, and the gap between consecutive campaigns), find the
split point that maximizes the two-sample t-statistic between the segment
before it and the segment after. If that maximum exceeds a threshold, a
change-point is reported at that campaign's epoch, with what changed (burn
size vs tempo), the direction, and the before/after means. Whichever of the
two metrics has the stronger split wins.

Honesty (§9): needs enough campaigns on both sides (MIN_SIDE each) or it
returns None — no change-point is invented from a handful of points. The
reported shift always cites the computed before/after values.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

MIN_SIDE = 3            # campaigns required on each side of a candidate split
T_THRESHOLD = 2.5       # min two-sample t-statistic to call it a real shift


@dataclass(frozen=True)
class ChangePoint:
    norad_id: int
    epoch_iso: str          # when the shift takes effect (first campaign of the new regime)
    metric: str             # 'burn size' | 'tempo'
    unit: str
    before_mean: float
    after_mean: float
    direction: str          # 'increased' | 'decreased'
    t_stat: float

    def summary(self) -> str:
        return (f"NORAD {self.norad_id}: {self.metric} {self.direction} from "
                f"{self.before_mean:.2f}{self.unit} to {self.after_mean:.2f}{self.unit} "
                f"around {self.epoch_iso[:10]} (t={self.t_stat:.1f})")


def _best_split(values: np.ndarray):
    """Return (k, t_stat, before_mean, after_mean) for the split index k
    that maximizes the two-sample t-statistic, or None if too short."""
    n = len(values)
    if n < 2 * MIN_SIDE:
        return None
    best = None
    for k in range(MIN_SIDE, n - MIN_SIDE + 1):
        a, b = values[:k], values[k:]
        ma, mb = float(a.mean()), float(b.mean())
        va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
        pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
        sp = np.sqrt(max(pooled, 1e-12)) * np.sqrt(1.0 / len(a) + 1.0 / len(b))
        t = abs(ma - mb) / sp if sp > 0 else 0.0
        if best is None or t > best[1]:
            best = (k, t, ma, mb)
    return best


def detect_changepoint(campaigns: list) -> Optional[ChangePoint]:
    """`campaigns`: pattern-of-life ManeuverCampaigns (chronological), each
    with .peak_dv_ms and .start_epoch. Returns the stronger of the burn-size
    and tempo change-points, or None if neither is significant."""
    if len(campaigns) < 2 * MIN_SIDE:
        return None
    norad_id = campaigns[0].norad_id
    epochs = [datetime.fromisoformat(c.start_epoch.replace("Z", "+00:00")) for c in campaigns]

    burns = np.array([c.peak_dv_ms for c in campaigns])
    gaps = np.array([(epochs[i] - epochs[i - 1]).total_seconds() / 86400.0
                     for i in range(1, len(epochs))])

    candidates = []
    b_split = _best_split(burns)
    if b_split is not None:
        k, t, ma, mb = b_split
        candidates.append(("burn size", " m/s", k, t, ma, mb, campaigns[k].start_epoch))
    g_split = _best_split(gaps)
    if g_split is not None:
        k, t, ma, mb = g_split
        # gaps[k] is the interval ending at campaign k+1 → epoch of campaign k+1
        candidates.append(("tempo", " days", k + 1, t, ma, mb, campaigns[k + 1].start_epoch))

    if not candidates:
        return None
    metric, unit, k, t, ma, mb, epoch = max(candidates, key=lambda c: c[3])
    if t < T_THRESHOLD:
        return None
    return ChangePoint(
        norad_id=norad_id, epoch_iso=epoch, metric=metric, unit=unit,
        before_mean=ma, after_mean=mb,
        direction="increased" if mb > ma else "decreased", t_stat=t,
    )
