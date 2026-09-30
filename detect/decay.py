"""Re-entry / orbital-decay prediction (bonus §10).

Tracks an object's perigee altitude across its stored TLE history and, if
it's decaying, extrapolates a re-entry window. Purely data-driven — the
decay signal is read from the object's OWN element history (atmospheric
drag steadily lowers the semi-major axis of a low-perigee object), not
from a drag model with assumed ballistic coefficients we don't have.

The model, and why it's not a naive line fit: atmospheric density rises
roughly exponentially as altitude drops (ρ ≈ ρ₀·exp(−Δh/H)), so decay
ACCELERATES near the end — a linear perigee extrapolation massively
overestimates remaining lifetime (validated live 2026-07-15: a real
Cosmos 1408 fragment that actually re-entered ~30 days out was predicted
~555 days out by a pure line fit). So when the history shows the decay
accelerating, we derive the object's OWN effective atmospheric scale
height H directly from its two-halves decay rates (H = Δh / ln(r_new/
r_old)) and integrate the exponential-atmosphere decay to re-entry —
data-driven, no assumed ballistic coefficient. Linear "current-rate" is
kept as a fallback (and reported as `linear_days_to_reentry`) when
there's no measurable acceleration to fit H from.

Still a SCREENING estimate, never an operational re-entry prediction
(those use high-fidelity drag models + space-weather forecasts), and
every result says so.

Refuse-over-fake (§9): non-decaying (or boosted, like the ISS) objects
return `None` rather than a fabricated date; too little history returns
`None`; a perigee already below the re-entry threshold reports imminent,
not a negative countdown.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from core.elements import mean_elements
from data.store import TleRecord

REENTRY_ALTITUDE_KM = 100.0     # Kármán-line-ish; perigee at/below here ⇒ re-entry
MAX_PERIGEE_FOR_DECAY_KM = 2000.0  # above this, atmospheric drag is negligible — a
                                     # perigee "trend" is TLE noise or other perturbations,
                                     # not decay. Found live 2026-07-15: a GEO satellite
                                     # (SJ-21, perigee 35,567 km) with 0.02 km/day of noise
                                     # was given a re-entry date ~1.7 million days out.
MIN_HISTORY_POINTS = 6          # need this many TLEs before fitting a trend
MIN_DECAY_RATE_KM_PER_DAY = 0.02  # slower perigee loss than this ⇒ treat as stable
ACCEL_SIGNIFICANCE = 1.15       # recent-half decay rate this× the older-half ⇒ "accelerating"
DEFAULT_SCALE_HEIGHT_KM = 50.0  # fallback H when acceleration can't be measured
MIN_SCALE_HEIGHT_KM = 20.0      # clamp the data-derived H to physically sane bounds
MAX_SCALE_HEIGHT_KM = 90.0


@dataclass(frozen=True)
class DecayPrediction:
    norad_id: int
    latest_epoch_iso: str
    latest_perigee_km: float
    decay_rate_km_per_day: float     # positive = losing altitude (recent-half rate)
    predicted_reentry_iso: Optional[str]
    days_to_reentry: Optional[float]
    linear_days_to_reentry: Optional[float]  # naive line-fit estimate, for comparison
    model: str                       # 'exponential-atmosphere' | 'linear'
    scale_height_km: Optional[float]  # data-derived H, when exponential model used
    accelerating: bool
    fit_quality_r2: float
    note: str

    def summary(self) -> str:
        if self.predicted_reentry_iso is None:
            return f"NORAD {self.norad_id}: {self.note}"
        model_bit = (f"exponential-atmosphere model (data-derived scale height "
                     f"{self.scale_height_km:.0f} km)" if self.model == "exponential-atmosphere"
                     else "linear current-rate model")
        return (
            f"NORAD {self.norad_id}: perigee {self.latest_perigee_km:.0f} km, losing "
            f"{self.decay_rate_km_per_day:.2f} km/day → predicted re-entry "
            f"~{self.predicted_reentry_iso[:10]} ({self.days_to_reentry:.0f} days) "
            f"via {model_bit}. {self.note}"
        )


def _epoch_dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _half_rates(days: np.ndarray, perigee: np.ndarray):
    """Decay rate (km/day, positive = losing) and mean altitude over the
    older half and the newer half of the history — the two (rate,
    altitude) points the exponential-atmosphere model fits H from."""
    mid = len(days) // 2
    old_rate = -float(np.polyfit(days[:mid], perigee[:mid], 1)[0])
    new_rate = -float(np.polyfit(days[mid:], perigee[mid:], 1)[0])
    old_alt = float(perigee[:mid].mean())
    new_alt = float(perigee[mid:].mean())
    return old_rate, new_rate, old_alt, new_alt


def predict_decay(tles: list[TleRecord]) -> Optional[DecayPrediction]:
    """Build the perigee-vs-time series from an object's TLE history and
    predict re-entry. The math lives in `predict_decay_from_series` so it's
    testable with synthetic series, independent of TLE parsing."""
    if len(tles) < MIN_HISTORY_POINTS:
        return None
    pts = []
    for rec in tles:
        try:
            el = mean_elements(rec)
        except Exception:
            continue
        pts.append((_epoch_dt(rec.epoch), el.perigee_km))
    if len(pts) < MIN_HISTORY_POINTS:
        return None
    pts.sort(key=lambda p: p[0])
    return predict_decay_from_series(
        tles[0].norad_id, [t for t, _ in pts], [p for _, p in pts]
    )


def predict_decay_from_series(
    norad_id: int, epochs: list[datetime], perigee_km: list[float]
) -> Optional[DecayPrediction]:
    if len(epochs) < MIN_HISTORY_POINTS:
        return None
    order = np.argsort([t.timestamp() for t in epochs])
    epochs = [epochs[i] for i in order]
    perigee = np.array([perigee_km[i] for i in order], dtype=float)

    t0 = epochs[0]
    days = np.array([(t - t0).total_seconds() / 86400.0 for t in epochs])
    pts = list(zip(epochs, perigee))

    slope, intercept = np.polyfit(days, perigee, 1)
    overall_rate = -float(slope)
    pred = slope * days + intercept
    ss_res = float(np.sum((perigee - pred) ** 2))
    ss_tot = float(np.sum((perigee - perigee.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    latest_t, latest_per = pts[-1]
    latest_iso = latest_t.strftime("%Y-%m-%dT%H:%M:%SZ")

    if latest_per > MAX_PERIGEE_FOR_DECAY_KM:
        return DecayPrediction(
            norad_id=norad_id, latest_epoch_iso=latest_iso, latest_perigee_km=latest_per,
            decay_rate_km_per_day=overall_rate, predicted_reentry_iso=None, days_to_reentry=None,
            linear_days_to_reentry=None, model="linear", scale_height_km=None,
            accelerating=False, fit_quality_r2=r2,
            note=f"perigee {latest_per:.0f} km is above the drag-dominated regime — "
                 "atmospheric decay does not apply, no re-entry prediction",
        )

    if overall_rate < MIN_DECAY_RATE_KM_PER_DAY:
        return DecayPrediction(
            norad_id=norad_id, latest_epoch_iso=latest_iso, latest_perigee_km=latest_per,
            decay_rate_km_per_day=overall_rate, predicted_reentry_iso=None, days_to_reentry=None,
            linear_days_to_reentry=None, model="linear", scale_height_km=None,
            accelerating=False, fit_quality_r2=r2,
            note="not measurably decaying (stable or boosted) — no re-entry prediction",
        )

    if latest_per <= REENTRY_ALTITUDE_KM:
        return DecayPrediction(
            norad_id=norad_id, latest_epoch_iso=latest_iso, latest_perigee_km=latest_per,
            decay_rate_km_per_day=overall_rate, predicted_reentry_iso=latest_iso,
            days_to_reentry=0.0, linear_days_to_reentry=0.0, model="linear", scale_height_km=None,
            accelerating=True, fit_quality_r2=r2,
            note="perigee already at/below re-entry altitude — re-entry imminent",
        )

    # Recent-half rate is the relevant current behavior; measure acceleration.
    recent_rate = overall_rate
    accelerating = False
    scale_height: Optional[float] = None
    if len(pts) >= 2 * MIN_HISTORY_POINTS:
        old_rate, new_rate, old_alt, new_alt = _half_rates(days, perigee)
        recent_rate = max(new_rate, MIN_DECAY_RATE_KM_PER_DAY)
        accelerating = new_rate > ACCEL_SIGNIFICANCE * max(old_rate, 1e-6)
        # Derive the object's OWN effective scale height from how much the
        # decay rate grew as it dropped from old_alt to new_alt:
        #   r(h) ∝ exp(−h/H)  ⇒  H = (old_alt − new_alt) / ln(new_rate/old_rate)
        if accelerating and new_alt < old_alt and new_rate > old_rate > 0:
            H = (old_alt - new_alt) / np.log(new_rate / old_rate)
            scale_height = float(np.clip(H, MIN_SCALE_HEIGHT_KM, MAX_SCALE_HEIGHT_KM))

    drop_km = latest_per - REENTRY_ALTITUDE_KM
    linear_days = drop_km / recent_rate

    if scale_height is not None:
        # Integrate dt = dh / r(h), r(h) = recent_rate·exp((latest_per − h)/H):
        #   t = (H/recent_rate)·(1 − exp(−drop/H))
        H = scale_height
        days_to_reentry = (H / recent_rate) * (1.0 - np.exp(-drop_km / H))
        model = "exponential-atmosphere"
        note = ("data-derived exponential-atmosphere estimate (decay accelerates as it "
                "descends); still a screening figure, not an operational re-entry prediction")
    else:
        days_to_reentry = linear_days
        model = "linear"
        note = ("LINEAR current-rate estimate — no acceleration measurable yet, so this is "
                "an UPPER BOUND on lifetime (real re-entry likely sooner once decay ramps)")

    reentry_dt = latest_t + timedelta(days=float(days_to_reentry))
    return DecayPrediction(
        norad_id=norad_id, latest_epoch_iso=latest_iso, latest_perigee_km=latest_per,
        decay_rate_km_per_day=recent_rate,
        predicted_reentry_iso=reentry_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        days_to_reentry=float(days_to_reentry), linear_days_to_reentry=float(linear_days),
        model=model, scale_height_km=scale_height, accelerating=accelerating, fit_quality_r2=r2,
        note=note,
    )
