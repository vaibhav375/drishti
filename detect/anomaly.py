"""Unsupervised statistical anomaly detection (bonus / AI layer).

An IsolationForest (scikit-learn — already a project dependency) trained on
the orbital features of the active catalogue, scoring how much of an
OUTLIER each object's orbit is relative to the whole population. This is
the "AI-based" complement to the rule-based threat score (F1.4): the
threat score encodes what analysts already know to look for; the
anomaly model finds objects that simply don't fit the statistical
distribution, including patterns no rule anticipated.

Honesty discipline (§9): an anomaly score is NOT a threat verdict. It
says "this orbit is statistically unusual versus its peers", nothing more
— a brand-new experimental orbit and a genuinely evasive object both look
unusual. The score is surfaced as exactly that, and it explains WHICH
feature drives it (the largest standardized deviation), so it is not a
black box.

Features (all from SATCAT, available for most objects, no propagation):
log10 apogee, log10 perigee, inclination, eccentricity proxy, period.
Standardized before fitting so no single unit dominates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import R_EARTH_KM

FEATURE_NAMES = ("apogee (log)", "perigee (log)", "inclination", "eccentricity", "period")


@dataclass(frozen=True)
class AnomalyScore:
    norad_id: int
    score: float                 # 0..1, higher = more anomalous
    is_outlier: bool             # IsolationForest hard label
    top_feature: str             # feature with the largest standardized deviation
    top_feature_z: float         # that deviation, in sigma


def _features(row: dict) -> Optional[list[float]]:
    apo, per, inc, period = (row.get("apogee_km"), row.get("perigee_km"),
                             row.get("inclination"), row.get("period_min"))
    if None in (apo, per, inc) or apo <= 0 or per <= 0:
        return None
    ra, rp = apo + R_EARTH_KM, per + R_EARTH_KM
    ecc = (ra - rp) / (ra + rp)
    per_min = period if period else 0.0
    return [math.log10(apo + 1), math.log10(per + 1), float(inc), float(ecc), float(per_min)]


class AnomalyModel:
    """Fit-once model over a population; then score any member or newcomer."""

    def __init__(self):
        self._ids: list[int] = []
        self._X = None
        self._mean = None
        self._std = None
        self._forest = None

    def fit(self, rows: list[dict]) -> "AnomalyModel":
        from sklearn.ensemble import IsolationForest

        feats, ids = [], []
        for r in rows:
            f = _features(r)
            if f is not None:
                feats.append(f)
                ids.append(r["norad_id"])
        if len(feats) < 20:
            raise ValueError("not enough objects with orbital data to fit an anomaly model")
        X = np.array(feats, dtype=float)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        Xs = (X - self._mean) / self._std
        self._forest = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        self._forest.fit(Xs)
        self._ids = ids
        self._X = X
        # normalize raw decision scores to 0..1 (higher = more anomalous)
        raw = -self._forest.score_samples(Xs)
        self._raw_lo, self._raw_hi = float(raw.min()), float(raw.max())
        # cache a score for every fitted object — computed in BATCH here so
        # the population ranking never re-scores object-by-object (which was
        # ~60s over the live catalogue; batched it is well under a second)
        labels = self._forest.predict(Xs)
        top_j = np.argmax(np.abs(Xs), axis=1)
        self._scores = [
            AnomalyScore(norad_id=ids[k], score=self._norm(float(raw[k])),
                         is_outlier=bool(labels[k] == -1),
                         top_feature=FEATURE_NAMES[int(top_j[k])],
                         top_feature_z=float(Xs[k, int(top_j[k])]))
            for k in range(len(ids))
        ]
        return self

    def all_scores(self) -> list["AnomalyScore"]:
        """Every fitted object's cached anomaly score, most anomalous first."""
        return sorted(self._scores, key=lambda s: s.score, reverse=True)

    @property
    def size(self) -> int:
        """Number of objects the model was fitted on."""
        return len(self._scores)

    def _norm(self, raw: float) -> float:
        span = self._raw_hi - self._raw_lo
        return 0.0 if span <= 0 else float(np.clip((raw - self._raw_lo) / span, 0, 1))

    def score_row(self, row: dict) -> Optional[AnomalyScore]:
        f = _features(row)
        if f is None or self._forest is None:
            return None
        x = np.array(f, dtype=float)
        xs = (x - self._mean) / self._std
        raw = -float(self._forest.score_samples(xs.reshape(1, -1))[0])
        label = self._forest.predict(xs.reshape(1, -1))[0] == -1
        j = int(np.argmax(np.abs(xs)))
        return AnomalyScore(
            norad_id=row["norad_id"], score=self._norm(raw), is_outlier=bool(label),
            top_feature=FEATURE_NAMES[j], top_feature_z=float(xs[j]),
        )

    def top_anomalies(self, rows: list[dict], n: int = 30) -> list[AnomalyScore]:
        scored = [s for s in (self.score_row(r) for r in rows) if s is not None]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:n]
