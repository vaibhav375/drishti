"""Conjunction-filtering cascade (F2.2) — mandatory before any all-pairs
screening at catalog scale (handoff §7: "naive all-pairs over ~25k objects
will not finish on an M1"). Two cheap, eliminating-first stages ahead of
any expensive per-time propagation:

1. `altitude_sieve` — perigee/apogee interval overlap. A HARD filter: if
   two orbits' radial-distance ranges never overlap, they can never
   coincide in position, full stop — no geometry or timing needed to
   rule this out. An interval sweep over objects sorted by perigee, so
   it's O(N log N + P) rather than O(N²) (P = surviving pairs).
2. `geometry_filter` — RELATIVE INCLINATION between the two orbital
   PLANES (closed-form: i, RAAN only — no sampling). Two orbits sharing
   an altitude band can still never meaningfully approach if their planes
   are far apart; this is O(1) per pair.

Design history (found live, 2026-07-15): stage 2 was originally a
point-cloud "minimum distance between fixed ellipses" (sample each
orbit's ellipse, take the closest pairwise point distance). Two bugs in
that approach before this: (a) a FIXED sample count under-resolved
GEO-scale orbits and spuriously filtered out the real SJ-21/BeiDou-2 G2
pair; (b) fixed with arc-length-adaptive sampling, it then OOM-killed on
the real ~570-object GEO belt, because hundreds of near-identical,
near-coplanar circular orbits sharing one thin shell put ENTIRE ARCS of
points within any reasonable threshold of each other simultaneously —
not sparse crossings, a combinatorial explosion. That isn't a fixable
resolution/performance bug; a point-cloud minimum-distance is the wrong
tool for a densely coplanar population. Relative inclination is cheap,
bounded, and asks the question stage 2 actually needs to ask: could
these planes ever bring the objects close, ignoring exactly how.

Honest limitation this leaves: within one densely populated, near-
coplanar regime (the GEO belt is the real example — hundreds of objects,
most near-0° inclination), relative inclination alone does NOT
discriminate much; nearly everyone shares a plane there by construction.
That's a true statement about GEO, not a gap in this filter — GEO
screening in practice needs RAAN/longitude-slot binning or straight
time-domain propagation, neither of which this module attempts. This
cascade's real value is eliminating CROSS-regime pairs (the literal
O(N²)-at-25k-objects problem §7 names) and diverging-plane SAME-regime
pairs; it is not a GEO-belt-specific solution.

Stage 3 (actual time-domain close-approach checking) is NOT reimplemented
here — pairs surviving both filters are few enough to hand to
detect/rpo.py::assess_rpo (or a future dedicated Pc screen) for the
expensive per-time analysis. This module's job ends at producing that
candidate list.

Honesty rule (§9): this is a coarse SCREEN, not a detector. A pair
surviving the cascade is a CANDIDATE for further analysis, not a
confirmed close approach — say so in every summary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.elements import MeanElements

ALTITUDE_MARGIN_KM = 50.0             # generous margin for TLE/element uncertainty
ORBIT_GEOMETRY_MAX_REL_INCL_DEG = 30.0  # generous plane-alignment gate — a sieve,
                                          # err toward not excluding real candidates


@dataclass(frozen=True)
class ScreenedPair:
    norad_a: int
    norad_b: int
    altitude_overlap_km: float
    relative_inclination_deg: float

    def summary(self) -> str:
        return (f"NORAD {self.norad_a} / {self.norad_b}: CANDIDATE for further analysis — "
                f"altitude bands overlap by {self.altitude_overlap_km:.0f} km, "
                f"relative orbital-plane inclination {self.relative_inclination_deg:.2f}° "
                "(coarse screen, not a confirmed close approach)")


def altitude_sieve(
    elements: list[MeanElements], margin_km: float = ALTITUDE_MARGIN_KM
) -> list[tuple[int, int]]:
    """All (norad_a, norad_b) pairs whose perigee/apogee bands overlap.

    Sweep over objects sorted by interval start; O(N log N + N*|active|),
    a large reduction from O(N²) whenever orbital regimes are diverse
    (LEO objects are never compared against GEO objects, etc.).
    """
    intervals = sorted(
        ((el.perigee_km - margin_km, el.apogee_km + margin_km, el.norad_id) for el in elements),
        key=lambda t: t[0],
    )
    pairs: list[tuple[int, int]] = []
    active: list[tuple[float, int]] = []  # (interval end, norad_id)
    for start, end, nid in intervals:
        active = [(e, n) for e, n in active if e >= start]
        for _, n in active:
            pairs.append((min(n, nid), max(n, nid)))
        active.append((end, nid))
    return pairs


def relative_inclination_deg(el_a: MeanElements, el_b: MeanElements) -> float:
    """Angle between the two orbital planes — closed-form, O(1), no
    sampling. Real SJ-21 vs BeiDou-2 G2 (i≈8.18°/8.15°, RAAN≈50.84°/50.87°)
    show ~0.03° here, correctly near-coplanar.
    """
    i_a, i_b = np.radians(el_a.incl_deg), np.radians(el_b.incl_deg)
    d_raan = np.radians(el_a.raan_deg - el_b.raan_deg)
    cos_rel = np.cos(i_a) * np.cos(i_b) + np.sin(i_a) * np.sin(i_b) * np.cos(d_raan)
    cos_rel = float(np.clip(cos_rel, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_rel)))


def geometry_filter(
    pairs: list[tuple[int, int]],
    elements_by_id: dict[int, MeanElements],
    max_relative_incl_deg: float = ORBIT_GEOMETRY_MAX_REL_INCL_DEG,
) -> list[ScreenedPair]:
    out: list[ScreenedPair] = []
    for a, b in pairs:
        el_a, el_b = elements_by_id.get(a), elements_by_id.get(b)
        if el_a is None or el_b is None:
            continue
        rel_incl = relative_inclination_deg(el_a, el_b)
        if rel_incl <= max_relative_incl_deg:
            out.append(ScreenedPair(
                norad_a=a,
                norad_b=b,
                altitude_overlap_km=(min(el_a.apogee_km, el_b.apogee_km)
                                      - max(el_a.perigee_km, el_b.perigee_km)),
                relative_inclination_deg=rel_incl,
            ))
    return out


def run_cascade(
    elements: list[MeanElements],
    altitude_margin_km: float = ALTITUDE_MARGIN_KM,
    max_relative_incl_deg: float = ORBIT_GEOMETRY_MAX_REL_INCL_DEG,
) -> list[ScreenedPair]:
    """Full cascade: mean elements (one per object, e.g. from each
    object's latest stored TLE) -> altitude sieve -> plane-alignment
    filter. Returns CANDIDATE pairs for further (expensive) time-domain
    analysis — this cascade does not itself confirm a close approach.
    """
    by_id = {el.norad_id: el for el in elements}
    coarse_pairs = altitude_sieve(elements, margin_km=altitude_margin_km)
    return geometry_filter(coarse_pairs, by_id, max_relative_incl_deg=max_relative_incl_deg)


def screen_top(
    elements: list[MeanElements],
    top_k: int = 60,
    altitude_margin_km: float = ALTITUDE_MARGIN_KM,
    max_relative_incl_deg: float = ORBIT_GEOMETRY_MAX_REL_INCL_DEG,
) -> tuple[int, list[ScreenedPair]]:
    """Same cascade as `run_cascade`, vectorized, for catalogue-scale views.

    Returns (total surviving pairs, the `top_k` most plane-aligned pairs).
    Found live (2026-09-30): at the full public Celestrak catalogue (~16.8k
    objects) the thousands of Starlink satellites sharing one ~550 km shell
    make `run_cascade` materialize tens of millions of Python tuples — the
    web page never returned. This computes the identical survivor set with
    numpy per object and only builds ScreenedPair objects for the top_k.
    """
    if not elements:
        return 0, []
    ids = np.array([el.norad_id for el in elements])
    peri = np.array([el.perigee_km for el in elements])
    apo = np.array([el.apogee_km for el in elements])
    inc = np.radians([el.incl_deg for el in elements])
    raan = np.radians([el.raan_deg for el in elements])
    order = np.argsort(peri - altitude_margin_km, kind="stable")
    ids, peri, apo, inc, raan = ids[order], peri[order], apo[order], inc[order], raan[order]
    start, end = peri - altitude_margin_km, apo + altitude_margin_km
    sin_i, cos_i = np.sin(inc), np.cos(inc)
    cos_gate = np.cos(np.radians(max_relative_incl_deg))

    # running count + running top_k only — storing every survivor peaked at
    # >500 MB on the real catalogue (Starlink shells), over a 512 MB host
    total = 0
    top_a = np.empty(0, dtype=np.int64)
    top_b = np.empty(0, dtype=np.int64)
    top_c = np.empty(0)  # cos(relative inclination); larger = more aligned
    for k in range(1, len(ids)):
        j = np.nonzero(end[:k] >= start[k])[0]
        if j.size == 0:
            continue
        c = np.clip(cos_i[j] * cos_i[k] + sin_i[j] * sin_i[k] * np.cos(raan[j] - raan[k]), -1.0, 1.0)
        keep = c >= cos_gate
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue
        total += n_keep
        if top_c.size >= top_k:
            keep &= c > top_c.min()
            if not keep.any():
                continue
        top_a = np.concatenate([top_a, j[keep]])
        top_b = np.concatenate([top_b, np.full(int(keep.sum()), k)])
        top_c = np.concatenate([top_c, c[keep]])
        if top_c.size > top_k:
            sel = np.argsort(-top_c, kind="stable")[:top_k]
            top_a, top_b, top_c = top_a[sel], top_b[sel], top_c[sel]
    if total == 0:
        return 0, []
    out = []
    for t in np.argsort(-top_c, kind="stable"):
        a, b = top_a[t], top_b[t]
        na, nb = int(ids[a]), int(ids[b])
        out.append(ScreenedPair(
            norad_a=min(na, nb),
            norad_b=max(na, nb),
            altitude_overlap_km=float(min(apo[a], apo[b]) - max(peri[a], peri[b])),
            relative_inclination_deg=float(np.degrees(np.arccos(top_c[t]))),
        ))
    return total, out
