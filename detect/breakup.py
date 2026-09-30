"""Breakup / fragmentation detection (F1.3).

Per handoff §5: watch catalog delta — a correlated burst of new NORAD IDs
sharing both launch lineage and orbital regime is a candidate fragmentation
event. Two entry points share one clustering core:

* `detect_from_new_ids` — the live/forward-looking path: feed it whatever
  norad_ids appeared between two `catalog_snapshot` polls (data/store.py
  owns the snapshot table itself; this module doesn't poll).
* `detect_from_launch_prefix` — retrospective validation: pull every
  catalogued object sharing a known launch designator directly from
  SATCAT. This is what lets a real historical event (Cosmos 1408, 2021)
  be validated today without ever having polled the catalog back then.

Grounded against real SATCAT data (confirmed live, 2026-07-15): Cosmos
1408's launch designator 1982-092 carries 1 payload + 1 rocket body + 1806
DEB-typed fragments, tightly clustered at ~82.5° inclination and 140-300 km
perigee/apogee — the textbook signature. Contrast: Cosmos 2542/2543's
launch (2019-079) has exactly 5 objects and zero DEB — an ordinary
multi-payload/sub-satellite launch, not a breakup. MIN_FRAGMENTS and the
DEB-type filter below are chosen against that real contrast, not guessed.

A second real contrast forced a second gate: ISS's own launch designator
(1998-067) carries 148 DEB-typed objects — individually-named EVA litter
(dropped tools, a camera, a socket extension) accumulated over 25 years of
operations, NOT a fragmentation. Inclination clustering alone doesn't tell
these apart (both cling to the parent's orbit plane). What does: NORAD
catalog numbers are assigned roughly in order of first tracking, so a
genuine fragmentation shows up as debris catalogued in a tight burst
(Cosmos 1408: norad_id MAD/median ≈1.3%; Fengyun-1C ≈4.1%), while slowly
accumulated litter is scattered across decades of catalog growth (ISS
≈18.5%). MAX_ID_RELATIVE_MAD below is set from that real three-point
comparison, not guessed — see validate/run_benchmark.py for the check
that keeps it honest against all three plus the Cosmos 2542/2543 control.

Honesty rule (§9): a launch prefix with too few DEB-typed objects, no
SATCAT rows, or debris not catalogued in one temporal burst returns None
— never a low-confidence fabrication.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

MIN_FRAGMENTS = 10             # DEB objects sharing a launch prefix to call it a breakup
MIN_NEW_FRAGMENTS_TRIGGER = 3  # new DEB ids in ONE polling delta before even checking —
                                # guards against re-alerting on routine incremental
                                # cataloging of an already-known debris cloud
REGIME_INCL_SPREAD_DEG = 5.0   # inclination std this tight = one coherent fragment cloud
MAX_ID_RELATIVE_MAD = 0.08     # norad_id MAD / median above this = catalogued over
                                # years, not one event — accumulated litter, not a breakup


@dataclass(frozen=True)
class BreakupCandidate:
    launch_prefix: str
    parent_norad_id: Optional[int]
    parent_name: Optional[str]
    fragment_norad_ids: tuple[int, ...]
    n_fragments: int
    mean_inclination_deg: float
    incl_std_deg: float
    perigee_km_range: tuple[float, float]
    apogee_km_range: tuple[float, float]
    confidence: str                        # 'low' | 'medium' | 'high'

    def summary(self) -> str:
        who = f"{self.parent_name} ({self.parent_norad_id})" if self.parent_norad_id else self.launch_prefix
        return (
            f"{who}: {self.n_fragments} debris fragments, "
            f"incl {self.mean_inclination_deg:.1f}°±{self.incl_std_deg:.1f}, "
            f"perigee {self.perigee_km_range[0]:.0f}-{self.perigee_km_range[1]:.0f} km — "
            f"confidence {self.confidence}"
        )


def cluster_from_rows(prefix: str, rows: list[dict]) -> Optional[BreakupCandidate]:
    """Core clustering logic: does this launch's SATCAT cohort look like a
    fragmentation? Pure function over SATCAT-shaped dict rows — no DB
    access — so it's testable with synthetic rows and reusable from both
    entry points below.
    """
    debris = [
        r for r in rows
        if r.get("object_type") == "DEB"
        and r.get("inclination") is not None
        and r.get("perigee_km") is not None
        and r.get("apogee_km") is not None
    ]
    if len(debris) < MIN_FRAGMENTS:
        return None

    ids = np.array([d["norad_id"] for d in debris])
    id_median = float(np.median(ids))
    id_mad = float(np.median(np.abs(ids - id_median)))
    id_relative_mad = id_mad / id_median if id_median > 0 else 0.0
    if id_relative_mad > MAX_ID_RELATIVE_MAD:
        # scattered across decades of catalog growth -> accumulated litter,
        # not debris catalogued together after one physical event
        return None

    incls = np.array([d["inclination"] for d in debris])
    incl_std = float(np.std(incls))
    n = len(debris)

    if n >= 100 and incl_std < REGIME_INCL_SPREAD_DEG:
        confidence = "high"
    elif n >= MIN_FRAGMENTS and incl_std < REGIME_INCL_SPREAD_DEG * 2:
        confidence = "medium"
    else:
        confidence = "low"

    parent = (next((r for r in rows if r.get("object_type") == "PAY"), None)
              or next((r for r in rows if r.get("object_type") == "R/B"), None))

    perigees = [d["perigee_km"] for d in debris]
    apogees = [d["apogee_km"] for d in debris]

    return BreakupCandidate(
        launch_prefix=prefix,
        parent_norad_id=parent["norad_id"] if parent else None,
        parent_name=parent["name"] if parent else None,
        fragment_norad_ids=tuple(sorted(d["norad_id"] for d in debris)),
        n_fragments=n,
        mean_inclination_deg=float(np.mean(incls)),
        incl_std_deg=incl_std,
        perigee_km_range=(min(perigees), max(perigees)),
        apogee_km_range=(min(apogees), max(apogees)),
        confidence=confidence,
    )


def detect_from_launch_prefix(prefix: str) -> Optional[BreakupCandidate]:
    """Retrospective validation: does a KNOWN launch designator's current
    SATCAT cohort look like a fragmentation? (e.g. '1982-092' for Cosmos
    1408). No snapshot history needed — SATCAT already encodes lineage
    via intl_desig, so this works for events years before we started
    polling.
    """
    from data.store import get_satcat_by_launch_prefix
    rows = get_satcat_by_launch_prefix(prefix)
    if not rows:
        return None
    return cluster_from_rows(prefix, rows)


def detect_from_new_ids(new_norad_ids: Iterable[int]) -> list[BreakupCandidate]:
    """Live/forward-looking path: given norad_ids that appeared between two
    catalog_snapshot polls, group the NEW debris-typed ones by shared
    launch designator. A prefix only triggers a check if it contributed
    at least MIN_NEW_FRAGMENTS_TRIGGER new DEB ids in THIS delta — a
    single newly-tracked fragment added to an old, already-known debris
    cloud isn't a fresh event. Once triggered, the FULL current cohort
    for that prefix is pulled and clustered (not just the new ids) for
    an accurate characterization.
    """
    from data.store import get_satcat_by_launch_prefix, get_satcat_many
    ids = list(new_norad_ids)
    by_id = get_satcat_many(ids)

    new_debris_by_prefix: dict[str, list[dict]] = {}
    for nid in ids:
        row = by_id.get(nid)
        if not row or not row.get("intl_desig") or row.get("object_type") != "DEB":
            continue
        prefix = row["intl_desig"][:8]
        new_debris_by_prefix.setdefault(prefix, []).append(row)

    candidates: list[BreakupCandidate] = []
    for prefix, new_rows in new_debris_by_prefix.items():
        if len(new_rows) < MIN_NEW_FRAGMENTS_TRIGGER:
            continue
        full_rows = get_satcat_by_launch_prefix(prefix)
        cand = cluster_from_rows(prefix, full_rows)
        if cand:
            candidates.append(cand)
    return candidates
