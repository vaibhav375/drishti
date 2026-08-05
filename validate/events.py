"""Labeled validation event set (§6) — the benchmark spine and the
paper's dataset. Dates are retrieval windows, not detection timestamps.

Week-1 rule (§4): before building any detector on an event, verify the
Space-Track archive actually covers it:

    python cli.py check-coverage --norad 44797 --norad 44835 --norad 39232 \
        --start 2019-12-01 --end 2020-03-15

If coverage is zero for any object, STOP and flag it — do not fake output.

NORAD IDs below are believed-correct from public catalogs but MUST be
confirmed against SATCAT during the week-1 coverage check before any
result is reported against them (per §9: never fabricate; verify).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class LabeledEvent:
    key: str
    description: str
    event_type: str          # 'rpo' | 'fragmentation' | 'conjunction' | 'routine'
    label: str               # 'positive' | 'control'
    window_start: date
    window_end: date
    actor_norad: list[int] = field(default_factory=list)   # the object(s) acting
    target_norad: list[int] = field(default_factory=list)  # object(s) acted upon
    notes: str = ""


EVENTS: list[LabeledEvent] = [
    LabeledEvent(
        key="cosmos2542_usa245",
        description="Cosmos 2542/2543 shadowing USA 245 (KH-11)",
        event_type="rpo", label="positive",
        window_start=date(2019, 12, 1), window_end=date(2020, 3, 15),
        actor_norad=[44797, 44835],   # Cosmos 2542, Cosmos 2543
        target_norad=[39232],         # USA 245 — confirmed via SATCAT 2026-07-14
                                       # (was wrongly 32711 = NAVSTAR 62/USA 201,
                                       # a GPS sat)
        notes="Sub-satellite 2543 deployed from 2542; both maneuvered near USA 245. "
              "CONFIRMED 2026-07-15 via check-coverage: 0 archived TLEs for 39232 "
              "in the window (Space-Track gp_history) — classified as predicted. "
              "44797/44835 have good coverage (236/227 TLEs) so actor maneuvers "
              "can still be analyzed; target ephemeris must be marked unavailable/"
              "approximate wherever this event is used, never fabricated.",
    ),
    LabeledEvent(
        key="luch_geo_drift",
        description="Luch (Olymp-K) GEO drift between comsats",
        event_type="rpo", label="positive",
        window_start=date(2014, 10, 1), window_end=date(2016, 12, 31),
        actor_norad=[40258],
        target_norad=[],  # multiple GEO comsats over time; resolve per sub-window
        notes="Long-baseline serial-inspector case — the temporal graph (F3.2) target.",
    ),
    LabeledEvent(
        key="sj21_beidou_tow",
        description="Shijian-21 tow of BeiDou-2 G2 to graveyard",
        event_type="rpo", label="positive",
        window_start=date(2021, 12, 1), window_end=date(2022, 2, 15),
        actor_norad=[49330],          # SJ-21
        target_norad=[34779],         # BeiDou-2 G2 — confirmed via SATCAT
                                       # 2026-07-14 (was wrongly 36287 = G1)
    ),
    LabeledEvent(
        key="cosmos1408_asat",
        description="Cosmos 1408 destroyed by Russian ASAT test",
        event_type="fragmentation", label="positive",
        window_start=date(2021, 10, 15), window_end=date(2022, 1, 31),
        actor_norad=[13552],
        notes="~1500 tracked fragments; catalog-delta burst signature (F1.3).",
    ),
    LabeledEvent(
        key="fengyun1c_asat",
        description="Fengyun-1C destroyed by Chinese ASAT test",
        event_type="fragmentation", label="positive",
        window_start=date(2006, 12, 1), window_end=date(2007, 6, 30),
        actor_norad=[25730],
        notes="Largest debris event on record; fragments catalogued over months.",
    ),
    LabeledEvent(
        key="iridium_cosmos_collision",
        description="Iridium 33 / Cosmos 2251 collision",
        event_type="conjunction", label="positive",
        window_start=date(2009, 1, 15), window_end=date(2009, 4, 30),
        actor_norad=[24946, 22675],   # Iridium 33, Cosmos 2251
        notes="Both conjunction screening (pre-event) and breakup (post-event) target.",
    ),
    LabeledEvent(
        key="iss_reboosts",
        description="ISS routine reboost maneuvers",
        event_type="routine", label="control",
        window_start=date(2023, 1, 1), window_end=date(2023, 12, 31),
        actor_norad=[25544],
        notes="Detector should FIND the burns but pattern-of-life (F1.2) must "
              "classify them as routine — controls matter as much as positives.",
    ),
]


def by_key(key: str) -> LabeledEvent:
    for e in EVENTS:
        if e.key == key:
            return e
    raise KeyError(key)


def all_norad_ids() -> list[int]:
    ids: set[int] = set()
    for e in EVENTS:
        ids.update(e.actor_norad)
        ids.update(e.target_norad)
    return sorted(ids)
