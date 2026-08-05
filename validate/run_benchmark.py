"""Validation benchmark runner (§6, §8) — runs detectors against real
Space-Track/SATCAT data already ingested into the local store, and checks
results against the labeled events in validate/events.py.

Currently covers F1.3 (breakup detection): two real fragmentation
positives (Cosmos 1408, Fengyun-1C) and two real controls — ordinary
launches that must NOT be flagged as breakups (the Cosmos 2542/2543
launch, and ISS's own launch). Run via `python cli.py validate-breakup`.

Honesty rule (§9): if an object's launch prefix can't be resolved (no
SATCAT row locally), the result is UNRESOLVED, never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from data.store import get_satcat_many
from detect.breakup import detect_from_launch_prefix
from validate.events import EVENTS

# Fragmentation positives: the ASAT/breakup events themselves.
FRAGMENTATION_EVENT_KEYS = ["cosmos1408_asat", "fengyun1c_asat"]

# Controls: ordinary launches (payload + rocket body, no debris cloud) that
# a breakup detector must correctly say NO to. Deliberately reuses launches
# already relevant to other events rather than inventing new ones.
CONTROL_EVENT_KEYS = ["cosmos2542_usa245", "iss_reboosts"]


@dataclass(frozen=True)
class BreakupBenchmarkResult:
    event_key: str
    expected: str                  # 'positive' | 'control'
    launch_prefix: Optional[str]
    found: bool
    n_fragments: Optional[int]
    confidence: Optional[str]
    passed: bool

    def summary(self) -> str:
        if self.launch_prefix is None:
            return f"{self.event_key}: UNRESOLVED — no SATCAT row for the reference object"
        detail = (f"{self.n_fragments} fragments, confidence {self.confidence}"
                   if self.found else "no cluster found")
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.event_key} (expected {self.expected}, prefix {self.launch_prefix}): {detail}"


def _launch_prefix_for(norad_id: int) -> Optional[str]:
    row = get_satcat_many([norad_id]).get(norad_id)
    if not row or not row.get("intl_desig"):
        return None
    return row["intl_desig"][:8]


def run_breakup_benchmark() -> list[BreakupBenchmarkResult]:
    by_key = {e.key: e for e in EVENTS}
    results: list[BreakupBenchmarkResult] = []

    for key, expected in (
        [(k, "positive") for k in FRAGMENTATION_EVENT_KEYS]
        + [(k, "control") for k in CONTROL_EVENT_KEYS]
    ):
        event = by_key[key]
        norad_id = event.actor_norad[0]
        prefix = _launch_prefix_for(norad_id)
        cand = detect_from_launch_prefix(prefix) if prefix else None
        found = cand is not None
        results.append(BreakupBenchmarkResult(
            event_key=key,
            expected=expected,
            launch_prefix=prefix,
            found=found,
            n_fragments=cand.n_fragments if cand else None,
            confidence=cand.confidence if cand else None,
            passed=(found if expected == "positive" else not found),
        ))
    return results
