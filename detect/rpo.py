"""RPO / inspector detection (F2.1) — THE differentiator (Tier 2, §2).

Per handoff §5/§6: in the RIC frame, detect SUSTAINED low relative
velocity that "parks" one object near another, as opposed to an ordinary
brief conjunction — real inspection/tow behavior keeps range AND relative
speed low together for days to weeks, not just a single close instant.
"Matched plane" is not a separate check here: two objects can't sustain
low relative speed at all unless their orbital planes are already close,
so the range+speed criterion below implicitly requires it.

Grounded against real archive data (2026-07-15): SJ-21's actual tow of
BeiDou-2 G2 shows range collapsing from ~32,000 km to <5 km over ~3.5
weeks (Dec 2021), then holding under ~2 km with relative speed
<0.001 km/s for nearly a month before release — the textbook signature.
CLOSE_RANGE_KM and LOITER_SPEED_KMS are set generously around that real
episode, not guessed. (Cosmos 2542/2543 shadowing USA 245 — the other
labeled RPO event — can't be validated this way: USA 245 has zero
archived TLEs, confirmed via check-coverage. Don't fabricate that case.)

Note on precision (§9): SGP4/TLE states carry km-level absolute error.
Sub-km range figures below describe the RELATIVE geometry of two nearby
objects propagated the same way at the same instant — directionally
meaningful for a real event like this, but not precise range in meters.

Method:
1. Align both objects' TLE histories onto a common time grid over their
   OVERLAPPING coverage window (core/propagation.py::propagate_aligned),
   skipping any point where either object's nearest TLE is too stale to
   trust.
2. Compute RIC relative range/speed over that grid
   (core/frames.py::relative_range_series).
3. Find contiguous "close+slow" episodes (range < CLOSE_RANGE_KM AND
   relative speed < LOITER_SPEED_KMS) and classify by total time spent
   in them — one brief fast flyby crosses close range only instantaneously
   and never accumulates sustained time; a real tow/shadow does.

Honesty rule (§9): too few aligned samples (short/sparse overlapping
coverage) returns None rather than a guess — mirrors detect_maneuvers.

Known limitation, found live against real data: this detector cannot
tell genuine RPO from sub-satellite DEPLOYMENT on range/speed alone.
Cosmos 2542 vs Cosmos 2543 (2543 was released FROM 2542) scores "likely"
from a 3.2-day close episode right at their real Dec 2019 deployment — a
true close+slow signature, wrong cause. `assess_rpo` now checks SATCAT
launch designators (same intl_desig-prefix trick as detect/breakup.py)
and attaches `same_launch_lineage` + a caveat in the summary when actor
and target share a launch — it doesn't suppress the result (a shared
launch doesn't PROVE it's deployment, and later RPO can still happen
between siblings), it flags it for an analyst to check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.frames import relative_range_series
from core.propagation import propagate_aligned
from data.store import TleRecord, get_satcat_many

MIN_ALIGNED_SAMPLES = 20      # need this many aligned points before assessing at all
CLOSE_RANGE_KM = 100.0        # range below this counts as "close"
LOITER_SPEED_KMS = 0.05       # relative speed below this during a close episode = loitering
MIN_SUSTAINED_DAYS_LIKELY = 3.0  # total close+slow time above this = 'likely', not just 'possible'


@dataclass(frozen=True)
class RpoEpisode:
    start_iso: str
    end_iso: str
    duration_days: float
    min_range_km: float
    mean_rel_speed_kms: float


@dataclass(frozen=True)
class RpoAssessment:
    actor_norad: int
    target_norad: int
    label: str                     # 'none' | 'possible' | 'likely'
    min_range_km: float
    n_episodes: int
    total_close_days: float
    episodes: tuple[RpoEpisode, ...]
    n_aligned_samples: int
    coverage_start_iso: str
    coverage_end_iso: str
    same_launch_lineage: Optional[bool]  # True/False if SATCAT resolved both, else None

    def summary(self) -> str:
        lines = [
            f"NORAD {self.actor_norad} vs {self.target_norad}: {self.label} "
            f"(min range {self.min_range_km:.1f} km, {self.n_episodes} close episode(s), "
            f"{self.total_close_days:.1f} total days close; {self.n_aligned_samples} aligned "
            f"samples over {self.coverage_start_iso}..{self.coverage_end_iso})"
        ]
        for e in self.episodes:
            lines.append(f"  - {e.start_iso} -> {e.end_iso} ({e.duration_days:.1f}d): "
                          f"min {e.min_range_km:.1f} km, "
                          f"mean rel speed {e.mean_rel_speed_kms * 1000:.2f} m/s")
        if self.label != "none" and self.same_launch_lineage:
            lines.append("  CAVEAT: actor and target share a launch designator (SATCAT) — "
                          "this closeness may be sub-satellite DEPLOYMENT, not inspection. "
                          "Confirm against launch/deployment records before calling it RPO.")
        return "\n".join(lines)


def classify_episodes(
    times: list, range_km, speed_kms,
) -> tuple[list[RpoEpisode], str, float]:
    """Pure classification core: given an aligned relative-motion time
    series, find contiguous close+slow episodes and label the pair.
    Pulled out from assess_rpo so it's testable with synthetic arrays,
    independent of TLE propagation. Returns (episodes, label, total_close_days).
    """
    close = (range_km < CLOSE_RANGE_KM) & (speed_kms < LOITER_SPEED_KMS)

    episodes: list[RpoEpisode] = []
    i, n = 0, len(times)
    while i < n:
        if not close[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and close[j + 1]:
            j += 1
        episodes.append(RpoEpisode(
            start_iso=times[i].strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_iso=times[j].strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_days=(times[j] - times[i]).total_seconds() / 86400.0,
            min_range_km=float(range_km[i:j + 1].min()),
            mean_rel_speed_kms=float(speed_kms[i:j + 1].mean()),
        ))
        i = j + 1

    total_close_days = sum(e.duration_days for e in episodes)
    if total_close_days >= MIN_SUSTAINED_DAYS_LIKELY:
        label = "likely"
    elif episodes:
        label = "possible"
    else:
        label = "none"

    return episodes, label, total_close_days


def assess_rpo(
    actor_id: int,
    target_id: int,
    actor_tles: list[TleRecord],
    target_tles: list[TleRecord],
    step_hours: float = 6.0,
) -> Optional[RpoAssessment]:
    times, r_a, v_a, r_b, v_b = propagate_aligned(actor_tles, target_tles, step_hours=step_hours)
    if len(times) < MIN_ALIGNED_SAMPLES:
        return None

    rel = relative_range_series(r_a, v_a, r_b, v_b)
    range_km = rel["range_km"]
    speed_kms = rel["rel_speed_kms"]
    episodes, label, total_close_days = classify_episodes(times, range_km, speed_kms)

    satcat = get_satcat_many([actor_id, target_id])
    actor_desig = satcat.get(actor_id, {}).get("intl_desig")
    target_desig = satcat.get(target_id, {}).get("intl_desig")
    same_launch_lineage = (
        None if not actor_desig or not target_desig
        else actor_desig[:8] == target_desig[:8]
    )

    return RpoAssessment(
        actor_norad=actor_id,
        target_norad=target_id,
        label=label,
        min_range_km=float(range_km.min()),
        n_episodes=len(episodes),
        total_close_days=total_close_days,
        episodes=tuple(episodes),
        n_aligned_samples=len(times),
        coverage_start_iso=times[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        coverage_end_iso=times[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        same_launch_lineage=same_launch_lineage,
    )
