"""Pattern-of-life baseline (F1.2).

Per handoff §5: per-object maneuver tempo and typical burn size, used to
flag maneuvers that deviate from an OBJECT'S OWN history — not a global
threshold. Reuses the median+MAD robust-baseline idiom from F1.1
(detect/maneuver.py) for the same reason: a routine station-keeping burn
must not be treated as the "normal" case that swamps true outliers, and
a true outlier must not be allowed to define what's normal for the object.

Input is the `events` list `detect_maneuvers` already found (chronological,
since it walks TLE pairs in epoch order) — this module classifies THOSE
detections, it does not re-detect from raw TLEs.

Campaign grouping (learned from real ISS 2023 archive data, not assumed):
dense Space-Track history often yields SEVERAL closely-timed flagged
residuals — hours apart — around one physical burn (multi-pulse reboosts,
or just multiple TLEs generated while the orbit is still settling). Treated
as independent "maneuvers", these inflate tempo with tight intra-burn gaps,
which then makes the real, regular gap BETWEEN burns (e.g. ISS's ~2-4 week
reboost cadence) look anomalous — exactly backwards, since that cadence is
the routine case §6 requires this detector to recognize. So detections
within CAMPAIGN_GAP_DAYS of each other are grouped into one ManeuverCampaign
before baselining or classifying; tempo and burn-size baselines are built
from campaigns, not raw per-pair residuals.

Honesty rule (§9): with fewer than MIN_MANEUVERS detected campaigns, there
is no basis for a pattern — build_baseline returns None and nothing is
classified, rather than judging deviation against an unfounded sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from detect.maneuver import ManeuverEvent

MIN_MANEUVERS = 3          # need this many campaigns before a pattern exists
DV_Z_ANOMALOUS = 3.0       # burn-size deviation from the object's own typical burn
TEMPO_Z_ANOMALOUS = 3.0    # gap deviation from the object's own typical tempo
CAMPAIGN_GAP_DAYS = 1.0    # flagged residuals closer together than this = one campaign
MIN_RELATIVE_SIGMA = 0.15  # sigma floor as a fraction of the baseline's own value


@dataclass(frozen=True)
class ManeuverCampaign:
    """One or more consecutive flagged residuals treated as a single burn."""
    norad_id: int
    start_epoch: str
    end_epoch: str
    peak_dv_ms: float
    peak_event: ManeuverEvent
    members: tuple[ManeuverEvent, ...]

    @property
    def n_pulses(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class ObjectBaseline:
    norad_id: int
    n_campaigns: int
    typical_dv_ms: float        # median campaign peak burn size, own history
    dv_mad_ms: float
    typical_tempo_days: float   # median days between consecutive campaigns
    tempo_mad_days: float
    first_epoch: str
    last_epoch: str


@dataclass(frozen=True)
class PatternClassification:
    campaign: ManeuverCampaign
    label: str                             # 'routine' | 'anomalous'
    dv_z: float                            # burn-size deviation from typical
    tempo_days_since_prior: Optional[float]
    tempo_z: Optional[float]
    reasons: tuple[str, ...]

    def summary(self) -> str:
        why = f" — {'; '.join(self.reasons)}" if self.reasons else ""
        pulses = f" ({self.campaign.n_pulses} pulses)" if self.campaign.n_pulses > 1 else ""
        return f"{self.campaign.start_epoch}: {self.label}{why}{pulses}"


def _epoch_dt(epoch_iso: str) -> datetime:
    return datetime.fromisoformat(epoch_iso)


def group_into_campaigns(
    events: list[ManeuverEvent], gap_days: float = CAMPAIGN_GAP_DAYS
) -> list[ManeuverCampaign]:
    """Collapse closely-timed flagged residuals into single campaigns.

    `events` must be chronological (detect_maneuvers' natural order).
    """
    if not events:
        return []

    groups: list[list[ManeuverEvent]] = [[events[0]]]
    for prev, cur in zip(events, events[1:]):
        gap = (_epoch_dt(cur.epoch_iso) - _epoch_dt(prev.epoch_iso)).total_seconds() / 86400.0
        if gap <= gap_days:
            groups[-1].append(cur)
        else:
            groups.append([cur])

    campaigns: list[ManeuverCampaign] = []
    for members in groups:
        peak = max(members, key=lambda e: e.dv_ms)
        campaigns.append(ManeuverCampaign(
            norad_id=members[0].norad_id,
            start_epoch=members[0].epoch_iso,
            end_epoch=members[-1].epoch_iso,
            peak_dv_ms=peak.dv_ms,
            peak_event=peak,
            members=tuple(members),
        ))
    return campaigns


def build_baseline(campaigns: list[ManeuverCampaign]) -> Optional[ObjectBaseline]:
    """Baseline from an object's own maneuver-campaign history."""
    if len(campaigns) < MIN_MANEUVERS:
        return None

    dvs = np.array([c.peak_dv_ms for c in campaigns])
    dv_med = float(np.median(dvs))
    dv_mad = float(np.median(np.abs(dvs - dv_med)))

    epochs = [_epoch_dt(c.start_epoch) for c in campaigns]
    gaps = np.array([(b - a).total_seconds() / 86400.0 for a, b in zip(epochs, epochs[1:])])
    tempo_med = float(np.median(gaps))
    tempo_mad = float(np.median(np.abs(gaps - tempo_med)))

    return ObjectBaseline(
        norad_id=campaigns[0].norad_id,
        n_campaigns=len(campaigns),
        typical_dv_ms=dv_med,
        dv_mad_ms=dv_mad,
        typical_tempo_days=tempo_med,
        tempo_mad_days=tempo_mad,
        first_epoch=campaigns[0].start_epoch,
        last_epoch=campaigns[-1].start_epoch,
    )


def classify(
    campaigns: list[ManeuverCampaign], baseline: ObjectBaseline
) -> list[PatternClassification]:
    """Classify each maneuver campaign against the object's own baseline.

    MAD→σ (1.4826×MAD) matches F1.1's convention. The floor is RELATIVE
    (MIN_RELATIVE_SIGMA × the baseline value), not a bare epsilon: with
    only a handful of campaigns, MAD frequently lands on exactly zero
    (ties at the median are common with small N) — a tiny absolute floor
    then turns an ordinary one-day drift in a ~5-day cadence into a
    thousand-sigma "anomaly". A relative floor keeps the threshold
    dimensionally sane for both m/s bursts and day-scale gaps, and still
    degrades to a near-zero absolute floor when the baseline value itself
    is near zero.
    """
    dv_sigma = max(1.4826 * baseline.dv_mad_ms, MIN_RELATIVE_SIGMA * baseline.typical_dv_ms, 1e-3)
    tempo_sigma = max(1.4826 * baseline.tempo_mad_days,
                       MIN_RELATIVE_SIGMA * baseline.typical_tempo_days, 1e-3)
    epochs = [_epoch_dt(c.start_epoch) for c in campaigns]

    out: list[PatternClassification] = []
    for idx, c in enumerate(campaigns):
        dv_z = (c.peak_dv_ms - baseline.typical_dv_ms) / dv_sigma
        tempo_days: Optional[float] = None
        tempo_z: Optional[float] = None
        reasons: list[str] = []

        if abs(dv_z) >= DV_Z_ANOMALOUS:
            reasons.append(f"burn size {c.peak_dv_ms:.2f} m/s is {dv_z:+.1f}σ from typical")

        if idx > 0:
            tempo_days = (epochs[idx] - epochs[idx - 1]).total_seconds() / 86400.0
            tempo_z = (tempo_days - baseline.typical_tempo_days) / tempo_sigma
            if abs(tempo_z) >= TEMPO_Z_ANOMALOUS:
                reasons.append(
                    f"{tempo_days:.1f}d since prior campaign vs "
                    f"{baseline.typical_tempo_days:.1f}d typical"
                )

        out.append(PatternClassification(
            campaign=c,
            label="anomalous" if reasons else "routine",
            dv_z=float(dv_z),
            tempo_days_since_prior=tempo_days,
            tempo_z=tempo_z,
            reasons=tuple(reasons),
        ))
    return out


def pattern_of_life(
    events: list[ManeuverEvent],
) -> tuple[list[PatternClassification], Optional[ObjectBaseline]]:
    """Group raw detections into campaigns, build the baseline, classify.

    Mirrors detect_maneuvers' (results, baseline_or_None) return shape.
    """
    campaigns = group_into_campaigns(events)
    baseline = build_baseline(campaigns)
    if baseline is None:
        return [], None
    return classify(campaigns, baseline), baseline
