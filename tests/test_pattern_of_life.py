"""Tests for F1.2 pattern-of-life baselining (detect/pattern_of_life.py).

Offline, synthetic — same fixture idiom as tests/test_core.py's F1.1
tests (self-consistent daily TLE history by string surgery over the ISS
element set). Real-event validation (ISS reboosts as the routine control)
lives in validate/ and needs the Space-Track archive.
"""
from __future__ import annotations

from detect.maneuver import detect_maneuvers
from detect.pattern_of_life import build_baseline, group_into_campaigns, pattern_of_life
from tests.test_core import _synthetic_history


def test_insufficient_campaigns_returns_no_baseline():
    # A single injected burn -> detect_maneuvers finds one event -> one
    # campaign, below MIN_MANEUVERS (3) -> no basis for a pattern yet.
    events, _ = detect_maneuvers(_synthetic_history(n_pairs=14, burn_at=8))
    assert len(events) < 3
    assert build_baseline(group_into_campaigns(events)) is None


def test_regular_station_keeping_burns_are_routine():
    # Several same-size burns at a roughly regular cadence — the ISS-reboost /
    # GEO station-keeping control case: the detector should find them, and
    # pattern-of-life should call all of them routine relative to each other.
    burns = {5: 0.010, 10: 0.011, 15: 0.0095, 20: 0.0105}
    events, _ = detect_maneuvers(_synthetic_history(n_pairs=24, burns=burns))
    assert len(events) == len(burns)

    classifications, baseline = pattern_of_life(events)
    assert baseline is not None
    assert baseline.n_campaigns == len(burns)
    assert all(c.label == "routine" for c in classifications)


def test_outsized_burn_flagged_anomalous_against_own_history():
    # Same small-burn cadence, but one burn is ~8x the others in size —
    # a real deviation from THIS object's own established pattern, not
    # just a global threshold.
    burns = {5: 0.010, 10: 0.011, 15: 0.080, 20: 0.0105}
    events, _ = detect_maneuvers(_synthetic_history(n_pairs=24, burns=burns))
    assert len(events) == len(burns)

    classifications, baseline = pattern_of_life(events)
    assert baseline is not None

    outsized = next(c for c in classifications
                     if c.campaign.peak_dv_ms == max(e.dv_ms for e in events))
    assert outsized.label == "anomalous"
    assert outsized.dv_z >= 3.0
    assert any("burn size" in r for r in outsized.reasons)

    # the routine burns around it should stay routine
    routine_count = sum(1 for c in classifications if c.label == "routine")
    assert routine_count == len(burns) - 1


def test_close_pulses_grouped_into_one_campaign_and_stay_routine():
    # Regression test for a real bug found against live ISS 2023 archive
    # data: dense Space-Track history yields several closely-timed flagged
    # residuals (hours apart) around one physical multi-pulse burn. Without
    # grouping, those tight intra-burn gaps set an artificially low
    # "typical tempo" baseline, which then made the real, regular gap
    # BETWEEN separate burns (here, and ISS's actual ~2-4 week reboost
    # cadence) look anomalous — exactly backwards from §6's requirement
    # that pattern-of-life classify routine reboosts as routine.
    #
    # Indices 10 and 11 are one day apart -> within CAMPAIGN_GAP_DAYS ->
    # one two-pulse campaign; the rest are lone single-pulse campaigns.
    burns = {5: 0.010, 10: 0.011, 11: 0.012, 16: 0.0105, 21: 0.0098}
    events, _ = detect_maneuvers(_synthetic_history(n_pairs=26, burns=burns))
    assert len(events) == len(burns)  # every pulse individually flagged

    campaigns = group_into_campaigns(events)
    assert len(campaigns) == 4
    multi_pulse = [c for c in campaigns if c.n_pulses > 1]
    assert len(multi_pulse) == 1
    assert multi_pulse[0].n_pulses == 2

    classifications, baseline = pattern_of_life(events)
    assert baseline is not None
    assert baseline.n_campaigns == 4
    assert all(c.label == "routine" for c in classifications)
