"""Tests for F2.1 RPO detection (detect/rpo.py).

Offline, synthetic time series for classify_episodes (the pure
classification core) — independent of TLE propagation. The propagation/
alignment path (propagate_aligned, assess_rpo end to end) was validated
against real archive data instead: SJ-21's documented tow of BeiDou-2 G2
("likely"), Cosmos 2542/2543's real Dec 2019 deployment ("likely", noted
as a known deployment-vs-RPO ambiguity), and Iridium 33 vs Cosmos 2251
pre-collision ("none", correctly distinguishing a fast conjunction from
loitering) — see README "Real-data findings".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from detect.rpo import (
    CLOSE_RANGE_KM,
    LOITER_SPEED_KMS,
    MIN_ALIGNED_SAMPLES,
    MIN_SUSTAINED_DAYS_LIKELY,
    assess_rpo,
    classify_episodes,
)


def _times(n: int, step_hours: float = 6.0):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(hours=i * step_hours) for i in range(n)]


def test_never_close_is_none():
    times = _times(40)
    range_km = np.full(40, 5000.0)
    speed_kms = np.full(40, 0.2)
    episodes, label, total_days = classify_episodes(times, range_km, speed_kms)
    assert episodes == []
    assert label == "none"
    assert total_days == 0.0


def test_brief_fast_flyby_is_none_not_possible():
    # Crosses under CLOSE_RANGE_KM briefly but relative speed stays high
    # throughout -- a fast conjunction, not loitering.
    n = 40
    times = _times(n)
    range_km = np.full(n, 5000.0)
    speed_kms = np.full(n, 1.0)  # well above LOITER_SPEED_KMS
    range_km[20] = 10.0  # instantaneously close
    episodes, label, total_days = classify_episodes(times, range_km, speed_kms)
    assert episodes == []
    assert label == "none"


def test_brief_close_and_slow_is_possible_not_likely():
    n = 40
    times = _times(n, step_hours=6.0)
    range_km = np.full(n, 5000.0)
    speed_kms = np.full(n, 0.2)
    # one close+slow sample (~6 hours, well under MIN_SUSTAINED_DAYS_LIKELY)
    range_km[20] = 10.0
    speed_kms[20] = 0.01
    episodes, label, total_days = classify_episodes(times, range_km, speed_kms)
    assert len(episodes) == 1
    assert label == "possible"
    assert total_days < MIN_SUSTAINED_DAYS_LIKELY


def test_sustained_close_and_slow_is_likely():
    # Mirrors the real SJ-21/BeiDou-2 G2 shape: several days of close+slow.
    step_hours = 6.0
    n = 40
    times = _times(n, step_hours=step_hours)
    range_km = np.full(n, 5000.0)
    speed_kms = np.full(n, 0.2)
    n_close = int(MIN_SUSTAINED_DAYS_LIKELY * 24 / step_hours) + 4
    range_km[5:5 + n_close] = 1.0
    speed_kms[5:5 + n_close] = 0.001
    episodes, label, total_days = classify_episodes(times, range_km, speed_kms)
    assert len(episodes) == 1
    assert label == "likely"
    assert total_days >= MIN_SUSTAINED_DAYS_LIKELY
    assert episodes[0].min_range_km == 1.0


def test_multiple_separate_episodes_accumulate_toward_likely():
    step_hours = 6.0
    n = 60
    times = _times(n, step_hours=step_hours)
    range_km = np.full(n, 5000.0)
    speed_kms = np.full(n, 0.2)
    # two separate close+slow episodes, far apart, each individually short
    range_km[5:9], speed_kms[5:9] = 1.0, 0.001
    range_km[40:44], speed_kms[40:44] = 1.0, 0.001
    episodes, label, total_days = classify_episodes(times, range_km, speed_kms)
    assert len(episodes) == 2
    assert total_days == 2 * (3 * step_hours / 24.0)


def test_assess_rpo_refuses_below_min_aligned_samples():
    assert assess_rpo(1, 2, [], []) is None


def test_summary_shows_launch_lineage_caveat_only_when_flagged():
    from detect.rpo import RpoAssessment
    shared = RpoAssessment(
        actor_norad=1, target_norad=2, label="likely", min_range_km=0.8,
        n_episodes=1, total_close_days=3.2, episodes=(), n_aligned_samples=50,
        coverage_start_iso="2020-01-01T00:00:00Z", coverage_end_iso="2020-01-05T00:00:00Z",
        same_launch_lineage=True,
    )
    assert "CAVEAT" in shared.summary()

    unrelated = RpoAssessment(
        actor_norad=1, target_norad=2, label="likely", min_range_km=0.8,
        n_episodes=1, total_close_days=3.2, episodes=(), n_aligned_samples=50,
        coverage_start_iso="2020-01-01T00:00:00Z", coverage_end_iso="2020-01-05T00:00:00Z",
        same_launch_lineage=False,
    )
    assert "CAVEAT" not in unrelated.summary()

    none_label = RpoAssessment(
        actor_norad=1, target_norad=2, label="none", min_range_km=5000.0,
        n_episodes=0, total_close_days=0.0, episodes=(), n_aligned_samples=50,
        coverage_start_iso="2020-01-01T00:00:00Z", coverage_end_iso="2020-01-05T00:00:00Z",
        same_launch_lineage=True,
    )
    assert "CAVEAT" not in none_label.summary()


def test_thresholds_are_the_documented_real_calibration():
    # Locks in the values grounded against real SJ-21/BeiDou-2 G2 data
    # (see module docstring) so a future edit can't silently drift them.
    assert CLOSE_RANGE_KM == 100.0
    assert LOITER_SPEED_KMS == 0.05
    assert MIN_SUSTAINED_DAYS_LIKELY == 3.0
    assert MIN_ALIGNED_SAMPLES == 20
