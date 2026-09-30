"""Tests for the watch agent's change detection (watch/agent.py).

Offline, synthetic state dicts fed to `diff_states` (the pure core). The
full cycle orchestration and the grounded brief were validated on real
stored data — see the watch-agent run against the SJ-21/Luch watchlist.
Watchlist CRUD is tested against a temp DB.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from watch.agent import diff_states


def _state(**kw):
    base = dict(norad_id=1, threat_score=20.0, threat_band="low",
               last_maneuver_epoch=None, n_maneuvers=0, rpo_label=None,
               rpo_min_range_km=None, decay_reentry_iso=None, disposal_status=None)
    base.update(kw)
    return base


def test_first_sight_is_baseline_no_findings():
    # prev is None -> the object's first cycle raises nothing.
    assert diff_states(None, _state(), "X") == []


def test_no_change_no_findings():
    s = _state(threat_band="moderate", n_maneuvers=3, last_maneuver_epoch="2024-01-01T00:00:00Z")
    assert diff_states(s, dict(s), "X") == []


def test_threat_band_rise_to_high_is_priority():
    prev = _state(threat_band="moderate", threat_score=50)
    cur = _state(threat_band="high", threat_score=70)
    f = diff_states(prev, cur, "X")
    assert len(f) == 1
    assert f[0].kind == "threat_up" and f[0].significance == "priority"


def test_threat_band_fall_is_routine():
    prev = _state(threat_band="high", threat_score=70)
    cur = _state(threat_band="low", threat_score=20)
    f = diff_states(prev, cur, "X")
    assert f[0].kind == "threat_down" and f[0].significance == "routine"


def test_new_maneuver_is_elevated():
    prev = _state(n_maneuvers=3, last_maneuver_epoch="2024-01-01T00:00:00Z")
    cur = _state(n_maneuvers=4, last_maneuver_epoch="2024-02-01T00:00:00Z")
    f = diff_states(prev, cur, "X")
    assert any(x.kind == "new_maneuver" and x.significance == "elevated" for x in f)


def test_rpo_appearing_likely_is_priority():
    prev = _state(rpo_label=None)
    cur = _state(rpo_label="likely", rpo_min_range_km=0.04)
    f = diff_states(prev, cur, "X")
    assert any(x.kind == "new_rpo" and x.significance == "priority" for x in f)
    # sub-km range renders in metres (0.04 km = 40 m)
    assert "40 m" in next(x for x in f if x.kind == "new_rpo").summary


def test_rpo_materially_closing_is_priority():
    prev = _state(rpo_label="possible", rpo_min_range_km=100.0)
    cur = _state(rpo_label="likely", rpo_min_range_km=2.0)
    f = diff_states(prev, cur, "X")
    assert any(x.kind == "rpo_closer" and x.significance == "priority" for x in f)


def test_disposal_change_to_noncompliant_is_elevated():
    prev = _state(disposal_status="OPERATIONAL_OR_ABANDONED_IN_PLACE")
    cur = _state(disposal_status="NON_COMPLIANT_SHALLOW")
    f = diff_states(prev, cur, "X")
    assert any(x.kind == "disposal_change" and x.significance == "elevated" for x in f)


def test_watchlist_crud_roundtrip():
    from data.store import add_to_watchlist, get_watchlist, is_watched, remove_from_watchlist
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.sqlite3"
        add_to_watchlist(49330, "SJ-21", db_path=db)
        add_to_watchlist(40258, "Luch", db_path=db)
        assert {w["norad_id"] for w in get_watchlist(db_path=db)} == {49330, 40258}
        assert is_watched(49330, db_path=db)
        remove_from_watchlist(49330, db_path=db)
        assert not is_watched(49330, db_path=db)
        assert {w["norad_id"] for w in get_watchlist(db_path=db)} == {40258}
