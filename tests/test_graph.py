"""Tests for F3.2 temporal proximity-interaction graph (detect/graph.py).

Offline, synthetic RpoAssessment objects — no propagation/DB dependency,
since this module only consumes RpoAssessment results computed elsewhere.
Real validation (does a real repeat-inspector pattern surface) — see
README "Real-data findings".
"""
from __future__ import annotations

from datetime import datetime, timezone

from detect.graph import build_interaction_graph, edges_in_window, find_serial_inspectors
from detect.rpo import RpoAssessment


def _assessment(actor, target, label, min_range=1.0, start="2020-01-01T00:00:00Z",
                 end="2020-01-05T00:00:00Z", same_launch_lineage=False):
    return RpoAssessment(
        actor_norad=actor, target_norad=target, label=label, min_range_km=min_range,
        n_episodes=1, total_close_days=3.0, episodes=(), n_aligned_samples=50,
        coverage_start_iso=start, coverage_end_iso=end,
        same_launch_lineage=same_launch_lineage,
    )


def test_none_labeled_assessments_produce_no_edges():
    g = build_interaction_graph([_assessment(1, 2, "none")])
    assert g.number_of_edges() == 0
    assert g.number_of_nodes() == 0


def test_possible_and_likely_produce_edges():
    g = build_interaction_graph([
        _assessment(1, 2, "possible"),
        _assessment(1, 3, "likely"),
    ])
    assert g.number_of_edges() == 2
    assert set(g.nodes) == {1, 2, 3}


def test_hva_flag_propagates_to_nodes():
    g = build_interaction_graph([_assessment(1, 2, "likely")], hva_ids={2})
    assert g.nodes[1]["is_hva"] is False
    assert g.nodes[2]["is_hva"] is True


def test_repeated_episodes_with_same_target_do_not_count_as_serial():
    # Same actor, same single target, multiple time windows -- sustained
    # RPO against ONE asset, not the "serial" (many different targets)
    # pattern this module looks for.
    g = build_interaction_graph([
        _assessment(1, 2, "likely", start="2020-01-01T00:00:00Z", end="2020-01-05T00:00:00Z"),
        _assessment(1, 2, "likely", start="2020-03-01T00:00:00Z", end="2020-03-05T00:00:00Z"),
    ])
    findings = find_serial_inspectors(g)
    assert findings == []


def test_multi_target_actor_is_flagged_as_serial_inspector():
    g = build_interaction_graph([
        _assessment(1, 2, "likely"),
        _assessment(1, 3, "possible"),
    ], hva_ids={2, 3})
    findings = find_serial_inspectors(g)
    assert len(findings) == 1
    f = findings[0]
    assert f.actor_norad == 1
    assert f.distinct_targets == 2
    assert f.distinct_hva_targets == 2
    assert f.confidence == "medium"


def test_confidence_scales_with_hva_and_target_count():
    many_targets = [_assessment(1, t, "likely") for t in range(2, 8)]  # 6 distinct targets
    g = build_interaction_graph(many_targets, hva_ids=set(range(2, 8)))
    findings = find_serial_inspectors(g)
    assert findings[0].confidence == "high"


def test_edges_in_window_filters_by_time_overlap():
    g = build_interaction_graph([
        _assessment(1, 2, "likely", start="2020-01-01T00:00:00Z", end="2020-01-05T00:00:00Z"),
        _assessment(1, 3, "likely", start="2020-06-01T00:00:00Z", end="2020-06-05T00:00:00Z"),
    ])
    tz = timezone.utc
    window = edges_in_window(g, datetime(2020, 1, 1, tzinfo=tz), datetime(2020, 2, 1, tzinfo=tz))
    assert len(window) == 1
    assert window[0][1] == 2  # only the January encounter, target 2
