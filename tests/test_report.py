"""Tests for F4.1 facts payload + F4.2 verification harness
(report/facts.py, report/verify.py). Offline, no LLM dependency — verify_report
is pure text-in/facts-in logic, independent of what produced the text. This
is deliberate: the anti-hallucination check must work the same whether the
sentence came from MLX, a different model, or (in these tests) was typed
by hand to simulate an injected hallucination.
"""
from __future__ import annotations

import pytest

from report.facts import Fact, build_report_facts
from report.verify import verify_report


def _facts():
    return build_report_facts(
        25544, "ISS (ZARYA)",
        [
            Fact("threat_score", 31, "/100", "Composite threat score"),
            Fact("maneuver_0_dv", 13.26, " m/s", "Maneuver delta-v"),
            Fact("rpo_min_range", 0.035, " km", "RPO minimum range"),
            Fact("rpo_coverage_start", "2021-12-01T00:00:00Z", "", "Coverage window start"),
        ],
    )


def test_build_report_facts_refuses_empty():
    with pytest.raises(ValueError):
        build_report_facts(1, "X")


def test_numeric_values_holds_measured_fact_quantities():
    facts = _facts()
    values = facts.numeric_values()
    assert 31.0 in values
    assert 13.26 in values
    assert 0.035 in values
    assert 25544.0 not in values  # identifier, lives in exact_values() instead


def test_exact_values_holds_norad_id_and_date_years():
    facts = _facts()
    values = facts.exact_values()
    assert 25544.0 in values
    assert 2021.0 in values


def test_grounded_sentence_passes():
    facts = _facts()
    text = "NORAD 25544 (ISS) shows a composite threat score of 31/100."
    result = verify_report(text, facts)
    assert result.all_grounded
    assert result.n_ungrounded == 0


def test_sentence_with_no_numbers_passes():
    facts = _facts()
    text = "This object shows routine, well-characterized behavior."
    result = verify_report(text, facts)
    assert result.all_grounded


def test_injected_hallucination_is_caught():
    # The core F4.2 acceptance test per handoff §6: a fabricated number
    # not present in the facts must be flagged.
    facts = _facts()
    text = "The object executed a 47.2 m/s maneuver, an unprecedented event."
    result = verify_report(text, facts)
    assert not result.all_grounded
    assert result.n_ungrounded == 1
    assert 47.2 in result.verdicts[0].unbacked_numbers


def test_mixed_report_flags_only_the_ungrounded_sentence():
    facts = _facts()
    text = (
        "NORAD 25544 shows a threat score of 31/100. "
        "It also completed a secret 99.9 m/s burn last week. "
        "The RPO minimum range recorded was 0.035 km."
    )
    result = verify_report(text, facts)
    assert result.n_sentences == 3
    assert result.n_ungrounded == 1
    assert not result.verdicts[1].grounded
    assert result.verdicts[0].grounded and result.verdicts[2].grounded


def test_hyphenated_object_name_suffix_is_not_read_as_a_negative_number():
    # Real bug, found running verify_report against actual MLX-generated
    # text: "SJ-21" was parsed as the number -21 and flagged as an
    # unbacked hallucination.
    facts = _facts()
    text = "The analyst evaluated SHIJIAN-21 (SJ-21) and found routine behavior."
    result = verify_report(text, facts)
    assert result.all_grounded


def test_clock_time_is_not_split_into_separate_numbers():
    # Real bug, found running verify_report against actual MLX-generated
    # text: "09:06:30 UTC" got split by the colons into three unrelated
    # numbers (9, 6, 30), none individually grounded, even though the
    # whole timestamp matches a coverage-start fact ("2021-12-01T00:06:30Z").
    facts = _facts()
    text = "The close encounter began in 2021 at around 09:06:30 UTC."
    result = verify_report(text, facts)
    assert result.all_grounded


def test_reformatted_number_within_tolerance_still_grounds():
    # "13.3" vs the fact's 13.26 -- LLM rounding, not hallucination.
    facts = _facts()
    text = "The object performed a maneuver of about 13.3 m/s."
    result = verify_report(text, facts)
    assert result.all_grounded


def test_object_name_number_is_not_flagged():
    # Real bug: "Vanguard 1" (NORAD 5) — the "1" in the object NAME was
    # flagged as an unbacked number. Name numbers are identifiers.
    from report.facts import build_report_facts, Fact
    facts = build_report_facts(5, "VANGUARD 1", [Fact("threat_score", 31, "/100", "score")])
    result = verify_report("VANGUARD 1 has a composite threat score of 31/100.", facts)
    assert result.all_grounded


def test_coverage_fraction_number_is_not_flagged():
    # Real bug: a "3/5" coverage fact, restated as "3 out of 5", flagged
    # the "3" because numbers inside a string fact value weren't grounded.
    from report.facts import build_report_facts, Fact
    facts = build_report_facts(1, "X", [Fact("cov", "3/5", "", "coverage")])
    result = verify_report("It was assessed in 3 out of 5 categories.", facts)
    assert result.all_grounded


def test_threat_band_mapping():
    from report.facts import threat_band
    assert threat_band(10) == "low"
    assert threat_band(33) == "low"
    assert threat_band(50) == "moderate"
    assert threat_band(70) == "high"


def test_facts_from_maneuvers_builder():
    from dataclasses import dataclass
    from report.facts import facts_from_maneuvers

    @dataclass
    class FakeEvent:
        epoch_iso: str
        dv_ms: float
        direction: str
        confidence: str
        zscore: float

    events = [FakeEvent("2023-07-05T02:37:56Z", 10.29, "retrograde", "high", 41.5)]
    facts = facts_from_maneuvers(events)
    values = {f.value for f in facts}
    assert 10.29 in values
    assert "retrograde" in values
