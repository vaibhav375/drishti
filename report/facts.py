"""Structured facts payload (F4.1 input, F4.2 ground truth).

Per handoff §9: "Never fabricate orbital numbers. Every quantity in a
report or the UI must trace to a computed value." This module is where
that traceability starts — every `Fact` wraps a number that a detect/ or
score/ module actually computed, nothing invented for the report layer.
`report/verify.py` checks generated text against exactly this payload,
so a sentence is "grounded" if and only if its numbers appear here.

One builder function per detector output type (F1.1 maneuvers, F1.4
threat score, F2.1 RPO, F3.1 Pc) — each returns a list[Fact] that
`build_report_facts` combines. Deliberately NOT one big function: a
report can combine any subset (e.g. threat score + RPO status for one
object), and each detect/ module stays independently droppable per §2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

FactValue = Union[float, int, str]


@dataclass(frozen=True)
class Fact:
    key: str
    value: FactValue
    unit: str
    label: str


@dataclass(frozen=True)
class ReportFacts:
    norad_id: int
    object_name: Optional[str]
    facts: tuple[Fact, ...]

    def as_prompt_block(self) -> str:
        return "\n".join(f"- {f.label}: {f.value}{f.unit}" for f in self.facts)

    def numeric_values(self) -> set[float]:
        """Measured/computed fact quantities — matched with a rounding
        TOLERANCE in report/verify.py (an LLM restating 13.26 as "13.3"
        is reformatting, not hallucination)."""
        return {float(f.value) for f in self.facts if isinstance(f.value, (int, float))}

    def exact_values(self) -> set[float]:
        """Identifiers and fixed-scale markers — the object's own
        norad_id, year/month/day embedded in ISO-date facts (found live:
        a real generated report wrote "February 14, 2022" from a
        coverage_end_iso fact of "2022-02-14T...", and the day number 14
        wasn't extracted, only the year — a real, dominant false-positive
        class once dates appear in prose, not a hypothetical), and
        numbers embedded in a UNIT string (e.g. the "100" in a "/100"
        score scale). These must match EXACTLY, not within a tolerance: a
        loose tolerance around a round marker like 100 would let a
        coincidentally-nearby fabricated number (e.g. a hallucinated
        "99.9") slip through as if grounded — found live while testing
        this exact case."""
        values = {float(self.norad_id)}
        # numbers that are part of the object's own NAME are identifiers,
        # not claims — "Vanguard 1", "OPS 7044", "MMS 1" (found live: the
        # "1" in "Vanguard 1" was wrongly flagged as an unbacked number).
        if self.object_name:
            for m in re.finditer(r"-?\d+\.?\d*", self.object_name):
                values.add(float(m.group(0)))
        for f in self.facts:
            # every number embedded in a STRING fact value is a literal from
            # a computed fact and is exactly restatable — dates ("2021-12-01"),
            # coverage ("3/5"), designators ("2019-079A"), etc. (found live:
            # "3 out of 5" from a "3/5" coverage fact was wrongly flagged).
            if isinstance(f.value, str):
                for m in re.finditer(r"\d+\.?\d*", f.value):
                    values.add(float(m.group(0)))
            for m in re.finditer(r"-?\d+\.?\d*", f.unit):
                values.add(float(m.group(0)))
        return values


def build_report_facts(norad_id: int, object_name: Optional[str], *fact_lists: list[Fact]) -> ReportFacts:
    combined = tuple(f for lst in fact_lists for f in lst)
    if not combined:
        raise ValueError("no facts provided — refuse to build a report with nothing to ground it")
    return ReportFacts(norad_id=norad_id, object_name=object_name, facts=combined)


def facts_from_maneuvers(events: list) -> list[Fact]:
    """From detect/maneuver.py::ManeuverEvent list."""
    facts = []
    for i, e in enumerate(events):
        facts.append(Fact(f"maneuver_{i}_dv", round(e.dv_ms, 2), " m/s",
                           f"Maneuver on {e.epoch_iso}: estimated delta-v"))
        facts.append(Fact(f"maneuver_{i}_direction", e.direction, "",
                           f"Maneuver on {e.epoch_iso}: direction"))
        facts.append(Fact(f"maneuver_{i}_confidence", e.confidence, "",
                           f"Maneuver on {e.epoch_iso}: confidence"))
        facts.append(Fact(f"maneuver_{i}_zscore", round(e.zscore, 1), "",
                           f"Maneuver on {e.epoch_iso}: z-score vs. noise floor"))
    return facts


def threat_band(score_0_100: float) -> str:
    """Map a composite threat score to a plain qualitative band. Giving the
    report a GROUNDED word for the level stops the model inventing its own
    (it once called a 10/100 score a 'significant threat') — it may only use
    this descriptor."""
    if score_0_100 >= 67:
        return "high"
    if score_0_100 >= 34:
        return "moderate"
    return "low"


def facts_from_threat_score(score) -> list[Fact]:
    """From score/threat.py::ThreatScore."""
    facts = []
    if score.score is not None:
        facts.append(Fact("threat_score", round(score.score), "/100", "Composite threat score"))
        facts.append(Fact("threat_level", threat_band(score.score), "",
                           "Threat level (the ONLY qualitative descriptor to use)"))
    facts.append(Fact("threat_factors_available", f"{score.n_factors_available}/{score.n_factors_total}",
                       "", "Threat score factor coverage"))
    for f in score.factors:
        if f.value is not None:
            facts.append(Fact(f"threat_factor_{f.name}", round(f.value, 2), "",
                               f"Threat factor: {f.name}"))
    return facts


def facts_from_rpo(assessment) -> list[Fact]:
    """From detect/rpo.py::RpoAssessment."""
    facts = [
        Fact("rpo_label", assessment.label, "", "RPO assessment"),
        Fact("rpo_min_range", round(assessment.min_range_km, 3), " km", "RPO minimum range"),
        Fact("rpo_episodes", assessment.n_episodes, "", "RPO close episodes"),
        Fact("rpo_total_close_days", round(assessment.total_close_days, 1), " days",
             "RPO total time spent close"),
        Fact("rpo_coverage_start", assessment.coverage_start_iso, "", "RPO coverage window start"),
        Fact("rpo_coverage_end", assessment.coverage_end_iso, "", "RPO coverage window end"),
    ]
    if assessment.same_launch_lineage:
        facts.append(Fact("rpo_same_launch", "true", "",
                           "Actor and target share a launch designator (possible deployment, not RPO)"))
    return facts


def facts_from_pc(assessment) -> list[Fact]:
    """From detect/conjunction.py::ConjunctionAssessment."""
    return [
        Fact("pc_estimate", f"{assessment.pc_estimate:.2e}", "", "Estimated probability of collision"),
        Fact("pc_tca", assessment.tca_iso, "", "Time of closest approach"),
        Fact("pc_min_range", round(assessment.min_range_km, 3), " km",
             "Deterministic minimum range at TCA"),
        Fact("pc_hard_body_radius", round(assessment.hard_body_radius_km * 1000, 1), " m",
             "Combined hard-body radius used"),
    ]


def facts_from_breakup(candidate) -> list[Fact]:
    """From detect/breakup.py::BreakupCandidate."""
    return [
        Fact("breakup_n_fragments", candidate.n_fragments, "", "Debris fragment count"),
        Fact("breakup_confidence", candidate.confidence, "", "Fragmentation confidence"),
        Fact("breakup_mean_inclination", round(candidate.mean_inclination_deg, 1), " deg",
             "Debris cloud mean inclination"),
    ]


def facts_from_regime(classification) -> list[Fact]:
    """From core/regime.py::RegimeClassification (bonus)."""
    facts = [
        Fact("regime_primary", classification.primary, "", "Orbital regime"),
        Fact("regime_period", round(classification.period_min), " min", "Orbital period"),
    ]
    for tag in classification.special_tags:
        facts.append(Fact(f"regime_tag_{tag.lower()}", tag, "", "Special-orbit tag"))
    return facts


def facts_from_decay(prediction) -> list[Fact]:
    """From detect/decay.py::DecayPrediction (bonus)."""
    facts = [
        Fact("decay_perigee", round(prediction.latest_perigee_km), " km", "Latest perigee altitude"),
        Fact("decay_rate", round(prediction.decay_rate_km_per_day, 2), " km/day", "Perigee decay rate"),
    ]
    if prediction.predicted_reentry_iso is not None and prediction.days_to_reentry:
        facts.append(Fact("decay_reentry_date", prediction.predicted_reentry_iso, "",
                           "Predicted re-entry date"))
        facts.append(Fact("decay_days_to_reentry", round(prediction.days_to_reentry), " days",
                           "Predicted days to re-entry"))
    return facts


def facts_from_disposal(assessment) -> list[Fact]:
    """From detect/graveyard.py::DisposalAssessment (bonus)."""
    return [
        Fact("disposal_status", assessment.status, "", "GEO disposal-compliance status"),
        Fact("disposal_km_above_geo", round(assessment.km_above_geo), " km",
             "Perigee altitude relative to GEO"),
    ]


def watch_facts(finding_dicts: list[dict], run_at: str):
    """Build the grounded facts payload for a watch-cycle brief. Each
    finding's summary is already a computed, human-readable string, so its
    numbers ground automatically (string-value extraction in exact_values).
    `finding_dicts`: {'summary','significance', ...}."""
    n = len(finding_dicts)
    counts = {"priority": 0, "elevated": 0, "routine": 0}
    for f in finding_dicts:
        counts[f["significance"]] = counts.get(f["significance"], 0) + 1
    facts = [
        Fact("cycle_run_at", run_at, "", "Watch cycle run time"),
        Fact("n_findings", n, "", "Findings this cycle"),
        Fact("n_priority", counts["priority"], "", "Priority findings"),
        Fact("n_elevated", counts["elevated"], "", "Elevated findings"),
    ]
    for i, f in enumerate(finding_dicts):
        facts.append(Fact(f"finding_{i}", f["summary"], "",
                           f"Finding ({f['significance']})"))
    return build_report_facts(0, "watch cycle", facts)
