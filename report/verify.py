"""Anti-hallucination verification harness (F4.2).

Per handoff §5/§9: reject any generated sentence carrying a numeric claim
that isn't backed by a fact in the payload that produced it. This is the
core discipline of the report layer — "a research contribution on its
own", not a formality — and it is fully independent of which LLM (or
even whether an LLM) produced the text, so it's testable offline with
zero model dependency.

Method: split generated text into sentences, extract every number in
each sentence, and check each against two pools from `ReportFacts`:
`numeric_values()` (measured quantities, matched with a TOLERANCE — LLMs
reformat numbers, "3.0" vs "3", or round a Δv to one fewer decimal
place, that's not hallucination) and `exact_values()` (identifiers and
fixed-scale markers — norad_id, dates, unit denominators like the "100"
in "/100" — matched EXACTLY, because a loose tolerance around a round
marker like 100 would let a coincidentally-nearby fabricated number
slip through as if grounded; found live while building this). A
sentence with ANY unbacked number is flagged; sentences with no numbers
pass (connective/framing text makes no quantitative claim to ground).

Honesty rule, applied to the harness itself: `verify_report` never
"fixes" a report — it only reports which sentences are grounded and
which aren't. Deciding what to do with an ungrounded sentence (redact,
regenerate, reject the whole report) is the caller's call.

Two false-positive classes found running this against REAL generated
text (not just hand-written examples), fixed by masking before number
extraction: (1) hyphenated identifier suffixes — "SJ-21" parsed as the
literal number -21; (2) clock-time substrings — "09:06:30 UTC" split by
the colons into three unrelated numbers (9, 6, 30), when the whole
timestamp is already grounded as one ISO-date fact. Both are masked out
before `_numbers_in` runs, not treated as numeric claims at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from report.facts import ReportFacts

NUMBER_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z]+-?\d+[A-Za-z]*\b")  # e.g. "SJ-21", "COSMOS2543"
CLOCK_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # e.g. "09:06:30"
RELATIVE_TOLERANCE = 0.02   # 2% — LLM rounding/reformatting slack
ABSOLUTE_TOLERANCE = 0.05   # floor so near-zero facts aren't impossibly strict


@dataclass(frozen=True)
class SentenceVerdict:
    sentence: str
    numbers_found: tuple[float, ...]
    grounded: bool
    unbacked_numbers: tuple[float, ...]


@dataclass(frozen=True)
class VerificationResult:
    verdicts: tuple[SentenceVerdict, ...]
    all_grounded: bool
    n_sentences: int
    n_ungrounded: int

    def summary(self) -> str:
        lines = [f"{self.n_sentences - self.n_ungrounded}/{self.n_sentences} sentences grounded"]
        for v in self.verdicts:
            if not v.grounded:
                lines.append(f"  UNGROUNDED: \"{v.sentence}\" — unbacked numbers: "
                              f"{list(v.unbacked_numbers)}")
        return "\n".join(lines)


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _numbers_in(sentence: str) -> list[float]:
    masked = IDENTIFIER_RE.sub(" ", sentence)
    masked = CLOCK_TIME_RE.sub(" ", masked)
    out = []
    for m in NUMBER_RE.findall(masked):
        try:
            out.append(float(m))
        except ValueError:
            continue
    return out


def _is_backed(value: float, tolerant_values: set[float], exact_values: set[float]) -> bool:
    for kv in exact_values:
        if abs(value - kv) < 1e-9:
            return True
    for kv in tolerant_values:
        tol = max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * abs(kv))
        if abs(value - kv) <= tol:
            return True
    return False


def verify_report(text: str, facts: ReportFacts) -> VerificationResult:
    tolerant_values = facts.numeric_values()
    exact_values = facts.exact_values()
    verdicts = []
    for sentence in _split_sentences(text):
        numbers = _numbers_in(sentence)
        unbacked = tuple(n for n in numbers if not _is_backed(n, tolerant_values, exact_values))
        verdicts.append(SentenceVerdict(
            sentence=sentence,
            numbers_found=tuple(numbers),
            grounded=(len(unbacked) == 0),
            unbacked_numbers=unbacked,
        ))
    n_ungrounded = sum(1 for v in verdicts if not v.grounded)
    return VerificationResult(
        verdicts=tuple(verdicts),
        all_grounded=(n_ungrounded == 0),
        n_sentences=len(verdicts),
        n_ungrounded=n_ungrounded,
    )
