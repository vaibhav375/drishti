"""Temporal proximity-interaction graph (F3.2).

Per handoff §5: objects as nodes; edges form on close low-relative-
velocity encounters (F2.1's RPO episodes), evolving over time. Graph
anomaly detection surfaces SERIAL INSPECTORS — objects that repeatedly
form close edges with many DIFFERENT high-value assets, not just one
(the real Luch/Olymp-K GEO-drift case in validate/events.py is exactly
this pattern: "Long-baseline serial-inspector case — the temporal graph
(F3.2) target").

This module builds and analyzes the graph — it does NOT run RPO
detection itself. Feed it `RpoAssessment`s already computed elsewhere
(detect/rpo.py::assess_rpo, one call per candidate pair — typically
cascade-screened pairs, detect/cascade.py, that also have archived TLE
history). Only 'possible'/'likely' assessments become edges; 'none'
contributes nothing (no encounter, no edge).

Deliberately NOT the "Temporal Graph Transformer" mentioned in the
handoff's background — that's an explicit BONUS/research-grade
extension (§10) on top of this baseline, not required for Tier 3. This
is a plain networkx MultiDiGraph with a simple degree-based anomaly
rule: honest, explainable, and enough to surface a real repeat-inspector
pattern (the acceptance target per §6).

Honesty rule (§9): a node's "serial inspector" status is a count over
whatever assessments were fed in — it says nothing about pairs never
screened. Report what was actually checked, not the full catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import networkx as nx

from detect.rpo import RpoAssessment

MIN_DISTINCT_TARGETS = 2  # >=2 different targets before "serial" is even considered


@dataclass(frozen=True)
class SerialInspectorFinding:
    actor_norad: int
    distinct_targets: int
    distinct_hva_targets: int
    target_norad_ids: tuple[int, ...]
    confidence: str  # 'low' | 'medium' | 'high'

    def summary(self) -> str:
        return (
            f"NORAD {self.actor_norad}: serial-inspector pattern — {self.distinct_targets} "
            f"distinct target(s) ({self.distinct_hva_targets} high-value), "
            f"confidence {self.confidence} — targets: {list(self.target_norad_ids)}"
        )


def build_interaction_graph(
    assessments: list[RpoAssessment], hva_ids: Optional[set[int]] = None
) -> nx.MultiDiGraph:
    """One edge per non-'none' RpoAssessment, actor -> target. Multiple
    episodes between the same pair (from re-running assess_rpo over
    different windows) become parallel edges — a MultiDiGraph, not a
    simple graph, so repeated encounters over time aren't collapsed away.
    """
    hva_ids = hva_ids or set()
    g = nx.MultiDiGraph()
    for a in assessments:
        if a.label == "none":
            continue
        g.add_node(a.actor_norad, is_hva=a.actor_norad in hva_ids)
        g.add_node(a.target_norad, is_hva=a.target_norad in hva_ids)
        g.add_edge(
            a.actor_norad, a.target_norad,
            label=a.label,
            min_range_km=a.min_range_km,
            total_close_days=a.total_close_days,
            start_iso=a.coverage_start_iso,
            end_iso=a.coverage_end_iso,
            same_launch_lineage=a.same_launch_lineage,
        )
    return g


def find_serial_inspectors(
    g: nx.MultiDiGraph, min_distinct_targets: int = MIN_DISTINCT_TARGETS
) -> list[SerialInspectorFinding]:
    """Nodes whose OUT-edges reach multiple DISTINCT targets — repeat
    encounters with the SAME single target don't count (that's sustained
    RPO against one asset, already what F2.1 reports; "serial" means
    spread across several different objects, the graph-level signal
    F2.1 alone can't see).
    """
    findings: list[SerialInspectorFinding] = []
    for node in g.nodes:
        targets = {v for _, v, _ in g.out_edges(node, keys=True)}
        targets.discard(node)  # a self-loop (data artifact) isn't a second target
        if len(targets) < min_distinct_targets:
            continue
        n_hva = sum(1 for t in targets if g.nodes[t].get("is_hva"))
        if n_hva >= 3 or len(targets) >= 5:
            confidence = "high"
        elif n_hva >= 1 or len(targets) >= 3:
            confidence = "medium"
        else:
            confidence = "low"
        findings.append(SerialInspectorFinding(
            actor_norad=node,
            distinct_targets=len(targets),
            distinct_hva_targets=n_hva,
            target_norad_ids=tuple(sorted(targets)),
            confidence=confidence,
        ))
    findings.sort(key=lambda f: (f.distinct_hva_targets, f.distinct_targets), reverse=True)
    return findings


def edges_in_window(g: nx.MultiDiGraph, start: datetime, end: datetime) -> list[tuple[int, int, dict]]:
    """Edges whose coverage window overlaps [start, end) — the "evolving
    over time" slice a caller (e.g. a future dashboard timeline) needs,
    without requiring the whole graph to be rebuilt per window.
    """
    out = []
    for u, v, data in g.edges(data=True):
        e_start = datetime.fromisoformat(data["start_iso"].replace("Z", "+00:00"))
        e_end = datetime.fromisoformat(data["end_iso"].replace("Z", "+00:00"))
        if e_start <= end and e_end >= start:
            out.append((u, v, data))
    return out
