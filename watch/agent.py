"""Autonomous "analyst on watch" agent (roadmap feature 1).

A repeatable watch CYCLE that answers "what changed since I last looked?"
across the analyst's watchlist and the catalogue population, and emits a
ranked findings board + a grounded, verified brief. Manual or cron-driven;
each run appends to a feed, giving the "agent watching over time" feel with
no always-on daemon.

Architecture = deterministic cycle + grounded narrative (design Approach 1):
detection is 100% deterministic and the local LLM only writes the brief
prose, which the F4.2 verifier checks. No LLM tool-calling — a 3B/4-bit
local model can't plan reliably, and "the agent hallucinated its
investigation" is the one headline this project can't afford.

The core idea is DIFFING: a finding is raised only when this cycle's
per-object state differs from the stored previous state (a new maneuver
epoch, a threat score crossing a band, an RPO that appeared or closed, a
disposal status change, a newly-predicted re-entry). The first cycle for an
object is a baseline — no findings — so run one never floods false "new"
alerts. Every finding traces to a computed detector output (§9).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from data.store import (
    get_last_object_states,
    get_snapshot_ids,
    get_watchlist,
    list_active_satcat_norad_ids,
    list_snapshot_timestamps,
    record_catalog_snapshot,
    save_watch_cycle,
)

_BAND_RANK = {"low": 0, "moderate": 1, "high": 2}
RPO_CLOSER_FRACTION = 0.5   # a new min range below this × the old = "materially closer"
DECAY_IMMINENT_DAYS = 30.0


@dataclass(frozen=True)
class Finding:
    norad_id: Optional[int]
    kind: str
    significance: str          # priority | elevated | routine
    summary: str


@dataclass(frozen=True)
class WatchCycleResult:
    cycle_id: int
    run_at: str
    refreshed: bool
    findings: tuple[Finding, ...]
    brief: str
    n_watched: int


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def object_state(norad_id: int) -> dict:
    """Extract the diffable signals for one object from its full
    characterization. Returns a dict whose keys match the
    watch_object_state columns (+ a transient `_name` for summaries)."""
    from report.facts import threat_band
    from web.characterize import characterize

    d = characterize(norad_id)
    t = d["threat"]
    maneuvers = d["maneuvers"]
    rpo = d["rpo"]
    decay = d["decay"]
    disposal = d["disposal"]
    band = threat_band(t.score) if (t is not None and t.score is not None) else None
    return {
        "norad_id": norad_id,
        "_name": d["name"],
        "threat_score": t.score if t is not None else None,
        "threat_band": band,
        "last_maneuver_epoch": maneuvers[-1].epoch_iso if maneuvers else None,
        "n_maneuvers": len(maneuvers),
        "rpo_label": rpo.label if rpo is not None else None,
        "rpo_min_range_km": rpo.min_range_km if rpo is not None else None,
        "decay_reentry_iso": decay.predicted_reentry_iso if decay is not None else None,
        "disposal_status": disposal.status if disposal is not None else None,
    }


def _fmt_range(km) -> str:
    if km is None:
        return "n/a"
    return f"{km * 1000:.0f} m" if km < 1.0 else f"{km:.1f} km"


def diff_states(prev: Optional[dict], cur: dict, name: str) -> list[Finding]:
    """Compare an object's previous stored state to its current state and
    raise a finding for each meaningful change. `prev is None` (first sight)
    yields nothing — a baseline, not a flood of 'new' alerts."""
    if prev is None:
        return []
    nid = cur["norad_id"]
    out: list[Finding] = []

    # --- threat band change -------------------------------------------
    pb, cb = prev.get("threat_band"), cur.get("threat_band")
    if pb and cb and pb != cb:
        rose = _BAND_RANK[cb] > _BAND_RANK[pb]
        sig = "priority" if (rose and cb == "high") else ("elevated" if rose else "routine")
        verb = "rose" if rose else "fell"
        score = cur.get("threat_score")
        out.append(Finding(nid, "threat_up" if rose else "threat_down", sig,
                            f"{name}: threat {verb} from {pb} to {cb}"
                            + (f" (score {score:.0f}/100)" if score is not None else "")))

    # --- new maneuver -------------------------------------------------
    if (cur.get("last_maneuver_epoch") and prev.get("last_maneuver_epoch")
            and cur["last_maneuver_epoch"] > prev["last_maneuver_epoch"]
            and (cur.get("n_maneuvers") or 0) > (prev.get("n_maneuvers") or 0)):
        n_new = cur["n_maneuvers"] - prev["n_maneuvers"]
        out.append(Finding(nid, "new_maneuver", "elevated",
                            f"{name}: {n_new} new maneuver detection(s), latest "
                            f"{cur['last_maneuver_epoch'][:10]}"))

    # --- RPO appeared or materially closed ----------------------------
    cur_rpo, prev_rpo = cur.get("rpo_label"), prev.get("rpo_label")
    if cur_rpo in ("likely", "possible"):
        if prev_rpo not in ("likely", "possible"):
            sig = "priority" if cur_rpo == "likely" else "elevated"
            out.append(Finding(nid, "new_rpo", sig,
                                f"{name}: began proximity operations ({cur_rpo}, min "
                                f"{_fmt_range(cur.get('rpo_min_range_km'))})"))
        else:
            pr, cr = prev.get("rpo_min_range_km"), cur.get("rpo_min_range_km")
            if pr is not None and cr is not None and cr < pr * RPO_CLOSER_FRACTION:
                out.append(Finding(nid, "rpo_closer", "priority",
                                    f"{name}: proximity closed from {_fmt_range(pr)} to "
                                    f"{_fmt_range(cr)}"))

    # --- re-entry newly predicted -------------------------------------
    if cur.get("decay_reentry_iso") and not prev.get("decay_reentry_iso"):
        days = None
        try:
            r = datetime.fromisoformat(cur["decay_reentry_iso"].replace("Z", "+00:00"))
            days = (r - _utcnow()).total_seconds() / 86400.0
        except Exception:
            pass
        sig = "priority" if (days is not None and days < DECAY_IMMINENT_DAYS) else "elevated"
        tail = f" in ~{days:.0f} days" if days is not None else ""
        out.append(Finding(nid, "new_decay", sig,
                            f"{name}: re-entry now predicted{tail} "
                            f"(~{cur['decay_reentry_iso'][:10]})"))

    # --- disposal status changed --------------------------------------
    pd, cd = prev.get("disposal_status"), cur.get("disposal_status")
    if pd and cd and pd != cd:
        sig = "elevated" if cd in ("NON_COMPLIANT_SHALLOW", "BELOW_GEO_DRIFT") else "routine"
        out.append(Finding(nid, "disposal_change", sig,
                            f"{name}: disposal status changed {pd} → {cd}"))
    return out


def _population_findings() -> list[Finding]:
    """Catalogue-level events since the previous snapshot — new debris
    bursts (candidate fragmentations) via the existing snapshot delta."""
    from detect.breakup import detect_from_new_ids
    stamps = list_snapshot_timestamps()
    if len(stamps) < 2:
        return []
    new_ids = get_snapshot_ids(stamps[-1]) - get_snapshot_ids(stamps[-2])
    out = []
    for c in detect_from_new_ids(new_ids):
        out.append(Finding(c.parent_norad_id, "fragmentation", "priority",
                           f"Candidate fragmentation: {c.n_fragments} new debris on launch "
                           f"{c.launch_prefix} (confidence {c.confidence})"))
    return out


def run_cycle(refresh: bool = False, generate_brief: bool = True) -> WatchCycleResult:
    """Run one watch cycle: (optionally refresh) → state each watched object
    → diff vs. last cycle → population events → grounded brief → persist."""
    run_at = _utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    watchlist = [w["norad_id"] for w in get_watchlist()]

    if refresh:
        from data.celestrak import ingest_object
        for nid in watchlist:
            try:
                ingest_object(nid)
            except Exception:
                pass
        active = list_active_satcat_norad_ids()
        if active:
            record_catalog_snapshot(active)

    prev_states = get_last_object_states()
    findings: list[Finding] = []
    new_states: list[dict] = []
    for nid in watchlist:
        try:
            cur = object_state(nid)
        except Exception:
            continue
        name = cur.pop("_name", str(nid))
        findings.extend(diff_states(prev_states.get(nid), cur, name))
        new_states.append(cur)

    findings.extend(_population_findings())
    findings.sort(key=lambda f: {"priority": 3, "elevated": 2, "routine": 1}[f.significance],
                  reverse=True)
    finding_dicts = [{"norad_id": f.norad_id, "kind": f.kind,
                      "significance": f.significance, "summary": f.summary} for f in findings]

    brief = ""
    if generate_brief and finding_dicts:
        from report.generate import generate_watch_brief
        brief = generate_watch_brief(finding_dicts, run_at)

    cycle_id = save_watch_cycle(run_at, refresh, brief, finding_dicts, new_states)
    return WatchCycleResult(cycle_id, run_at, refresh, tuple(findings), brief, len(watchlist))
