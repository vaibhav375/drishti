# DRISHTI Watch Agent — design spec

**Status:** approved 2026-07-16. First of a roadmap of "AI/ML/agent/online"
features. Build order: watch agent → NL query → change-point/clustering →
alerting/watchlist → external enrichment → standardized products.

## Anchor stance (agreed)
Keep both project anchors. Offline core stays the default; online is an
OPTIONAL layer that degrades gracefully. Firmly decision-support, never
targeting. Deepen AI/ML/agent depth *within* those rails.

## Feature 1 — Autonomous "analyst on watch" agent

**Purpose.** A repeatable watch *cycle* that answers "what changed since I
last looked?" across a watchlist of objects and the catalog population,
and emits a grounded, verified intelligence brief + a ranked findings
board. Manual or cron-driven; each run appends to an accumulating feed
(the "agent watching over time" without an always-on daemon).

**Architecture = Approach 1 (deterministic cycle + grounded narrative).**
Detection is 100% deterministic; the local LLM only writes the brief prose
and is checked by the F4.2 verifier. No LLM tool-calling (unreliable on a
3B/4-bit local model, and "the agent hallucinated its investigation" is the
one headline this project can't afford).

### Components
- `watch/agent.py` — `object_state(norad_id)`, `diff_states(prev, cur)`,
  `significance(kind)`, `run_cycle(refresh)`. Reuses `web.characterize`.
- `data/store.py` tables (+ migration): `watch_list`, `watch_cycle`,
  `watch_finding`, `watch_object_state` (per-object diff snapshot).
- Brief via `report/` (facts → generate → verify).

### Change detection (core idea)
A finding is raised only when this cycle DIFFERS from the stored previous
state: a new maneuver epoch, a threat score crossing a band, an RPO that
appeared or closed, a disposal status change, a newly-predicted re-entry,
a newly-flagged anomaly. First cycle for an object = baseline, no findings.

### Significance (deterministic; priority/elevated/routine)
new "likely" RPO → priority; threat band low→high → priority; new
fragmentation → priority; decay <30d newly predicted → priority; new
maneuver → elevated; disposal became non-compliant → elevated; anomaly
newly flagged → elevated; drift within a band → routine.

### Interfaces
- CLI: `python cli.py watch [--refresh] [--add N] [--remove N] [--list]`.
- Web: `/watch` — I&W-style findings board (ranked), the grounded brief,
  the watchlist, the feed of past cycles. "+ watch" button on dossiers,
  "Run cycle" button.

### Offline / online
Offline default (diff current detector output vs. last stored cycle).
`--refresh` optionally pulls fresh Celestrak TLEs + a snapshot first;
degrades gracefully offline.

### Honesty & testing
Every finding traces to a computed detector output; brief runs through the
verifier; significance deterministic + explainable; stale/decayed guards
reused; first-cycle baseline avoids false "new" alerts. Tests: synthetic
state-A-vs-B diffs, significance, watchlist CRUD (offline); cycle
orchestration validated on real stored data.

### Out of scope (first cut)
No daemon, no LLM tool-calling, no push/email, no external enrichment, no
multi-user — later features.
```
