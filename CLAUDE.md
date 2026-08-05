# CLAUDE.md — DRISHTI project memory

> **Maintenance contract:** this file must be kept current by Claude after every
> meaningful change in this repo — new feature, real bug found/fixed, design
> decision, or status change. Update it in the same turn as the work, not
> deferred. A fresh Claude Code session should be able to read only this file
> and know exactly where things stand before touching anything else.

## What this is

DRISHTI — an offline Space Domain Awareness (SDA) system. Ingests public TLEs
(Celestrak, Space-Track, SATCAT), characterizes on-orbit *behavior* (maneuvers,
RPO/inspection, fragmentation), scores threat explainably, and generates
grounded analyst reports via a local LLM. Framing: **decision-support, not
targeting** — this matters for every design choice, especially anything
touching "ownership"/nation attribution.

Full spec: `SDA_DASHBOARD_HANDOFF.md`. Session-1 snapshot (superseded by this
file and README.md, kept for history): `PROGRESS_HANDOFF.md`. Detailed
real-data findings log (every real bug found this session, with root cause):
`README.md` → "Real-data findings" section. **Read README's findings section
before assuming any constant/threshold in the code is a guess** — most were
calibrated against real archive data, and the finding explains why.

## Current status (2026-07-15) — Tiers 0-4 complete + first 3 bonus features done

| Tier | Status |
|---|---|
| Tier 0 (ingestion, propagation, frames, dashboard skeleton) | ✅ done |
| Tier 1 (F1.1 maneuvers, F1.2 pattern-of-life, F1.3 breakup, F1.4 threat score) | ✅ done |
| Tier 2 (F2.1 RPO, F2.2 conjunction cascade) | ✅ done |
| Tier 3 (F3.1 Pc, F3.2 temporal graph) | ✅ done |
| Tier 4 (F4.1 grounded reports, F4.2 verification harness) | ✅ done |
| Bonus §10: regime auto-classification, re-entry/decay prediction, graveyard-compliance | ✅ done |
| Bonus: decision-support mitigation strategies (score/mitigations.py) | ✅ done |
| Multi-page interactive web console (web/app.py) — overview, catalog, orbit-map, dossiers, threat board, conjunctions, disposal monitor, events, country rollup; SVG visualizations; real-time Celestrak refresh | ✅ done |

Every capability has been run against REAL archived data (not just synthetic
tests) and validated against at least one real historical event or real
generated output — see README's status table and findings log for specifics
(Cosmos 1408/Fengyun-1C fragmentation, SJ-21/BeiDou-2 G2 tow, Iridium 33/
Cosmos 2251, Luch/Olymp-K serial-inspector pattern, real ISS 2023 reboost
cadence, a real MLX report at 5/5 grounded sentences after 3 real harness
bugs fixed, real Cosmos 1408 fragment re-entry dates predicted within days,
Sentinel SSO / Molniya / BeiDou-2 G2 graveyard classification, etc.).

**The three bonus features are integrated, not siloed**: regime feeds the
threat score (F1.4) with strategic-orbit bumps, and all three (regime,
decay, disposal) flow into the F4 grounded-report pipeline as fact builders
(`report/facts.py`), so a single `generate-report` narrates the full
characterization with hallucination-checking. A recurring real-data
through-line: BeiDou-2 G2 (the object SJ-21 towed in F2.1/F3.1) independently
comes out GRAVEYARD (regime) and COMPLIANT_GRAVEYARD (disposal) — three
features confirming one real event.

Remaining work: more bonus features if wanted, the dashboard update, git
commits — see "What's left".

Test suite: `python -m pytest tests/ -v` — all offline/synthetic, run this
after every change. Real-event validation runs separately via CLI commands
(`validate-breakup`, `assess-rpo`, `estimate-pc`, `screen-graph`) since those
need live Space-Track credentials and the local SQLite archive.

## Architecture quick-reference

```
config.py          paths, .env loading, constants
cli.py              all CLI commands (source of truth for how to invoke everything)
data/               store.py (SQLite), celestrak.py, spacetrack_client.py, tle_parse.py
core/               propagation.py, frames.py (RIC/CW), elements.py, groundtrack.py,
                    regime.py (bonus: physics-based regime auto-classifier)
detect/             maneuver, pattern_of_life, breakup, rpo, cascade, conjunction, graph,
                    decay (bonus: re-entry prediction), graveyard (bonus: disposal
                    compliance) — one module per capability, independently droppable
score/              threat.py (F1.4 composite score; regime-aware via core/regime.py),
                    mitigations.py (bonus: decision-support response playbook +
                    response_posture() — real public coordination frameworks, NOT
                    fabricated actions)
watch/agent.py      Autonomous "analyst on watch" agent (online-roadmap feature 1):
                    a deterministic watch CYCLE — object_state() per watched object →
                    diff_states() vs. the last stored cycle → significance → grounded
                    verified brief (report/generate.py::generate_watch_brief). Findings
                    only on CHANGE (new maneuver, threat band cross, RPO onset/close,
                    disposal change, new decay). First cycle = baseline (no false alerts).
                    Persisted in data/store.py (watch_list/watch_cycle/watch_finding/
                    watch_object_state). CLI: `cli.py watch [--add/--remove/--list/
                    --refresh/--no-brief]`. Web: /watch (I&W board + brief + cycle feed),
                    "+ add to watch" on every dossier. Offline by default; --refresh pulls
                    fresh Celestrak TLEs first. Design spec: docs/superpowers/specs/.
query/nlquery.py    Natural-language catalogue query (online-roadmap feature 2): the
                    local LLM's ONLY job is parse_query(text) -> QuerySpec (a validated,
                    constrained filter). Execution is 100% deterministic over SATCAT
                    (web/app.py::nl_query), so the model cannot hallucinate results — it
                    only produces a filter, which spec_from_json() validates (unknown
                    keys ignored, invalid values dropped, limit clamped). The interpreted
                    spec is always shown back (describe()). Web: /query ("ask" nav).
                    Validated live: "Chinese payloads in GEO", "most anomalous LEO
                    objects", "decayed Cosmos debris", "SSO near 98°" all parse correctly.
detect/anomaly.py   unsupervised ML: IsolationForest over catalogue orbital features;
                    scores how statistical-outlier an orbit is (NOT a threat verdict).
                    Scores are cached at fit time (batched) — never re-score per object.
detect/changepoint.py  Behavioral change-point detection (roadmap feature 3): over an
                    object's maneuver-campaign series, finds the split that maximizes the
                    two-sample t-statistic on burn size AND tempo; reports a regime SHIFT
                    (e.g. tempo tightened, burns stepped up) if t >= 2.5. Complements F1.2
                    (which flags single anomalous campaigns). Surfaced on the dossier
                    ("Behavioral shift" panel) and in characterize().
web/serialize.py +  JSON API (roadmap feature 4): /api/object/<id>, /api/threat-board,
web/app.py routes   /api/query?q=, /api/watch/latest, /api/conjunction/<a>/<b>.cdm.
                    Makes the analytic core programmatically consumable. The .cdm export
                    is a CCSDS-CDM-STYLE conjunction message (interoperability) with an
                    explicit caveat that covariance is modeled — NOT an operational CDM.
                    Serializers are pure/offline-testable.
report/             facts.py (structured facts payload), generate.py (F4.1,
                    local MLX generation), verify.py (F4.2, anti-hallucination
                    harness — pure text-in/facts-in, no LLM dependency)
validate/           events.py (labeled real events), run_benchmark.py
web/                index.html — static showcase / analyst-briefing page (telemetry-plate
                    aesthetic; hero is the REAL SJ-21→BeiDou-2 G2 relative-range trace).
                    app.py — multi-page Flask console (run: python web/app.py). Routes:
                    / overview, /catalog (searchable SATCAT browser), /object/<id> (the
                    flagship "detailed threat report" dossier — regime, elements, maneuvers,
                    pattern-of-life, threat-score factor breakdown, decay, disposal, RPO,
                    ranked mitigations, + on-demand grounded report), /object/<id>/report,
                    /threat-board (featured objects ranked live). characterize.py = the
                    per-object data-assembly layer that runs every applicable detector.
                    static/app.css + templates/ = the shared design system (same
                    telemetry-plate language as index.html). Threat board takes ~11s
                    (characterizes ~37 objects live) — a caching pass is a known TODO.
                    MORE PAGES (2nd increment): /orbit-map (altitude×inclination scatter
                    of the catalogue — the orbital shells), /disposal (GEO-belt compliance
                    monitor + belt scatter), /conjunctions (cascade-screened pairs),
                    /events + /events/<key> (validation events as case pages), /country
                    (operator/nation rollup), /object/<id>/refresh (pull latest TLE from
                    Celestrak live). VISUALS: web/plots.py — hand-composed SVG generators
                    (orbit scatter, per-object orbit diagram, RPO range trace, GEO belt
                    scatter, threat×altitude scatter, regime bars); self-contained, no JS
                    charting lib, native <title> tooltips.
                    3D: web/static/orbit3d.js + vendored three.js r128
                    (web/static/vendor/, offline). Dossier "Orbit — live 3D" panel renders
                    the real Keplerian orbit(s) around a wireframe Earth, rotatable/auto-
                    orbiting; RPO dossiers show both orbits. Marker phase is schematic
                    (stated in the caption), geometry is real.
                    3rd increment pages: /anomalies (IsolationForest outlier ranking),
                    and dossier now has: 3D orbit, statistical-anomaly panel, and a
                    response-posture & coordination panel (real frameworks).
                    CATALOG-CLICK FIX: opening a dossier for an active object with no
                    stored TLE now auto-fetches it from Celestrak (data.celestrak.ingest_object)
                    and caches it — every catalogue row populates on first open.
dashboard/          Streamlit app (F0.4) — NOT updated since early session; only
                    knows about F1.1 maneuvers, doesn't reflect F1.2-F3.2 yet
tests/               offline/synthetic pytest suite, one file per detect/score module
```

## Pipeline-correctness audit (2026-07-16) — real bugs found & fixed

A full validation pass for hallucinated values, data leakage, and pipeline
errors surfaced these (all fixed, all regression-tested):

- **Report editorializing** (the big one): the local LLM called a 10/100
  (low) object a "significant threat / notable concern." The
  anti-hallucination harness only checks NUMBERS, not qualitative spin, so
  it passed. Fix: a grounded `threat_level` fact (low/moderate/high via
  `report.facts.threat_band`) + a tightened system prompt forbidding the
  model from inventing severity language. Vanguard 1 now reads
  "unremarkable, does not pose significant concern."
- **Verification false-positives**: numbers inside the object NAME
  ("Vanguard 1" → 1) and inside string fact values ("3/5" coverage →
  "3 out of 5") were wrongly flagged. Fix: both pools added to
  `exact_values()`. Vanguard 1 report went 3/4 → 6/6 grounded.
- **Proximity used stale TLEs**: `_proximity_factor` propagated an object's
  latest element set to wall-clock "now". For objects last tracked years
  ago (Luch +3484d, BeiDou-2 G2 +1612d, decayed Cosmos 2542) this produced
  meaningless distances. Fix: omit the factor when the TLE is >30 days old.
- **Proximity was non-deterministic**: evaluating at "now" made the score
  jump run-to-run for fast/eccentric orbits (Vanguard 1 swung 10→52). Fix:
  evaluate at the object's OWN element epoch — deterministic, zero
  propagation error.
- **Decayed objects** were shown as if live. Fix: a DECAYED banner +
  "last known orbit" framing; proximity already omitted via staleness.
- **Data leakage: NONE found.** Decay validation uses a proper temporal
  holdout (train data ends before the decay date, forward-only
  extrapolation). The anomaly IsolationForest is unsupervised (no target to
  leak). Maneuver/pattern-of-life are self-referential by design, not
  predictions with a holdout.
- Known minor imprecision (not a bug): the regime classifier labels
  inclined-geosynchronous orbits (QZSS/IGSO) "OTHER"; the disposal
  classifier correctly calls them INCLINED_GEOSYNCHRONOUS. "OTHER" is an
  honest catch-all, so left as-is.

## Working conventions specific to this repo (learned, not guessed)

- **Never fabricate.** Every detector refuses (returns `None`/empty) rather
  than guessing on insufficient data — `detect_maneuvers`, `pattern_of_life`,
  `detect/breakup.py`, `detect/rpo.py`, `detect/conjunction.py` all do this.
  Preserve this pattern in any new code.
- **Validate against real data whenever credentials/archive access allow it.**
  This session found real bugs in nearly every module that only real data
  surfaced (wrong NORAD IDs, a retired Space-Track API class, under-sampling
  in the geometry filter, a silently-broken SATCAT column, TCA grids too
  coarse for Pc, and — for F4.2 — 3 separate false-positive classes in the
  verification harness's number-extraction regex that only showed up
  against genuine MLX-generated prose, not hand-written test sentences:
  hyphenated identifiers like "SJ-21" read as -21, clock times like
  "09:06:30" split into 3 unrelated numbers, and a word-boundary regex bug
  that silently failed to match ISO datetimes at all). Synthetic tests alone
  did not catch these. Treat "passes offline tests" as necessary, not
  sufficient — for report generation specifically, always sanity-check the
  harness against REAL generated text, not just hand-crafted examples.
- **State every approximation's limitation explicitly**, in both code
  docstrings and README, not just once. E.g. Pc's modeled (not measured)
  covariance, F2.2's GEO-belt screening limits, F1.4's incomplete HVA pool.
- **Ownership/attribution factors must stay nation-neutral** — catalog
  completeness only, never a per-country weighting. This is a hard ethical
  boundary from §1 of the spec, not a style preference.
- Tests: offline/synthetic in `tests/`, one file per module, mirroring
  `detect/`/`score/` layout. Pure logic gets extracted into testable
  functions when DB/propagation coupling would otherwise block offline
  testing (e.g. `combine_factors` in threat.py, `classify_episodes` in
  rpo.py, `cluster_from_rows` in breakup.py).
- Local venv at `.venv/` (already set up, `source .venv/bin/activate`).
  Credentials in `.env` (gitignored — Space-Track user/pass already
  configured). SQLite DB at `storage/drishti.sqlite3` (gitignored).
- Git: a fresh repo was initialized scoped to `drishti/` this session
  (previously the repo root was accidentally the whole `~/Downloads`
  folder). No commits made yet as of this writing — ask before committing.

## Environment constraints

- **Machine: 8GB RAM, Apple Silicon M1 MacBook Air.** Confirmed this session
  (`uname`: arm64, T8103) — Claude Code runs natively on the actual target
  machine, not a remote sandbox, so MLX/local-LLM work is genuinely testable
  here, not just theoretical.
- F4's LLM is **local MLX** (`mlx-community/Qwen2.5-3B-Instruct-4bit`, ~1.6GB
  on disk), per explicit user preference: no paid API, prefer free/local.
  Confirmed it fits comfortably in 8GB and generates in a few seconds once
  loaded. Qwen2.5 was chosen over Llama-3.2-3B specifically because Meta's
  models are gated on Hugging Face (require accepting a license via an HF
  account) — Qwen is ungated and downloads with no login needed.

## What's left

- **Web console — next pages** (the multi-page app has overview/catalog/
  dossier/threat-board/report; these are the planned additions):
  events timeline (the 7 validation events as case pages with the real
  plots — the SJ-21 range trace already exists in web/index.html and can
  be reused), disposal/graveyard belt monitor (scan_geo_belt() already
  returns the data), conjunctions screen (cascade + Pc on screened pairs),
  country/operator threat rollup (aggregate by SATCAT owner), and a
  real-time refresh button (pull latest TLE from Celestrak on demand —
  the ingest path already exists). Also: cache the threat board (~11s live).
- **Bonus features** (handoff §10) — 4 done (regime, decay, graveyard,
  mitigations). Remaining strong candidates reusing existing infra:
  natural-language query interface (LLM→structured query, reuses F4
  report/facts + MLX), launch-detection/attribution (reuses F1.3
  catalog_snapshot delta), constellation-scale pattern-of-life (reuses
  F1.2), ground-station pass prediction (skyfield). Research-grade ones
  (federated fusion, temporal graph transformer, Bayesian change-point)
  are heavier lifts.
- **Dashboard** (`dashboard/app.py`) has not been updated since early in the
  session — still only wired to F1.1 maneuvers. Updating it to surface
  F1.2-F4.2 outputs (pattern-of-life, breakup, threat score, RPO, Pc, graph,
  generated reports) is unstarted, real work, not yet scheduled.
- **No commits made to git yet** this session — everything is uncommitted
  working-tree changes on the freshly-initialized `drishti/`-scoped repo.
  Ask before committing.
- The paper/validation-set angle (handoff §8) — the labeled event set
  (`validate/events.py`) and the real findings logged in README are the
  actual dataset-building work; formal write-up hasn't started.
