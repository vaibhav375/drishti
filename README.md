# DRISHTI — Space Domain Awareness & Behavioral Threat Analysis

**Live demo: https://drishti-25tc.onrender.com** — free-tier host, so the first
visit after idle takes ~1 min to wake and heavy pages take 20–60 s. The demo
uses public Celestrak data only and has no local LLM (reports show the grounded
facts instead of a narrative) — see [Deploying](#deploying-render-free-plan).

**DRISHTI reads what a satellite is *doing*, not just where it is** — and it
runs entirely offline, on a laptop, from public data. It ingests public
two-line element sets, characterizes on-orbit behavior (maneuvers, rendezvous
& proximity operations, fragmentation, disposal), scores threat explainably,
and writes grounded analyst reports with a local LLM whose every sentence is
checked against computed facts. Validated end-to-end against real historical
events — the SJ-21 tow of BeiDou-2 G2, the Cosmos 1408 ASAT breakup, Luch's
serial GEO inspections, the ISS as a routine control.

**Framing:** decision-support and situational awareness. **Not targeting** —
an ethical boundary enforced in code (attribution is nation-neutral; the
mitigation engine has no offensive actions in its vocabulary).
**Data:** public only (Celestrak, Space-Track, SATCAT). **Runs offline.**

### What's in it
- **Detection engine** (`detect/`, `score/`) — maneuvers, pattern-of-life,
  fragmentation, RPO, conjunction cascade, Monte-Carlo Pc, interaction graph,
  regime classification, re-entry prediction, disposal compliance, decision-
  support mitigations, unsupervised ML anomaly detection, change-point.
- **Grounded reports** (`report/`) — a local MLX model writes the narrative;
  an anti-hallucination harness rejects any sentence with an unbacked number.
- **Watch agent** (`watch/`) — an autonomous cycle that reports what *changed*
  since it last looked, with a verified brief.
- **Natural-language query** (`query/`) — the LLM parses English into a
  validated filter; execution is deterministic, so results can't be hallucinated.
- **Web console** (`web/`) — a multi-page Flask app: catalog, orbit map,
  per-object dossiers with a live 3D orbit view, threat board, anomalies,
  conjunctions, disposal monitor, the watch floor, and a JSON API.
- **~140 offline tests**; every real-data finding and bug is logged below.

### Try it in 60 seconds
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # optional: add free Space-Track creds
python cli.py ingest-celestrak --group active --group geo   # pull public TLEs
python cli.py ingest-satcat                # object metadata
python web/app.py                          # → open http://127.0.0.1:5000
```

> **Project memory:** `CLAUDE.md` is a maintained, agent-facing summary of
> status/architecture/conventions — read it first if picking this up cold.
> `SDA_DASHBOARD_HANDOFF.md` is the original full spec.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes mlx/mlx-lm (Apple Silicon only, F4)
cp .env.example .env      # fill in Space-Track credentials (free account:
                           # space-track.org/auth/createAccount)
```

F4's local report generation (`generate-report`) downloads a ~1.6GB model
(`mlx-community/Qwen2.5-3B-Instruct-4bit`) to the Hugging Face cache on
first run — no API key, no login required (ungated, unlike Meta's Llama
models). Confirmed to run well within 8GB RAM on an M1 MacBook Air.

## Quick start

```bash
# 1. Pull current catalog groups (no auth needed)
python cli.py ingest-celestrak --group stations --group active

# 2. Pull object metadata
python cli.py ingest-satcat

# 3. Week-1 gate: verify historical archive coverage for validation events
python cli.py check-coverage --norad 44797 --norad 44835 --start 2019-12-01 --end 2020-03-15

# 4. Pull historical TLEs for an event window
python cli.py ingest-historical --norad 25544 --start 2023-01-01 --end 2023-12-31

# 5. Run maneuver detection on stored history
python cli.py detect-maneuvers --norad 25544

# 5b. Classify those maneuvers against the object's own pattern of life
python cli.py pattern-of-life --norad 25544

# 6. F1.3 breakup benchmark against real Cosmos 1408 / Fengyun-1C + controls
python cli.py validate-breakup

# 6b. Live catalog-delta monitoring (run periodically, e.g. daily via cron)
python cli.py snapshot-catalog
python cli.py check-catalog-delta

# 6c. F1.4 composite threat score with a mandatory per-factor breakdown
python cli.py ingest-celestrak --group geo --group gps-ops  # a real high-value-asset pool
python cli.py threat-score --norad 25544

# 6d. F2.1 RPO assessment for an actor/target pair (needs overlapping archive coverage)
python cli.py assess-rpo --actor 49330 --target 34779   # SJ-21 vs BeiDou-2 G2

# 6e. F2.2 conjunction-filtering cascade over the current live catalog
python cli.py screen-conjunctions --limit 10

# 6f. F3.1 Monte Carlo Pc for a specific pair (needs overlapping archive coverage)
python cli.py estimate-pc --actor 49330 --target 34779   # SJ-21 vs BeiDou-2 G2

# 6g. F3.2 serial-inspector screen: one actor vs many candidate targets
python cli.py screen-graph --actor 40258 --target 34710 --target 26927 \
    --start 2014-10-01 --end 2016-12-31   # Luch vs two real EUTELSAT candidates

# 6h. F4.1/F4.2 grounded report + verification (first run downloads ~1.6GB local model)
#     folds in regime + decay + disposal facts automatically, all hallucination-checked
python cli.py generate-report --norad 34779 --rpo-target 49330   # BeiDou-2 G2 (SJ-21 towed it)

# Bonus features (§10):
python cli.py classify-regime --norad 40697        # regime auto-classification (Sentinel → SSO)
python cli.py predict-decay --norad 50632          # re-entry prediction from decay trend
python cli.py check-graveyard --norad 34779        # GEO disposal compliance (one object)
python cli.py check-graveyard                       # ...or a belt-wide compliance scan

# 6i. Autonomous watch agent (roadmap feature 1) — runs the pipeline over a
#     watchlist and reports what CHANGED since the last cycle, with a grounded brief
python cli.py watch --add 49330          # add SJ-21 to the watchlist
python cli.py watch                      # run a cycle (offline); --refresh pulls fresh TLEs

# 7. Interactive web console (multi-page Flask app). Pages: overview,
#    catalog, ask (natural-language query → LLM-parsed filter), orbit-map,
#    per-object dossiers (3D orbit + RPO trace + threat factors + anomaly +
#    behavioral-shift + mitigations + response posture), watch floor (I&W
#    board + brief + cycle feed), threat board, anomalies, conjunctions,
#    disposal monitor, events, country rollup. Live "refresh from Celestrak".
#    JSON API: /api/object/<id>, /api/threat-board, /api/query?q=,
#    /api/watch/latest, /api/conjunction/<a>/<b>.cdm (CCSDS-CDM-style export).
python web/app.py           # then open http://127.0.0.1:5000

# 7b. Static one-page showcase (open directly in a browser)
open web/index.html

# 7c. Original Streamlit dashboard (F0.4 skeleton) — DEPRECATED, superseded
#     by the Flask console above; kept for history only.
streamlit run dashboard/app.py
```

## Deploying (Render free plan)

```bash
python deploy/build_space.py --target render   # bundle -> build/render/, public Celestrak-only DB
# push build/render/ as the orphan `render` branch, then in Render:
# New -> Blueprint -> this repo, branch `render` (render.yaml is at its root)
```

The deploy ships **only public Celestrak data** — Space-Track TLEs are stripped
at build time (their user agreement forbids redistribution), so detectors that
need long element-set history often (correctly) refuse there. The 512 MB host
has no LLM (`DRISHTI_LLM_BACKEND=none`): reports show the grounded facts block
instead, and "ask" falls back to name search. `--target space` builds a
Hugging Face Docker Space variant with a CPU LLM (requires HF PRO).
Found while deploying: at full-catalogue scale (~16.8k objects) per-connection
schema setup and one-query-per-object loops made dossiers ~10x slower, and
`run_cascade` never finished on the Starlink shells — fixed with a bulk TLE
query and the vectorized, equivalence-tested `detect/cascade.py::screen_top`.

## Status vs handoff tiers

| Feature | Status |
|---|---|
| F0.1 Ingestion (Celestrak + Space-Track + SATCAT, dedupe by norad_id+epoch) | ✅ built, exercised against live endpoints |
| F0.2 Propagation engine (sgp4, TEME) | ✅ built |
| F0.3 Frames — RIC basis, relative motion, CW equations (no poliastro) | ✅ built + tested |
| F0.4 Dashboard skeleton (ground track + 3D orbit) | ✅ built, smoke-tested |
| F1.1 Maneuver detection + Δv + confidence | ✅ tested vs synthetic burn AND calibrated against real ISS 2023 + Cosmos 2542/2543 archive history |
| F1.2 Pattern-of-life baseline (detect/pattern_of_life.py) | ✅ campaign-grouped baseline; validated against real ISS 2023 reboost cadence |
| F1.3 Breakup detection (detect/breakup.py) | ✅ retrospective (SATCAT launch-lineage) + live (catalog_snapshot delta) paths; validated 4/4 against real Cosmos 1408, Fengyun-1C, and two controls |
| F1.4 Threat score (score/threat.py) | ✅ 5-factor composite with mandatory per-factor breakdown; ran against real ISS/Cosmos 2542/2543 |
| F2.1 RPO / inspector detection (detect/rpo.py) | ✅ validated against SJ-21's real tow of BeiDou-2 G2 ("likely"), Iridium 33/Cosmos 2251 pre-collision ("none"); flags shared-launch pairs (deployment vs. inspection ambiguity) via SATCAT lineage |
| F2.2 Conjunction-filtering cascade (detect/cascade.py) | ✅ altitude sieve + relative-inclination filter; runs the real ~626-object live catalog (195,625 possible pairs) in ~1.5s |
| F3.1 Probabilistic conjunction assessment / Pc (detect/conjunction.py) | ✅ Monte Carlo Pc with modeled RIC covariance + RCS-derived hard-body radius; fine-grained TCA search validated against real SJ-21/BeiDou-2 G2 (found a 35 m approach) and Iridium 33/Cosmos 2251 (correctly found no data spans the actual collision instant) |
| F3.2 Temporal proximity-interaction graph (detect/graph.py) | ✅ networkx MultiDiGraph + degree-based serial-inspector rule; **found the real Luch/Olymp-K repeat-inspector pattern** against real archived data — the exact acceptance case §6 names |
| F4.1 Grounded report generation (report/generate.py) | ✅ local MLX (Qwen2.5-3B-Instruct-4bit, ~1.6GB), confirmed to fit and run well within the real 8GB M1 target machine |
| F4.2 Anti-hallucination verification harness (report/verify.py) | ✅ pure text-in/facts-in, no LLM dependency; catches an injected hallucination in tests, and a real generated report reaches 5/5 grounded sentences after 3 real regex bugs were found and fixed |
| Bonus: regime auto-classification (core/regime.py) | ✅ physics-based (J2 nodal precession for SSO, critical-inclination + period for Molniya/Tundra); real Sentinel→SSO, Molniya sats→HEO[MOLNIYA], BeiDou-2 G2→GRAVEYARD; feeds the threat score |
| Bonus: re-entry/decay prediction (detect/decay.py) | ✅ exponential-atmosphere model with data-derived scale height; predicted 3 real Cosmos 1408 fragment re-entries within 2-9 days (vs. 400-700 days for a naive line fit) |
| Bonus: graveyard-disposal compliance (detect/graveyard.py) | ✅ IADC-floor check, single-object + belt-wide scan; real live scan found compliant graveyard objects, a shallow non-compliant re-orbit, and 23 uncontrolled drifters |

**All spec tiers (0-4) complete and validated; first 3 bonus features (§10) done and integrated.**

Tests: `python -m pytest tests/ -v` — 96 passing, all offline/synthetic
(frames orthonormality, transport-term validation, propagation sanity,
injected-burn detection, quiet-history false-positive check,
insufficient-history guard, pattern-of-life campaign grouping and
routine/anomalous classification, breakup clustering incl. the
catalog-number-dispersion gate, threat-score factor logic and weighted-
average combination, RPO episode classification and launch-lineage
caveat, cascade altitude/plane filtering incl. a dense-population
performance regression test, Pc hard-body-radius/covariance/TCA logic,
graph edge-building and serial-inspector detection, F4 fact-grounding
and hallucination-detection logic incl. 3 real regex regression tests,
regime SSO/Molniya/graveyard classification, decay linear-vs-exponential
model selection, graveyard disposal-status logic).
Real-event validation that needs the live archive runs separately:
`python cli.py validate-breakup`,
`python cli.py assess-rpo --actor 49330 --target 34779`,
`python cli.py screen-conjunctions`,
`python cli.py estimate-pc --actor 49330 --target 34779`.

## Real-data findings this session (2026-07-15)

Two Space-Track credentials arrived and the whole pipeline was run against
live data for the first time. This surfaced real bugs no synthetic test
could have — recorded here so they aren't rediscovered:

- **Space-Track retired the `tle`/`tle_latest` classes.** `data/spacetrack_client.py`
  now uses `gp_history`, the current historical-archive class. The old
  `tle` class returns `400 Your Class Does Not Exist`.
- **Two NORAD IDs in `validate/events.py` were wrong**, confirmed against a
  live SATCAT pull: USA 245 is **39232**, not 32711 (which is NAVSTAR
  62/USA 201, a GPS satellite). BeiDou-2 G2 is **34779**, not 36287 (which
  is G1).
- **F1.2's tempo baseline broke on real archive density.** Dense
  Space-Track history yields several closely-timed residuals (hours apart)
  per physical burn (multi-pulse reboosts). Without grouping, those tight
  intra-burn gaps set an absurdly low "typical tempo", which then flagged
  the real, regular gap *between* burns — ISS's actual ~2-4 week reboost
  cadence — as anomalous. Fixed by grouping same-burn residuals into
  `ManeuverCampaign`s before baselining (`detect/pattern_of_life.py`).
- **MAD can be exactly zero with few samples**, turning an ordinary 1-day
  tempo drift into a thousand-sigma "anomaly". Fixed with a sigma floor
  relative to the baseline's own magnitude (`MIN_RELATIVE_SIGMA`), not a
  bare epsilon.
- **Inclination clustering alone can't tell a fragmentation from decades of
  incidental litter.** ISS's own launch designator carries 148 DEB-typed
  objects (individually-lost EVA tools, a camera) that cluster in
  inclination just like real debris would, since they share the parent's
  orbit plane. What separates them: catalog numbers are assigned roughly
  in order of first tracking, so a genuine fragmentation's fragments get
  catalogued in a tight burst (Cosmos 1408: norad_id MAD/median ≈1.3%;
  Fengyun-1C ≈4.1%) while slowly-accumulated litter is scattered across
  decades of catalog growth (ISS ≈18.5%). `detect/breakup.py` gates on this.
- **Confirmed real signal, unprompted:** F1.1 (built before any RPO
  detector existed) flags a 36.9 m/s and 49.8 m/s *radial* burn for Cosmos
  2543 on 2019-12-14/16 — matching the publicly documented close-approach
  maneuvers toward USA 245.
- **F1.4's proximity factor self-matched on multi-module space stations.**
  Scoring ISS found ~0 km to POISK (country='ISS'), then — after excluding
  that — to NAUKA (country='CIS'): SATCAT's `country` field is inconsistent
  across ISS modules, so a country-code exclusion is fragile. Fixed with a
  general distance floor (`MIN_PHYSICAL_SEPARATION_KM`): anything under
  ~2 km at single-epoch granularity is the same docked structure, not a
  distinct object — ISS itself only spans ~100 m. Genuine close-approach
  behavior over multiple passes belongs to F2.1, not this coarse snapshot.
- **Sanity-checked F1.4 against the three richest real objects**: Cosmos
  2543 (53/100, the actual RPO actor with the Dec 2019 close-approach
  burns) scores highest, Cosmos 2542 (41/100) next, ISS lowest (29/100,
  well-attributed and maneuvers often but almost all routine) — an
  explainable ranking that matches what's publicly documented about these
  three objects' behavior.
- **F2.1 validated against three real pairs.** SJ-21 vs BeiDou-2 G2:
  range collapses from ~32,000 km to <5 km over ~3.5 weeks (Dec 2021),
  then holds under ~2 km with relative speed <1 m/s for nearly a month —
  correctly "likely" (19 total days close). Iridium 33 vs Cosmos 2251
  (Jan-Apr 2009, before their Feb 10 collision): minimum separation
  664 km, correctly "none" — the eventual collision was a fast conjunction,
  not loitering, and the detector doesn't confuse the two. Cosmos 2542 vs
  Cosmos 2543: correctly "likely" too, but for the WRONG underlying
  reason — it's their real Dec 2019 sub-satellite deployment, not RPO
  inspection. Fixed: `assess_rpo` now checks SATCAT launch designators
  (same intl_desig-prefix trick as `detect/breakup.py`) and attaches a
  `same_launch_lineage` flag + caveat in the summary — it doesn't
  suppress the result (siblings can still do genuine RPO later), it
  flags it for an analyst to check.
- **F2.2 went through two real design failures before working.** V1
  sampled each orbit's ellipse at a FIXED point count (72) and took the
  minimum pairwise point distance — this under-resolved GEO-scale orbits
  (~3,700 km between samples at GEO radius) and spuriously filtered out
  the real SJ-21/BeiDou-2 G2 pair. V2 made sampling arc-length-adaptive,
  which fixed that — but then OOM-killed on the real ~570-object GEO
  belt, because hundreds of near-identical, near-coplanar circular orbits
  put ENTIRE ARCS of points within any reasonable threshold of each other
  simultaneously (not sparse crossings — a combinatorial explosion no
  amount of KD-tree indexing fixes, since the problem is the point count
  itself, not the search algorithm). Root cause: a point-cloud minimum-
  distance is the wrong tool for a densely coplanar population. V3
  replaced stage 2 entirely with closed-form RELATIVE INCLINATION between
  the two orbital planes (i, RAAN only, O(1) per pair, no sampling) —
  correctly keeps SJ-21/BeiDou-2 G2 (0.02° apart) and runs the full real
  catalog (626 objects, 195,625 possible pairs) in ~1.5s.
- **SATCAT's RCS field was silently empty for all 69,871 rows.**
  `data/celestrak.py` parsed `RCS_SIZE` (the legacy SMALL/MEDIUM/LARGE
  category), but the live Celestrak CSV column is `RCS` — a numeric radar
  cross-section in m² (e.g. ISS ≈399 m², BeiDou-2 G2 ≈1.26 m², Iridium 33
  ≈2.52 m², all physically plausible once fixed). ~47% of objects publish
  it. Fixed with a schema migration (`rcs_class TEXT` → `rcs_m2 REAL`,
  `data/store.py::_migrate` upgrades existing local DBs without losing
  already-ingested archive history) — this directly feeds F3.1's
  hard-body-radius estimate.
- **F2.1's 6-hour RPO grid is too coarse for Pc.** At ~7.5 km/s relative
  velocity that's ~160,000 km between samples — a genuine sub-km
  encounter falls between them. F3.1's `find_time_of_closest_approach`
  uses a finer coarse pass (default 1h) with local regridding down to
  ~1s resolution, and found the real SJ-21/BeiDou-2 G2 approach at just
  **35 m** — tighter than F2.1's own episode-level 0.2 km minimum.
- **Confirmed a real, honest data gap, didn't paper over it.** Searching
  for Iridium 33/Cosmos 2251's TCA across their full Jan-Apr 2009 archive
  window converges on Feb 12 (123 km) — two days AFTER their actual
  Feb 10, 16:56 UTC collision, not on it. Checked directly: both objects'
  archived TLEs jump straight from Feb 10 ~18:xx to Feb 12 — a real
  ~1.5-day gap in the archive spanning the collision (plausible: after an
  unplanned catastrophic fragmentation, re-establishing tracked orbit
  determinations for what's left of the original catalog objects takes
  time). F3.1 correctly reports the closest approach the DATA supports,
  and does not fabricate a near-miss around a moment no data spans.
- **F3.2's serial-inspector acceptance case validated against real data,
  not synthetic.** `validate/events.py` names Luch/Olymp-K's GEO drift as
  "the temporal graph (F3.2) target" — no specific historical incidents
  were recalled from memory (that would risk fabricating claims); instead
  a real, data-driven candidate list was built (149 real GEO comsats
  launched before 2014, still active through 2016, per live SATCAT), a
  sample of 20 EUTELSAT/INTELSAT objects was archive-ingested for Luch's
  2014-2016 drift window, and `assess_rpo` was run for real against each.
  Result: Luch shows **"likely" against EUTELSAT 10A — 28.3 km minimum
  range, 51 separate episodes, 57.0 total days close** — and "possible"
  against EUTELSAT 12 WEST B. `detect/graph.py` correctly flags Luch as a
  serial inspector with 2 distinct targets from real archived data — the
  exact pattern §6 asks this tier to surface. (Confidence reads "low" via
  the plain CLI command specifically because these two EUTELSAT objects
  aren't in the CURRENT live-ingested HVA pool — same disclosed
  incompleteness as F1.4's proximity factor; the underlying signal, 2
  distinct real targets with substantial sustained closeness, is
  unaffected by that labeling detail.)
- **F4.2's verification harness had 3 real bugs, all found by testing
  against genuine MLX output instead of only hand-written examples.**
  (1) A real generated sentence read "SHIJIAN-21 (SJ-21)"; the regex read
  the hyphenated identifier suffix "-21" as the literal number -21 and
  flagged it as an unbacked hallucination. Fixed by masking hyphenated
  alphanumeric identifiers before number extraction. (2) "09:06:30 UTC"
  got split by its colons into three unrelated numbers (9, 6, 30), none
  individually grounded, even though the whole timestamp matched a
  coverage-window fact. Fixed by masking clock-time substrings first.
  (3) While fixing (2), a date-extraction regex requiring a trailing `\b`
  silently failed to match ISO datetimes at all — `\b` doesn't fire
  between "01" and "T" in "2021-12-01T00:00:00Z" since both are word
  characters — so day/month numbers (e.g. "February 14" flagging the
  day "14") were never being grounded. All three are now regression-
  tested with real generated sentences, not just synthetic ones. Net
  result: the same real SJ-21/BeiDou-2 G2 report went from 3/5 grounded
  sentences before these fixes to 5/5 after.
- **A 2%-relative-tolerance design flaw, caught before it shipped:**
  giving EVERY known fact value the same rounding tolerance meant a
  fixed scale-marker like the "100" in a "/100" threat score let a
  *coincidentally nearby fabricated number* (a hallucinated "99.9")
  slip through as "grounded" — exactly the kind of plausible hallucination
  the harness exists to catch. Fixed by splitting facts into two pools:
  measured quantities (`numeric_values()`, tolerant — LLMs legitimately
  round 13.26 to 13.3) and identifiers/scale-markers (`exact_values()`,
  zero tolerance — a "100" must be exactly 100, not "close to" 100).
- **Bonus regime classifier caught a real Molniya-vs-GTO mislabel.** Real
  Molniya 1-29 (64° inclination, 12h period, ecc 0.69) first classified
  as primary "GTO" because it shares the LEO-perigee/GEO-apogee band with
  a transfer orbit. Inclination is the real discriminator: a genuine GTO
  is a LOW-inclination temporary transfer; a high-inclination high-ecc
  orbit is operational Molniya-class HEO. Fixed → HEO[MOLNIYA]. SSO
  detection is pure physics: Sentinel-2A/3A at 98.6° matched the J2-derived
  sun-synchronous inclination of 98.6° for their altitude exactly.
- **Decay predictor: a naive line fit was 400-700 days wrong; the fix is
  physically grounded, not just tuned.** Real Cosmos 1408 fragments that
  re-entered ~30 days out were predicted ~555 days out by linear perigee
  extrapolation — because atmospheric density (and thus decay rate) rises
  roughly exponentially as altitude drops, so decay accelerates near the
  end. The fix derives each object's OWN effective atmospheric scale
  height from how much its decay rate grew across the two halves of its
  history (H = Δaltitude / ln(rate_new/rate_old)) and integrates the
  exponential-atmosphere decay. Result: 3 real fragments predicted within
  **2-9 days** of their actual re-entry dates, no assumed ballistic
  coefficient. Data-derived scale heights (20-27 km) are physically sane
  for the terminal-decay regime.
- **BeiDou-2 G2 is the same real object across four features now.** SJ-21
  towed it (F2.1 "likely" RPO, F3.1 35 m closest approach); the regime
  classifier independently calls it GRAVEYARD (~292 km above GEO); the
  disposal-compliance monitor calls it COMPLIANT_GRAVEYARD; and a grounded
  report weaves all of that together at 7/7 verified sentences. One real
  event, cross-confirmed by independent detectors — the kind of
  corroboration the whole "behavioral characterization" thesis is about.
- **A graveyard operational-box ordering bug, caught by testing.** An
  object re-orbited only +120 km above GEO (a real insufficient-disposal
  case) was first read as "operational" because the operational-box check
  used the full ±200 km protected-region width. Real active satellites
  station-keep to a much tighter box (~±75 km), so the box was narrowed to
  ±100 km — now a +120 km object correctly reads NON_COMPLIANT_SHALLOW.
  A live belt scan then found real examples: 23 uncontrolled drifters, a
  real shallow re-orbit (NORAD 41838, +107 km), and 4 compliant graveyard
  objects.
- **The web disposal monitor surfaced a real misclassification.** Building
  the belt-compliance page, the "23 uncontrolled drifters" turned out to be
  mostly active QZSS, BeiDou-IGSO, and IRNSS satellites — these are
  *inclined geosynchronous* (IGSO), a legitimate operational orbit type,
  inclined by design so their perigee dips below GEO. Calling them
  "uncontrolled drift" was wrong. Fixed: `detect/graveyard.py` now
  recognizes a geosynchronous-period, near-GEO, inclined object as
  INCLINED_GEOSYNCHRONOUS (operational, not a disposal case) before the
  drift fallback — the classic case of a new visualization exposing a
  labeling error the tabular output had hidden.

## Honesty rules (enforced in code)

- Δv estimates carry a z-score against the object's **own** noise floor and a
  confidence label. TLEs are low-precision; we say so.
- `detect_maneuvers` returns nothing when history is too thin to baseline —
  it never flags against an unfounded baseline. `pattern_of_life` likewise
  refuses below 3 maneuver campaigns.
- `detect/breakup.py` returns `None` rather than a low-confidence guess when
  fragment count, regime clustering, or catalog-burst tightness don't
  support a fragmentation claim.
- `check-coverage` exists so no validation event is built on missing data —
  confirmed live: USA 245 has **zero** archived TLEs in the Cosmos 2542
  window (classified, as predicted), all 10 other event objects have solid
  coverage (119–2103 TLEs each).
- Credentials live in `.env` (gitignored). Nothing hardcoded.

## Known limitations / notes for next session

- Δv magnitude is now calibrated against real ISS 2023 data (typical burn
  ≈3.0 m/s per campaign, noise floor ≈0.22 m/s) rather than assumed — the
  previously-unused `MANEUVER_NOISE_FLOOR_MS` config placeholder was
  removed since the adaptive per-object baseline supersedes it.
- USA 245 archive coverage is confirmed zero for the Cosmos 2542 window —
  target ephemeris for that event must stay marked unavailable, never
  fabricated, wherever it's used later (F2.1 RPO).
- F1.3's live path (`snapshot-catalog` / `check-catalog-delta`) is wired
  and smoke-tested but not yet run on a real schedule (cron) or against a
  real in-progress breakup — only validated retrospectively.
- `astropy` TEME→GCRS conversion isn't wired yet; ground tracks go through
  skyfield (correct), and both RPO relative motion (F2.1) and Pc (F3.1)
  stay in consistent TEME (valid for these relative comparisons — see
  `detect/conjunction.py` docstring for why this doesn't block Pc after
  all, correcting an earlier assumption). Only add astropy conversion if
  something needs to compare against an externally-sourced ABSOLUTE-frame
  value, which nothing does yet.
- F1.4's high-value-asset pool is currently whatever we've live-ingested
  (stations + geo + gps-ops groups, ~600 objects, filtered to SATCAT
  object_type=='PAY') — a real, data-driven starting pool, not exhaustive.
  Expand by ingesting more Celestrak groups.
- F1.4's `ownership`/attribution factor deliberately does NOT weight by
  nation — only by catalog completeness (known vs. UNK country/type). A
  per-nation weighting would cross into the targeting-flavored territory
  §1 explicitly rules out for this project.
- F2.1's Cosmos 2542/2543-vs-USA-245 acceptance case (§6) can't be
  validated with real data — USA 245 has zero archived TLEs, confirmed.
  The SJ-21/BeiDou-2 G2 tow is the real validation case instead.
- F2.2's relative-inclination filter does NOT meaningfully narrow down
  pairs WITHIN one densely populated, near-coplanar regime — the GEO belt
  is the real example (~151,000 of 195,625 real pairs survive, mostly
  GEO-GEO). That's an honest fact about GEO, not a bug: nearly everyone
  shares a plane there by construction. Real GEO screening needs RAAN/
  longitude-slot binning or straight time-domain propagation, neither
  built yet. The cascade's real value today is eliminating CROSS-regime
  pairs (the literal problem §7 names) and diverging-plane same-regime
  pairs.
- F2.2 (screen-conjunctions) and F3.1 (estimate-pc) are not yet wired
  together end-to-end: the cascade runs against the LIVE-only catalog
  (whatever we've `ingest-celestrak`'d), while Pc needs archived TLE
  HISTORY for both objects (only available for `validate/events.py`
  objects so far). Chaining "cascade output -> Pc on survivors" needs
  either broader historical ingestion or accepting Pc only runs on
  cascade candidates we happen to have archive history for.
- F3.1's covariance model (`SIGMA_*_KM` in `detect/conjunction.py`) and
  hard-body-radius default (5 m, used for the ~53% of objects with no
  published RCS — including SJ-21 itself) are DISCLOSED starting
  assumptions, not fit to any real covariance data (none is public).
  Real Pc calibration would need CDM data DRISHTI deliberately doesn't
  have access to, by design (public-data-only, §1).
- F3.2 (`detect/graph.py`) is a plain networkx MultiDiGraph with a
  degree-based rule (distinct-target count + HVA fraction), deliberately
  NOT the "Temporal Graph Transformer" the handoff's background
  mentions — that's an explicit bonus/research-grade extension (§10) on
  top of this baseline, not required for Tier 3.
- F3.2 doesn't run RPO detection itself — it consumes `RpoAssessment`s
  computed elsewhere (`screen-graph` calls `assess_rpo` per candidate
  target). The real Luch screen used a hand-picked sample of 20 real
  EUTELSAT/INTELSAT candidates (out of 149 real ones matching the launch/
  decay window), not an exhaustive search — a genuinely serial pattern
  against a target outside that sample would be missed. Widening the
  candidate list (more operators, more of the 149) is straightforward but
  costs more Space-Track archive calls per run.
- F4's facts payload (`report/facts.py`) currently has builders for
  maneuvers, threat score, RPO, Pc, and breakup — not every detect/
  module (e.g. no dedicated builder for cascade `ScreenedPair` or graph
  `SerialInspectorFinding` yet, though nothing structural blocks adding
  one the same way).
- `report/verify.py`'s day/month extraction only handles the ISO
  `YYYY-MM-DD` pattern already present in every date fact's ISO string;
  a report writing a date in some other format entirely wouldn't ground
  against it. Not observed in practice yet (Qwen2.5 consistently restates
  dates plausibly), but worth knowing as a boundary of the fix.
- Dashboard, bonus features, and git commits are what's left overall —
  see `CLAUDE.md` "What's left" for the current, maintained list (kept
  more up to date turn-to-turn than this section of the README).
