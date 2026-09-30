"""DRISHTI command-line interface: ingestion + week-1 archive checks.

Examples
--------
python cli.py ingest-celestrak --group stations --group active
python cli.py ingest-satcat
python cli.py ingest-historical --norad 44797 --norad 44835 --start 2020-01-01 --end 2020-03-01
python cli.py check-coverage --norad 13552 --start 2021-10-15 --end 2021-12-15
python cli.py detect-maneuvers --norad 25544
python cli.py pattern-of-life --norad 25544
python cli.py validate-breakup
python cli.py threat-score --norad 25544
python cli.py assess-rpo --actor 49330 --target 34779
python cli.py screen-conjunctions --limit 10
python cli.py estimate-pc --actor 49330 --target 34779
python cli.py screen-graph --actor 40258 --target 34710 --target 26927 --start 2014-10-01 --end 2016-12-31
python cli.py generate-report --norad 44835 --rpo-target 39232
"""
from __future__ import annotations

import argparse
from datetime import date


def main() -> None:
    p = argparse.ArgumentParser(prog="drishti")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("ingest-celestrak", help="Pull current TLE groups (no auth)")
    c.add_argument("--group", action="append", help="Celestrak group; repeatable")

    sub.add_parser("ingest-satcat", help="Pull SATCAT metadata (no auth)")

    h = sub.add_parser("ingest-historical", help="Pull archived TLEs from Space-Track")
    h.add_argument("--norad", action="append", type=int, required=True)
    h.add_argument("--start", type=date.fromisoformat, required=True)
    h.add_argument("--end", type=date.fromisoformat, required=True)

    k = sub.add_parser("check-coverage", help="Week-1: verify archive coverage per object")
    k.add_argument("--norad", action="append", type=int, required=True)
    k.add_argument("--start", type=date.fromisoformat, required=True)
    k.add_argument("--end", type=date.fromisoformat, required=True)

    cr = sub.add_parser("classify-regime", help="Bonus: auto-classify an object's orbital regime")
    cr.add_argument("--norad", type=int, required=True)

    dc = sub.add_parser("predict-decay", help="Bonus: re-entry/decay prediction from TLE history")
    dc.add_argument("--norad", type=int, required=True)

    gv = sub.add_parser("check-graveyard",
                         help="Bonus: GEO disposal-compliance for one object (--norad) "
                              "or a belt-wide scan (no --norad)")
    gv.add_argument("--norad", type=int, default=None)

    m = sub.add_parser("detect-maneuvers", help="Run F1.1 on stored TLE history")
    m.add_argument("--norad", type=int, required=True)

    pol = sub.add_parser("pattern-of-life", help="Run F1.2 on an object's detected maneuvers")
    pol.add_argument("--norad", type=int, required=True)

    sub.add_parser("validate-breakup", help="F1.3 benchmark: real Cosmos 1408/Fengyun-1C vs. controls")

    sub.add_parser("snapshot-catalog", help="F1.3: record which active objects exist right now")
    sub.add_parser("check-catalog-delta", help="F1.3: diff latest two snapshots, flag breakup candidates")

    ts = sub.add_parser("threat-score", help="Run F1.4 composite threat score for an object")
    ts.add_argument("--norad", type=int, required=True)

    rpo = sub.add_parser("assess-rpo", help="Run F2.1 RPO assessment for an actor/target pair")
    rpo.add_argument("--actor", type=int, required=True)
    rpo.add_argument("--target", type=int, required=True)

    cas = sub.add_parser("screen-conjunctions",
                          help="F2.2: cascade-screen the current catalog for candidate pairs")
    cas.add_argument("--limit", type=int, default=20, help="max candidate pairs to print")

    pc = sub.add_parser("estimate-pc", help="Run F3.1 Monte Carlo Pc for an actor/target pair")
    pc.add_argument("--actor", type=int, required=True)
    pc.add_argument("--target", type=int, required=True)

    rep = sub.add_parser("generate-report",
                          help="F4.1/F4.2: grounded analyst report + verification for an object")
    rep.add_argument("--norad", type=int, required=True)
    rep.add_argument("--rpo-target", type=int, default=None,
                      help="optional: also include an RPO assessment vs. this target")

    sg = sub.add_parser("screen-graph",
                         help="F3.2: RPO-screen one actor against many targets, "
                              "build the interaction graph, flag serial inspectors")
    sg.add_argument("--actor", type=int, required=True)
    sg.add_argument("--target", action="append", type=int, required=True,
                     help="candidate target norad_id; repeatable")
    sg.add_argument("--start", type=date.fromisoformat, required=True)
    sg.add_argument("--end", type=date.fromisoformat, required=True)

    w = sub.add_parser("watch", help="Autonomous watch agent: run a cycle, or manage the watchlist")
    w.add_argument("--refresh", action="store_true", help="pull fresh Celestrak TLEs before the cycle")
    w.add_argument("--add", type=int, default=None, help="add a NORAD id to the watchlist")
    w.add_argument("--remove", type=int, default=None, help="remove a NORAD id from the watchlist")
    w.add_argument("--list", action="store_true", help="show the watchlist and exit")
    w.add_argument("--no-brief", action="store_true", help="skip the LLM brief (faster)")

    args = p.parse_args()

    if args.cmd == "ingest-celestrak":
        from data.celestrak import ingest_groups
        print(ingest_groups(args.group))

    elif args.cmd == "ingest-satcat":
        from data.celestrak import ingest_satcat
        print(f"SATCAT rows upserted: {ingest_satcat()}")

    elif args.cmd == "ingest-historical":
        from data.spacetrack_client import ingest_historical
        print(ingest_historical(args.norad, args.start, args.end))

    elif args.cmd == "check-coverage":
        from data.spacetrack_client import check_archive_coverage
        counts = check_archive_coverage(args.norad, args.start, args.end)
        for nid, n in counts.items():
            flag = "" if n > 0 else "   <-- NO COVERAGE — do not build on this event"
            print(f"NORAD {nid}: {n} archived TLEs{flag}")

    elif args.cmd == "classify-regime":
        from data.store import get_latest_tle
        from core.elements import mean_elements
        from core.regime import classify_regime
        tle = get_latest_tle(args.norad)
        if tle is None:
            print(f"No stored TLE for {args.norad}. Ingest it first.")
            return
        print(classify_regime(mean_elements(tle)).summary())

    elif args.cmd == "predict-decay":
        from data.store import get_tles
        from detect.decay import predict_decay
        tles = get_tles(args.norad)
        p = predict_decay(tles)
        if p is None:
            print(f"Not enough TLE history for {args.norad} ({len(tles)} stored) to fit a decay trend.")
            return
        print(p.summary())

    elif args.cmd == "check-graveyard":
        from core.elements import mean_elements
        from data.store import get_latest_tle
        from detect.graveyard import assess_disposal, scan_geo_belt
        if args.norad is not None:
            tle = get_latest_tle(args.norad)
            if tle is None:
                print(f"No stored TLE for {args.norad}. Ingest it first.")
                return
            print(assess_disposal(mean_elements(tle)).summary())
        else:
            grouped = scan_geo_belt()
            order = ["COMPLIANT_GRAVEYARD", "NON_COMPLIANT_SHALLOW", "BELOW_GEO_DRIFT",
                     "OPERATIONAL_OR_ABANDONED_IN_PLACE", "NOT_GEO_BELT"]
            print("GEO disposal-compliance scan over the current live catalog:")
            for status in order:
                items = grouped.get(status, [])
                if items:
                    print(f"  {status}: {len(items)}")
            # surface the non-compliant ones explicitly — they're the point
            for status in ("NON_COMPLIANT_SHALLOW", "BELOW_GEO_DRIFT"):
                for a in grouped.get(status, [])[:5]:
                    print(f"    - {a.summary()}")

    elif args.cmd == "detect-maneuvers":
        from data.store import get_tles
        from detect.maneuver import detect_maneuvers
        tles = get_tles(args.norad)
        events, floor = detect_maneuvers(tles)
        if floor is None:
            print(f"Not enough history ({len(tles)} TLEs stored). Ingest more first.")
            return
        print(f"Noise floor: {floor:.2f} m/s over {len(tles)} TLEs")
        if not events:
            print("No anomalous maneuvers detected.")
        for e in events:
            print(e.summary())

    elif args.cmd == "pattern-of-life":
        from data.store import get_tles
        from detect.maneuver import detect_maneuvers
        from detect.pattern_of_life import pattern_of_life
        tles = get_tles(args.norad)
        events, floor = detect_maneuvers(tles)
        if floor is None:
            print(f"Not enough TLE history ({len(tles)} TLEs stored) to detect maneuvers.")
            return
        classifications, baseline = pattern_of_life(events)
        if baseline is None:
            print(f"Only {len(events)} maneuver detection(s) — need >= 3 campaigns to "
                  "establish a pattern. No baseline built.")
            return
        print(f"Baseline over {baseline.n_campaigns} maneuver campaigns: "
              f"typical burn {baseline.typical_dv_ms:.2f} m/s, "
              f"typical tempo {baseline.typical_tempo_days:.1f} days")
        for c in classifications:
            print(c.summary())

    elif args.cmd == "validate-breakup":
        from validate.run_benchmark import run_breakup_benchmark
        results = run_breakup_benchmark()
        for r in results:
            print(r.summary())
        n_pass = sum(1 for r in results if r.passed)
        print(f"\n{n_pass}/{len(results)} passed")

    elif args.cmd == "snapshot-catalog":
        from data.store import list_active_satcat_norad_ids, record_catalog_snapshot
        ids = list_active_satcat_norad_ids()
        if not ids:
            print("No SATCAT rows stored — run ingest-satcat first.")
            return
        ts = record_catalog_snapshot(ids)
        print(f"Recorded snapshot at {ts}: {len(ids)} active objects.")

    elif args.cmd == "check-catalog-delta":
        from data.store import get_snapshot_ids, list_snapshot_timestamps
        from detect.breakup import detect_from_new_ids
        stamps = list_snapshot_timestamps()
        if len(stamps) < 2:
            print(f"Need >= 2 snapshots to diff (have {len(stamps)}). "
                  "Run snapshot-catalog again after some time has passed.")
            return
        prev_ids = get_snapshot_ids(stamps[-2])
        curr_ids = get_snapshot_ids(stamps[-1])
        new_ids = curr_ids - prev_ids
        print(f"{stamps[-2]} -> {stamps[-1]}: {len(new_ids)} new object(s)")
        candidates = detect_from_new_ids(new_ids)
        if not candidates:
            print("No breakup candidates in this delta.")
        for c in candidates:
            print(c.summary())

    elif args.cmd == "threat-score":
        from score.threat import compute_threat_score
        print(compute_threat_score(args.norad).summary())

    elif args.cmd == "assess-rpo":
        from data.store import get_tles
        from detect.rpo import assess_rpo
        actor_tles = get_tles(args.actor)
        target_tles = get_tles(args.target)
        result = assess_rpo(args.actor, args.target, actor_tles, target_tles)
        if result is None:
            print(f"Not enough overlapping TLE coverage between {args.actor} and "
                  f"{args.target} to assess (actor has {len(actor_tles)}, "
                  f"target has {len(target_tles)} stored TLEs).")
            return
        print(result.summary())

    elif args.cmd == "screen-conjunctions":
        from data.store import get_latest_tle, list_current_objects
        from core.elements import mean_elements
        from detect.cascade import run_cascade
        current = list_current_objects()
        if not current:
            print("No live-tracked objects — run ingest-celestrak first.")
            return
        elements = []
        for c in current:
            tle = get_latest_tle(c["norad_id"])
            if tle is None:
                continue
            try:
                elements.append(mean_elements(tle))
            except RuntimeError:
                continue
        n = len(elements)
        candidates = run_cascade(elements)
        print(f"{n} objects, {n * (n - 1) // 2} possible pairs -> "
              f"{len(candidates)} candidates survived the cascade")
        for c in sorted(candidates, key=lambda c: c.relative_inclination_deg)[:args.limit]:
            print(c.summary())

    elif args.cmd == "estimate-pc":
        from data.store import get_satcat_many, get_tles
        from detect.conjunction import estimate_pc
        actor_tles = get_tles(args.actor)
        target_tles = get_tles(args.target)
        satcat = get_satcat_many([args.actor, args.target])
        result = estimate_pc(
            args.actor, args.target, actor_tles, target_tles,
            actor_rcs_m2=satcat.get(args.actor, {}).get("rcs_m2"),
            target_rcs_m2=satcat.get(args.target, {}).get("rcs_m2"),
        )
        if result is None:
            print(f"Not enough overlapping TLE coverage between {args.actor} and "
                  f"{args.target} to assess (actor has {len(actor_tles)}, "
                  f"target has {len(target_tles)} stored TLEs).")
            return
        print(result.summary())

    elif args.cmd == "screen-graph":
        from data.store import get_tles, list_high_value_asset_ids
        from detect.rpo import assess_rpo
        from detect.graph import build_interaction_graph, find_serial_inspectors
        actor_tles = get_tles(args.actor, start=args.start.isoformat(), end=args.end.isoformat())
        assessments = []
        for tid in args.target:
            target_tles = get_tles(tid, start=args.start.isoformat(), end=args.end.isoformat())
            r = assess_rpo(args.actor, tid, actor_tles, target_tles, step_hours=12.0)
            if r is None:
                print(f"NORAD {tid}: insufficient overlapping coverage, skipped")
                continue
            print(f"NORAD {tid}: {r.label} (min range {r.min_range_km:.1f} km, "
                  f"{r.n_episodes} episodes, {r.total_close_days:.1f}d close)")
            assessments.append(r)
        hva_ids = set(list_high_value_asset_ids())
        g = build_interaction_graph(assessments, hva_ids=hva_ids)
        print(f"\nGraph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        findings = find_serial_inspectors(g)
        if not findings:
            print("No serial-inspector pattern found among the screened targets.")
        for f in findings:
            print(f.summary())

    elif args.cmd == "watch":
        from data.store import add_to_watchlist, get_watchlist, remove_from_watchlist
        # management flags manage the watchlist and exit — they don't run a cycle
        if args.add is not None or args.remove is not None or args.list:
            if args.add is not None:
                add_to_watchlist(args.add)
                print(f"Added NORAD {args.add} to the watchlist.")
            if args.remove is not None:
                remove_from_watchlist(args.remove)
                print(f"Removed NORAD {args.remove} from the watchlist.")
            wl = get_watchlist()
            print(f"Watchlist ({len(wl)}): {', '.join(str(w['norad_id']) for w in wl) or 'empty'}")
            return
        from watch.agent import run_cycle
        if not get_watchlist():
            print("Watchlist is empty — add objects with `watch --add <norad>` first.")
            return
        print("Running watch cycle" + (" (refreshing from Celestrak)" if args.refresh else "") + "...")
        result = run_cycle(refresh=args.refresh, generate_brief=not args.no_brief)
        print(f"\nCycle {result.cycle_id} · {result.run_at} · {result.n_watched} watched · "
              f"{len(result.findings)} finding(s)")
        for f in result.findings:
            print(f"  [{f.significance.upper()}] {f.summary}")
        if not result.findings:
            print("  No changes since the last cycle.")
        if result.brief:
            print("\n--- BRIEF ---")
            print(result.brief)

    elif args.cmd == "generate-report":
        from core.elements import mean_elements
        from core.regime import classify_regime
        from data.store import get_latest_tle, get_satcat_many, get_tles
        from detect.decay import predict_decay
        from detect.graveyard import assess_disposal
        from detect.rpo import assess_rpo
        from report.facts import (
            build_report_facts, facts_from_decay, facts_from_disposal,
            facts_from_regime, facts_from_rpo, facts_from_threat_score,
        )
        from report.generate import generate_report
        from report.verify import verify_report
        from score.threat import compute_threat_score

        score = compute_threat_score(args.norad)
        fact_lists = [facts_from_threat_score(score)]

        # Bonus-feature context, folded into the same grounded report:
        # regime is always available; decay/disposal when applicable.
        latest = get_latest_tle(args.norad)
        if latest is not None:
            regime = classify_regime(mean_elements(latest))
            fact_lists.append(facts_from_regime(regime))
            if regime.primary in ("GEO", "GRAVEYARD"):
                fact_lists.append(facts_from_disposal(assess_disposal(mean_elements(latest))))
            decay = predict_decay(get_tles(args.norad))
            if decay is not None and decay.predicted_reentry_iso is not None:
                fact_lists.append(facts_from_decay(decay))

        if args.rpo_target is not None:
            actor_tles = get_tles(args.norad)
            target_tles = get_tles(args.rpo_target)
            rpo = assess_rpo(args.norad, args.rpo_target, actor_tles, target_tles)
            if rpo is not None:
                fact_lists.append(facts_from_rpo(rpo))
            else:
                print(f"(no RPO assessment: insufficient overlapping coverage with {args.rpo_target})")

        name = get_satcat_many([args.norad]).get(args.norad, {}).get("name")
        facts = build_report_facts(args.norad, name, *fact_lists)

        print("Loading local model (first run downloads ~1.6 GB)...")
        text = generate_report(facts)
        print("\n--- GENERATED REPORT ---")
        print(text)

        result = verify_report(text, facts)
        print("\n--- F4.2 VERIFICATION ---")
        print(result.summary())


if __name__ == "__main__":
    main()
