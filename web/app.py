"""DRISHTI interactive web console (multi-page Flask app).

Run:  python web/app.py   →   http://127.0.0.1:5000

Pages:
  /                 overview console (live catalog stats + section index)
  /catalog          searchable/filterable object browser (SATCAT)
  /orbit-map        altitude × inclination scatter of the catalogue (where + what kind)
  /object/<id>      per-object dossier — the detailed threat report: orbit
                    diagram, regime, elements, maneuvers, pattern-of-life,
                    threat score + factor breakdown, decay, disposal, RPO
                    range trace, and ranked decision-support mitigations
  /object/<id>/report   on-demand grounded LLM narrative + verification
  /object/<id>/refresh  pull the latest element set from Celestrak on demand
  /threat-board     objects with full behavioral data, ranked by threat (+ scatter)
  /conjunctions     cascade-screened close-approach candidates
  /disposal         GEO-belt disposal-compliance monitor (+ belt scatter)
  /events, /events/<key>   the labeled validation events as case pages
  /country          active objects rolled up by operator/nation

Every number shown traces to a computed value — same discipline as the
code and the CLI. Visualizations are hand-composed SVG (web/plots.py),
self-contained, no JS charting library.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
from pathlib import Path

# make the project root importable when run as `python web/app.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for  # noqa: E402

from config import DB_PATH, R_EARTH_KM  # noqa: E402
from data.store import get_latest_tle, get_satcat_many, get_tles, list_current_objects  # noqa: E402
from validate.events import EVENTS, by_key  # noqa: E402
from web import plots  # noqa: E402
from web.characterize import characterize  # noqa: E402

app = Flask(__name__)

# ------------------------------------------------------------------ helpers
REGIME_SQL = {
    "LEO":  "apogee_km < 2000",
    "MEO":  "perigee_km >= 2000 AND apogee_km < 34000",
    "GEO":  "apogee_km BETWEEN 34000 AND 38000 AND perigee_km > 33000",
    "HEO":  "apogee_km > 35000 AND perigee_km < 2000",
}
OBJECT_TYPES = ["PAY", "DEB", "R/B", "UNK"]

_COUNTRY_CACHE: list[str] | None = None
_TTL_CACHE: dict[str, tuple[float, object]] = {}
_TTL_S = float(os.environ.get("DRISHTI_CACHE_TTL_S", "600"))


def _ttl_cached(key: str, fn):
    """Catalogue-wide scans (conjunctions, disposal) are identical between
    Celestrak refreshes — reuse the result for _TTL_S seconds."""
    import time
    hit = _TTL_CACHE.get(key)
    if hit is None or time.monotonic() - hit[0] > _TTL_S:
        hit = (time.monotonic(), fn())
        _TTL_CACHE[key] = hit
    return hit[1]


def _top_countries(n: int = 20) -> list[str]:
    """Return the top-N operator/country codes by active object count."""
    global _COUNTRY_CACHE
    if _COUNTRY_CACHE is None:
        with _conn() as c:
            rows = c.execute(
                "SELECT country, COUNT(*) n FROM satcat "
                "WHERE decay_date IS NULL AND country IS NOT NULL "
                "GROUP BY country ORDER BY n DESC LIMIT ?", (n,)
            ).fetchall()
        _COUNTRY_CACHE = [r["country"] for r in rows]
    return _COUNTRY_CACHE


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def coarse_regime(row) -> str:
    apo, per = row["apogee_km"], row["perigee_km"]
    if apo is None or per is None:
        return "—"
    if apo < 2000:
        return "LEO"
    if per >= 2000 and apo < 34000:
        return "MEO"
    if 34000 <= apo <= 38000 and per > 33000:
        return "GEO"
    if apo > 35000 and per < 2000:
        return "HEO"
    return "OTHER"


def featured_norads() -> list[int]:
    """Objects rich enough for the full behavioral pipeline: those with
    real TLE history, plus every validation-event actor/target."""
    ids: set[int] = set()
    with _conn() as c:
        for r in c.execute("SELECT norad_id, COUNT(*) n FROM tle GROUP BY norad_id HAVING n>=8"):
            ids.add(r["norad_id"])
    for e in EVENTS:
        ids.update(e.actor_norad)
        ids.update(e.target_norad)
    # keep only ones that actually have some history stored
    return sorted(nid for nid in ids if get_tles(nid))


_ANOMALY_MODEL = None


def anomaly_model():
    """Lazily fit an IsolationForest over the active catalogue's orbital
    features, once, and cache it for the process lifetime."""
    global _ANOMALY_MODEL
    if _ANOMALY_MODEL is None:
        from detect.anomaly import AnomalyModel
        with _conn() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT norad_id,apogee_km,perigee_km,inclination,period_min FROM satcat "
                "WHERE decay_date IS NULL AND apogee_km IS NOT NULL AND perigee_km IS NOT NULL "
                "AND inclination IS NOT NULL")]
        _ANOMALY_MODEL = AnomalyModel().fit(rows)
    return _ANOMALY_MODEL


def catalog_stats() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM satcat").fetchone()[0]
        active = c.execute("SELECT COUNT(*) FROM satcat WHERE decay_date IS NULL").fetchone()[0]
        payloads = c.execute("SELECT COUNT(*) FROM satcat WHERE object_type='PAY' AND decay_date IS NULL").fetchone()[0]
        debris = c.execute("SELECT COUNT(*) FROM satcat WHERE object_type='DEB' AND decay_date IS NULL").fetchone()[0]
    return {"total": total, "active": active, "payloads": payloads, "debris": debris,
            "featured": len(featured_norads()), "events": len(EVENTS)}


# ------------------------------------------------------------------- routes
@app.route("/")
def overview():
    return render_template("overview.html", stats=catalog_stats(), nav="overview")


@app.route("/catalog")
def catalog():
    q = (request.args.get("q") or "").strip()
    regime = request.args.get("regime") or ""
    obj_type = request.args.get("type") or ""
    country = request.args.get("country") or ""
    page = max(1, int(request.args.get("page", 1)))
    per_page = 60

    where, params = ["1=1"], []
    if q:
        if q.isdigit():
            where.append("norad_id = ?"); params.append(int(q))
        else:
            where.append("name LIKE ?"); params.append(f"%{q.upper()}%")
    if regime in REGIME_SQL:
        where.append(REGIME_SQL[regime])
    if obj_type in OBJECT_TYPES:
        where.append("object_type = ?"); params.append(obj_type)
    if country:
        where.append("country = ?"); params.append(country)
    where.append("decay_date IS NULL")
    clause = " AND ".join(where)

    with _conn() as c:
        total = c.execute(f"SELECT COUNT(*) FROM satcat WHERE {clause}", params).fetchone()[0]
        rows = c.execute(
            f"SELECT norad_id,name,country,object_type,apogee_km,perigee_km,inclination "
            f"FROM satcat WHERE {clause} ORDER BY norad_id LIMIT ? OFFSET ?",
            (*params, per_page, (page - 1) * per_page),
        ).fetchall()

    items = [{**dict(r), "regime": coarse_regime(r)} for r in rows]
    return render_template("catalog.html", items=items, q=q, regime=regime,
                           obj_type=obj_type, country=country,
                           regimes=list(REGIME_SQL), obj_types=OBJECT_TYPES,
                           countries=_top_countries(),
                           page=page, per_page=per_page,
                           total=total, pages=(total + per_page - 1) // per_page, nav="catalog")


@app.route("/query")
def nl_query():
    q = (request.args.get("q") or "").strip()
    if not q:
        return render_template("query.html", q="", spec=None, results=[], nav="query")
    from query.nlquery import QuerySpec, parse_query
    spec = parse_query(q)
    used_llm = spec is not None
    if spec is None:  # LLM produced no parseable filter → fall back to a name search
        spec = QuerySpec(name_contains=q, limit=50)
    where, params = spec.where()
    clause = " AND ".join(where) if where else "1=1"
    with _conn() as c:
        rows = [{**dict(r), "regime": coarse_regime(r)} for r in c.execute(
            f"SELECT norad_id,name,country,object_type,apogee_km,perigee_km,inclination "
            f"FROM satcat WHERE {clause} LIMIT 3000", params)]
    if spec.sort == "anomaly":
        scores = {s.norad_id: s.score for s in anomaly_model().all_scores()}
        rows.sort(key=lambda r: scores.get(r["norad_id"], -1.0), reverse=True)
        for r in rows:
            r["anomaly"] = scores.get(r["norad_id"])
    elif spec.sort == "altitude_desc":
        rows.sort(key=lambda r: (r["apogee_km"] or 0) + (r["perigee_km"] or 0), reverse=True)
    elif spec.sort == "altitude_asc":
        rows.sort(key=lambda r: (r["apogee_km"] or 0) + (r["perigee_km"] or 0))
    from report.llm import available
    return render_template("query.html", q=q, spec=spec, results=rows[:spec.limit],
                           used_llm=used_llm, llm_off=not available(), total=len(rows), nav="query")


@app.route("/object/<int:norad_id>")
def object_dossier(norad_id: int):
    data = characterize(norad_id)
    if data["satcat"] is None and not data["has_tle"]:
        abort(404)
    # Most catalogue objects have SATCAT metadata but no stored element set,
    # so a dossier would be empty. If the object is active (not decayed),
    # pull its current TLE from Celestrak on demand, then re-characterize —
    # this makes every clickable catalogue row populate on first open, and
    # caches the TLE so later opens are instant.
    fetched = False
    if not data["has_tle"] and data["satcat"] and not data["satcat"].get("decay_date"):
        from data.celestrak import ingest_object
        try:
            if ingest_object(norad_id):
                data = characterize(norad_id)
                fetched = True
        except Exception:
            pass
    svg = {}
    orbit3d = None
    if data["elements"] is not None:
        svg["orbit"] = plots.orbit_diagram(data["elements"])
        orbit3d = {"earth_radius_km": R_EARTH_KM, "orbits": [_orbit3d_entry(data["elements"], 0xE83A00)]}
        # add the RPO target's orbit so the encounter shows in 3D
        if data.get("rpo") is not None:
            from core.elements import mean_elements
            ttle = get_latest_tle(data["rpo"].target_norad)
            if ttle is not None:
                orbit3d["orbits"].append(_orbit3d_entry(mean_elements(ttle), 0x1A4F9C))
    if data.get("rpo_series"):
        svg["range"] = plots.range_trace(
            data["rpo_series"]["days"], data["rpo_series"]["ranges_km"])
    anomaly = None
    anomaly_n = 0
    if data["satcat"] is not None:
        try:
            am = anomaly_model()
            anomaly = am.score_row(data["satcat"])
            anomaly_n = am.size
        except Exception:
            pass
    posture = None
    if data["mitigations"]:
        from score.mitigations import response_posture
        posture = response_posture(data["mitigations"])
    from data.store import is_watched
    return render_template("dossier.html", d=data, svg=svg, orbit3d=orbit3d,
                           anomaly=anomaly, anomaly_n=anomaly_n, posture=posture,
                           fetched=fetched, watched=is_watched(norad_id), nav="")


def _orbit3d_entry(el, color: int) -> dict:
    return {"a": el.semi_major_km, "e": el.ecc, "i": el.incl_deg,
            "raan": el.raan_deg, "argp": el.argp_deg, "color": color}


@app.route("/object/<int:norad_id>/refresh")
def object_refresh(norad_id: int):
    from data.celestrak import ingest_object
    try:
        ingest_object(norad_id)
    except Exception:
        pass
    return redirect(url_for("object_dossier", norad_id=norad_id))


@app.route("/object/<int:norad_id>/report")
def object_report(norad_id: int):
    from report.facts import (build_report_facts, facts_from_decay, facts_from_disposal,
                              facts_from_regime, facts_from_rpo, facts_from_threat_score)
    from report.generate import generate_report
    from report.verify import verify_report

    d = characterize(norad_id)
    if d["threat"] is None:
        abort(404)
    fact_lists = [facts_from_threat_score(d["threat"]), facts_from_regime(d["regime"])]
    if d["disposal"] is not None:
        fact_lists.append(facts_from_disposal(d["disposal"]))
    if d["decay"] is not None:
        fact_lists.append(facts_from_decay(d["decay"]))
    if d["rpo"] is not None:
        fact_lists.append(facts_from_rpo(d["rpo"]))
    facts = build_report_facts(norad_id, d["name"], *fact_lists)
    from report.llm import available
    if not available():
        # no model on this host: show the grounded facts the model WOULD narrate
        return render_template("report.html", d=d, text=None, verify=None,
                               facts_block=facts.as_prompt_block(), nav="")
    text = generate_report(facts)
    result = verify_report(text, facts)
    return render_template("report.html", d=d, text=text, verify=result, nav="")


_SIG_RANK = {"priority": 3, "elevated": 2, "routine": 1}


@app.route("/watch")
def watch_floor():
    from data.store import get_cycle_findings, get_last_object_states, get_watch_cycles, get_watchlist
    watchlist = get_watchlist()
    names = get_satcat_many([w["norad_id"] for w in watchlist])
    cycles = get_watch_cycles(limit=15)
    latest = cycles[0] if cycles else None
    findings = get_cycle_findings(latest["cycle_id"]) if latest else []
    findings.sort(key=lambda f: _SIG_RANK.get(f["significance"], 0), reverse=True)
    states = get_last_object_states()
    return render_template("watch.html", watchlist=watchlist, names=names, cycles=cycles,
                           latest=latest, findings=findings, states=states, nav="watch")


@app.route("/watch/run")
def watch_run():
    from watch.agent import run_cycle
    refresh = request.args.get("refresh") == "1"
    run_cycle(refresh=refresh, generate_brief=True)
    return redirect(url_for("watch_floor"))


@app.route("/object/<int:norad_id>/watch")
def object_watch_toggle(norad_id: int):
    from data.store import add_to_watchlist, is_watched, remove_from_watchlist
    if is_watched(norad_id):
        remove_from_watchlist(norad_id)
    else:
        add_to_watchlist(norad_id)
    return redirect(url_for("object_dossier", norad_id=norad_id))


def _ranked_featured() -> list[dict]:
    ranked = []
    for nid in featured_norads():
        d = characterize(nid)
        if d["threat"] is None or d["threat"].score is None:
            continue
        ranked.append(d)
    ranked.sort(key=lambda d: d["threat"].score, reverse=True)
    return ranked


@app.route("/threat-board")
def threat_board():
    ranked = _ttl_cached("threat_board", _ranked_featured)
    scatter = plots.threat_scatter(ranked)
    return render_template("threatboard.html", items=ranked, scatter=scatter, nav="threat")


@app.route("/orbit-map")
def orbit_map():
    regime = request.args.get("regime") or ""
    where = ["decay_date IS NULL", "apogee_km IS NOT NULL", "perigee_km IS NOT NULL",
             "inclination IS NOT NULL", "object_type='PAY'"]
    if regime in REGIME_SQL:
        where.append(REGIME_SQL[regime])
    with _conn() as c:
        rows = c.execute(
            f"SELECT norad_id,name,apogee_km,perigee_km,inclination FROM satcat "
            f"WHERE {' AND '.join(where)} ORDER BY norad_id",
        ).fetchall()
    # sample to keep the scatter legible/fast
    step = max(1, len(rows) // 2200)
    objs = [{**dict(r), "regime": coarse_regime(r)} for r in rows[::step]]
    counts: dict[str, int] = {}
    for o in objs:
        counts[o["regime"]] = counts.get(o["regime"], 0) + 1
    return render_template("orbitmap.html", scatter=plots.orbit_scatter(objs),
                           bars=plots.regime_bars(counts), n=len(objs), total=len(rows),
                           regime=regime, regimes=list(REGIME_SQL), nav="orbit")


@app.route("/disposal")
def disposal():
    from detect.graveyard import scan_geo_belt
    grouped = _ttl_cached("disposal", scan_geo_belt)
    flat = [a for lst in grouped.values() for a in lst if a.status != "NOT_GEO_BELT"]
    order = ["COMPLIANT_GRAVEYARD", "NON_COMPLIANT_SHALLOW", "BELOW_GEO_DRIFT",
             "INCLINED_GEOSYNCHRONOUS", "OPERATIONAL_OR_ABANDONED_IN_PLACE"]
    counts = {s: len(grouped.get(s, [])) for s in order}
    flagged = sorted((a for a in flat if a.compliant is False),
                     key=lambda a: a.km_above_geo)
    names = get_satcat_many([a.norad_id for a in flagged])
    # the belt scatter is the ±20° protected-region view; high-inclination
    # IGSO sits off that scale by design, so it's shown as a count, not a dot
    scatter_items = [a for a in flat if a.inclination_deg <= 20]
    return render_template("disposal.html", scatter=plots.belt_scatter(scatter_items),
                           counts=counts, flagged=flagged, names=names, nav="disposal")


@app.route("/events")
def events_index():
    return render_template("events.html", events=EVENTS, nav="events")


@app.route("/events/<key>")
def event_detail(key):
    try:
        ev = by_key(key)
    except KeyError:
        abort(404)
    actor = ev.actor_norad[0] if ev.actor_norad else None
    d = characterize(actor) if actor else None
    svg = {}
    if d and d.get("rpo_series"):
        svg["range"] = plots.range_trace(d["rpo_series"]["days"], d["rpo_series"]["ranges_km"])
    names = get_satcat_many(list(ev.actor_norad) + list(ev.target_norad))
    return render_template("event_detail.html", ev=ev, d=d, svg=svg, names=names, nav="events")


@app.route("/anomalies")
def anomalies():
    model = anomaly_model()
    top = model.all_scores()[:40]
    names = get_satcat_many([a.norad_id for a in top])
    items = [{"a": a, "row": names.get(a.norad_id, {})} for a in top]
    return render_template("anomalies.html", items=items,
                           total=len(model.all_scores()), nav="anomalies")


@app.route("/country")
def country_rollup():
    with _conn() as c:
        rows = c.execute(
            "SELECT country, COUNT(*) n, "
            "SUM(CASE WHEN object_type='PAY' THEN 1 ELSE 0 END) pay, "
            "SUM(CASE WHEN object_type='DEB' THEN 1 ELSE 0 END) deb, "
            "SUM(CASE WHEN apogee_km BETWEEN 34000 AND 38000 AND perigee_km>33000 THEN 1 ELSE 0 END) geo "
            "FROM satcat WHERE decay_date IS NULL AND country IS NOT NULL AND country != '' "
            "GROUP BY country ORDER BY n DESC LIMIT 30"
        ).fetchall()
    return render_template("country.html", rows=[dict(r) for r in rows], nav="country")


@app.route("/conjunctions")
def conjunctions():
    from core.elements import mean_elements
    from data.store import get_latest_tles_many, list_current_objects
    from detect.cascade import screen_top

    def _screen():
        elements = []
        current = list_current_objects()
        tles = get_latest_tles_many(c["norad_id"] for c in current)
        for c in current:
            tle = tles.get(c["norad_id"])
            if tle is None:
                continue
            try:
                elements.append(mean_elements(tle))
            except Exception:
                continue
        # vectorized cascade: at full-catalogue scale run_cascade never returns
        return (len(elements), *screen_top(elements, top_k=60))

    n, total, cands = _ttl_cached("conjunctions", _screen)
    names = get_satcat_many([p.norad_a for p in cands] + [p.norad_b for p in cands])
    return render_template("conjunctions.html", cands=cands, total=total,
                           n=n, pairs=n * (n - 1) // 2, names=names, nav="conjunctions")


# ------------------------------------------------------------------- JSON API
# Makes the whole analytic core programmatically consumable (the "APIs /
# workflows" layer). Every field is a computed value — same discipline.

@app.route("/api/object/<int:norad_id>")
def api_object(norad_id: int):
    from web.serialize import object_json
    d = characterize(norad_id)
    if d["satcat"] is None and not d["has_tle"]:
        abort(404)
    return jsonify(object_json(d))


@app.route("/api/threat-board")
def api_threat_board():
    out = []
    for d in _ttl_cached("threat_board", _ranked_featured):
        t, nid = d["threat"], d["norad_id"]
        out.append({"norad_id": nid, "name": d["name"], "threat_score": t.score,
                    "regime": d["regime"].label,
                    "top_mitigation": d["mitigations"][0].title if d["mitigations"] else None,
                    "top_urgency": d["mitigations"][0].urgency if d["mitigations"] else None})
    out.sort(key=lambda x: x["threat_score"], reverse=True)
    return jsonify({"count": len(out), "objects": out})


@app.route("/api/query")
def api_query():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "provide ?q=<question>"}), 400
    from query.nlquery import QuerySpec, parse_query
    spec = parse_query(q)
    used_llm = spec is not None
    if spec is None:
        spec = QuerySpec(name_contains=q, limit=50)
    where, params = spec.where()
    clause = " AND ".join(where) if where else "1=1"
    with _conn() as c:
        rows = [{**dict(r), "regime": coarse_regime(r)} for r in c.execute(
            f"SELECT norad_id,name,country,object_type,apogee_km,perigee_km,inclination "
            f"FROM satcat WHERE {clause} LIMIT ?", (*params, spec.limit))]
    return jsonify({"query": q, "parsed_by_llm": used_llm, "interpretation": spec.describe(),
                    "count": len(rows), "results": rows})


@app.route("/api/watch/latest")
def api_watch_latest():
    from data.store import get_cycle_findings, get_watch_cycles
    cycles = get_watch_cycles(limit=1)
    if not cycles:
        return jsonify({"cycles": 0, "findings": []})
    latest = cycles[0]
    findings = get_cycle_findings(latest["cycle_id"])
    return jsonify({"cycle_id": latest["cycle_id"], "run_at": latest["run_at"],
                    "brief": latest["brief"], "findings": findings})


@app.route("/api/conjunction/<int:a>/<int:b>.cdm")
def api_conjunction_cdm(a: int, b: int):
    from detect.conjunction import estimate_pc
    from web.serialize import conjunction_cdm
    ta, tb = get_tles(a), get_tles(b)
    satcat = get_satcat_many([a, b])
    ra = satcat.get(a, {}).get("rcs_m2")
    rb = satcat.get(b, {}).get("rcs_m2")
    result = estimate_pc(a, b, ta, tb, actor_rcs_m2=ra, target_rcs_m2=rb)
    if result is None:
        abort(404)
    cdm = conjunction_cdm(result, satcat.get(a, {}).get("name", str(a)),
                          satcat.get(b, {}).get("name", str(b)))
    return Response(cdm, mimetype="text/plain")


@app.template_filter("pct")
def pct(v):
    return f"{v*100:.0f}" if v is not None else "—"


def _prewarm() -> None:
    """Fill the process caches (anomaly model, catalogue-wide scans, threat
    board) in the background so the first visitor on a small host doesn't
    pay for them. Opt-in via DRISHTI_PREWARM=1 (set on the Render deploy)."""
    with app.test_client() as c:
        for url in ("/anomalies", "/threat-board", "/disposal", "/conjunctions"):
            try:
                c.get(url)
            except Exception as e:  # never let warm-up take the server down
                app.logger.warning("prewarm %s failed: %s", url, e)


if os.environ.get("DRISHTI_PREWARM") == "1":
    threading.Thread(target=_prewarm, daemon=True, name="drishti-prewarm").start()


if __name__ == "__main__":
    app.run(debug=False, port=5000)
