"""Server-side SVG plot generators for the web console.

Self-contained: every chart is hand-composed SVG with inline colours from
the telemetry-plate palette and native <title> tooltips — no JS charting
library, no external requests, works offline (matches the project ethos).
Coordinates are computed in Python; templates inject the returned markup
with |safe.

All plots answer the three questions the analyst asks: WHERE an object is
(orbit diagram, altitude scatter), HOW CLOSE it got (relative-range
trace), and WHAT KIND it is (regime colouring, belt compliance).
"""
from __future__ import annotations

import math
from html import escape

# palette (kept in sync with web/static/app.css)
INK = "#141d27"; SLATE = "#5f6b78"; HAIR = "#c8d0d9"; HAIRSOFT = "#dde3e9"
SIGNAL = "#e83a00"; REF = "#1a4f9c"; OK = "#1d7d55"; WARN = "#b5730a"
PLATE = "#f7f8fa"

REGIME_COLOR = {
    "LEO": REF, "MEO": "#6a8fc7", "GEO": SIGNAL, "GRAVEYARD": WARN,
    "HEO": "#8a5cc0", "GTO": "#c05c8a", "OTHER": SLATE, "—": SLATE, "DECAYING": SIGNAL,
}
STATUS_COLOR = {
    "COMPLIANT_GRAVEYARD": OK, "NON_COMPLIANT_SHALLOW": WARN,
    "BELOW_GEO_DRIFT": SIGNAL, "OPERATIONAL_OR_ABANDONED_IN_PLACE": REF,
    "INCLINED_GEOSYNCHRONOUS": "#3a9c8c", "NOT_GEO_BELT": SLATE,
}
R_EARTH = 6378.137
GEO_R = 42164.0


def _log_alt_y(alt_km, y_top, y_bot, lo=2.0, hi=5.0):
    """Map altitude (km, log scale) to a y pixel; higher alt = higher up."""
    a = max(alt_km, 10.0)
    f = (math.log10(a) - lo) / (hi - lo)
    f = min(1.0, max(0.0, f))
    return y_bot - f * (y_bot - y_top)


# ---------------------------------------------------------------- orbit scatter
def orbit_scatter(objects, w=940, h=460):
    """Altitude (log) × inclination scatter of the catalogue — 'where +
    what kind'. `objects`: dicts with apogee_km, perigee_km, inclination,
    regime, name, norad_id."""
    ml, mr, mt, mb = 58, 84, 26, 40
    x0, x1, y0, y1 = ml, w - mr, mt, h - mb
    incl_max = 140.0

    def xpos(incl):
        return x0 + min(max(incl, 0), incl_max) / incl_max * (x1 - x0)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Catalogue altitude versus inclination scatter">']
    # altitude gridlines / regime bands
    for alt, lab in [(400, "LEO"), (2000, ""), (20200, "MEO"), (35786, "GEO"), (100000, "")]:
        y = _log_alt_y(alt, y0, y1)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{HAIRSOFT}" stroke-width="1"/>')
        parts.append(f'<text x="{x1+6}" y="{y+3:.1f}" font-family="IBM Plex Mono,monospace" font-size="10.5" fill="{SLATE}">{alt:,} km</text>')
        if lab:
            parts.append(f'<text x="{x0+4}" y="{y-5:.1f}" font-family="IBM Plex Mono,monospace" font-size="10.5" fill="{SLATE}" font-weight="600">{lab}</text>')
    # inclination ticks
    for inc in (0, 28, 52, 63, 90, 98, 120):
        x = xpos(inc)
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{HAIRSOFT}" stroke-width="1" opacity="0.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{y1+16}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}">{inc}°</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{h-4}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="10.5" fill="{SLATE}">inclination</text>')

    for o in objects:
        apo, per, inc = o.get("apogee_km"), o.get("perigee_km"), o.get("inclination")
        if apo is None or per is None or inc is None:
            continue
        mean_alt = (apo + per) / 2
        col = REGIME_COLOR.get(o.get("regime", "—"), SLATE)
        x, y = xpos(inc), _log_alt_y(mean_alt, y0, y1)
        r = 2.4 if not o.get("hi") else 4.2
        op = 0.42 if not o.get("hi") else 0.95
        title = escape(f'{o.get("name","")} ({o.get("norad_id","")}) — {o.get("regime","")}, {mean_alt:,.0f} km, {inc:.1f}°')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}" opacity="{op}"><title>{title}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- orbit diagram
def orbit_diagram(el, w=360, h=300):
    """Side-view of one orbit: Earth at a focus, the object's ellipse, and
    a dashed GEO reference ring — 'where this object lives'."""
    a = el.semi_major_km; e = el.ecc
    r_apo = a * (1 + e); r_per = a * (1 - e)
    b = a * math.sqrt(max(1 - e * e, 1e-6))
    c = a * e
    cx, cy = w / 2, h / 2
    max_r = max(r_apo, GEO_R * 1.02)
    scale = (min(w, h) / 2 - 24) / max_r

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Orbit diagram">']
    # GEO reference ring
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{GEO_R*scale:.1f}" fill="none" stroke="{HAIR}" stroke-width="1" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{cx}" y="{cy - GEO_R*scale - 4:.1f}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="9.5" fill="{SLATE}">GEO</text>')
    # orbit ellipse — Earth at right focus, so shift centre left by c
    ex = cx - c * scale
    parts.append(f'<ellipse cx="{ex:.1f}" cy="{cy}" rx="{a*scale:.1f}" ry="{b*scale:.1f}" fill="none" stroke="{SIGNAL}" stroke-width="1.8"/>')
    # Earth at focus
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{max(R_EARTH*scale,4):.1f}" fill="{REF}" opacity="0.85"/>')
    # perigee / apogee markers — labels point INWARD so they never clip the edge
    px = cx + r_per * scale  # perigee to the right of focus
    ax = cx - r_apo * scale
    parts.append(f'<circle cx="{px:.1f}" cy="{cy}" r="3" fill="{SIGNAL}"/>')
    parts.append(f'<text x="{px-8:.1f}" y="{cy-8}" text-anchor="end" font-family="IBM Plex Mono,monospace" font-size="9.5" fill="{SLATE}">per {el.perigee_km:,.0f} km</text>')
    parts.append(f'<circle cx="{ax:.1f}" cy="{cy}" r="3" fill="{SIGNAL}"/>')
    parts.append(f'<text x="{ax+8:.1f}" y="{cy-8}" text-anchor="start" font-family="IBM Plex Mono,monospace" font-size="9.5" fill="{SLATE}">apo {el.apogee_km:,.0f} km</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------- range trace
def range_trace(days, ranges_km, w=920, h=320, close_km=100.0):
    """Log relative-range vs time — the RPO 'how close' signature. `days`
    = float days from start, `ranges_km` aligned."""
    if not days:
        return ""
    ml, mr, mt, mb = 40, 66, 26, 30
    x0, x1, y0, y1 = ml, w - mr, mt, h - mb
    dmax = max(days) or 1.0

    def xp(d):
        return x0 + d / dmax * (x1 - x0)

    def yp(r):
        return _log_alt_y(max(r, 0.05), y0, y1, lo=-1.0, hi=5.0)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Relative range over time">']
    for r, lab in [(100000, "100k"), (10000, "10k"), (1000, "1k"), (100, "100"), (10, "10"), (1, "1 km")]:
        y = yp(r)
        if y0 <= y <= y1:
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{HAIRSOFT}" stroke-width="1"/>')
            parts.append(f'<text x="{x1+6}" y="{y+3:.1f}" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}">{lab}</text>')
    # close-approach band (below close_km)
    yb = yp(close_km)
    parts.append(f'<rect x="{x0}" y="{yb:.1f}" width="{x1-x0}" height="{y1-yb:.1f}" fill="{SIGNAL}" opacity="0.07"/>')
    parts.append(f'<text x="{x0+6}" y="{y1-6}" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}">&lt; {close_km:.0f} km</text>')
    # trace
    pts = " ".join(f"{xp(d):.1f},{yp(r):.1f}" for d, r in zip(days, ranges_km))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{SIGNAL}" stroke-width="2" stroke-linejoin="round"/>')
    # min marker
    imin = min(range(len(ranges_km)), key=lambda i: ranges_km[i])
    mr_km = ranges_km[imin]
    lab = f"{mr_km*1000:.0f} m" if mr_km < 1 else f"{mr_km:.1f} km"
    parts.append(f'<circle cx="{xp(days[imin]):.1f}" cy="{yp(mr_km):.1f}" r="4" fill="none" stroke="{SIGNAL}" stroke-width="1.4"/>')
    parts.append(f'<text x="{xp(days[imin]):.1f}" y="{yp(mr_km)-9:.1f}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="10.5" fill="{INK}" font-weight="600">min {lab}</text>')
    parts.append(f'<text x="{x0}" y="{h-4}" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}">day 0</text>')
    parts.append(f'<text x="{x1}" y="{h-4}" text-anchor="end" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}">day {dmax:.0f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------- belt scatter
def belt_scatter(assessments, w=920, h=420):
    """GEO-belt disposal map: inclination × km-above-GEO, coloured by
    disposal status. Reference lines at GEO (0) and the IADC floor (+235)."""
    ml, mr, mt, mb = 66, 60, 26, 40
    x0, x1, y0, y1 = ml, w - mr, mt, h - mb
    incl_max = 20.0
    lo, hi = -400.0, 600.0  # km above GEO range

    def xp(inc):
        return x0 + min(max(inc, 0), incl_max) / incl_max * (x1 - x0)

    def yp(km):
        f = (min(max(km, lo), hi) - lo) / (hi - lo)
        return y1 - f * (y1 - y0)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="GEO belt disposal scatter">']
    for km, lab, col in [(235, "IADC re-orbit floor", OK), (0, "GEO", INK), (-200, "protected region", SIGNAL)]:
        y = yp(km)
        dash = "" if km == 0 else 'stroke-dasharray="4 3"'
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{col}" stroke-width="1" {dash} opacity="0.6"/>')
        parts.append(f'<text x="{x1+6}" y="{y+3:.1f}" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}">{km:+d} km</text>')
        parts.append(f'<text x="{x0+4}" y="{y-4:.1f}" font-family="IBM Plex Mono,monospace" font-size="9.5" fill="{SLATE}">{lab}</text>')
    for inc in (0, 5, 10, 15, 20):
        x = xp(inc)
        parts.append(f'<text x="{x:.1f}" y="{y1+16}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}">{inc}°</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{h-4}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="10.5" fill="{SLATE}">inclination</text>')
    for a in assessments:
        col = STATUS_COLOR.get(a.status, SLATE)
        x, y = xp(a.inclination_deg), yp(a.km_above_geo)
        title = escape(f'{a.norad_id} — {a.status} · {a.km_above_geo:+.0f} km · {a.inclination_deg:.1f}°')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{col}" opacity="0.8"><title>{title}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------- threat scatter
def threat_scatter(items, w=940, h=380):
    """Threat score × mean altitude (log), dot colour by regime — 'where
    the threat concentrates'. `items`: characterization dicts."""
    ml, mr, mt, mb = 58, 84, 26, 40
    x0, x1, y0, y1 = ml, w - mr, mt, h - mb

    def xp(score):
        return x0 + min(max(score, 0), 100) / 100 * (x1 - x0)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Threat versus altitude scatter">']
    for alt, lab in [(400, "LEO"), (20200, "MEO"), (35786, "GEO"), (100000, "")]:
        y = _log_alt_y(alt, y0, y1)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{HAIRSOFT}" stroke-width="1"/>')
        parts.append(f'<text x="{x1+6}" y="{y+3:.1f}" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}">{alt:,} km</text>')
        if lab:
            parts.append(f'<text x="{x0+4}" y="{y-4:.1f}" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}" font-weight="600">{lab}</text>')
    for s in (0, 25, 50, 75, 100):
        x = xp(s)
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{HAIRSOFT}" stroke-width="1" opacity="0.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{y1+16}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="10" fill="{SLATE}">{s}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{h-4}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="10.5" fill="{SLATE}">composite threat score →</text>')
    for d in items:
        el = d.get("elements"); th = d.get("threat")
        if el is None or th is None or th.score is None:
            continue
        mean_alt = (el.apogee_km + el.perigee_km) / 2
        col = REGIME_COLOR.get(d["regime"].primary, SLATE)
        x, y = xp(th.score), _log_alt_y(mean_alt, y0, y1)
        title = escape(f'{d["name"]} ({d["norad_id"]}) — threat {th.score:.0f}, {d["regime"].primary}')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" opacity="0.85"><title>{title}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------- regime bars
def regime_bars(counts, w=920, h=120):
    total = sum(counts.values()) or 1
    order = ["LEO", "MEO", "GEO", "HEO", "GTO", "OTHER"]
    items = [(k, counts.get(k, 0)) for k in order if counts.get(k, 0)]
    x = 0.0
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Regime distribution">']
    bw = w
    for k, n in items:
        seg = n / total * bw
        col = REGIME_COLOR.get(k, SLATE)
        parts.append(f'<rect x="{x:.1f}" y="18" width="{max(seg-2,1):.1f}" height="30" rx="1.5" fill="{col}" opacity="0.9"><title>{k}: {n:,} ({n/total*100:.0f}%)</title></rect>')
        if seg > 60:  # only label segments wide enough to hold the text
            parts.append(f'<text x="{x+8:.1f}" y="37" font-family="IBM Plex Mono,monospace" font-size="11.5" fill="{PLATE}" font-weight="600">{k} {n:,}</text>')
        x += seg
    # compact legend row below, evenly spaced (no per-segment collision)
    lx = 0.0
    for k, n in items:
        col = REGIME_COLOR.get(k, SLATE)
        parts.append(f'<circle cx="{lx+5:.1f}" cy="66" r="4" fill="{col}"/>')
        parts.append(f'<text x="{lx+14:.1f}" y="70" font-family="IBM Plex Mono,monospace" font-size="11" fill="{SLATE}">{k} {n:,}</text>')
        lx += 150
    parts.append("</svg>")
    return "".join(parts)
