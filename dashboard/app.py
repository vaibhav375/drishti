"""DRISHTI dashboard skeleton (F0.4) — DEPRECATED.

⚠ Superseded by the multi-page Flask web console (`python web/app.py`),
which surfaces the full pipeline (F1.1–F4.2 + bonus + roadmap features).
This Streamlit skeleton only ever wired up F0.4/F1.1 and is kept for
history. Use the Flask console instead.

Run (legacy):  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import R_EARTH_KM
from core.elements import mean_elements
from core.groundtrack import ground_track
from core.propagation import propagate_window
from data.store import get_latest_tle, get_tles, list_objects
from detect.maneuver import detect_maneuvers

st.set_page_config(page_title="DRISHTI — SDA", layout="wide")
st.title("DRISHTI — Space Domain Awareness")
st.caption(
    "Behavioral characterization from public TLE data. "
    "Decision-support / situational awareness — not targeting."
)

objects = list_objects()
if not objects:
    st.warning(
        "No TLEs in the local store yet. Ingest first:\n\n"
        "`python cli.py ingest-celestrak --group stations`"
    )
    st.stop()

labels = {f"{o['name']} ({o['norad_id']})": o["norad_id"] for o in objects}
choice = st.sidebar.selectbox("Object", sorted(labels))
norad = labels[choice]
hours = st.sidebar.slider("Propagation window (hours)", 1, 72, 24)

rec = get_latest_tle(norad)
start = datetime.now(timezone.utc)
end = start + timedelta(hours=hours)

el = mean_elements(rec)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Regime", el.regime)
c2.metric("Perigee", f"{el.perigee_km:,.0f} km")
c3.metric("Apogee", f"{el.apogee_km:,.0f} km")
c4.metric("Period", f"{el.period_min:.1f} min")
st.caption(f"Latest TLE epoch: {rec.epoch} (source: {rec.source})")

left, right = st.columns(2)

with left:
    st.subheader("Ground track")
    gt = ground_track(rec, start, end, step_s=60.0)
    fig = go.Figure(
        go.Scattergeo(
            lat=gt["lat_deg"], lon=gt["lon_deg"], mode="lines",
            line=dict(width=1.5),
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("3D orbit (TEME)")
    eph = propagate_window(rec, start, end, step_s=60.0)
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    fig3d = go.Figure()
    fig3d.add_trace(
        go.Surface(
            x=R_EARTH_KM * np.cos(u) * np.sin(v),
            y=R_EARTH_KM * np.sin(u) * np.sin(v),
            z=R_EARTH_KM * np.cos(v),
            showscale=False, opacity=0.35,
        )
    )
    fig3d.add_trace(
        go.Scatter3d(
            x=eph.r_teme[:, 0], y=eph.r_teme[:, 1], z=eph.r_teme[:, 2],
            mode="lines", line=dict(width=3),
            name=f"NORAD {norad}",
        )
    )
    fig3d.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=420,
        scene=dict(aspectmode="data"),
    )
    st.plotly_chart(fig3d, use_container_width=True)

st.divider()
st.subheader("Maneuver history (F1.1 — needs historical TLEs)")
history = get_tles(norad)
if len(history) < 8:
    st.info(
        f"Only {len(history)} TLE(s) stored for this object. Ingest a historical "
        "window from Space-Track to enable maneuver detection:\n\n"
        f"`python cli.py ingest-historical --norad {norad} --start 2026-05-01 --end 2026-07-14`"
    )
else:
    events, noise_floor = detect_maneuvers(history)
    if noise_floor is None:
        st.info("Not enough valid TLE pairs to establish a per-object noise baseline.")
    elif not events:
        st.success(
            f"No anomalous maneuvers detected. Residual noise floor ≈ {noise_floor:.2f} m/s."
        )
    else:
        st.warning(f"{len(events)} candidate maneuver(s) detected")
        st.dataframe(
            [
                {
                    "epoch": e.epoch_iso,
                    "Δv (m/s)": round(e.dv_ms, 2),
                    "direction": e.direction,
                    "z-score": round(e.zscore, 1),
                    "confidence": e.confidence,
                }
                for e in events
            ]
        )
