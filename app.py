# app.py — Full PinasPath app (auto-bidir + walking connectors, safe parsing)

import streamlit as st
import pandas as pd
import networkx as nx
import heapq
import math
from datetime import datetime, timedelta
from streamlit_folium import st_folium
import folium
import os

st.set_page_config(page_title="PinasPath — Streamlit Prototype", layout="wide")

st.markdown("<h1 style='margin-bottom:6px;'>PinasPath</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='margin-top:0;color:#555;'>Quick prototype — shortest-travel-time route using local CSVs (stops.csv + routes.csv).</p>",
    unsafe_allow_html=True,
)

# ----------------- small CSS for nicer look -----------------
st.markdown(
    """
    <style>
    .mode-badge {display:inline-block;padding:4px 8px;border-radius:6px;color:white;font-weight:600;margin-right:6px;}
    .mode-bus {background:#1f77b4;}
    .mode-train {background:#2ca02c;}
    .mode-jeepney {background:#ff7f0e;}
    .mode-walk {background:#7f7f7f;}
    .panel {background:#ffffff;border-radius:8px;padding:12px;box-shadow: 0 2px 8px rgba(0,0,0,0.06);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Data loader -----------------
@st.cache_data
def load_data(stops_path="stops.csv", routes_path="routes.csv"):
    """
    Load stops.csv and routes.csv.
    - Ignores lines starting with '#' so appended comment blocks won't break parsing.
    - Coerces lat/lon to numeric (NaNs allowed).
    """
    if not os.path.exists(stops_path):
        raise FileNotFoundError(f"{stops_path} not found.")
    if not os.path.exists(routes_path):
        raise FileNotFoundError(f"{routes_path} not found.")

    stops = pd.read_csv(stops_path, dtype=str, comment="#")
    routes = pd.read_csv(routes_path, dtype=str, comment="#")

    if "lat" in stops.columns and "lon" in stops.columns:
        stops["lat"] = pd.to_numeric(stops["lat"], errors="coerce")
        stops["lon"] = pd.to_numeric(stops["lon"], errors="coerce")
    else:
        stops["lat"] = None
        stops["lon"] = None

    routes["travel_time"] = pd.to_numeric(routes["travel_time"], errors="coerce").fillna(1.0)

    if "stop_id" in stops.columns:
        stops["stop_id"] = stops["stop_id"].astype(str)
    if "from_stop" in routes.columns and "to_stop" in routes.columns:
        routes["from_stop"] = routes["from_stop"].astype(str)
        routes["to_stop"] = routes["to_stop"].astype(str)

    return stops, routes


try:
    stops, routes = load_data()
except Exception as e:
    st.error(f"Error loading CSVs: {e}")
    st.stop()

# ----------------- Sidebar: inputs & explanation -----------------
st.sidebar.header("Plan a trip")

stop_names = stops["stop_name"].tolist()
origin_name = st.sidebar.selectbox("Origin", stop_names, index=0)
destination_name = st.sidebar.selectbox(
    "Destination", stop_names, index=1 if len(stop_names) > 1 else 0
)

transfer_penalty = st.sidebar.number_input(
    "Transfer penalty (min)",
    min_value=0,
    max_value=30,
    value=2,
    step=1,
    help="Extra minutes added each time the traveler changes vehicle/route (models waiting/walking).",
)

st.sidebar.markdown(
    "<small>Increase the penalty to prefer fewer transfers even if travel time rises slightly.</small>",
    unsafe_allow_html=True,
)

show_map = st.sidebar.checkbox("Show map", value=True)

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.markdown("<b>Mode colors</b>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<span class="mode-badge mode-train">Train</span> '
    '<span class="mode-badge mode-bus">Bus</span> '
    '<span class="mode-badge mode-jeepney">Jeepney</span> '
    '<span class="mode-badge mode-walk">Walk</span>',
    unsafe_allow_html=True,
)

# ----------------- Helpers -----------------
def name_to_id(name):
    row = stops[stops["stop_name"] == name]
    if row.empty:
        return None
    return str(row["stop_id"].values[0])


origin_id = name_to_id(origin_name)
destination_id = name_to_id(destination_name)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))

# --- Graph builder + auto-fixes ---
def build_full_graph(stops_df, routes_df, add_walk_links=True, walk_thresh_m=700):
    G = nx.DiGraph()

    for _, r in stops_df.iterrows():
        sid = str(r["stop_id"])
        lat = float(r["lat"]) if pd.notna(r["lat"]) and r["lat"] != "" else None
        lon = float(r["lon"]) if pd.notna(r["lon"]) and r["lon"] != "" else None
        G.add_node(sid, name=r["stop_name"], lat=lat, lon=lon)

    for _, r in routes_df.iterrows():
        u = str(r["from_stop"])
        v = str(r["to_stop"])
        try:
            w = float(r["travel_time"])
        except Exception:
            w = 1.0

        mode = r.get("mode", "") or ""
        rn = r.get("route_name", "") or ""

        if G.has_edge(u, v):
            existing = G[u][v]
            if w < existing.get("travel_time", float("inf")):
                existing["travel_time"] = w
            routes_list = existing.get("route_names", [])
            if not any(
                x.get("route_name") == rn and x.get("mode") == mode
                for x in routes_list
            ):
                routes_list.append({"route_name": rn, "mode": mode})
            existing["route_names"] = routes_list
        else:
            G.add_edge(
                u,
                v,
                travel_time=w,
                route_names=[{"route_name": rn, "mode": mode}],
            )

    edges_to_add = []
    for u, v, data in list(G.edges(data=True)):
        if not G.has_edge(v, u):
            edges_to_add.append(
                (
                    v,
                    u,
                    {
                        "travel_time": data.get("travel_time", 1.0),
                        "route_names": data.get("route_names", []),
                    },
                )
            )

    for a, b, attrs in edges_to_add:
        G.add_edge(a, b, **attrs)

    if add_walk_links:
        coords = []
        for n, d in G.nodes(data=True):
            if d.get("lat") is not None and d.get("lon") is not None:
                coords.append((n, float(d["lat"]), float(d["lon"])))

        th_km = walk_thresh_m / 1000.0
        for i in range(len(coords)):
            id1, lat1, lon1 = coords[i]
            for j in range(i + 1, len(coords)):
                id2, lat2, lon2 = coords[j]
                dist_km = haversine_km(lat1, lon1, lat2, lon2)
                if dist_km <= th_km:
                    walk_time_min = max(1.0, (dist_km * 1000) / 80.0)
                    if not G.has_edge(id1, id2):
                        G.add_edge(
                            id1,
                            id2,
                            travel_time=walk_time_min,
                            route_names=[{"route_name": "walk", "mode": "walk"}],
                        )
                    if not G.has_edge(id2, id1):
                        G.add_edge(
                            id2,
                            id1,
                            travel_time=walk_time_min,
                            route_names=[{"route_name": "walk", "mode": "walk"}],
                        )

    return G


G = build_full_graph(stops, routes, add_walk_links=True, walk_thresh_m=700)
