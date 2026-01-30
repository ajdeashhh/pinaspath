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

# ----------------- Sidebar -----------------
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
)

show_map = st.sidebar.checkbox("Show map", value=True)

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

# ----------------- Graph -----------------
def build_full_graph(stops_df, routes_df, add_walk_links=True, walk_thresh_m=700):
    G = nx.DiGraph()

    for _, r in stops_df.iterrows():
        G.add_node(
            str(r["stop_id"]),
            name=r["stop_name"],
            lat=r["lat"],
            lon=r["lon"],
        )

    for _, r in routes_df.iterrows():
        G.add_edge(
            str(r["from_stop"]),
            str(r["to_stop"]),
            travel_time=float(r["travel_time"]),
            route_names=[{"route_name": r.get("route_name", ""), "mode": r.get("mode", "")}],
        )

    for u, v, d in list(G.edges(data=True)):
        if not G.has_edge(v, u):
            G.add_edge(v, u, **d)

    return G


G = build_full_graph(stops, routes)

# ----------------- Shortest path -----------------
def shortest_path_with_transfer_penalty(G, origin, destination, transfer_penalty=0):
    pq = [(0, origin, None, None, [origin], [])]
    visited = {}

    while pq:
        cost, node, prev_mode, prev_route, path, legs = heapq.heappop(pq)

        if node == destination:
            return {"total_cost": cost, "path": path, "legs": legs}

        for nbr in G.neighbors(node):
            e = G[node][nbr]
            rn = e["route_names"][0]
            mode = rn["mode"]
            route = rn["route_name"]
            penalty = transfer_penalty if prev_mode and mode != prev_mode else 0
            new_cost = cost + e["travel_time"] + penalty

            heapq.heappush(
                pq,
                (
                    new_cost,
                    nbr,
                    mode,
                    route,
                    path + [nbr],
                    legs + [{
                        "from_id": node,
                        "to_id": nbr,
                        "from_name": G.nodes[node]["name"],
                        "to_name": G.nodes[nbr]["name"],
                        "mode": mode,
                        "route_name": route,
                        "travel_time": e["travel_time"],
                        "penalty": penalty,
                    }],
                ),
            )

    return None


# ----------------- Layout -----------------
left_col, right_col = st.columns([2, 1])

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ----------------- LEFT: MAP -----------------
with left_col:
    if st.session_state["last_result"] is None:
        if show_map:
            m = folium.Map(location=[14.6, 121.0], zoom_start=12)
            for _, r in stops.dropna(subset=["lat", "lon"]).iterrows():
                folium.CircleMarker(
                    location=(r["lat"], r["lon"]),
                    radius=4,
                    popup=r["stop_name"],
                ).add_to(m)
            st_folium(m, width=900, height=700)
    else:
        if show_map:
            m = folium.Map(location=[14.6, 121.0], zoom_start=13)
            for leg in st.session_state["last_result"]["legs"]:
                u = leg["from_id"]
                v = leg["to_id"]
                if G.nodes[u]["lat"] and G.nodes[v]["lat"]:
                    folium.PolyLine(
                        [
                            (G.nodes[u]["lat"], G.nodes[u]["lon"]),
                            (G.nodes[v]["lat"], G.nodes[v]["lon"]),
                        ]
                    ).add_to(m)
            st_folium(m, width=900, height=700)

# ----------------- RIGHT: CONTROLS -----------------
with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Trip control")

    # 🔴 🔴 🔴 SURGICAL FIX — ONLY ADDITION 🔴 🔴 🔴
    if st.session_state["last_result"] is not None:
        if st.button("Clear route / Back to overview"):
            st.session_state["last_result"] = None
            st.rerun()
    # 🔴 🔴 🔴 END FIX 🔴 🔴 🔴

    if st.button("Plan route"):
        r = shortest_path_with_transfer_penalty(
            G, origin_id, destination_id, transfer_penalty
        )
        if r:
            st.session_state["last_result"] = r
            st.rerun()

    if st.session_state["last_result"]:
        st.markdown("### Recommended route")
        st.write(f"Estimated time: {st.session_state['last_result']['total_cost']:.1f} min")

    st.markdown("</div>", unsafe_allow_html=True)
