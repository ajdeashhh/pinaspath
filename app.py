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

# ----------------- CSS -----------------
st.markdown(
    """
    <style>
    .mode-badge {display:inline-block;padding:4px 8px;border-radius:6px;color:white;font-weight:600;margin-right:6px;}
    .mode-bus {background:#1f77b4;}
    .mode-train {background:#2ca02c;}
    .mode-jeepney {background:#ff7f0e;}
    .mode-walk {background:#7f7f7f;}
    .panel {background:#ffffff;border-radius:8px;padding:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Data loader -----------------
@st.cache_data
def load_data(stops_path="stops.csv", routes_path="routes.csv"):
    stops = pd.read_csv(stops_path, dtype=str, comment="#")
    routes = pd.read_csv(routes_path, dtype=str, comment="#")

    stops["lat"] = pd.to_numeric(stops["lat"], errors="coerce")
    stops["lon"] = pd.to_numeric(stops["lon"], errors="coerce")
    routes["travel_time"] = pd.to_numeric(routes["travel_time"], errors="coerce").fillna(1.0)

    stops["stop_id"] = stops["stop_id"].astype(str)
    routes["from_stop"] = routes["from_stop"].astype(str)
    routes["to_stop"] = routes["to_stop"].astype(str)

    return stops, routes


stops, routes = load_data()

# ----------------- Sidebar -----------------
st.sidebar.header("Plan a trip")

stop_names = stops["stop_name"].tolist()
origin_name = st.sidebar.selectbox("Origin", stop_names)
destination_name = st.sidebar.selectbox("Destination", stop_names, index=1)

transfer_penalty = st.sidebar.number_input(
    "Transfer penalty (min)", min_value=0, max_value=30, value=2
)

show_map = st.sidebar.checkbox("Show map", value=True)

# ----------------- Helpers -----------------
def name_to_id(name):
    return stops.loc[stops["stop_name"] == name, "stop_id"].values[0]


origin_id = name_to_id(origin_name)
destination_id = name_to_id(destination_name)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))

# ----------------- Graph -----------------
def build_full_graph(stops, routes, walk_thresh_m=700):
    G = nx.DiGraph()

    for _, r in stops.iterrows():
        G.add_node(
            r["stop_id"],
            name=r["stop_name"],
            lat=r["lat"],
            lon=r["lon"],
        )

    for _, r in routes.iterrows():
        G.add_edge(
            r["from_stop"],
            r["to_stop"],
            travel_time=r["travel_time"],
            route_names=[{"route_name": r.get("route_name", ""), "mode": r.get("mode", "")}],
        )

    # auto-bidirectional
    for u, v, d in list(G.edges(data=True)):
        if not G.has_edge(v, u):
            G.add_edge(v, u, **d)

    # walking connectors
    coords = [
        (n, d["lat"], d["lon"])
        for n, d in G.nodes(data=True)
        if pd.notna(d["lat"]) and pd.notna(d["lon"])
    ]

    for i in range(len(coords)):
        id1, lat1, lon1 = coords[i]
        for j in range(i + 1, len(coords)):
            id2, lat2, lon2 = coords[j]
            dist = haversine_km(lat1, lon1, lat2, lon2)
            if dist * 1000 <= walk_thresh_m:
                t = max(1.0, (dist * 1000) / 80)
                G.add_edge(
                    id1,
                    id2,
                    travel_time=t,
                    route_names=[{"route_name": "walk", "mode": "walk"}],
                )
                G.add_edge(
                    id2,
                    id1,
                    travel_time=t,
                    route_names=[{"route_name": "walk", "mode": "walk"}],
                )

    return G


G = build_full_graph(stops, routes)

# ----------------- Shortest path -----------------
def shortest_path(G, origin, destination, penalty):
    pq = [(0, origin, None, None, [], [])]

    visited = {}

    while pq:
        cost, node, pmode, proute, path, legs = heapq.heappop(pq)

        if node == destination:
            return {"total": cost, "legs": legs}

        for nbr in G.neighbors(node):
            e = G[node][nbr]
            rn = e["route_names"][0]
            mode = rn["mode"]
            route = rn["route_name"]
            add = penalty if pmode and (mode != pmode or route != proute) else 0
            nc = cost + e["travel_time"] + add

            state = (nbr, mode, route)
            if visited.get(state, 1e9) <= nc:
                continue
            visited[state] = nc

            legs2 = legs + [{
                "from": node,
                "to": nbr,
                "from_name": G.nodes[node]["name"],
                "to_name": G.nodes[nbr]["name"],
                "mode": mode,
                "route": route,
                "time": e["travel_time"],
            }]

            heapq.heappush(pq, (nc, nbr, mode, route, path + [nbr], legs2))

    return None


# ----------------- COMPRESS LEGS (KEY FIX) -----------------
def legs_to_instructions(legs):
    out = []
    cur = legs[0]

    for l in legs[1:]:
        if l["mode"] == cur["mode"] and l["route"] == cur["route"]:
            cur["to_name"] = l["to_name"]
            cur["time"] += l["time"]
        else:
            out.append(cur)
            cur = l

    out.append(cur)
    return out


# ----------------- Layout -----------------
left, right = st.columns([2, 1])

if "last" not in st.session_state:
    st.session_state["last"] = None

# ----------------- MAP -----------------
legend_html = """
<div style="position:fixed;bottom:40px;left:10px;background:white;
padding:8px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.2)">
<b>Legend</b><br>
<span style="color:#2ca02c">■</span> Train<br>
<span style="color:#1f77b4">■</span> Bus<br>
<span style="color:#ff7f0e">■</span> Jeepney<br>
<span style="color:#7f7f7f">■</span> Walk
</div>
"""

with left:
    if show_map:
        m = folium.Map(location=[stops["lat"].mean(), stops["lon"].mean()], zoom_start=12)
        m.get_root().html.add_child(folium.Element(legend_html))

        if st.session_state["last"]:
            for l in st.session_state["last"]["legs"]:
                u, v = l["from"], l["to"]
                if pd.notna(G.nodes[u]["lat"]) and pd.notna(G.nodes[v]["lat"]):
                    color = "#7f7f7f"
                    if "train" in l["mode"].lower():
                        color = "#2ca02c"
                    elif "bus" in l["mode"].lower():
                        color = "#1f77b4"
                    elif "jeep" in l["mode"].lower():
                        color = "#ff7f0e"

                    folium.PolyLine(
                        [(G.nodes[u]["lat"], G.nodes[u]["lon"]),
                         (G.nodes[v]["lat"], G.nodes[v]["lon"])],
                        color=color,
                        weight=6,
                    ).add_to(m)

        st_folium(m, width=900, height=700)

# ----------------- RIGHT PANEL -----------------
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # SURGICAL FIX
    if st.session_state["last"]:
        if st.button("Clear route / Back to overview"):
            st.session_state["last"] = None
            st.rerun()

    if st.button("Plan route"):
        res = shortest_path(G, origin_id, destination_id, transfer_penalty)
        if res:
            st.session_state["last"] = res
            st.rerun()

    if st.session_state["last"]:
        st.markdown("### Steps")
        steps = legs_to_instructions(st.session_state["last"]["legs"])

        for s in steps:
            badge = "mode-walk"
            if "train" in s["mode"].lower():
                badge = "mode-train"
            elif "bus" in s["mode"].lower():
                badge = "mode-bus"
            elif "jeep" in s["mode"].lower():
                badge = "mode-jeepney"

            st.markdown(
                f"<span class='mode-badge {badge}'>{s['mode']}</span> "
                f"{s['from_name']} → {s['to_name']} "
                f"{s['time']:.1f} min",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
