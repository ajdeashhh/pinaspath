# app.py — Full PinasPath app (restored original behavior + surgical fixes)

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
    "<p style='margin-top:0;color:#555;'>Quick prototype — shortest-travel-time route using local CSVs.</p>",
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
    if not os.path.exists(stops_path):
        raise FileNotFoundError(f"{stops_path} not found.")
    if not os.path.exists(routes_path):
        raise FileNotFoundError(f"{routes_path} not found.")

    stops = pd.read_csv(stops_path, dtype=str, comment="#")
    routes = pd.read_csv(routes_path, dtype=str, comment="#")

    stops["lat"] = pd.to_numeric(stops.get("lat"), errors="coerce")
    stops["lon"] = pd.to_numeric(stops.get("lon"), errors="coerce")
    routes["travel_time"] = pd.to_numeric(routes.get("travel_time"), errors="coerce").fillna(1.0)

    stops["stop_id"] = stops["stop_id"].astype(str)
    routes["from_stop"] = routes["from_stop"].astype(str)
    routes["to_stop"] = routes["to_stop"].astype(str)

    return stops, routes


try:
    stops, routes = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

# ----------------- Sidebar -----------------
st.sidebar.header("Plan a trip")

stop_names = stops["stop_name"].tolist()
origin_name = st.sidebar.selectbox("Origin", stop_names, index=0)
destination_name = st.sidebar.selectbox(
    "Destination", stop_names, index=1 if len(stop_names) > 1 else 0
)

transfer_penalty = st.sidebar.number_input(
    "Transfer penalty (min)", min_value=0, max_value=30, value=2
)

show_map = st.sidebar.checkbox("Show map", value=True)

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
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
    return None if row.empty else row["stop_id"].values[0]


origin_id = name_to_id(origin_name)
destination_id = name_to_id(destination_name)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dl / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


# ----------------- Graph builder -----------------
def build_graph(stops_df, routes_df):
    G = nx.DiGraph()

    for _, r in stops_df.iterrows():
        G.add_node(
            r["stop_id"],
            name=r["stop_name"],
            lat=r["lat"],
            lon=r["lon"],
        )

    for _, r in routes_df.iterrows():
        G.add_edge(
            r["from_stop"],
            r["to_stop"],
            travel_time=float(r["travel_time"]),
            route_name=r.get("route_name", ""),
            mode=r.get("mode", ""),
        )

    return G


G = build_graph(stops, routes)

# ----------------- Shortest path -----------------
def shortest_path(G, origin, destination, penalty):
    pq = [(0, origin, None, None, [], [])]
    visited = {}

    while pq:
        cost, node, pmode, proute, path, legs = heapq.heappop(pq)

        if (node, pmode, proute) in visited:
            continue
        visited[(node, pmode, proute)] = cost

        if node == destination:
            return {"total_cost": cost, "legs": legs}

        for nbr in G.neighbors(node):
            e = G[node][nbr]
            mode = e.get("mode", "")
            route = e.get("route_name", "")
            add = penalty if pmode and (mode != pmode or route != proute) else 0

            heapq.heappush(
                pq,
                (
                    cost + e["travel_time"] + add,
                    nbr,
                    mode,
                    route,
                    path + [nbr],
                    legs
                    + [
                        {
                            "from": node,
                            "to": nbr,
                            "from_name": G.nodes[node]["name"],
                            "to_name": G.nodes[nbr]["name"],
                            "mode": mode,
                            "route": route,
                            "time": e["travel_time"],
                            "penalty": add,
                        }
                    ],
                ),
            )

    return None


# ----------------- COMPRESS LEGS (ORIGINAL BEHAVIOR) -----------------
def legs_to_instructions(legs):
    if not legs:
        return []

    out = []
    cur = legs[0].copy()

    for leg in legs[1:]:
        if leg["mode"] == cur["mode"] and leg["route"] == cur["route"]:
            cur["to_name"] = leg["to_name"]
            cur["time"] += leg["time"]
            cur["penalty"] += leg["penalty"]
        else:
            out.append(cur)
            cur = leg.copy()

    out.append(cur)
    return out


# ----------------- Layout -----------------
left, right = st.columns([2, 1])

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ----------------- Map -----------------
with left:
    m = folium.Map(location=[14.6, 121.0], zoom_start=12, tiles="CartoDB positron")

    legend = """
    <div style="position:fixed;bottom:40px;left:10px;background:white;
    padding:8px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.2)">
    <b>Legend</b><br>
    <span style="color:#2ca02c">■</span> Train<br>
    <span style="color:#1f77b4">■</span> Bus<br>
    <span style="color:#ff7f0e">■</span> Jeepney<br>
    <span style="color:#7f7f7f">■</span> Walk
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    if show_map:
        st_folium(m, width=900, height=700)

# ----------------- Right panel -----------------
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Trip control")

    if st.button("Plan route"):
        st.session_state["last_result"] = shortest_path(
            G, origin_id, destination_id, transfer_penalty
        )
        st.rerun()

    if st.session_state["last_result"]:
        if st.button("Clear route / Back to overview"):
            st.session_state["last_result"] = None
            st.rerun()

        res = st.session_state["last_result"]
        steps = legs_to_instructions(res["legs"])

        st.markdown("### Recommended route")

        for step in steps:
            mode = step["mode"].lower()
            cls = (
                "mode-train"
                if "train" in mode
                else "mode-bus"
                if "bus" in mode
                else "mode-jeepney"
                if "jeep" in mode
                else "mode-walk"
            )

            st.markdown(
                f"""
<div>
<span class="mode-badge {cls}">{step["mode"]}</span>
<b>{step["from_name"]}</b> → <b>{step["to_name"]}</b><br>
<small>{step["route"]} • {step["time"]:.1f} min</small>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
