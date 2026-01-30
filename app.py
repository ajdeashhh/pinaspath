# app.py — PinasPath (map-accurate where data allows)

import streamlit as st
import pandas as pd
import networkx as nx
import heapq
import math
from datetime import datetime, timedelta
from streamlit_folium import st_folium
import folium
import os
import requests

# ----------------- Page setup -----------------
st.set_page_config(page_title="PinasPath", layout="wide")
st.title("PinasPath")
st.caption(
    "🗺️ Walking & jeepney paths follow real roads. Train lines are schematic (station-to-station)."
)

# ----------------- Data loader -----------------
@st.cache_data
def load_data(stops_path="stops.csv", routes_path="routes.csv"):
    if not os.path.exists(stops_path) or not os.path.exists(routes_path):
        st.error("stops.csv or routes.csv not found")
        st.stop()

    stops = pd.read_csv(stops_path, dtype=str, comment="#")
    routes = pd.read_csv(routes_path, dtype=str, comment="#")

    stops["lat"] = pd.to_numeric(stops.get("lat"), errors="coerce")
    stops["lon"] = pd.to_numeric(stops.get("lon"), errors="coerce")
    routes["travel_time"] = pd.to_numeric(routes["travel_time"], errors="coerce").fillna(1)

    stops["stop_id"] = stops["stop_id"].astype(str)
    routes["from_stop"] = routes["from_stop"].astype(str)
    routes["to_stop"] = routes["to_stop"].astype(str)

    return stops, routes


stops, routes = load_data()

# ----------------- Sidebar -----------------
st.sidebar.header("Plan a trip")

stop_names = stops["stop_name"].tolist()
origin_name = st.sidebar.selectbox("Origin", stop_names, 0)
destination_name = st.sidebar.selectbox("Destination", stop_names, 1)

transfer_penalty = st.sidebar.number_input(
    "Transfer penalty (min)", min_value=0, max_value=30, value=2
)

show_map = st.sidebar.checkbox("Show map", True)


def name_to_id(name):
    return stops.loc[stops["stop_name"] == name, "stop_id"].values[0]


origin_id = name_to_id(origin_name)
destination_id = name_to_id(destination_name)

# ----------------- OSRM geometry -----------------
@st.cache_data(show_spinner=False)
def osrm_geometry(lat1, lon1, lat2, lon2, profile):
    try:
        url = (
            f"https://router.project-osrm.org/route/v1/"
            f"{profile}/{lon1},{lat1};{lon2},{lat2}"
            "?overview=full&geometries=geojson"
        )
        r = requests.get(url, timeout=10).json()
        coords = r["routes"][0]["geometry"]["coordinates"]
        return [(lat, lon) for lon, lat in coords]
    except Exception:
        return None


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
            mode=r["mode"],
            route_name=r["route_name"],
        )

    return G


G = build_graph(stops, routes)

# ----------------- Shortest path -----------------
def shortest_path(G, origin, destination, penalty):
    pq = [(0, origin, None, None, [])]
    visited = {}

    while pq:
        cost, node, prev_mode, prev_route, legs = heapq.heappop(pq)
        key = (node, prev_mode, prev_route)

        if key in visited and visited[key] <= cost:
            continue
        visited[key] = cost

        if node == destination:
            return {"total": cost, "legs": legs}

        for nbr in G.neighbors(node):
            e = G[node][nbr]
            add = 0
            if prev_mode and (e["mode"] != prev_mode or e["route_name"] != prev_route):
                add = penalty

            new_leg = {
                "from": node,
                "to": nbr,
                "mode": e["mode"],
                "route": e["route_name"],
                "time": e["travel_time"],
                "penalty": add,
            }

            heapq.heappush(
                pq,
                (
                    cost + e["travel_time"] + add,
                    nbr,
                    e["mode"],
                    e["route_name"],
                    legs + [new_leg],
                ),
            )

    return None


def compress_legs(legs):
    if not legs:
        return []

    out = [legs[0].copy()]
    for l in legs[1:]:
        last = out[-1]
        if l["mode"] == last["mode"] and l["route"] == last["route"]:
            last["to"] = l["to"]
            last["time"] += l["time"]
            last["penalty"] += l["penalty"]
        else:
            out.append(l.copy())
    return out


# ----------------- Layout -----------------
left, right = st.columns([2, 1])

if "result" not in st.session_state:
    st.session_state.result = None

# ----------------- Controls -----------------
with right:
    if st.button("Plan route"):
        st.session_state.result = shortest_path(
            G, origin_id, destination_id, transfer_penalty
        )
        st.rerun()

    if st.session_state.result:
        legs = compress_legs(st.session_state.result["legs"])
        total = sum(l["time"] + l["penalty"] for l in legs)

        st.markdown("### Route")
        st.write(f"**Estimated total:** {total:.1f} min")

        for l in legs:
            st.write(f"{l['mode']} {l['route']} → {l['time']:.1f} min")

# ----------------- Map -----------------
with left:
    if show_map and st.session_state.result:
        legs = compress_legs(st.session_state.result["legs"])
        first = legs[0]["from"]
        start = G.nodes[first]

        m = folium.Map(
            location=[start["lat"], start["lon"]],
            zoom_start=13,
            tiles="CartoDB positron",
        )

        for l in legs:
            u = G.nodes[l["from"]]
            v = G.nodes[l["to"]]
            mode = l["mode"].lower()

            color = "#7f7f7f"
            if "train" in mode:
                color = "#2ca02c"
            elif "bus" in mode:
                color = "#1f77b4"
            elif "jeep" in mode:
                color = "#ff7f0e"

            geom = None
            if mode == "walk":
                geom = osrm_geometry(u["lat"], u["lon"], v["lat"], v["lon"], "foot")
            elif "jeep" in mode:
                geom = osrm_geometry(u["lat"], u["lon"], v["lat"], v["lon"], "car")

            if geom:
                folium.PolyLine(geom, color=color, weight=4).add_to(m)
            else:
                folium.PolyLine(
                    [(u["lat"], u["lon"]), (v["lat"], v["lon"])],
                    color=color,
                    weight=6,
                ).add_to(m)

        st_folium(m, width=900, height=650)

