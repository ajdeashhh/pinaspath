# app.py (fixed: do not print st_folium return value)
import streamlit as st
import pandas as pd
import networkx as nx
from streamlit_folium import st_folium
import folium
import heapq
import os

st.set_page_config(page_title="PinasPath — Streamlit Prototype", layout="wide")
st.title("PinasPath — Streamlit Prototype")
st.write("Shortest-travel-time route using local CSVs (stops.csv + routes.csv). Map persists after finding a route.")

# ----------------- Data loader -----------------
@st.cache_data
def load_data(stops_path="stops.csv", routes_path="routes.csv"):
    if not os.path.exists(stops_path):
        raise FileNotFoundError(f"{stops_path} not found.")
    if not os.path.exists(routes_path):
        raise FileNotFoundError(f"{routes_path} not found.")
    stops = pd.read_csv(stops_path, dtype=str)
    routes = pd.read_csv(routes_path, dtype=str)
    stops["lat"] = stops["lat"].astype(float)
    stops["lon"] = stops["lon"].astype(float)
    routes["travel_time"] = routes["travel_time"].astype(float)
    stops["stop_id"] = stops["stop_id"].astype(str)
    routes["from_stop"] = routes["from_stop"].astype(str)
    routes["to_stop"] = routes["to_stop"].astype(str)
    return stops, routes

try:
    stops, routes = load_data()
except Exception as e:
    st.error(f"Error loading CSVs: {e}")
    st.stop()

# ----------------- Sidebar inputs -----------------
st.sidebar.header("Trip input")
stop_names = stops["stop_name"].tolist()
origin_name = st.sidebar.selectbox("Origin", stop_names, index=0)
destination_name = st.sidebar.selectbox("Destination", stop_names, index=1 if len(stop_names) > 1 else 0)
show_map = st.sidebar.checkbox("Show map", value=True)
transfer_penalty = st.sidebar.number_input(
    "Transfer penalty (minutes)",
    min_value=0,
    max_value=60,
    value=2,
    step=1,
    help="Penalty added when the mode or route changes between legs."
)

# ----------------- Helpers -----------------
def name_to_id(name):
    row = stops[stops["stop_name"] == name]
    if row.empty:
        return None
    return str(row["stop_id"].values[0])

origin_id = name_to_id(origin_name)
destination_id = name_to_id(destination_name)

def build_graph(stops_df, routes_df, auto_bidirectional=False):
    G = nx.DiGraph()
    for _, r in stops_df.iterrows():
        G.add_node(str(r["stop_id"]), name=r["stop_name"], lat=float(r["lat"]), lon=float(r["lon"]))
    for _, r in routes_df.iterrows():
        u = str(r["from_stop"]); v = str(r["to_stop"])
        w = float(r["travel_time"])
        mode = r.get("mode", "")
        rn = r.get("route_name", "")
        if G.has_edge(u, v):
            existing = G[u][v]
            if w < existing.get("travel_time", float("inf")):
                existing["travel_time"] = w
            routes_list = existing.get("route_names", [])
            if not any(x.get("route_name")==rn and x.get("mode")==mode for x in routes_list):
                routes_list.append({"route_name": rn, "mode": mode})
            existing["route_names"] = routes_list
        else:
            G.add_edge(u, v, travel_time=w, route_names=[{"route_name": rn, "mode": mode}])
        if auto_bidirectional and not G.has_edge(v, u):
            G.add_edge(v, u, travel_time=w, route_names=[{"route_name": rn, "mode": mode}])
    return G

# Build graph (no auto-bidirectional by default)
G = build_graph(stops, routes, auto_bidirectional=False)

def shortest_path_with_transfer_penalty(G, origin, destination, transfer_penalty=0):
    pq = []
    heapq.heappush(pq, (0.0, origin, None, None, [origin], []))
    visited = {}
    while pq:
        cost, node, prev_mode, prev_route, path, legs = heapq.heappop(pq)
        key = (node, prev_mode, prev_route)
        if key in visited and visited[key] <= cost:
            continue
        visited[key] = cost
        if node == destination:
            return {"total_cost": cost, "path": path, "legs": legs}
        for nbr in G.neighbors(node):
            e = G[node][nbr]
            travel_time = float(e.get("travel_time", 0.0))
            rn_list = e.get("route_names", [])
            if rn_list:
                mode = rn_list[0].get("mode")
                route_name = rn_list[0].get("route_name")
            else:
                mode = e.get("mode", "")
                route_name = ""
            add = 0.0
            if prev_mode is not None:
                if mode != prev_mode:
                    add = transfer_penalty
                else:
                    if prev_route and route_name and prev_route != route_name:
                        add = transfer_penalty
            new_cost = cost + travel_time + add
            new_path = path + [nbr]
            new_legs = legs + [{
                "from_id": node,
                "to_id": nbr,
                "from_name": G.nodes[node]["name"],
                "to_name": G.nodes[nbr]["name"],
                "mode": mode,
                "route_name": route_name,
                "travel_time": travel_time,
                "penalty": add,
                "route_options": rn_list
            }]
            heapq.heappush(pq, (new_cost, nbr, mode, route_name, new_path, new_legs))
    return None

# ----------------- Persistent UI placeholders -----------------
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

map_placeholder = st.empty()
text_placeholder = st.empty()

# Always show a base map (so map area doesn't vanish).
# IMPORTANT: call st_folium(...) but DO NOT pass its returned dict to st.write or map_placeholder.write.
# Assign return value to a variable if you want to inspect map interaction later (but don't st.write it).
with map_placeholder.container():
    base_center = [stops["lat"].mean(), stops["lon"].mean()]
    base_map = folium.Map(location=base_center, zoom_start=12)
    sample = stops.sample(min(300, len(stops)), random_state=1) if len(stops) > 300 else stops
    for _, r in sample.iterrows():
        folium.CircleMarker(
            location=(float(r["lat"]), float(r["lon"])),
            radius=2,
            color="#444",
            fill=True,
            fill_opacity=0.6
        ).add_to(base_map)
    # render map and capture returned state (but do NOT display that state)
    _map_state = st_folium(base_map, width=900, height=600)

# ----------------- Find Route action -----------------
if st.button("Find Route"):
    if origin_id is None or destination_id is None:
        st.warning("Invalid origin or destination selection.")
    elif origin_id == destination_id:
        st.warning("Origin and destination are the same.")
    else:
        res = shortest_path_with_transfer_penalty(G, origin_id, destination_id, transfer_penalty=transfer_penalty)
        if not res:
            st.error("No path found between selected stops.")
        else:
            st.session_state["last_result"] = res
            total_travel = sum(leg["travel_time"] for leg in res["legs"])
            total_penalty = sum(leg["penalty"] for leg in res["legs"])
            text_placeholder.subheader("Route legs")
            text_placeholder.write(f"Estimated total (including penalties): **{res['total_cost']:.1f} minutes**")
            for i, leg in enumerate(res["legs"], 1):
                text = f"{i}. {leg['from_name']} → {leg['to_name']} — {leg['mode']} ({leg['route_name']}) | {leg['travel_time']:.1f} min"
                if leg['penalty'] > 0:
                    text += f" + {leg['penalty']:.1f} min transfer penalty"
                text_placeholder.write(text)
            text_placeholder.write(f"**Total travel time (legs only):** {total_travel:.1f} minutes")
            text_placeholder.write(f"**Total transfer penalty:** {total_penalty:.1f} minutes")

# If we have a last_result, render its map persistently below (DO NOT st.write the returned dict)
if st.session_state.get("last_result") and show_map:
    res = st.session_state["last_result"]
    orig_node = res["path"][0]
    lat0 = G.nodes[orig_node]["lat"]
    lon0 = G.nodes[orig_node]["lon"]
    m = folium.Map(location=[lat0, lon0], zoom_start=13)
    coords = []
    for node in res["path"]:
        node_data = G.nodes[node]
        coords.append((node_data["lat"], node_data["lon"]))
        route_opts = []
        for leg in res["legs"]:
            if leg["from_id"] == node:
                route_opts = [f"{r.get('route_name')}({r.get('mode')})" for r in leg.get('route_options', [])]
                break
        popup_html = f"{node_data['name']}<br>options: {', '.join(route_opts)}"
        folium.CircleMarker(
            location=(node_data["lat"], node_data["lon"]),
            radius=6,
            popup=popup_html,
            tooltip=node_data["name"]
        ).add_to(m)
    folium.PolyLine(coords, weight=6, color="green", opacity=0.8).add_to(m)
    # render route map and capture returned state (but do NOT display it)
    _route_map_state = st_folium(m, width=900, height=600)
