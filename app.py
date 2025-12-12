# app.py
import streamlit as st
import pandas as pd
import networkx as nx
from streamlit_folium import st_folium
import folium
import heapq
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="PinasPath — Streamlit Prototype", layout="wide")
st.markdown("<h1 style='margin-bottom:6px;'>PinasPath</h1>", unsafe_allow_html=True)
st.markdown("<p style='margin-top:0;color:#555;'>Quick prototype — shortest-travel-time route using local CSVs (stops.csv + routes.csv).</p>", unsafe_allow_html=True)

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

# ----------------- Sidebar: inputs & explanation -----------------
st.sidebar.header("Plan a trip")
stop_names = stops["stop_name"].tolist()
origin_name = st.sidebar.selectbox("Origin", stop_names, index=0)
destination_name = st.sidebar.selectbox("Destination", stop_names, index=1 if len(stop_names) > 1 else 0)

transfer_penalty = st.sidebar.number_input(
    "Transfer penalty (min)",
    min_value=0,
    max_value=30,
    value=2,
    step=1,
    help="Extra minutes added each time the traveler changes vehicle/route (models waiting/walking)."
)
st.sidebar.markdown(
    "<small>Increase the penalty to prefer fewer transfers even if travel time rises slightly.</small>",
    unsafe_allow_html=True,
)
show_map = st.sidebar.checkbox("Show map", value=True)

# small legend
st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.markdown("<b>Mode colors</b>", unsafe_allow_html=True)
st.sidebar.markdown('<span class="mode-badge mode-train">Train</span> <span class="mode-badge mode-bus">Bus</span> <span class="mode-badge mode-jeepney">Jeepney</span> <span class="mode-badge mode-walk">Walk</span>', unsafe_allow_html=True)

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

# ----------------- UI layout: map left, details right -----------------
left_col, right_col = st.columns([2, 1])

# persistent state
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# Render map in left column (only one map rendered)
with left_col:
    if st.session_state.get("last_result") is None:
        # show base map
        base_center = [stops["lat"].mean(), stops["lon"].mean()]
        base_map = folium.Map(location=base_center, zoom_start=12, tiles="CartoDB positron")
        # show stops markers
        for _, r in stops.iterrows():
            folium.CircleMarker(
                location=(r["lat"], r["lon"]),
                radius=4,
                color="#2b5876",
                fill=True,
                fill_opacity=0.8,
                popup=f"{r['stop_id']}: {r['stop_name']}"
            ).add_to(base_map)
        # draw legend box
        legend_html = """
         <div style="position: fixed; 
                     bottom: 50px; left: 10px; width:150px; height:110px; 
                     background-color: white; z-index:9999; font-size:12px; border-radius:8px; padding:8px; box-shadow:0 2px 6px rgba(0,0,0,0.15)">
         <b>Legend</b><br>
         <span style="background:#2ca02c;color:white;padding:3px 6px;border-radius:5px"> Train </span>&nbsp; Train<br>
         <span style="background:#1f77b4;color:white;padding:3px 6px;border-radius:5px"> Bus </span>&nbsp; Bus<br>
         <span style="background:#ff7f0e;color:white;padding:3px 6px;border-radius:5px"> Jeepney </span>&nbsp; Jeepney<br>
         <span style="background:#7f7f7f;color:white;padding:3px 6px;border-radius:5px"> Walk </span>&nbsp; Walk
         </div>
         """
        base_map.get_root().html.add_child(folium.Element(legend_html))
        # render map but DO NOT st.write the returned dict
        _ = st_folium(base_map, width=900, height=700)
    else:
        # show route map
        res = st.session_state["last_result"]
        orig = res["path"][0]
        lat0 = G.nodes[orig]["lat"]; lon0 = G.nodes[orig]["lon"]
        m = folium.Map(location=[lat0, lon0], zoom_start=13, tiles="CartoDB positron")
        coords = []
        # draw polyline segments with colored styling per mode
        for idx, leg in enumerate(res["legs"]):
            u = leg["from_id"]; v = leg["to_id"]
            udata = G.nodes[u]; vdata = G.nodes[v]
            coords.append((udata["lat"], udata["lon"]))
            # color by mode
            mode = leg.get("mode", "").lower()
            color = "#7f7f7f"
            if "train" in mode: color = "#2ca02c"
            elif "bus" in mode: color = "#1f77b4"
            elif "jeep" in mode or "jeepney" in mode: color = "#ff7f0e"
            elif "walk" in mode: color = "#7f7f7f"
            # segment line
            folium.PolyLine(
                [(udata["lat"], udata["lon"]), (vdata["lat"], vdata["lon"])],
                weight=6 if mode != "walk" else 3,
                color=color,
                opacity=0.9 if mode != "walk" else 0.6
            ).add_to(m)
            # stop marker
            popup = f"<b>{udata['name']}</b><br>{leg['mode']} {leg['route_name']}<br>{leg['travel_time']:.1f} min"
            folium.CircleMarker(location=(udata["lat"], udata["lon"]), radius=6, color=color, fill=True, fill_color=color, popup=popup).add_to(m)
        # add last node marker
        last_node = res["path"][-1]
        last_nd = G.nodes[last_node]
        folium.CircleMarker(location=(last_nd["lat"], last_nd["lon"]), radius=6, color="#d62728", fill=True, fill_color="#d62728", popup=last_nd["name"]).add_to(m)

        # render map
        _ = st_folium(m, width=900, height=700)

# Right column: route controls and summary
with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Trip control")
    st.write(f"**Origin:** {origin_name}")
    st.write(f"**Destination:** {destination_name}")
    st.write(f"**Transfer penalty:** {transfer_penalty} minutes")
    if st.button("Plan route"):
        if origin_id is None or destination_id is None:
            st.warning("Invalid origin or destination.")
        elif origin_id == destination_id:
            st.warning("Origin and destination are the same.")
        else:
            r = shortest_path_with_transfer_penalty(G, origin_id, destination_id, transfer_penalty=transfer_penalty)
            if not r:
                st.error("No path found between selected stops.")
            else:
                # store and trigger map refresh
                st.session_state["last_result"] = r
                st.experimental_rerun()

    if st.session_state.get("last_result"):
        res = st.session_state["last_result"]
        st.markdown("---")
        st.markdown("### Recommended route")
        total_travel = sum(leg["travel_time"] for leg in res["legs"])
        total_penalty = sum(leg["penalty"] for leg in res["legs"])
        est_total = res["total_cost"]
        eta = datetime.now() + timedelta(minutes=est_total)
        st.write(f"**Estimated total time:** {est_total:.1f} min")
        st.write(f"**Legs travel:** {total_travel:.1f} min • **Transfer penalty total:** {total_penalty:.1f} min")
        st.write(f"**ETA (now + travel):** {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("#### Steps")
        for i, leg in enumerate(res["legs"], start=1):
            # pretty mode badge
            mode = (leg.get("mode") or "unknown").lower()
            badge_class = "mode-walk"
            if "train" in mode: badge_class = "mode-train"
            elif "bus" in mode: badge_class = "mode-bus"
            elif "jeep" in mode: badge_class = "mode-jeepney"
            st.markdown(f"<div style='margin-bottom:6px;'><span class='mode-badge {badge_class}'>{leg.get('mode')}</span> <b>{leg.get('from_name')}</b> → <b>{leg.get('to_name')}</b><br>"
                        f"<small>{leg.get('route_name')} • {leg.get('travel_time'):.1f} min"
                        f"{' • +'+str(int(leg.get('penalty'))) + ' min transfer' if leg.get('penalty') and leg.get('penalty')>0 else ''}</small></div>", unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown("No route planned yet. Use the controls above to plan a trip.")
    st.markdown('</div>', unsafe_allow_html=True)
