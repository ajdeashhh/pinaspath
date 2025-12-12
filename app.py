# app.py — Full PinasPath app (safe map rendering, ignores commented CSV lines)
import streamlit as st
import pandas as pd
import networkx as nx
import heapq
from datetime import datetime, timedelta
from streamlit_folium import st_folium
import folium
import os

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
    """
    Load stops.csv and routes.csv.
    - Ignores lines starting with '#' so appended comment blocks won't break parsing.
    - Coerces lat/lon to numeric (NaNs allowed).
    """
    if not os.path.exists(stops_path):
        raise FileNotFoundError(f"{stops_path} not found.")
    if not os.path.exists(routes_path):
        raise FileNotFoundError(f"{routes_path} not found.")
    # ignore comment lines that start with '#'
    stops = pd.read_csv(stops_path, dtype=str, comment="#")
    routes = pd.read_csv(routes_path, dtype=str, comment="#")
    # normalize and coerce numeric fields
    if "lat" in stops.columns and "lon" in stops.columns:
        stops["lat"] = pd.to_numeric(stops["lat"], errors="coerce")
        stops["lon"] = pd.to_numeric(stops["lon"], errors="coerce")
    else:
        stops["lat"] = None
        stops["lon"] = None
    routes["travel_time"] = pd.to_numeric(routes["travel_time"], errors="coerce").fillna(1.0)
    # ensure ID columns exist and are strings
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
        sid = str(r["stop_id"])
        # set safe lat/lon as float if present
        lat = float(r["lat"]) if pd.notna(r["lat"]) else None
        lon = float(r["lon"]) if pd.notna(r["lon"]) else None
        G.add_node(sid, name=r["stop_name"], lat=lat, lon=lon)
    for _, r in routes_df.iterrows():
        u = str(r["from_stop"]); v = str(r["to_stop"])
        w = float(r["travel_time"])
        mode = r.get("mode", "") if r.get("mode", "") is not None else ""
        rn = r.get("route_name", "") if r.get("route_name", "") is not None else ""
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

# build graph (do not auto-add reverse edges because you appended both directions earlier)
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

# Helper: only use valid lat/lon rows
valid_coords = stops[stops["lat"].notna() & stops["lon"].notna()]

# Render map in left column (only one map rendered)
with left_col:
    if st.session_state.get("last_result") is None:
        # show base map — use only valid coordinates, fall back to info if none
        if not valid_coords.empty and show_map:
            base_center = [valid_coords["lat"].astype(float).mean(), valid_coords["lon"].astype(float).mean()]
            base_map = folium.Map(location=base_center, zoom_start=12, tiles="CartoDB positron")
            # show stops markers (only where lat/lon present)
            for _, r in valid_coords.iterrows():
                folium.CircleMarker(
                    location=(float(r["lat"]), float(r["lon"])),
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
            st.info("No valid lat/lon coordinates found in stops.csv — map disabled. Please add numeric lat,lon for stops or ensure stops.csv has no comment lines inside data rows.")
    else:
        # show route map
        res = st.session_state["last_result"]
        orig = res["path"][0]
        lat0 = G.nodes[orig]["lat"]; lon0 = G.nodes[orig]["lon"]
        # if lat/lon missing for origin, fall back to overall mean if available
        if lat0 is None or lon0 is None:
            if not valid_coords.empty:
                lat0 = float(valid_coords["lat"].mean())
                lon0 = float(valid_coords["lon"].mean())
            else:
                # can't show map
                st.info("Route found but no coordinates available to render map.")
                lat0 = None; lon0 = None
        if lat0 is not None and lon0 is not None and show_map:
            m = folium.Map(location=[lat0, lon0], zoom_start=13, tiles="CartoDB positron")
            coords = []
            # draw polyline segments with colored styling per mode
            for idx, leg in enumerate(res["legs"]):
                u = leg["from_id"]; v = leg["to_id"]
                udata = G.nodes[u]; vdata = G.nodes[v]
                # only plot if these nodes have numeric coords
                if udata.get("lat") is not None and udata.get("lon") is not None:
                    coords.append((udata["lat"], udata["lon"]))
                # color by mode
                mode = (leg.get("mode") or "").lower()
                color = "#7f7f7f"
                if "train" in mode: color = "#2ca02c"
                elif "bus" in mode: color = "#1f77b4"
                elif "jeep" in mode or "jeepney" in mode: color = "#ff7f0e"
                elif "walk" in mode: color = "#7f7f7f"
                # add segment if both endpoints have coords
                if udata.get("lat") is not None and udata.get("lon") is not None and vdata.get("lat") is not None and vdata.get("lon") is not None:
                    folium.PolyLine(
                        [(udata["lat"], udata["lon"]), (vdata["lat"], vdata["lon"])],
                        weight=6 if mode != "walk" else 3,
                        color=color,
                        opacity=0.9 if mode != "walk" else 0.6
                    ).add_to(m)
                # add marker for the from-node if coords exist
                if udata.get("lat") is not None and udata.get("lon") is not None:
                    popup = f"<b>{udata['name']}</b><br>{leg['mode']} {leg['route_name']}<br>{leg['travel_time']:.1f} min"
                    folium.CircleMarker(location=(udata["lat"], udata["lon"]), radius=6, color=color, fill=True, fill_color=color, popup=popup).add_to(m)
            # add last node marker if coords exist
            last_node = res["path"][-1]
            last_nd = G.nodes[last_node]
            if last_nd.get("lat") is not None and last_nd.get("lon") is not None:
                folium.CircleMarker(location=(last_nd["lat"], last_nd["lon"]), radius=6, color="#d62728", fill=True, fill_color="#d62728", popup=last_nd["name"]).add_to(m)
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
            st.rerun()

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

# helpful note
st.markdown("<small>Note: CSV comment lines starting with '#' are ignored. Ensure stops.csv contains numeric lat,lon for map rendering.</small>", unsafe_allow_html=True)
