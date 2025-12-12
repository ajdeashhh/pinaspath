# app.py — Full-network loader but filenames remain stops.csv & routes.csv
import streamlit as st
import pandas as pd
import networkx as nx
import heapq
from datetime import datetime, timedelta
from streamlit_folium import st_folium
import folium
import os

st.set_page_config(page_title="PinasPath", layout="wide")
st.markdown("<h1 style='margin-bottom:6px;'>PinasPath</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;margin-top:0;'>Full network mode — uses <code>stops.csv</code> and <code>routes.csv</code>.</p>", unsafe_allow_html=True)

# ---------------- CSS
st.markdown("""
<style>
.mode-badge {display:inline-block;padding:4px 8px;border-radius:6px;color:white;font-weight:600;margin-right:6px;}
.mode-bus {background:#1f77b4;} .mode-train {background:#2ca02c;} .mode-jeepney {background:#ff7f0e;} .mode-walk {background:#7f7f7f;}
.panel {background:#fff;border-radius:8px;padding:12px;box-shadow:0 2px 10px rgba(0,0,0,0.06);}
</style>
""", unsafe_allow_html=True)

# ---------------- Load CSVs
@st.cache_data
def load_data(stops_path="stops.csv", routes_path="routes.csv", auto_bidir=True):
    if not os.path.exists(stops_path) or not os.path.exists(routes_path):
        raise FileNotFoundError("Make sure stops.csv and routes.csv exist in this folder.")
    stops = pd.read_csv(stops_path, dtype=str)
    routes = pd.read_csv(routes_path, dtype=str)
    # parse numeric columns
    if "lat" in stops.columns and "lon" in stops.columns:
        stops["lat"] = pd.to_numeric(stops["lat"], errors="coerce")
        stops["lon"] = pd.to_numeric(stops["lon"], errors="coerce")
    else:
        stops["lat"] = None; stops["lon"] = None
    routes["travel_time"] = pd.to_numeric(routes["travel_time"], errors="coerce").fillna(1.0)
    # optionally add reverse edges (make bidirectional) if CSV has only single direction rows
    if auto_bidir:
        extra = []
        for _, r in routes.iterrows():
            extra.append({"from_stop": r["to_stop"], "to_stop": r["from_stop"], "travel_time": r["travel_time"], "mode": r.get("mode",""), "route_name": r.get("route_name","")})
        if extra:
            routes = pd.concat([routes, pd.DataFrame(extra)], ignore_index=True)
    return stops, routes

try:
    stops, routes = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---------------- Build graph
def build_graph(routes_df):
    G = nx.DiGraph()
    for _, r in routes_df.iterrows():
        u = str(r["from_stop"]); v = str(r["to_stop"])
        G.add_edge(u, v, travel_time=float(r["travel_time"]), mode=r.get("mode",""), route_name=r.get("route_name",""))
    return G

G = build_graph(routes)

# map-availability check
coords_available = stops[["lat","lon"]].dropna().shape[0] > 0

# ---------------- UI: sidebar
st.sidebar.header("Trip planner")
stop_names = stops["stop_name"].tolist()
origin_name = st.sidebar.selectbox("Origin", stop_names, index=0)
destination_name = st.sidebar.selectbox("Destination", stop_names, index=1 if len(stop_names)>1 else 0)
transfer_penalty = st.sidebar.number_input("Transfer penalty (min)", min_value=0, max_value=30, value=2, step=1, help="Extra minutes added on mode/route change.")
show_map = st.sidebar.checkbox("Show map (if coords available)", value=True)

# helpers
def name_to_id(name):
    row = stops[stops["stop_name"]==name]
    return row.iloc[0]["stop_id"] if not row.empty else None

origin = name_to_id(origin_name)
destination = name_to_id(destination_name)

# ---------------- Shortest path with transfer penalty (Dijkstra-like)
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
            tt = float(e.get("travel_time", 0.0))
            mode = e.get("mode","")
            route = e.get("route_name","")
            add = 0.0
            if prev_mode is not None:
                if mode != prev_mode:
                    add = transfer_penalty
                else:
                    if prev_route and route and prev_route != route:
                        add = transfer_penalty
            new_cost = cost + tt + add
            new_path = path + [nbr]
            new_legs = legs + [{
                "from": node, "to": nbr,
                "from_name": stops.loc[stops.stop_id==node, "stop_name"].values[0] if not stops.loc[stops.stop_id==node].empty else node,
                "to_name": stops.loc[stops.stop_id==nbr, "stop_name"].values[0] if not stops.loc[stops.stop_id==nbr].empty else nbr,
                "mode": mode, "route": route, "travel_time": tt, "penalty": add
            }]
            heapq.heappush(pq, (new_cost, nbr, mode, route, new_path, new_legs))
    return None

# compress contiguous legs into ride instructions
def legs_to_instructions(legs):
    if not legs:
        return []
    instr = []
    cur = legs[0].copy()
    for leg in legs[1:]:
        if leg["mode"] == cur["mode"] and leg["route"] == cur["route"]:
            cur["to_name"] = leg["to_name"]
            cur["travel_time"] += leg["travel_time"]
            cur["penalty"] += leg["penalty"]
        else:
            instr.append(cur)
            cur = leg.copy()
    instr.append(cur)
    return instr

# ---------------- Layout
left, right = st.columns([2,1])

# left: map area (single map)
with left:
    if show_map and coords_available and "last_route" not in st.session_state:
        center = [stops["lat"].dropna().mean(), stops["lon"].dropna().mean()]
        m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
        for _, r in stops.iterrows():
            if pd.notna(r["lat"]) and pd.notna(r["lon"]):
                folium.CircleMarker(location=(float(r["lat"]), float(r["lon"])), radius=3, color="#2b5876", fill=True, popup=r["stop_name"]).add_to(m)
        _ = st_folium(m, width=900, height=700)
    elif show_map and not coords_available:
        st.info("No coordinates found in stops.csv — map disabled. Add lat/lon to enable map.")
    else:
        st.write("")  # placeholder

# right: controls + instructions
with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Trip control")
    st.write(f"**Origin:** {origin_name}")
    st.write(f"**Destination:** {destination_name}")
    st.write(f"**Transfer penalty:** {transfer_penalty} min")
    if st.button("Plan route"):
        if origin is None or destination is None:
            st.warning("Invalid origin or destination.")
        elif origin == destination:
            st.warning("Origin and destination are the same.")
        else:
            res = shortest_path_with_transfer_penalty(G, origin, destination, transfer_penalty=transfer_penalty)
            if not res:
                st.error("No path found.")
            else:
                st.session_state["last_route"] = res
                st.experimental_rerun()

    if st.session_state.get("last_route"):
        res = st.session_state["last_route"]
        st.markdown("---")
        st.markdown("### Recommended route")
        total_travel = sum(l["travel_time"] for l in res["legs"])
        total_penalty = sum(l["penalty"] for l in res["legs"])
        total = res["total_cost"]
        sakay = sum(1 for leg in legs_to_instructions(res["legs"]) if leg["mode"].lower() != "walk")
        eta = datetime.now() + timedelta(minutes=total)
        st.write(f"**Estimated total:** {total:.1f} min  •  **Sakay:** {sakay}  •  **ETA:** {eta.strftime('%Y-%m-%d %H:%M')}")
        st.markdown("#### Steps")
        instructions = legs_to_instructions(res["legs"])
        for i, leg in enumerate(instructions, 1):
            mode = leg["mode"] or "walk"
            badge = "mode-walk"
            if "train" in mode.lower(): badge = "mode-train"
            elif "bus" in mode.lower(): badge = "mode-bus"
            elif "jeep" in mode.lower(): badge = "mode-jeepney"
            txt = f"{i}. <span class='mode-badge {badge}'>{mode}</span> <b>{leg['from_name']}</b> → <b>{leg['to_name']}</b> — {leg['travel_time']:.1f} min"
            if leg.get("penalty",0) > 0:
                txt += f" (+{leg['penalty']:.1f} min transfer)"
            st.markdown(txt, unsafe_allow_html=True)
        st.markdown("---")
        # draw route map if coords available
        if show_map and coords_available:
            path_coords = []
            for n in res["path"]:
                row = stops[stops.stop_id==n]
                if not row.empty and pd.notna(row.iloc[0]["lat"]):
                    path_coords.append((float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])))
            # center map on path
            if path_coords:
                m2 = folium.Map(location=path_coords[0], zoom_start=13, tiles="CartoDB positron")
                for idx, (lat, lon) in enumerate(path_coords):
                    folium.CircleMarker(location=(lat, lon), radius=6, color="#2b5876", fill=True, popup=f"{idx+1}").add_to(m2)
                folium.PolyLine(path_coords, color="green", weight=6).add_to(m2)
                _ = st_folium(m2, width=900, height=500)
    else:
        st.info("No route planned. Use 'Plan route' to generate instructions.")
    st.markdown('</div>', unsafe_allow_html=True)

# bottom: helpful note
st.markdown("<small>Note: this app reads your full network from <code>stops.csv</code> and <code>routes.csv</code>. Ensure they contain the expanded stations & edges you want.</small>", unsafe_allow_html=True)
