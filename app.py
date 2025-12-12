# app_full.py — PinasPath full (rail topology + connectors, exact ride instructions)
import streamlit as st
import pandas as pd
import networkx as nx
import heapq
from datetime import datetime, timedelta
from streamlit_folium import st_folium
import folium
import os

st.set_page_config(page_title="PinasPath — Full Rail & Instructions", layout="wide")
st.title("PinasPath — Full Rail Network (LRT/MRT) & Exact Ride Instructions")
st.markdown("This prototype uses built-in LRT-1, MRT-3, and LRT-2 station topology (CSV files). It computes exact ride-by-ride instructions and sakay counts for any origin → destination.")

# ---------------- Load CSVs ----------------
@st.cache_data
def load_data(stops_path="stops_full.csv", routes_path="routes_full.csv"):
    if not os.path.exists(stops_path):
        raise FileNotFoundError(stops_path + " missing.")
    if not os.path.exists(routes_path):
        raise FileNotFoundError(routes_path + " missing.")
    stops = pd.read_csv(stops_path, dtype=str)
    routes = pd.read_csv(routes_path, dtype=str)
    # cast travel_time numeric
    routes["travel_time"] = routes["travel_time"].astype(float)
    # clean lat/lon columns (some may be empty)
    if "lat" in stops.columns and "lon" in stops.columns:
        try:
            stops["lat"] = pd.to_numeric(stops["lat"], errors="coerce")
            stops["lon"] = pd.to_numeric(stops["lon"], errors="coerce")
        except Exception:
            stops["lat"] = None; stops["lon"] = None
    else:
        stops["lat"] = None; stops["lon"] = None
    return stops, routes

stops, routes = load_data()

# Build graph (bidirectional edges are explicit in CSV; still allow auto-bidir option)
def build_graph(routes_df):
    G = nx.DiGraph()
    for _, r in routes_df.iterrows():
        u = r["from_stop"]; v = r["to_stop"]
        G.add_edge(u, v, travel_time=float(r["travel_time"]), mode=r["mode"], route_name=r["route_name"])
    return G

G = build_graph(routes)

# UI inputs
st.sidebar.header("Trip Planner")
stop_names = stops["stop_name"].tolist()
origin = st.sidebar.selectbox("Origin", stop_names, index=stop_names.index("Monumento") if "Monumento" in stop_names else 0)
destination = st.sidebar.selectbox("Destination", stop_names, index=stop_names.index("Cubao") if "Cubao" in stop_names else (1 if len(stop_names)>1 else 0))
transfer_penalty = st.sidebar.number_input("Transfer penalty (min)", value=2, min_value=0, max_value=30, step=1)
compute_all = st.sidebar.button("Compute ALL OD (save to CSV)")

# helper map display flag: only if lat/lon present
show_map = st.sidebar.checkbox("Show map (if coordinates available)", value=False)

def id_from_name(name):
    row = stops[stops["stop_name"]==name]
    if row.empty:
        return None
    return row.iloc[0]["stop_id"]

origin_id = id_from_name(origin)
destination_id = id_from_name(destination)

# shortest path with transfer penalty; returns legs list
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
            travel_time = float(e["travel_time"])
            mode = e.get("mode", "")
            route_name = e.get("route_name", "")
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
                "from": node,
                "to": nbr,
                "from_name": stops.loc[stops.stop_id==node, "stop_name"].values[0] if not stops.loc[stops.stop_id==node].empty else node,
                "to_name": stops.loc[stops.stop_id==nbr, "stop_name"].values[0] if not stops.loc[stops.stop_id==nbr].empty else nbr,
                "mode": mode,
                "route_name": route_name,
                "travel_time": travel_time,
                "penalty": add
            }]
            heapq.heappush(pq, (new_cost, nbr, mode, route_name, new_path, new_legs))
    return None

# helper to compress legs into ride instructions
def legs_to_instructions(legs):
    if not legs:
        return []
    ins = []
    cur_mode = legs[0]["mode"]
    cur_route = legs[0]["route_name"]
    start = legs[0]["from_name"]
    travel = legs[0]["travel_time"]
    end = legs[0]["to_name"]
    penalty = legs[0]["penalty"]
    for leg in legs[1:]:
        if leg["mode"] == cur_mode and leg["route_name"] == cur_route:
            # extend same ride
            travel += leg["travel_time"]
            end = leg["to_name"]
            penalty += leg["penalty"]
        else:
            ins.append({"mode": cur_mode, "route": cur_route, "from": start, "to": end, "travel_time": travel, "penalty": penalty})
            cur_mode = leg["mode"]; cur_route = leg["route_name"]; start = leg["from_name"]; end = leg["to_name"]; travel = leg["travel_time"]; penalty = leg["penalty"]
    ins.append({"mode": cur_mode, "route": cur_route, "from": start, "to": end, "travel_time": travel, "penalty": penalty})
    return ins

# UI: compute single route
col_map, col_text = st.columns([2,1])
with col_text:
    st.header("Trip summary")
    if st.button("Plan route"):
        if origin_id is None or destination_id is None:
            st.error("Invalid origin/destination selection.")
        elif origin_id == destination_id:
            st.warning("Origin and destination are the same.")
        else:
            res = shortest_path_with_transfer_penalty(G, origin_id, destination_id, transfer_penalty=transfer_penalty)
            if not res:
                st.error("No path found.")
            else:
                instructions = legs_to_instructions(res["legs"])
                # sakay count = number of ride segments where mode != walk
                sakay = sum(1 for i in instructions if i["mode"].lower() != "walk")
                total_travel = sum(l["travel_time"] for l in res["legs"])
                total_penalty = sum(l["penalty"] for l in res["legs"])
                est_total = res["total_cost"]
                eta = datetime.now() + timedelta(minutes=est_total)
                st.subheader(f"{origin} → {destination}")
                st.markdown(f"**Estimated total:** {est_total:.1f} min • **Sakay:** {sakay} • **ETA:** {eta.strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown("**Exact instructions:**")
                step = 1
                for instr in instructions:
                    mode = instr["mode"]
                    route = instr["route"]
                    frm = instr["from"]
                    to = instr["to"]
                    t = instr["travel_time"]
                    pen = instr["penalty"]
                    if mode.lower() == "walk":
                        st.write(f"{step}. Walk: {frm} → {to} — ≈ {t:.1f} min")
                    else:
                        st.write(f"{step}. Ride {route} ({mode}): {frm} → {to} — ≈ {t:.1f} min" + (f" + {pen:.1f} min transfer penalty" if pen>0 else ""))
                    step += 1
                st.markdown("---")
                # prepare map view if coordinates exist for some stations
                coords_exist = stops[["lat","lon"]].dropna().shape[0] > 0
                if coords_exist:
                    with col_map:
                        mcenter = None
                        # compute simple center from available nodes in path
                        lats = []
                        lons = []
                        for n in res["path"]:
                            row = stops[stops.stop_id==n]
                            if not row.empty and pd.notna(row.iloc[0]["lat"]):
                                lats.append(float(row.iloc[0]["lat"])); lons.append(float(row.iloc[0]["lon"]))
                        if lats and lons:
                            mcenter = [sum(lats)/len(lats), sum(lons)/len(lons)]
                        else:
                            mcenter = [14.6, 121.0]
                        fmap = folium.Map(location=mcenter, zoom_start=12, tiles="CartoDB positron")
                        # draw path nodes
                        for n in res["path"]:
                            row = stops[stops.stop_id==n]
                            if not row.empty and pd.notna(row.iloc[0]["lat"]):
                                name = row.iloc[0]["stop_name"]
                                lat = float(row.iloc[0]["lat"]); lon = float(row.iloc[0]["lon"])
                                folium.CircleMarker(location=(lat, lon), radius=5, popup=name, tooltip=name, color="#2b5876", fill=True).add_to(fmap)
                        # draw lines
                        path_coords = []
                        for n in res["path"]:
                            row = stops[stops.stop_id==n]
                            if not row.empty and pd.notna(row.iloc[0]["lat"]):
                                path_coords.append((float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])))
                        if path_coords:
                            folium.PolyLine(path_coords, color="green", weight=5, opacity=0.8).add_to(fmap)
                        _ = st_folium(fmap, width=700, height=600)
                else:
                    with col_map:
                        st.info("No coordinates for stations — map disabled. Provide lat/lon in stops_full.csv to enable map.")
    else:
        st.info("Click 'Plan route' to get exact ride instructions and sakay count.")

# Compute ALL OD (optional)
if compute_all:
    st.sidebar.info("Computing all OD pairs — this may take a moment...")
    stop_ids = list(stops["stop_id"].values)
    records = []
    for a in stop_ids:
        for b in stop_ids:
            if a == b: continue
            r = shortest_path_with_transfer_penalty(G, a, b, transfer_penalty=transfer_penalty)
            if r:
                instr = legs_to_instructions(r["legs"])
                sakay = sum(1 for i in instr if i["mode"].lower() != "walk")
                records.append({
                    "origin": a,
                    "destination": b,
                    "total_time_min": r["total_cost"],
                    "sakay_count": sakay,
                    "route_sequence": " -> ".join([f"{x['route']}({x['mode']})" for x in instr]),
                    "path_nodes": "->".join(r["path"])
                })
            else:
                records.append({"origin": a, "destination": b, "total_time_min": None, "sakay_count": None, "route_sequence": None, "path_nodes": None})
    outdf = pd.DataFrame(records)
    outpath = "all_od_routes_full.csv"
    outdf.to_csv(outpath, index=False)
    st.sidebar.success(f"All OD computed and saved to {outpath} (download from your project folder).")
