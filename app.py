# app.py
# PinasPath — Google-Maps-like autocomplete + route recommendation + ETA
# Run: streamlit run app.py
# Requirements: see requirements.txt below

import streamlit as st
import requests
import pandas as pd
import networkx as nx
import math
import time
import heapq
from streamlit_folium import st_folium
import folium
from datetime import datetime, timedelta

st.set_page_config(page_title="PinasPath — Autocomplete + Route", layout="wide")
st.title("PinasPath — Autocomplete & Route Recommendation")
st.write("Type addresses and select suggestions (appears while typing). Then click **Plan route** to get a recommended commute and ETA.")

# ----------------- Utilities -----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# Autocomplete using Nominatim "search" endpoint (behaves like suggestions)
@st.cache_data(ttl=300)
def nominatim_suggest(q, limit=6):
    if not q or len(q.strip()) < 3:
        return []
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": limit, "addressdetails": 0}
    headers = {"User-Agent": "PinasPathPrototype/1.0 (youremail@example.com)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json()
            time.sleep(0.4)  # be polite
            out = [{"display_name": it.get("display_name"), "lat": float(it.get("lat")), "lon": float(it.get("lon"))} for it in data]
            return out
    except Exception:
        return []
    return []

def overpass_fetch_stops_and_routes(lat, lon, radius_m=2000):
    overpass_url = "https://overpass-api.de/api/interpreter"
    node_q = (
        f"[out:json][timeout:60];"
        f"("
        f"node(around:{radius_m},{lat},{lon})[highway=bus_stop];"
        f"node(around:{radius_m},{lat},{lon})[public_transport=platform];"
        f"node(around:{radius_m},{lat},{lon})[railway=station];"
        f"node(around:{radius_m},{lat},{lon})[railway=halt];"
        f");out body;"
    )
    rel_q = (
        f"[out:json][timeout:60];"
        f"relation(around:{radius_m},{lat},{lon})[route~\"bus|tram|train|subway|light_rail|ferry\"];"
        f"out body; >; out skel qt;"
    )
    nodes = {}
    relations = []
    try:
        nr = requests.post(overpass_url, data=node_q, timeout=60)
        nr.raise_for_status()
        nd = nr.json()
        for el in nd.get("elements", []):
            if el["type"] == "node":
                nid = el["id"]
                nodes[nid] = {"id": nid, "lat": el.get("lat"), "lon": el.get("lon"), "tags": el.get("tags", {})}
        rr = requests.post(overpass_url, data=rel_q, timeout=60)
        rr.raise_for_status()
        rd = rr.json()
        for el in rd.get("elements", []):
            if el["type"] == "relation":
                rel = {"id": el["id"], "tags": el.get("tags", {}), "members": []}
                for m in el.get("members", []):
                    if m.get("type") == "node":
                        rel["members"].append({"type": "node", "ref": m.get("ref"), "role": m.get("role")})
                relations.append(rel)
        for el in rd.get("elements", []):
            if el["type"] == "node":
                nid = el["id"]
                if nid not in nodes:
                    nodes[nid] = {"id": nid, "lat": el.get("lat"), "lon": el.get("lon"), "tags": el.get("tags", {})}
    except Exception as e:
        st.warning("Overpass fetch failed or timed out. Try increasing radius or retry.")
    return nodes, relations

# Build graph from fetched nodes & relations
def build_graph_from_osm(nodes, relations, walking_threshold_km=0.6):
    # nodes: dict id-> {lat,lon,tags}
    G = nx.DiGraph()
    # add nodes
    for nid, nd in nodes.items():
        G.add_node(str(nid), name=nd["tags"].get("name") or nd["tags"].get("ref") or f"stop_{nid}",
                   lat=float(nd["lat"]), lon=float(nd["lon"]))
    # add transit edges from relations (sequence-based)
    transit_added = 0
    for rel in relations:
        rtags = rel.get("tags", {})
        route_type = rtags.get("route", "bus")
        route_name = rtags.get("ref") or rtags.get("name") or f"route_{rel['id']}"
        seq = [str(m["ref"]) for m in rel.get("members", []) if m.get("ref") in nodes]
        speed_kmph = 25.0
        if route_type in ("train", "subway", "light_rail"):
            speed_kmph = 50.0
        elif route_type == "tram":
            speed_kmph = 30.0
        elif route_type == "ferry":
            speed_kmph = 20.0
        for i in range(len(seq)-1):
            a = seq[i]; b = seq[i+1]
            la, lo = nodes[int(a)]["lat"], nodes[int(a)]["lon"]
            lb, lb2 = nodes[int(b)]["lat"], nodes[int(b)]["lon"]
            dist_km = haversine(la, lo, lb, lb2)
            travel_min = (dist_km / speed_kmph) * 60.0
            G.add_edge(a, b, travel_time=travel_min, mode=route_type, route_name=route_name)
            G.add_edge(b, a, travel_time=travel_min, mode=route_type, route_name=route_name)
            transit_added += 1
    # walking edges between nearby stops
    items = list(nodes.items())
    for i in range(len(items)):
        id_a, a = items[i]
        for j in range(i+1, len(items)):
            id_b, b = items[j]
            d = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if d <= walking_threshold_km:
                walk_min = (d / 5.0) * 60.0  # fixed walking speed 5 km/h
                G.add_edge(str(id_a), str(id_b), travel_time=walk_min, mode="walk", route_name="walk")
                G.add_edge(str(id_b), str(id_a), travel_time=walk_min, mode="walk", route_name="walk")
    return G, transit_added

# Dijkstra-like pathfinder that charges transfer penalty when mode/route changes
def find_fastest_route(G, origin_stop, dest_stop, transfer_penalty=2.0):
    pq = []
    heapq.heappush(pq, (0.0, origin_stop, None, None, [origin_stop], []))
    visited = {}
    while pq:
        cost, node, prev_mode, prev_route, path, legs = heapq.heappop(pq)
        key = (node, prev_mode, prev_route)
        if key in visited and visited[key] <= cost:
            continue
        visited[key] = cost
        if node == dest_stop:
            return {"total_cost": cost, "path": path, "legs": legs}
        for nbr in G.neighbors(node):
            e = G[node][nbr]
            mode = e.get("mode")
            route = e.get("route_name")
            t = float(e.get("travel_time", 0.0))
            add = 0.0
            if prev_mode is not None and (mode != prev_mode or route != prev_route):
                add = float(transfer_penalty)
            new_cost = cost + t + add
            new_path = path + [nbr]
            new_legs = legs + [{"from": node, "to": nbr, "mode": mode, "route": route, "travel_time": t, "penalty": add}]
            heapq.heappush(pq, (new_cost, nbr, mode, route, new_path, new_legs))
    return None

# ----------------- Layout: left = inputs, right = map & results -----------------
col1, col2 = st.columns([1, 2])
with col1:
    st.header("Origin & Destination")

    # ORIGIN input + live suggestions
    origin_input = st.text_input("Origin (type an address or place)", key="origin_input")
    origin_suggestions = []
    if origin_input and len(origin_input.strip()) >= 3:
        # call suggestions (cached)
        origin_suggestions = nominatim_suggest(origin_input, limit=6)
    if origin_suggestions:
        labels = [f"{i+1}. {s['display_name']}" for i, s in enumerate(origin_suggestions)]
        sel = st.selectbox("Select origin suggestion", ["(choose)"] + labels, key="origin_sel")
        if sel != "(choose)":
            idx = labels.index(sel)
            choice = origin_suggestions[idx]
            st.session_state["origin_choice"] = choice
            st.success("Origin selected")
    # show chosen origin (if any)
    if st.session_state.get("origin_choice"):
        oc = st.session_state["origin_choice"]
        st.markdown(f"**Chosen origin:** {oc['display_name']}  \n(lat: {oc['lat']:.6f}, lon: {oc['lon']:.6f})")
    else:
        st.info("Start typing to see suggestions (3+ chars)")

    st.write("---")

    # DESTINATION input + live suggestions
    dest_input = st.text_input("Destination (type an address or place)", key="dest_input")
    dest_suggestions = []
    if dest_input and len(dest_input.strip()) >= 3:
        dest_suggestions = nominatim_suggest(dest_input, limit=6)
    if dest_suggestions:
        labels2 = [f"{i+1}. {s['display_name']}" for i, s in enumerate(dest_suggestions)]
        sel2 = st.selectbox("Select destination suggestion", ["(choose)"] + labels2, key="dest_sel")
        if sel2 != "(choose)":
            idx2 = labels2.index(sel2)
            choice2 = dest_suggestions[idx2]
            st.session_state["dest_choice"] = choice2
            st.success("Destination selected")
    if st.session_state.get("dest_choice"):
        dc = st.session_state["dest_choice"]
        st.markdown(f"**Chosen destination:** {dc['display_name']}  \n(lat: {dc['lat']:.6f}, lon: {dc['lon']:.6f})")
    else:
        st.info("Type destination to show suggestions (3+ chars)")

    st.write("---")
    st.write("Options:")
    radius_km = st.number_input("Stops fetch radius (km)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    transfer_penalty = st.number_input("Transfer penalty (min)", min_value=0, max_value=10, value=2, step=1)
    plan_btn = st.button("Plan route")

with col2:
    st.header("Map & Route")
    # empty map placeholder, will update later
    map_placeholder = st.empty()

# ----------------- Plan route action -----------------
if plan_btn:
    if not st.session_state.get("origin_choice") or not st.session_state.get("dest_choice"):
        st.error("Please select both origin and destination from suggestions.")
    else:
        origin = st.session_state["origin_choice"]
        dest = st.session_state["dest_choice"]
        ox, oy = origin["lat"], origin["lon"]
        dx, dy = dest["lat"], dest["lon"]
        center_lat = (ox + dx) / 2.0
        center_lon = (oy + dy) / 2.0

        st.info("Fetching nearby stops & building graph (OSM Overpass). This may take a few seconds...")
        nodes, relations = overpass_fetch_stops_and_routes(center_lat, center_lon, radius_m=int(radius_km*1000))
        if not nodes:
            st.error("No nearby stops found (try increasing radius).")
        else:
            G, transit_edges_count = build_graph_from_osm(nodes, relations, walking_threshold_km=0.6)
            st.success(f"Built graph — {len(G.nodes)} stops, ~{transit_edges_count} transit edges (edges also include walking links).")

            # find nearest stops to origin & dest
            stops_df = pd.DataFrame([{"stop_id": nid, "lat": nd["lat"], "lon": nd["lon"], "name": nd["tags"].get("name") or nd["tags"].get("ref") or f"stop_{nid}"} for nid, nd in nodes.items()])
            stops_df["dist_o"] = stops_df.apply(lambda r: haversine(ox, oy, float(r["lat"]), float(r["lon"])), axis=1)
            stops_df["dist_d"] = stops_df.apply(lambda r: haversine(dx, dy, float(r["lat"]), float(r["lon"])), axis=1)
            origin_row = stops_df.loc[stops_df["dist_o"].idxmin()]
            dest_row = stops_df.loc[stops_df["dist_d"].idxmin()]
            o_stop = str(int(origin_row["stop_id"]))
            d_stop = str(int(dest_row["stop_id"]))
            walk_o_min = (origin_row["dist_o"] / 5.0) * 60.0
            walk_d_min = (dest_row["dist_d"] / 5.0) * 60.0

            st.write(f"Nearest origin stop: **{origin_row['name']}** — {origin_row['dist_o']*1000:.0f} m away (walk ≈ {walk_o_min:.1f} min)")
            st.write(f"Nearest dest stop: **{dest_row['name']}** — {dest_row['dist_d']*1000:.0f} m away (walk ≈ {walk_d_min:.1f} min)")

            # run pathfinder
            res = find_fastest_route(G, o_stop, d_stop, transfer_penalty=float(transfer_penalty))
            if not res:
                st.error("No transit route found between nearest stops. The app still shows walking options between stops if available.")
            else:
                total_transit = sum([leg["travel_time"] for leg in res["legs"] if leg["mode"] != "walk"])
                total_walk_internal = sum([leg["travel_time"] for leg in res["legs"] if leg["mode"] == "walk"])
                total_penalties = sum([leg["penalty"] for leg in res["legs"]])
                estimated_total_min = walk_o_min + total_transit + total_walk_internal + walk_d_min + total_penalties

                st.markdown("## Recommended Route (fastest estimate)")
                st.markdown(f"**Estimated total duration:** {estimated_total_min:.1f} minutes")
                eta = datetime.now() + timedelta(minutes=estimated_total_min)
                st.markdown(f"**Estimated arrival time (ETA):** {eta.strftime('%Y-%m-%d %H:%M:%S')}")

                st.markdown("### Legs (walk + transit)")
                # first walking to origin stop
                st.write(f"1. Walk from origin input to stop **{origin_row['name']}** — ≈ {walk_o_min:.1f} min")
                idx = 2
                for leg in res["legs"]:
                    from_name = G.nodes[leg["from"]]["name"]
                    to_name = G.nodes[leg["to"]]["name"]
                    mode = leg["mode"]
                    route = leg["route"]
                    t = leg["travel_time"]
                    penalty = leg["penalty"]
                    if mode == "walk":
                        st.write(f"{idx}. Walk: {from_name} → {to_name} — {t:.1f} min")
                    else:
                        st.write(f"{idx}. {mode.title()}: {from_name} → {to_name} — {route} — {t:.1f} min" + (f" + {penalty:.1f} min transfer" if penalty>0 else ""))
                    idx += 1
                st.write(f"{idx}. Walk from stop **{dest_row['name']}** to destination input — ≈ {walk_d_min:.1f} min")

                # Map visualization
                m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                # add origin & dest markers
                folium.Marker(location=(ox, oy), popup="Origin (selected)", icon=folium.Icon(color="blue")).add_to(m)
                folium.Marker(location=(dx, dy), popup="Destination (selected)", icon=folium.Icon(color="red")).add_to(m)
                # add nearest stops markers and small markers for all stops
                for _, r in stops_df.iterrows():
                    folium.CircleMarker(location=(float(r["lat"]), float(r["lon"])), radius=2, color="#666", fill=True, fill_opacity=0.5).add_to(m)
                folium.Marker(location=(float(origin_row["lat"]), float(origin_row["lon"])), popup=f"Origin stop: {origin_row['name']}", icon=folium.Icon(color="green")).add_to(m)
                folium.Marker(location=(float(dest_row["lat"]), float(dest_row["lon"])), popup=f"Dest stop: {dest_row['name']}", icon=folium.Icon(color="green")).add_to(m)

                # draw transit polyline along legs
                coords = []
                for leg in res["legs"]:
                    node = G.nodes[leg["from"]]
                    coords.append((node["lat"], node["lon"]))
                coords.append((G.nodes[res["path"][-1]]["lat"], G.nodes[res["path"][-1]]["lon"]))
                folium.PolyLine(coords, color="green", weight=5, opacity=0.8).add_to(m)

                # walking lines to/from user points
                folium.PolyLine([(ox, oy), (float(origin_row["lat"]), float(origin_row["lon"]))], color="blue", weight=3, dash_array="5,8").add_to(m)
                folium.PolyLine([(float(dest_row["lat"]), float(dest_row["lon"])), (dx, dy)], color="red", weight=3, dash_array="5,8").add_to(m)

                map_placeholder.write(st_folium(m, width=900, height=600))

# small footer
st.markdown("---")
st.caption("Autocomplete uses Nominatim (OpenStreetMap). Routes and stops are inferred from OSM (Overpass). Times are estimates. For production-grade routing use GTFS + routing engine + licensed geocoding.")
