# app.py
# PinasPath — Manila-wide stops: autocomplete + route recommendation + possible rides
# Run: streamlit run app.py
# Requirements: streamlit, pandas, networkx, folium, streamlit-folium, requests

import streamlit as st
import requests
import pandas as pd
import networkx as nx
import math, time, heapq
from streamlit_folium import st_folium
import folium
from datetime import datetime, timedelta

st.set_page_config(page_title="PinasPath — Manila-wide", layout="wide")
st.title("PinasPath — Manila-wide Route Recommendation")
st.write("Autocomplete addresses, pick origin/destination, then get a recommended commute across Manila with ETA and possible rides.")

# ------------------ Utilities ------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# Resilient Nominatim suggestions (cached)
@st.cache_data(ttl=300)
def nominatim_suggest(q, limit=6):
    if not q or len(q.strip()) < 3:
        return []
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q + " Manila", "format": "json", "limit": limit, "addressdetails": 0}  # bias toward Manila
    headers = {"User-Agent": "PinasPath/1.0 (contact@example.com)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            time.sleep(0.15)
            return [{"display_name": it.get("display_name"), "lat": float(it.get("lat")), "lon": float(it.get("lon"))} for it in r.json()]
    except Exception:
        return []
    return []

# ------------------ Overpass: fetch Manila stops & routes ------------------
# This query fetches nodes (bus stops/platforms/stations) and relations for routes inside the City of Manila boundary.
# It's a fairly large query but we cache results for 24 hours to avoid repeating.
@st.cache_data(ttl=24*3600)
def fetch_manila_osm(radius_m=3000):
    """
    Fetch nodes and relations for Manila from Overpass.
    Returns (nodes_dict, relations_list)
    nodes_dict: id -> {"id", "lat","lon","tags"}
    relations_list: list of relations with members (node refs)
    """
    overpass = "https://overpass-api.de/api/interpreter"

    # Attempt to find area for Manila (admin boundary). If missing, fallback to bounding box roughly covering Manila.
    # First, try area by name "Manila" and admin_level 8 (city).
    area_query = ('[out:json][timeout:120];'
                  'area["name"="Manila"]["admin_level"="8"];out ids;')
    try:
        ar = requests.post(overpass, data=area_query, timeout=60)
        ar.raise_for_status()
        area_resp = ar.json()
        if area_resp.get("elements"):
            area_id = area_resp["elements"][0]["id"]
            area_osm_id = area_id
            # Overpass area ids need to be '360' + relation id for area() usage sometimes; use area(area_id)
            # Compose node and relation queries scoped to area
            node_q = f'[out:json][timeout:120];(node(area:{area_osm_id})[highway=bus_stop];node(area:{area_osm_id})[public_transport=platform];node(area:{area_osm_id})[railway=station];node(area:{area_osm_id})[railway=halt];);out body;'
            rel_q = f'[out:json][timeout:120];relation(area:{area_osm_id})[route~"bus|tram|train|subway|light_rail|ferry"];out body;>;out skel qt;'
        else:
            raise Exception("Area not found")
    except Exception:
        # Fallback bounding box roughly around City of Manila
        # bbox: south, west, north, east
        # Using approximate Manila extents
        bbox = "14.4960,120.9567,14.6760,121.0500"
        node_q = f'[out:json][timeout:120];(node({bbox})[highway=bus_stop];node({bbox})[public_transport=platform];node({bbox})[railway=station];node({bbox})[railway=halt];);out body;'
        rel_q = f'[out:json][timeout:120];relation({bbox})[route~"bus|tram|train|subway|light_rail|ferry"];out body;>;out skel qt;'

    nodes = {}
    relations = []
    # Fetch nodes
    try:
        nr = requests.post(overpass, data=node_q, timeout=120)
        nr.raise_for_status()
        nd = nr.json()
        for el in nd.get("elements", []):
            if el.get("type") == "node":
                nid = el["id"]
                nodes[nid] = {"id": nid, "lat": el.get("lat"), "lon": el.get("lon"), "tags": el.get("tags", {})}
    except Exception as e:
        st.warning(f"Overpass nodes fetch failed: {e}")
    # Fetch relations (routes)
    try:
        rr = requests.post(overpass, data=rel_q, timeout=120)
        rr.raise_for_status()
        rd = rr.json()
        # accumulate relations and also any additional nodes included in the response
        for el in rd.get("elements", []):
            if el.get("type") == "relation":
                rel = {"id": el["id"], "tags": el.get("tags", {}), "members": []}
                for m in el.get("members", []):
                    if m.get("type") == "node":
                        rel["members"].append({"type": "node", "ref": m.get("ref"), "role": m.get("role")})
                relations.append(rel)
            elif el.get("type") == "node":
                nid = el["id"]
                if nid not in nodes:
                    nodes[nid] = {"id": nid, "lat": el.get("lat"), "lon": el.get("lon"), "tags": el.get("tags", {})}
    except Exception as e:
        st.warning(f"Overpass relations fetch failed: {e}")

    return nodes, relations

# ------------------ Build Manila-wide graph ------------------
@st.cache_data(ttl=24*3600)
def build_manila_graph():
    nodes, relations = fetch_manila_osm()
    if not nodes:
        return None, None, None
    # Build nodes dataframe
    stops_list = []
    for nid, nd in nodes.items():
        name = nd["tags"].get("name") or nd["tags"].get("ref") or f"stop_{nid}"
        stops_list.append({"stop_id": str(nid), "stop_name": name, "lat": nd["lat"], "lon": nd["lon"], "tags": nd["tags"]})
    stops_df = pd.DataFrame(stops_list)

    # Graph
    G = nx.DiGraph()
    for _, r in stops_df.iterrows():
        G.add_node(r["stop_id"], name=r["stop_name"], lat=float(r["lat"]), lon=float(r["lon"]), tags=r["tags"])

    # Transit edges from relations (ordered members)
    transit_edges = 0
    for rel in relations:
        rtags = rel.get("tags", {})
        route_type = rtags.get("route", "bus")
        route_name = rtags.get("ref") or rtags.get("name") or f"route_{rel['id']}"
        # sequence of node refs present in nodes
        seq = [str(m["ref"]) for m in rel.get("members", []) if m.get("ref") in nodes]
        # speed heuristics
        speed_kmph = 25.0
        if route_type in ("train", "subway", "light_rail"):
            speed_kmph = 45.0
        elif route_type == "tram":
            speed_kmph = 30.0
        for i in range(len(seq)-1):
            a, b = seq[i], seq[i+1]
            la, lo = nodes[int(a)]["lat"], nodes[int(a)]["lon"]
            lb, lb2 = nodes[int(b)]["lat"], nodes[int(b)]["lon"]
            dist_km = haversine(la, lo, lb, lb2)
            travel_min = (dist_km / speed_kmph) * 60.0
            # Add edge and record route_name in edge attributes; allow multiple route_names per (u,v)
            if G.has_edge(a, b):
                # append route_name to list in attribute
                existing = G[a][b].get("route_names", [])
                existing.append({"route_name": route_name, "mode": route_type})
                G[a][b]["route_names"] = existing
            else:
                G.add_edge(a, b, travel_time=travel_min, mode=route_type, route_names=[{"route_name": route_name, "mode": route_type}])
            # also add reverse (bidirectional)
            if G.has_edge(b, a):
                existing = G[b][a].get("route_names", [])
                existing.append({"route_name": route_name, "mode": route_type})
                G[b][a]["route_names"] = existing
            else:
                G.add_edge(b, a, travel_time=travel_min, mode=route_type, route_names=[{"route_name": route_name, "mode": route_type}])
            transit_edges += 1

    # Walking edges: connect stops within threshold (0.6 km)
    walking_threshold_km = 0.6
    items = list(nodes.items())
    for i in range(len(items)):
        id_a, a = items[i]
        for j in range(i+1, len(items)):
            id_b, b = items[j]
            d = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if d <= walking_threshold_km:
                walk_min = (d / 5.0) * 60.0
                if not G.has_edge(str(id_a), str(id_b)):
                    G.add_edge(str(id_a), str(id_b), travel_time=walk_min, mode="walk", route_names=[{"route_name":"walk","mode":"walk"}])
                if not G.has_edge(str(id_b), str(id_a)):
                    G.add_edge(str(id_b), str(id_a), travel_time=walk_min, mode="walk", route_names=[{"route_name":"walk","mode":"walk"}])

    return G, stops_df, transit_edges

# ------------------ Pathfinding with transfer handling ------------------
def find_fastest_path(G, origin_stop, dest_stop, transfer_penalty=2.0):
    """
    Dijkstra-like search that tracks previous route/mode and adds penalty on changes.
    It returns legs with route_names lists where possible.
    """
    pq = []
    heapq.heappush(pq, (0.0, origin_stop, None, None, [origin_stop], []))
    visited = dict()
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
            t = float(e.get("travel_time", 0.0))
            # pick mode from first route_names entry if exists
            rn = e.get("route_names", [])
            mode = rn[0]["mode"] if rn else e.get("mode", "unknown")
            # if route_names exists, multiple possible rides connect this link; keep the full list for "possible rides"
            possible_rides = rn if rn else [{"route_name":"walk","mode":"walk"}]
            add = 0.0
            if prev_mode is not None:
                # increment penalty if switching mode or changing route identity (best-effort)
                if mode != prev_mode:
                    add = transfer_penalty
                else:
                    # if we had prev_route and current possible_rides don't include it, treat as transfer
                    if prev_route and all(pr.get("route_name") != prev_route for pr in possible_rides):
                        add = transfer_penalty
            new_cost = cost + t + add
            new_path = path + [nbr]
            new_legs = legs + [{
                "from": node,
                "to": nbr,
                "travel_time": t,
                "penalty": add,
                "possible_rides": possible_rides
            }]
            heapq.heappush(pq, (new_cost, nbr, mode, possible_rides[0].get("route_name"), new_path, new_legs))
    return None

# ------------------ UI & flow ------------------
st.sidebar.header("Options")
st.sidebar.write("You can provide a Google Places API key if Nominatim is slow (optional).")
google_api_key = st.sidebar.text_input("Google Places API key (optional)", type="password")
radius_km = st.sidebar.number_input("Stops search radius (km) [used for fallback bbox]", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
transfer_penalty = st.sidebar.number_input("Transfer penalty (min)", min_value=0, max_value=10, value=2, step=1)

col1, col2 = st.columns([1, 2])
with col1:
    st.header("Origin & Destination (autocomplete)")

    origin_q = st.text_input("Origin (3+ chars)", key="origin_q")
    dest_q = st.text_input("Destination (3+ chars)", key="dest_q")

    # choose suggestion mechanism
    use_google = bool(google_api_key and google_api_key.strip())
    origin_sugs = []
    dest_sugs = []
    if origin_q and len(origin_q.strip()) >= 3:
        if use_google:
            # Google Places autocomplete + details
            try:
                # Autocomplete
                pa = requests.get("https://maps.googleapis.com/maps/api/place/autocomplete/json",
                                  params={"input": origin_q, "key": google_api_key, "types":"geocode"}, timeout=6).json()
                for pred in pa.get("predictions", [])[:6]:
                    # details
                    pd = requests.get("https://maps.googleapis.com/maps/api/place/details/json",
                                      params={"place_id": pred["place_id"], "key": google_api_key, "fields":"formatted_address,geometry"}, timeout=6).json()
                    res = pd.get("result", {})
                    geom = res.get("geometry", {}).get("location")
                    if geom:
                        origin_sugs.append({"display_name": res.get("formatted_address") or pred.get("description"), "lat": geom["lat"], "lon": geom["lng"]})
            except Exception:
                origin_sugs = []
        else:
            origin_sugs = nominatim_suggest(origin_q, limit=6)

    if dest_q and len(dest_q.strip()) >= 3:
        if use_google:
            try:
                pa = requests.get("https://maps.googleapis.com/maps/api/place/autocomplete/json",
                                  params={"input": dest_q, "key": google_api_key, "types":"geocode"}, timeout=6).json()
                for pred in pa.get("predictions", [])[:6]:
                    pd = requests.get("https://maps.googleapis.com/maps/api/place/details/json",
                                      params={"place_id": pred["place_id"], "key": google_api_key, "fields":"formatted_address,geometry"}, timeout=6).json()
                    res = pd.get("result", {})
                    geom = res.get("geometry", {}).get("location")
                    if geom:
                        dest_sugs.append({"display_name": res.get("formatted_address") or pred.get("description"), "lat": geom["lat"], "lon": geom["lng"]})
            except Exception:
                dest_sugs = []
        else:
            dest_sugs = nominatim_suggest(dest_q, limit=6)

    st.markdown("**Origin suggestions**")
    for i, s in enumerate(origin_sugs):
        label = f"{s['display_name']}"
        if st.button(label, key=f"orig_{i}"):
            st.session_state["origin_choice"] = s
            st.success("Origin selected")
    if st.session_state.get("origin_choice"):
        oc = st.session_state["origin_choice"]
        st.markdown(f"**Chosen origin:** {oc['display_name']}  \n(lat: {oc['lat']:.6f}, lon: {oc['lon']:.6f})")

    st.markdown("---")
    st.markdown("**Destination suggestions**")
    for i, s in enumerate(dest_sugs):
        label = f"{s['display_name']}"
        if st.button(label, key=f"dest_{i}"):
            st.session_state["dest_choice"] = s
            st.success("Destination selected")
    if st.session_state.get("dest_choice"):
        dc = st.session_state["dest_choice"]
        st.markdown(f"**Chosen destination:** {dc['display_name']}  \n(lat: {dc['lat']:.6f}, lon: {dc['lon']:.6f})")

    st.write("---")
    if st.button("Load Manila stops (cached)"):
        with st.spinner("Fetching Manila stops & routes from OSM (cached for 24h)..."):
            G_manila, stops_df, tcount = build_manila_graph()
            if G_manila is None:
                st.error("Failed to fetch Manila OSM data.")
            else:
                st.success(f"Loaded Manila graph: {len(G_manila.nodes)} stops, transit edges ≈ {tcount}")

    plan = st.button("Plan route (city-wide)")

with col2:
    st.header("Map & Recommendation")
    map_box = st.empty()

# If user asked to plan route
if plan:
    if not st.session_state.get("origin_choice") or not st.session_state.get("dest_choice"):
        st.error("Choose both origin and destination from suggestions first.")
    else:
        origin = st.session_state["origin_choice"]
        dest = st.session_state["dest_choice"]
        ox, oy = float(origin["lat"]), float(origin["lon"])
        dx, dy = float(dest["lat"]), float(dest["lon"])

        # Ensure Manila graph ready
        with st.spinner("Building/Loading Manila graph (this is cached — only the first time may take long)..."):
            G_manila, stops_df, tcount = build_manila_graph()
        if G_manila is None:
            st.error("Manila graph not available. Try clicking 'Load Manila stops' first or try again later.")
        else:
            # find nearest stops in the city graph
            # compute distances
            stops_df["dist_o"] = stops_df.apply(lambda r: haversine(ox, oy, float(r["lat"]), float(r["lon"])), axis=1)
            stops_df["dist_d"] = stops_df.apply(lambda r: haversine(dx, dy, float(r["lat"]), float(r["lon"])), axis=1)
            origin_row = stops_df.loc[stops_df["dist_o"].idxmin()]
            dest_row = stops_df.loc[stops_df["dist_d"].idxmin()]
            o_stop = str(origin_row["stop_id"]); d_stop = str(dest_row["stop_id"])
            walk_o = (origin_row["dist_o"] / 5.0) * 60.0
            walk_d = (dest_row["dist_d"] / 5.0) * 60.0

            st.markdown("## Nearest stops & walking to stops")
            st.write(f"Nearest origin stop: **{origin_row['stop_name']}** ({origin_row['dist_o']*1000:.0f} m) — walk ≈ {walk_o:.1f} min")
            st.write(f"Nearest destination stop: **{dest_row['stop_name']}** ({dest_row['dist_d']*1000:.0f} m) — walk ≈ {walk_d:.1f} min")

            # pathfinding
            res = find_fastest_path(G_manila, o_stop, d_stop, transfer_penalty=float(transfer_penalty))
            if not res:
                st.error("No path found between nearest stops in the Manila graph. The network may be sparse in that area.")
            else:
                total_transit = sum([leg["travel_time"] for leg in res["legs"] if any(rn["mode"]!="walk" for rn in leg["possible_rides"])])
                total_walk_internal = sum([leg["travel_time"] for leg in res["legs"] if all(rn["mode"]=="walk" for rn in leg["possible_rides"])])
                total_penalties = sum([leg["penalty"] for leg in res["legs"]])
                estimated_total = walk_o + total_transit + total_walk_internal + walk_d + total_penalties
                eta = datetime.now() + timedelta(minutes=estimated_total)

                st.markdown("## Recommended commute")
                st.write(f"**Estimated total time:** {estimated_total:.1f} min • **ETA:** {eta.strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown("### Steps & possible rides")
                # Step 1: walk to origin stop
                st.write(f"1. Walk: origin → **{origin_row['stop_name']}** — ≈ {walk_o:.1f} min")
                step = 2
                for leg in res["legs"]:
                    from_name = G_manila.nodes[leg["from"]]["name"]
                    to_name = G_manila.nodes[leg["to"]]["name"]
                    # list possible rides
                    rides = leg["possible_rides"]
                    # group unique route_name + mode strings
                    ride_strs = []
                    for r in rides:
                        rn = r.get("route_name") or "(unnamed)"
                        rm = r.get("mode") or "unknown"
                        ride_strs.append(f"{rn} ({rm})")
                    ride_strs = sorted(list(dict.fromkeys(ride_strs)))  # unique preserve order
                    if all(r.get("mode")=="walk" for r in rides):
                        st.write(f"{step}. Walk: {from_name} → {to_name} — {leg['travel_time']:.1f} min")
                    else:
                        st.write(f"{step}. Transit: {from_name} → {to_name} — options: {', '.join(ride_strs)} — {leg['travel_time']:.1f} min" + (f" + {leg['penalty']:.1f} min transfer" if leg['penalty']>0 else ""))
                    step += 1
                st.write(f"{step}. Walk: **{dest_row['stop_name']}** → destination — ≈ {walk_d:.1f} min")

                # Map: show user points, nearest stops, and route polyline
                center_lat = (ox + dx) / 2.0; center_lon = (oy + dy) / 2.0
                m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
                folium.Marker(location=(ox, oy), popup="Origin (selected)", icon=folium.Icon(color="blue")).add_to(m)
                folium.Marker(location=(dx, dy), popup="Destination (selected)", icon=folium.Icon(color="red")).add_to(m)
                # add nearest stops markers
                folium.Marker(location=(float(origin_row["lat"]), float(origin_row["lon"])), popup=f"Origin stop: {origin_row['stop_name']}", icon=folium.Icon(color="green")).add_to(m)
                folium.Marker(location=(float(dest_row["lat"]), float(dest_row["lon"])), popup=f"Dest stop: {dest_row['stop_name']}", icon=folium.Icon(color="green")).add_to(m)
                # small markers for first N stops to avoid clutter
                sample = stops_df.sample(min(500, len(stops_df)), random_state=1) if len(stops_df) > 500 else stops_df
                for _, r in sample.iterrows():
                    folium.CircleMarker(location=(float(r["lat"]), float(r["lon"])), radius=1.5, color="#666", fill=True, fill_opacity=0.4).add_to(m)
                # polyline along path (use nodes in legs)
                coords = []
                for leg in res["legs"]:
                    node = G_manila.nodes[leg["from"]]
                    coords.append((node["lat"], node["lon"]))
                coords.append((G_manila.nodes[res["path"][-1]]["lat"], G_manila.nodes[res["path"][-1]]["lon"]))
                folium.PolyLine(coords, color="green", weight=5, opacity=0.8).add_to(m)
                # walking to/from user points
                folium.PolyLine([(ox, oy), (float(origin_row["lat"]), float(origin_row["lon"]))], color="blue", weight=3, dash_array="5,8").add_to(m)
                folium.PolyLine([(float(dest_row["lat"]), float(dest_row["lon"])), (dx, dy)], color="red", weight=3, dash_array="5,8").add_to(m)

                map_box.write(st_folium(m, width=900, height=600))

st.markdown("---")
st.caption("This app fetches Manila stops & route relations from OpenStreetMap (Overpass) and uses Nominatim (or Google if you supply an API key) for address suggestions. Travel times are estimates from distance + assumed speeds. For production-grade ETA use GTFS + routing engine.")
