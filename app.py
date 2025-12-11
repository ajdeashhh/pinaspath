# app.py
# PinasPath — Autocomplete with robust Nominatim (retries/backoff) + optional Google Places
# Includes dynamic-stop route planning (Overpass) and ETA
# Run: streamlit run app.py
# Requirements: streamlit, pandas, networkx, folium, streamlit-folium, requests

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

st.set_page_config(page_title="PinasPath — Robust Autocomplete", layout="wide")
st.title("PinasPath — Autocomplete + Route Recommendation (robust)")

# ---------------- Utilities ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# ---------- Robust Nominatim (retries + backoff) ----------
@st.cache_data(ttl=300)
def nominatim_suggest_resilient(q, limit=6, max_retries=3, timeout=15):
    if not q or len(q.strip()) < 3:
        return []
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": limit, "addressdetails": 0}
    headers = {"User-Agent": "PinasPathPrototype/1.0 (contact@example.com)"}
    backoff = 0.6
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                # minimal polite wait but not blocking heavy
                time.sleep(0.15)
                out = [{"display_name": it.get("display_name"), "lat": float(it.get("lat")), "lon": float(it.get("lon"))} for it in data]
                return out
            else:
                # Non-200: wait & retry
                time.sleep(backoff)
                backoff *= 2
        except requests.exceptions.RequestException as e:
            # network error or timeout, backoff and retry
            if attempt == max_retries:
                # last attempt failed — return empty and let caller handle message
                return []
            time.sleep(backoff)
            backoff *= 2
    return []

# ---------- Google Places suggestion (optional) ----------
# If user enters an API key, the app will use Google Places for autocomplete suggestions.
# You must enable Places API, and optionally Places Details for geometry.
@st.cache_data(ttl=300)
def google_places_suggest(q, api_key, limit=6):
    if not q or len(q.strip()) < 1 or not api_key:
        return []
    try:
        # Autocomplete endpoint
        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params = {"input": q, "key": api_key, "types": "geocode", "components": ""}  # components can be used for country filter
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        preds = resp.json().get("predictions", [])[:limit]
        results = []
        # For each prediction, request place details to get geometry
        for p in preds:
            pid = p.get("place_id")
            # Place details
            durl = "https://maps.googleapis.com/maps/api/place/details/json"
            dparams = {"place_id": pid, "key": api_key, "fields": "geometry,name,formatted_address"}
            dresp = requests.get(durl, params=dparams, timeout=8)
            dresp.raise_for_status()
            pdict = dresp.json().get("result", {})
            geom = pdict.get("geometry", {}).get("location")
            if geom:
                results.append({
                    "display_name": pdict.get("formatted_address") or pdict.get("name"),
                    "lat": float(geom["lat"]),
                    "lon": float(geom["lng"])
                })
            # be polite to Google too (very small)
            time.sleep(0.05)
        return results
    except Exception:
        return []

# ---------- Overpass fetch (same as before) ----------
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
        st.warning(f"Overpass error: {e}")
    return nodes, relations

# ---------- Graph builders & pathfinder (same) ----------
def build_graph_from_osm(nodes, relations, walking_threshold_km=0.6):
    G = nx.DiGraph()
    for nid, nd in nodes.items():
        G.add_node(str(nid), name=nd["tags"].get("name") or nd["tags"].get("ref") or f"stop_{nid}",
                   lat=float(nd["lat"]), lon=float(nd["lon"]))
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
    items = list(nodes.items())
    for i in range(len(items)):
        id_a, a = items[i]
        for j in range(i+1, len(items)):
            id_b, b = items[j]
            d = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if d <= walking_threshold_km:
                walk_min = (d / 5.0) * 60.0
                G.add_edge(str(id_a), str(id_b), travel_time=walk_min, mode="walk", route_name="walk")
                G.add_edge(str(id_b), str(id_a), travel_time=walk_min, mode="walk", route_name="walk")
    return G, transit_added

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

# ---------------- UI ----------------
col1, col2 = st.columns([1, 2])
with col1:
    st.header("Inputs: autocomplete (3+ chars)")
    st.info("Tip: provide a Google Places API key in the sidebar to use Google's autocomplete (recommended for heavy/reliable use).")

    origin_text = st.text_input("Origin", key="origin_text")
    dest_text = st.text_input("Destination", key="dest_text")

    # Google API key (optional)
    google_api_key = st.text_input("Google Places API key (optional)", type="password")
    use_google = bool(google_api_key and len(google_api_key.strip())>0)

    # suggestions area
    st.markdown("**Origin suggestions:**")
    origin_suggestions = []
    if origin_text and len(origin_text.strip())>=3:
        if use_google:
            origin_suggestions = google_places_suggest(origin_text, google_api_key, limit=6)
        else:
            origin_suggestions = nominatim_suggest_resilient(origin_text, limit=6)
            if origin_suggestions == []:
                st.warning("Nominatim timed out or returned no results. Try again or provide a Google API key in the sidebar.")
    # clickable buttons for origin suggestions
    for i, s in enumerate(origin_suggestions):
        label = f"{s['display_name']}"
        if st.button(label, key=f"orig_btn_{i}"):
            st.session_state["origin_choice"] = s
            st.success("Origin selected from suggestions")
    if st.session_state.get("origin_choice"):
        oc = st.session_state["origin_choice"]
        st.markdown(f"Chosen origin: **{oc['display_name']}** — ({oc['lat']:.6f}, {oc['lon']:.6f})")

    st.markdown("---")
    st.markdown("**Destination suggestions:**")
    dest_suggestions = []
    if dest_text and len(dest_text.strip())>=3:
        if use_google:
            dest_suggestions = google_places_suggest(dest_text, google_api_key, limit=6)
        else:
            dest_suggestions = nominatim_suggest_resilient(dest_text, limit=6)
            if dest_suggestions == []:
                st.warning("Nominatim timed out or returned no results for destination. Try again or provide a Google API key.")
    for i, s in enumerate(dest_suggestions):
        label = f"{s['display_name']}"
        if st.button(label, key=f"dest_btn_{i}"):
            st.session_state["dest_choice"] = s
            st.success("Destination selected from suggestions")
    if st.session_state.get("dest_choice"):
        dc = st.session_state["dest_choice"]
        st.markdown(f"Chosen destination: **{dc['display_name']}** — ({dc['lat']:.6f}, {dc['lon']:.6f})")

    st.write("---")
    radius_km = st.number_input("Stops search radius (km)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    transfer_penalty = st.number_input("Transfer penalty (min)", min_value=0, max_value=10, value=2, step=1)
    plan_btn = st.button("Plan route")

with col2:
    st.header("Map & Result")
    map_placeholder = st.empty()

# Plan button action
if plan_btn:
    if not st.session_state.get("origin_choice") or not st.session_state.get("dest_choice"):
        st.error("Select suggestions for both origin and destination before planning.")
    else:
        origin = st.session_state["origin_choice"]; dest = st.session_state["dest_choice"]
        ox, oy = origin["lat"], origin["lon"]; dx, dy = dest["lat"], dest["lon"]
        center_lat = (ox + dx) / 2.0; center_lon = (oy + dy) / 2.0

        st.info("Fetching nearby stops (Overpass) and building route graph...")
        nodes, relations = overpass_fetch_stops_and_routes(center_lat, center_lon, radius_m=int(radius_km*1000))
        if not nodes:
            st.error("No stops found in area. Increase radius or try a different area.")
        else:
            G, transit_count = build_graph_from_osm(nodes, relations, walking_threshold_km=0.6)
            st.success(f"Graph built — {len(G.nodes)} stops, transit edges ≈ {transit_count}.")

            # nearest stops
            stops_df = pd.DataFrame([{"stop_id": nid, "lat": nd["lat"], "lon": nd["lon"], "name": nd["tags"].get("name") or nd["tags"].get("ref") or f"stop_{nid}"} for nid, nd in nodes.items()])
            stops_df["dist_o"] = stops_df.apply(lambda r: haversine(ox, oy, float(r["lat"]), float(r["lon"])), axis=1)
            stops_df["dist_d"] = stops_df.apply(lambda r: haversine(dx, dy, float(r["lat"]), float(r["lon"])), axis=1)
            origin_row = stops_df.loc[stops_df["dist_o"].idxmin()]
            dest_row = stops_df.loc[stops_df["dist_d"].idxmin()]
            o_stop = str(int(origin_row["stop_id"])); d_stop = str(int(dest_row["stop_id"]))
            walk_o_min = (origin_row["dist_o"] / 5.0) * 60.0; walk_d_min = (dest_row["dist_d"] / 5.0) * 60.0

            st.write(f"Nearest origin stop: **{origin_row['name']}** — {origin_row['dist_o']*1000:.0f} m (walk ≈ {walk_o_min:.1f} min)")
            st.write(f"Nearest dest stop: **{dest_row['name']}** — {dest_row['dist_d']*1000:.0f} m (walk ≈ {walk_d_min:.1f} min)")

            res = find_fastest_route(G, o_stop, d_stop, transfer_penalty=float(transfer_penalty))
            if not res:
                st.error("No transit path found between nearest stops.")
            else:
                total_transit = sum([leg["travel_time"] for leg in res["legs"] if leg["mode"] != "walk"])
                total_walk_internal = sum([leg["travel_time"] for leg in res["legs"] if leg["mode"] == "walk"])
                total_penalties = sum([leg["penalty"] for leg in res["legs"]])
                estimated_total_min = walk_o_min + total_transit + total_walk_internal + walk_d_min + total_penalties

                st.markdown("## Recommended Route")
                st.markdown(f"**Estimated duration:** {estimated_total_min:.1f} minutes")
                eta = datetime.now() + timedelta(minutes=estimated_total_min)
                st.markdown(f"**ETA:** {eta.strftime('%Y-%m-%d %H:%M:%S')}")

                st.markdown("### Legs (walk + transit)")
                st.write(f"1. Walk: origin input → {origin_row['name']} — ≈ {walk_o_min:.1f} min")
                idx = 2
                for leg in res["legs"]:
                    from_name = G.nodes[leg["from"]]["name"]; to_name = G.nodes[leg["to"]]["name"]
                    if leg["mode"] == "walk":
                        st.write(f"{idx}. Walk: {from_name} → {to_name} — {leg['travel_time']:.1f} min")
                    else:
                        st.write(f"{idx}. {leg['mode'].title()}: {from_name} → {to_name} — {leg['route']} — {leg['travel_time']:.1f} min" + (f" + {leg['penalty']:.1f} min transfer" if leg['penalty']>0 else ""))
                    idx += 1
                st.write(f"{idx}. Walk: {dest_row['name']} → destination input — ≈ {walk_d_min:.1f} min")

                # map
                m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                folium.Marker(location=(ox, oy), popup="Origin", icon=folium.Icon(color="blue")).add_to(m)
                folium.Marker(location=(dx, dy), popup="Destination", icon=folium.Icon(color="red")).add_to(m)
                for _, r in stops_df.iterrows():
                    folium.CircleMarker(location=(float(r["lat"]), float(r["lon"])), radius=2, color="#666", fill=True, fill_opacity=0.5).add_to(m)
                folium.Marker(location=(float(origin_row["lat"]), float(origin_row["lon"])), popup=f"Origin stop: {origin_row['name']}", icon=folium.Icon(color="green")).add_to(m)
                folium.Marker(location=(float(dest_row["lat"]), float(dest_row["lon"])), popup=f"Dest stop: {dest_row['name']}", icon=folium.Icon(color="green")).add_to(m)
                coords = []
                for leg in res["legs"]:
                    node = G.nodes[leg["from"]]; coords.append((node["lat"], node["lon"]))
                coords.append((G.nodes[res["path"][-1]]["lat"], G.nodes[res["path"][-1]]["lon"]))
                folium.PolyLine(coords, color="green", weight=5, opacity=0.8).add_to(m)
                folium.PolyLine([(ox, oy), (float(origin_row["lat"]), float(origin_row["lon"]))], color="blue", weight=3, dash_array="5,8").add_to(m)
                folium.PolyLine([(float(dest_row["lat"]), float(dest_row["lon"])), (dx, dy)], color="red", weight=3, dash_array="5,8").add_to(m)

                map_placeholder.write(st_folium(m, width=900, height=600))

st.markdown("---")
st.caption("Autocomplete uses Nominatim (resilient) or Google Places (optional). Overpass fetches stops. For production-grade, use Google Places + routing engine + GTFS.")
