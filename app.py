# app.py
# PinasPath — Streamlit prototype with address suggestions (Nominatim search)
# Requirements: streamlit, pandas, networkx, folium, streamlit-folium, requests
# Run: streamlit run app.py

import streamlit as st
import requests
import pandas as pd
import networkx as nx
import math
import time
from streamlit_folium import st_folium
import folium
import heapq

st.set_page_config(page_title="PinasPath (autocomplete addresses)", layout="wide")
st.title("PinasPath — Address Suggestions Prototype")
st.write(
    "Type addresses and click **Search** to get address suggestions. Pick the correct match, then Plan trip."
)

# ---------- Utilities ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def nominatim_search(q, limit=6, pause=1.0):
    """Return list of results: dicts with display_name, lat, lon"""
    if not q or str(q).strip()=="":
        return []
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": limit, "addressdetails": 0}
    headers = {"User-Agent": "PinasPathPrototype/1.0 (youremail@example.com)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            time.sleep(pause)  # be polite to free service
            results = []
            for it in data:
                results.append({
                    "display_name": it.get("display_name"),
                    "lat": float(it.get("lat")),
                    "lon": float(it.get("lon"))
                })
            return results
    except Exception as e:
        st.warning(f"Nominatim search failed: {e}")
    return []

def overpass_fetch_stops_and_routes(lat, lon, radius_m=2000):
    """
    Fetch nearby stop nodes and route relations using Overpass API.
    Returns dict: nodes (id->attrs) and relations (list)
    """
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
                nodes[nid] = {
                    "id": nid,
                    "lat": el.get("lat"),
                    "lon": el.get("lon"),
                    "tags": el.get("tags", {})
                }
        rr = requests.post(overpass_url, data=rel_q, timeout=60)
        rr.raise_for_status()
        rd = rr.json()
        for el in rd.get("elements", []):
            if el["type"] == "relation":
                rel = {
                    "id": el["id"],
                    "tags": el.get("tags", {}),
                    "members": []
                }
                for m in el.get("members", []):
                    if m.get("type") == "node":
                        rel["members"].append({"type": "node", "ref": m.get("ref"), "role": m.get("role")})
                relations.append(rel)
        for el in rd.get("elements", []):
            if el["type"] == "node":
                nid = el["id"]
                if nid not in nodes:
                    nodes[nid] = {
                        "id": nid,
                        "lat": el.get("lat"),
                        "lon": el.get("lon"),
                        "tags": el.get("tags", {})
                    }
    except Exception as e:
        st.warning(f"Overpass query failed or timed out: {e}")
    return nodes, relations

# ---------- UI controls ----------
st.sidebar.header("Trip inputs")
origin_text = st.sidebar.text_input("Origin address")
if st.sidebar.button("Search origin"):
    origin_results = nominatim_search(origin_text, limit=6)
    st.session_state["origin_candidates"] = origin_results
    if origin_results:
        st.success(f"Found {len(origin_results)} origin matches")
    else:
        st.warning("No origin matches found")

destination_text = st.sidebar.text_input("Destination address")
if st.sidebar.button("Search destination"):
    dest_results = nominatim_search(destination_text, limit=6)
    st.session_state["destination_candidates"] = dest_results
    if dest_results:
        st.success(f"Found {len(dest_results)} destination matches")
    else:
        st.warning("No destination matches found")

# allow map clicks as alternative
click_mode = st.sidebar.radio("Map click sets:", ("None", "Origin", "Destination"))

# search radius for Overpass
radius_km = st.sidebar.slider("Stops search radius (km)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
radius_m = int(radius_km * 1000)

# transfer penalty (keepable)
transfer_penalty = st.sidebar.number_input("Transfer penalty (min)", min_value=0, max_value=30, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Tips: type a street address, landmark, or place name and click Search. Pick the best match from the list that appears.")

# ---------- session init ----------
if "origin_candidates" not in st.session_state:
    st.session_state["origin_candidates"] = []
if "destination_candidates" not in st.session_state:
    st.session_state["destination_candidates"] = []
if "origin_choice" not in st.session_state:
    st.session_state["origin_choice"] = None
if "destination_choice" not in st.session_state:
    st.session_state["destination_choice"] = None
if "origin" not in st.session_state:
    st.session_state["origin"] = None
if "destination" not in st.session_state:
    st.session_state["destination"] = None
if "fetched_nodes" not in st.session_state:
    st.session_state["fetched_nodes"] = None
if "fetched_relations" not in st.session_state:
    st.session_state["fetched_relations"] = None

# show candidate pickers (if results exist)
if st.session_state["origin_candidates"]:
    options = [f"{i+1}. {r['display_name']}" for i, r in enumerate(st.session_state["origin_candidates"])]
    sel = st.sidebar.selectbox("Choose origin match", options)
    idx = options.index(sel)
    chosen = st.session_state["origin_candidates"][idx]
    st.session_state["origin_choice"] = chosen
    st.sidebar.markdown(f"Selected origin: **{chosen['display_name']}**")

if st.session_state["destination_candidates"]:
    options = [f"{i+1}. {r['display_name']}" for i, r in enumerate(st.session_state["destination_candidates"])]
    sel = st.sidebar.selectbox("Choose destination match", options)
    idx = options.index(sel)
    chosen = st.session_state["destination_candidates"][idx]
    st.session_state["destination_choice"] = chosen
    st.sidebar.markdown(f"Selected destination: **{chosen['display_name']}**")

# if user confirms chosen candidate, copy to origin/destination point
if st.sidebar.button("Set chosen origin"):
    if st.session_state["origin_choice"]:
        st.session_state["origin"] = {"lat": st.session_state["origin_choice"]["lat"],
                                      "lon": st.session_state["origin_choice"]["lon"],
                                      "display_name": st.session_state["origin_choice"]["display_name"],
                                      "source": "search"}
        st.success("Origin set from search result")
    else:
        st.warning("No origin candidate selected")

if st.sidebar.button("Set chosen destination"):
    if st.session_state["destination_choice"]:
        st.session_state["destination"] = {"lat": st.session_state["destination_choice"]["lat"],
                                           "lon": st.session_state["destination_choice"]["lon"],
                                           "display_name": st.session_state["destination_choice"]["display_name"],
                                           "source": "search"}
        st.success("Destination set from search result")
    else:
        st.warning("No destination candidate selected")

# ---------- Map display & click handling ----------
# center fallback
center_lat, center_lon = 14.5995, 120.9842
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# add markers if chosen
if st.session_state.get("origin"):
    folium.Marker(location=(st.session_state["origin"]["lat"], st.session_state["origin"]["lon"]),
                  popup=f"Origin: {st.session_state['origin'].get('display_name','(manual)')}",
                  icon=folium.Icon(color="green")).add_to(m)
if st.session_state.get("destination"):
    folium.Marker(location=(st.session_state["destination"]["lat"], st.session_state["destination"]["lon"]),
                  popup=f"Destination: {st.session_state['destination'].get('display_name','(manual)')}",
                  icon=folium.Icon(color="red")).add_to(m)

st.subheader("Map (click to set origin/destination)")
map_result = st_folium(m, width=900, height=600)

if map_result and map_result.get("last_clicked") and click_mode != "None":
    c = map_result["last_clicked"]
    if click_mode == "Origin":
        st.session_state["origin"] = {"lat": c["lat"], "lon": c["lng"], "display_name": "map click", "source": "click"}
        st.success("Origin set by map click")
    elif click_mode == "Destination":
        st.session_state["destination"] = {"lat": c["lat"], "lon": c["lng"], "display_name": "map click", "source": "click"}
        st.success("Destination set by map click")

# ---------- dynamic stops & graph builder (same logic, fixed walking speed) ----------
def build_dynamic_graph(origin, destination, radius_m=2000, walking_speed_kmph=5.0, transfer_penalty=2):
    if not origin or not destination:
        return None
    center_lat = (origin["lat"] + destination["lat"]) / 2.0
    center_lon = (origin["lon"] + destination["lon"]) / 2.0
    with st.spinner("Fetching nearby stops & routes from OSM (Overpass)..."):
        nodes, relations = overpass_fetch_stops_and_routes(center_lat, center_lon, radius_m=radius_m)
    if not nodes:
        st.warning("No stops found in the area. Try increasing search radius.")
        return None
    stops_list = []
    for nid, nd in nodes.items():
        name = nd["tags"].get("name") or nd["tags"].get("ref") or f"stop_{nid}"
        stops_list.append({"stop_id": str(nid), "stop_name": name, "lat": nd["lat"], "lon": nd["lon"], "tags": nd["tags"]})
    stops_df = pd.DataFrame(stops_list)
    G = nx.DiGraph()
    for _, row in stops_df.iterrows():
        G.add_node(row["stop_id"], name=row["stop_name"], lat=float(row["lat"]), lon=float(row["lon"]))
    transit_added = 0
    for rel in relations:
        rtags = rel.get("tags", {})
        route_type = rtags.get("route", "bus")
        route_name = rtags.get("ref") or rtags.get("name") or f"route_{rel['id']}"
        seq = [str(m["ref"]) for m in rel.get("members", []) if str(m.get("ref")) in nodes]
        speed_kmph = 25.0
        if route_type in ("train", "subway", "light_rail"):
            speed_kmph = 50.0
        elif route_type == "tram":
            speed_kmph = 30.0
        elif route_type == "ferry":
            speed_kmph = 20.0
        for i in range(len(seq)-1):
            a = seq[i]
            b = seq[i+1]
            la, lo = nodes[int(a)]["lat"], nodes[int(a)]["lon"]
            lb, lb2 = nodes[int(b)]["lat"], nodes[int(b)]["lon"]
            dist_km = haversine(la, lo, lb, lb2)
            travel_min = (dist_km / speed_kmph) * 60.0
            G.add_edge(a, b, travel_time=travel_min, mode=route_type, route_name=route_name)
            G.add_edge(b, a, travel_time=travel_min, mode=route_type, route_name=route_name)
            transit_added += 1
    walking_threshold_km = 0.6
    nodes_items = list(nodes.items())
    for i in range(len(nodes_items)):
        id_a, a = nodes_items[i]
        for j in range(i+1, len(nodes_items)):
            id_b, b = nodes_items[j]
            d = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if d <= walking_threshold_km:
                walk_min = (d / walking_speed_kmph) * 60.0
                G.add_edge(str(id_a), str(id_b), travel_time=walk_min, mode="walk", route_name="walk")
                G.add_edge(str(id_b), str(id_a), travel_time=walk_min, mode="walk", route_name="walk")
    meta = {"stops_df": stops_df, "nodes_raw": nodes, "relations_raw": relations, "transit_edges": transit_added}
    return G, meta

def dijkstra_with_transfers(G, origin_stop_id, dest_stop_id, transfer_penalty=2):
    pq = []
    heapq.heappush(pq, (0.0, origin_stop_id, None, None, [origin_stop_id], []))
    visited = dict()
    while pq:
        cost, node, prev_mode, prev_route, path, legs = heapq.heappop(pq)
        key = (node, prev_mode, prev_route)
        if key in visited and visited[key] <= cost:
            continue
        visited[key] = cost
        if node == dest_stop_id:
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

# ---------- Plan trip ----------
if st.button("Plan trip with chosen locations"):
    if not st.session_state.get("origin") or not st.session_state.get("destination"):
        st.error("Set both origin and destination (via search & set or map click).")
    else:
        Gmeta = build_dynamic_graph(st.session_state["origin"], st.session_state["destination"],
                                    radius_m=radius_m, walking_speed_kmph=5.0, transfer_penalty=transfer_penalty)
        if not Gmeta:
            st.stop()
        G, meta = Gmeta
        st.session_state["fetched_nodes"] = meta["nodes_raw"]
        st.session_state["fetched_relations"] = meta["relations_raw"]
        st.success(f"Fetched {len(meta['stops_df'])} stops, created ~{meta['transit_edges']} transit edges.")
        origin = st.session_state["origin"]
        dest = st.session_state["destination"]
        sd = meta["stops_df"]
        sd["dist_to_origin"] = sd.apply(lambda r: haversine(origin["lat"], origin["lon"], float(r["lat"]), float(r["lon"])), axis=1)
        sd["dist_to_dest"] = sd.apply(lambda r: haversine(dest["lat"], dest["lon"], float(r["lat"]), float(r["lon"])), axis=1)
        o_row = sd.loc[sd["dist_to_origin"].idxmin()]
        d_row = sd.loc[sd["dist_to_dest"].idxmin()]
        origin_stop = str(o_row["stop_id"])
        dest_stop = str(d_row["stop_id"])
        st.session_state["origin_stop"] = {"id": origin_stop, "name": o_row["stop_name"], "dist_km": o_row["dist_to_origin"]}
        st.session_state["dest_stop"] = {"id": dest_stop, "name": d_row["stop_name"], "dist_km": d_row["dist_to_dest"]}
        res = dijkstra_with_transfers(G, origin_stop, dest_stop, transfer_penalty=transfer_penalty)
        if not res:
            st.error("No route found between nearest stops.")
        else:
            st.session_state["plan"] = {"graph": G, "meta": meta, "path_result": res,
                                       "origin": origin, "destination": dest}
            st.success("Planned route successfully.")

# ---------- Show results ----------
if st.session_state.get("plan"):
    plan = st.session_state["plan"]
    G = plan["graph"]
    meta = plan["meta"]
    result = plan["path_result"]
    st.markdown("## Route summary")
    st.write(f"Origin: {plan['origin'].get('display_name','(point)')} ({plan['origin']['lat']:.6f}, {plan['origin']['lon']:.6f})")
    st.write(f"Destination: {plan['destination'].get('display_name','(point)')} ({plan['destination']['lat']:.6f}, {plan['destination']['lon']:.6f})")
    st.write(f"Nearest origin stop: {st.session_state['origin_stop']['name']} — {st.session_state['origin_stop']['dist_km']*1000:.0f} m")
    st.write(f"Nearest dest stop: {st.session_state['dest_stop']['name']} — {st.session_state['dest_stop']['dist_km']*1000:.0f} m")
    total_transit = sum([leg["travel_time"] for leg in result["legs"] if leg["mode"] != "walk"])
    total_walk = sum([leg["travel_time"] for leg in result["legs"] if leg["mode"] == "walk"])
    total_penalty = sum([leg["penalty"] for leg in result["legs"]])
    st.write(f"Transit: {total_transit:.1f} min • Walk between stops: {total_walk:.1f} min • Transfers: {total_penalty:.1f} min")
    st.write(f"Estimated total: {result['total_cost']:.1f} min")

    st.markdown("### Legs")
    for i, leg in enumerate(result["legs"], 1):
        from_name = G.nodes[leg["from"]]["name"]
        to_name = G.nodes[leg["to"]]["name"]
        st.write(f"{i}. {from_name} → {to_name} — {leg['mode']} ({leg['route']}) | {leg['travel_time']:.1f} min" +
                 (f" + {leg['penalty']:.1f} min transfer" if leg['penalty']>0 else ""))

    # map
    m2 = folium.Map(location=[(plan["origin"]["lat"] + plan["destination"]["lat"]) / 2,
                              (plan["origin"]["lon"] + plan["destination"]["lon"]) / 2], zoom_start=13)
    for _, r in meta["stops_df"].iterrows():
        folium.CircleMarker(location=(float(r["lat"]), float(r["lon"])), radius=2, color="#666", fill=True, fill_opacity=0.5).add_to(m2)
    folium.Marker(location=(plan["origin"]["lat"], plan["origin"]["lon"]), popup="Origin", icon=folium.Icon(color="blue")).add_to(m2)
    folium.Marker(location=(plan["destination"]["lat"], plan["destination"]["lon"]), popup="Destination", icon=folium.Icon(color="darkred")).add_to(m2)
    o_s = meta["stops_df"].loc[meta["stops_df"]["stop_id"] == int(st.session_state["origin_stop"]["id"])]
    d_s = meta["stops_df"].loc[meta["stops_df"]["stop_id"] == int(st.session_state["dest_stop"]["id"])]
    if not o_s.empty:
        orow = o_s.iloc[0]
        folium.Marker(location=(float(orow["lat"]), float(orow["lon"])), popup=f"Origin stop: {orow['stop_name']}", icon=folium.Icon(color="green")).add_to(m2)
    if not d_s.empty:
        drow = d_s.iloc[0]
        folium.Marker(location=(float(drow["lat"]), float(drow["lon"])), popup=f"Dest stop: {drow['stop_name']}", icon=folium.Icon(color="green")).add_to(m2)
    coords = []
    for leg in result["legs"]:
        from_node = G.nodes[leg["from"]]
        coords.append((from_node["lat"], from_node["lon"]))
    coords.append((G.nodes[result["path"][-1]]["lat"], G.nodes[result["path"][-1]]["lon"]))
    folium.PolyLine(coords, color="green", weight=5, opacity=0.8).add_to(m2)
    # walking lines
    folium.PolyLine([(plan["origin"]["lat"], plan["origin"]["lon"]), (G.nodes[result["path"][0]]["lat"], G.nodes[result["path"][0]]["lon"])],
                    color="blue", weight=3, dash_array="5, 8").add_to(m2)
    folium.PolyLine([(G.nodes[result["path"][-1]]["lat"], G.nodes[result["path"][-1]]["lon"]), (plan["destination"]["lat"], plan["destination"]["lon"])],
                    color="red", weight=3, dash_array="5, 8").add_to(m2)
    st.subheader("Map")
    st_folium(m2, width=900, height=600)

st.markdown("---")
st.caption("Address suggestions use Nominatim (OpenStreetMap). Overpass fetches nearby stops. Limits and coverage vary by area.")
