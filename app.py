import streamlit as st
import pandas as pd
import networkx as nx
from streamlit_folium import st_folium
import folium
import requests
import math
import time

st.set_page_config(page_title="PinasPath — Map-based Prototype", layout="wide")
st.title("PinasPath — Map-based Prototype")
st.write("Enter addresses or click the map to set origin and destination. The app snaps to nearest stops and computes the transit + walking route.")

# ---------- Utilities ----------
@st.cache_data
def load_data(stops_path="stops.csv", routes_path="routes.csv"):
    stops = pd.read_csv(stops_path)
    routes = pd.read_csv(routes_path)
    return stops, routes

def haversine(lat1, lon1, lat2, lon2):
    # returns distance in kilometers
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def find_nearest_stop(lat, lon, stops_df):
    dists = stops_df.apply(lambda r: haversine(lat, lon, float(r["lat"]), float(r["lon"])), axis=1)
    idx = dists.idxmin()
    return stops_df.loc[idx, "stop_id"], stops_df.loc[idx, "stop_name"], dists[idx]

def geocode_address_nominatim(address, pause=1.0):
    # Simple Nominatim forward geocoding (no API key). Respect usage policy.
    # Returns (lat, lon) or None
    if not address or str(address).strip()=="":
        return None
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "PinasPathPrototype/1.0 (youremail@example.com)"}
    params = {"q": address, "format": "json", "limit": 1}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                # be polite to the service
                time.sleep(pause)
                return lat, lon
    except Exception as e:
        st.warning(f"Geocoding error: {e}")
    return None

# ---------- Load data and build graph ----------
st.sidebar.header("Data & options")
st.sidebar.write("This demo snaps geocoded / clicked locations to the nearest transit stops in the dataset.")
stops, routes = load_data()
G = nx.DiGraph()
for _, r in stops.iterrows():
    G.add_node(r["stop_id"], name=r["stop_name"], lat=float(r["lat"]), lon=float(r["lon"]))
for _, r in routes.iterrows():
    G.add_edge(r["from_stop"], r["to_stop"], travel_time=float(r["travel_time"]), mode=r["mode"], route_name=r["route_name"])
    G.add_edge(r["to_stop"], r["from_stop"], travel_time=float(r["travel_time"]), mode=r["mode"], route_name=r["route_name"])

transfer_penalty = st.sidebar.number_input("Transfer penalty (min)", min_value=0, max_value=30, value=0, step=1)
walking_speed_kmph = st.sidebar.number_input("Walking speed (km/h)", min_value=2.0, max_value=7.0, value=5.0, step=0.1)

# ---------- UI for origin/destination ----------
st.sidebar.subheader("Set origin & destination")
origin_address = st.sidebar.text_input("Origin address (free text)")
if st.sidebar.button("Geocode origin"):
    geo = geocode_address_nominatim(origin_address)
    if geo:
        st.session_state["origin_geocode"] = {"lat": geo[0], "lon": geo[1], "source": "address"}
    else:
        st.warning("Origin geocoding failed.")

destination_address = st.sidebar.text_input("Destination address (free text)")
if st.sidebar.button("Geocode destination"):
    geo = geocode_address_nominatim(destination_address)
    if geo:
        st.session_state["destination_geocode"] = {"lat": geo[0], "lon": geo[1], "source": "address"}
    else:
        st.warning("Destination geocoding failed.")

# Map click mode
click_mode = st.sidebar.radio("Map click sets:", ("None", "Origin", "Destination"))
st.sidebar.markdown("Click the map to place the selected point. Use the geocode buttons or map clicks — whichever you prefer.")

# Initialize session state for geocodes if missing
if "origin_geocode" not in st.session_state:
    st.session_state["origin_geocode"] = None
if "destination_geocode" not in st.session_state:
    st.session_state["destination_geocode"] = None

# ---------- Map display and click capture ----------
# Determine initial map center
mean_lat = stops["lat"].astype(float).mean()
mean_lon = stops["lon"].astype(float).mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

# Add all stops as small markers (optional)
for _, r in stops.iterrows():
    folium.CircleMarker(location=(float(r["lat"]), float(r["lon"])),
                        radius=3,
                        color="#444",
                        fill=True,
                        fill_opacity=0.6,
                        popup=f'{r["stop_name"]} ({r["stop_id"]})').add_to(m)

# If there are geocoded points in session, show them
def add_user_marker(m, lat, lon, label, color="blue"):
    folium.Marker(location=(lat, lon), popup=label, tooltip=label,
                  icon=folium.Icon(color=color, icon="map-marker")).add_to(m)

# Pre-add markers if geocoded
if st.session_state.get("origin_geocode"):
    og = st.session_state["origin_geocode"]
    add_user_marker(m, og["lat"], og["lon"], "Origin (chosen)", color="green")

if st.session_state.get("destination_geocode"):
    dg = st.session_state["destination_geocode"]
    add_user_marker(m, dg["lat"], dg["lon"], "Destination (chosen)", color="red")

# Show interactive map and capture clicks
st.subheader("Map (click to set origin/destination)")
map_result = st_folium(m, width=900, height=600)

# If user clicked and click_mode set, record the clicked point
if map_result and map_result.get("last_clicked") and click_mode != "None":
    clicked = map_result["last_clicked"]
    lat_clicked = clicked["lat"]
    lon_clicked = clicked["lng"]
    if click_mode == "Origin":
        st.session_state["origin_geocode"] = {"lat": lat_clicked, "lon": lon_clicked, "source": "click"}
        st.success(f"Origin set by click: {lat_clicked:.6f}, {lon_clicked:.6f}")
    elif click_mode == "Destination":
        st.session_state["destination_geocode"] = {"lat": lat_clicked, "lon": lon_clicked, "source": "click"}
        st.success(f"Destination set by click: {lat_clicked:.6f}, {lon_clicked:.6f}")

# ---------- Snap to nearest stops and compute route ----------
def build_path_summary(origin_point, destination_point):
    # origin_point and destination_point are dicts {"lat":..., "lon":..., "source":...}
    if not origin_point or not destination_point:
        return {"error": "Origin or destination not set."}

    # Find nearest stops
    o_stop_id, o_stop_name, o_stop_dist = find_nearest_stop(origin_point["lat"], origin_point["lon"], stops)
    d_stop_id, d_stop_name, d_stop_dist = find_nearest_stop(destination_point["lat"], destination_point["lon"], stops)

    # Walking times (minutes)
    walk_time_origin = (o_stop_dist / walking_speed_kmph) * 60.0
    walk_time_dest = (d_stop_dist / walking_speed_kmph) * 60.0

    # Transit path between stops using Dijkstra on travel_time
    try:
        path_nodes = nx.shortest_path(G, o_stop_id, d_stop_id, weight="travel_time")
    except nx.NetworkXNoPath:
        return {"error": "No transit path found between nearest stops."}

    # Build legs for transit
    transit_legs = []
    total_transit = 0.0
    for i in range(len(path_nodes)-1):
        a = path_nodes[i]
        b = path_nodes[i+1]
        e = G[a][b]
        transit_legs.append({
            "from_id": a,
            "to_id": b,
            "from_name": G.nodes[a]["name"],
            "to_name": G.nodes[b]["name"],
            "mode": e["mode"],
            "route_name": e["route_name"],
            "travel_time": e["travel_time"]
        })
        total_transit += e["travel_time"]

    # Total estimate
    total_est = walk_time_origin + total_transit + walk_time_dest

    return {
        "origin_point": origin_point,
        "destination_point": destination_point,
        "origin_stop": {"id": o_stop_id, "name": o_stop_name, "dist_km": o_stop_dist, "walk_min": walk_time_origin},
        "destination_stop": {"id": d_stop_id, "name": d_stop_name, "dist_km": d_stop_dist, "walk_min": walk_time_dest},
        "transit": transit_legs,
        "total_transit_min": total_transit,
        "estimated_total_min": total_est
    }

# If both points set, provide a "Plan trip" button
st.sidebar.write("---")
if st.sidebar.button("Plan trip"):
    if not st.session_state.get("origin_geocode") or not st.session_state.get("destination_geocode"):
        st.sidebar.error("Set both origin and destination (by geocode or map click).")
    else:
        summary = build_path_summary(st.session_state["origin_geocode"], st.session_state["destination_geocode"])
        st.session_state["last_summary"] = summary

# Show results if available
if st.session_state.get("last_summary"):
    s = st.session_state["last_summary"]
    if "error" in s:
        st.error(s["error"])
    else:
        st.markdown("## Trip summary")
        st.markdown(f"**Origin (input) :** {s['origin_point']['lat']:.6f}, {s['origin_point']['lon']:.6f}  ")
        st.markdown(f"**Nearest stop (origin) :** {s['origin_stop']['name']} ({s['origin_stop']['id']}) — {s['origin_stop']['dist_km']*1000:.0f} m — walk ≈ {s['origin_stop']['walk_min']:.1f} min")
        st.markdown(f"**Destination (input) :** {s['destination_point']['lat']:.6f}, {s['destination_point']['lon']:.6f}  ")
        st.markdown(f"**Nearest stop (destination) :** {s['destination_stop']['name']} ({s['destination_stop']['id']}) — {s['destination_stop']['dist_km']*1000:.0f} m — walk ≈ {s['destination_stop']['walk_min']:.1f} min")
        st.markdown("### Transit legs")
        for i, leg in enumerate(s["transit"], 1):
            st.write(f"{i}. **{leg['from_name']} → {leg['to_name']}** — {leg['mode']} ({leg['route_name']}) | {leg['travel_time']} min")
        st.markdown(f"**Transit total:** {s['total_transit_min']:.1f} min")
        st.markdown(f"**Estimated total (walk + transit):** {s['estimated_total_min']:.1f} min")

        # Draw a new map with route: origin (blue), origin-stop (green), stops polyline, dest-stop (green), destination (red)
        m2 = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)
        # all stops (light)
        for _, r in stops.iterrows():
            folium.CircleMarker(location=(float(r["lat"]), float(r["lon"])), radius=2, color="#888", fill=True, fill_opacity=0.4).add_to(m2)

        # origin point marker
        op = s["origin_point"]
        add_user_marker(m2, op["lat"], op["lon"], "Origin (you)", color="blue")
        # origin nearest stop marker
        osid = s["origin_stop"]["id"]
        osn = G.nodes[osid]
        add_user_marker(m2, osn["lat"], osn["lon"], f"Origin stop: {osn['name']}", color="green")
        # destination point marker
        dp = s["destination_point"]
        add_user_marker(m2, dp["lat"], dp["lon"], "Destination (you)", color="darkred")
        # dest nearest stop
        dsid = s["destination_stop"]["id"]
        dsn = G.nodes[dsid]
        add_user_marker(m2, dsn["lat"], dsn["lon"], f"Dest stop: {dsn['name']}", color="green")

        # draw walking lines: origin->origin_stop and dest_stop->destination
        folium.PolyLine([(op["lat"], op["lon"]), (osn["lat"], osn["lon"])], color="blue", weight=3, dash_array="5,8").add_to(m2)
        folium.PolyLine([(dp["lat"], dp["lon"]), (dsn["lat"], dsn["lon"])], color="red", weight=3, dash_array="5,8").add_to(m2)

        # draw transit polyline through stops
        coords = []
        for leg in s["transit"]:
            a = G.nodes[leg["from_id"]]
            coords.append((a["lat"], a["lon"]))
        # add last stop
        coords.append((G.nodes[s["transit"][-1]["to_id"]]["lat"], G.nodes[s["transit"][-1]["to_id"]]["lon"]))
        folium.PolyLine(coords, color="green", weight=5, opacity=0.8).add_to(m2)

        st.subheader("Map view of planned trip")
        st_folium(m2, width=900, height=600)

# Footer note
st.markdown("---")
st.caption("Geocoding via Nominatim (OpenStreetMap). Please avoid heavy automated use; this demo includes a short polite pause between requests. For production use obtain proper geocoding credentials / service.")
