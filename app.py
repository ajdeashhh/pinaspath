import streamlit as st
import pandas as pd
import networkx as nx
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="PinasPath — Streamlit Prototype", layout="centered")
st.title("PinasPath — Streamlit Prototype")
st.write("Single-file prototype: shortest travel-time route on local CSV data, visualized on a map.")

@st.cache_data
def load_data(stops_path="stops.csv", routes_path="routes.csv"):
    stops = pd.read_csv(stops_path)
    routes = pd.read_csv(routes_path)
    return stops, routes

stops, routes = load_data()

st.sidebar.header("Trip input")
stop_names = stops["stop_name"].tolist()
origin_name = st.sidebar.selectbox("Origin", stop_names, index=0)
destination_name = st.sidebar.selectbox("Destination", stop_names, index=1)
show_map = st.sidebar.checkbox("Show map", value=True)
transfer_penalty = st.sidebar.number_input("Transfer penalty (minutes)", min_value=0, max_value=30, value=0, step=1,
                                           help="Optional penalty added when the mode or route changes between legs.")

# helper: get ID by name
def name_to_id(name):
    row = stops[stops["stop_name"] == name]
    if row.empty:
        return None
    return row["stop_id"].values[0]

origin_id = name_to_id(origin_name)
destination_id = name_to_id(destination_name)

# Build directed weighted graph
def build_graph(stops_df, routes_df, transfer_penalty=0):
    G = nx.DiGraph()
    # add nodes
    for _, r in stops_df.iterrows():
        G.add_node(r["stop_id"], name=r["stop_name"], lat=float(r["lat"]), lon=float(r["lon"]))
    # add edges (bidirectional)
    for _, r in routes_df.iterrows():
        from_s = r["from_stop"]
        to_s = r["to_stop"]
        weight = float(r["travel_time"])
        G.add_edge(from_s, to_s, travel_time=weight, mode=r["mode"], route_name=r["route_name"])
        G.add_edge(to_s, from_s, travel_time=weight, mode=r["mode"], route_name=r["route_name"])
    # Optionally encode transfer penalty at runtime when evaluating paths (we'll handle it in path cost)
    return G

G = build_graph(stops, routes, transfer_penalty)

# Custom shortest-path that includes transfer penalty (mode/route change)
def shortest_path_with_transfer_penalty(G, origin, destination, transfer_penalty=0):
    import heapq
    # state: (cost, node, prev_mode, prev_route, path_list, legs_list)
    pq = []
    heapq.heappush(pq, (0, origin, None, None, [origin], []))
    visited = dict()  # visited[(node, prev_mode, prev_route)] = best_cost

    while pq:
        cost, node, prev_mode, prev_route, path, legs = heapq.heappop(pq)
        if node == destination:
            return {"total_cost": cost, "path": path, "legs": legs}
        state_key = (node, prev_mode, prev_route)
        if state_key in visited and visited[state_key] <= cost:
            continue
        visited[state_key] = cost

        for nbr in G.neighbors(node):
            e = G[node][nbr]
            leg_time = float(e.get("travel_time", 0))
            mode = e.get("mode")
            route_name = e.get("route_name")
            add = 0
            # if changing mode or route compared to previous leg, add transfer penalty
            if prev_mode is not None and (mode != prev_mode or route_name != prev_route):
                add += transfer_penalty
            new_cost = cost + leg_time + add
            new_path = path + [nbr]
            new_legs = legs + [{
                "from_id": node,
                "to_id": nbr,
                "from_name": G.nodes[node]["name"],
                "to_name": G.nodes[nbr]["name"],
                "mode": mode,
                "route_name": route_name,
                "travel_time": leg_time,
                "penalty": add
            }]
            heapq.heappush(pq, (new_cost, nbr, mode, route_name, new_path, new_legs))
    return None

if st.button("Find Route"):
    if origin_id == destination_id:
        st.warning("Origin and destination are the same.")
    else:
        result = shortest_path_with_transfer_penalty(G, origin_id, destination_id, transfer_penalty=transfer_penalty)
        if not result:
            st.error("No path found between selected stops.")
        else:
            st.success(f"Route found — estimated total (including penalties): {result['total_cost']} minutes")
            # Show textual legs and totals
            total_travel = sum(leg["travel_time"] for leg in result["legs"])
            total_penalty = sum(leg["penalty"] for leg in result["legs"])
            st.subheader("Route legs")
            for i, leg in enumerate(result["legs"], 1):
                st.write(f"{i}. **{leg['from_name']} → {leg['to_name']}** — {leg['mode']} ({leg['route_name']}) | "
                         f"{leg['travel_time']} min" + (f" + {leg['penalty']} transfer penalty" if leg['penalty']>0 else ""))
            st.write(f"**Total travel time (legs only):** {total_travel} minutes")
            st.write(f"**Total transfer penalty:** {total_penalty} minutes")
            st.write(f"**Estimated total time:** {result['total_cost']} minutes")

            # Map visualization
            if show_map:
                # center map between origin and destination
                orig_node = result["path"][0]
                dest_node = result["path"][-1]
                lat0 = G.nodes[orig_node]["lat"]
                lon0 = G.nodes[orig_node]["lon"]
                m = folium.Map(location=[lat0, lon0], zoom_start=12)
                # add all stops in path as markers and connect with polyline
                coords = []
                for node in result["path"]:
                    node_data = G.nodes[node]
                    coords.append((node_data["lat"], node_data["lon"]))
                    folium.CircleMarker(
                        location=(node_data["lat"], node_data["lon"]),
                        radius=6,
                        popup=node_data["name"],
                        tooltip=node_data["name"]
                    ).add_to(m)
                folium.PolyLine(coords, weight=5, opacity=0.8).add_to(m)
                # show map
                st_folium(m, width=700, height=500)
