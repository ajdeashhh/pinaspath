# app.py — Full PinasPath app (accurate map where possible)

import streamlit as st
import pandas as pd
import networkx as nx
import heapq
import math
from datetime import datetime, timedelta
from streamlit_folium import st_folium
import folium
import os
import requests

st.set_page_config(page_title="PinasPath — Streamlit Prototype", layout="wide")
st.markdown("<h1>PinasPath</h1>", unsafe_allow_html=True)
st.caption(
    "🗺️ Map accuracy: walking & jeepney paths follow real roads; train lines are schematic (station-to-station)."
)

# ----------------- CSS -----------------
st.markdown("""
<style>
.mode-badge {display:inline-block;padding:4px 8px;border-radius:6px;color:white;font-weight:600;margin-right:6px;}
.mode-bus {background:#1f77b4;}
.mode-train {background:#2ca02c;}
.mode-jeepney {background:#ff7f0e;}
.mode-walk {background:#7f7f7f;}
.panel {background:#ffffff;border-radius:8px;padding:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}
</style>
""", unsafe_allow_html=True)

# ----------------- Data loader -----------------
@st.cache_data
def load_data(stops_path="stops.csv", routes_path="routes.csv"):
    stops = pd.read_csv(stops_path, dtype=str, comment="#")
    routes = pd.read_csv(routes_path, dtype=str, comment="#")

    stops["lat"] = pd.to_numeric(stops.get("lat"), errors="coerce")
    stops["lon"] = pd.to_numeric(stops.get("lon"), errors="coerce")
    routes["travel_time"] = pd.to_numeric(routes["travel_time"], errors="coerce").fillna(1.0)

    stops["stop_id"] = stops["stop_id"].astype(str)
    routes["from_stop"] = routes["from_stop"].astype(str)
    routes["to_stop"] = routes["to_stop"].astype(str)
    return stops, routes

stops, routes = load_data()

# ----------------- Sidebar -----------------
st.sidebar.header("Plan a trip")
stop_names = stops["stop_name"].tolist()
origin_name = st.sidebar.selectbox("Origin", stop_names, 0)
destination_name = st.sidebar.selectbox("Destination", stop_names, 1)
transfer_penalty = st.sidebar.number_input("Transfer penalty (min)", 0, 30, 2)
show_map = st.sidebar.checkbox("Show map", True)

def name_to_id(name):
    return stops.loc[stops["stop_name"] == name, "stop_id"].values[0]

origin_id = name_to_id(origin_name)
destination_id = name_to_id(destination_name)

# ----------------- Helpers -----------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

@st.cache_data(show_spinner=False)
def osrm_geometry(lat1, lon1, lat2, lon2, profile):
    try:
        url = (
            f"https://router.project-osrm.org/route/v1/"
            f"{profile}/{lon1},{lat1};{lon2},{lat2}"
            "?overview=full&geometries=geojson"
        )
        r = requests.get(url, timeout=10).json()
        coords = r["routes"][0]["geometry"]["coordinates"]
        return [(lat, lon) for lon, lat in coords]
    except Exception:
        return None

# ----------------- Graph builder -----------------
def build_graph(stops_df, routes_df):
    G = nx.DiGraph()
    for _, r in stops_
