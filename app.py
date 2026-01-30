# app.py — PinasPath (map-accurate where data allows)

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

# ----------------- Page setup -----------------
st.set_page_config(page_title="PinasPath", layout="wide")
st.title("PinasPath")
st.caption(
    "🗺️ Walking & jeepney paths follow real roads. Train lines are schematic (station-to-station)."
)

# ----------------- Data loader -----------------
@st.cache_data
def load_data(stops_path="stops.csv", routes_path="routes.csv"):
    if not os.path.exists(stops_path) or not os.path.exists(routes_path):
        st.error("stops.csv or routes.csv not found")
        st.stop()

    stops = pd.read_csv(stops_path, dtype=str, comment="#")
    routes = pd.read_csv(routes_path, dtype=str, comment="#")

    stops["lat"] = pd.to_numeric(stops.get("lat"), errors="coerce")
    stops["lon"] = pd.to_numeric(stops.get("lon"), errors=_
