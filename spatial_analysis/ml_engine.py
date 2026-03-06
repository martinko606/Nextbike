import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from math import radians, cos, sin, asin, sqrt
import requests
import time
import math
import json
import joblib
import unicodedata


def _repo_root():
    """Return the repository root — one level above the spatial_analysis/ folder."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _city_slug(city_name):
    normalized = unicodedata.normalize("NFD", city_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().replace(" ", "_").replace("-", "_")


def _cache_dir(subfolder):
    """Return path to a cache subfolder, creating it if it doesn't exist."""
    path = os.path.join(_repo_root(), "cache", subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    return 2 * R * asin(sqrt(sin(dp/2)**2 + cos(p1) * cos(p2) * sin(dl/2)**2))


def get_elevations_cached(coords_list, cache_file):
    if os.path.exists(cache_file):
        print(f"  -> Loading elevations from cache: {os.path.basename(cache_file)}")
        with open(cache_file, "r") as f:
            return json.load(f)

    if not coords_list:
        return {}

    elevations_dict = {}
    batch_size    = 100
    total_batches = math.ceil(len(coords_list) / batch_size)
    print(f"  -> Cache not found. Downloading {len(coords_list)} elevations "
          f"({total_batches} batches)...")

    for i in range(0, len(coords_list), batch_size):
        batch_num = (i // batch_size) + 1
        if batch_num % 10 == 0 or batch_num == total_batches:
            print(f"     ... processed {batch_num}/{total_batches} batches")

        batch = coords_list[i:i + batch_size]
        locations_str = "|".join([f"{lat},{lng}" for lat, lng in batch])
        url = f"https://api.opentopodata.org/v1/aster30m?locations={locations_str}"

        for attempt in range(3):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for idx, result in enumerate(data.get("results", [])):
                        key = f"{batch[idx][0]},{batch[idx][1]}"
                        elevations_dict[key] = result.get("elevation", 0.0)
                    break
                else:
                    time.sleep(2)
            except Exception:
                if attempt == 2:
                    print(f"     [!] Batch {batch_num} timed out after 3 attempts.")
                time.sleep(2)
        time.sleep(1.2)

    with open(cache_file, "w") as f:
        json.dump(elevations_dict, f)
    print(f"  -> Saved: cache/elevations/{os.path.basename(cache_file)}")
    return elevations_dict


def get_or_train_local_model(file_path, sheet_name, city_name, sample_size=800,
                              duration_min=60, duration_max=1800):
    slug           = _city_slug(city_name)
    models_dir     = _cache_dir("models")
    model_filename = os.path.join(models_dir, f"{slug}_rf_model.pkl")

    # Load cached model
    if os.path.exists(model_filename):
        print(f"\n--- PHASE 1: LOADING CACHED ML MODEL — {city_name.upper()} ---")
        print(f"  -> Found: cache/models/{slug}_rf_model.pkl")
        return joblib.load(model_filename)

    # Train from scratch
    print(f"\n--- PHASE 1: TRAINING LOCAL ML MODEL — {city_name.upper()} ---")

    if not os.path.exists(file_path):
        print(f"  [!] Excel file not found: {file_path}")
        return None

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except ValueError:
        print(f"  [!] Sheet '{sheet_name}' not found in {file_path}.")
        print(f"      Run pd.ExcelFile(file_path).sheet_names to list available sheets.")
        return None

    df.columns = df.columns.str.strip()
    required_cols = ["start_lat", "start_lng", "end_lat", "end_lng", "duration"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  [!] Missing columns in sheet '{sheet_name}': {missing}")
        return None

    # Drop round trips and duration outliers
    df = df[~((df["start_lat"] == df["end_lat"]) & (df["start_lng"] == df["end_lng"]))]
    df = df[
        (df["duration"] >= duration_min) & (df["duration"] <= duration_max)
    ].dropna(subset=required_cols)

    print(f"  -> Trips after filtering: {len(df)} "
          f"(duration {duration_min}-{duration_max}s, sample_size={sample_size})")

    if df.empty:
        print(f"  [!] No valid trips found for {city_name} after filtering.")
        return None

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Fetch elevations
    coords = list(
        set(zip(df["start_lat"], df["start_lng"])) |
        set(zip(df["end_lat"],   df["end_lng"]))
    )
    elev_cache = os.path.join(_cache_dir("elevations"), f"{slug}_training_elevations.json")
    elev_dict  = get_elevations_cached(coords, cache_file=elev_cache)

    df["elev_start"] = df.apply(
        lambda r: elev_dict.get(f"{r['start_lat']},{r['start_lng']}", 0.0), axis=1)
    df["elev_end"] = df.apply(
        lambda r: elev_dict.get(f"{r['end_lat']},{r['end_lng']}", 0.0), axis=1)
    df["elevation_delta_m"] = df["elev_end"] - df["elev_start"]

    # Straight-line distance + detour ratio
    df["straight_line_m"] = df.apply(
        lambda r: _haversine(r["start_lat"], r["start_lng"], r["end_lat"], r["end_lng"]), axis=1)

    # Network distances
    print(f"  -> Calculating historic network distances for {city_name}...")
    n = df["start_lat"].max() + 0.02
    s = df["start_lat"].min() - 0.02
    e = df["start_lng"].max() + 0.02
    w = df["start_lng"].min() - 0.02
    G = ox.graph_from_bbox(bbox=(w, s, e, n), network_type="bike")

    distances = []
    for index, row in df.iterrows():
        try:
            start_node = ox.distance.nearest_nodes(G, X=row["start_lng"], Y=row["start_lat"])
            end_node   = ox.distance.nearest_nodes(G, X=row["end_lng"],   Y=row["end_lat"])
            dist = nx.shortest_path_length(G, start_node, end_node, weight="length")
            distances.append((index, dist))
        except Exception:
            distances.append((index, np.nan))

    df = df.join(
        pd.DataFrame(distances, columns=["index", "network_distance_m"]).set_index("index")
    )
    df = df.dropna(subset=["network_distance_m"])

    if df.empty:
        print(f"  [!] No routable trips found for {city_name}. Skipping model training.")
        return None

    df["detour_ratio"] = df["network_distance_m"] / df["straight_line_m"].clip(lower=1)

    features = ["network_distance_m", "elevation_delta_m", "straight_line_m", "detour_ratio"]
    X = df[features]
    y = df["duration"]

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    score = model.score(X, y)
    print(f"  -> Random Forest trained for {city_name}! (R2 = {round(score, 3)})")

    joblib.dump(model, model_filename)
    print(f"  -> Saved: cache/models/{slug}_rf_model.pkl")

    return model