import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
import requests
import time
import math
import os
import json
import joblib


def get_elevations_cached(coords_list, cache_file):
    """Fetches elevations, saving them locally so we never wait for the API again."""
    elevations_dict = {}

    if os.path.exists(cache_file):
        print(f"  -> SUCCESS: Loading elevations from local cache: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    batch_size = 100
    total_batches = math.ceil(len(coords_list) / batch_size)
    print(f"  -> Cache not found. Downloading {len(coords_list)} elevations ({total_batches} batches)...")

    for i in range(0, len(coords_list), batch_size):
        batch_num = (i // batch_size) + 1
        if batch_num % 10 == 0 or batch_num == total_batches:
            print(f"     ... processed {batch_num}/{total_batches} batches")

        batch = coords_list[i:i + batch_size]
        locations_str = "|".join([f"{lat},{lng}" for lat, lng in batch])
        url = f"https://api.opentopodata.org/v1/aster30m?locations={locations_str}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for idx, result in enumerate(data.get('results', [])):
                        key = f"{batch[idx][0]},{batch[idx][1]}"
                        elevations_dict[key] = result.get('elevation', 0.0)
                    break
                else:
                    time.sleep(2)
            except Exception:
                if attempt == max_retries - 1:
                    print(f"     [!] Batch {batch_num} timed out.")
                time.sleep(2)
        time.sleep(1.2)

    with open(cache_file, 'w') as f:
        json.dump(elevations_dict, f)
    print(f"  -> Elevation data permanently saved to {cache_file}")

    return elevations_dict


def get_or_train_local_model(file_path, sheet_name, city_name, sample_size=800):
    """Trains a city-specific AI model, or loads it if already trained."""
    model_filename = f"{city_name.lower()}_rf_model.pkl"

    # Check if we already trained this city's model!
    if os.path.exists(model_filename):
        print(f"\n--- PHASE 1: LOADING CACHED ML MODEL FOR {city_name.upper()} ---")
        return joblib.load(model_filename)

    print(f"\n--- PHASE 1: TRAINING LOCAL ML MODEL FOR {city_name.upper()} ---")

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except ValueError:
        print(f"     [!] Error: Could not find sheet '{sheet_name}' in the Excel file.")
        return None

    df.columns = df.columns.str.strip()
    required_cols = ['start_lat', 'start_lng', 'end_lat', 'end_lng', 'duration']
    df = df[~((df['start_lat'] == df['end_lat']) & (df['start_lng'] == df['end_lng']))]
    df = df[(df['duration'] >= 120) & (df['duration'] <= 2700)].dropna(subset=required_cols)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    print(f"     Loaded {len(df)} valid historical trips for {city_name}.")

    coords = list(set(zip(df['start_lat'], df['start_lng'])) | set(zip(df['end_lat'], df['end_lng'])))
    cache_filename = f"{city_name.lower()}_training_elevations.json"
    elev_dict = get_elevations_cached(coords, cache_file=cache_filename)

    df['elev_start'] = df.apply(lambda r: elev_dict.get(f"{r['start_lat']},{r['start_lng']}", 0.0), axis=1)
    df['elev_end'] = df.apply(lambda r: elev_dict.get(f"{r['end_lat']},{r['end_lng']}", 0.0), axis=1)
    df['elevation_delta_m'] = df['elev_end'] - df['elev_start']

    print("     Calculating historic network distances for training...")
    distances = []
    n, s = df['start_lat'].max() + 0.02, df['start_lat'].min() - 0.02
    e, w = df['start_lng'].max() + 0.02, df['start_lng'].min() - 0.02
    G = ox.graph_from_bbox(bbox=(w, s, e, n), network_type='bike')

    for index, row in df.iterrows():
        try:
            start_node = ox.distance.nearest_nodes(G, X=row['start_lng'], Y=row['start_lat'])
            end_node = ox.distance.nearest_nodes(G, X=row['end_lng'], Y=row['end_lat'])
            dist = nx.shortest_path_length(G, start_node, end_node, weight='length')
            distances.append((index, dist))
        except Exception:
            distances.append((index, np.nan))

    df = df.join(pd.DataFrame(distances, columns=['index', 'network_distance_m']).set_index('index'))
    df = df.dropna(subset=['network_distance_m'])

    X = df[['network_distance_m', 'elevation_delta_m']]
    y = df['duration']
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    score = model.score(X, y)
    print(f"-> Localized Random Forest Trained! (R² Score: {round(score, 3)})")

    # Save the trained brain to the hard drive!
    joblib.dump(model, model_filename)
    print(f"-> Model saved to {model_filename}")

    return model