import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
import requests
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from ml_engine import get_elevations_cached, get_or_train_local_model

warnings.filterwarnings("ignore")

MIN_WALK_SECONDS  = 120   # skip stations with < 2 min remaining walk
WALK_BUFFER_M     = 20    # walking polygon edge buffer (metres, projected)
BIKE_BUFFER_M     = 40    # own-bike polygon edge buffer (metres, projected)
DISSOLVE_BRIDGE_M = 100   # buffer to bridge gaps between walk blobs (island fix)
MAX_WORKERS       = 6     # parallel workers for station walk polygons


# ==============================================================================
# DATA + GRAPH HELPERS
# ==============================================================================

def get_official_nextbike_stations(city_name):
    print(f"  -> Fetching live Nextbike data for {city_name}...")
    url = "https://api.nextbike.net/maps/nextbike-live.json"
    stations = []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for country in data.get("countries", []):
                for city in country.get("cities", []):
                    if city["name"] == city_name:
                        for p in city.get("places", []):
                            if p.get("spot"):
                                stations.append({
                                    "uid":  p["uid"],
                                    "name": p["name"],
                                    "lat":  float(p["lat"]),
                                    "lon":  float(p["lng"]),
                                })
                        return stations
    except Exception as e:
        print(f"     [!] API error: {e}")
    return stations


def resolve_origin_from_station_name(stations, station_name):
    """Look up GPS coords by Nextbike station name (exact, then partial match)."""
    for s in stations:
        if s["name"].strip().lower() == station_name.strip().lower():
            print(f"  -> Origin resolved: '{s['name']}' -> ({s['lat']}, {s['lon']})")
            return s["lat"], s["lon"]
    partial = [s for s in stations if station_name.strip().lower() in s["name"].strip().lower()]
    if partial:
        suggestions = ", ".join(f"'{s['name']}'" for s in partial[:5])
        raise ValueError(f"Station '{station_name}' not found. Did you mean: {suggestions}?")
    raise ValueError(
        f"Station '{station_name}' not found in {len(stations)} stations. "
        f"Run [s['name'] for s in stations] to list all available names."
    )


# ==============================================================================
# ISOCHRONE GENERATORS
# ==============================================================================

def generate_own_bike_polygon(G_bike, predicted_times_dict, total_mins, output_filename):
    print(f"\n  -> Generating OWN BIKE | {total_mins} mins...")
    total_sec = total_mins * 60
    valid_node_ids = [n for n, t in predicted_times_dict.items() if t <= total_sec]
    sub_graph = G_bike.subgraph(valid_node_ids)
    edges = ox.graph_to_gdfs(sub_graph, nodes=False, edges=True)
    if not edges.empty:
        poly = edges.to_crs(epsg=32633).geometry.buffer(BIKE_BUFFER_M).unary_union
        final_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:32633").to_crs(epsg=4326)
        final_gdf.to_file(output_filename, driver="GeoJSON")
        print(f"     Saved: {output_filename}")
        return final_gdf.iloc[0].geometry
    return None


def _project_graph_edges(G_walk):
    """Project all walk edges to EPSG:32633 once; reused by every station worker."""
    return ox.graph_to_gdfs(G_walk, nodes=False, edges=True).to_crs(epsg=32633)


def _station_walk_polygon(walk_node, remaining_sec, G_walk, projected_edges_gdf):
    """Compute walking reachability polygon for one station (runs in a thread)."""
    subgraph = nx.ego_graph(G_walk, walk_node, radius=remaining_sec, distance="travel_time")
    if subgraph.number_of_edges() == 0:
        return None
    sub_nodes = set(subgraph.nodes())
    mask = (
        projected_edges_gdf.index.get_level_values("u").isin(sub_nodes) &
        projected_edges_gdf.index.get_level_values("v").isin(sub_nodes)
    )
    sub_edges = projected_edges_gdf[mask]
    if sub_edges.empty:
        return None
    return sub_edges.geometry.buffer(WALK_BUFFER_M).unary_union


def _dissolve_and_clip(polygons_proj, own_bike_poly_wgs84):
    """
    Island fix: union all blobs, bridge small gaps with a +/- buffer pass,
    reproject to WGS-84, then clip to the own-bike boundary.
    """
    raw_union = gpd.GeoSeries(polygons_proj, crs="EPSG:32633").unary_union
    dissolved = raw_union.buffer(DISSOLVE_BRIDGE_M).buffer(-DISSOLVE_BRIDGE_M)
    final_gdf = gpd.GeoDataFrame(geometry=[dissolved], crs="EPSG:32633").to_crs(epsg=4326)
    final_gdf.geometry = final_gdf.geometry.intersection(own_bike_poly_wgs84)
    return final_gdf


def generate_shared_bike_isochrone(G_bike, G_walk, own_bike_poly, predicted_times_dict,
                                    stations, origin_lat, origin_lon, total_mins,
                                    output_filename, projected_walk_edges=None):
    print(f"\n  -> Generating SHARED BIKE | {total_mins} mins...")
    total_sec = total_mins * 60
    polygons_proj = []
    reached_stations = []

    # 1. Origin walk polygon
    origin_walk_node = ox.distance.nearest_nodes(G_walk, X=origin_lon, Y=origin_lat)
    walk_sub = nx.ego_graph(G_walk, origin_walk_node, radius=total_sec, distance="travel_time")
    walk_edges = ox.graph_to_gdfs(walk_sub, nodes=False, edges=True)
    if not walk_edges.empty:
        polygons_proj.append(walk_edges.to_crs(epsg=32633).geometry.buffer(WALK_BUFFER_M).unary_union)

    # 2. Pre-project walk edges if not supplied by caller
    if projected_walk_edges is None:
        projected_walk_edges = _project_graph_edges(G_walk)

    # 3. Filter stations to those inside the own-bike polygon
    stations_df = pd.DataFrame(stations)
    stations_gdf = gpd.GeoDataFrame(stations_df,
        geometry=gpd.points_from_xy(stations_df.lon, stations_df.lat), crs="EPSG:4326")
    valid_stations_gdf = stations_gdf[stations_gdf.geometry.within(own_bike_poly)].copy()
    print(f"     Filtered to {len(valid_stations_gdf)} stations inside the bike zone.")

    if valid_stations_gdf.empty:
        if polygons_proj:
            _dissolve_and_clip(polygons_proj, own_bike_poly).to_file(output_filename, driver="GeoJSON")
        return

    # 4. Vectorised nearest_nodes — one batch call for all stations
    valid_stations_gdf["walk_node"] = ox.distance.nearest_nodes(
        G_walk, X=valid_stations_gdf["lon"].values, Y=valid_stations_gdf["lat"].values)
    valid_stations_gdf["bike_node"] = ox.distance.nearest_nodes(
        G_bike, X=valid_stations_gdf["lon"].values, Y=valid_stations_gdf["lat"].values)

    # 5. Compute remaining walk budget; drop stations below threshold
    def _remaining(row):
        bike_sec = predicted_times_dict.get(row["bike_node"], float("inf"))
        return total_sec - (bike_sec + 20)

    valid_stations_gdf["remaining_walk_sec"] = valid_stations_gdf.apply(_remaining, axis=1)
    schedulable = valid_stations_gdf[valid_stations_gdf["remaining_walk_sec"] >= MIN_WALK_SECONDS]
    print(f"     {len(schedulable)} stations have >= {MIN_WALK_SECONDS}s remaining walk time.")

    for _, s in schedulable.iterrows():
        reached_stations.append({
            "uid":   s.get("uid", "unknown"),
            "name":  s.get("name", "Nextbike Station"),
            "geometry": s["geometry"],
            "bike_travel_time_min": round((total_sec - s["remaining_walk_sec"]) / 60, 1),
            "remaining_walk_min":   round(s["remaining_walk_sec"] / 60, 1),
        })

    # 6. Parallel walk polygon computation
    print(f"     Computing {len(schedulable)} walk polygons in parallel (workers={MAX_WORKERS})...")
    rows = list(schedulable.iterrows())

    def _task(idx_row):
        _, row = idx_row
        return _station_walk_polygon(int(row["walk_node"]), float(row["remaining_walk_sec"]),
                                     G_walk, projected_walk_edges)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_task, item): item for item in rows}
        for future in as_completed(futures):
            poly = future.result()
            if poly is not None:
                polygons_proj.append(poly)

    # 7. Dissolve (island fix) and save
    if polygons_proj:
        final_gdf = _dissolve_and_clip(polygons_proj, own_bike_poly)
        final_gdf.to_file(output_filename, driver="GeoJSON")
        print(f"     Saved Area: {output_filename}")

    if reached_stations:
        stations_out_gdf = gpd.GeoDataFrame(reached_stations, crs="EPSG:4326")
        stations_filename = output_filename.replace("isochrone_", "stations_")
        stations_out_gdf.to_file(stations_filename, driver="GeoJSON")
        print(f"     Saved Stations: {stations_filename}")


# ==============================================================================
# MASTER EXECUTION
# ==============================================================================

if __name__ == "__main__":

    # ── CONFIGURATION — edit only this section ─────────────────────────────────
    FILE_NAME = r"C:\Users\vaclavikmartin\PycharmProjects\Nextbike\data\nextbike_data_VSB_Vaclavik.xlsx"

    CITY_SHEET_MAPPING = {
        # "Ostrava":           "Vypujcky_Ostrava",
        # "Ostrava hlavní":    "Vypujcky_Ostrava",
        # "Brno":              "Vypujcky_Brno",
        # "Přerov":            "Vypujcky_Prerov",
        "Valašské Meziříčí": "Vypujcky_ValMez",
    }

    CITY_ORIGINS = {
        # "Ostrava":        ["SV-Svinov nádraží *(navíc 15min na odjezd)"],
        # "Ostrava hlavní": ["MOAP-Hlavní nádraží"],
        # "Brno":           ["Hlavní nádraží - Hlavní vstup", "Hlavní nádraží - pošta", "Bajkazyl 666"],
        # "Přerov":         ["Nádraží"],
        "Valašské Meziříčí": ["Vlakové nádraží Valašské Meziříčí (nové umístění)"],
    }

    TIME_BUDGETS = [15, 30]   # minutes — add/remove freely

    OUTPUT_DIR = os.path.join("results", "spatial_analysis")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── PER-CITY LOOP ──────────────────────────────────────────────────────────
    for MAPPING_CITY_NAME, ORIGIN_STATION_NAMES in CITY_ORIGINS.items():

        TRAINING_SHEET = CITY_SHEET_MAPPING.get(MAPPING_CITY_NAME)
        if not TRAINING_SHEET:
            print(f"  [!] '{MAPPING_CITY_NAME}' missing from CITY_SHEET_MAPPING, skipping.")
            continue

        # ── PHASE 1: ML model (once per city) ─────────────────────────────────
        ai_model = get_or_train_local_model(
            FILE_NAME, TRAINING_SHEET, MAPPING_CITY_NAME, sample_size=800)

        # ── PHASE 2: Stations + graphs (once per city) ────────────────────────
        stations = get_official_nextbike_stations(MAPPING_CITY_NAME)
        if not stations:
            print(f"  [!] No stations returned for {MAPPING_CITY_NAME}, skipping.")
            continue

        # Use the first listed origin as the geographic centre for graph download
        first_lat, first_lon = resolve_origin_from_station_name(stations, ORIGIN_STATION_NAMES[0])

        city_key        = MAPPING_CITY_NAME.lower().replace(" ", "_")
        bike_graph_file = f"{city_key}_bike_G.graphml"
        walk_graph_file = f"{city_key}_walk_G.graphml"

        print(f"\n--- PHASE 2: GRAPHS FOR {MAPPING_CITY_NAME.upper()} ---")
        if os.path.exists(bike_graph_file):
            print(f"  -> Loading cached bike graph: {bike_graph_file}")
            G_bike = ox.load_graphml(bike_graph_file)
        else:
            print(f"  -> Downloading bike graph for {MAPPING_CITY_NAME}...")
            urban_bike_filter = (
                '["highway"]["area"!~"yes"]["access"!~"private"]'
                '["highway"!~"motorway|motorway_link|trunk|trunk_link|track|steps"]'
                '["surface"!~"unpaved|dirt|sand|grass|gravel|mud"]'
            )
            G_bike = ox.graph_from_point(
                (first_lat, first_lon), dist=8500, custom_filter=urban_bike_filter)
            ox.save_graphml(G_bike, bike_graph_file)

        if os.path.exists(walk_graph_file):
            print(f"  -> Loading cached walk graph: {walk_graph_file}")
            G_walk = ox.load_graphml(walk_graph_file)
        else:
            print(f"  -> Downloading walk graph for {MAPPING_CITY_NAME}...")
            G_walk = ox.graph_from_point(
                (first_lat, first_lon), dist=8500, network_type="walk")
            U_walk = G_walk.to_undirected()
            largest_cc = max(nx.connected_components(U_walk), key=len)
            G_walk = G_walk.subgraph(largest_cc).copy()
            print("     Topology: removed isolated pedestrian islands.")
            G_walk = ox.add_edge_speeds(G_walk, fallback=5.0)
            G_walk = ox.add_edge_travel_times(G_walk)
            ox.save_graphml(G_walk, walk_graph_file)

        # ── PHASE 3: Master time dictionary (once per city) ───────────────────
        print(f"\n--- PHASE 3: MASTER TIME DICTIONARY FOR {MAPPING_CITY_NAME.upper()} ---")
        origin_bike_node = ox.distance.nearest_nodes(G_bike, X=first_lon, Y=first_lat)
        distances_dict   = nx.single_source_dijkstra_path_length(
            G_bike, origin_bike_node, weight="length")
        reachable_nodes  = {n: d for n, d in distances_dict.items() if d <= 15000}

        coords_to_fetch  = [(G_bike.nodes[n]["y"], G_bike.nodes[n]["x"]) for n in reachable_nodes]
        coords_to_fetch.insert(0, (first_lat, first_lon))
        coords_to_fetch += [(s["lat"], s["lon"]) for s in stations]
        coords_to_fetch  = list(set(coords_to_fetch))

        network_elev_cache = f"{city_key}_network_elevations.json"
        elev_dict   = get_elevations_cached(coords_to_fetch, cache_file=network_elev_cache)
        origin_elev = elev_dict.get(f"{first_lat},{first_lon}", 0.0)

        nodes_data = [
            [n, dist,
             elev_dict.get(f"{G_bike.nodes[n]['y']},{G_bike.nodes[n]['x']}", 0.0) - origin_elev]
            for n, dist in reachable_nodes.items()
        ]
        node_features_df = pd.DataFrame(
            nodes_data, columns=["node_id", "network_distance_m", "elevation_delta_m"]
        ).set_index("node_id")

        print("  -> Predicting travel times for all intersections...")
        all_predictions = ai_model.predict(
            node_features_df[["network_distance_m", "elevation_delta_m"]])
        predicted_times_dict = dict(zip(node_features_df.index, all_predictions))

        # ── PHASE 4: Pre-project walk edges (once per city) ───────────────────
        print(f"\n--- PHASE 4: PRE-PROJECTING WALK GRAPH ---")
        projected_walk_edges = _project_graph_edges(G_walk)
        print(f"  -> {len(projected_walk_edges)} walk edges ready.")

        # ── PHASE 5: Isochrones for each origin station ───────────────────────
        print(f"\n--- PHASE 5: ISOCHRONES FOR {MAPPING_CITY_NAME.upper()} | {TIME_BUDGETS} MIN ---")
        for ORIGIN_STATION_NAME in ORIGIN_STATION_NAMES:
            ORIGIN_LAT, ORIGIN_LON = resolve_origin_from_station_name(stations, ORIGIN_STATION_NAME)
            station_slug = (ORIGIN_STATION_NAME.lower()
                            .replace(" ", "_").replace("-", "").replace("/", "_"))
            city_prefix  = f"{city_key}_{station_slug}"

            for mins in TIME_BUDGETS:
                poly_own = generate_own_bike_polygon(
                    G_bike, predicted_times_dict, mins,
                    os.path.join(OUTPUT_DIR, f"isochrone_{city_prefix}_own_bike_{mins}min.geojson"),
                )
                if poly_own is not None:
                    generate_shared_bike_isochrone(
                        G_bike, G_walk, poly_own,
                        predicted_times_dict, stations,
                        ORIGIN_LAT, ORIGIN_LON, mins,
                        os.path.join(OUTPUT_DIR, f"isochrone_{city_prefix}_shared_bike_{mins}min.geojson"),
                        projected_walk_edges=projected_walk_edges,
                    )

    print("\n*** ALL MAPS SUCCESSFULLY COMPLETED! ***")