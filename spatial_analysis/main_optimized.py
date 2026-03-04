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

MIN_WALK_SECONDS      = 120   # skip stations with < 2 min remaining walk
WALK_BUFFER_M         = 20    # walking polygon edge buffer (metres, projected)
BIKE_BUFFER_M         = 40    # own-bike polygon edge buffer (metres, projected)
DISSOLVE_BRIDGE_M     = 100   # buffer to bridge gaps between walk blobs (island fix)
STATION_ENVELOPE_M    = 600   # convex-hull buffer around dock stations (bbox fix)
                               # tune: 400=tight, 600=default ~7min walk, 1000=loose
MAX_WORKERS           = 6     # parallel workers for station walk polygons


# ==============================================================================
# DATA + GRAPH HELPERS
# ==============================================================================

def get_official_nextbike_stations(city_names):
    """
    Fetch stations for one city name or a list of city names.
    Use a list when a bikesharing system is split across multiple
    API city entries (e.g. Valašské Meziříčí + its outlying villages).
    """
    if isinstance(city_names, str):
        city_names = [city_names]

    url = "https://api.nextbike.net/maps/nextbike-live.json"
    all_stations = []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"     [!] API returned status {response.status_code}")
            return []
        data = response.json()
        for country in data.get("countries", []):
            for city in country.get("cities", []):
                if city["name"] in city_names:
                    found = []
                    for p in city.get("places", []):
                        if p.get("spot"):
                            found.append({
                                "uid":  p["uid"],
                                "name": p["name"],
                                "lat":  float(p["lat"]),
                                "lon":  float(p["lng"]),
                            })
                    print(f"  -> '{city['name']}': {len(found)} stations fetched.")
                    all_stations.extend(found)
    except Exception as e:
        print(f"     [!] API error: {e}")

    print(f"  -> Total stations fetched: {len(all_stations)}")
    return all_stations


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


def _station_envelope(stations_gdf, buffer_m=STATION_ENVELOPE_M):
    """
    Build a WGS-84 polygon: convex hull around all reachable dock stations
    expanded by buffer_m metres. Clips the shared-bike result so it can never
    extend far along rural bike paths beyond where stations actually exist.

    Tune STATION_ENVELOPE_M at the top of the file:
      400 m  -> tight, stays very close to stations
      600 m  -> default, ~7 min walk margin around the station cluster
      1000 m -> loose, allows more extension along paths near stations
    """
    projected = stations_gdf.to_crs(epsg=32633)
    hull = projected.geometry.unary_union.convex_hull.buffer(buffer_m)
    return gpd.GeoDataFrame(geometry=[hull], crs="EPSG:32633").to_crs(epsg=4326).iloc[0].geometry


def _dissolve_and_clip(polygons_proj, own_bike_poly_wgs84, station_envelope=None):
    """
    Island fix: union all blobs, bridge small gaps with a +/- buffer pass,
    reproject to WGS-84, clip to the own-bike boundary, then optionally
    clip to the station envelope to prevent runaway rural extensions.
    """
    raw_union = gpd.GeoSeries(polygons_proj, crs="EPSG:32633").unary_union
    dissolved = raw_union.buffer(DISSOLVE_BRIDGE_M).buffer(-DISSOLVE_BRIDGE_M)
    final_gdf = gpd.GeoDataFrame(geometry=[dissolved], crs="EPSG:32633").to_crs(epsg=4326)

    # Clip 1: own-bike boundary
    final_gdf.geometry = final_gdf.geometry.intersection(own_bike_poly_wgs84)

    # Clip 2: station envelope — keeps result near actual dock locations
    if station_envelope is not None:
        final_gdf.geometry = final_gdf.geometry.intersection(station_envelope)

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
    walk_sub_lengths = nx.single_source_dijkstra_path_length(
        G_walk, origin_walk_node, cutoff=total_sec, weight="travel_time"
    )
    origin_sub_nodes = set(walk_sub_lengths.keys())
    mask = (
        projected_walk_edges.index.get_level_values("u").isin(origin_sub_nodes) &
        projected_walk_edges.index.get_level_values("v").isin(origin_sub_nodes)
    ) if projected_walk_edges is not None else None

    # 2. Pre-project walk edges if not supplied by caller
    if projected_walk_edges is None:
        projected_walk_edges = _project_graph_edges(G_walk)
        mask = (
            projected_walk_edges.index.get_level_values("u").isin(origin_sub_nodes) &
            projected_walk_edges.index.get_level_values("v").isin(origin_sub_nodes)
        )

    origin_edges = projected_walk_edges[mask]
    if not origin_edges.empty:
        polygons_proj.append(origin_edges.geometry.buffer(WALK_BUFFER_M).unary_union)

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

    # 4. Build station envelope — convex hull around reachable stations + buffer
    envelope = _station_envelope(valid_stations_gdf)
    print(f"     Station envelope built ({STATION_ENVELOPE_M}m buffer, {len(valid_stations_gdf)} stations).")

    # 5. Vectorised nearest_nodes — one batch call for all stations
    valid_stations_gdf["walk_node"] = ox.distance.nearest_nodes(
        G_walk, X=valid_stations_gdf["lon"].values, Y=valid_stations_gdf["lat"].values)
    valid_stations_gdf["bike_node"] = ox.distance.nearest_nodes(
        G_bike, X=valid_stations_gdf["lon"].values, Y=valid_stations_gdf["lat"].values)

    # 6. Compute remaining walk budget; drop stations below threshold
    def _remaining(row):
        bike_sec = predicted_times_dict.get(row["bike_node"], float("inf"))
        return total_sec - (bike_sec + 20)

    valid_stations_gdf["remaining_walk_sec"] = valid_stations_gdf.apply(_remaining, axis=1)
    schedulable = valid_stations_gdf[
        valid_stations_gdf["remaining_walk_sec"] >= MIN_WALK_SECONDS
    ].copy()
    print(f"     {len(schedulable)} stations have >= {MIN_WALK_SECONDS}s remaining walk time.")

    for _, s in schedulable.iterrows():
        reached_stations.append({
            "uid":   s.get("uid", "unknown"),
            "name":  s.get("name", "Nextbike Station"),
            "geometry": s["geometry"],
            "bike_travel_time_min": round((total_sec - s["remaining_walk_sec"]) / 60, 1),
            "remaining_walk_min":   round(s["remaining_walk_sec"] / 60, 1),
        })

        # 7. Walk polygon computation with corrected dominance pruning
        schedulable_sorted = schedulable.sort_values("remaining_walk_sec", ascending=False)

        walk_node_to_remaining = {
            int(row["walk_node"]): float(row["remaining_walk_sec"])
            for _, row in schedulable_sorted.iterrows()
        }
        all_walk_nodes = set(walk_node_to_remaining.keys())
        skipped = set()
        computed = 0

        print(f"     Computing walk polygons with dominance pruning...")
        for _, row in schedulable_sorted.iterrows():
            walk_node = int(row["walk_node"])
            remaining = float(row["remaining_walk_sec"])

            if walk_node in skipped:
                continue

            try:
                reachable = nx.single_source_dijkstra_path_length(
                    G_walk, walk_node, cutoff=remaining, weight="travel_time"
                )
            except nx.NetworkXError:
                continue

            reachable_nodes = set(reachable.keys())
            computed += 1

            # Corrected dominance: A dominates B only if A reaches B_node
            # AND still has enough time left to cover B's full polygon
            #   travel_time(A → B) ≤ A_remaining - B_remaining
            for other_node, other_remaining in walk_node_to_remaining.items():
                if other_node == walk_node or other_node in skipped:
                    continue
                travel_to_other = reachable.get(other_node)  # None if not reachable
                if travel_to_other is not None:
                    spare_time = remaining - travel_to_other
                    if spare_time >= other_remaining:  # A covers everything B would
                        skipped.add(other_node)

            mask = (
                    projected_walk_edges.index.get_level_values("u").isin(reachable_nodes) &
                    projected_walk_edges.index.get_level_values("v").isin(reachable_nodes)
            )
            sub_edges = projected_walk_edges[mask]
            if not sub_edges.empty:
                polygons_proj.append(sub_edges.geometry.buffer(WALK_BUFFER_M).unary_union)

        print(f"     Polygons computed: {computed} | Dominated & skipped: {len(skipped)}")

    # 8. Dissolve (island fix) + clip to own-bike boundary + clip to station envelope
    if polygons_proj:
        final_gdf = _dissolve_and_clip(polygons_proj, own_bike_poly, station_envelope=envelope)
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

    # Scripts live in spatial_analysis/, data/ is one level up at repo root
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FILE_NAME = os.path.join(REPO_ROOT, "data", "nextbike_data_VSB_Vaclavik.xlsx")

    CITY_SHEET_MAPPING = {
        # "Ostrava":           "Vypujcky_Ostrava",
        # "Ostrava hlavní":    "Vypujcky_Ostrava",
        # "Brno":              "Vypujcky_Brno",
        # "Přerov":            "Vypujcky_Prerov",
        "Valašské Meziříčí": "Vypujcky_ValMez",
    }

    # Maps the city label to ALL Nextbike API city names that belong to the same system.
    # Single string = one API city. List = merge multiple API cities into one system.
    CITY_API_NAMES = {
        # "Ostrava":           "Ostrava",
        # "Ostrava hlavní":    "Ostrava hlavní",
        # "Brno":              "Brno",
        # "Přerov":            "Přerov",
        "Valašské Meziříčí": ["Valašské Meziříčí", "Poličná", "Krhová", "Zašová"],
    }

    CITY_ORIGINS = {
        # "Ostrava":        ["SV-Svinov nádraží *(navíc 15min na odjezd)"],
        # "Ostrava hlavní": ["MOAP-Hlavní nádraží"],
        # "Brno":           ["Hlavní nádraží - Hlavní vstup", "Hlavní nádraží - pošta", "Bajkazyl 666"],
        # "Přerov":         ["Nádraží"],
        "Valašské Meziříčí": ["Vlakové nádraží Valašské Meziříčí (nové umístění)"],
    }
    TIME_BUDGETS = [15, 30]   # minutes — add/remove freely

    OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "spatial_analysis")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Verify data file exists before doing any work ──────────────────────────
    if not os.path.exists(FILE_NAME):
        raise FileNotFoundError(
            f"Data file not found: {FILE_NAME}\n"
            f"Make sure 'data/nextbike_data_VSB_Vaclavik.xlsx' exists in repo root: {REPO_ROOT}"
        )

    # ── PER-CITY LOOP ──────────────────────────────────────────────────────────
    for MAPPING_CITY_NAME, ORIGIN_STATION_NAMES in CITY_ORIGINS.items():

            TRAINING_SHEET = CITY_SHEET_MAPPING.get(MAPPING_CITY_NAME)
            if not TRAINING_SHEET:
                print(f"  [!] '{MAPPING_CITY_NAME}' missing from CITY_SHEET_MAPPING, skipping.")
                continue

            # ── PHASE 1: ML model (once per city) ─────────────────────────────────
            ai_model = get_or_train_local_model(
                FILE_NAME, TRAINING_SHEET, MAPPING_CITY_NAME, sample_size=800)

            if ai_model is None:
                raise RuntimeError(
                    f"Model training failed for '{MAPPING_CITY_NAME}'.\n"
                    f"  Check 1 — file exists:  {FILE_NAME}\n"
                    f"  Check 2 — sheet exists: '{TRAINING_SHEET}'\n"
                    f"  Check 3 — sheet has columns: start_lat, start_lng, end_lat, end_lng, duration"
                )

            # ── PHASE 2: Stations + graphs (once per city) ────────────────────────
            api_names = CITY_API_NAMES.get(MAPPING_CITY_NAME, MAPPING_CITY_NAME)
            stations = get_official_nextbike_stations(api_names)
            if not stations:
                print(f"  [!] No stations returned for {MAPPING_CITY_NAME}, skipping.")
                continue

            # Use the first listed origin as the geographic centre for graph download
            first_lat, first_lon = resolve_origin_from_station_name(stations, ORIGIN_STATION_NAMES[0])

            city_key = MAPPING_CITY_NAME.lower().replace(" ", "_")
            bike_graph_file = os.path.join(REPO_ROOT, f"{city_key}_bike_G.graphml")
            walk_graph_file = os.path.join(REPO_ROOT, f"{city_key}_walk_G.graphml")

            print(f"\n--- PHASE 2: GRAPHS FOR {MAPPING_CITY_NAME.upper()} ---")
            if os.path.exists(bike_graph_file):
                print(f"  -> Loading cached bike graph: {bike_graph_file}")
                G_bike = ox.load_graphml(bike_graph_file)
            else:
                print(f"  -> Downloading bike graph for {MAPPING_CITY_NAME}...")
                urban_bike_filter = (
                    '["highway"]["area"!~"yes"]["access"!~"private"]'
                    '["highway"!~"motorway|motorway_link|trunk|trunk_link|steps"]'
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
            distances_dict = nx.single_source_dijkstra_path_length(
                G_bike, origin_bike_node, weight="length")
            reachable_nodes = {n: d for n, d in distances_dict.items() if d <= 15000}

            coords_to_fetch = [(G_bike.nodes[n]["y"], G_bike.nodes[n]["x"]) for n in reachable_nodes]
            coords_to_fetch.insert(0, (first_lat, first_lon))
            coords_to_fetch += [(s["lat"], s["lon"]) for s in stations]
            coords_to_fetch = list(set(coords_to_fetch))

            network_elev_cache = os.path.join(REPO_ROOT, f"{city_key}_network_elevations.json")
            elev_dict = get_elevations_cached(coords_to_fetch, cache_file=network_elev_cache)
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
                city_prefix = f"{city_key}_{station_slug}"

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