import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
import requests
import warnings
from math import radians, cos, sin, asin, sqrt as msqrt
from shapely.ops import unary_union

from ml_engine import get_elevations_cached, get_or_train_local_model

warnings.filterwarnings("ignore")

MIN_WALK_SECONDS      = 60    # skip stations with < 1 min remaining walk
WALK_BUFFER_M         = 20    # internal walk edge buffer (metres) — OSMnx fallback
BOUNDARY_BUFFER_M     = 30    # frontier walk edge buffer — OSMnx fallback
BIKE_BUFFER_M         = 40    # own-bike polygon edge buffer (metres, projected)
DISSOLVE_BRIDGE_M     = 150   # gap-bridging buffer for dissolve pass
WALK_SPEED_MS         = 5 / 3.6    # 5 km/h in m/s

# Physics floor: predictions cannot imply faster than this average speed.
MAX_CYCLING_SPEED_MS  = 15 / 3.6   # 15 km/h in m/s

# r5py walk routing — used for egress walk polygons from dock stations.
# Set USE_R5PY_WALK = False to fall back to OSMnx boundary-edge method.
USE_R5PY_WALK         = True
WALK_SPEED_KMH        = 5.0   # must match WALK_SPEED_MS above


# ==============================================================================
# HELPERS
# ==============================================================================

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    return 2 * R * asin(msqrt(sin(dp/2)**2 + cos(p1) * cos(p2) * sin(dl/2)**2))


def get_official_nextbike_stations(city_names):
    """
    Fetch stations for one city name or a list of city names.
    Use a list when the system is split across multiple API entries
    (e.g. Valasske Mezirici + outlying villages Policna, Krhova, Zasova).
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
# R5PY WALK ROUTING
# ==============================================================================

def _build_r5py_walk_network(osm_pbf_path):
    """
    Build an r5py TransportNetwork for walk-only routing.
    No GTFS needed — pass an empty list for transit feeds.
    Cached automatically by r5py as a .dat file next to the .pbf.
    """
    try:
        import r5py
        print(f"  -> Building r5py walk network from {os.path.basename(osm_pbf_path)}...")
        print(f"     (First run builds cache ~1-3 min; subsequent runs load instantly.)")
        network = r5py.TransportNetwork(osm_pbf_path, [])
        print(f"  -> r5py walk network ready.")
        return network
    except ImportError:
        print("  [!] r5py not available — falling back to OSMnx boundary-edge walk method.")
        return None
    except Exception as e:
        print(f"  [!] r5py network build failed ({e}) — falling back to OSMnx method.")
        return None


def _r5py_walk_polygons(r5py_network, station_rows, total_sec, depart_datetime):
    """
    For each schedulable dock station, compute a walk isochrone using r5py.

    R5 routes egress walk on real OSM streets — much more accurate than
    circle buffers, and geometrically correct unlike the OSMnx boundary-edge
    method which only approximates the frontier.

    Parameters
    ----------
    r5py_network  : r5py.TransportNetwork (walk-only)
    station_rows  : list of dicts with keys: name, lat, lon, remaining_walk_sec
    total_sec     : total time budget in seconds (used for departure time)
    depart_datetime : datetime of the bike departure from origin

    Returns
    -------
    list of Shapely polygons (EPSG:32633) or None if r5py fails
    """
    import r5py
    import datetime

    if not station_rows:
        return []

    origins_gdf = gpd.GeoDataFrame(
        {
            "id":                [r["name"] for r in station_rows],
            "remaining_walk_sec": [r["remaining_walk_sec"] for r in station_rows],
        },
        geometry=gpd.points_from_xy(
            [r["lon"] for r in station_rows],
            [r["lat"] for r in station_rows],
        ),
        crs="EPSG:4326",
    )

    polygons_proj = []

    for _, row in origins_gdf.iterrows():
        remaining_sec = row["remaining_walk_sec"]
        remaining_min = remaining_sec / 60

        origin_single = gpd.GeoDataFrame(
            {"id": [row["id"]]},
            geometry=[row.geometry],
            crs="EPSG:4326",
        )

        try:
            # Isochrone: walk from dock station for remaining_walk_sec
            # Uses actual OSM street network — no circle approximation
            isochrone = r5py.Isochrones(
                r5py_network,
                origins=origin_single,
                departure=depart_datetime,
                departure_time_window=datetime.timedelta(minutes=1),
                transport_modes=[r5py.TransportMode.WALK],
                isochrones=[datetime.timedelta(seconds=remaining_sec)],
                speed_walking=WALK_SPEED_KMH,
            )

            if isochrone.empty:
                continue

            # Reproject to EPSG:32633 for consistent metric operations
            poly_proj = isochrone.to_crs(epsg=32633).geometry.unary_union
            if poly_proj and not poly_proj.is_empty:
                polygons_proj.append(poly_proj)

        except Exception as e:
            print(f"     [!] r5py walk failed for '{row['id']}': {e}")
            continue

    return polygons_proj


# ==============================================================================
# ISOCHRONE GENERATORS — OSMnx FALLBACK
# ==============================================================================

def _project_graph_edges(G_walk):
    """Project all walk edges to EPSG:32633 once; reused by every station computation."""
    return ox.graph_to_gdfs(G_walk, nodes=False, edges=True).to_crs(epsg=32633)


def _isochrone_polygon_from_reachable(in_range, projected_edges_gdf):
    """
    Boundary-edge isochrone approach (Baum et al.):
      - Internal edges (both u and v in range) -> fill the reachable area
      - Boundary edges (exactly one endpoint in range) -> close the outer shell
    Used as fallback when r5py is unavailable.
    """
    u_vals = projected_edges_gdf.index.get_level_values("u")
    v_vals = projected_edges_gdf.index.get_level_values("v")

    u_in = np.isin(u_vals, list(in_range))
    v_in = np.isin(v_vals, list(in_range))

    internal_edges = projected_edges_gdf[u_in & v_in]
    boundary_edges = projected_edges_gdf[u_in ^ v_in]

    polys = []
    if not internal_edges.empty:
        polys.append(internal_edges.geometry.buffer(WALK_BUFFER_M).unary_union)
    if not boundary_edges.empty:
        polys.append(boundary_edges.geometry.buffer(BOUNDARY_BUFFER_M).unary_union)

    return unary_union(polys) if polys else None


def _osmnx_walk_polygons(G_walk, projected_walk_edges, station_rows):
    """
    Compute walk polygons using OSMnx boundary-edge method.
    Used as fallback when r5py is unavailable.
    Includes corrected dominance pruning.
    """
    schedulable_sorted = sorted(
        station_rows, key=lambda r: r["remaining_walk_sec"], reverse=True)
    walk_node_to_remaining = {
        int(r["walk_node"]): float(r["remaining_walk_sec"])
        for r in schedulable_sorted
    }
    skipped  = set()
    computed = 0
    polygons = []

    for r in schedulable_sorted:
        walk_node = int(r["walk_node"])
        remaining = float(r["remaining_walk_sec"])

        if walk_node in skipped:
            continue

        try:
            reachable = nx.single_source_dijkstra_path_length(
                G_walk, walk_node, cutoff=remaining, weight="travel_time")
        except nx.NetworkXError:
            continue

        in_range = set(reachable.keys())
        computed += 1

        for other_node, other_remaining in walk_node_to_remaining.items():
            if other_node == walk_node or other_node in skipped:
                continue
            travel_to_other = reachable.get(other_node)
            if travel_to_other is not None:
                if (remaining - travel_to_other) >= other_remaining:
                    skipped.add(other_node)

        poly = _isochrone_polygon_from_reachable(in_range, projected_walk_edges)
        if poly is not None:
            polygons.append(poly)

    print(f"     Polygons computed: {computed} | Dominated & skipped: {len(skipped)}")
    return polygons


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def generate_own_bike_polygon(G_bike_undirected, predicted_times_dict,
                               total_mins, output_filename):
    print(f"\n  -> Generating OWN BIKE | {total_mins} mins...")
    total_sec = total_mins * 60
    valid_node_ids = [n for n, t in predicted_times_dict.items() if t <= total_sec]
    sub_graph = G_bike_undirected.subgraph(valid_node_ids)
    edges = ox.graph_to_gdfs(sub_graph, nodes=False, edges=True)
    if not edges.empty:
        proj = edges.to_crs(epsg=32633).geometry.buffer(BIKE_BUFFER_M).unary_union
        poly = proj.buffer(DISSOLVE_BRIDGE_M).buffer(-DISSOLVE_BRIDGE_M)
        final_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:32633").to_crs(epsg=4326)
        final_gdf.to_file(output_filename, driver="GeoJSON")
        print(f"     Saved: {os.path.relpath(output_filename)}")
        return final_gdf.iloc[0].geometry
    return None


def _dissolve_and_clip(polygons_proj, own_bike_poly_wgs84, station_envelope=None):
    """
    Union all blobs, bridge small gaps, reproject to WGS-84,
    clip to own-bike boundary, then clip to station envelope.
    """
    raw_union = gpd.GeoSeries(polygons_proj, crs="EPSG:32633").unary_union
    dissolved = raw_union.buffer(DISSOLVE_BRIDGE_M).buffer(-DISSOLVE_BRIDGE_M)
    final_gdf = gpd.GeoDataFrame(geometry=[dissolved], crs="EPSG:32633").to_crs(epsg=4326)
    final_gdf.geometry = final_gdf.geometry.intersection(own_bike_poly_wgs84)
    if station_envelope is not None:
        final_gdf.geometry = final_gdf.geometry.intersection(station_envelope)
    return final_gdf


def _station_envelope(schedulable_gdf):
    """
    Per-station walk budget circles unioned together.
    Each station buffered by remaining_walk_sec * WALK_SPEED_MS metres.
    """
    projected = schedulable_gdf.to_crs(epsg=32633).copy()
    radii = schedulable_gdf["remaining_walk_sec"].values * WALK_SPEED_MS
    per_station_buffers = projected.geometry.buffer(radii)
    envelope_proj = per_station_buffers.unary_union
    return gpd.GeoDataFrame(
        geometry=[envelope_proj], crs="EPSG:32633"
    ).to_crs(epsg=4326).iloc[0].geometry


def generate_shared_bike_isochrone(G_bike_undirected, G_walk, own_bike_poly,
                                    predicted_times_dict, stations,
                                    origin_lat, origin_lon, total_mins,
                                    output_filename,
                                    projected_walk_edges=None,
                                    r5py_network=None,
                                    depart_datetime=None):
    print(f"\n  -> Generating SHARED BIKE | {total_mins} mins...")
    total_sec = total_mins * 60
    polygons_proj = []
    reached_stations = []

    # Determine walk method
    use_r5 = USE_R5PY_WALK and r5py_network is not None
    print(f"     Walk method: {'r5py (real streets)' if use_r5 else 'OSMnx boundary-edge (fallback)'}")

    # 1. Pre-project walk edges if using OSMnx fallback
    if not use_r5 and projected_walk_edges is None:
        projected_walk_edges = _project_graph_edges(G_walk)

    # 2. Origin walk polygon
    if use_r5:
        # r5py walk from origin for full time budget
        import r5py, datetime as dt
        origin_gdf = gpd.GeoDataFrame(
            {"id": ["origin"]},
            geometry=gpd.points_from_xy([origin_lon], [origin_lat]),
            crs="EPSG:4326",
        )
        try:
            iso = r5py.Isochrones(
                r5py_network,
                origins=origin_gdf,
                departure=depart_datetime,
                departure_time_window=dt.timedelta(minutes=1),
                transport_modes=[r5py.TransportMode.WALK],
                isochrones=[dt.timedelta(seconds=total_sec)],
                speed_walking=WALK_SPEED_KMH,  # ← was walk_speed
            )
            if not iso.empty:
                polygons_proj.append(iso.to_crs(epsg=32633).geometry.unary_union)
        except Exception as e:
            print(f"     [!] r5py origin walk failed: {e}")
    else:
        origin_walk_node = ox.distance.nearest_nodes(G_walk, X=origin_lon, Y=origin_lat)
        try:
            origin_reachable = nx.single_source_dijkstra_path_length(
                G_walk, origin_walk_node, cutoff=total_sec, weight="travel_time")
            origin_poly = _isochrone_polygon_from_reachable(
                set(origin_reachable.keys()), projected_walk_edges)
            if origin_poly is not None:
                polygons_proj.append(origin_poly)
        except nx.NetworkXError:
            pass

    # 3. Filter stations inside own-bike polygon
    stations_df = pd.DataFrame(stations)
    stations_gdf = gpd.GeoDataFrame(stations_df,
        geometry=gpd.points_from_xy(stations_df.lon, stations_df.lat), crs="EPSG:4326")
    valid_stations_gdf = stations_gdf[stations_gdf.geometry.within(own_bike_poly)].copy()
    print(f"     Filtered to {len(valid_stations_gdf)} stations inside the bike zone.")

    if valid_stations_gdf.empty:
        if polygons_proj:
            _dissolve_and_clip(polygons_proj, own_bike_poly).to_file(
                output_filename, driver="GeoJSON")
        return

    # 4. Nearest nodes (OSMnx — always needed for bike node lookup)
    valid_stations_gdf["bike_node"] = ox.distance.nearest_nodes(
        G_bike_undirected,
        X=valid_stations_gdf["lon"].values,
        Y=valid_stations_gdf["lat"].values)

    if not use_r5:
        valid_stations_gdf["walk_node"] = ox.distance.nearest_nodes(
            G_walk,
            X=valid_stations_gdf["lon"].values,
            Y=valid_stations_gdf["lat"].values)

    # 5. Remaining walk budget
    def _remaining(row):
        bike_sec = predicted_times_dict.get(row["bike_node"], float("inf"))
        return total_sec - (bike_sec + 20)

    valid_stations_gdf["remaining_walk_sec"] = valid_stations_gdf.apply(_remaining, axis=1)
    schedulable = valid_stations_gdf[
        valid_stations_gdf["remaining_walk_sec"] >= MIN_WALK_SECONDS].copy()
    print(f"     {len(schedulable)} stations have >= {MIN_WALK_SECONDS}s remaining walk time.")

    for _, s in schedulable.iterrows():
        reached_stations.append({
            "uid":   s.get("uid", "unknown"),
            "name":  s.get("name", "Nextbike Station"),
            "geometry": s["geometry"],
            "bike_travel_time_min": round((total_sec - s["remaining_walk_sec"]) / 60, 1),
            "remaining_walk_min":   round(s["remaining_walk_sec"] / 60, 1),
        })

    # 6. Station envelope (per-station walk budget circles)
    if not schedulable.empty:
        envelope = _station_envelope(schedulable)
        print(f"     Station envelope built from {len(schedulable)} stations "
              f"with individual walk budgets.")
    else:
        envelope = None

    # 7. Walk polygons from dock stations
    print(f"     Computing dock station walk polygons...")
    if use_r5:
        # r5py: real street-following walk polygons from each schedulable station
        station_rows = [
            {
                "name":               row["name"],
                "lat":                row["lat"],
                "lon":                row["lon"],
                "remaining_walk_sec": row["remaining_walk_sec"],
            }
            for _, row in schedulable.iterrows()
        ]
        r5_polys = _r5py_walk_polygons(
            r5py_network, station_rows, total_sec, depart_datetime)
        polygons_proj.extend(r5_polys)
        print(f"     r5py walk polygons computed: {len(r5_polys)}")
    else:
        # OSMnx fallback with dominance pruning
        station_rows = [
            {
                "walk_node":          int(row["walk_node"]),
                "remaining_walk_sec": float(row["remaining_walk_sec"]),
            }
            for _, row in schedulable.iterrows()
        ]
        osm_polys = _osmnx_walk_polygons(G_walk, projected_walk_edges, station_rows)
        polygons_proj.extend(osm_polys)

    # 8. Dissolve + clip to own-bike boundary + clip to station envelope
    if polygons_proj:
        final_gdf = _dissolve_and_clip(polygons_proj, own_bike_poly, station_envelope=envelope)
        final_gdf.to_file(output_filename, driver="GeoJSON")
        print(f"     Saved: {os.path.relpath(output_filename)}")

    if reached_stations:
        stations_out_gdf = gpd.GeoDataFrame(reached_stations, crs="EPSG:4326")
        stations_filename = output_filename.replace("isochrone_", "stations_")
        stations_out_gdf.to_file(stations_filename, driver="GeoJSON")
        print(f"     Saved: {os.path.relpath(stations_filename)}")


# ==============================================================================
# MASTER EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import datetime

    # ── CONFIGURATION — edit only this section ─────────────────────────────────

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FILE_NAME = os.path.join(REPO_ROOT, "data", "nextbike_data_VSB_Vaclavik.xlsx")

    CACHE_GRAPHS = os.path.join(REPO_ROOT, "cache", "graphs")
    CACHE_ELEV   = os.path.join(REPO_ROOT, "cache", "elevations")
    OSM_DIR      = os.path.join(REPO_ROOT, "data")
    OUTPUT_DIR   = os.path.join(REPO_ROOT, "results", "spatial_analysis")
    for d in [CACHE_GRAPHS, CACHE_ELEV, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

    # Departure datetime — used by r5py for walk routing.
    # Date does not affect walk-only routing but is required by the API.
    DEPART_DATETIME = datetime.datetime(2025, 4, 1, 8, 0)

    CITY_SHEET_MAPPING = {
        # "Ostrava":           "Vypujcky_Ostrava",
        # "Ostrava hlavni":    "Vypujcky_Ostrava",
        # "Brno":              "Vypujcky_Brno",
        # "Prerov":            "Vypujcky_Prerov",
        "Valasske Mezirici": "Vypujcky_ValMez",
    }

    # Values must match Nextbike API city names EXACTLY (with diacritics)
    CITY_API_NAMES = {
        # "Ostrava":           "Ostrava",
        # "Ostrava hlavni":    "Ostrava hlavní",
        # "Brno":              "Brno",
        # "Prerov":            "Přerov",
        "Valasske Mezirici": ["Valašské Meziříčí", "Poličná", "Krhová", "Zašová"],
    }

    # Station names must match Nextbike API EXACTLY (with diacritics)
    CITY_ORIGINS = {
        # "Ostrava":        ["SV-Svinov nádraží *(navíc 15min na odjezd)"],
        # "Ostrava hlavni": ["MOAP-Hlavní nádraží"],
        # "Brno":           ["Hlavní nádraží - Hlavní vstup", "Hlavní nádraží - pošta"],
        # "Prerov":         ["Nádraží"],
        "Valasske Mezirici": ["Vlakové nádraží Valašské Meziříčí (nové umístění)"],
    }

    # OSM .pbf file per city — place in data/osm/
    # All Czech cities can share one file: czech-republic-latest.osm.pbf
    # Download: https://download.geofabrik.de/europe/czech-republic.html
    CITY_OSM_FILE = {
        "Valasske Mezirici": "czech-republic-260303.osm.pbf",
        # "Ostrava":           "czech-republic-260303.osm.pbf",
        # "Brno":              "czech-republic-260303.osm.pbf",
        # "Prerov":            "czech-republic-260303.osm.pbf",
    }

    # Per-city duration filter (seconds).
    CITY_DURATION_FILTER = {
        "Valasske Mezirici": (60,  1800),
        "Prerov":            (60,  1800),
        "Ostrava":           (120, 3600),
        "Ostrava hlavni":    (120, 3600),
        "Brno":              (120, 4500),
    }

    TIME_BUDGETS = [15, 30]   # minutes

    # ── Verify data file ───────────────────────────────────────────────────────
    if not os.path.exists(FILE_NAME):
        raise FileNotFoundError(
            f"Data file not found: {FILE_NAME}\n"
            f"Make sure 'data/nextbike_data_VSB_Vaclavik.xlsx' exists in: {REPO_ROOT}"
        )

    # ── PER-CITY LOOP ──────────────────────────────────────────────────────────
    for MAPPING_CITY_NAME, ORIGIN_STATION_NAMES in CITY_ORIGINS.items():

        TRAINING_SHEET = CITY_SHEET_MAPPING.get(MAPPING_CITY_NAME)
        if not TRAINING_SHEET:
            print(f"  [!] '{MAPPING_CITY_NAME}' missing from CITY_SHEET_MAPPING, skipping.")
            continue

        # ── PHASE 1: ML model ─────────────────────────────────────────────────
        dur_min, dur_max = CITY_DURATION_FILTER.get(MAPPING_CITY_NAME, (60, 1800))
        ai_model = get_or_train_local_model(
            FILE_NAME, TRAINING_SHEET, MAPPING_CITY_NAME,
            sample_size=2000,
            duration_min=dur_min,
            duration_max=dur_max,
        )

        if ai_model is None:
            raise RuntimeError(
                f"Model training failed for '{MAPPING_CITY_NAME}'.\n"
                f"  Check 1 - file exists:  {FILE_NAME}\n"
                f"  Check 2 - sheet exists: '{TRAINING_SHEET}'\n"
                f"  Check 3 - columns: start_lat, start_lng, end_lat, end_lng, duration"
            )

        # ── PHASE 2: Stations + graphs ────────────────────────────────────────
        api_names = CITY_API_NAMES.get(MAPPING_CITY_NAME, MAPPING_CITY_NAME)
        stations  = get_official_nextbike_stations(api_names)
        if not stations:
            print(f"  [!] No stations returned for {MAPPING_CITY_NAME}, skipping.")
            continue

        first_lat, first_lon = resolve_origin_from_station_name(
            stations, ORIGIN_STATION_NAMES[0])

        city_key        = MAPPING_CITY_NAME.lower().replace(" ", "_")
        bike_graph_file = os.path.join(CACHE_GRAPHS, f"{city_key}_bike_G.graphml")
        walk_graph_file = os.path.join(CACHE_GRAPHS, f"{city_key}_walk_G.graphml")

        print(f"\n--- PHASE 2: GRAPHS FOR {MAPPING_CITY_NAME.upper()} ---")
        if os.path.exists(bike_graph_file):
            print(f"  -> Loading cached bike graph.")
            G_bike = ox.load_graphml(bike_graph_file)
        else:
            print(f"  -> Downloading bike graph for {MAPPING_CITY_NAME}...")
            urban_bike_filter = (
                '["highway"]["area"!~"yes"]["access"!~"private"]'
                '["highway"!~"motorway|motorway_link|trunk|trunk_link|steps"]'
                '["surface"!~"dirt|sand|grass|mud"]'
            )
            G_bike = ox.graph_from_point(
                (first_lat, first_lon), dist=12000, custom_filter=urban_bike_filter)
            ox.save_graphml(G_bike, bike_graph_file)
            print(f"  -> Saved: cache/graphs/{city_key}_bike_G.graphml")

        if os.path.exists(walk_graph_file):
            print(f"  -> Loading cached walk graph.")
            G_walk = ox.load_graphml(walk_graph_file)
        else:
            print(f"  -> Downloading walk graph for {MAPPING_CITY_NAME}...")
            G_walk = ox.graph_from_point(
                (first_lat, first_lon), dist=12000, network_type="walk")
            U_walk = G_walk.to_undirected()
            largest_cc = max(nx.connected_components(U_walk), key=len)
            G_walk = G_walk.subgraph(largest_cc).copy()
            G_walk = ox.add_edge_speeds(G_walk, fallback=5.0)
            G_walk = ox.add_edge_travel_times(G_walk)
            ox.save_graphml(G_walk, walk_graph_file)
            print(f"  -> Saved: cache/graphs/{city_key}_walk_G.graphml")

        G_bike_undirected = ox.convert.to_undirected(G_bike)

        # ── PHASE 3: r5py walk network (if enabled) ───────────────────────────
        r5py_network = None
        if USE_R5PY_WALK:
            osm_file = CITY_OSM_FILE.get(MAPPING_CITY_NAME)
            if osm_file:
                osm_path = os.path.join(OSM_DIR, osm_file)
                if os.path.exists(osm_path):
                    print(f"\n--- PHASE 3: R5PY WALK NETWORK ---")
                    r5py_network = _build_r5py_walk_network(osm_path)
                else:
                    print(f"\n--- PHASE 3: R5PY WALK NETWORK ---")
                    print(f"  [!] OSM file not found: {osm_path}")
                    print(f"      Download from https://download.geofabrik.de/europe/czech-republic.html")
                    print(f"      Falling back to OSMnx walk method.")
            else:
                print(f"\n--- PHASE 3: No OSM file configured for {MAPPING_CITY_NAME} ---")
                print(f"      Add to CITY_OSM_FILE to enable r5py walk routing.")

        # ── PHASE 4: Master time dictionary ───────────────────────────────────
        print(f"\n--- PHASE 4: MASTER TIME DICTIONARY FOR {MAPPING_CITY_NAME.upper()} ---")
        origin_bike_node = ox.distance.nearest_nodes(
            G_bike_undirected, X=first_lon, Y=first_lat)
        distances_dict   = nx.single_source_dijkstra_path_length(
            G_bike_undirected, origin_bike_node, weight="length")
        reachable_nodes  = {n: d for n, d in distances_dict.items() if d <= 15000}

        coords_to_fetch  = [(G_bike_undirected.nodes[n]["y"],
                             G_bike_undirected.nodes[n]["x"]) for n in reachable_nodes]
        coords_to_fetch.insert(0, (first_lat, first_lon))
        coords_to_fetch += [(s["lat"], s["lon"]) for s in stations]
        coords_to_fetch  = list(set(coords_to_fetch))

        network_elev_cache = os.path.join(CACHE_ELEV, f"{city_key}_network_elevations.json")
        elev_dict   = get_elevations_cached(coords_to_fetch, cache_file=network_elev_cache)
        origin_elev = elev_dict.get(f"{first_lat},{first_lon}", 0.0)

        nodes_data = [
            [n, dist,
             elev_dict.get(
                 f"{G_bike_undirected.nodes[n]['y']},{G_bike_undirected.nodes[n]['x']}",
                 0.0) - origin_elev]
            for n, dist in reachable_nodes.items()
        ]
        node_features_df = pd.DataFrame(
            nodes_data, columns=["node_id", "network_distance_m", "elevation_delta_m"]
        ).set_index("node_id")

        node_features_df["straight_line_m"] = node_features_df.apply(
            lambda r: _haversine(
                first_lat, first_lon,
                G_bike_undirected.nodes[r.name]["y"],
                G_bike_undirected.nodes[r.name]["x"]), axis=1)
        node_features_df["detour_ratio"] = (
            node_features_df["network_distance_m"] /
            node_features_df["straight_line_m"].clip(lower=1)
        )

        print("  -> Predicting travel times for all intersections...")
        features = ["network_distance_m", "elevation_delta_m", "straight_line_m", "detour_ratio"]
        raw_predictions = ai_model.predict(node_features_df[features])

        physics_floor       = node_features_df["network_distance_m"].values / MAX_CYCLING_SPEED_MS
        floored_predictions = np.maximum(raw_predictions, physics_floor)
        n_floored = int((floored_predictions > raw_predictions).sum())
        if n_floored > 0:
            print(f"  -> Physics floor applied to {n_floored} nodes.")

        predicted_times_dict = dict(zip(node_features_df.index, floored_predictions))

        # ── PHASE 5: Pre-project walk edges (OSMnx fallback only) ─────────────
        projected_walk_edges = None
        if not USE_R5PY_WALK or r5py_network is None:
            print(f"\n--- PHASE 5: PRE-PROJECTING WALK GRAPH ---")
            projected_walk_edges = _project_graph_edges(G_walk)
            print(f"  -> {len(projected_walk_edges)} walk edges ready.")

        # ── PHASE 6: Isochrones ───────────────────────────────────────────────
        print(f"\n--- PHASE 6: ISOCHRONES FOR {MAPPING_CITY_NAME.upper()} "
              f"| {TIME_BUDGETS} MIN ---")
        for ORIGIN_STATION_NAME in ORIGIN_STATION_NAMES:
            ORIGIN_LAT, ORIGIN_LON = resolve_origin_from_station_name(
                stations, ORIGIN_STATION_NAME)
            station_slug = (ORIGIN_STATION_NAME.lower()
                            .replace(" ", "_").replace("-", "").replace("/", "_"))
            city_prefix  = f"{city_key}_{station_slug}"

            for mins in TIME_BUDGETS:
                poly_own = generate_own_bike_polygon(
                    G_bike_undirected, predicted_times_dict, mins,
                    os.path.join(OUTPUT_DIR,
                                 f"isochrone_{city_prefix}_own_bike_{mins}min.geojson"),
                )
                if poly_own is not None:
                    generate_shared_bike_isochrone(
                        G_bike_undirected, G_walk, poly_own,
                        predicted_times_dict, stations,
                        ORIGIN_LAT, ORIGIN_LON, mins,
                        os.path.join(OUTPUT_DIR,
                                     f"isochrone_{city_prefix}_shared_bike_{mins}min.geojson"),
                        projected_walk_edges=projected_walk_edges,
                        r5py_network=r5py_network,
                        depart_datetime=DEPART_DATETIME,
                    )

    print("\n*** ALL MAPS SUCCESSFULLY COMPLETED! ***")