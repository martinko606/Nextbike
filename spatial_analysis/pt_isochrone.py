import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datetime
import warnings
import requests
import re
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import polygonize, unary_union

warnings.filterwarnings("ignore")

# ==============================================================================
# DEPENDENCIES CHECK
# r5py requires:
#   pip install r5py
#   Java JDK 11+ must be installed and on PATH
#
# OSM data (.osm.pbf) for each city must be placed in data/osm/:
#   Brno:    https://download.geofabrik.de/europe/czech-republic.html
#            (download South Moravian Region or full Czech Republic)
#   Ostrava: same source (Moravian-Silesian Region)
#   Rename to brno.osm.pbf / ostrava.osm.pbf
# ==============================================================================

# ==============================================================================
# CONFIGURATION — edit only this section
# ==============================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GTFS_DIR = os.path.join(REPO_ROOT, "data", "gtfs")
OSM_DIR = os.path.join(REPO_ROOT, "data", "osm")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "spatial_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Departure configuration ───────────────────────────────────────────────────
# Must fall within the GTFS validity period (e.g. late August working Tuesday)
DEPARTURE_DATETIME = datetime.datetime(2026, 2, 25, 8, 0)

# How many minutes around the departure time R5 samples to account for
# timetable uncertainty (e.g. 10 = samples departures 08:00–08:10 every minute,
# returns median travel time). Increase for more robust results, decrease for speed.
TIME_WINDOW_MINUTES = 20

TIME_BUDGETS = [15, 30]  # minutes — total budget including transit + walking

# ── Walking parameters ────────────────────────────────────────────────────────
WALK_SPEED_KMH = 5.0  # km/h — used for access, egress and transfers
MAX_WALK_MINUTES = 15  # maximum walking time for any single leg

# ── City configuration ────────────────────────────────────────────────────────
CITY_CONFIG = {
    "Ostrava": {
        "gtfs_file": "ostrava.zip",
        "osm_file": "czech-republic-260303.osm.pbf",
        "api_names": "Ostrava",
        "origins": ["SV-Svinov nádraží *(navíc 15min na odjezd)"],
    },
    "Ostrava_hlavni": {
        "gtfs_file": "ostrava.zip",
        "osm_file": "czech-republic-260303.osm.pbf",
        "api_names": "Ostrava",
        "origins": ["MOAP-Hlavní nádraží"],
    },
    #"Brno": {
    #    "gtfs_file": "brno.zip",
    #    "osm_file": "czech-republic-260303.osm.pbf",
    #    "api_names": "Brno",
    #    "origins": ["Hlavní nádraží - Hlavní vstup"],
    #},
    # Commented out due to corrupted GTFS file (missing coordinates/keys)
    # "Přerov": {
    # "gtfs_file": "prerov.zip",
    # "osm_file": "czech-republic-260303.osm.pbf",
    # "api_names": "Přerov",
    # "origins": ["Nádraží"],
    # },
    # "ValMez": {
    # "gtfs_file": "XXXX.zip",
    # "osm_file": "czech-republic-260303.osm.pbf",
    # "api_names": ["Valašské Meziříčí", "Poličná", "Krhová", "Zašová"],
    # "origins": ["Vlakové nádraží Valašské Meziříčí (nové umístění)"],
    # },
}


# ==============================================================================
# NEXTBIKE ORIGIN RESOLVER
# ==============================================================================

def resolve_nextbike_origin(city_api_names, station_name):
    if isinstance(city_api_names, str):
        city_api_names = [city_api_names]

    url = "https://api.nextbike.net/maps/nextbike-live.json"
    stations = []
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        for country in data.get("countries", []):
            for city in country.get("cities", []):
                if city["name"] in city_api_names:
                    for p in city.get("places", []):
                        if p.get("spot"):
                            stations.append({
                                "name": p["name"],
                                "lat": float(p["lat"]),
                                "lon": float(p["lng"]),
                            })
    except Exception as e:
        raise RuntimeError(f"Nextbike API error: {e}")

    if not stations:
        raise RuntimeError(f"No Nextbike stations found for {city_api_names}.")

    for s in stations:
        if s["name"].strip().lower() == station_name.strip().lower():
            print(f"  -> Origin resolved: '{s['name']}' -> ({s['lat']}, {s['lon']})")
            return s["lat"], s["lon"]

    partial = [s for s in stations if station_name.lower() in s["name"].lower()]
    if partial:
        suggestions = ", ".join(f"'{s['name']}'" for s in partial[:5])
        raise ValueError(f"Station '{station_name}' not found. Did you mean: {suggestions}?")
    raise ValueError(f"Station '{station_name}' not found among {len(stations)} Nextbike stops.")


def ensure_solid_polygon(geom):
    """
    Converts r5py LineStrings into a solid Polygon by applying a tiny buffer
    to connect the street network, unioning it, and then extracting
    only the exterior boundary to remove all internal holes.
    """
    if geom is None or geom.is_empty:
        return geom

    # Step 1: If it is lines (streets), buffer them slightly to make them overlap
    # into a continuous shape (50 meters usually covers city blocks).
    if geom.geom_type in ['LineString', 'MultiLineString']:
        # Project to metric CRS for accurate buffering in meters
        gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs="EPSG:4326")
        gdf_metric = gdf.to_crs(epsg=32633)
        buffered_metric = gdf_metric.geometry.buffer(50).unary_union

        # Bring it back to WGS84
        geom = gpd.GeoSeries([buffered_metric], crs="EPSG:32633").to_crs(epsg=4326).iloc[0]

    # Step 2: Now that it is a polygon (or if it already was), remove all internal holes
    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior)
    elif geom.geom_type == 'MultiPolygon':
        # Drop tiny isolated artifacts and fill holes of the main blobs
        filled_parts = [Polygon(p.exterior) for p in geom.geoms if p.area > 0.00001]
        return MultiPolygon(filled_parts) if len(filled_parts) > 1 else (filled_parts[0] if filled_parts else geom)

    return geom


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    try:
        import r5py
    except ImportError:
        raise ImportError(
            "r5py is not installed. Run:  pip install r5py\n"
            "Also ensure Java JDK 11+ is installed and on your PATH.\n"
            "Verify with:  java -version"
        )

    for city_name, cfg in CITY_CONFIG.items():
        print(f"\n{'=' * 60}")
        print(f"  PROCESSING: {city_name.upper()}")
        print(f"{'=' * 60}")

        # ── File checks ───────────────────────────────────────────────────────
        gtfs_path = os.path.join(GTFS_DIR, cfg["gtfs_file"])
        osm_path = os.path.join(OSM_DIR, cfg["osm_file"])

        missing = []
        if not os.path.exists(gtfs_path):
            missing.append(f"GTFS: {gtfs_path}")
        if not os.path.exists(osm_path):
            missing.append(
                f"OSM PBF: {osm_path}\n"
                f"  Download from https://download.geofabrik.de/europe/czech-republic.html\n"
                f"  and rename to {cfg['osm_file']}"
            )
        if missing:
            print(f"  [!] Missing files for {city_name} — skipping:")
            for m in missing:
                print(f"      {m}")
            continue

        city_key = city_name.lower().replace(" ", "_")

        # ── PHASE 1: Resolve origins ──────────────────────────────────────────
        print(f"\n--- PHASE 1: RESOLVING ORIGINS ---")
        origin_coords = []
        for station_name in cfg["origins"]:
            try:
                lat, lon = resolve_nextbike_origin(cfg["api_names"], station_name)
                origin_coords.append((station_name, lat, lon))
            except Exception as e:
                print(f"  [!] Could not resolve '{station_name}': {e}")

        if not origin_coords:
            print(f"  [!] No valid origins for {city_name}, skipping.")
            continue

        # ── PHASE 2: Build transport network (once per city) ─────────────────
        print(f"\n--- PHASE 2: BUILDING TRANSPORT NETWORK ---")
        print(f"  -> OSM:  {cfg['osm_file']}")
        print(f"  -> GTFS: {cfg['gtfs_file']}")
        print(f"  -> (First run builds network cache — takes 1-3 min, "
              f"subsequent runs load instantly)")

        transport_network = r5py.TransportNetwork(
            osm_path,
            [gtfs_path],
        )

        # ── PHASE 3: Isochrones per origin per time budget ───────────────────
        city_output_dir = os.path.join(OUTPUT_DIR, city_key)
        os.makedirs(city_output_dir, exist_ok=True)

        print(f"\n--- PHASE 3: ISOCHRONES | Depart {DEPARTURE_DATETIME} "
              f"| Budgets {TIME_BUDGETS} min ---")

        for station_name, origin_lat, origin_lon in origin_coords:
            # Build origin as a GeoDataFrame
            origin_gdf = gpd.GeoDataFrame(
                {"id": [station_name]},
                geometry=gpd.points_from_xy([origin_lon], [origin_lat]),
                crs="EPSG:4326",
            )

            isochrones = r5py.Isochrones(
                transport_network,
                origins=origin_gdf,
                departure=DEPARTURE_DATETIME,
                departure_time_window=datetime.timedelta(minutes=TIME_WINDOW_MINUTES),
                transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
                isochrones=[datetime.timedelta(minutes=m) for m in TIME_BUDGETS],
                speed_walking=WALK_SPEED_KMH,
                max_time_walking=datetime.timedelta(minutes=MAX_WALK_MINUTES),
            )

            # Save according to new naming convention
            for mins in TIME_BUDGETS:
                budget_td = datetime.timedelta(minutes=mins)
                layer = isochrones[isochrones["travel_time"] == budget_td].copy()

                if layer.empty:
                    continue

                layer["travel_time_min"] = mins
                layer = layer.drop(columns=["travel_time"])

                # --- NEW: Convert Lines to Solid Polygons ---
                layer["geometry"] = layer["geometry"].apply(ensure_solid_polygon)
                # ------------------------------------------------

                # Standardized filename: isochrone_pt_15min.geojson
                out_file = os.path.join(city_output_dir, f"isochrone_pt_{mins}min.geojson")
                layer.to_file(out_file, driver="GeoJSON")
                print(f"     Saved PT Isochrone (Solid): {os.path.relpath(out_file)}")

    print("\n*** ALL PT ISOCHRONES COMPLETED! ***")