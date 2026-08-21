import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datetime
import warnings
import requests
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import polygonize, unary_union

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM_DIR = os.path.join(REPO_ROOT, "data", "osm")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "spatial_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Departure configuration ───────────────────────────────────────────────────
# Date doesn't strictly matter for walk routing, but r5py requires one.
DEPARTURE_DATETIME = datetime.datetime(2026, 5, 25, 8, 0)
TIME_BUDGETS = [15, 30]  # minutes
WALK_SPEED_KMH = 5.0  # km/h

# ── City configuration ────────────────────────────────────────────────────────
# Configure ONLY the cities lacking GTFS data for walk analysis
CITY_CONFIG = {
    "Přerov": {
        "osm_file": "czech-republic-260303.osm.pbf",
        "api_names": "Přerov",
        "origins": ["Nádraží"],
    },
    "ValMez": {
        "osm_file": "czech-republic-260303.osm.pbf",
        "api_names": ["Valašské Meziříčí", "Poličná", "Krhová", "Zašová"],
        "origins": ["Vlakové nádraží Valašské Meziříčí (nové umístění)"],
    },
}


# ==============================================================================
# GEOMETRY & API RESOLVERS
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

    for s in stations:
        if s["name"].strip().lower() == station_name.strip().lower():
            print(f"  -> Origin resolved: '{s['name']}' -> ({s['lat']}, {s['lon']})")
            return s["lat"], s["lon"]
    raise ValueError(f"Station '{station_name}' not found.")


def ensure_solid_polygon(geom):
    """
    Converts r5py LineStrings into a solid Polygon by applying a tiny buffer
    to connect the street network, unioning it, and then extracting
    only the exterior boundary to remove all internal holes.
    """
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type in ['LineString', 'MultiLineString']:
        gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs="EPSG:4326")
        gdf_metric = gdf.to_crs(epsg=32633)
        buffered_metric = gdf_metric.geometry.buffer(50).unary_union
        geom = gpd.GeoSeries([buffered_metric], crs="EPSG:32633").to_crs(epsg=4326).iloc[0]

    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior)
    elif geom.geom_type == 'MultiPolygon':
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
        raise ImportError("r5py is not installed.")

    for city_name, cfg in CITY_CONFIG.items():
        print(f"\n{'=' * 60}")
        print(f"  PROCESSING WALK ISOCHRONES: {city_name.upper()}")
        print(f"{'=' * 60}")

        osm_path = os.path.join(OSM_DIR, cfg["osm_file"])
        if not os.path.exists(osm_path):
            print(f"  [!] Missing OSM file: {osm_path} — skipping.")
            continue

        city_key = city_name.lower().replace(" ", "_")

        # ── PHASE 1: Resolve origin ───────────────────────────────────────────
        origin_coords = []
        for station_name in cfg["origins"]:
            try:
                lat, lon = resolve_nextbike_origin(cfg["api_names"], station_name)
                origin_coords.append((station_name, lat, lon))
            except Exception as e:
                print(f"  [!] Could not resolve '{station_name}': {e}")

        if not origin_coords:
            continue

        # ── PHASE 2: Build walk-only transport network ────────────────────────
        print(f"\n--- PHASE 2: BUILDING WALK-ONLY NETWORK ---")
        # THE TRICK: Passing an empty list [] instead of GTFS paths forces r5py
        # to build a pure pedestrian network from the OSM file!
        transport_network = r5py.TransportNetwork(osm_path, [])

        city_output_dir = os.path.join(OUTPUT_DIR, city_key)
        os.makedirs(city_output_dir, exist_ok=True)

        print(f"\n--- PHASE 3: GENERATING WALK ISOCHRONES | {TIME_BUDGETS} min ---")

        for station_name, origin_lat, origin_lon in origin_coords:
            origin_gdf = gpd.GeoDataFrame(
                {"id": [station_name]},
                geometry=gpd.points_from_xy([origin_lon], [origin_lat]),
                crs="EPSG:4326",
            )

            # Route using ONLY the WALK transport mode
            isochrones = r5py.Isochrones(
                transport_network,
                origins=origin_gdf,
                departure=DEPARTURE_DATETIME,
                transport_modes=[r5py.TransportMode.WALK],  # Walk only!
                isochrones=[datetime.timedelta(minutes=m) for m in TIME_BUDGETS],
                speed_walking=WALK_SPEED_KMH,
            )

            for mins in TIME_BUDGETS:
                budget_td = datetime.timedelta(minutes=mins)
                layer = isochrones[isochrones["travel_time"] == budget_td].copy()

                if layer.empty:
                    continue

                layer = layer.drop(columns=["travel_time"])

                # --- Fill holes to create solid polygons (matching PT script) ---
                layer["geometry"] = layer["geometry"].apply(ensure_solid_polygon)

                out_file = os.path.join(city_output_dir, f"isochrone_walk_{mins}min.geojson")
                layer.to_file(out_file, driver="GeoJSON")
                print(f"     Saved Walk Isochrone (Solid): {os.path.relpath(out_file)}")

    print("\n*** ALL WALK ISOCHRONES COMPLETED! ***")