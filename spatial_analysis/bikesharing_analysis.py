"""
================================================================================
  Nextbike × Public Transport — Validation & Synergy Analysis (QGIS Filled)
  Author  : Urban Mobility / GIS Research Pipeline
  Purpose : Thesis results — geometric synergy, model validation, demand-coverage
  CRS     : EPSG:4326 for I/O  |  EPSG:32633 (UTM 33N) for metric calculations
================================================================================
"""

# ── standard library ─────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from pyproj import Transformer

warnings.filterwarnings("ignore", category=FutureWarning)

# ═════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  — adjust these before running
# ═════════════════════════════════════════════════════════════════════════════

# ── Dynamic Project Root Resolution ──────────────────────────────────────────
# This prevents PyCharm "Working Directory" errors by forcing absolute paths.
# It walks up the folder tree until it finds the 'data' directory.
_current_dir = Path(__file__).resolve().parent
while _current_dir.name and not (_current_dir / "data").exists():
    _current_dir = _current_dir.parent
PROJECT_ROOT = _current_dir

# ── cities to process ────────────────────────────────────────────────────────
CITIES = ["brno", "ostrava_svinov", "ostrava_hlavni", "prerov", "valmez"]

# ── root of the GeoJSON isochrone tree ───────────────────────────────────────
ISOCHRONE_ROOT = PROJECT_ROOT / "data" / "analysis data"

# ── Excel trip data ──────────────────────────────────────────────────────────
EXCEL_PATH = PROJECT_ROOT / "nextbike_data_VSB_Vaclavik.xlsx"

# ── Population Grid Data (NEW) ───────────────────────────────────────────────
POPULATION_GRID_PATH = PROJECT_ROOT / "data" / "analysis data" / "grid_obyvatelstvo_sldb2021_20210326.gpkg"
POP_COLUMN = "g131620000" # CZSO SLDB 2021 code for Total Population (Obyvatelstvo celkem)

# ── Study-origin coordinates (WGS-84) ────────────────────────────────────────
STUDY_ORIGINS = {
    "ostrava_svinov": {"lat": 49.8265, "lon": 18.1850, "label": "Svinov Nádraží"},
    "ostrava_hlavni": {"lat": 49.8517, "lon": 18.2686, "label": "Ostrava hlavní nádraží"},
    "brno":           {"lat": 49.1952, "lon": 16.6079, "label": "Brno hlavní nádraží"},
    "prerov":         {"lat": 49.4475, "lon": 17.4442, "label": "Přerov nádraží"},
    "valmez":         {"lat": 49.4758, "lon": 17.9625, "label": "Valašské Meziříčí nádraží"},
}

# Prefix mapping for QGIS batch output files
FILE_PREFIXES = {
    "brno":           "brno",
    "ostrava_svinov": "svinov",
    "ostrava_hlavni": "ov_hl",
    "prerov":         "prerov",
    "valmez":         "valmez"
}

ORIGIN_BUFFER_M = 100

# ── Coordinate reference systems ─────────────────────────────────────────────
CRS_WGS84  = "EPSG:4326"
CRS_METRIC = "EPSG:32633"   # UTM zone 33N — valid across Czech Republic

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR    = PROJECT_ROOT / "results" / "spatial_analysis"
REPORT_CSV    = OUTPUT_DIR / "Validation_Report.csv"
SYNERGY_GJSON = OUTPUT_DIR / "synergy_analysis.geojson"


# ═════════════════════════════════════════════════════════════════════════════
# 1.  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_isochrone(city: str, mode: str, minutes: int) -> gpd.GeoDataFrame | None:
    """
    Load a single GeoJSON isochrone file based on QGIS batch output names.
    """
    prefix = FILE_PREFIXES.get(city, city)
    filename = f"{prefix}_isochrone_{mode}_{minutes}min_filled.geojson"

    # Check if files are in city subfolders or flattened in the root directory
    path_city = ISOCHRONE_ROOT / city / filename
    path_root = ISOCHRONE_ROOT / filename

    target_path = path_city if path_city.exists() else path_root

    if not target_path.exists():
        print(f"  [WARN] Missing isochrone: {filename}")
        return None

    gdf = gpd.read_file(target_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    else:
        gdf = gdf.to_crs(CRS_WGS84)

    print(f"  [OK]   Loaded {filename}  ({len(gdf)} feature(s))")
    return gdf

def load_stations(city: str, minutes: int) -> gpd.GeoDataFrame | None:
    """
    Load the explicitly provided station GeoJSON files for accurate infrastructure counts.
    """
    prefix = FILE_PREFIXES.get(city, city)
    filename = f"{prefix}_stations_shared_bike_{minutes}min.geojson"

    path_city = ISOCHRONE_ROOT / city / filename
    path_root = ISOCHRONE_ROOT / filename

    target_path = path_city if path_city.exists() else path_root

    if not target_path.exists():
        print(f"  [WARN] Missing explicit station file: {filename}")
        return None

    gdf = gpd.read_file(target_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    else:
        gdf = gdf.to_crs(CRS_WGS84)

    print(f"  [OK]   Loaded {filename}  ({len(gdf)} stations)")
    return gdf

def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(CRS_METRIC)


def polygon_area_km2(gdf: gpd.GeoDataFrame) -> float:
    return gdf.geometry.area.sum() / 1e6


def dissolve_to_single(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Adding buffer(0) to fix any residual self-intersections
    unified = unary_union(gdf.geometry).buffer(0)
    return gpd.GeoDataFrame(geometry=[unified], crs=gdf.crs)


def calculate_population_covered(poly_m_gdf: gpd.GeoDataFrame, grid_m_gdf: gpd.GeoDataFrame, pop_col: str) -> float:
    """
    Calculates population using area-weighted interpolation.
    Assumes both inputs are in the metric CRS.
    """
    if poly_m_gdf is None or poly_m_gdf.empty or grid_m_gdf is None or grid_m_gdf.empty:
        return 0.0

    # Ensure single multipolygon to avoid duplicates
    unified_poly = dissolve_to_single(poly_m_gdf)

    # Intersect polygon with the grid
    intersection = gpd.overlay(grid_m_gdf, unified_poly, how='intersection')
    if intersection.empty:
        return 0.0

    # Area-weighted calculation
    intersection['intersect_area'] = intersection.geometry.area
    intersection['pop_ratio'] = intersection['intersect_area'] / intersection['orig_area']
    intersection['pop_covered'] = intersection[pop_col] * intersection['pop_ratio']

    return float(intersection['pop_covered'].sum())


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TASK A — GEOMETRIC SYNERGY & INTERSECTIONS
# ═════════════════════════════════════════════════════════════════════════════

def analyse_geometric_synergy(city: str, minutes: int = 15, pop_grid_m: gpd.GeoDataFrame = None, stations_gdf: gpd.GeoDataFrame = None) -> dict:
    """
    Compute areas of all polygons, intersections, SMI, Transit Gap,
    and demographically covered populations.
    """
    print(f"\n{'─'*60}")
    print(f"  [Task A] Areas & Demographics — {city.upper()} @ {minutes} min")
    print(f"{'─'*60}")

    # Dynamic baseline based on project context (Walking for secondary cities)
    baseline_mode = "walk" if city in ["prerov", "valmez"] else "pt"

    base_gdf = load_isochrone(city, baseline_mode, minutes)
    sb_gdf   = load_isochrone(city, "shared_bike", minutes)
    ob_gdf   = load_isochrone(city, "own_bike", minutes)

    if base_gdf is None or sb_gdf is None:
        print(f"  [SKIP] Cannot compute full synergy — {baseline_mode.upper()} or Shared Bike missing.")
        return {}

    # ── project to metric CRS ────────────────────────────────────────────────
    base_m = to_metric(dissolve_to_single(base_gdf))
    sb_m   = to_metric(dissolve_to_single(sb_gdf))
    ob_m   = to_metric(dissolve_to_single(ob_gdf)) if ob_gdf is not None else None

    # ── individual areas ──────────────────────────────────────────────────────
    area_base = polygon_area_km2(base_m)
    area_sb   = polygon_area_km2(sb_m)
    area_ob   = polygon_area_km2(ob_m) if ob_m is not None else 0.0

    # ── union (Baseline ∪ Shared Bike) ────────────────────────────────────────
    union_geom = unary_union([base_m.geometry.iloc[0], sb_m.geometry.iloc[0]])
    union_gdf  = gpd.GeoDataFrame(geometry=[union_geom], crs=CRS_METRIC)
    area_union = polygon_area_km2(union_gdf)

    # ── Spatial Multiplier Index (SMI) ────────────────────────────────────────
    smi = area_union / area_base if area_base > 0 else np.nan

    # ── Transit Gap (Shared Bike − Baseline) ──────────────────────────────────
    gap_geom = sb_m.geometry.iloc[0].difference(base_m.geometry.iloc[0])
    gap_gdf  = gpd.GeoDataFrame(geometry=[gap_geom], crs=CRS_METRIC)
    area_gap = polygon_area_km2(gap_gdf)

    # ── Intersections ─────────────────────────────────────────────────────────
    overlap_base_sb_geom = sb_m.geometry.iloc[0].intersection(base_m.geometry.iloc[0])
    area_overlap_base_sb = gpd.GeoDataFrame(geometry=[overlap_base_sb_geom], crs=CRS_METRIC).geometry.area.sum() / 1e6

    area_overlap_base_ob = 0.0
    if ob_m is not None:
        overlap_base_ob_geom = ob_m.geometry.iloc[0].intersection(base_m.geometry.iloc[0])
        area_overlap_base_ob = gpd.GeoDataFrame(geometry=[overlap_base_ob_geom], crs=CRS_METRIC).geometry.area.sum() / 1e6

    area_overlap_sb_ob = 0.0
    if ob_m is not None:
        overlap_sb_ob_geom = ob_m.geometry.iloc[0].intersection(sb_m.geometry.iloc[0])
        area_overlap_sb_ob = gpd.GeoDataFrame(geometry=[overlap_sb_ob_geom], crs=CRS_METRIC).geometry.area.sum() / 1e6

    # ── Population Analysis ───────────────────────────────────────────────────
    pop_base = pop_sb = pop_ob = pop_union = pop_gap = 0
    if pop_grid_m is not None and not pop_grid_m.empty:
        pop_base  = calculate_population_covered(base_m, pop_grid_m, POP_COLUMN)
        pop_sb    = calculate_population_covered(sb_m, pop_grid_m, POP_COLUMN)
        pop_union = calculate_population_covered(union_gdf, pop_grid_m, POP_COLUMN)
        pop_gap   = calculate_population_covered(gap_gdf, pop_grid_m, POP_COLUMN)
        if ob_m is not None:
            pop_ob = calculate_population_covered(ob_m, pop_grid_m, POP_COLUMN)

    # ── Station Supply Analysis ───────────────────────────────────────────────
    stations_covered = 0
    if stations_gdf is not None and not stations_gdf.empty:
        # Reproject stations to metric to match the shared bike polygon
        stations_m = stations_gdf.to_crs(CRS_METRIC)
        sb_union_gdf = dissolve_to_single(sb_m)
        # Spatial join to count stations sitting inside the polygon.
        # Using "intersects" ensures points sitting exactly on the border are counted.
        joined_stations = gpd.sjoin(stations_m, sb_union_gdf, how="inner", predicate="intersects")
        stations_covered = len(joined_stations)

    # ── report ────────────────────────────────────────────────────────────────
    print(f"  [Areas & Populations]")
    print(f"  Baseline ({baseline_mode})     : {area_base:.3f} km²  | {int(pop_base):,} people")
    print(f"  Shared Bike ({minutes} min)    : {area_sb:.3f} km²  | {int(pop_sb):,} people  | {stations_covered} stations")
    print(f"  Own Bike ({minutes} min)       : {area_ob:.3f} km²  | {int(pop_ob):,} people")
    print(f"  \n  [Intersections]")
    print(f"  Baseline ∩ Shared Bike     : {area_overlap_base_sb:.3f} km²")
    print(f"  Baseline ∩ Own Bike        : {area_overlap_base_ob:.3f} km²")
    print(f"  Shared Bike ∩ Own Bike     : {area_overlap_sb_ob:.3f} km²")
    print(f"  \n  [Synergy & Gaps]")
    print(f"  Union (Baseline ∪ Bike)    : {area_union:.3f} km²  | {int(pop_union):,} people")
    print(f"  Transit Gap (Bike only)    : {area_gap:.3f} km²  | {int(pop_gap):,} people")
    print(f"  Spatial Multiplier Index   : {smi:.4f}  (×{smi:.2f} Baseline alone)")

    return {
        "city":                   city,
        "minutes":                minutes,
        "baseline_mode":          baseline_mode,
        "area_baseline_km2":      round(area_base, 4),
        "area_shared_bike_km2":   round(area_sb, 4),
        "area_own_bike_km2":      round(area_ob, 4),
        "area_union_km2":         round(area_union, 4),
        "area_transit_gap_km2":   round(area_gap, 4),
        "overlap_base_shared_km2":round(area_overlap_base_sb, 4),
        "overlap_base_own_km2":   round(area_overlap_base_ob, 4),
        "overlap_shared_own_km2": round(area_overlap_sb_ob, 4),
        "pop_baseline":           int(pop_base),
        "pop_shared_bike":        int(pop_sb),
        "pop_own_bike":           int(pop_ob),
        "pop_union":              int(pop_union),
        "pop_transit_gap":        int(pop_gap),
        "stations_in_shared_bike":int(stations_covered),
        "SMI":                    round(smi, 4),
        "_gap_gdf":               gap_gdf.to_crs(CRS_WGS84),
        "_union_gdf":             union_gdf.to_crs(CRS_WGS84),
    }

# ═════════════════════════════════════════════════════════════════════════════
# 3.  TASK B — MODEL VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def load_trip_data(excel_path: Path) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print(f"  [Task B] Loading trip data from: {excel_path}")
    print(f"{'─'*60}")

    if not excel_path.exists():
        print(f"  [WARN] Excel file not found: {excel_path}")
        return pd.DataFrame()

    df = pd.read_excel(excel_path, engine="openpyxl")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"start_lat", "start_lng", "end_lat", "end_lng"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=list(required))
    return df


def build_point_gdf(df: pd.DataFrame, lat_col: str, lon_col: str) -> gpd.GeoDataFrame:
    geom = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df.copy(), geometry=geom, crs=CRS_WGS84)


def filter_trips_from_origin(trips_df: pd.DataFrame, origin_lat: float, origin_lon: float, buffer_m: float = ORIGIN_BUFFER_M) -> pd.DataFrame:
    starts_gdf = build_point_gdf(trips_df, "start_lat", "start_lng")
    starts_m   = starts_gdf.to_crs(CRS_METRIC)

    transformer = Transformer.from_crs(CRS_WGS84, CRS_METRIC, always_xy=True)
    ox, oy      = transformer.transform(origin_lon, origin_lat)
    origin_buf  = Point(ox, oy).buffer(buffer_m)

    mask = starts_m.geometry.within(origin_buf)
    print(f"  Origin buffer ({buffer_m} m) captured {mask.sum()} trip start(s).")
    return trips_df[mask].copy()


def point_in_polygon_accuracy(trips_df: pd.DataFrame, isochrone_gdf: gpd.GeoDataFrame, label: str) -> dict:
    if trips_df.empty:
        return {"label": label, "total": 0, "inside": 0, "accuracy_pct": np.nan}

    ends_gdf = build_point_gdf(trips_df, "end_lat", "end_lng")
    iso_union = dissolve_to_single(isochrone_gdf)
    joined   = gpd.sjoin(ends_gdf, iso_union, how="left", predicate="within")

    n_inside = joined["index_right"].notna().sum()
    n_total  = len(trips_df)
    accuracy = (n_inside / n_total * 100) if n_total > 0 else np.nan

    print(f"  {label}: {n_inside}/{n_total} ends inside → accuracy {accuracy:.1f}%")
    return {
        "label":        label,
        "total_trips":  n_total,
        "ends_inside":  int(n_inside),
        "accuracy_pct": round(accuracy, 2),
    }


def validate_model(city: str, trips_df: pd.DataFrame) -> list[dict]:
    if trips_df.empty:
        return []

    origin = STUDY_ORIGINS.get(city)
    if origin is None:
        return []

    print(f"\n  Study origin: {origin['label']} ({origin['lat']}, {origin['lon']})")
    origin_trips = filter_trips_from_origin(trips_df, origin["lat"], origin["lon"])

    results = []
    for minutes in [15, 30]:
        iso = load_isochrone(city, "own_bike", minutes)
        if iso is not None:
            res = point_in_polygon_accuracy(
                origin_trips, iso,
                label=f"{city} | Own Bike {minutes} min | origin={origin['label']}"
            )
            res.update({"city": city, "minutes": minutes, "origin": origin["label"]})
            results.append(res)
    return results

# ═════════════════════════════════════════════════════════════════════════════
# 4.  TASK C — DEMAND vs. COVERAGE
# ═════════════════════════════════════════════════════════════════════════════

def compute_station_flow(trips_df: pd.DataFrame, lat_col: str = "start_lat", lon_col: str = "start_lng") -> gpd.GeoDataFrame:
    """
    Aggregate trip starts by exact coordinate pair to determine flow.
    """
    if trips_df.empty:
        return gpd.GeoDataFrame()

    # Group by exact coordinates without any rounding/clustering
    flow = trips_df.groupby([lat_col, lon_col]).size().reset_index(name="flow_count")
    geom = [Point(lon, lat) for lon, lat in zip(flow[lon_col], flow[lat_col])]

    flow_gdf = gpd.GeoDataFrame(flow, geometry=geom, crs=CRS_WGS84).rename(
        columns={lat_col: "station_lat", lon_col: "station_lon"}
    )

    print(f"  Aggregated {len(trips_df)} trips → {len(flow_gdf)} exact unique station(s).")
    return flow_gdf


def attach_isochrone_reach(stations_gdf: gpd.GeoDataFrame, shared_bike_iso_gdf: gpd.GeoDataFrame, minutes: int = 15) -> gpd.GeoDataFrame:
    if stations_gdf.empty or shared_bike_iso_gdf is None:
        return stations_gdf

    iso_m = to_metric(dissolve_to_single(shared_bike_iso_gdf))
    iso_area = polygon_area_km2(iso_m)

    iso_wgs = iso_m.to_crs(CRS_WGS84)
    joined  = gpd.sjoin(stations_gdf, iso_wgs, how="left", predicate="within")

    stations_gdf = stations_gdf.copy()
    stations_gdf[f"in_shared_bike_{minutes}min"] = joined["index_right"].notna().values
    stations_gdf[f"iso_area_km2_{minutes}min"] = np.where(stations_gdf[f"in_shared_bike_{minutes}min"], iso_area, 0.0)
    return stations_gdf


def demand_coverage_summary(stations_gdf: gpd.GeoDataFrame, city: str, minutes: int = 15, flow_threshold_pct: float = 25.0, area_threshold_pct: float = 75.0) -> pd.DataFrame:
    if stations_gdf.empty:
        return pd.DataFrame()

    area_col = f"iso_area_km2_{minutes}min"
    if area_col not in stations_gdf.columns:
        return pd.DataFrame()

    df = stations_gdf[["station_lat", "station_lon", "flow_count", area_col]].copy()
    df["city"]    = city
    df["minutes"] = minutes

    low_flow_q  = np.percentile(df["flow_count"], flow_threshold_pct)
    high_area_q = np.percentile(df[area_col],    area_threshold_pct)
    low_area_q  = np.percentile(df[area_col],    100 - area_threshold_pct)
    high_flow_q = np.percentile(df["flow_count"], 100 - flow_threshold_pct)

    df["high_reach_low_flow"] = ((df[area_col] >= high_area_q) & (df["flow_count"] <= low_flow_q))
    df["low_reach_high_flow"] = ((df[area_col] <= low_area_q)  & (df["flow_count"] >= high_flow_q))

    df["mismatch_type"] = "balanced"
    df.loc[df["high_reach_low_flow"], "mismatch_type"] = "HIGH_REACH_LOW_FLOW"
    df.loc[df["low_reach_high_flow"], "mismatch_type"] = "LOW_REACH_HIGH_FLOW"

    print(f"  HIGH reach / LOW flow stations  : {df['high_reach_low_flow'].sum()}")
    print(f"  LOW reach  / HIGH flow stations : {df['low_reach_high_flow'].sum()}")
    return df.drop(columns=["high_reach_low_flow", "low_reach_high_flow"])

# ═════════════════════════════════════════════════════════════════════════════
# 5.  SYNERGY GEOJSON EXPORT
# ═════════════════════════════════════════════════════════════════════════════

def build_synergy_geojson(synergy_results: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for r in synergy_results:
        if not r:
            continue

        # ── Union layer ───────────────────────────────────────────────────────
        union_gdf = r.get("_union_gdf")
        if union_gdf is not None and not union_gdf.empty:
            rows.append({
                "geometry":          union_gdf.geometry.iloc[0],
                "city":              r["city"],
                "minutes":           r["minutes"],
                "layer_type":        f"union_{r['baseline_mode']}_SharedBike",
                "area_km2":          r["area_union_km2"],
                "area_baseline_km2": r["area_baseline_km2"],
                "area_bike_km2":     r["area_shared_bike_km2"],
                "overlap_base_bike_km2": r["overlap_base_shared_km2"],
                "area_gap_km2":      r["area_transit_gap_km2"],
                "SMI":               r["SMI"],
            })

        # ── Transit-Gap layer ─────────────────────────────────────────────────
        gap_gdf = r.get("_gap_gdf")
        if gap_gdf is not None and not gap_gdf.empty:
            rows.append({
                "geometry":     gap_gdf.geometry.iloc[0],
                "city":         r["city"],
                "minutes":      r["minutes"],
                "layer_type":   "transit_gap",
                "area_km2":     r["area_transit_gap_km2"],
                "area_baseline_km2": r["area_baseline_km2"],
                "area_bike_km2":r["area_shared_bike_km2"],
                "overlap_base_bike_km2": r["overlap_base_shared_km2"],
                "area_gap_km2": r["area_transit_gap_km2"],
                "SMI":          r["SMI"],
            })

    return gpd.GeoDataFrame(rows, crs=CRS_WGS84) if rows else gpd.GeoDataFrame()

# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Nextbike × PT/Walk — Validation & Synergy Analysis")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_synergy_results   = []
    all_validation_results= []
    all_demand_rows       = []
    geojson_inputs        = []

    # ── Load and prep population grid ─────────────────────────────────────────
    pop_grid_m = gpd.GeoDataFrame()
    if POPULATION_GRID_PATH.exists():
        print(f"  Loading population grid from {POPULATION_GRID_PATH.name}...")
        pop_grid = gpd.read_file(POPULATION_GRID_PATH)
        pop_grid_m = pop_grid.to_crs(CRS_METRIC)
        # Pre-calculate original area for areal interpolation weighting
        pop_grid_m['orig_area'] = pop_grid_m.geometry.area
        print(f"  [OK] Population grid loaded ({len(pop_grid_m)} cells).")
    else:
        print(f"  [WARN] Population grid {POPULATION_GRID_PATH.name} not found. Demographics will be 0.")

    trips_df = load_trip_data(EXCEL_PATH)

    for city in CITIES:
        print(f"\n{'═'*65}")
        print(f"  CITY: {city.upper()}")
        print(f"{'═'*65}")

        # ── Pre-compute empirical stations for Task C Demand ──────────────────
        # Fixes the string mismatch (Excel says "Ostrava", but city key is "ostrava_svinov")
        if "city" in trips_df.columns:
            if "ostrava" in city:
                city_trips = trips_df[trips_df["city"].str.lower() == "ostrava"]
            elif city == "valmez":
                city_trips = trips_df[trips_df["city"].str.lower().str.contains("val", na=False)]
            else:
                city_trips = trips_df[trips_df["city"].str.lower() == city]
        else:
            city_trips = trips_df

        empirical_stations_gdf = compute_station_flow(city_trips)

        # ── Task A: Areas, Intersections & Demographics ───────────────────────
        for minutes in [15, 30]:
            # Explicitly load the station files you added for the raw dock count
            dock_stations_gdf = load_stations(city, minutes)

            res = analyse_geometric_synergy(city, minutes, pop_grid_m, dock_stations_gdf)
            if res:
                scalar_res = {k: v for k, v in res.items() if not k.startswith("_")}
                all_synergy_results.append(scalar_res)
                geojson_inputs.append(res)

        # ── Task B: Model Validation ──────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  [Task B] Model Validation — {city.upper()}")
        print(f"{'─'*60}")
        val_results = validate_model(city, trips_df)
        all_validation_results.extend(val_results)

        # ── Task C: Demand vs. Coverage ───────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  [Task C] Demand vs Coverage — {city.upper()}")
        print(f"{'─'*60}")

        for minutes in [15, 30]:
            iso_shared = load_isochrone(city, "shared_bike", minutes)
            if iso_shared is not None and not empirical_stations_gdf.empty:
                # Use empirical_stations_gdf here so we keep the flow_count data for Task C
                stations_with_reach = attach_isochrone_reach(empirical_stations_gdf, iso_shared, minutes)
                summary = demand_coverage_summary(stations_with_reach, city, minutes=minutes)
                if not summary.empty:
                    all_demand_rows.append(summary)

    # ═════════════════════════════════════════════════════════════════════════
    # 7.  OUTPUTS
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*65}")
    print("  Writing outputs …")
    print(f"{'═'*65}")

    report_sections = []

    if all_synergy_results:
        synergy_df = pd.DataFrame(all_synergy_results)
        synergy_df.insert(0, "analysis", "geometric_synergy")
        report_sections.append(synergy_df)

    if all_validation_results:
        val_df = pd.DataFrame(all_validation_results)
        val_df.insert(0, "analysis", "model_validation")
        report_sections.append(val_df)

    if all_demand_rows:
        demand_df = pd.concat(all_demand_rows, ignore_index=True)
        demand_df.insert(0, "analysis", "demand_coverage")
        report_sections.append(demand_df)

    if report_sections:
        full_report = pd.concat(report_sections, ignore_index=True, sort=False)
        full_report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
        print(f"  [SAVED] {REPORT_CSV}  ({len(full_report)} rows)")

    synergy_gdf = build_synergy_geojson(geojson_inputs)
    if not synergy_gdf.empty:
        synergy_gdf.to_file(SYNERGY_GJSON, driver="GeoJSON")
        print(f"  [SAVED] {SYNERGY_GJSON}  ({len(synergy_gdf)} feature(s))")

    print(f"\n{'═'*65}")
    print("  SUMMARY TABLE — Geometric Synergy")
    print(f"{'═'*65}")
    if all_synergy_results:
        df = pd.DataFrame(all_synergy_results)
        cols = ["city", "minutes", "baseline_mode", "area_baseline_km2", "area_shared_bike_km2", "area_transit_gap_km2", "SMI", "pop_transit_gap", "stations_in_shared_bike"]
        print(df[cols].to_string(index=False))

    print(f"\n{'═'*65}")
    print("  Done. ✓")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()