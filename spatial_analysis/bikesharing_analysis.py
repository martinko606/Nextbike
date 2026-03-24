"""
================================================================================
  Nextbike × Public Transport — Validation & Synergy Analysis
  Author  : Urban Mobility / GIS Research Pipeline
  Purpose : Thesis results — geometric synergy, model validation, demand-coverage
  CRS     : EPSG:4326 for I/O  |  EPSG:32633 (UTM 33N) for metric calculations
================================================================================

Expected file-tree
------------------
results/
└── spatial_analysis/
    ├── brno/
    │   ├── isochrone_own_bike_15min.geojson
    │   ├── isochrone_own_bike_30min.geojson
    │   ├── isochrone_shared_bike_15min.geojson
    │   ├── isochrone_shared_bike_30min.geojson
    │   ├── isochrone_pt_15min.geojson
    │   └── isochrone_pt_30min.geojson
    └── ostrava/
        └── <same naming convention>

nextbike_data_VSB_Vaclavik.xlsx   ← historical trip data (any reachable path)

Outputs (written next to this script OR into results/spatial_analysis/)
-----------------------------------------------------------------------
Validation_Report.csv
synergy_analysis.geojson
"""

# ── standard library ─────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
from pyproj import Transformer


warnings.filterwarnings("ignore", category=FutureWarning)

# ═════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  — adjust these before running
# ═════════════════════════════════════════════════════════════════════════════

# ── cities to process ────────────────────────────────────────────────────────
CITIES = ["brno", "ostrava"]

# ── root of the GeoJSON isochrone tree ───────────────────────────────────────
ISOCHRONE_ROOT = Path("results/spatial_analysis")

# ── Excel trip data ───────────────────────────────────────────────────────────
EXCEL_PATH = Path("nextbike_data_VSB_Vaclavik.xlsx")

# ── Study-origin coordinates (WGS-84) — Svinov Nádraží as default ────────────
#    Add/modify entries for each city as needed.
STUDY_ORIGINS = {
    "ostrava": {"lat": 49.8265, "lon": 18.1850, "label": "Svinov Nádraží"},
    "brno":    {"lat": 49.1952, "lon": 16.6079, "label": "Brno hlavní nádraží"},
}

# Buffer radius around the study origin used to "snap" trip starts (metres)
ORIGIN_BUFFER_M = 100

# ── Coordinate reference systems ─────────────────────────────────────────────
CRS_WGS84  = "EPSG:4326"
CRS_METRIC = "EPSG:32633"   # UTM zone 33N — valid across Czech Republic

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path("results/spatial_analysis")
REPORT_CSV    = OUTPUT_DIR / "Validation_Report.csv"
SYNERGY_GJSON = OUTPUT_DIR / "synergy_analysis.geojson"


# ═════════════════════════════════════════════════════════════════════════════
# 1.  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_isochrone(city: str, mode: str, minutes: int) -> gpd.GeoDataFrame | None:
    """
    Load a single GeoJSON isochrone file.

    Parameters
    ----------
    city    : 'brno' | 'ostrava'
    mode    : 'own_bike' | 'shared_bike' | 'pt'
    minutes : 15 | 30

    Returns
    -------
    GeoDataFrame in WGS-84, or None if the file is missing.
    """
    fname = ISOCHRONE_ROOT / city / f"isochrone_{mode}_{minutes}min.geojson"
    if not fname.exists():
        print(f"  [WARN] Missing isochrone: {fname}")
        return None

    gdf = gpd.read_file(fname)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    else:
        gdf = gdf.to_crs(CRS_WGS84)

    print(f"  [OK]   Loaded {fname.name}  ({len(gdf)} feature(s))")
    return gdf


def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Re-project a GeoDataFrame to the metric CRS."""
    return gdf.to_crs(CRS_METRIC)


def polygon_area_km2(gdf: gpd.GeoDataFrame) -> float:
    """
    Return the total area (km²) of all geometries in *metric* CRS.
    The caller is responsible for projecting to CRS_METRIC first.
    """
    return gdf.geometry.area.sum() / 1e6


def dissolve_to_single(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Dissolve all features into one unified polygon.
    Works even if the GeoDataFrame has no 'dissolve_key' column.
    """
    unified = unary_union(gdf.geometry)
    return gpd.GeoDataFrame(geometry=[unified], crs=gdf.crs)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TASK A — GEOMETRIC SYNERGY  (PT ∪ Shared Bike)
# ═════════════════════════════════════════════════════════════════════════════

def analyse_geometric_synergy(city: str, minutes: int = 15) -> dict:
    """
    Compute the Spatial Multiplier Index (SMI) and Transit-Gap geometry.

    Returns a dict with scalar results and the gap GeoDataFrame.
    """
    print(f"\n{'─'*60}")
    print(f"  [Task A] Geometric Synergy — {city.upper()} @ {minutes} min")
    print(f"{'─'*60}")

    pt_gdf   = load_isochrone(city, "pt",          minutes)
    bike_gdf = load_isochrone(city, "shared_bike", minutes)

    if pt_gdf is None or bike_gdf is None:
        print("  [SKIP] Cannot compute synergy — isochrone(s) missing.")
        return {}

    # ── project to metric CRS ────────────────────────────────────────────────
    pt_m   = to_metric(dissolve_to_single(pt_gdf))
    bike_m = to_metric(dissolve_to_single(bike_gdf))

    # ── individual areas ──────────────────────────────────────────────────────
    area_pt   = polygon_area_km2(pt_m)
    area_bike = polygon_area_km2(bike_m)

    # ── union (PT ∪ Shared Bike) ──────────────────────────────────────────────
    union_geom  = unary_union([pt_m.geometry.iloc[0], bike_m.geometry.iloc[0]])
    union_gdf   = gpd.GeoDataFrame(geometry=[union_geom], crs=CRS_METRIC)
    area_union  = polygon_area_km2(union_gdf)

    # ── Spatial Multiplier Index (SMI) ────────────────────────────────────────
    #    SMI > 1  → combined reach larger than PT alone
    smi = area_union / area_pt if area_pt > 0 else np.nan

    # ── Transit Gap  =  Shared-Bike reach  MINUS  PT reach ───────────────────
    #    "Areas accessible by bike but NOT covered by PT"
    gap_geom  = bike_m.geometry.iloc[0].difference(pt_m.geometry.iloc[0])
    gap_gdf   = gpd.GeoDataFrame(geometry=[gap_geom], crs=CRS_METRIC)
    area_gap  = polygon_area_km2(gap_gdf)

    # ── Overlap (intersection) ────────────────────────────────────────────────
    overlap_geom = bike_m.geometry.iloc[0].intersection(pt_m.geometry.iloc[0])
    area_overlap = gpd.GeoDataFrame(
        geometry=[overlap_geom], crs=CRS_METRIC
    ).geometry.area.sum() / 1e6

    # ── report ────────────────────────────────────────────────────────────────
    print(f"  Area PT ({minutes} min)        : {area_pt:.3f} km²")
    print(f"  Area Shared Bike ({minutes} min): {area_bike:.3f} km²")
    print(f"  Area Union (PT ∪ Bike)      : {area_union:.3f} km²")
    print(f"  Spatial Multiplier Index    : {smi:.4f}  (×{smi:.2f} PT alone)")
    print(f"  Transit Gap (bike only)     : {area_gap:.3f} km²")
    print(f"  Overlap (PT ∩ Bike)         : {area_overlap:.3f} km²")

    return {
        "city":              city,
        "minutes":           minutes,
        "area_pt_km2":       round(area_pt,      4),
        "area_shared_bike_km2": round(area_bike, 4),
        "area_union_km2":    round(area_union,   4),
        "area_overlap_km2":  round(area_overlap, 4),
        "area_transit_gap_km2": round(area_gap,  4),
        "SMI":               round(smi,           4),
        # store gap GDF for QGIS export (back to WGS-84)
        "_gap_gdf":          gap_gdf.to_crs(CRS_WGS84),
        "_union_gdf":        union_gdf.to_crs(CRS_WGS84),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3.  TASK B — MODEL VALIDATION  (Empirical trips vs. Theoretical isochrones)
# ═════════════════════════════════════════════════════════════════════════════

def load_trip_data(excel_path: Path) -> pd.DataFrame:
    """
    Load and sanitise the historical Nextbike trip Excel file.
    Expected columns: start_lat, start_lng, end_lat, end_lng, duration
    """
    print(f"\n{'─'*60}")
    print(f"  [Task B] Loading trip data from: {excel_path}")
    print(f"{'─'*60}")

    if not excel_path.exists():
        print(f"  [WARN] Excel file not found: {excel_path}")
        return pd.DataFrame()

    df = pd.read_excel(excel_path, engine="openpyxl")

    # ── normalise column names (strip whitespace, lowercase) ─────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"start_lat", "start_lng", "end_lat", "end_lng"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Excel file is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # ── drop rows with null coordinates ──────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=list(required))
    after  = len(df)
    print(f"  Loaded {before} rows, {after} valid after dropping NaN coords.")

    return df


def build_point_gdf(df: pd.DataFrame, lat_col: str, lon_col: str) -> gpd.GeoDataFrame:
    """Convert lat/lon columns into a Point GeoDataFrame (WGS-84)."""
    geom = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df.copy(), geometry=geom, crs=CRS_WGS84)


def filter_trips_from_origin(
    trips_df: pd.DataFrame,
    origin_lat: float,
    origin_lon: float,
    buffer_m: float = ORIGIN_BUFFER_M,
) -> pd.DataFrame:
    """
    Return only trips whose start point lies within `buffer_m` metres of the
    study origin (using the metric CRS for accurate distance).
    """
    # Build metric GDF of all start points
    starts_gdf = build_point_gdf(trips_df, "start_lat", "start_lng")
    starts_m   = starts_gdf.to_crs(CRS_METRIC)

    # Build origin buffer in metric CRS
    transformer = Transformer.from_crs(CRS_WGS84, CRS_METRIC, always_xy=True)
    ox, oy      = transformer.transform(origin_lon, origin_lat)
    origin_buf  = Point(ox, oy).buffer(buffer_m)

    # Filter
    mask = starts_m.geometry.within(origin_buf)
    n    = mask.sum()
    print(f"  Origin buffer ({buffer_m} m) captured {n} trip start(s).")
    return trips_df[mask].copy()


def point_in_polygon_accuracy(
    trips_df: pd.DataFrame,
    isochrone_gdf: gpd.GeoDataFrame,
    label: str,
) -> dict:
    """
    Point-in-polygon test: what fraction of trip end-points fall inside the
    given isochrone polygon?

    Returns a dict with counts and the accuracy rate.
    """
    if trips_df.empty:
        return {"label": label, "total": 0, "inside": 0, "accuracy_pct": np.nan}

    ends_gdf = build_point_gdf(trips_df, "end_lat", "end_lng")

    # Dissolve isochrone to one polygon, keep in WGS-84
    iso_union = dissolve_to_single(isochrone_gdf)

    # Spatial join — keep only points that fall inside the polygon
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
    """
    Full model validation for one city:
      1. Filter trips from the study origin.
      2. Test against 15-min and 30-min Own Bike isochrones.
    """
    if trips_df.empty:
        return []

    origin = STUDY_ORIGINS.get(city)
    if origin is None:
        print(f"  [WARN] No study origin defined for {city}.")
        return []

    print(f"\n  Study origin: {origin['label']}  "
          f"({origin['lat']}, {origin['lon']})")

    # ── filter trips from origin ──────────────────────────────────────────────
    origin_trips = filter_trips_from_origin(
        trips_df, origin["lat"], origin["lon"]
    )

    results = []
    for minutes in [15, 30]:
        iso = load_isochrone(city, "own_bike", minutes)
        if iso is None:
            continue
        res = point_in_polygon_accuracy(
            origin_trips, iso,
            label=f"{city} | Own Bike {minutes} min | origin={origin['label']}"
        )
        res["city"]    = city
        res["minutes"] = minutes
        res["origin"]  = origin["label"]
        results.append(res)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TASK C — DEMAND vs. COVERAGE  (Station flow × isochrone reach)
# ═════════════════════════════════════════════════════════════════════════════

def compute_station_flow(
    trips_df: pd.DataFrame,
    lat_col: str = "start_lat",
    lon_col: str = "start_lng",
    snap_m: float = 50.0,
) -> gpd.GeoDataFrame:
    """
    Aggregate trip starts by unique coordinate pair (acting as a proxy for
    bike-share station locations) within a snapping tolerance.

    Returns a point GeoDataFrame with a 'flow_count' column.
    """
    if trips_df.empty:
        return gpd.GeoDataFrame()

    # Round coordinates to ~10 m precision to merge near-duplicate stations
    trips_df = trips_df.copy()
    trips_df["_lat_r"] = trips_df[lat_col].round(4)
    trips_df["_lon_r"] = trips_df[lon_col].round(4)

    flow = (
        trips_df.groupby(["_lat_r", "_lon_r"])
        .size()
        .reset_index(name="flow_count")
    )

    geom = [Point(lon, lat) for lon, lat in zip(flow["_lon_r"], flow["_lat_r"])]
    flow_gdf = gpd.GeoDataFrame(flow, geometry=geom, crs=CRS_WGS84)
    flow_gdf = flow_gdf.rename(columns={"_lat_r": "station_lat",
                                         "_lon_r": "station_lon"})
    print(f"  Aggregated {len(trips_df)} trips → {len(flow_gdf)} unique station(s).")
    return flow_gdf


def attach_isochrone_reach(
    stations_gdf: gpd.GeoDataFrame,
    shared_bike_iso_gdf: gpd.GeoDataFrame,
    minutes: int = 15,
) -> gpd.GeoDataFrame:
    """
    For each station point, compute the area (km²) of the shared-bike isochrone
    feature that *contains* the station (proxy for local network reach).

    If isochrones are pre-computed per station this is a direct lookup;
    when a single city-wide isochrone is used, every station receives the same
    area (still useful as a constant 'coverage' flag).
    """
    if stations_gdf.empty or shared_bike_iso_gdf is None:
        return stations_gdf

    iso_m = to_metric(dissolve_to_single(shared_bike_iso_gdf))
    iso_area = polygon_area_km2(iso_m)

    # Spatial join — flag stations that lie inside the isochrone
    iso_wgs = iso_m.to_crs(CRS_WGS84)
    joined  = gpd.sjoin(
        stations_gdf, iso_wgs, how="left", predicate="within"
    )
    stations_gdf = stations_gdf.copy()
    stations_gdf[f"in_shared_bike_{minutes}min"] = (
        joined["index_right"].notna().values
    )
    stations_gdf[f"iso_area_km2_{minutes}min"] = np.where(
        stations_gdf[f"in_shared_bike_{minutes}min"], iso_area, 0.0
    )
    return stations_gdf


def demand_coverage_summary(
    stations_gdf: gpd.GeoDataFrame,
    city: str,
    minutes: int = 15,
    flow_threshold_pct: float = 25.0,
    area_threshold_pct: float = 75.0,
) -> pd.DataFrame:
    """
    Build a summary table flagging stations with:
      - HIGH reach (iso area ≥ 75th percentile) but LOW flow (≤ 25th percentile)
      - LOW reach  (iso area ≤ 25th percentile) but HIGH flow (≥ 75th percentile)

    These are the operationally interesting imbalances.
    """
    if stations_gdf.empty:
        return pd.DataFrame()

    area_col = f"iso_area_km2_{minutes}min"
    df = stations_gdf[
        ["station_lat", "station_lon", "flow_count", area_col]
    ].copy()
    df["city"]    = city
    df["minutes"] = minutes

    low_flow_q  = np.percentile(df["flow_count"], flow_threshold_pct)
    high_area_q = np.percentile(df[area_col],    area_threshold_pct)
    low_area_q  = np.percentile(df[area_col],    100 - area_threshold_pct)
    high_flow_q = np.percentile(df["flow_count"], 100 - flow_threshold_pct)

    df["high_reach_low_flow"] = (
        (df[area_col]    >= high_area_q) &
        (df["flow_count"] <= low_flow_q)
    )
    df["low_reach_high_flow"] = (
        (df[area_col]    <= low_area_q)  &
        (df["flow_count"] >= high_flow_q)
    )
    df["mismatch_type"] = "balanced"
    df.loc[df["high_reach_low_flow"], "mismatch_type"] = "HIGH_REACH_LOW_FLOW"
    df.loc[df["low_reach_high_flow"], "mismatch_type"] = "LOW_REACH_HIGH_FLOW"

    n_hr_lf = df["high_reach_low_flow"].sum()
    n_lr_hf = df["low_reach_high_flow"].sum()
    print(f"  HIGH reach / LOW flow stations  : {n_hr_lf}")
    print(f"  LOW reach  / HIGH flow stations : {n_lr_hf}")
    return df.drop(columns=["high_reach_low_flow", "low_reach_high_flow"])


# ═════════════════════════════════════════════════════════════════════════════
# 5.  SYNERGY GEOJSON EXPORT  (for QGIS)
# ═════════════════════════════════════════════════════════════════════════════

def build_synergy_geojson(synergy_results: list[dict]) -> gpd.GeoDataFrame:
    """
    Assemble a multi-feature GeoDataFrame containing:
      - The union polygon  (PT ∪ Shared Bike)
      - The transit-gap polygon  (Shared Bike  −  PT)
    with attribute columns for city, minutes, SMI, areas, and layer type.
    """
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
                "layer_type":        "union_PT_SharedBike",
                "area_km2":          r["area_union_km2"],
                "area_pt_km2":       r["area_pt_km2"],
                "area_bike_km2":     r["area_shared_bike_km2"],
                "area_overlap_km2":  r["area_overlap_km2"],
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
                "area_pt_km2":  r["area_pt_km2"],
                "area_bike_km2":r["area_shared_bike_km2"],
                "area_overlap_km2": r["area_overlap_km2"],
                "area_gap_km2": r["area_transit_gap_km2"],
                "SMI":          r["SMI"],
            })

    if not rows:
        print("  [WARN] No synergy geometries to export.")
        return gpd.GeoDataFrame()

    out_gdf = gpd.GeoDataFrame(rows, crs=CRS_WGS84)
    return out_gdf


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Nextbike × PT — Validation & Synergy Analysis")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_synergy_results   = []   # Task A scalar rows
    all_validation_results= []   # Task B scalar rows
    all_demand_rows       = []   # Task C scalar rows
    geojson_inputs        = []   # for QGIS export

    # ── load trip data once (shared across cities) ────────────────────────────
    trips_df = load_trip_data(EXCEL_PATH)

    # ── per-city loop ─────────────────────────────────────────────────────────
    for city in CITIES:
        print(f"\n{'═'*65}")
        print(f"  CITY: {city.upper()}")
        print(f"{'═'*65}")

        # ── Task A: Geometric Synergy ─────────────────────────────────────────
        for minutes in [15, 30]:
            res = analyse_geometric_synergy(city, minutes)
            if res:
                # strip internal GDFs before storing scalars
                scalar_res = {k: v for k, v in res.items()
                              if not k.startswith("_")}
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

        # Filter trips for this city if a 'city' column exists, else use all
        city_trips = trips_df
        if "city" in trips_df.columns:
            city_trips = trips_df[trips_df["city"].str.lower() == city]

        stations_gdf = compute_station_flow(city_trips)

        for minutes in [15, 30]:
            iso_shared = load_isochrone(city, "shared_bike", minutes)
            if iso_shared is not None and not stations_gdf.empty:
                stations_gdf = attach_isochrone_reach(
                    stations_gdf, iso_shared, minutes
                )

        if not stations_gdf.empty:
            summary = demand_coverage_summary(stations_gdf, city, minutes=15)
            all_demand_rows.append(summary)

    # ═════════════════════════════════════════════════════════════════════════
    # 7.  OUTPUTS
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n{'═'*65}")
    print("  Writing outputs …")
    print(f"{'═'*65}")

    # ── Validation_Report.csv ─────────────────────────────────────────────────
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
        # Concatenate with fill so mismatched columns are NaN-padded
        full_report = pd.concat(report_sections, ignore_index=True, sort=False)
        full_report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
        print(f"  [SAVED] {REPORT_CSV}  ({len(full_report)} rows)")
    else:
        print("  [WARN] No results to write to CSV.")

    # ── synergy_analysis.geojson ──────────────────────────────────────────────
    synergy_gdf = build_synergy_geojson(geojson_inputs)
    if not synergy_gdf.empty:
        synergy_gdf.to_file(SYNERGY_GJSON, driver="GeoJSON")
        print(f"  [SAVED] {SYNERGY_GJSON}  ({len(synergy_gdf)} feature(s))")
    else:
        print("  [WARN] synergy_analysis.geojson is empty — check isochrone files.")

    # ── Pretty console summary ────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  SUMMARY TABLE — Geometric Synergy")
    print(f"{'═'*65}")
    if all_synergy_results:
        df = pd.DataFrame(all_synergy_results)
        cols = ["city", "minutes", "area_pt_km2", "area_shared_bike_km2",
                "area_union_km2", "area_transit_gap_km2", "SMI"]
        print(df[cols].to_string(index=False))

    print(f"\n{'═'*65}")
    print("  SUMMARY TABLE — Model Validation")
    print(f"{'═'*65}")
    if all_validation_results:
        df = pd.DataFrame(all_validation_results)
        cols = ["city", "minutes", "origin", "total_trips",
                "ends_inside", "accuracy_pct"]
        print(df[cols].to_string(index=False))
    else:
        print("  (no validation results — check Excel path and study origins)")

    print(f"\n{'═'*65}")
    print("  Done. ✓")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()