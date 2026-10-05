import sys, os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point

# ==========================================
# 1. SETUP & DATA
# ==========================================
# Dynamically resolve repo root to prevent file path errors
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define output directories
HTML_DIR = os.path.join(REPO_ROOT, "results", "spatial_analysis", "flows_html")
GEOJSON_DIR = os.path.join(REPO_ROOT, "results", "spatial_analysis", "flows_geojson")
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(GEOJSON_DIR, exist_ok=True)

file_path = os.path.join(REPO_ROOT, "data", "nextbike_data_VSB_Vaclavik.xlsx")

# Station Names
STATION_SVINOV = 'SV-Svinov nádraží *(navíc 15min na odjezd)'
STATION_MOAP = 'MOAP-Hlavní nádraží'
STATION_VALMEZ = 'Vlakové nádraží Valašské Meziříčí (nové umístění)'
STATION_PREROV = 'Nádraží'

BRNO_STATIONS = [
    'Hlavní nádraží - Hlavní vstup',
    'Hlavní nádraží - pošta',
    'Bajkazyl 666'
]

print("Step 1: Loading data...")
try:
    all_sheets = pd.read_excel(file_path, sheet_name=None)
except FileNotFoundError:
    raise FileNotFoundError(f"Cannot find data file at {file_path}. Please check directory structure.")

valid_sheets = [df for df in all_sheets.values() if not df.empty]
all_trips = pd.concat(valid_sheets, ignore_index=True)

all_trips.columns = [c.strip() for c in all_trips.columns]
col_lat_start, col_lng_start = 'start_lat', 'start_lng'
col_lat_end, col_lng_end = 'end_lat', 'end_lng'

# Global Outlier Removal
all_trips = all_trips[
    (all_trips['start_place'] != "BIKE 487804") &
    (all_trips['end_place'] != "BIKE 487804")
    ]


# ==========================================
# 2. MATH HELPERS
# ==========================================
def calculate_distance_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# NOTE: get_offset_geo has been removed to ensure exact station point coordinate usage.

def get_arrow_head(lat1, lon1, lat2, lon2, size_scale=0.00015):
    mx = lon1 + (lon2 - lon1) * 0.55
    my = lat1 + (lat2 - lat1) * 0.55
    dx, dy = lon2 - lon1, lat2 - lat1
    angle = np.arctan2(dy, dx)
    a1, a2 = angle + np.radians(150), angle - np.radians(150)
    w1x, w1y = mx + np.cos(a1) * size_scale * 1.5, my + np.sin(a1) * size_scale
    w2x, w2y = mx + np.cos(a2) * size_scale * 1.5, my + np.sin(a2) * size_scale
    return [w1y, my, w2y], [w1x, mx, w2x]


# ==========================================
# 3. HELPER: DRAWING FUNCTION (Reusable)
# ==========================================
def add_traces_to_fig(fig, hub_name, partner_data, h_lat, h_lng, traces_list, geojson_features, top_k=None,
                      global_valid_partners=None, global_max_trips=None):
    """Calculates lines and arrows for a specific hub-partner pair using exact coordinates"""

    if global_valid_partners is None:
        counts = partner_data['partner'].value_counts()
        valid_partners = counts.head(top_k).index.tolist() if top_k else counts.index.tolist()
        max_trips = counts.max() if not counts.empty else 1
    else:
        valid_partners = global_valid_partners
        max_trips = global_max_trips

    for p in partner_data['partner'].unique():
        if p == hub_name: continue
        if p not in valid_partners: continue

        rank = valid_partners.index(p) + 1

        try:
            pr = all_trips[all_trips['start_place'] == p].iloc[0]
            p_lat, p_lng = pr[col_lat_start], pr[col_lng_start]
        except:
            try:
                pr = all_trips[all_trips['end_place'] == p].iloc[0]
                p_lat, p_lng = pr[col_lat_end], pr[col_lng_end]
            except:
                continue

        if calculate_distance_km(h_lat, h_lng, p_lat, p_lng) > 20: continue

        out_c = len(partner_data[(partner_data['start_place'] == hub_name) & (partner_data['end_place'] == p)])
        in_c = len(partner_data[(partner_data['start_place'] == p) & (partner_data['end_place'] == hub_name)])

        # Exact Coordinates Override
        sl, slo = h_lat, h_lng
        el, elo = p_lat, p_lng

        total_c = out_c + in_c

        if total_c > 0:
            width = max(1.5, (total_c / max_trips) * 12)
            arrow_size = 0.0001 + (width * 0.000015)

            # Base combined line
            traces_list.append(go.Scattermap(
                mode="lines", lat=[sl, el], lon=[slo, elo],
                line=dict(width=width, color='#AA00FF'), opacity=0.8,
                hoverinfo='text', text=f"TOTAL: {total_c} | OUT: {out_c} | IN: {in_c}", showlegend=False
            ))

            # Arrows for directionality (Blue = Out, Pink = In)
            if out_c > 0:
                ay, ax = get_arrow_head(sl, slo, el, elo, size_scale=arrow_size)
                traces_list.append(go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=width, color='#00BFFF'),
                                                 hoverinfo='skip', showlegend=False))

            if in_c > 0:
                ay, ax = get_arrow_head(el, elo, sl, slo, size_scale=arrow_size)
                traces_list.append(go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=width, color='#FF1493'),
                                                 hoverinfo='skip', showlegend=False))

            # Record GeoJSON feature for OUTBOUND flow
            if out_c > 0:
                geojson_features.append({
                    "name": f"{hub_name} -> {p}",
                    "type": "flow_out",
                    "from": hub_name,
                    "to": p,
                    "trip_count": out_c,
                    "rank": rank,
                    "geometry": LineString([(slo, sl), (elo, el)])  # Hub to Partner
                })

            # Record GeoJSON feature for INBOUND flow
            if in_c > 0:
                geojson_features.append({
                    "name": f"{p} -> {hub_name}",
                    "type": "flow_in",
                    "from": p,
                    "to": hub_name,
                    "trip_count": in_c,
                    "rank": rank,
                    "geometry": LineString([(elo, el), (slo, sl)])  # Partner to Hub (REVERSED COORDS)
                })

        traces_list.append(go.Scattermap(
            mode="markers+text", lat=[p_lat], lon=[p_lng],
            marker=dict(size=6, color='white'), text=f"{p} (#{rank})", textposition="top center",
            textfont=dict(size=10, color="#ccc"), showlegend=False
        ))


# ==========================================
# 4. MAP GENERATOR: STANDARD / SINGLE STATION
# ==========================================
def generate_standard_map(center_station, file_label, folder_prefix="", top_k=None):
    suffix = "Top10" if top_k else "All"
    title_suffix = "(Top 10 Hubs)" if top_k else "(All Hubs)"
    print(f"--- Generating Map: {file_label} {title_suffix} ---")

    subset = all_trips[(all_trips['start_place'] == center_station) | (all_trips['end_place'] == center_station)].copy()
    if len(subset) == 0: return

    try:
        r = all_trips[all_trips['start_place'] == center_station].iloc[0]
        h_lat, h_lng = r[col_lat_start], r[col_lng_start]
    except:
        try:
            r = all_trips[all_trips['end_place'] == center_station].iloc[0]
            h_lat, h_lng = r[col_lat_end], r[col_lng_end]
        except:
            return

    subset['partner'] = subset.apply(
        lambda x: x['end_place'] if x['start_place'] == center_station else x['start_place'], axis=1)

    counts = subset['partner'].value_counts()
    valid_partners = counts.head(top_k).index.tolist() if top_k else counts.index.tolist()
    max_trips = counts.max() if not counts.empty else 1

    traces = []
    geojson_features = []

    for p in subset['partner'].unique():
        if p == center_station: continue
        if p not in valid_partners: continue

        rank = valid_partners.index(p) + 1

        try:
            pr = all_trips[all_trips['start_place'] == p].iloc[0]
            p_lat, p_lng = pr[col_lat_start], pr[col_lng_start]
        except:
            try:
                pr = all_trips[all_trips['end_place'] == p].iloc[0]
                p_lat, p_lng = pr[col_lat_end], pr[col_lng_end]
            except:
                continue

        if calculate_distance_km(h_lat, h_lng, p_lat, p_lng) > 20: continue

        out_c = len(subset[(subset['start_place'] == center_station) & (subset['end_place'] == p)])
        in_c = len(subset[(subset['start_place'] == p) & (subset['end_place'] == center_station)])

        # Exact Coordinates Override
        sl, slo = h_lat, h_lng
        el, elo = p_lat, p_lng

        total_c = out_c + in_c

        if total_c > 0:
            width = max(1.5, (total_c / max_trips) * 12)
            arrow_size = 0.0001 + (width * 0.000015)

            # Base combined line
            traces.append(go.Scattermap(
                mode="lines", lat=[sl, el], lon=[slo, elo],
                line=dict(width=width, color='#AA00FF'), opacity=0.8,
                hoverinfo='text', text=f"TOTAL: {total_c} | OUT: {out_c} | IN: {in_c}", showlegend=False
            ))

            if out_c > 0:
                ay, ax = get_arrow_head(sl, slo, el, elo, size_scale=arrow_size)
                traces.append(go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=width, color='#00BFFF'),
                                            hoverinfo='skip', showlegend=False))

            if in_c > 0:
                ay, ax = get_arrow_head(el, elo, sl, slo, size_scale=arrow_size)
                traces.append(go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=width, color='#FF1493'),
                                            hoverinfo='skip', showlegend=False))

            # Record GeoJSON feature for OUTBOUND flow
            if out_c > 0:
                geojson_features.append({
                    "name": f"{center_station} -> {p}",
                    "type": "flow_out",
                    "from": center_station,
                    "to": p,
                    "trip_count": out_c,
                    "rank": rank,
                    "geometry": LineString([(slo, sl), (elo, el)])  # Hub to Partner
                })

            # Record GeoJSON feature for INBOUND flow
            if in_c > 0:
                geojson_features.append({
                    "name": f"{p} -> {center_station}",
                    "type": "flow_in",
                    "from": p,
                    "to": center_station,
                    "trip_count": in_c,
                    "rank": rank,
                    "geometry": LineString([(elo, el), (slo, sl)])  # Partner to Hub (REVERSED COORDS)
                })

        traces.append(
            go.Scattermap(mode="markers+text", lat=[p_lat], lon=[p_lng], marker=dict(size=6, color='white'),
                          text=f"{p} (#{rank})",
                          textposition="top center", textfont=dict(size=10, color="#ccc"), showlegend=False))

    hub_dot = go.Scattermap(
        mode="markers+text", lat=[h_lat], lon=[h_lng],
        marker=dict(size=25, color='#FFD700', symbol='star'),
        text=center_station, textposition="bottom center",
        textfont=dict(size=14, color='#FFD700', family="Arial Black"),
        name="Hub"
    )
    traces.append(hub_dot)

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f"<b>Flows: {file_label} {title_suffix}</b>", title_font_color="white",
        paper_bgcolor="black", plot_bgcolor="black",
        map=dict(style="carto-darkmatter", center=dict(lat=h_lat, lon=h_lng), zoom=13),
        margin=dict(l=0, r=0, t=50, b=0), height=800
    )

    fname_html = os.path.join(HTML_DIR, f"{folder_prefix}Map_{file_label}_{suffix}.html")
    fig.write_html(fname_html)
    print(f"  -> HTML Saved: {fname_html}")

    if geojson_features:
        gdf = gpd.GeoDataFrame(geojson_features, crs="EPSG:4326")
        fname_geo = os.path.join(GEOJSON_DIR, f"{folder_prefix}Flows_{file_label}_{suffix}.geojson")
        gdf.to_file(fname_geo, driver="GeoJSON")
        print(f"  -> GeoJSON Saved: {fname_geo}")


# ==========================================
# 5. MAP GENERATOR: BRNO COMBINED (SEPARATE NODES)
# ==========================================
def generate_brno_combined_separate_nodes(top_k=None):
    suffix = "Top10" if top_k else "All"
    title_suffix = "(Top 10 Global Partners)" if top_k else "(All Connections)"
    print(f"--- Generating Map: Brno Combined Separate Nodes {title_suffix} ---")

    fig = go.Figure()
    all_traces = []
    lats, lngs = [], []
    geojson_features = []

    # Calculate global top partners across ALL 3 Brno stations combined
    combined_subset = all_trips[
        all_trips['start_place'].isin(BRNO_STATIONS) | all_trips['end_place'].isin(BRNO_STATIONS)]

    # Extract all partners (destinations when starting in Brno, or origins when ending in Brno)
    partner_series = pd.concat([
        combined_subset.loc[combined_subset['start_place'].isin(BRNO_STATIONS), 'end_place'],
        combined_subset.loc[combined_subset['end_place'].isin(BRNO_STATIONS), 'start_place']
    ])

    # Exclude internal trips between the 3 hubs to find pure external partners
    partner_counts = partner_series[~partner_series.isin(BRNO_STATIONS)].value_counts()

    global_valid_partners = partner_counts.head(top_k).index.tolist() if top_k else partner_counts.index.tolist()
    global_max_trips = partner_counts.max() if not partner_counts.empty else 1

    for hub in BRNO_STATIONS:
        try:
            r = all_trips[all_trips['start_place'] == hub].iloc[0]
            h_lat, h_lng = r[col_lat_start], r[col_lng_start]
            lats.append(h_lat)
            lngs.append(h_lng)
        except:
            continue

        subset = all_trips[(all_trips['start_place'] == hub) | (all_trips['end_place'] == hub)].copy()
        subset['partner'] = subset.apply(lambda x: x['end_place'] if x['start_place'] == hub else x['start_place'],
                                         axis=1)

        # Pass the global top 10 list into the drawing function so it filters correctly
        add_traces_to_fig(fig, hub, subset, h_lat, h_lng, all_traces, geojson_features, top_k, global_valid_partners,
                          global_max_trips)

        all_traces.append(go.Scattermap(
            mode="markers+text", lat=[h_lat], lon=[h_lng],
            marker=dict(size=15, color='#FFD700'),
            text=hub, textposition="bottom center", textfont=dict(size=11, color="yellow"),
            name=hub
        ))

    for t in all_traces: fig.add_trace(t)

    c_lat, c_lng = np.mean(lats), np.mean(lngs)

    fig.update_layout(
        title=f"<b>Brno Network (All 3 Stations Active) {title_suffix}</b>", title_font_color="white",
        paper_bgcolor="black", plot_bgcolor="black",
        map=dict(style="carto-darkmatter", center=dict(lat=c_lat, lon=c_lng), zoom=14),
        margin=dict(l=0, r=0, t=50, b=0), height=900
    )

    fname_html = os.path.join(HTML_DIR, f"Brno_Combined_SeparateNodes_{suffix}.html")
    fig.write_html(fname_html)
    print(f"  -> HTML Saved: {fname_html}")

    if geojson_features:
        gdf = gpd.GeoDataFrame(geojson_features, crs="EPSG:4326")
        fname_geo = os.path.join(GEOJSON_DIR, f"Flows_Brno_Combined_SeparateNodes_{suffix}.geojson")
        gdf.to_file(fname_geo, driver="GeoJSON")
        print(f"  -> GeoJSON Saved: {fname_geo}")


# ==========================================
# 6. MAP GENERATOR: BRNO CLUSTERED (MERGED)
# ==========================================
def generate_brno_clustered(top_k=None):
    suffix = "Top10" if top_k else "All"
    print(f"--- Generating Map: Brno Clustered (Merged) - {suffix} ---")

    cluster_name = "Brno Master Hub"
    temp_df = all_trips.copy()

    temp_df.loc[temp_df['start_place'].isin(BRNO_STATIONS), 'start_place'] = cluster_name
    temp_df.loc[temp_df['end_place'].isin(BRNO_STATIONS), 'end_place'] = cluster_name

    c_data = all_trips[all_trips['start_place'].isin(BRNO_STATIONS)]
    c_lat, c_lng = c_data[col_lat_start].mean(), c_data[col_lng_start].mean()

    generate_standard_map_clustered_logic(temp_df, cluster_name, c_lat, c_lng, f"Brno_Clustered_Total_{suffix}", top_k)


def generate_standard_map_clustered_logic(dataframe, center_name, h_lat, h_lng, filename, top_k=None):
    title_suffix = "(Top 10 Hubs)" if top_k else "(All Hubs)"
    subset = dataframe[(dataframe['start_place'] == center_name) | (dataframe['end_place'] == center_name)].copy()
    subset['partner'] = subset.apply(lambda x: x['end_place'] if x['start_place'] == center_name else x['start_place'],
                                     axis=1)

    counts = subset['partner'].value_counts()
    valid_partners = counts.head(top_k).index.tolist() if top_k else counts.index.tolist()
    max_trips = counts.max() if not counts.empty else 1

    traces = []
    geojson_features = []

    for p in subset['partner'].unique():
        if p == center_name: continue
        if p not in valid_partners: continue

        rank = valid_partners.index(p) + 1

        try:
            pr = all_trips[all_trips['start_place'] == p].iloc[0]
            p_lat, p_lng = pr[col_lat_start], pr[col_lng_start]
        except:
            try:
                pr = all_trips[all_trips['end_place'] == p].iloc[0]
                p_lat, p_lng = pr[col_lat_end], pr[col_lng_end]
            except:
                continue

        if calculate_distance_km(h_lat, h_lng, p_lat, p_lng) > 20: continue

        out_c = len(subset[(subset['start_place'] == center_name) & (subset['end_place'] == p)])
        in_c = len(subset[(subset['start_place'] == p) & (subset['end_place'] == center_name)])

        # Exact Coordinates Override
        sl, slo = h_lat, h_lng
        el, elo = p_lat, p_lng

        total_c = out_c + in_c

        if total_c > 0:
            width = max(1.5, (total_c / max_trips) * 12)
            arrow_size = 0.0001 + (width * 0.000015)

            # Base combined line
            traces.append(go.Scattermap(
                mode="lines", lat=[sl, el], lon=[slo, elo],
                line=dict(width=width, color='#AA00FF'), opacity=0.8,
                hoverinfo='text', text=f"TOTAL: {total_c} | OUT: {out_c} | IN: {in_c}", showlegend=False
            ))

            if out_c > 0:
                ay, ax = get_arrow_head(sl, slo, el, elo, size_scale=arrow_size)
                traces.append(go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=width, color='#00BFFF'),
                                            hoverinfo='skip', showlegend=False))

            if in_c > 0:
                ay, ax = get_arrow_head(el, elo, sl, slo, size_scale=arrow_size)
                traces.append(go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=width, color='#FF1493'),
                                            hoverinfo='skip', showlegend=False))

            # Record GeoJSON feature for OUTBOUND flow
            if out_c > 0:
                geojson_features.append({
                    "name": f"{center_name} -> {p}",
                    "type": "flow_out",
                    "from": center_name,
                    "to": p,
                    "trip_count": out_c,
                    "rank": rank,
                    "geometry": LineString([(slo, sl), (elo, el)])  # Hub to Partner
                })

            # Record GeoJSON feature for INBOUND flow
            if in_c > 0:
                geojson_features.append({
                    "name": f"{p} -> {center_name}",
                    "type": "flow_in",
                    "from": p,
                    "to": center_name,
                    "trip_count": in_c,
                    "rank": rank,
                    "geometry": LineString([(elo, el), (slo, sl)])  # Partner to Hub (REVERSED COORDS)
                })

        traces.append(
            go.Scattermap(mode="markers+text", lat=[p_lat], lon=[p_lng], marker=dict(size=6, color='white'),
                          text=f"{p} (#{rank})",
                          textposition="top center", textfont=dict(size=10, color="#ccc"), showlegend=False))

    hub_dot = go.Scattermap(
        mode="markers+text", lat=[h_lat], lon=[h_lng],
        marker=dict(size=25, color='#FFD700', symbol='star'),
        text="<b>BRNO CLUSTER</b>", textposition="bottom center",
        textfont=dict(size=14, color='#FFD700'), name="Cluster Hub"
    )
    traces.append(hub_dot)

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f"<b>Flows: {center_name} {title_suffix}</b>", title_font_color="white",
        paper_bgcolor="black", plot_bgcolor="black",
        map=dict(style="carto-darkmatter", center=dict(lat=h_lat, lon=h_lng), zoom=13),
        margin=dict(l=0, r=0, t=50, b=0), height=800
    )

    fname_html = os.path.join(HTML_DIR, f"{filename}.html")
    fig.write_html(fname_html)
    print(f"  -> HTML Saved: {fname_html}")

    if geojson_features:
        gdf = gpd.GeoDataFrame(geojson_features, crs="EPSG:4326")
        fname_geo = os.path.join(GEOJSON_DIR, f"Flows_{filename}.geojson")
        gdf.to_file(fname_geo, driver="GeoJSON")
        print(f"  -> GeoJSON Saved: {fname_geo}")


# ==========================================
# 7. EXECUTION
# ==========================================

print(f"--- Outputting maps to: {HTML_DIR} and {GEOJSON_DIR} ---")

for k in [10, None]:
    # 1. Brno: 3 Separate Maps (Individual Stations)
    for station in BRNO_STATIONS:
        safe_name = "Brno_" + station.replace(" ", "_").replace("-", "_")
        generate_standard_map(station, safe_name, top_k=k)

    # 2. Brno: 1 Map with All 3 Stations (Separate Nodes)
    generate_brno_combined_separate_nodes(top_k=k)

    # 3. Brno: 1 Clustered Map (Merged Data)
    generate_brno_clustered(top_k=k)

    # 4. Other Cities (Standard Maps)
    generate_standard_map(STATION_SVINOV, "Ostrava_Svinov", top_k=k)
    generate_standard_map(STATION_MOAP, "Ostrava_MOAP", top_k=k)
    generate_standard_map(STATION_VALMEZ, "Valasske_Mezirici", top_k=k)
    generate_standard_map(STATION_PREROV, "Prerov", top_k=k)

print("\nAll interactive maps AND QGIS-ready flow lines generated successfully!")