import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os  # <--- Added to handle directories

# ==========================================
# 1. SETUP & DATA
# ==========================================
# Define output directory
OUTPUT_DIR = os.path.join("../results", "maps")
# Create directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

file_path = '/data/nextbike_data_VSB_Vaclavik.xlsx'

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
all_sheets = pd.read_excel(file_path, sheet_name=None)
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


def get_offset_geo(lat1, lon1, lat2, lon2, offset_meters=30):
    d_lat = 1 / 111111
    d_lon = 1 / (111111 * np.cos(np.radians((lat1 + lat2) / 2)))
    dx = (lon2 - lon1) / d_lon
    dy = (lat2 - lat1) / d_lat
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if dist == 0: return (lat1, lon1), (lat2, lon2)
    px, py = -dy / dist, dx / dist
    s_lat = lat1 + py * offset_meters * d_lat
    s_lon = lon1 + px * offset_meters * d_lon
    e_lat = lat2 + py * offset_meters * d_lat
    e_lon = lon2 + px * offset_meters * d_lat
    return (s_lat, s_lon), (e_lat, e_lon)


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
def add_traces_to_fig(fig, hub_name, partner_data, h_lat, h_lng, traces_list):
    """Calculates lines and arrows for a specific hub-partner pair"""
    for p in partner_data['partner'].unique():
        if p == hub_name: continue

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

        if out_c > 0:
            (sl, slo), (el, elo) = get_offset_geo(h_lat, h_lng, p_lat, p_lng, 25)
            traces_list.append(go.Scattermap(
                mode="lines", lat=[sl, el], lon=[slo, elo],
                line=dict(width=min(out_c / 2, 6), color='#00BFFF'), opacity=0.7,
                hoverinfo='text', text=f"OUT: {out_c} -> {p}", showlegend=False
            ))
            ay, ax = get_arrow_head(sl, slo, el, elo)
            traces_list.append(
                go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=2, color='#00BFFF'), hoverinfo='skip',
                              showlegend=False))

        if in_c > 0:
            (sl, slo), (el, elo) = get_offset_geo(p_lat, p_lng, h_lat, h_lng, 25)
            traces_list.append(go.Scattermap(
                mode="lines", lat=[sl, el], lon=[slo, elo],
                line=dict(width=min(in_c / 2, 6), color='#FF1493'), opacity=0.7,
                hoverinfo='text', text=f"IN: {in_c} <- {p}", showlegend=False
            ))
            ay, ax = get_arrow_head(sl, slo, el, elo)
            traces_list.append(
                go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=2, color='#FF1493'), hoverinfo='skip',
                              showlegend=False))

        traces_list.append(go.Scattermap(
            mode="markers+text", lat=[p_lat], lon=[p_lng],
            marker=dict(size=6, color='white'), text=p, textposition="top center",
            textfont=dict(size=9, color="#ccc"), showlegend=False
        ))


# ==========================================
# 4. MAP GENERATOR: STANDARD / SINGLE STATION
# ==========================================
def generate_standard_map(center_station, file_label, folder_prefix=""):
    print(f"--- Generating Map: {file_label} ---")

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
    top20_list = counts.head(20).index.tolist()

    traces_top20 = []
    traces_other = []

    temp_traces_top = []
    temp_traces_other = []

    for p in subset['partner'].unique():
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

        target_list = temp_traces_top if p in top20_list else temp_traces_other

        if out_c > 0:
            (sl, slo), (el, elo) = get_offset_geo(h_lat, h_lng, p_lat, p_lng, 25)
            target_list.append(go.Scattermap(mode="lines", lat=[sl, el], lon=[slo, elo],
                                             line=dict(width=min(out_c / 2, 6), color='#00BFFF'), opacity=0.7,
                                             hoverinfo='text', text=f"OUT: {out_c}", showlegend=False))
            ay, ax = get_arrow_head(sl, slo, el, elo)
            target_list.append(
                go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=2, color='#00BFFF'), hoverinfo='skip',
                              showlegend=False))

        if in_c > 0:
            (sl, slo), (el, elo) = get_offset_geo(p_lat, p_lng, h_lat, h_lng, 25)
            target_list.append(go.Scattermap(mode="lines", lat=[sl, el], lon=[slo, elo],
                                             line=dict(width=min(in_c / 2, 6), color='#FF1493'), opacity=0.7,
                                             hoverinfo='text', text=f"IN: {in_c}", showlegend=False))
            ay, ax = get_arrow_head(sl, slo, el, elo)
            target_list.append(
                go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=2, color='#FF1493'), hoverinfo='skip',
                              showlegend=False))

        target_list.append(
            go.Scattermap(mode="markers+text", lat=[p_lat], lon=[p_lng], marker=dict(size=6, color='white'), text=p,
                          textposition="top center", textfont=dict(size=9, color="#ccc"), showlegend=False))

    hub_dot = go.Scattermap(
        mode="markers+text", lat=[h_lat], lon=[h_lng],
        marker=dict(size=25, color='#FFD700', symbol='star'),
        text=center_station, textposition="bottom center",
        textfont=dict(size=14, color='#FFD700', family="Arial Black"),
        name="Hub"
    )
    temp_traces_top.append(hub_dot)
    temp_traces_other.append(hub_dot)

    fig = go.Figure()
    for t in temp_traces_top: fig.add_trace(t)
    for t in temp_traces_other: fig.add_trace(t)

    n_top = len(temp_traces_top)
    n_other = len(temp_traces_other)

    vis_top = [True] * n_top + [False] * n_other
    vis_all = [True] * (n_top + n_other)

    for i in range(len(fig.data)): fig.data[i].visible = vis_top[i]

    fig.update_layout(
        title=f"<b>Flows: {file_label}</b>", title_font_color="white",
        paper_bgcolor="black", plot_bgcolor="black",
        map=dict(style="carto-darkmatter", center=dict(lat=h_lat, lon=h_lng), zoom=13),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.01, y=1.05, bgcolor="white",
            buttons=[
                dict(label="Show Top 20", method="update", args=[{"visible": vis_top}]),
                dict(label="Show All", method="update", args=[{"visible": vis_all}])
            ]
        )],
        margin=dict(l=0, r=0, t=50, b=0), height=800
    )

    # UPDATED: Save to OUTPUT_DIR
    fname = os.path.join(OUTPUT_DIR, f"{folder_prefix}Map_{file_label}.html")
    fig.write_html(fname)
    print(f"Saved: {fname}")


# ==========================================
# 5. MAP GENERATOR: BRNO COMBINED (SEPARATE NODES)
# ==========================================
def generate_brno_combined_separate_nodes():
    print("--- Generating Map: Brno Combined (Separate Nodes) ---")

    fig = go.Figure()
    all_traces = []
    lats, lngs = [], []

    for hub in BRNO_STATIONS:
        try:
            r = all_trips[all_trips['start_place'] == hub].iloc[0]
            h_lat, h_lng = r[col_lat_start], r[col_lng_start]
            lats.append(h_lat);
            lngs.append(h_lng)
        except:
            continue

        subset = all_trips[(all_trips['start_place'] == hub) | (all_trips['end_place'] == hub)].copy()
        subset['partner'] = subset.apply(lambda x: x['end_place'] if x['start_place'] == hub else x['start_place'],
                                         axis=1)

        add_traces_to_fig(fig, hub, subset, h_lat, h_lng, all_traces)

        all_traces.append(go.Scattermap(
            mode="markers+text", lat=[h_lat], lon=[h_lng],
            marker=dict(size=15, color='#FFD700'),
            text=hub, textposition="bottom center", textfont=dict(size=11, color="yellow"),
            name=hub
        ))

    for t in all_traces: fig.add_trace(t)

    c_lat, c_lng = np.mean(lats), np.mean(lngs)

    fig.update_layout(
        title="<b>Brno Network (All 3 Stations Active)</b>", title_font_color="white",
        paper_bgcolor="black", plot_bgcolor="black",
        map=dict(style="carto-darkmatter", center=dict(lat=c_lat, lon=c_lng), zoom=14),
        margin=dict(l=0, r=0, t=50, b=0), height=900
    )
    # UPDATED: Save to OUTPUT_DIR
    fname = os.path.join(OUTPUT_DIR, "Brno_Combined_SeparateNodes.html")
    fig.write_html(fname)
    print(f"Saved: {fname}")


# ==========================================
# 6. MAP GENERATOR: BRNO CLUSTERED (MERGED)
# ==========================================
def generate_brno_clustered():
    print("--- Generating Map: Brno Clustered (Merged) ---")

    cluster_name = "Brno Master Hub"
    temp_df = all_trips.copy()

    temp_df.loc[temp_df['start_place'].isin(BRNO_STATIONS), 'start_place'] = cluster_name
    temp_df.loc[temp_df['end_place'].isin(BRNO_STATIONS), 'end_place'] = cluster_name

    c_data = all_trips[all_trips['start_place'].isin(BRNO_STATIONS)]
    c_lat, c_lng = c_data[col_lat_start].mean(), c_data[col_lng_start].mean()

    generate_standard_map_clustered_logic(temp_df, cluster_name, c_lat, c_lng, "Brno_Clustered_Total")


def generate_standard_map_clustered_logic(dataframe, center_name, h_lat, h_lng, filename):
    subset = dataframe[(dataframe['start_place'] == center_name) | (dataframe['end_place'] == center_name)].copy()
    subset['partner'] = subset.apply(lambda x: x['end_place'] if x['start_place'] == center_name else x['start_place'],
                                     axis=1)

    counts = subset['partner'].value_counts()
    top20_list = counts.head(20).index.tolist()

    traces_top = []
    traces_other = []

    for p in subset['partner'].unique():
        if p == center_name: continue
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

        target = traces_top if p in top20_list else traces_other

        if out_c > 0:
            (sl, slo), (el, elo) = get_offset_geo(h_lat, h_lng, p_lat, p_lng, 25)
            target.append(go.Scattermap(mode="lines", lat=[sl, el], lon=[slo, elo],
                                        line=dict(width=min(out_c / 2, 8), color='#00BFFF'), opacity=0.8,
                                        hoverinfo='text', text=f"OUT: {out_c}", showlegend=False))
            ay, ax = get_arrow_head(sl, slo, el, elo)
            target.append(
                go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=2, color='#00BFFF'), hoverinfo='skip',
                              showlegend=False))

        if in_c > 0:
            (sl, slo), (el, elo) = get_offset_geo(p_lat, p_lng, h_lat, h_lng, 25)
            target.append(go.Scattermap(mode="lines", lat=[sl, el], lon=[slo, elo],
                                        line=dict(width=min(in_c / 2, 8), color='#FF1493'), opacity=0.8,
                                        hoverinfo='text', text=f"IN: {in_c}", showlegend=False))
            ay, ax = get_arrow_head(sl, slo, el, elo)
            target.append(
                go.Scattermap(mode="lines", lat=ay, lon=ax, line=dict(width=2, color='#FF1493'), hoverinfo='skip',
                              showlegend=False))

        target.append(
            go.Scattermap(mode="markers+text", lat=[p_lat], lon=[p_lng], marker=dict(size=6, color='white'), text=p,
                          textposition="top center", textfont=dict(size=9, color="#ccc"), showlegend=False))

    hub_dot = go.Scattermap(
        mode="markers+text", lat=[h_lat], lon=[h_lng],
        marker=dict(size=25, color='#FFD700', symbol='star'),
        text="<b>BRNO CLUSTER</b>", textposition="bottom center",
        textfont=dict(size=14, color='#FFD700'), name="Cluster Hub"
    )
    traces_top.append(hub_dot)
    traces_other.append(hub_dot)

    fig = go.Figure()
    for t in traces_top: fig.add_trace(t)
    for t in traces_other: fig.add_trace(t)

    n_top, n_other = len(traces_top), len(traces_other)
    vis_top = [True] * n_top + [False] * n_other
    vis_all = [True] * (n_top + n_other)

    for i in range(len(fig.data)): fig.data[i].visible = vis_top[i]

    fig.update_layout(
        title=f"<b>Flows: {center_name}</b>", title_font_color="white",
        paper_bgcolor="black", plot_bgcolor="black",
        map=dict(style="carto-darkmatter", center=dict(lat=h_lat, lon=h_lng), zoom=13),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.01, y=1.05, bgcolor="white",
            buttons=[
                dict(label="Show Top 20", method="update", args=[{"visible": vis_top}]),
                dict(label="Show All", method="update", args=[{"visible": vis_all}])
            ]
        )],
        margin=dict(l=0, r=0, t=50, b=0), height=800
    )

    # UPDATED: Save to OUTPUT_DIR
    fname = os.path.join(OUTPUT_DIR, f"{filename}.html")
    fig.write_html(fname)
    print(f"Saved: {fname}")


# ==========================================
# 7. EXECUTION
# ==========================================

print(f"--- Outputting maps to: {OUTPUT_DIR} ---")

# 1. Brno: 3 Separate Maps (Individual Stations)
for station in BRNO_STATIONS:
    safe_name = "Brno_" + station.replace(" ", "_").replace("-", "_")
    generate_standard_map(station, safe_name)

# 2. Brno: 1 Map with All 3 Stations (Separate Nodes)
generate_brno_combined_separate_nodes()

# 3. Brno: 1 Clustered Map (Merged Data)
generate_brno_clustered()

# 4. Other Cities (Standard Maps)
generate_standard_map(STATION_SVINOV, "Ostrava_Svinov")
generate_standard_map(STATION_MOAP, "Ostrava_MOAP")
generate_standard_map(STATION_VALMEZ, "Valasske_Mezirici")
generate_standard_map(STATION_PREROV, "Prerov")

print(f"\nAll maps generated successfully in {OUTPUT_DIR}!")