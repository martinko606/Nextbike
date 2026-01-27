import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal.windows import gaussian
import os
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. UNIVERSAL CONFIGURATION
# ==========================================
CITIES = {
    'Ostrava-Svinov': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava-Svinov',
        'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)']
    },
    'Ostrava hl.n.': {
        'file_trains': 'data/Pohyby_Ostrava_hl.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava hl.n.',
        'bike_stations': ['MOAP-Hlavní nádraží']
    },
    'Brno': {
        'file_trains': 'data/Pohyby_Brno.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'file_weather': 'data/Brno_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Brno',
        'train_station': 'Brno hl.n.',
        'bike_stations': ['Hlavní nádraží - Hlavní vstup', 'Hlavní nádraží - pošta', 'Bajkazyl 666']
    },
    'Prerov': {
        'file_trains': 'data/Pohyby_Prerov.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'file_weather': 'data/Prerov_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Prerov',
        'train_station': 'Přerov os.n.',
        'bike_stations': ['Nádraží']
    },
    'ValMez': {
        'file_trains': 'data/Pohyby_ValMez.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'file_weather': 'data/ValMez_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_ValMez',
        'train_station': 'Valašské Meziříčí',
        'bike_stations': ['Vlakové nádraží Valašské Meziříčí (nové umístění)']
    }
}

VIS_START = '2025-09-15 00:00'
VIS_END = '2025-10-19 23:59'

SAVE_FOLDER = 'results/3D_heatmaps'
if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)


# ==========================================
# 2. DATA LOADING
# ==========================================
def load_data(cfg):
    # --- TRAINS ---
    try:
        df_p = pd.read_excel(cfg['file_trains'], header=None, nrows=20)
        h_idx = 0
        for i, row in df_p.iterrows():
            if any('čas' in str(x).lower() for x in row.values):
                h_idx = i
                break
        df_t = pd.read_excel(cfg['file_trains'], header=h_idx)

        cols = {}
        for c in df_t.columns:
            cl = str(c).lower().strip()
            if 'příjezdu' in cl:
                cols[c] = 'arr'
            elif 'odjezdu' in cl:
                cols[c] = 'dep'
            elif 'druh' in cl:
                cols[c] = 'type'
            elif 'bod' in cl:
                cols[c] = 'station'
        df_t.rename(columns=cols, inplace=True)

        if 'station' in df_t.columns:
            if isinstance(df_t['station'], pd.DataFrame):
                st_col = df_t['station'].iloc[:, 0]
            else:
                st_col = df_t['station']
            df_t = df_t[st_col.astype(str).str.contains(cfg['train_station'], case=False, na=False)]

        df_t['arr'] = pd.to_datetime(df_t['arr'], errors='coerce')
        df_t['dep'] = pd.to_datetime(df_t['dep'], errors='coerce')

    except Exception as e:
        print(f"   ❌ Train Load Error: {e}")
        return None, None, None

    # --- BIKES ---
    try:
        df_b = pd.read_excel(cfg['file_bikes'], sheet_name=cfg['sheet_rentals'])
        df_b.columns = [str(c).lower().strip() for c in df_b.columns]

        valid = [s.lower() for s in cfg['bike_stations']]
        mask = df_b['start_place'].astype(str).str.lower().isin(valid) | \
               df_b['end_place'].astype(str).str.lower().isin(valid)
        df_b = df_b[mask].copy()

        df_b['start_time'] = pd.to_datetime(df_b['start_time'])
        df_b['end_time'] = pd.to_datetime(df_b['end_time'])

    except Exception as e:
        print(f"   ❌ Bike Load Error: {e}")
        return None, None, None

    # --- WEATHER (Fixed Detection) ---
    try:
        df_w = pd.read_excel(cfg['file_weather'])
        df_w.columns = [str(c).lower().strip() for c in df_w.columns]

        t_col = next((c for c in df_w.columns if any(x in c for x in ['date', 'time', 'cas'])), None)
        p_col = next((c for c in df_w.columns if any(x in c for x in ['rain', 'precip', 'uhrn'])), None)

        if t_col:
            df_w['ts'] = pd.to_datetime(df_w[t_col])
        if p_col:
            df_w['rain'] = pd.to_numeric(df_w[p_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        else:
            df_w['rain'] = 0
            print("   ⚠️ No rain column found (checked 'rain', 'precip', 'uhrn')")

    except:
        df_w = pd.DataFrame({'ts': [], 'rain': []})

    return df_t, df_b, df_w


# ==========================================
# 3. 3D GENERATOR
# ==========================================
def generate_3d_swarm(city, cfg):
    print(f"\n🚀 Generating 3D Swarm for: {city}...")

    df_t, df_b, df_w = load_data(cfg)
    if df_t is None: return

    # Define Timeline
    zoom_start = pd.to_datetime(VIS_START)
    zoom_end = pd.to_datetime(VIS_END)
    timeline_1min = pd.date_range(zoom_start, zoom_end, freq='1min')

    # --- A. CALCULATE PRESSURE ---
    print("   ...Calculating High-Res Pressure")
    s_press = pd.Series(0.0, index=timeline_1min)
    w_map = {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1.0, 'IC': 1.0}

    mask_t = (df_t['arr'] >= zoom_start) & (df_t['arr'] <= zoom_end)
    subset_t = df_t[mask_t]

    for _, r in subset_t.iterrows():
        w = w_map.get(r.get('type', 'Os'), 1.0)
        if pd.notnull(r['arr']):
            try:
                s_press.at[r['arr'].round('min')] += w
            except:
                pass
        if pd.notnull(r['dep']):
            try:
                s_press.at[r['dep'].round('min')] += w
            except:
                pass

    gauss = gaussian(30, std=4)
    gauss /= gauss.sum()
    s_smooth = pd.Series(np.convolve(s_press.values, gauss, mode='same'), index=timeline_1min)

    max_val = s_smooth.max()
    scale_factor = 10.0 / max_val if max_val > 0 else 1.0
    s_scaled = s_smooth * scale_factor

    # --- B. CREATE TERRAIN MESH ---
    print("   ...Building Terrain Mesh")

    # Grid for visuals (15 min)
    df_grid = pd.DataFrame({'pressure': s_scaled})
    df_mesh = df_grid.resample('15min').mean()

    df_mesh['date'] = df_mesh.index.date
    df_mesh['time_dec'] = df_mesh.index.hour + df_mesh.index.minute / 60.0

    z_matrix = df_mesh.pivot(index='date', columns='time_dec', values='pressure')

    all_hours = np.arange(0, 24, 0.25)
    z_matrix = z_matrix.reindex(columns=all_hours, fill_value=0).fillna(0)

    all_dates = pd.date_range(zoom_start.date(), zoom_end.date()).date
    z_matrix = z_matrix.reindex(index=all_dates, fill_value=0).fillna(0)

    z_values = z_matrix.values
    x_values = z_matrix.columns.values
    y_values = [str(d) for d in z_matrix.index]

    # --- C. PROCESS RAIN LAYER (ROBUST) ---
    print("   ...Processing Rain Layer")
    rain_threshold = 0.1  # Lower threshold to capture light rain
    rain_height = 10.1

    rain_surface = np.full(z_values.shape, np.nan)  # Start empty (transparent)

    if not df_w.empty:
        w_cut = df_w[(df_w['ts'] >= zoom_start) & (df_w['ts'] <= zoom_end)]
        print(f"      -> Max Rain in Window: {w_cut['rain'].max()} mm")

        if not w_cut.empty:
            # 1. Create a Series with same index as 1-min timeline, fill with 0
            w_series = pd.Series(0.0, index=timeline_1min)

            # 2. Map loaded rain to this timeline
            # (Round to nearest minute to ensure matching)
            w_loaded = w_cut.set_index('ts')['rain']
            w_loaded.index = w_loaded.index.round('min')

            # 3. Update the 1-min timeline with actual rain values
            w_series.update(w_loaded)

            # 4. Resample to 15-min to match the Grid (Max value in that 15 min block)
            w_15min = w_series.resample('15min').max()

            # 5. Create DataFrame for pivoting
            df_rain_grid = pd.DataFrame({'rain': w_15min})
            df_rain_grid['date'] = df_rain_grid.index.date
            df_rain_grid['time_dec'] = df_rain_grid.index.hour + df_rain_grid.index.minute / 60.0

            # 6. Pivot exactly like Terrain
            r_matrix = df_rain_grid.pivot(index='date', columns='time_dec', values='rain')
            r_matrix = r_matrix.reindex(index=all_dates, columns=all_hours, fill_value=0).fillna(0)

            # 7. Apply Mask
            is_raining = r_matrix.values > rain_threshold
            rain_surface[is_raining] = rain_height

            print(f"      -> Painted {np.sum(is_raining)} rain slots (Grid Cells).")
    else:
        print("      ⚠️ No weather data available.")

    # --- D. PROCESS DOTS ---
    print("   ...Placing Dots")
    valid_st = [s.lower() for s in cfg['bike_stations']]

    def get_dots(df, time_col, station_col, z_add):
        mask = df[station_col].astype(str).str.lower().isin(valid_st) & \
               (df[time_col] >= zoom_start) & (df[time_col] <= zoom_end)
        sub = df[mask]

        x, y, z, t = [], [], [], []
        for _, row in sub.iterrows():
            ts = row[time_col]
            x.append(ts.hour + ts.minute / 60.0 + ts.second / 3600.0)
            y.append(str(ts.date()))

            lookup_t = ts.round('min')
            if lookup_t in s_scaled.index:
                base_h = s_scaled.loc[lookup_t]
            else:
                base_h = 0

            z.append(base_h + z_add)
            t.append(ts.strftime('%d.%m %H:%M'))
        return x, y, z, t

    rx, ry, rz, rt = get_dots(df_b, 'start_time', 'start_place', z_add=0.3)
    ex, ey, ez, et = get_dots(df_b, 'end_time', 'end_place', z_add=0.3)

    # --- E. RENDER ---
    print("   ...Rendering")
    fig = go.Figure()

    # Terrain (Orange)
    fig.add_trace(go.Surface(
        z=z_values, x=x_values, y=y_values,
        colorscale='Oranges', opacity=0.6, name='Pressure',
        colorbar=dict(title='Pressure', x=0, len=0.5)
    ))

    # Rain (Blue)
    fig.add_trace(go.Surface(
        z=rain_surface, x=x_values, y=y_values,
        colorscale=[[0, '#87CEFA'], [1, '#00008B']],
        opacity=0.4, showscale=False, name='Rain',
        hoverinfo='skip'
    ))

    # Dots
    fig.add_trace(go.Scatter3d(
        x=rx, y=ry, z=rz, mode='markers',
        marker=dict(size=2, color='#2ca02c'),
        name='Rental', text=rt
    ))
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='markers',
        marker=dict(size=2, color='#1f77b4'),
        name='Return', text=et
    ))

    fig.update_layout(
        title=f"3D Probabilistic Heatmap: {city}",
        scene=dict(
            xaxis_title='Time (Hour)',
            yaxis_title='Date',
            zaxis_title='Intensity',
            xaxis=dict(tickvals=list(range(0, 25, 4))),
            aspectmode='manual',
            aspectratio=dict(x=1, y=3.0, z=0.5)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=900
    )

    fname = f"{SAVE_FOLDER}/3D_Swarm_{city.replace(' ', '_').replace('.', '')}.html"
    fig.write_html(fname)
    print(f"   ✅ Saved to {fname}")


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    for city, cfg in CITIES.items():
        generate_3d_swarm(city, cfg)