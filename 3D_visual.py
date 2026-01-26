import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal.windows import gaussian
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 0. HELPER FUNCTIONS
# ==========================================
def standard_columns(df):
    """Renames Czech columns to standard English names."""
    rename_map = {
        'název bodu výběru': 'station', 'číslo vlaku': 'train_no',
        'druh vlaku': 'train_type', 'název výchozí stanice vlaku': 'origin',
        'název cílové stanice vlaku': 'destination', 'čas příjezdu': 'actual_arrival',
        'čas odjezdu': 'actual_departure', 'datum': 'date'
    }
    df.columns = [str(c).lower().strip() for c in df.columns]
    new_cols = {}
    for col in df.columns:
        for key, val in rename_map.items():
            if key in col:
                new_cols[col] = val
                break
    return df.rename(columns=new_cols)


def generate_signals(trains, timeline, params):
    """Generates smoothed pressure signals."""
    s_arr = pd.Series(0.0, index=timeline)
    s_dep = pd.Series(0.0, index=timeline)
    weights = params.get('weights', {'Os': 1, 'Sp': 1, 'R': 1.5, 'Ex': 2})

    for _, train in trains.iterrows():
        w = weights.get(train.get('train_type', 'Os'), 1.0)
        if pd.notnull(train.get('actual_arrival')):
            try:
                s_arr.at[train['actual_arrival'].round('min')] += w
            except:
                pass
        if pd.notnull(train.get('actual_departure')):
            try:
                s_dep.at[train['actual_departure'].round('min')] += w
            except:
                pass

    window_size = 30
    gauss_kernel = gaussian(window_size, std=4)
    gauss_kernel /= gauss_kernel.sum()
    s_arr_smooth = pd.Series(np.convolve(s_arr.values, gauss_kernel, mode='same'), index=timeline)
    s_dep_smooth = pd.Series(np.convolve(s_dep.values, gauss_kernel, mode='same'), index=timeline)
    return s_arr_smooth, s_dep_smooth


# ==========================================
# 1. CONFIGURATION
# ==========================================
NETWORKS = {
    'Ostrava-Svinov': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava-Svinov',
        'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
        'params': {'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1}},
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    },
    'Ostrava hl.n.': {
        'file_trains': 'data/Pohyby_Ostrava_hl.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava hl.n.',
        'bike_stations': ['MOAP-Hlavní nádraží'],
        'params': {'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1}},
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    },
    'Brno hl.n.': {
        'file_trains': 'data/Pohyby_Brno.xlsx',
        'file_weather': 'data/Brno_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Brno',
        'train_station': 'Brno hl.n.',
        'bike_stations': ['Hlavní nádraží - Hlavní vstup', 'Hlavní nádraží - pošta', 'Bajkazyl 666'],
        'params': {'weights': {'Os': 2.5, 'Sp': 2.0, 'R': 1.5, 'Ex': 1}},
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    },
    'Prerov': {
        'file_trains': 'data/Pohyby_Prerov.xlsx',
        'file_weather': 'data/Prerov_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Prerov',
        'train_station': 'Přerov os.n.',
        'bike_stations': ['Nádraží'],
        'params': {'weights': {'Os': 1.7, 'Sp': 1.4, 'R': 1.2, 'Ex': 1}},
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    },
    'ValMez': {
        'file_trains': 'data/Pohyby_ValMez.xlsx',
        'file_weather': 'data/ValMez_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_ValMez',
        'train_station': 'Valašské Meziříčí',
        'bike_stations': ['Vlakové nádraží Valašské Meziříčí (nové umístění)'],
        'params': {'weights': {'Os': 1.8, 'Sp': 1.6, 'R': 1.4, 'Ex': 1.6}},
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    }
}


# ==========================================
# 2. CORE GENERATOR FUNCTION
# ==========================================
def generate_3d_swarm(city_name, config):
    print(f"\n🚀 Generating 3D Swarm for: {city_name}...")

    # --- A. LOAD DATA ---
    try:
        # 1. Trains
        df_preview = pd.read_excel(config['file_trains'], header=None, nrows=20)
        header_idx = 0
        for idx, row in df_preview.iterrows():
            if any('čas' in str(x).lower() for x in row.values):
                header_idx = idx
                break
        df_trains = pd.read_excel(config['file_trains'], header=header_idx)
        df_trains = standard_columns(df_trains)

        if 'station' in df_trains.columns:
            df_trains = df_trains[
                df_trains['station'].astype(str).str.contains(config['train_station'], case=False, na=False)]

        df_trains['actual_arrival'] = pd.to_datetime(df_trains['actual_arrival'], errors='coerce')
        df_trains['actual_departure'] = pd.to_datetime(df_trains['actual_departure'], errors='coerce')

        # 2. Bikes
        bike_file = config.get('file_bikes', 'data/nextbike_data_VSB_Vaclavik.xlsx')
        df_bikes = pd.read_excel(bike_file, sheet_name=config['sheet_rentals'])
        df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]
        df_bikes['start_time'] = pd.to_datetime(df_bikes['start_time'])
        df_bikes['end_time'] = pd.to_datetime(df_bikes['end_time'])

        # 3. Weather
        df_weather = pd.read_excel(config['file_weather'])
        df_weather.columns = [str(c).lower().strip() for c in df_weather.columns]

        t_col = next((c for c in df_weather.columns if any(x in c for x in ['datetime', 'date', 'datum', 'cas'])), None)
        p_col = next((c for c in df_weather.columns if any(x in c for x in ['rain', 'precip', 'sraz', 'uhrn'])), None)

        if t_col and p_col:
            df_weather['ts'] = pd.to_datetime(df_weather[t_col], errors='coerce')
            df_weather['val'] = df_weather[p_col].astype(str).str.replace(',', '.', regex=False)
            df_weather['val'] = pd.to_numeric(df_weather['val'], errors='coerce').fillna(0)
            df_weather = df_weather.sort_values('ts')
        else:
            print("      ⚠️ Weather columns missing. Skipping rain.")
            df_weather = pd.DataFrame({'ts': [], 'val': []})

    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return

    # --- B. PROCESS TERRAIN ---
    print("   ...Processing Terrain (1-Min Resolution)")
    zoom_start = pd.to_datetime(config['vis_start'])
    zoom_end = pd.to_datetime(config['vis_end'])

    # 1. Timeline
    timeline_1min = pd.date_range(zoom_start, zoom_end, freq='1min')
    t_mask = (df_trains['actual_arrival'] >= zoom_start) & (df_trains['actual_arrival'] <= zoom_end)
    subset_trains = df_trains[t_mask].copy()

    s_arr, s_dep = generate_signals(subset_trains, timeline_1min, config['params'])
    sig_total = s_arr + s_dep

    # 2. Normalize
    max_pressure = sig_total.max()
    scaling_factor = 10.0 / max_pressure if max_pressure > 0 else 1.0
    sig_total_scaled = sig_total * scaling_factor

    # 3. Grid
    df_grid = pd.DataFrame({'pressure': sig_total_scaled, 'ts': timeline_1min}).set_index('ts')
    df_grid['date'] = df_grid.index.date
    df_grid['time_dec'] = df_grid.index.hour + df_grid.index.minute / 60.0

    z_matrix = df_grid.pivot(index='date', columns='time_dec', values='pressure')

    # Force Full Grid
    all_minutes = [h + m / 60.0 for h in range(24) for m in range(60)]
    z_matrix = z_matrix.reindex(columns=all_minutes, fill_value=0).fillna(0)

    all_dates = pd.date_range(zoom_start.date(), zoom_end.date()).date
    z_matrix = z_matrix.reindex(index=all_dates, fill_value=0).fillna(0)

    z_values = z_matrix.values
    x_values = z_matrix.columns.values
    y_values = [str(d) for d in z_matrix.index]
    pressure_lookup = df_grid['pressure']

    # --- C. PROCESS DOTS ---
    print("   ...Processing Dots")
    valid_stations = [s.strip().lower() for s in config['bike_stations']]

    def is_valid_station(series):
        return series.astype(str).str.strip().str.lower().isin(valid_stations)

    df_bikes_filtered = df_bikes[
        (is_valid_station(df_bikes['start_place']) | is_valid_station(df_bikes['end_place'])) &
        (df_bikes['start_time'] >= zoom_start) & (df_bikes['end_time'] <= zoom_end)
        ].copy()

    def get_dots(df, time_col, station_col, z_offset=0.2):
        subset = df[is_valid_station(df[station_col])].copy()
        x, y, z, t = [], [], [], []
        for _, row in subset.iterrows():
            ts = row[time_col]
            x.append(ts.hour + ts.minute / 60.0 + ts.second / 3600.0)
            y.append(str(ts.date()))

            lookup = ts.round('min')
            if lookup in pressure_lookup.index:
                base = pressure_lookup.loc[lookup]
            else:
                base = 0

            z.append(base + z_offset)
            t.append(ts.strftime('%H:%M'))
        return x, y, z, t

    rx, ry, rz, rt = get_dots(df_bikes_filtered, 'start_time', 'start_place')
    ex, ey, ez, et = get_dots(df_bikes_filtered, 'end_time', 'end_place')

    # --- D. PROCESS WEATHER ---
    print("   ...Building Weather Mask")
    rain_threshold = 0.2

    mask = (df_weather['ts'] >= zoom_start) & (df_weather['ts'] <= zoom_end)
    df_w_subset = df_weather[mask].copy()

    if not df_w_subset.empty:
        rain_series = df_w_subset.set_index('ts')['val'].sort_index()

        # DEBUG: Print Max Rain
        print(f"      -> Max Rain Detected in Data: {rain_series.max()} mm")

        # Resample to 1-Min grid
        rain_1min = rain_series.resample('1h').max().resample('1min').ffill()
        rain_aligned = rain_1min.reindex(df_grid.index, fill_value=0)

        # Pivot
        df_rain_grid = pd.DataFrame({'rain': rain_aligned})
        df_rain_grid['date'] = df_rain_grid.index.date
        df_rain_grid['time_dec'] = df_rain_grid.index.hour + df_rain_grid.index.minute / 60.0

        rain_matrix = df_rain_grid.pivot(index='date', columns='time_dec', values='rain')
        rain_matrix = rain_matrix.reindex(index=all_dates, columns=all_minutes, fill_value=0).fillna(0)

        is_raining = rain_matrix.values > rain_threshold

        rain_surface = np.full(z_values.shape, np.nan)
        # FORCE RAIN TO BE ABOVE MAX TERRAIN
        rain_z_height = 10.1
        rain_surface[is_raining] = rain_z_height

        print(f"      -> Painted {np.sum(is_raining)} rain minutes.")
    else:
        rain_surface = np.full(z_values.shape, np.nan)
        print("      -> No rain data in window.")

    # --- E. PLOT ---
    print("   ...Rendering")
    fig = go.Figure()

    # Terrain (Semi-Transparent)
    fig.add_trace(go.Surface(
        z=z_values, x=x_values, y=y_values,
        colorscale='Oranges',
        opacity=0.6,  # <--- TRANSPARENCY FIX
        name='Pressure',
        colorbar=dict(title='Pressure (Norm)', x=0, len=0.5)
    ))

    # Rain
    fig.add_trace(go.Surface(
        z=rain_surface, x=x_values, y=y_values,
        colorscale=[[0, '#87CEFA'], [1, '#00008B']],
        opacity=0.4, showscale=False, name=f'Rain > {rain_threshold}mm', hoverinfo='skip'
    ))

    # Dots
    fig.add_trace(
        go.Scatter3d(x=rx, y=ry, z=rz, mode='markers', marker=dict(size=1.5, color='#2ca02c'), name='Rental', text=rt))
    fig.add_trace(
        go.Scatter3d(x=ex, y=ey, z=ez, mode='markers', marker=dict(size=1.5, color='#1f77b4'), name='Return', text=et))

    fig.update_layout(
        title=f"3D Commuter Swarm: {city_name}",
        scene=dict(
            xaxis_title='Time (Hour)',
            yaxis_title='Date',
            zaxis_title='Intensity (Norm)',
            xaxis=dict(tickvals=list(range(0, 25, 4))),
            aspectmode='manual',
            aspectratio=dict(x=1, y=3.0, z=0.6)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=900
    )

    if not os.path.exists('results'): os.makedirs('results')
    filename = f"results/3D_Swarm_{city_name.replace(' ', '_').replace('.', '')}.html"
    fig.write_html(filename)
    print(f"   ✅ Saved to {filename}")


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    for city, cfg in NETWORKS.items():
        generate_3d_swarm(city, cfg)