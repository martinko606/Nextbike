import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches  # <--- FIXED IMPORT
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

SAVE_FOLDER = 'results/2D_heatmaps'
if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)


# ==========================================
# 2. DATA LOADING
# ==========================================
def load_data(cfg):
    try:
        # TRAINS
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

        # BIKES
        df_b = pd.read_excel(cfg['file_bikes'], sheet_name=cfg['sheet_rentals'])
        df_b.columns = [str(c).lower().strip() for c in df_b.columns]
        valid = [s.lower() for s in cfg['bike_stations']]
        mask = df_b['start_place'].astype(str).str.lower().isin(valid) | \
               df_b['end_place'].astype(str).str.lower().isin(valid)
        df_b = df_b[mask].copy()
        df_b['start_time'] = pd.to_datetime(df_b['start_time'])
        df_b['end_time'] = pd.to_datetime(df_b['end_time'])

        # WEATHER
        try:
            df_w = pd.read_excel(cfg['file_weather'])
            df_w.columns = [str(c).lower().strip() for c in df_w.columns]
            t_col = next((c for c in df_w.columns if any(x in c for x in ['date', 'time', 'cas'])), None)
            p_col = next((c for c in df_w.columns if any(x in c for x in ['rain', 'precip', 'uhrn'])), None)
            if t_col: df_w['ts'] = pd.to_datetime(df_w[t_col])
            if p_col:
                df_w['rain'] = pd.to_numeric(df_w[p_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
            else:
                df_w['rain'] = 0
        except:
            df_w = pd.DataFrame()

        return df_t, df_b, df_w
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


# ==========================================
# 3. 2D RASTER GENERATOR
# ==========================================
def generate_2d_raster(city, cfg):
    print(f"\n🚀 Generating 2D Raster for: {city}...")

    df_t, df_b, df_w = load_data(cfg)
    if df_t is None: return

    zoom_start = pd.to_datetime(VIS_START)
    zoom_end = pd.to_datetime(VIS_END)

    # 1. Timeline & Grid
    timeline_1min = pd.date_range(zoom_start, zoom_end, freq='1min')

    # 2. Pressure Calc
    print("   ...Calculating Pressure")
    s_press = pd.Series(0.0, index=timeline_1min)
    w_map = {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1.0, 'IC': 1.0}

    mask_t = (df_t['arr'] >= zoom_start) & (df_t['arr'] <= zoom_end)
    for _, r in df_t[mask_t].iterrows():
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

    # Normalize
    s_norm = s_smooth / s_smooth.max() if s_smooth.max() > 0 else s_smooth

    # 3. Prepare Heatmap Matrix
    print("   ...Building Matrix")
    # Resample to 5-min for display performance
    s_res = s_norm.resample('5min').mean()

    df_grid = pd.DataFrame({'pressure': s_res})
    df_grid['date'] = df_grid.index.date
    df_grid['time_dec'] = df_grid.index.hour + df_grid.index.minute / 60.0

    pivot_press = df_grid.pivot(index='date', columns='time_dec', values='pressure').fillna(0)

    # Force full ranges
    all_dates = pd.date_range(zoom_start.date(), zoom_end.date()).date
    all_times = np.arange(0, 24, 5 / 60.0)  # 5 min steps

    pivot_press = pivot_press.reindex(index=all_dates).reindex(columns=all_times).fillna(0)

    # 4. Prepare Rain Overlay
    rain_mask = np.zeros_like(pivot_press.values)
    if not df_w.empty:
        w_cut = df_w[(df_w['ts'] >= zoom_start) & (df_w['ts'] <= zoom_end)]
        if not w_cut.empty:
            w_series = w_cut.set_index('ts')['rain'].resample('5min').max()
            df_r = pd.DataFrame({'rain': w_series})
            df_r['date'] = df_r.index.date
            df_r['time_dec'] = df_r.index.hour + df_r.index.minute / 60.0

            piv_rain = df_r.pivot(index='date', columns='time_dec', values='rain')
            piv_rain = piv_rain.reindex(index=all_dates).reindex(columns=all_times).fillna(0)

            rain_mask = np.where(piv_rain.values > 0.1, 1, 0)

    # 5. Prepare Bike Dots
    print("   ...Mapping Dots")
    valid_st = [s.lower() for s in cfg['bike_stations']]

    def get_coords(df, time_col, station_col):
        mask = df[station_col].astype(str).str.lower().isin(valid_st) & \
               (df[time_col] >= zoom_start) & (df[time_col] <= zoom_end)
        sub = df[mask]

        x = sub[time_col].dt.hour + sub[time_col].dt.minute / 60.0
        date_map = {d: i for i, d in enumerate(all_dates)}
        y = sub[time_col].dt.date.map(date_map)

        valid = y.notna()
        return x[valid], y[valid]

    rx, ry = get_coords(df_b, 'start_time', 'start_place')
    ex, ey = get_coords(df_b, 'end_time', 'end_place')

    # 6. PLOTTING
    print("   ...Rendering")
    fig, ax = plt.subplots(figsize=(15, len(all_dates) * 0.3 + 2))

    # Heatmap
    im = ax.imshow(pivot_press.values, aspect='auto', cmap='Oranges',
                   extent=[0, 24, len(all_dates), 0], vmin=0, vmax=1)

    # Rain Overlay
    rain_img = np.zeros((rain_mask.shape[0], rain_mask.shape[1], 4))
    rain_img[rain_mask == 1] = [0, 0, 0.8, 0.3]
    ax.imshow(rain_img, aspect='auto', extent=[0, 24, len(all_dates), 0])

    # Dots (Centered on rows)
    ax.scatter(rx, ry + 0.5, s=15, color='#2ca02c', label='Rentals', edgecolors='white', linewidth=0.3)
    ax.scatter(ex, ey + 0.5, s=15, color='#1f77b4', label='Returns', edgecolors='white', linewidth=0.3)

    # Formatting
    ax.set_xlim(0, 24)
    ax.set_ylim(len(all_dates), 0)

    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.set_xlabel("Time of Day")

    ax.set_yticks(np.arange(len(all_dates)) + 0.5)
    ax.set_yticklabels([d.strftime('%a %d.%m') for d in all_dates], fontsize=9)

    ax.grid(which='major', axis='x', linestyle=':', alpha=0.3, color='black')
    ax.vlines(range(24), 0, len(all_dates), colors='black', linestyles=':', alpha=0.1)

    plt.title(f"Transport Rhythm: {city}\nOrange=Train Pressure | Blue Zones=Rain | Dots=Bikes",
              fontsize=14, fontweight='bold', pad=15)

    # Custom Legend (FIXED)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', label='Bike Rental'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label='Bike Return'),
        mpatches.Patch(facecolor='orange', label='High Train Traffic'),  # <--- USING mpatches
        mpatches.Patch(facecolor='blue', alpha=0.3, label='Rain Event'),  # <--- USING mpatches
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.tight_layout()
    fname = f"{SAVE_FOLDER}/2D_Raster_{city.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=200)
    print(f"   ✅ Saved {fname}")
    plt.close()


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    for city, cfg in CITIES.items():
        generate_2d_raster(city, cfg)