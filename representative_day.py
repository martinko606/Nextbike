import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.transforms import ScaledTranslation
import numpy as np
import os
import warnings
import sys

# === CRITICAL IMPORT ===
# This ensures we use the EXACT same math as the main analysis
import transport_analytics as ta

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION (Must match run_city_analysis.py)
# ==========================================
CITIES = {
    'Ostrava-Svinov': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava-Svinov',
        'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
        'params': {
            'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1},
            'pulse_arr_reg': 15, 'pulse_dep_reg': 20,
            'pulse_arr_ld': 20, 'pulse_dep_ld': 35,
            'rush_am': 2.0, 'rush_pm': 1.8
        },
    },
    'Ostrava hl.n.': {
        'file_trains': 'data/Pohyby_Ostrava_hl.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava hl.n.',
        'bike_stations': ['MOAP-Hlavní nádraží'],
        'params': {
            'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1},
            'pulse_arr_reg': 25, 'pulse_dep_reg': 25,
            'pulse_arr_ld': 30, 'pulse_dep_ld': 35,
            'rush_am': 2.0, 'rush_pm': 1.7
        },
    },
    'Brno hl.n.': {
        'file_trains': 'data/Pohyby_Brno.xlsx',
        'file_weather': 'data/Brno_pocasi.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_rentals': 'Vypujcky_Brno',
        'train_station': 'Brno hl.n.',
        'bike_stations': ['Hlavní nádraží - Hlavní vstup', 'Hlavní nádraží - pošta', 'Bajkazyl 666'],
        'params': {
            'weights': {'Os': 2.5, 'Sp': 2.0, 'R': 1.5, 'Ex': 1},
            'pulse_arr_reg': 12, 'pulse_dep_reg': 25,
            'pulse_arr_ld': 30, 'pulse_dep_ld': 35,
            'rush_am': 2.5, 'rush_pm': 2.0
        },
    },
    'Přerov': {
        'file_trains': 'data/Pohyby_Prerov.xlsx',
        'file_weather': 'data/Prerov_pocasi.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_rentals': 'Vypujcky_Prerov',
        'train_station': 'Přerov os.n.',
        'bike_stations': ['Nádraží'],
        'params': {
            'weights': {'Os': 1.7, 'Sp': 1.4, 'R': 1.2, 'Ex': 1},
            'pulse_arr_reg': 12, 'pulse_dep_reg': 20,
            'pulse_arr_ld': 15, 'pulse_dep_ld': 30,
            'rush_am': 1.7, 'rush_pm': 1.4
        },
    },
    'Valašské Meziříčí': {
        'file_trains': 'data/Pohyby_ValMez.xlsx',
        'file_weather': 'data/ValMez_pocasi.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_rentals': 'Vypujcky_ValMez',
        'train_station': 'Valašské Meziříčí',
        'bike_stations': ['Vlakové nádraží Valašské Meziříčí (nové umístění)'],
        'params': {
            'weights': {'Os': 1.8, 'Sp': 1.6, 'R': 1.4, 'Ex': 1.6},
            'pulse_arr_reg': 12, 'pulse_dep_reg': 20,
            'pulse_arr_ld': 20, 'pulse_dep_ld': 25,
            'rush_am': 1.8, 'rush_pm': 1.4
        },
    },
}

SAVE_FOLDER = 'results/paper_graphs_highres'
if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)


# ==========================================
# 2. DATA LOADING (Using TA module)
# ==========================================
def load_data_via_ta(cfg):
    """
    Uses pandas to load raw files, then passes them to TA for standardization.
    Identical flow to run_city_analysis.
    """
    # 1. TRAINS
    try:
        df_p = pd.read_excel(cfg['file_trains'], header=None, nrows=20)
        h_idx = 0
        for i, row in df_p.iterrows():
            if any('čas' in str(x).lower() for x in row.values): h_idx = i; break
        df_trains = pd.read_excel(cfg['file_trains'], header=h_idx)
    except:
        return None, None, None

    # 2. BIKES
    try:
        df_bikes = pd.read_excel(cfg['file_bikes'], sheet_name=cfg['sheet_rentals'])
        # Pre-clean for TA compatibility
        df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]
        df_bikes['start_time'] = pd.to_datetime(df_bikes['start_time'])
        df_bikes['end_time'] = pd.to_datetime(df_bikes['end_time'])
    except:
        return None, None, None

    # 3. WEATHER
    try:
        df_weather = pd.read_excel(cfg['file_weather'])
    except:
        df_weather = None

    return df_trains, df_bikes, df_weather


# ==========================================
# 3. MAIN LOGIC (Imported Calculation)
# ==========================================
def find_representative_day(city_name, cfg):
    print(f"\n🔍 Processing: {city_name}...")

    # A. Load Raw Data
    df_t_raw, df_b_raw, df_w_raw = load_data_via_ta(cfg)
    if df_t_raw is None: return

    # B. Standardize & Filter (USING TA MODULE)
    df_t = ta.standard_columns(df_t_raw)

    # Filter Station
    if 'station' in df_t.columns:
        df_t = df_t[df_t['station'].astype(str).str.contains(cfg['train_station'], case=False, na=False)]

    # Define Timeline (Global)
    t_min = min(df_t['actual_arrival'].min(), df_b_raw['start_time'].min())
    t_max = max(df_t['actual_arrival'].max(), df_b_raw['end_time'].max())
    global_line = pd.date_range(t_min.floor('h'), t_max.ceil('h'), freq='1min')

    print("   🌍 Calculating Global Correlation (Exact TA Logic)...")

    # C. Generate Signals (USING TA MODULE)
    s_arr, s_dep = ta.generate_signals(df_t, global_line, cfg['params'])
    s_total_global = s_arr + s_dep

    # D. Generate Bike Counts (Manual aggregate to match TA)
    valid_st = [s.lower() for s in cfg['bike_stations']]
    starts = df_b_raw[df_b_raw['start_place'].str.lower().isin(valid_st)]
    ends = df_b_raw[df_b_raw['end_place'].str.lower().isin(valid_st)]

    c_rent = starts.set_index('start_time').resample('1min').size().reindex(global_line, fill_value=0)
    c_ret = ends.set_index('end_time').resample('1min').size().reindex(global_line, fill_value=0)
    c_total_global = c_rent + c_ret

    # E. Calculate Target (Weather Masking)
    mask_available = pd.Series(True, index=global_line)
    obs_weights = pd.Series(1.0, index=global_line)

    if df_w_raw is not None:
        try:
            df_w_raw.columns = [str(c).lower().strip() for c in df_w_raw.columns]
            t_col = next((c for c in df_w_raw.columns if c in ['datetime', 'date', 'cas']), None)
            p_col = next((c for c in df_w_raw.columns if c in ['precip', 'rain', 'srazky']), None)
            if t_col and p_col:
                df_w_raw['ts'] = pd.to_datetime(df_w_raw[t_col]).dt.round('1min')
                w_res = df_w_raw.set_index('ts').resample('1min').ffill().reindex(global_line, method='ffill')
                mask_bad = (w_res[p_col] > 0.2)
                obs_weights[mask_bad] = 0.5
        except:
            pass

    valid_mask = mask_available

    target_corr = ta.weighted_pearson_corr(
        s_total_global[valid_mask].values,
        c_total_global[valid_mask].values,
        obs_weights[valid_mask].values
    )

    print(f"      -> Global Target Correlation: {target_corr:.6f}")

    # ---------------------------------------------------------
    # 2. SEARCH FOR MATCHING DAY
    # ---------------------------------------------------------
    print("   📊 Scanning days for best match...")
    unique_days = df_t['actual_arrival'].dt.date.unique()
    unique_days = sorted([d for d in unique_days if pd.notnull(d)])

    best_day = None
    min_diff = float('inf')
    best_day_corr = 0

    for d in unique_days:
        d_start = pd.Timestamp(d)
        d_end = d_start + pd.Timedelta(hours=23, minutes=59)

        day_mask = (global_line >= d_start) & (global_line <= d_end)
        if day_mask.sum() < 1000: continue

        s_slice = s_total_global[day_mask]
        c_slice = c_total_global[day_mask]
        w_slice = obs_weights[day_mask]

        if s_slice.sum() == 0 or c_slice.sum() == 0: continue

        day_corr = ta.weighted_pearson_corr(s_slice.values, c_slice.values, w_slice.values)

        if pd.isna(day_corr): continue

        diff = abs(day_corr - target_corr)
        if diff < min_diff:
            min_diff = diff
            best_day = d
            best_day_corr = day_corr

    print(f"      -> Selected Day: {best_day} (r={best_day_corr:.4f} | Diff: {min_diff:.6f})")

    # ---------------------------------------------------------
    # 3. VISUALIZE
    # ---------------------------------------------------------
    plot_representative_day(city_name, best_day, s_total_global, c_rent, c_ret, df_w_raw, df_t, best_day_corr,
                            target_corr)


def plot_representative_day(city_name, date, s_total, c_rent, c_ret, df_w, df_t, day_corr, global_corr):
    print(f"   🎨 Generating Plots (With and Without Headings)...")

    start_plot = pd.Timestamp(date) + pd.Timedelta(hours=0)
    end_plot = pd.Timestamp(date) + pd.Timedelta(hours=23, minutes=59)

    # 1. Slice Signals
    plot_mask = (s_total.index >= start_plot) & (s_total.index <= end_plot)
    s_plot = s_total[plot_mask]
    rent_plot = c_rent[plot_mask]
    ret_plot = c_ret[plot_mask]

    # 2. Slice Trains (Independent Filtering)
    mask_arr = (df_t['actual_arrival'] >= start_plot) & (df_t['actual_arrival'] <= end_plot)
    arrs = df_t.loc[mask_arr & pd.notnull(df_t['actual_arrival']), 'actual_arrival']

    mask_dep = (df_t['actual_departure'] >= start_plot) & (df_t['actual_departure'] <= end_plot)
    deps = df_t.loc[mask_dep & pd.notnull(df_t['actual_departure']), 'actual_departure']

    # 3. Weather Processing
    w_rain, w_temp = pd.DataFrame(), pd.DataFrame()
    if df_w is not None and not df_w.empty:
        try:
            t_col = next((c for c in df_w.columns if c in ['datetime', 'date', 'cas']), None)
            p_col = next((c for c in df_w.columns if c in ['precip', 'rain', 'srazky']), None)
            tm_col = next((c for c in df_w.columns if c in ['temp', 'teplota']), None)

            if t_col:
                df_w['ts'] = pd.to_datetime(df_w[t_col]).dt.round('1min')
                mask_w = (df_w['ts'] >= start_plot) & (df_w['ts'] <= end_plot)
                w_cut = df_w.loc[mask_w].set_index('ts')

                if p_col in w_cut.columns: w_rain = w_cut[p_col].resample('15min').mean()
                if tm_col in w_cut.columns: w_temp = w_cut[tm_col].resample('1h').first()
        except:
            pass

    # --- PLOTTING ---
    fig, (ax_w, ax_main) = plt.subplots(2, 1, figsize=(24, 10), sharex=True, gridspec_kw={'height_ratios': [1, 4]})

    # A. Weather Panel
    if not w_rain.empty:
        ax_w.bar(w_rain.index, w_rain.values, color='#4682B4', width=0.007, label='Rain')
        ax_w.set_ylabel('Rain (mm)', color='#4682B4', fontsize=16)
        ax_w.tick_params(axis='y', labelsize=14)
        ax_w.set_ylim(0, max(w_rain.max() * 1.5, 0.5))
    if not w_temp.empty:
        ax_wt = ax_w.twinx()
        ax_wt.plot(w_temp.index, w_temp.values, color='#DC143C', marker='o', alpha=0.5, ls=':')
        ax_wt.set_ylabel('Temp (°C)', color='#DC143C', fontsize=16)
        ax_wt.tick_params(axis='y', labelsize=16)

    # B. Main Panel
    x = s_plot.index

    # 1. Total Pressure
    ax_main.fill_between(x, s_plot, color='#FF8C00', alpha=0.15, label='Passenger Pressure')
    ax_main.plot(x, s_plot, color='#FF8C00', lw=2)

    # 2. Bike Events
    ax_b = ax_main.twinx()
    ax_main.set_ylim(bottom=0)
    ax_b.set_ylim(bottom=0)

    if not rent_plot.empty:
        ax_b.bar(rent_plot.index, rent_plot.values, width=0.002, color='#2ca02c', alpha=0.6, label='Rentals', zorder=1)
    if not ret_plot.empty:
        ax_b.bar(ret_plot.index, ret_plot.values, width=0.002, color='#1f77b4', alpha=0.6, label='Returns', zorder=1)

    ax_b.set_ylim(0, 10)
    ax_b.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_b.set_ylabel("Bike Events", color='#333333', fontweight='bold')

    # 3. Train Markers (Fixed on Axis)
    trans_up = ScaledTranslation(0, 10 / 72, fig.dpi_scale_trans)
    trans_down = ScaledTranslation(0, -10 / 72, fig.dpi_scale_trans)

    ax_main.scatter(deps, [0] * len(deps), marker='^', color='#D62728', s=80,
                    zorder=10, label='Departure', clip_on=False, transform=ax_main.transData + trans_up)

    ax_main.scatter(arrs, [0] * len(arrs), marker='v', color='black', s=80,
                    zorder=10, label='Arrival', clip_on=False, transform=ax_main.transData + trans_down)

    # 4. Formatting
    ax_main.set_xlim(start_plot, end_plot)
    ax_main.set_ylabel("Passenger Pressure", color='#FF8C00', fontweight='bold', fontsize=18)
    ax_main.tick_params(axis='both', which='major', labelsize=16, pad=15)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    lines_1, labels_1 = ax_main.get_legend_handles_labels()
    lines_2, labels_2 = ax_b.get_legend_handles_labels()
    by_label = dict(zip(labels_1 + labels_2, lines_1 + lines_2))
    ax_main.legend(by_label.values(), by_label.keys(), loc='upper left', ncol=5, frameon=True, fontsize=18)

    safe = city_name.replace(' ', '_').replace('.', '')

    # ==========================================
    # EXPORT 1: "NO HEADING" VERSION (For Paper)
    # ==========================================
    plt.tight_layout()  # Standard layout with no extra top room

    png_path_clean = f"{SAVE_FOLDER}/Heartbeat_MatchGlobal_{safe}_NoHeading.png"
    svg_path_clean = f"{SAVE_FOLDER}/Heartbeat_MatchGlobal_{safe}_NoHeading.svg"

    plt.savefig(png_path_clean, dpi=600, bbox_inches='tight')
    plt.savefig(svg_path_clean, format='svg', bbox_inches='tight')
    print(f"   ✅ Saved No-Heading PNG: {png_path_clean}")

    # ==========================================
    # EXPORT 2: "WITH HEADING" VERSION (For Reference)
    # ==========================================
    # Now we add the title dynamically without redrawing the whole graph
    plt.suptitle(
        f"Synchronized Transport Heartbeat: {city_name}\nDate: {date} (Day r={day_corr:.3f} | Global Target r={global_corr:.3f})",
        fontweight='bold', fontsize=22)

    # Adjust layout to make room for the newly added suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path_titled = f"{SAVE_FOLDER}/Heartbeat_MatchGlobal_{safe}_Titled.png"

    plt.savefig(png_path_titled, dpi=600, bbox_inches='tight')
    print(f"   ✅ Saved Titled PNG: {png_path_titled}")

    # Close figure to free memory
    plt.close()


if __name__ == "__main__":
    for city, cfg in CITIES.items():
        find_representative_day(city, cfg)