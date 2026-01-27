import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
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

SAVE_FOLDER = 'results/paper_graphs_highres'
if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_and_process(cfg):
    # --- A. TRAINS ---
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
                station_col = df_t['station'].iloc[:, 0]
            else:
                station_col = df_t['station']
            df_t = df_t[station_col.astype(str).str.contains(cfg['train_station'], case=False, na=False)]

        df_t['arr'] = pd.to_datetime(df_t['arr'], errors='coerce')
        df_t['dep'] = pd.to_datetime(df_t['dep'], errors='coerce')

    except Exception as e:
        print(f"   ❌ Error loading Trains: {e}")
        return None, None, None

    # --- B. BIKES ---
    try:
        df_b = pd.read_excel(cfg['file_bikes'], sheet_name=cfg['sheet_rentals'])
        df_b.columns = [str(c).lower().strip() for c in df_b.columns]

        valid_stats = [s.lower() for s in cfg['bike_stations']]
        mask_start = df_b['start_place'].astype(str).str.lower().isin(valid_stats)
        mask_end = df_b['end_place'].astype(str).str.lower().isin(valid_stats)
        df_b = df_b[mask_start | mask_end].copy()

        df_b['start_time'] = pd.to_datetime(df_b['start_time'])
        df_b['end_time'] = pd.to_datetime(df_b['end_time'])
    except Exception as e:
        print(f"   ❌ Error loading Bikes: {e}")
        return None, None, None

    # --- C. WEATHER ---
    try:
        df_w = pd.read_excel(cfg['file_weather'])
        df_w.columns = [str(c).lower().strip() for c in df_w.columns]

        t_col = next((c for c in df_w.columns if any(x in c for x in ['date', 'time', 'cas'])), None)
        p_col = next((c for c in df_w.columns if any(x in c for x in ['rain', 'precip', 'uhrn'])), None)
        temp_col = next((c for c in df_w.columns if any(x in c for x in ['temp', 'teplota'])), None)

        if t_col:
            df_w['ts'] = pd.to_datetime(df_w[t_col])
        if p_col:
            df_w['rain'] = pd.to_numeric(df_w[p_col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        else:
            df_w['rain'] = 0
        if temp_col:
            df_w['temp'] = pd.to_numeric(df_w[temp_col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df_w['temp'] = np.nan
    except Exception as e:
        print(f"   ⚠️ Warning: Weather load failed ({e}).")
        df_w = pd.DataFrame({'ts': [], 'rain': [], 'temp': []})

    return df_t, df_b, df_w


def generate_signals(df_t, df_b, date_start, date_end):
    timeline = pd.date_range(date_start, date_end, freq='1min')

    # 1. Train Pressure
    s_press = pd.Series(0.0, index=timeline)
    w_map = {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1.0, 'IC': 1.0}

    mask_t = (df_t['arr'] >= date_start) & (df_t['arr'] <= date_end)
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

    window = 30
    gauss = gaussian(window, std=4)
    gauss /= gauss.sum()
    s_press_smooth = np.convolve(s_press.values, gauss, mode='same')

    # 2. Bike Movement
    b_start = df_b[(df_b['start_time'] >= date_start) & (df_b['start_time'] <= date_end)]
    b_end = df_b[(df_b['end_time'] >= date_start) & (df_b['end_time'] <= date_end)]

    counts_s = b_start.set_index('start_time').resample('1min').size()
    counts_e = b_end.set_index('end_time').resample('1min').size()

    return pd.Series(s_press_smooth, index=timeline), counts_s, counts_e


# ==========================================
# 3. MAIN LOOP
# ==========================================
def analyze_city_heartbeat(city_name, cfg):
    print(f"\n🔍 Processing: {city_name}...")

    df_t, df_b, df_w = load_and_process(cfg)
    if df_t is None or df_b is None: return

    # --- A. Find Best Day ---
    print("   📊 Finding Representative Day...")
    unique_days = df_t['arr'].dt.date.unique()
    unique_days = sorted([d for d in unique_days if pd.notnull(d)])

    daily_corrs = []

    for d in unique_days:
        d_start = pd.Timestamp(d)
        d_end = d_start + pd.Timedelta(hours=23, minutes=59)

        try:
            sig_p, c_s, c_e = generate_signals(df_t, df_b, d_start, d_end)
            sig_b = c_s.reindex(sig_p.index, fill_value=0) + c_e.reindex(sig_p.index, fill_value=0)

            if sig_p.sum() > 5 and sig_b.sum() > 5:
                corr = sig_p.corr(sig_b)
                if not pd.isna(corr):
                    daily_corrs.append({'date': d, 'corr': corr})
        except:
            continue

    if not daily_corrs:
        print("   ❌ No valid days found.")
        return

    df_corr = pd.DataFrame(daily_corrs)
    avg_corr = df_corr['corr'].mean()

    best_day_row = df_corr.iloc[(df_corr['corr'] - avg_corr).abs().argsort()[:1]].iloc[0]
    best_date = best_day_row['date']

    print(f"      -> Best Day: {best_date}")
    print(f"      -> Day Corr: {best_day_row['corr']:.3f} | Global Avg Corr: {avg_corr:.3f}")

    # --- B. Generate Plot ---
    print(f"   🎨 Drawing High-Res Heartbeat...")

    start_plot = pd.Timestamp(best_date) + pd.Timedelta(hours=4)
    end_plot = pd.Timestamp(best_date) + pd.Timedelta(hours=23, minutes=59)

    sig_p, c_rent, c_ret = generate_signals(df_t, df_b, start_plot, end_plot)

    # --- PROCESS WEATHER (SEPARATED) ---
    w_rain = pd.DataFrame()
    w_temp = pd.DataFrame()

    if not df_w.empty:
        mask_w = (df_w['ts'] >= start_plot) & (df_w['ts'] <= end_plot)
        w_cut = df_w.loc[mask_w].set_index('ts')

        # 1. Rain (Resample to 15min bars)
        if 'rain' in w_cut.columns:
            w_rain = w_cut['rain'].resample('15min').mean()

        # 2. Temp (Keep Hourly Points)
        if 'temp' in w_cut.columns:
            # Resample to 1H to get exactly one point per hour
            w_temp = w_cut['temp'].resample('1h').first()

    # --- PLOTTING ---
    fig, (ax_w, ax_main) = plt.subplots(2, 1, figsize=(24, 10), sharex=True, gridspec_kw={'height_ratios': [1, 4]})

    # 1. Weather (Top)
    # Plot Rain Bars
    if not w_rain.empty:
        ax_w.bar(w_rain.index, w_rain.values, color='#4682B4', alpha=0.6, width=0.007, label='Rain')
        ax_w.set_ylabel('Rain (mm)', color='#4682B4')
        ax_w.set_ylim(0, max(w_rain.max() * 1.5, 0.5))
    else:
        ax_w.text(0.5, 0.5, "No Rain Data", ha='center', transform=ax_w.transAxes)

    # Plot Temp Points
    if not w_temp.empty:
        ax_wt = ax_w.twinx()
        # Plot as Red Dots
        ax_wt.scatter(w_temp.index, w_temp.values, color='#DC143C', s=50, zorder=5, label='Temp')
        ax_wt.set_ylabel('Temp (°C)', color='#DC143C')

        # Optional: Connect dots faintly if desired, remove ls=':' if you want pure dots
        ax_wt.plot(w_temp.index, w_temp.values, color='#DC143C', alpha=0.3, lw=1, ls=':')
    else:
        pass  # No temp data

    # 2. Train Pressure
    x = sig_p.index
    ax_main.fill_between(x, sig_p, color='#FF8C00', alpha=0.15, label='Aggregated Passenger Pressure')
    ax_main.plot(x, sig_p, color='#FF8C00', lw=2)

    # 3. Bike Events
    ax_b = ax_main.twinx()

    if not c_rent.empty:
        ax_b.bar(c_rent.index, c_rent.values, width=0.0007, color='#2ca02c', alpha=0.8, label='Rental Event')

    if not c_ret.empty:
        ax_b.bar(c_ret.index, c_ret.values, width=0.0007, color='#1f77b4', alpha=0.8, label='Return Event')

    # Fix: Limit to 10 & Integer
    ax_b.set_ylabel("Bike Events (Count/min)", color='#333333', fontweight='bold')
    ax_b.set_ylim(0, 10)
    ax_b.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 4. Train Markers
    mask_ev = (df_t['arr'] >= start_plot) & (df_t['arr'] <= end_plot)
    t_events = df_t[mask_ev]
    max_p = sig_p.max() if sig_p.max() > 0 else 1
    y_offset = max_p * 0.02

    arrs = t_events[pd.notnull(t_events['arr'])]
    deps = t_events[pd.notnull(t_events['dep'])]

    ax_main.scatter(arrs['arr'], [y_offset] * len(arrs), marker='v', color='black', s=40, zorder=10, label='Arrival')
    ax_main.scatter(deps['dep'], [y_offset] * len(deps), marker='^', color='#D62728', s=40, zorder=10,
                    label='Departure')

    # Styles
    ax_main.set_ylabel("Passenger Pressure (Smoothed)", color='#FF8C00', fontweight='bold')
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_main.set_xlim(start_plot, end_plot)
    ax_main.set_ylim(bottom=0)
    ax_main.grid(True, linestyle=':', alpha=0.6)

    # Legend
    lines1, labels1 = ax_main.get_legend_handles_labels()
    lines2, labels2 = ax_b.get_legend_handles_labels()
    by_label = dict(zip(labels1 + labels2, lines1 + lines2))
    ax_main.legend(by_label.values(), by_label.keys(), loc='upper left', ncol=5, frameon=True)

    plt.suptitle(
        f"Synchronized Transport Heartbeat: {city_name}\nDate: {best_date} (Day Corr: {best_day_row['corr']:.3f} | Global Avg: {avg_corr:.3f})",
        fontweight='bold')

    safe_name = city_name.replace(' ', '_').replace('.', '')
    plt.savefig(f"{SAVE_FOLDER}/Heartbeat_HighRes_{safe_name}.png", dpi=300)
    plt.close()
    print(f"   ✅ Saved Heartbeat_HighRes_{safe_name}.png")


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    for city, cfg in CITIES.items():
        analyze_city_heartbeat(city, cfg)