import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal.windows import gaussian
import os
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
CITIES = {
    'Ostrava-Svinov': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'train_station': 'Ostrava-Svinov',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)']
    },
    'Ostrava hl.n.': {
        'file_trains': 'data/Pohyby_Ostrava_hl.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'train_station': 'Ostrava hl.n.',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'bike_stations': ['MOAP-Hlavní nádraží']
    },
    'Brno': {
        'file_trains': 'data/Pohyby_Brno.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'train_station': 'Brno hl.n.',
        'sheet_rentals': 'Vypujcky_Brno',
        'bike_stations': ['Hlavní nádraží - Hlavní vstup', 'Hlavní nádraží - pošta', 'Bajkazyl 666']
    },
    'Prerov': {
        'file_trains': 'data/Pohyby_Prerov.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'train_station': 'Přerov os.n.',
        'sheet_rentals': 'Vypujcky_Prerov',
        'bike_stations': ['Nádraží']
    },
    'ValMez': {
        'file_trains': 'data/Pohyby_ValMez.xlsx',
        'file_bikes': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'train_station': 'Valašské Meziříčí',
        'sheet_rentals': 'Vypujcky_ValMez',
        'bike_stations': ['Vlakové nádraží Valašské Meziříčí (nové umístění)']
    }
}

OUTPUT_DIR = 'results/correlation_1min'
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_and_process(cfg):
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
            elif 'bod' in cl or 'stanice' in cl:
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
        print(f"   ❌ Train Load Error: {e}")
        return None, None

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
        return None, None

    return df_t, df_b


# ==========================================
# 3. ANALYSIS LOOP (1-MINUTE PRECISION)
# ==========================================
results = []
print("Starting High-Precision Analysis (1-Min Window)...")

# Define Exact Window
start = pd.to_datetime('2025-09-15 00:00')
end = pd.to_datetime('2025-10-19 23:59')
# This is now our master timeline: 1-minute steps
timeline_1min = pd.date_range(start, end, freq='1min')

for city, cfg in CITIES.items():
    print(f"Processing {city}...")
    df_t, df_b = load_and_process(cfg)

    if df_t is None or df_b is None or df_t.empty or df_b.empty:
        print(f"   Skipping {city} (Data Missing)")
        continue

    # --- 1. TRAIN PRESSURE (1-min) ---
    # We maintain 1-minute resolution throughout. No resampling.
    s_arr = pd.Series(0.0, index=timeline_1min)
    s_dep = pd.Series(0.0, index=timeline_1min)
    w_map = {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1.0, 'IC': 1.0}

    mask_t = (df_t['arr'] >= start) & (df_t['arr'] <= end)
    for _, r in df_t[mask_t].iterrows():
        w = w_map.get(r.get('type', 'Os'), 1.0)
        if pd.notnull(r['arr']):
            try:
                s_arr.at[r['arr'].round('min')] += w
            except:
                pass
        if pd.notnull(r['dep']):
            try:
                s_dep.at[r['dep'].round('min')] += w
            except:
                pass

    # Gaussian Smoothing (Essential for 1-min correlation)
    # Without this, matching a bike at 10:02 to a train at 10:00 would result in 0 correlation.
    gauss = gaussian(30, std=4)
    gauss /= gauss.sum()

    press_arr = pd.Series(np.convolve(s_arr.values, gauss, mode='same'), index=timeline_1min)
    press_dep = pd.Series(np.convolve(s_dep.values, gauss, mode='same'), index=timeline_1min)

    # --- 2. BIKE COUNTS (1-min) ---
    valid_st = [s.lower() for s in cfg['bike_stations']]

    # Rentals
    b_rent = df_b[df_b['start_place'].astype(str).str.lower().isin(valid_st)]
    # We use .size() on 1min resample. Result is sparse (mostly 0s, some 1s)
    cnt_rent = b_rent.set_index('start_time').resample('1min').size().reindex(timeline_1min, fill_value=0)

    # Returns
    b_ret = df_b[df_b['end_place'].astype(str).str.lower().isin(valid_st)]
    cnt_ret = b_ret.set_index('end_time').resample('1min').size().reindex(timeline_1min, fill_value=0)

    # --- 3. GLOBAL (1-min) ---
    press_global = press_arr + press_dep
    move_global = cnt_rent + cnt_ret

    # --- 4. CORRELATION (1-min) ---
    # Correlating continuous curve vs sparse spikes
    c1 = press_arr.corr(cnt_rent)
    c2 = press_dep.corr(cnt_ret)
    c3 = press_global.corr(move_global)

    results.append({
        'City': city,
        'Arrivals vs Rentals': c1,
        'Departures vs Returns': c2,
        'Global Pressure vs Movement': c3
    })
    print(f"   -> Global Corr (1-min): {c3:.3f}")

# ==========================================
# 4. PLOTTING
# ==========================================
if results:
    df_res = pd.DataFrame(results)

    # Save CSV
    df_res.to_csv(f"{OUTPUT_DIR}/Correlation_Table_1min.csv", index=False)

    # Melt
    df_melt = df_res.melt(id_vars='City', var_name='Correlation Type', value_name='Pearson r')

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(data=df_melt, x='City', y='Pearson r', hue='Correlation Type', palette='viridis')

    plt.title('Correlation Analysis (High-Precision 1-Minute Resolution)', fontsize=16, fontweight='bold')
    plt.ylabel('Pearson Correlation Coefficient (r)', fontsize=12)
    plt.xlabel('')
    plt.ylim(0, 0.3)  # 1-min correlations are typically much lower, scale Y accordingly (0.3 is usually plenty)
    plt.legend(title='Interaction Type')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Correlation_Summary_1min.png", dpi=300)
    print(f"\n✅ Saved chart to {OUTPUT_DIR}/Correlation_Summary_1min.png")
else:
    print("❌ No results generated.")