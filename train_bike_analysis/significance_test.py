import pandas as pd
import numpy as np
import os
from scipy import stats
import sys

# --- IMPORT EXISTING MODULE ---
import transport_analytics as ta

# ==========================================
# 1. CONFIGURATION
# ==========================================
FILE_BIKE_DATA = '../data/nextbike_data_VSB_Vaclavik.xlsx'
RESULTS_DIR = '../results'
ALPHA = 0.05

CITIES = {
    'Ostrava-Svinov': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava-Svinov',
        'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
        'params': {
            'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1},
            'pulse_arr_reg': 15, 'pulse_dep_reg': 20,
            'pulse_arr_ld': 20, 'pulse_dep_ld': 35
        },
    },
    'Ostrava hl.n.': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava-Svinov',  # Check filter
        'bike_stations': ['MOAP-Hlavní nádraží'],
        'params': {
            'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1},
            'pulse_arr_reg': 25, 'pulse_dep_reg': 25,
            'pulse_arr_ld': 30, 'pulse_dep_ld': 35
        },
    },
    'Brno hl.n.': {
        'file_trains': 'data/Pohyby_Brno.xlsx',
        'file_weather': 'data/Brno_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Brno',
        'train_station': 'Brno hl.n.',
        'bike_stations': ['Hlavní nádraží - Hlavní vstup', 'Hlavní nádraží - pošta', 'Bajkazyl 666'],
        'params': {
            'weights': {'Os': 2.5, 'Sp': 2.0, 'R': 1.5, 'Ex': 1},
            'pulse_arr_reg': 12, 'pulse_dep_reg': 25,
            'pulse_arr_ld': 30, 'pulse_dep_ld': 35
        },
    },
    'Přerov': {
        'file_trains': 'data/Pohyby_Prerov.xlsx',
        'file_weather': 'data/Prerov_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Prerov',
        'train_station': 'Přerov os.n.',
        'bike_stations': ['Nádraží'],
        'params': {
            'weights': {'Os': 1.7, 'Sp': 1.4, 'R': 1.2},
            'pulse_arr_reg': 12, 'pulse_dep_reg': 20
        },
    },
    'Valašské Meziříčí': {
        'file_trains': 'data/Pohyby_ValMez.xlsx',
        'file_weather': 'data/ValMez_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_ValMez',
        'train_station': 'Valašské Meziříčí',
        'bike_stations': ['Vlakové nádraží Valašské Meziříčí (nové umístění)'],
        'params': {
            'weights': {'Os': 1.8, 'Sp': 1.6, 'R': 1.4},
            'pulse_arr_reg': 12, 'pulse_dep_reg': 20
        },
    }
}


# ==========================================
# 2. LOCAL STATISTICAL FUNCTIONS
# ==========================================
def calculate_p_value_locally(x, y, w):
    """Calculates Weighted Pearson r AND P-Value."""
    x, y, w = np.asanyarray(x), np.asanyarray(y), np.asanyarray(w)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[mask], y[mask], w[mask]

    if len(x) < 3: return 0.0, 1.0, 0
    sum_w = np.sum(w)
    if sum_w == 0: return 0.0, 1.0, 0

    mean_x, mean_y = np.average(x, weights=w), np.average(y, weights=w)
    xm, ym = x - mean_x, y - mean_y

    cov = np.sum(w * xm * ym) / sum_w
    var_x = np.sum(w * xm ** 2) / sum_w
    var_y = np.sum(w * ym ** 2) / sum_w

    if var_x <= 0 or var_y <= 0: return 0.0, 1.0, int(sum_w)

    r = cov / np.sqrt(var_x * var_y)
    r = max(min(r, 1.0), -1.0)

    n_eff = (sum_w ** 2) / np.sum(w ** 2)
    df = max(n_eff - 2, 1)

    if abs(r) >= 1.0:
        p = 0.0
    else:
        t_stat = r * np.sqrt(df / (1 - r ** 2))
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return r, p, n_eff


# ==========================================
# 3. ROBUST FILE LOADER (THE FIX)
# ==========================================
def load_train_file_smart(filepath):
    """
    Scans the first 20 rows of an Excel file to find the header.
    Looks for keywords like 'druh vlaku', 'čas příjezdu', etc.
    """
    try:
        # 1. Read first 20 rows without header
        preview = pd.read_excel(filepath, header=None, nrows=20)

        # 2. Define keywords that MUST exist in the header row (lowercase)
        keywords = ['čas', 'cas', 'time', 'druh', 'type', 'vlak', 'train', 'příjezd', 'odjezd']

        header_row_idx = 0  # Default to 0 if not found

        # 3. Scan row by row
        for idx, row in preview.iterrows():
            # Convert entire row to a single lowercase string
            row_str = " ".join(row.astype(str)).lower()

            # Count how many keywords appear in this row
            matches = sum(1 for k in keywords if k in row_str)

            # If we find at least 2 keywords, this is definitely the header
            if matches >= 2:
                header_row_idx = idx
                print(f"    -> Found Header at Row {idx + 1}")
                break

        # 4. Reload file with the correct header
        return pd.read_excel(filepath, header=header_row_idx)

    except Exception as e:
        print(f"    [WARN] Smart load failed: {e}. Trying default.")
        return pd.read_excel(filepath)


# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def run_significance_test():
    print("=== 📊 P-VALUE SIGNIFICANCE TEST ===")
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    results = []

    for city, cfg in CITIES.items():
        print(f"\nProcessing {city}...")

        # --- A. LOAD DATA ---
        try:
            # 1. Trains (With Header Hunting)
            if not os.path.exists(cfg['file_trains']):
                print(f"  [SKIP] File missing: {cfg['file_trains']}")
                continue

            # USE SMART LOADER HERE
            df_t = load_train_file_smart(cfg['file_trains'])
            df_t = ta.standard_columns(df_t)  # Clean Columns

            # DEBUG CHECK
            if 'actual_arrival' not in df_t.columns:
                print(f"  [CRITICAL ERROR] Column 'actual_arrival' still missing!")
                print(f"  Columns found: {list(df_t.columns)}")
                continue

            # Convert Dates
            df_t['actual_arrival'] = pd.to_datetime(df_t['actual_arrival'], errors='coerce')
            if 'actual_departure' in df_t.columns:
                df_t['actual_departure'] = pd.to_datetime(df_t['actual_departure'], errors='coerce')

            # Filter Station
            if 'station' in df_t.columns and cfg.get('train_station'):
                mask = df_t['station'].astype(str).str.contains(cfg['train_station'], case=False, na=False)
                df_t = df_t[mask]

            # 2. Bikes
            if not os.path.exists(FILE_BIKE_DATA):
                print(f"  [CRITICAL] Bike file missing.")
                return

            df_b = pd.read_excel(FILE_BIKE_DATA, sheet_name=cfg['sheet_rentals'])
            df_b.columns = [str(c).lower().strip() for c in df_b.columns]
            df_b['start_time'] = pd.to_datetime(df_b['start_time'], errors='coerce')
            df_b['end_time'] = pd.to_datetime(df_b['end_time'], errors='coerce')

            # 3. Weather
            df_w = pd.DataFrame()
            if os.path.exists(cfg['file_weather']):
                df_w = pd.read_excel(cfg['file_weather'])
                df_w.columns = [str(c).lower().strip() for c in df_w.columns]

        except Exception as e:
            print(f"  [ERROR] Loading failed: {e}")
            continue

        # --- B. ALIGN TIMELINE ---
        try:
            t_min_t = df_t['actual_arrival'].dropna().min()
            t_min_b = df_b['start_time'].dropna().min()
            t_max_t = df_t['actual_arrival'].dropna().max()
            t_max_b = df_b['end_time'].dropna().max()

            if pd.isna(t_min_t) or pd.isna(t_min_b):
                print("  [ERROR] No valid dates found in data.")
                continue

            t_start = max(t_min_t, t_min_b).floor('h')
            t_end = min(t_max_t, t_max_b).ceil('h')

            if t_start >= t_end:
                print(f"  [ERROR] Date mismatch (No overlap).")
                continue

            timeline = pd.date_range(t_start, t_end, freq='1min')
            print(f"  -> Timeline: {len(timeline)} min")

        except Exception as e:
            print(f"  [ERROR] Alignment crash: {e}")
            continue

        # --- C. GENERATE SIGNALS ---
        print("  -> Convolving Passenger Pressure...")
        s_arr, s_dep = ta.generate_signals(df_t, timeline, cfg['params'])
        pressure_total = s_arr + s_dep

        # --- D. AGGREGATE BIKES ---
        print("  -> Aggregating Bike Flows...")
        valid_stations = [s.lower() for s in cfg['bike_stations']]

        starts = df_b[df_b['start_place'].astype(str).str.lower().isin(valid_stations)]
        ends = df_b[df_b['end_place'].astype(str).str.lower().isin(valid_stations)]

        ts_starts = starts.set_index('start_time').resample('1min').size().reindex(timeline, fill_value=0)
        ts_ends = ends.set_index('end_time').resample('1min').size().reindex(timeline, fill_value=0)
        bike_total = ts_starts + ts_ends

        # --- E. WEATHER WEIGHTS ---
        weights = pd.Series(1.0, index=timeline)
        if not df_w.empty:
            p_col = next((c for c in df_w.columns if any(x in c for x in ['precip', 'rain', 'srazky'])), None)
            t_col = next((c for c in df_w.columns if any(x in c for x in ['date', 'cas', 'time'])), None)

            if p_col and t_col:
                df_w['ts'] = pd.to_datetime(df_w[t_col], errors='coerce')
                df_w = df_w.dropna(subset=['ts'])
                df_w['ts'] = df_w['ts'].dt.round('1min')
                w_res = df_w.set_index('ts').resample('1min').ffill().reindex(timeline, method='ffill')
                weights[w_res[p_col] > 0.2] = 0.5

        # --- F. CALCULATE STATS ---
        r, p, n_eff = calculate_p_value_locally(pressure_total.values, bike_total.values, weights.values)
        r_arr, p_arr, _ = calculate_p_value_locally(s_arr.values, ts_starts.values, weights.values)
        r_dep, p_dep, _ = calculate_p_value_locally(s_dep.values, ts_ends.values, weights.values)

        is_sig = "YES" if p < ALPHA else "NO"

        results.append({
            'City': city,
            'Global_r': r,
            'Global_p': p,
            'Significant': is_sig,
            'N_Eff': int(n_eff),
            'Arr_r': r_arr,
            'Arr_p': p_arr,
            'Dep_r': r_dep,
            'Dep_p': p_dep
        })
        print(f"     [RESULT] r={r:.4f}, p={p:.4e} -> {is_sig}")

    if results:
        df_res = pd.DataFrame(results)
        out_path = os.path.join(RESULTS_DIR, 'Significance_Report.csv')
        df_res.to_csv(out_path, index=False)
        print("\n=== FINAL REPORT ===")
        print(df_res[['City', 'Global_r', 'Global_p', 'Significant']].to_string(index=False))
        print(f"\n✅ Saved to {out_path}")


if __name__ == "__main__":
    run_significance_test()