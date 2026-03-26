import pandas as pd
import numpy as np
from scipy import stats
import os
import transport_analytics as ta

# ==========================================
# 1. CONFIGURATION
# ==========================================
FILE_BIKE_DATA = '../data/nextbike_data_VSB_Vaclavik.xlsx'
RESULTS_DIR = '../results'

# Define Rush Hours (Adjust these hours if your local rush is different)
AM_START, AM_END = 5, 9  # 05:00 to 08:59
PM_START, PM_END = 13, 17  # 13:00 to 16:59

# Czech Train Types
REGIONAL_TYPES = ['os', 'sp']
LD_TYPES = ['r', 'ex', 'ic', 'ec', 'sc', 'rj', 'rx', 'le', 'rgj']

CITIES = {
    'Ostrava-Svinov': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava-Svinov',
        'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
        'params': {'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1}, 'pulse_arr_reg': 15, 'pulse_dep_reg': 20,
                   'pulse_arr_ld': 20, 'pulse_dep_ld': 35}
    },
    'Ostrava hl.n.': {
        'file_trains': 'data/Pohyby_Ostrava_hl.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava hl.n.',
        'bike_stations': ['MOAP-Hlavní nádraží'],
        'params': {'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1}, 'pulse_arr_reg': 25, 'pulse_dep_reg': 25,
                   'pulse_arr_ld': 30, 'pulse_dep_ld': 35}
    },
    'Brno hl.n.': {
        'file_trains': 'data/Pohyby_Brno.xlsx',
        'file_weather': 'data/Brno_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Brno',
        'train_station': 'Brno hl.n.',
        'bike_stations': ['Hlavní nádraží - Hlavní vstup', 'Hlavní nádraží - pošta', 'Bajkazyl 666'],
        'params': {'weights': {'Os': 2.5, 'Sp': 2.0, 'R': 1.5, 'Ex': 1}, 'pulse_arr_reg': 12, 'pulse_dep_reg': 25,
                   'pulse_arr_ld': 30, 'pulse_dep_ld': 35}
    },
    'Přerov': {
        'file_trains': 'data/Pohyby_Prerov.xlsx',
        'file_weather': 'data/Prerov_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Prerov',
        'train_station': 'Přerov os.n.',
        'bike_stations': ['Nádraží'],
        'params': {'weights': {'Os': 1.7, 'Sp': 1.4, 'R': 1.2, 'Ex': 1}, 'pulse_arr_reg': 12, 'pulse_dep_reg': 20,
                   'pulse_arr_ld': 15, 'pulse_dep_ld': 30}
    },
    'Valašské Meziříčí': {
        'file_trains': 'data/Pohyby_ValMez.xlsx',
        'file_weather': 'data/ValMez_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_ValMez',
        'train_station': 'Valašské Meziříčí',
        'bike_stations': ['Vlakové nádraží Valašské Meziříčí (nové umístění)'],
        'params': {'weights': {'Os': 1.8, 'Sp': 1.6, 'R': 1.4, 'Ex': 1.6}, 'pulse_arr_reg': 12, 'pulse_dep_reg': 20,
                   'pulse_arr_ld': 20, 'pulse_dep_ld': 25}
    }
}


# ==========================================
# 2. LOCAL MATH & LOADERS
# ==========================================
def calculate_p_value_locally(x, y, w):
    """Calculates Weighted Pearson r AND P-Value."""
    x, y, w = np.asanyarray(x), np.asanyarray(y), np.asanyarray(w)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[mask], y[mask], w[mask]

    if len(x) < 3: return np.nan, np.nan
    sum_w = np.sum(w)
    if sum_w == 0: return np.nan, np.nan

    mean_x, mean_y = np.average(x, weights=w), np.average(y, weights=w)
    xm, ym = x - mean_x, y - mean_y

    cov = np.sum(w * xm * ym) / sum_w
    var_x = np.sum(w * xm ** 2) / sum_w
    var_y = np.sum(w * ym ** 2) / sum_w

    if var_x <= 0 or var_y <= 0: return np.nan, np.nan

    r = cov / np.sqrt(var_x * var_y)
    r = max(min(r, 1.0), -1.0)

    n_eff = (sum_w ** 2) / np.sum(w ** 2)
    df = max(n_eff - 2, 1)

    if abs(r) >= 1.0:
        p = 0.0
    else:
        t_stat = r * np.sqrt(df / (1 - r ** 2))
        p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return r, p


def load_train_file_smart(filepath):
    try:
        preview = pd.read_excel(filepath, header=None, nrows=20)
        keywords = ['čas', 'cas', 'time', 'druh', 'type', 'vlak', 'train', 'příjezd', 'odjezd']
        header_row_idx = next((idx for idx, row in preview.iterrows() if
                               sum(1 for k in keywords if k in " ".join(row.astype(str)).lower()) >= 2), 0)
        return pd.read_excel(filepath, header=header_row_idx)
    except:
        return pd.read_excel(filepath)


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def run_directional_analysis():
    print("=== 🚴 DIRECTIONAL COMMUTER RUSH ANALYSIS (CATEGORIZED) 🚆 ===")
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    results = []

    for city, cfg in CITIES.items():
        print(f"\nProcessing {city}...")

        # 1. LOAD DATA
        try:
            df_t_master = load_train_file_smart(cfg['file_trains'])
            df_t_master = ta.standard_columns(df_t_master)
            df_t_master['actual_arrival'] = pd.to_datetime(df_t_master.get('actual_arrival'), errors='coerce')
            df_t_master['actual_departure'] = pd.to_datetime(df_t_master.get('actual_departure'), errors='coerce')

            if 'station' in df_t_master.columns:
                df_t_master = df_t_master[
                    df_t_master['station'].astype(str).str.contains(cfg['train_station'], case=False, na=False)]

            df_b = pd.read_excel(FILE_BIKE_DATA, sheet_name=cfg['sheet_rentals'])
            df_b.columns = [str(c).lower().strip() for c in df_b.columns]
            df_b['start_time'] = pd.to_datetime(df_b['start_time'], errors='coerce')
            df_b['end_time'] = pd.to_datetime(df_b['end_time'], errors='coerce')
        except Exception as e:
            print(f"  [ERROR] Loading failed: {e}")
            continue

        # 2. ALIGN TIMELINE
        t_min = min(df_t_master['actual_arrival'].dropna().min(), df_b['start_time'].dropna().min()).floor('h')
        t_max = max(df_t_master['actual_arrival'].dropna().max(), df_b['end_time'].dropna().max()).ceil('h')
        timeline = pd.date_range(t_min, t_max, freq='1min')

        # 3. AGGREGATE BIKES (Static for all train categories)
        valid_stations = [s.lower() for s in cfg['bike_stations']]
        starts = df_b[df_b['start_place'].astype(str).str.lower().isin(valid_stations)]
        ends = df_b[df_b['end_place'].astype(str).str.lower().isin(valid_stations)]

        c_rent = starts.set_index('start_time').resample('1min').size().reindex(timeline, fill_value=0)
        c_ret = ends.set_index('end_time').resample('1min').size().reindex(timeline, fill_value=0)

        # 4. TIME MASKS
        weights = pd.Series(1.0, index=timeline)
        workdays = timeline.dayofweek < 5
        mask_am = workdays & (timeline.hour >= AM_START) & (timeline.hour < AM_END)
        mask_pm = workdays & (timeline.hour >= PM_START) & (timeline.hour < PM_END)

        # 5. LOOP BY CATEGORY
        for category in ['Global', 'Regional', 'Long Distance']:
            print(f"  -> Category: {category}")

            # Filter Trains
            if category == 'Global':
                df_t_sub = df_t_master.copy()
            else:
                if 'train_type' not in df_t_master.columns:
                    print(f"     [!] 'train_type' column missing. Skipping {category}.")
                    continue

                # Extract first word to catch 'Os 1234' -> 'os'
                t_types = df_t_master['train_type'].astype(str).str.lower().str.strip().str.split().str[0]

                if category == 'Regional':
                    df_t_sub = df_t_master[t_types.isin(REGIONAL_TYPES)]
                elif category == 'Long Distance':
                    df_t_sub = df_t_master[t_types.isin(LD_TYPES)]

            if df_t_sub.empty:
                print(f"     [!] No trains found for {category}. Skipping.")
                continue

            # Generate Category Signals
            s_arr, s_dep = ta.generate_signals(df_t_sub, timeline, cfg['params'])

            # Calculate Correlations
            am_inbound_r, _ = calculate_p_value_locally(s_arr[mask_am], c_rent[mask_am], weights[mask_am])
            am_outbound_r, _ = calculate_p_value_locally(s_dep[mask_am], c_ret[mask_am], weights[mask_am])

            pm_outbound_r, _ = calculate_p_value_locally(s_dep[mask_pm], c_ret[mask_pm], weights[mask_pm])
            pm_inbound_r, _ = calculate_p_value_locally(s_arr[mask_pm], c_rent[mask_pm], weights[mask_pm])

            results.append({
                'City': city,
                'Category': category,
                'AM_Inbound (Arr->Rent)': round(am_inbound_r, 4),
                'AM_Outbound (Ret->Dep)': round(am_outbound_r, 4),
                'PM_Outbound (Ret->Dep)': round(pm_outbound_r, 4),
                'PM_Inbound (Arr->Rent)': round(pm_inbound_r, 4)
            })

    # 6. EXPORT
    if results:
        df_res = pd.DataFrame(results)

        # Sort values nicely before saving
        df_res['Cat_Rank'] = df_res['Category'].map({'Global': 1, 'Regional': 2, 'Long Distance': 3})
        df_res = df_res.sort_values(by=['City', 'Cat_Rank']).drop(columns=['Cat_Rank'])

        out_path = os.path.join(RESULTS_DIR, 'Directional_Rush_Analysis.xlsx')
        df_res.to_excel(out_path, index=False)

        print("\n=== 📊 DIRECTIONAL COMMUTER CORRELATIONS ===")
        print(df_res.to_string(index=False))
        print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    run_directional_analysis()