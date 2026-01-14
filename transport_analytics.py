import pandas as pd
import numpy as np
from scipy.stats import gamma
import sys


# ==========================================
# 1. HELPER: COLUMN STANDARDIZER
# ==========================================
def standard_columns(df):
    """
    Renames Czech/variable columns to standard English analysis names.
    """
    df.columns = [str(c).lower().strip() for c in df.columns]

    # DEBUG PRINT (You can comment this out later if you want less noise)
    print(f"    [DEBUG] Train File Columns found: {list(df.columns)}")

    default_map = {
        'actual_arrival': [
            'skutečný příjezd', 'příjezd', 'příj._skut.', 'cas_pr_skut',
            'čas příjezdu', 'actual_arrival', 'prij._skut.', 'příjezd skut.',
            'cas prijezdu', 'skut.příj.'
        ],
        'actual_departure': [
            'skutečný odjezd', 'odjezd', 'odj._skut.', 'cas_od_skut',
            'čas odjezdu', 'actual_departure', 'odj._skut.', 'odjezd skut.',
            'cas odjezdu', 'skut.odj.'
        ],
        'train_type': [
            'druh_vlaku', 'typ', 'kategorie', 'train_type', 'druh', 'druh vlaku'
        ],
        'station': [
            'stanice', 'nazev_stanice', 'název bodu výběru', 'station', 'bod',
            'nazev bodu vyberu'
        ]
    }

    for target_col, possible_names in default_map.items():
        if target_col in df.columns: continue

        for candidate in possible_names:
            candidate_clean = candidate.lower()
            if candidate_clean in df.columns:
                df.rename(columns={candidate_clean: target_col}, inplace=True)
                break

        if target_col not in df.columns:
            for col in df.columns:
                if target_col == 'actual_arrival' and 'příj' in col and ('čas' in col or 'skut' in col):
                    df.rename(columns={col: target_col}, inplace=True)
                elif target_col == 'actual_departure' and 'odj' in col and ('čas' in col or 'skut' in col):
                    df.rename(columns={col: target_col}, inplace=True)

    return df


# ==========================================
# 2. SIGNAL PROCESSING ENGINE
# ==========================================
def get_dynamic_kernels(p_arr, p_dep, peak_arr, peak_dep):
    a_arr = (peak_arr / 2.0) + 1
    x_arr = np.linspace(0, p_arr, p_arr)
    k_arr = gamma.pdf(x_arr, a=a_arr, scale=2)
    k_arr = k_arr / k_arr.max() if k_arr.max() > 0 else k_arr

    a_dep = (peak_dep / 2.0) + 1
    x_dep = np.linspace(0, p_dep, p_dep)
    k_dep = gamma.pdf(x_dep, a=a_dep, scale=2)
    k_dep = (k_dep / k_dep.max())[::-1] if k_dep.max() > 0 else k_dep

    return k_arr, k_dep


def generate_signals(subset_trains, timeline, params):
    dur_arr = params['pulse_arr']
    dur_dep = params['pulse_dep']
    boost_am = params['rush_am']
    boost_pm = params['rush_pm']
    peak_arr = params.get('peak_arr', 4)
    peak_dep = params.get('peak_dep', 15)

    sig_arr = pd.Series(0.0, index=timeline)
    sig_dep = pd.Series(0.0, index=timeline)
    k_arr, k_dep = get_dynamic_kernels(dur_arr, dur_dep, peak_arr, peak_dep)

    for _, t in subset_trains.iterrows():
        if pd.notnull(t.get('actual_arrival')):
            try:
                t_idx = sig_arr.index.get_loc(t['actual_arrival'].round('1min'))
                boost = boost_am if (6 <= t['actual_arrival'].hour <= 9) else 1.0
                end = min(t_idx + dur_arr, len(sig_arr))
                L = end - t_idx
                if L > 0: sig_arr.iloc[t_idx:end] += (k_arr[:L] * boost)
            except KeyError:
                pass

        if pd.notnull(t.get('actual_departure')):
            try:
                start_t = t['actual_departure'] - pd.Timedelta(minutes=dur_dep)
                t_idx = sig_dep.index.get_loc(start_t.round('1min'))
                boost = boost_pm if (14 <= t['actual_departure'].hour <= 18) else 1.0
                end = min(t_idx + dur_dep, len(sig_dep))
                L = end - t_idx
                if L > 0: sig_dep.iloc[t_idx:end] += (k_dep[:L] * boost)
            except KeyError:
                pass

    return sig_arr, sig_dep


# ==========================================
# 3. ANALYSIS RUNNER (LOGIC ONLY)
# ==========================================
def analyze_city_data(city_name, config, df_trains, df_bikes, df_logs, df_weather, categories):
    print(f"  > Processing Algorithms for {city_name}...")

    # 1. Standardize Columns
    df_trains = standard_columns(df_trains)

    # 2. Filter Trains
    if 'station' in df_trains.columns:
        if len(df_trains['station'].unique()) > 1:
            hub_trains = df_trains[df_trains['station'] == config['train_station']]
        else:
            hub_trains = df_trains
    else:
        hub_trains = df_trains

    if hub_trains.empty:
        print("    WARNING: No trains found in dataframe after filtering.")
        return []

    # 3. Filter Bike Rides
    df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]
    starts = df_bikes[df_bikes['start_place'].isin(config['bike_stations'])]
    ends = df_bikes[df_bikes['end_place'].isin(config['bike_stations'])]

    # 4. Timeline Setup
    try:
        start_t = hub_trains['actual_arrival'].min().floor('h')
        end_t = hub_trains['actual_arrival'].max().ceil('h')
        timeline = pd.date_range(start_t, end_t, freq='1min')
    except KeyError:
        print("    [!] ABORTING: 'actual_arrival' column still missing. Check debug output above.")
        return []

    # 5. ROBUST LOGS PROCESSING
    hub_logs = pd.DataFrame()
    if df_logs is not None and not df_logs.empty:
        df_logs.columns = [str(c).lower().strip() for c in df_logs.columns]

        # A. FIND TIMESTAMP
        time_col = next(
            (c for c in df_logs.columns if c in ['čas', 'cas', 'time', 'timestamp', 'date', 'interval_start']), None)

        if time_col:
            temp_logs = pd.DataFrame()
            temp_logs['timestamp'] = pd.to_datetime(df_logs[time_col])

            # B. WIDE vs LONG Check
            if 'place_name' in df_logs.columns:
                target_rows = df_logs[df_logs['place_name'].isin(config['bike_stations'])]
                cnt_col = next((c for c in df_logs.columns if c in ['bikes', 'count', 'pocet_kol', 'bikes_available']),
                               None)
                if cnt_col:
                    temp_logs['bikes'] = target_rows[cnt_col]
                    hub_logs = temp_logs.dropna()
            else:
                found_cols = []
                for target in config['bike_stations']:
                    match = next((col for col in df_logs.columns if target.lower() in col), None)
                    if match: found_cols.append(match)

                if found_cols:
                    temp_logs['bikes'] = df_logs[found_cols].fillna(0).sum(axis=1)
                    hub_logs = temp_logs

    # =========================================================
    # 6. MASKS (Availability & Weather)
    # =========================================================

    # DEFAULT: Assume everything is OK (Safety fallback)
    mask_available = pd.Series(True, index=timeline)
    mask_good = pd.Series(True, index=timeline)
    final_mask = pd.Series(True, index=timeline)  # <--- INITIALIZED HERE SO IT NEVER CRASHES

    # A. Availability Mask
    if not hub_logs.empty and 'bikes' in hub_logs.columns:
        stock = hub_logs.set_index('timestamp')['bikes'].resample('1min').ffill().reindex(timeline, method='nearest')
        mask_available = stock >= 1

    # B. Weather Mask
    if df_weather is not None and not df_weather.empty:
        df_weather.columns = [str(c).lower().strip() for c in df_weather.columns]

        # Search for datetime column
        possible_time_cols = ['datetime', 'timestamp', 'datum', 'cas', 'čas', 'date', 'time', 'local_time']
        w_time_col = next((c for c in df_weather.columns if c in possible_time_cols), None)

        if w_time_col:
            # FORCE DATETIME CONVERSION (Fixes 'Index' error)
            df_weather['timestamp'] = pd.to_datetime(df_weather[w_time_col], errors='coerce')
            df_weather = df_weather.dropna(subset=['timestamp'])  # Remove bad dates

            # Find data columns
            temp_col = next((c for c in df_weather.columns if 'temp' in c or 'tepl' in c), None)
            rain_col = next((c for c in df_weather.columns if
                             'precip' in c or 'sraz' in c or 'sráž' in c or 'prset' in c or 'rain' in c), None)

            if not temp_col and len(df_weather.columns) > 1: temp_col = df_weather.columns[2]
            if not rain_col and len(df_weather.columns) > 6: rain_col = df_weather.columns[6]

            try:
                w_res = df_weather.set_index('timestamp').resample('1min').ffill().reindex(timeline, method='nearest')

                t_val = w_res[temp_col] if temp_col else 15
                r_val = w_res[rain_col] if rain_col else 0

                mask_good = (r_val <= 0) & (t_val >= 10)
            except Exception as e:
                print(f"    [!] Weather Processing Error: {e}")

    # Final combined mask
    final_mask = mask_good & mask_available

    # 7. MAIN LOOP
    results = []
    ts_starts = starts.set_index('start_time').resample('1min').size().reindex(timeline, fill_value=0)
    ts_ends = ends.set_index('end_time').resample('1min').size().reindex(timeline, fill_value=0)

    for cat_name, type_list in categories.items():
        if type_list:
            subset = hub_trains[hub_trains['train_type'].isin(type_list)]
        else:
            subset = hub_trains

        s_arr, s_dep = generate_signals(subset, timeline, config['params'])
        idx_valid = final_mask & s_arr.index.isin(timeline)

        if idx_valid.sum() > 60:
            r_arr = s_arr[idx_valid].corr(ts_starts[idx_valid])
            r_dep = s_dep[idx_valid].corr(ts_ends[idx_valid])
        else:
            r_arr, r_dep = np.nan, np.nan

        results.append({
            'City': city_name,
            'Category': cat_name,
            'Arr_Corr': r_arr,
            'Dep_Corr': r_dep
        })

    return results