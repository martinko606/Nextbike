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
    """
    Generates theoretical pressure signals with Station-Specific Weights and Timings.
    """
    # --- A. DEFAULTS (Fallback if not specified in config) ---
    # Timing Defaults
    def_p_arr = params.get('pulse_arr', 30)
    def_p_dep = params.get('pulse_dep', 45)
    def_peak_arr = params.get('peak_arr', 4)
    def_peak_dep = params.get('peak_dep', 15)

    # --- B. READ STATION-SPECIFIC TIMING ---
    # 1. Regional (Standard)
    dur_arr_reg = int(params.get('pulse_arr_reg', def_p_arr))
    dur_dep_reg = int(params.get('pulse_dep_reg', def_p_dep))
    peak_arr_reg = params.get('peak_arr_reg', def_peak_arr)
    peak_dep_reg = params.get('peak_dep_reg', def_peak_dep)

    # 2. Long Distance (Optional override)
    # If not set, defaults to 1.5x Regional
    dur_arr_ld = int(params.get('pulse_arr_ld', dur_arr_reg * 1.5))
    dur_dep_ld = int(params.get('pulse_dep_ld', dur_dep_reg * 1.5))
    peak_arr_ld = params.get('peak_arr_ld', peak_arr_reg + 5)
    peak_dep_ld = params.get('peak_dep_ld', peak_dep_reg + 10)

    # --- C. PRE-CALCULATE KERNELS ---
    k_arr_reg, k_dep_reg = get_dynamic_kernels(dur_arr_reg, dur_dep_reg, peak_arr_reg, peak_dep_reg)
    k_arr_ld, k_dep_ld = get_dynamic_kernels(dur_arr_ld, dur_dep_ld, peak_arr_ld, peak_dep_ld)

    # --- D. READ STATION-SPECIFIC WEIGHTS ---
    # This allows Brno to have different multipliers than Prerov
    default_weights = {
        'Os': 2.0, 'Sp': 1.5,
        'R': 1, 'Ex': 1, 'Rx': 1,
        'EC': 1, '?': 1.0
    }
    # MERGE: Start with defaults, overwrite with station specific weights
    weights = default_weights.copy()
    if 'weights' in params:
        weights.update(params['weights'])

    # Long Distance Types Trigger
    ld_types = ['R', 'Ex', 'Rx', 'IC', 'EC', 'rj', 'SC', 'EN', 'NJ']

    # --- E. INITIALIZE OUTPUT ---
    sig_arr = pd.Series(0.0, index=timeline)
    sig_dep = pd.Series(0.0, index=timeline)

    boost_am = params.get('rush_am', 1.0)
    boost_pm = params.get('rush_pm', 1.0)

    # --- F. MAIN LOOP ---
    for _, t in subset_trains.iterrows():
        # Get Train Properties
        t_type = str(t.get('train_type', '?')).strip()

        # 1. Get Weight (Intensity)
        w = weights.get(t_type, 1.0)

        # 2. Select Kernel (Timing)
        is_ld = t_type in ld_types
        if is_ld:
            k_arr, dur_arr = k_arr_ld, dur_arr_ld
            k_dep, dur_dep = k_dep_ld, dur_dep_ld
        else:
            k_arr, dur_arr = k_arr_reg, dur_arr_reg
            k_dep, dur_dep = k_dep_reg, dur_dep_reg

        # 3. Apply Signal (Arrival)
        if pd.notnull(t.get('actual_arrival')):
            try:
                t_idx = sig_arr.index.get_loc(t['actual_arrival'].round('1min'))
                time_boost = boost_am if (6 <= t['actual_arrival'].hour <= 9) else 1.0

                end = min(t_idx + dur_arr, len(sig_arr))
                L = end - t_idx
                if L > 0:
                    sig_arr.iloc[t_idx:end] += (k_arr[:L] * w * time_boost)
            except KeyError:
                pass

        # 4. Apply Signal (Departure)
        if pd.notnull(t.get('actual_departure')):
            try:
                start_t = t['actual_departure'] - pd.Timedelta(minutes=dur_dep)
                t_idx = sig_dep.index.get_loc(start_t.round('1min'))
                time_boost = boost_pm if (14 <= t['actual_departure'].hour <= 18) else 1.0

                end = min(t_idx + dur_dep, len(sig_dep))
                L = end - t_idx
                if L > 0:
                    sig_dep.iloc[t_idx:end] += (k_dep[:L] * w * time_boost)
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
    if 'station' in df_trains.columns and len(df_trains['station'].unique()) > 1:
        hub_trains = df_trains[df_trains['station'] == config['train_station']]
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
        print("    [!] ABORTING: 'actual_arrival' column still missing.")
        return []

    # =========================================================
    # 5. CALCULATE FLOWS & STOCK
    # =========================================================
    # A. Individual Flows
    ts_starts = starts.set_index('start_time').resample('1min').size().reindex(timeline, fill_value=0)
    ts_ends = ends.set_index('end_time').resample('1min').size().reindex(timeline, fill_value=0)

    # B. TOTAL BIKE ACTIVITY (Rentals + Returns)
    ts_total_activity = ts_starts + ts_ends

    # C. Hybrid Stock Simulation
    net_flow = ts_ends - ts_starts
    sim_stock = net_flow.cumsum()

    log_stock = pd.Series(np.nan, index=timeline)
    if df_logs is not None and not df_logs.empty:
        df_logs.columns = [str(c).lower().strip() for c in df_logs.columns]
        time_col = next((c for c in df_logs.columns if c in ['čas', 'cas', 'time', 'timestamp', 'date']), None)
        if time_col:
            df_logs['timestamp'] = pd.to_datetime(df_logs[time_col])
            # Station column logic
            if 'place_name' in df_logs.columns:
                target_rows = df_logs[df_logs['place_name'].isin(config['bike_stations'])]
                cnt_col = next((c for c in df_logs.columns if c in ['bikes', 'count', 'pocet_kol']), None)
                if cnt_col:
                    temp = target_rows.set_index('timestamp')[cnt_col]
                    log_stock.update(temp.groupby(level=0).last())
            else:
                found_cols = []
                for target in config['bike_stations']:
                    match = next((col for col in df_logs.columns if target.lower() in col), None)
                    if match: found_cols.append(match)
                if found_cols:
                    temp = df_logs.set_index('timestamp')[found_cols].fillna(0).sum(axis=1)
                    log_stock.update(temp.groupby(level=0).last())

    offsets = log_stock - sim_stock
    offsets = offsets.ffill().bfill().fillna(0)
    final_stock = sim_stock + offsets

    # Mask Available
    mask_available = (final_stock >= 1) | (ts_starts >= 1)

    # =========================================================
    # 6. MASKS (Weather, Workday, Peak)
    # =========================================================
    mask_good_weather = pd.Series(True, index=timeline)
    mask_bad_weather = pd.Series(False, index=timeline)

    if df_weather is not None and not df_weather.empty:
        try:
            df_weather.columns = [str(c).lower().strip() for c in df_weather.columns]
            w_time_col = next((c for c in df_weather.columns if c in ['datetime', 'timestamp', 'date', 'cas']), None)
            temp_col = next((c for c in df_weather.columns if c in ['temp', 'temperature', 'teplota']), None)
            rain_col = next((c for c in df_weather.columns if c in ['precip', 'rain', 'srazky']), None)

            if w_time_col and temp_col and rain_col:
                for col in [temp_col, rain_col]:
                    if df_weather[col].dtype == 'object':
                        df_weather[col] = df_weather[col].astype(str).str.replace(',', '.')
                    df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')

                df_weather['timestamp'] = pd.to_datetime(df_weather[w_time_col], errors='coerce')
                df_weather = df_weather.dropna(subset=['timestamp'])
                # Round time to nearest minute first to kill jitter
                df_weather['timestamp'] = df_weather['timestamp'].dt.round('1min')
                # Use ffill (Forward Fill) instead of nearest
                w_res = df_weather.set_index('timestamp').resample('1min').ffill().reindex(timeline, method='ffill')

                mask_bad_logic = (w_res[rain_col] > 0.2) | (w_res[temp_col] < 10)
                mask_bad_weather = mask_bad_logic.fillna(False)
                mask_good_weather = ~mask_bad_weather
        except Exception as e:
            print(f"    [!] Weather Processing Warning: {e}")

    mask_workday = timeline.dayofweek < 5
    mask_weekend = timeline.dayofweek >= 5

    hours = timeline.hour
    mask_am_rush = (hours >= 6) & (hours < 9)
    mask_pm_rush = (hours >= 14) & (hours < 18)

    mask_workday_am = mask_workday & mask_am_rush
    mask_workday_pm = mask_workday & mask_pm_rush

    # =========================================================
    # 7. CORRELATION LOOP
    # =========================================================
    results = []

    for cat_name, type_list in categories.items():
        if type_list:
            subset = hub_trains[hub_trains['train_type'].isin(type_list)]
        else:
            subset = hub_trains  # ALL Trains

        s_arr, s_dep = generate_signals(subset, timeline, config['params'])

        # --- NEW: TOTAL TRAIN PRESSURE (Arr + Dep) ---
        s_total = s_arr + s_dep

        def get_corr(signal, actuals, context_mask):
            valid_idx = context_mask & mask_available & signal.index.isin(timeline)
            if valid_idx.sum() < 60: return np.nan
            s_vec = signal[valid_idx]
            a_vec = actuals[valid_idx]
            if s_vec.std() == 0 or a_vec.std() == 0: return np.nan
            return s_vec.corr(a_vec)

        # 1. TOTAL GLOBAL (The Combined Metric)
        #    Signal: Arr + Dep
        #    Actuals: Rentals + Returns
        r_total_global = get_corr(s_total, ts_total_activity, pd.Series(True, index=timeline))

        # 2. Directional Baselines
        r_arr_global = get_corr(s_arr, ts_starts, pd.Series(True, index=timeline))
        r_dep_global = get_corr(s_dep, ts_ends, pd.Series(True, index=timeline))

        # 3. Weather
        r_arr_bad = get_corr(s_arr, ts_starts, mask_bad_weather)
        r_dep_bad = get_corr(s_dep, ts_ends, mask_bad_weather)

        # 4. Day Type
        r_arr_work = get_corr(s_arr, ts_starts, mask_workday)
        r_dep_work = get_corr(s_dep, ts_ends, mask_workday)
        r_arr_wknd = get_corr(s_arr, ts_starts, mask_weekend)
        r_dep_wknd = get_corr(s_dep, ts_ends, mask_weekend)

        # 5. Peaks
        r_arr_am = get_corr(s_arr, ts_starts, mask_workday_am)
        r_dep_pm = get_corr(s_dep, ts_ends, mask_workday_pm)

        results.append({
            'City': city_name,
            'Category': cat_name,

            # === THE NEW METRIC ===
            'Total_Activity_Corr': r_total_global,
            # ======================

            'Arr_Global': r_arr_global,
            'Dep_Global': r_dep_global,
            'Arr_BadWeather': r_arr_bad,
            'Dep_BadWeather': r_dep_bad,
            'Arr_Workday': r_arr_work,
            'Dep_Workday': r_dep_work,
            'Arr_Weekend': r_arr_wknd,
            'Dep_Weekend': r_dep_wknd,
            'Arr_Workday_AM': r_arr_am,
            'Dep_Workday_PM': r_dep_pm
        })

    return results