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
        print("    [!] ABORTING: 'actual_arrival' column still missing.")
        return []

    # =========================================================
    # 5. CALCULATE FLOWS (Rentals & Returns)
    # =========================================================
    # We need these immediately to calculate stock
    ts_starts = starts.set_index('start_time').resample('1min').size().reindex(timeline, fill_value=0)
    ts_ends = ends.set_index('end_time').resample('1min').size().reindex(timeline, fill_value=0)

    # =========================================================
    # 6. HYBRID STOCK CALCULATION (Rent Sheet + Log Correction)
    # =========================================================
    # A. Calculate Net Flow (Simulation)
    net_flow = ts_ends - ts_starts
    sim_stock = net_flow.cumsum()

    # B. Load Logs (Ground Truth Checkpoints)
    log_stock = pd.Series(np.nan, index=timeline)

    if df_logs is not None and not df_logs.empty:
        df_logs.columns = [str(c).lower().strip() for c in df_logs.columns]
        time_col = next((c for c in df_logs.columns if c in ['čas', 'cas', 'time', 'timestamp', 'date']), None)

        if time_col:
            df_logs['timestamp'] = pd.to_datetime(df_logs[time_col])

            # Identify relevant log rows
            if 'place_name' in df_logs.columns:
                target_rows = df_logs[df_logs['place_name'].isin(config['bike_stations'])]
                cnt_col = next((c for c in df_logs.columns if c in ['bikes', 'count', 'pocet_kol']), None)
                if cnt_col:
                    temp = target_rows.set_index('timestamp')[cnt_col]
                    # Map logs to timeline (duplicates handled by taking last value)
                    log_stock.update(temp.groupby(level=0).last())
            else:
                # Wide format fallback
                found_cols = []
                for target in config['bike_stations']:
                    match = next((col for col in df_logs.columns if target.lower() in col), None)
                    if match: found_cols.append(match)
                if found_cols:
                    temp = df_logs.set_index('timestamp')[found_cols].fillna(0).sum(axis=1)
                    log_stock.update(temp.groupby(level=0).last())

    # C. Calculate Correction Offset
    # Where we have a log, Offset = Log - Sim
    offsets = log_stock - sim_stock

    # Forward Fill Offset (Carry the correction forward until the next log)
    # Backfill allows us to estimate stock before the first log
    offsets = offsets.ffill().bfill().fillna(0)

    # D. Final Hybrid Stock
    final_stock = sim_stock + offsets

    # E. Create Availability Mask
    # Rule 1: Stock > 0
    # Rule 2: Rental Occurred (Safety override: If rental happened, bike WAS there)
    mask_available = (final_stock >= 1) | (ts_starts >= 1)

    # =========================================================
    # 7. WEATHER STRATIFICATION (Robust Comma Fix)
    # =========================================================
    # Defaults: Assume Good Weather if processing fails
    mask_good_weather = pd.Series(True, index=timeline)
    mask_bad_weather = pd.Series(False, index=timeline)

    if df_weather is not None and not df_weather.empty:
        try:
            # 1. Clean Column Names
            df_weather.columns = [str(c).lower().strip() for c in df_weather.columns]

            # 2. Find Columns (Explicit Search)
            # Time
            w_time_col = next((c for c in df_weather.columns if c in ['datetime', 'timestamp', 'date', 'cas']), None)

            # Temp (Explicitly look for 'temp' or 'teplota')
            temp_col = next((c for c in df_weather.columns if c in ['temp', 'temperature', 'teplota']), None)

            # Rain (Explicitly look for 'precip' or 'srazky')
            rain_col = next((c for c in df_weather.columns if c in ['precip', 'rain', 'srazky', 'precipprob']), None)

            if w_time_col and temp_col and rain_col:
                # 3. FIX COMMAS (The Critical Step)
                # Convert columns to string, replace comma with dot, then back to numeric
                for col in [temp_col, rain_col]:
                    if df_weather[col].dtype == 'object':
                        df_weather[col] = df_weather[col].astype(str).str.replace(',', '.')
                    df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')

                # 4. Process Timeline
                df_weather['timestamp'] = pd.to_datetime(df_weather[w_time_col], errors='coerce')
                df_weather = df_weather.dropna(subset=['timestamp'])

                # Resample
                w_res = df_weather.set_index('timestamp').resample('1min').ffill().reindex(timeline, method='nearest')

                # 5. Extract Values
                t_val = w_res[temp_col]
                r_val = w_res[rain_col]

                # 6. Create Masks
                # Bad Weather = Rain > 0.2mm OR Temp < 10C
                mask_bad_logic = (r_val > 0.2) | (t_val < 10)

                # Assign to our series
                mask_bad_weather = mask_bad_logic.fillna(False)
                mask_good_weather = ~mask_bad_weather

                print(f"    [Weather] Bad weather detected: {mask_bad_weather.sum()} mins")

        except Exception as e:
            print(f"    [!] Weather Processing Warning: {e}")
            # If this fails, mask_bad_weather remains False (NaN result)
    # =========================================================
    # 8. CORRELATION LOOP
    # =========================================================
    results = []

    for cat_name, type_list in categories.items():
        if type_list:
            subset = hub_trains[hub_trains['train_type'].isin(type_list)]
        else:
            subset = hub_trains

        s_arr, s_dep = generate_signals(subset, timeline, config['params'])

        def get_corr(signal, actuals, mask):
            # Combine availability with the specific weather mask
            valid_idx = mask & mask_available & signal.index.isin(timeline)
            if valid_idx.sum() > 60:
                return signal[valid_idx].corr(actuals[valid_idx])
            return np.nan

        # A. GLOBAL (All Weather)
        mask_all = pd.Series(True, index=timeline)
        r_arr_global = get_corr(s_arr, ts_starts, mask_all)
        r_dep_global = get_corr(s_dep, ts_ends, mask_all)

        # B. IDEAL (Good Weather)
        r_arr_good = get_corr(s_arr, ts_starts, mask_good_weather)
        r_dep_good = get_corr(s_dep, ts_ends, mask_good_weather)

        # C. ADVERSE (Bad Weather)
        r_arr_bad = get_corr(s_arr, ts_starts, mask_bad_weather)
        r_dep_bad = get_corr(s_dep, ts_ends, mask_bad_weather)

        results.append({
            'City': city_name,
            'Category': cat_name,
            'Arr_Corr_Global': r_arr_global,
            'Dep_Corr_Global': r_dep_global,
            'Arr_Corr_Good': r_arr_good,
            'Dep_Corr_Good': r_dep_good,
            'Arr_Corr_Bad': r_arr_bad,
            'Dep_Corr_Bad': r_dep_bad
        })

    return results