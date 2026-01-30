import pandas as pd
import numpy as np
from scipy.stats import gamma
import sys


# ==========================================
# 1. HELPER: COLUMN STANDARDIZER
# ==========================================
def standard_columns(df):
    """
    Renames columns to standard names with robust cleaning.
    """
    df.columns = [str(c).lower().strip().replace('\n', ' ').replace('_', ' ').replace('.', ' ') for c in df.columns]

    # Mappings
    mappings = {
        'actual_arrival': ['skutečný příjezd', 'příjezd', 'příj skut', 'cas pr skut', 'čas příjezdu', 'actual arrival',
                           'prij skut'],
        'actual_departure': ['skutečný odjezd', 'odjezd', 'odj skut', 'cas od skut', 'čas odjezdu', 'actual departure',
                             'odjezd skut'],
        'train_type': ['druh vlaku', 'typ', 'kategorie', 'train type', 'druh'],
        'station': ['stanice', 'nazev stanice', 'název bodu výběru', 'station', 'bod']
    }

    for target, candidates in mappings.items():
        if target in df.columns: continue
        for c in df.columns:
            if any(cand in c for cand in candidates):
                df.rename(columns={c: target}, inplace=True)
                break

    # Fallback for Scheduled if Actual missing
    if 'actual_arrival' not in df.columns:
        for c in df.columns:
            if 'příjezd' in c or 'arrival' in c:
                df.rename(columns={c: 'actual_arrival'}, inplace=True);
                break

    if 'actual_departure' not in df.columns:
        for c in df.columns:
            if 'odjezd' in c or 'departure' in c:
                df.rename(columns={c: 'actual_departure'}, inplace=True);
                break

    return df


# ==========================================
# 2. SIGNAL PROCESSING
# ==========================================
def get_dynamic_kernels(duration_arr, duration_dep):
    # Arrival: Forward looking
    x_arr = np.linspace(0, 10, duration_arr)
    k_arr = gamma.pdf(x_arr, a=2.5, scale=2)
    k_arr = k_arr / k_arr.max() if k_arr.max() > 0 else k_arr

    # Departure: Backward looking
    x_dep = np.linspace(0, 10, duration_dep)
    k_dep = gamma.pdf(x_dep, a=3.5, scale=2)
    k_dep = (k_dep / k_dep.max())[::-1] if k_dep.max() > 0 else k_dep
    return k_arr, k_dep


def generate_signals(subset_trains, timeline, params):
    dur_arr = int(params.get('pulse_arr_reg', 20))
    dur_dep = int(params.get('pulse_dep_reg', 30))
    k_arr, k_dep = get_dynamic_kernels(dur_arr, dur_dep)

    weights = params.get('weights', {})
    sig_arr = pd.Series(0.0, index=timeline)
    sig_dep = pd.Series(0.0, index=timeline)

    boost_am = params.get('rush_am', 1.0)
    boost_pm = params.get('rush_pm', 1.0)

    # Pre-calculate timeline index for O(1) lookups
    timeline_idx = set(timeline)

    for t in subset_trains.itertuples(index=False):
        t_type = str(getattr(t, 'train_type', 'Os')).strip()
        w = weights.get(t_type, 1.0)

        # Arrival
        arr_val = getattr(t, 'actual_arrival', pd.NaT)
        if pd.notnull(arr_val):
            t_time = arr_val.round('min')
            if t_time in timeline_idx:
                try:
                    start_idx = sig_arr.index.get_loc(t_time)
                    # Time Boost
                    c_boost = boost_am if (6 <= t_time.hour <= 9) else 1.0
                    end_idx = min(start_idx + dur_arr, len(sig_arr))
                    length = end_idx - start_idx
                    if length > 0:
                        sig_arr.iloc[start_idx:end_idx] += (k_arr[:length] * w * c_boost)
                except:
                    pass

        # Departure
        dep_val = getattr(t, 'actual_departure', pd.NaT)
        if pd.notnull(dep_val):
            t_time = dep_val.round('min')
            if t_time in timeline_idx:
                try:
                    end_idx = sig_dep.index.get_loc(t_time) + 1
                    c_boost = boost_pm if (14 <= t_time.hour <= 18) else 1.0
                    start_idx = max(end_idx - dur_dep, 0)
                    length = end_idx - start_idx
                    if length > 0:
                        sig_dep.iloc[start_idx:end_idx] += (k_dep[-length:] * w * c_boost)
                except:
                    pass

    return sig_arr, sig_dep


# ==========================================
# 3. MATH HELPER
# ==========================================
def weighted_pearson_corr(x, y, w):
    def weighted_mean(v, w): return np.sum(v * w) / np.sum(w)

    def weighted_cov(x, y, w, xm, ym): return np.sum(w * (x - xm) * (y - ym)) / np.sum(w)

    xm = weighted_mean(x, w)
    ym = weighted_mean(y, w)
    cov_xy = weighted_cov(x, y, w, xm, ym)
    cov_xx = weighted_cov(x, x, w, xm, xm)
    cov_yy = weighted_cov(y, y, w, ym, ym)

    if cov_xx == 0 or cov_yy == 0: return np.nan
    return cov_xy / np.sqrt(cov_xx * cov_yy)


# ==========================================
# 4. ANALYSIS RUNNER
# ==========================================
def analyze_city_data(city_name, config, df_trains, df_bikes, df_logs, df_weather, categories):
    print(f"  > Processing Algorithms for {city_name}...")
    df_trains = standard_columns(df_trains)

    if 'actual_arrival' not in df_trains.columns:
        print("    [!] Skipping: 'actual_arrival' column missing.")
        return []

    if 'station' in df_trains.columns:
        hub_trains = df_trains[
            df_trains['station'].astype(str).str.contains(config['train_station'], case=False, na=False)]
    else:
        hub_trains = df_trains

    if hub_trains.empty: return []

    df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]
    valid_stations = [s.lower() for s in config['bike_stations']]

    t_min = min(hub_trains['actual_arrival'].min(), df_bikes['start_time'].min())
    t_max = max(hub_trains['actual_arrival'].max(), df_bikes['end_time'].max())
    timeline = pd.date_range(t_min.floor('h'), t_max.ceil('h'), freq='1min')

    # --- FLOWS ---
    starts = df_bikes[df_bikes['start_place'].str.lower().isin(valid_stations)]
    ts_starts = starts.set_index('start_time').resample('1min').size().reindex(timeline, fill_value=0)

    ends = df_bikes[df_bikes['end_place'].str.lower().isin(valid_stations)]
    ts_ends = ends.set_index('end_time').resample('1min').size().reindex(timeline, fill_value=0)

    ts_total = ts_starts + ts_ends

    # --- MASKS & WEIGHTS ---
    mask_available = pd.Series(True, index=timeline)
    obs_weights = pd.Series(1.0, index=timeline)
    mask_bad_weather = pd.Series(False, index=timeline)

    if df_weather is not None and not df_weather.empty:
        try:
            df_weather.columns = [str(c).lower().strip() for c in df_weather.columns]
            t_col = next((c for c in df_weather.columns if c in ['datetime', 'date', 'cas']), None)
            p_col = next((c for c in df_weather.columns if c in ['precip', 'rain', 'srazky']), None)

            if t_col and p_col:
                df_weather['ts'] = pd.to_datetime(df_weather[t_col]).dt.round('1min')
                w_res = df_weather.set_index('ts').resample('1min').ffill().reindex(timeline, method='ffill')
                mask_bad_weather = (w_res[p_col] > 0.2)
                obs_weights[mask_bad_weather] = 0.5
        except:
            pass

    # Time Masks
    hours = timeline.hour
    mask_workday = timeline.dayofweek < 5
    mask_weekend = ~mask_workday
    mask_am_rush = (hours >= 6) & (hours < 9)
    mask_pm_rush = (hours >= 14) & (hours < 18)

    # Combined Contexts
    mask_workday_am = mask_workday & mask_am_rush
    mask_workday_pm = mask_workday & mask_pm_rush
    mask_good_weather = ~mask_bad_weather

    # --- CORRELATION LOOP ---
    results = []

    for cat_name, type_list in categories.items():
        if type_list:
            subset = hub_trains[hub_trains['train_type'].isin(type_list)]
        else:
            subset = hub_trains

        if subset.empty: continue

        s_arr, s_dep = generate_signals(subset, timeline, config['params'])
        s_total = s_arr + s_dep

        # Generic Correlation Runner
        def run_corr(sig, act, ctx_mask):
            valid_mask = mask_available & ctx_mask
            s = sig[valid_mask]
            a = act[valid_mask]
            w = obs_weights[valid_mask]
            if len(s) < 30 or s.std() == 0 or a.std() == 0: return np.nan
            return weighted_pearson_corr(s.values, a.values, w.values)

        # 1. GLOBAL (Arr+Dep vs Total Movement)
        r_glob_all = run_corr(s_total, ts_total, pd.Series(True, index=timeline))
        r_glob_wd = run_corr(s_total, ts_total, mask_workday)
        r_glob_wk = run_corr(s_total, ts_total, mask_weekend)
        r_glob_am = run_corr(s_total, ts_total, mask_workday_am)
        r_glob_pm = run_corr(s_total, ts_total, mask_workday_pm)
        r_glob_good = run_corr(s_total, ts_total, mask_good_weather)
        r_glob_bad = run_corr(s_total, ts_total, mask_bad_weather)

        # 2. ARRIVAL (Arr vs Rentals)
        r_arr_all = run_corr(s_arr, ts_starts, pd.Series(True, index=timeline))
        r_arr_wd = run_corr(s_arr, ts_starts, mask_workday)
        r_arr_wk = run_corr(s_arr, ts_starts, mask_weekend)

        # 3. DEPARTURE (Dep vs Returns)
        r_dep_all = run_corr(s_dep, ts_ends, pd.Series(True, index=timeline))
        r_dep_wd = run_corr(s_dep, ts_ends, mask_workday)
        r_dep_wk = run_corr(s_dep, ts_ends, mask_weekend)

        results.append({
            'City': city_name,
            'Category': cat_name,

            # Global
            'Global_All': r_glob_all,
            'Global_Workday': r_glob_wd,
            'Global_Weekend': r_glob_wk,
            'Global_AM_Rush': r_glob_am,
            'Global_PM_Rush': r_glob_pm,
            'Global_GoodWeather': r_glob_good,
            'Global_BadWeather': r_glob_bad,

            # Directional
            'Arr_All': r_arr_all,
            'Arr_Workday': r_arr_wd,
            'Arr_Weekend': r_arr_wk,

            'Dep_All': r_dep_all,
            'Dep_Workday': r_dep_wd,
            'Dep_Weekend': r_dep_wk
        })

    return results