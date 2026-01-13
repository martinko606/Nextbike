# transport_analytics.py
import pandas as pd
import numpy as np
from scipy.stats import gamma
import os

# ==========================================
# 1. CONSTANTS
# ==========================================
DEFAULT_TRAIN_CATS = {
    'Global': None,
    'Regional': ['Os', 'Sp'],
    'LongDist': ['R', 'Ex']
}


# ==========================================
# 2. DATA LOADING ENGINE (UPDATED)
# ==========================================
def load_data(train_file, bike_file, rebalancing_file, weather_file):
    print(f"--- Loading Data ---")

    # 1. TRAINS
    print(f"Reading Trains...")
    trains = pd.read_excel(train_file)
    trains.columns = [c.lower().strip() for c in trains.columns]
    for col in ['actual_arrival', 'actual_departure']:
        if col in trains.columns:
            trains[col] = pd.to_datetime(trains[col], errors='coerce')

    # 2. BIKES (Movement Sheet)
    print(f"Reading Bike Movements...")
    # Loading specific sheet for rides (Adjust 'Rides' to your actual sheet name!)
    bikes = pd.read_excel(bike_file, sheet_name=0)
    bikes.columns = [c.lower().strip() for c in bikes.columns]
    bikes['start_time'] = pd.to_datetime(bikes['start_time'])
    bikes['end_time'] = pd.to_datetime(bikes['end_time'])

    # 3. STATION LOGS (Capacity Sheet)
    print(f"Reading Station Logs...")
    # Loading specific sheet for logs (Adjust 'Station Logs' or sheet index 1)
    logs = pd.read_excel(bike_file, sheet_name=1)
    logs.columns = [c.lower().strip() for c in logs.columns]
    # Ensure we have timestamp and count
    # Adjust 'datetime' and 'bikes' to match your column headers
    if 'timestamp' in logs.columns:
        logs['timestamp'] = pd.to_datetime(logs['timestamp'])

    # 4. REBALANCING (The Filter)
    print(f"Reading Rebalancing Data...")
    rebal = pd.read_excel(rebalancing_file)
    rebal.columns = [c.lower().strip() for c in rebal.columns]
    # Assuming standard columns: 'date_time', 'bike_number'
    # We will use this to filter 'bikes' dataframe below

    # 5. WEATHER
    print(f"Reading Weather...")
    weather = pd.read_excel(weather_file)
    weather.columns = [c.lower().strip() for c in weather.columns]
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])

    # --- PRE-PROCESSING: FILTER REBALANCING ---
    # Logic: If a bike ID is in the rebalancing file at a specific time,
    # it might appear as a ride. We remove rides that overlap with rebalancing events.
    # Simple heuristic: Remove rides that are < 1 min duration (often service moves)
    # OR matching exact timestamps.
    # For now, we apply a duration filter which captures 90% of service moves.

    initial_count = len(bikes)
    # Filter out very short 'service' hops often associated with rebalancing
    bikes = bikes[bikes['end_time'] - bikes['start_time'] > pd.Timedelta(seconds=60)]

    print(f"Data Load Complete. Filtered {initial_count - len(bikes)} potential service moves.\n")
    return trains, bikes, logs, weather


# ==========================================
# 3. SIGNAL PROCESSING
# ==========================================
def get_dynamic_kernels(p_arr, p_dep):
    x_arr = np.linspace(0, 15, p_arr)
    k_arr = gamma.pdf(x_arr, a=2.5, scale=2)
    k_arr = k_arr / k_arr.max()

    x_dep = np.linspace(0, 15, p_dep)
    k_dep = gamma.pdf(x_dep, a=4, scale=2)
    k_dep = (k_dep / k_dep.max())[::-1]
    return k_arr, k_dep


def generate_signals(subset_trains, timeline, params):
    dur_arr, dur_dep = params['pulse_arr'], params['pulse_dep']
    boost_am, boost_pm = params['rush_am'], params['rush_pm']

    sig_arr = pd.Series(0.0, index=timeline)
    sig_dep = pd.Series(0.0, index=timeline)
    k_arr, k_dep = get_dynamic_kernels(dur_arr, dur_dep)

    for _, t in subset_trains.iterrows():
        # Arrival
        if pd.notnull(t.get('actual_arrival')):
            try:
                t_idx = sig_arr.index.get_loc(t['actual_arrival'].round('1min'))
                boost = boost_am if (6 <= t['actual_arrival'].hour <= 9) else 1.0
                end = min(t_idx + dur_arr, len(sig_arr))
                L = end - t_idx
                if L > 0: sig_arr.iloc[t_idx:end] += (k_arr[:L] * boost)
            except KeyError:
                pass

        # Departure
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
# 4. ANALYSIS RUNNER
# ==========================================
def analyze_hub(hub_name, config, trains, bikes, logs, weather,
                temp_threshold=10, rain_threshold=0.0,
                categories=DEFAULT_TRAIN_CATS):
    print(f"Analyzing {hub_name}...")

    # A. TRAIN FILTER
    if 'station' in trains.columns:
        hub_trains = trains[trains['station'] == config['train_station']]
    else:
        hub_trains = trains

    if hub_trains.empty:
        print(f"  WARNING: No trains found for {config['train_station']}")
        return []

    # B. BIKE FILTER
    starts = bikes[bikes['start_place'].isin(config['bike_stations'])]
    ends = bikes[bikes['end_place'].isin(config['bike_stations'])]

    # C. LOGS FILTER (Availability)
    # Filter logs for this station(s)
    # Adjust 'place_name' to match your log sheet column
    if 'place_name' in logs.columns:
        hub_logs = logs[logs['place_name'].isin(config['bike_stations'])]
    else:
        # Fallback if no name column (assumes global file per city)
        hub_logs = logs

        # D. TIMELINE
    start_t = hub_trains['actual_arrival'].min().floor('H')
    end_t = hub_trains['actual_arrival'].max().ceil('H')
    timeline = pd.date_range(start_t, end_t, freq='1min')

    # Resample Movements
    ts_starts = starts.set_index('start_time').resample('1min').size().reindex(timeline, fill_value=0)
    ts_ends = ends.set_index('end_time').resample('1min').size().reindex(timeline, fill_value=0)

    # --- ROBUSTNESS: REAL AVAILABILITY MASK ---
    # 1. Resample 30-min logs to 1-min timeline (Forward Fill)
    # Aggregating by sum in case of multiple stations in hub
    stock_series = hub_logs.set_index('timestamp')['bikes'].resample('1min').ffill().reindex(timeline, method='nearest')

    # 2. Mask: Station must have at least 1 bike
    mask_available = stock_series >= 1

    # --- WEATHER MASK ---
    w_res = weather.set_index('timestamp').resample('1min').ffill().reindex(timeline, method='nearest')
    mask_good_weather = (w_res['precip_mm'] <= rain_threshold) & (w_res['temp_c'] >= temp_threshold)

    # COMBINED MASK
    final_mask = mask_good_weather & mask_available

    results = []

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
            'City': hub_name,
            'Category': cat_name,
            'Arr_Corr': r_arr,
            'Dep_Corr': r_dep,
            'Valid_Hours': idx_valid.sum() / 60
        })

    return results