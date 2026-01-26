import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
from scipy.signal import correlate
from scipy.signal.windows import gaussian  # <--- FIXED IMPORT HERE


# ==========================================
# 0. HELPER FUNCTIONS
# ==========================================
def standard_columns(df):
    """Renames Czech columns to standard English names."""
    rename_map = {
        'název bodu výběru': 'station', 'číslo vlaku': 'train_no',
        'druh vlaku': 'train_type', 'název výchozí stanice vlaku': 'origin',
        'název cílové stanice vlaku': 'destination', 'čas příjezdu': 'actual_arrival',
        'čas odjezdu': 'actual_departure', 'datum': 'date'
    }
    df.columns = [str(c).lower().strip() for c in df.columns]
    new_cols = {}
    for col in df.columns:
        for key, val in rename_map.items():
            if key in col:
                new_cols[col] = val
                break
    return df.rename(columns=new_cols)


def generate_signals(trains, timeline, params):
    """Generates smoothed pressure signals for Arrivals and Departures."""
    s_arr = pd.Series(0.0, index=timeline)
    s_dep = pd.Series(0.0, index=timeline)
    weights = params.get('weights', {'Os': 1, 'Sp': 1, 'R': 1.5, 'Ex': 2})

    for _, train in trains.iterrows():
        w = weights.get(train.get('train_type', 'Os'), 1.0)
        if pd.notnull(train.get('actual_arrival')):
            try:
                s_arr.at[train['actual_arrival'].round('min')] += w
            except:
                pass
        if pd.notnull(train.get('actual_departure')):
            try:
                s_dep.at[train['actual_departure'].round('min')] += w
            except:
                pass

    # Gaussian smoothing
    window_size = 30
    gauss_kernel = gaussian(window_size, std=4)
    gauss_kernel /= gauss_kernel.sum()
    s_arr_smooth = pd.Series(np.convolve(s_arr.values, gauss_kernel, mode='same'), index=timeline)
    s_dep_smooth = pd.Series(np.convolve(s_dep.values, gauss_kernel, mode='same'), index=timeline)
    return s_arr_smooth, s_dep_smooth


# ==========================================
# 1. CONFIGURATION
# ==========================================
CITY_KEY = 'Ostrava-Svinov'
VIS_START = '2025-09-15'
VIS_END = '2025-10-19'
ZOOM_DAY_START = '2025-09-27 00:00'
ZOOM_DAY_END = '2025-09-27 23:59'

FILE_TRAINS = 'data/Pohyby_Svinov.xlsx'
FILE_BIKES = 'data/nextbike_data_VSB_Vaclavik.xlsx'
FILE_WEATHER = 'data/Ostrava_pocasi.xlsx'
SHEET_RENTALS = 'Vypujcky_Ostrava'

STATION_CONFIG = {
    'train_station': 'Ostrava-Svinov',
    'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
    'max_wait_time': 45,
    'params': {
        'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1}
    }
}

# ==========================================
# 2. DATA LOADING
# ==========================================
print(f"📊 Generating Full Dashboard (Weather + Type Split)...")

# Load Trains
try:
    df_preview = pd.read_excel(FILE_TRAINS, header=None, nrows=20)
    header_idx = 0
    for idx, row in df_preview.iterrows():
        row_str = row.astype(str).str.lower().values
        if any('čas' in x or 'cas' in x or 'druh' in x for x in row_str):
            header_idx = idx
            break
    df_trains = pd.read_excel(FILE_TRAINS, header=header_idx)
    df_trains = standard_columns(df_trains)
    df_trains['actual_arrival'] = pd.to_datetime(df_trains['actual_arrival'], errors='coerce')
    df_trains['actual_departure'] = pd.to_datetime(df_trains['actual_departure'], errors='coerce')
except:
    print("Error loading trains.")
    exit()

# Load Bikes
df_bikes = pd.read_excel(FILE_BIKES, sheet_name=SHEET_RENTALS)
df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]
df_bikes['start_time'] = pd.to_datetime(df_bikes['start_time'])
df_bikes['end_time'] = pd.to_datetime(df_bikes['end_time'])

# Load Weather
try:
    df_weather = pd.read_excel(FILE_WEATHER)
    df_weather.columns = [str(c).lower().strip() for c in df_weather.columns]
    df_weather['timestamp'] = pd.to_datetime(df_weather['datetime'])
    df_weather['date_only'] = df_weather['timestamp'].dt.date
    daily_weather = df_weather.groupby('date_only').agg({'precip': 'sum', 'temp': 'mean'})
except:
    print("Warning: Weather file not found or invalid. Using dummy data.")
    dates = pd.date_range(VIS_START, VIS_END).date
    daily_weather = pd.DataFrame({'precip': 0}, index=dates)

# Global Filter
zoom_start = pd.to_datetime(VIS_START)
zoom_end = pd.to_datetime(VIS_END)
df_bikes = df_bikes[(df_bikes['start_time'] >= zoom_start) & (df_bikes['end_time'] <= zoom_end)].copy()
df_trains_filtered = df_trains.copy()

# ==========================================
# 3. ANALYSIS PART A: HEATMAPS (Regional vs LongDist)
# ==========================================
connections = []
df_arr_sorted = df_trains_filtered.dropna(subset=['actual_arrival']).sort_values('actual_arrival')
svinov_rentals = df_bikes[(df_bikes['start_place'].isin(STATION_CONFIG['bike_stations']))]

for idx, row in svinov_rentals.iterrows():
    rental_time = row['start_time']
    window_start = rental_time - pd.Timedelta(minutes=STATION_CONFIG['max_wait_time'])

    start_idx = df_arr_sorted['actual_arrival'].searchsorted(window_start)
    end_idx = df_arr_sorted['actual_arrival'].searchsorted(rental_time)
    potential_trains = df_arr_sorted.iloc[start_idx:end_idx]

    if not potential_trains.empty:
        for _, train in potential_trains.iterrows():
            lag = (rental_time - train['actual_arrival']).total_seconds() / 60.0
            cat = 'Regional' if train['train_type'] in ['Os', 'Sp'] else 'LongDist'

            connections.append({
                'time_of_day': rental_time.hour + rental_time.minute / 60.0,
                'diff_time': lag,
                'cat': cat
            })
df_conns = pd.DataFrame(connections)

# ==========================================
# 4. ANALYSIS PART B: MICRO-ZOOM
# ==========================================
z_start = pd.to_datetime(ZOOM_DAY_START)
z_end = pd.to_datetime(ZOOM_DAY_END)
timeline_1min = pd.date_range(z_start, z_end, freq='1min')

# Signals
t_mask = (df_trains['actual_arrival'] >= z_start) & (df_trains['actual_arrival'] <= z_end)
subset_trains = df_trains[t_mask].copy()
s_arr, s_dep = generate_signals(subset_trains, timeline_1min, STATION_CONFIG['params'])
sig_total = s_arr + s_dep

# Bikes
b_starts = df_bikes[(df_bikes['start_time'] >= z_start) & (df_bikes['start_time'] <= z_end) & (
    df_bikes['start_place'].isin(STATION_CONFIG['bike_stations']))]
b_ends = df_bikes[(df_bikes['end_time'] >= z_start) & (df_bikes['end_time'] <= z_end) & (
    df_bikes['end_place'].isin(STATION_CONFIG['bike_stations']))]

ts_rentals = b_starts.set_index('start_time').resample('1min').size().reindex(timeline_1min, fill_value=0)
ts_returns = b_ends.set_index('end_time').resample('1min').size().reindex(timeline_1min, fill_value=0)

# Correlation
if s_arr.std() > 0 and ts_rentals.std() > 0:
    norm_train = (s_arr - s_arr.mean()) / s_arr.std()
    norm_bike = (ts_rentals - ts_rentals.mean()) / ts_rentals.std()
    lags = np.arange(0, 30)
    corr = [norm_train.corr(norm_bike.shift(l)) for l in lags]
    best_lag = lags[np.argmax(corr)]
    max_corr = max(corr)
else:
    best_lag, max_corr = 0, 0

# ==========================================
# 5. COMPOSITE PLOTTING
# ==========================================
fig = plt.figure(figsize=(16, 16))
# 3 Rows: Weather (small), Heatmaps (medium), Heartbeat (medium)
gs = gridspec.GridSpec(3, 2, height_ratios=[0.3, 1, 1], hspace=0.3)

# --- ROW 1: WEATHER ---
ax_weather = plt.subplot(gs[0, :])
unique_dates = pd.date_range(zoom_start, zoom_end).date
rain_vals = [daily_weather.loc[d, 'precip'] if d in daily_weather.index else 0 for d in unique_dates]
x_indices = np.arange(len(unique_dates))
ax_weather.bar(x_indices, rain_vals, color='#4682b4', alpha=0.7)
ax_weather.set_title("Daily Rainfall (Context for Heatmaps)", fontsize=10, fontweight='bold', loc='left')
ax_weather.set_ylabel("Rain (mm)")
ax_weather.set_xlim(-0.5, len(unique_dates) - 0.5)
ax_weather.set_xticks([])  # Hide dates
ax_weather.grid(axis='y', linestyle='--', alpha=0.5)

# --- ROW 2: HEATMAPS (Reg vs LongDist) ---
ax1 = plt.subplot(gs[1, 0])
reg_data = df_conns[df_conns['cat'] == 'Regional']
if not reg_data.empty:
    h1 = ax1.hist2d(reg_data['time_of_day'], reg_data['diff_time'], bins=[24, 45], range=[[0, 24], [0, 45]],
                    cmap='Greens', cmin=1)
    plt.colorbar(h1[3], ax=ax1, label='Density')
ax1.set_title("A. Regional Sync (Commuters)", fontsize=12, fontweight='bold', loc='left')
ax1.set_ylabel("Transfer Time (min)")
ax1.set_xlabel("Time of Day (Hour)")
ax1.set_xticks(np.arange(0, 25, 4))

ax2 = plt.subplot(gs[1, 1])
ld_data = df_conns[df_conns['cat'] == 'LongDist']
if not ld_data.empty:
    h2 = ax2.hist2d(ld_data['time_of_day'], ld_data['diff_time'], bins=[24, 45], range=[[0, 24], [0, 45]], cmap='Blues',
                    cmin=1)
    plt.colorbar(h2[3], ax=ax2, label='Density')
ax2.set_title("B. Long Dist Sync (Travelers)", fontsize=12, fontweight='bold', loc='left')
ax2.set_xlabel("Time of Day (Hour)")
ax2.set_xticks(np.arange(0, 25, 4))

# --- ROW 3: HEARTBEAT (Zoom) ---
ax3 = plt.subplot(gs[2, :])
ax3_twin = ax3.twinx()

# 1. Total Pressure
ax3.fill_between(timeline_1min, sig_total, color='#d55e00', alpha=0.15, label='Total Pressure')
ax3.plot(timeline_1min, sig_total, color='#d55e00', linewidth=1.5, alpha=0.8)

# 2. Train Markers (Arr/Dep)
marker_offset = 0
arr_events = subset_trains.dropna(subset=['actual_arrival'])
dep_events = subset_trains.dropna(subset=['actual_departure'])

ax3.scatter(arr_events['actual_arrival'], [marker_offset] * len(arr_events),
            marker='v', s=60, color='#d55e00', edgecolors='white', linewidth=0.5, label='Arrival', zorder=10)
ax3.scatter(dep_events['actual_departure'], [marker_offset] * len(dep_events),
            marker='^', s=60, color='#8B0000', edgecolors='white', linewidth=0.5, label='Departure', zorder=10)

# 3. Bike Bars (0-10 Axis)
ax3_twin.bar(timeline_1min, ts_rentals, width=0.0006, color='#2ca02c', alpha=0.7, label='Rentals', align='center')
ax3_twin.bar(timeline_1min, ts_returns, width=0.0006, color='#1f77b4', alpha=0.7, label='Returns', align='center',
             bottom=ts_rentals)

ax3_twin.set_ylim(0, 10)  # <--- FIXED 0-10
ax3_twin.yaxis.set_major_locator(MaxNLocator(integer=True))

# Formatting
ax3.set_ylabel("Passenger Pressure", color='#d55e00', fontweight='bold')
ax3.set_ylim(0, sig_total.max() * 1.2 if sig_total.max() > 0 else 1)
ax3.tick_params(axis='y', labelcolor='#d55e00')
ax3_twin.set_ylabel("Bike Qty", color='#333333', fontweight='bold')
ax3_twin.tick_params(axis='y', labelcolor='#333333')

ax3.set_title(f"C. Rush Hour Zoom ({z_start.strftime('%a %d.%m %H:%M')} - {z_end.strftime('%H:%M')})", fontsize=12,
              fontweight='bold', loc='left')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3.set_xlim(z_start, z_end)
ax3.grid(True, alpha=0.25)

# Combined Legend
from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color='#d55e00', lw=2, label='Pressure'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#d55e00', markersize=8, label='Arrival'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#8B0000', markersize=8, label='Departure'),
    Line2D([0], [0], color='#2ca02c', lw=4, alpha=0.7, label='Rental'),
    Line2D([0], [0], color='#1f77b4', lw=4, alpha=0.7, label='Return')
]
ax3.legend(handles=custom_lines, loc='upper left', ncol=5, frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig(f'results/Dashboard_Full_{CITY_KEY}.png', dpi=300)
print(f"✅ Saved Full Dashboard to results/Dashboard_Full_{CITY_KEY}.png")
plt.show()