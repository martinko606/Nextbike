import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy.signal.windows import gaussian


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

FILE_TRAINS = 'data/Pohyby_Svinov.xlsx'
FILE_BIKES = 'data/nextbike_data_VSB_Vaclavik.xlsx'
SHEET_RENTALS = 'Vypujcky_Ostrava'

STATION_CONFIG = {
    'train_station': 'Ostrava-Svinov',
    'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
    'params': {
        'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1}
    }
}

# ==========================================
# 2. DATA LOADING
# ==========================================
print(f"📊 Generating Flow Balance Map for {CITY_KEY}...")

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
except Exception as e:
    print(f"Error loading trains: {e}")
    exit()

# Load Bikes
df_bikes = pd.read_excel(FILE_BIKES, sheet_name=SHEET_RENTALS)
df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]
df_bikes['start_time'] = pd.to_datetime(df_bikes['start_time'])
df_bikes['end_time'] = pd.to_datetime(df_bikes['end_time'])

# ==========================================
# 3. DATA PROCESSING
# ==========================================
zoom_start = pd.to_datetime(VIS_START)
zoom_end = pd.to_datetime(VIS_END)
timeline_1min = pd.date_range(zoom_start, zoom_end, freq='1min')

# --- A. Prepare Background Heatmap (Train Pressure) ---
t_mask = (df_trains['actual_arrival'] >= zoom_start) & (df_trains['actual_arrival'] <= zoom_end)
subset_trains = df_trains[t_mask].copy()

# Generate Signal
s_arr, s_dep = generate_signals(subset_trains, timeline_1min, STATION_CONFIG['params'])
sig_total = s_arr + s_dep

# Create Matrix (Resample to 15min)
df_sig = pd.DataFrame({'pressure': sig_total, 'ts': timeline_1min})
df_sig = df_sig.set_index('ts').resample('15min').mean()
df_sig['date'] = df_sig.index.date
df_sig['time_dec'] = df_sig.index.hour + df_sig.index.minute / 60.0

# Pivot: Rows=Time, Cols=Date
heatmap_data = df_sig.pivot(index='time_dec', columns='date', values='pressure').sort_index(ascending=True)

# --- B. Prepare Foreground Pie Clusters (Bike Flow) ---
BIN_SIZE = '15min'

# Filter bikes to station and window
b_starts = df_bikes[(df_bikes['start_time'] >= zoom_start) & (df_bikes['start_time'] <= zoom_end) &
                    (df_bikes['start_place'].isin(STATION_CONFIG['bike_stations']))]
b_ends = df_bikes[(df_bikes['end_time'] >= zoom_start) & (df_bikes['end_time'] <= zoom_end) &
                  (df_bikes['end_place'].isin(STATION_CONFIG['bike_stations']))]

# Create "Events" DataFrame for clustering
events = []
for t in b_starts['start_time']: events.append({'time': t, 'type': 'rental'})
for t in b_ends['end_time']: events.append({'time': t, 'type': 'return'})
df_events = pd.DataFrame(events)

if df_events.empty:
    print("No bike events found in this window/station.")
    exit()

# Binning
df_events['bin_time'] = df_events['time'].dt.round(BIN_SIZE)
cluster_counts = df_events.groupby(['bin_time', 'type']).size().unstack(fill_value=0)

# Ensure columns exist
if 'rental' not in cluster_counts.columns: cluster_counts['rental'] = 0
if 'return' not in cluster_counts.columns: cluster_counts['return'] = 0
cluster_counts['total'] = cluster_counts['rental'] + cluster_counts['return']
cluster_counts = cluster_counts[cluster_counts['total'] > 0]

# ==========================================
# 4. PLOTTING
# ==========================================
fig = plt.figure(figsize=(24, 12))
ax = plt.subplot(111)

# --- A. DRAW HEATMAP (Background) ---
# Use seaborn for the grid. Note: This sets the axis coordinates to 0, 1, 2...
sns.heatmap(heatmap_data, ax=ax, cmap='Oranges', cbar_kws={'label': 'Train Passenger Pressure'}, alpha=0.6)

# --- B. DRAW PIE CHART MARKERS (Foreground) ---
# We must map Date/Time to the Heatmap's coordinate system
# X-Axis: 0.5, 1.5, ... corresponding to columns (dates)
# Y-Axis: 0 to N rows (time).
# Heatmap rows are indices of `heatmap_data`.
# We need to calculate which row index corresponds to a specific hour.

unique_dates = heatmap_data.columns
date_map = {d: i + 0.5 for i, d in enumerate(unique_dates)}

# Time Mapping:
# heatmap_data index is 'time_dec' (e.g. 0.0, 0.25, 0.5 ... 23.75)
# We need to find the integer row location for a given time
time_index_map = {t: i for i, t in enumerate(heatmap_data.index)}

for ts, row in cluster_counts.iterrows():
    d = ts.date()

    # 1. Get X Coordinate
    if d not in date_map: continue
    x_pos = date_map[d]

    # 2. Get Y Coordinate
    # Find closest time bin in heatmap index
    t_dec = ts.hour + ts.minute / 60.0
    # Round to nearest 0.25 (since we resampled to 15min)
    t_rounded = round(t_dec * 4) / 4
    if t_rounded >= 24.0: t_rounded = 23.75  # Cap at midnight

    if t_rounded in time_index_map:
        y_pos = time_index_map[t_rounded] + 0.5  # Center in cell
    else:
        continue

    # 3. Data for Pie
    n_rentals = row['rental']
    total = row['total']

    # 4. Size Scaling (Square root scaling for area)
    # Adjust multiplier (0.35) to change max bubble size
    radius = 0.35 * (total ** 0.5)
    if radius > 0.8: radius = 0.8  # Prevent overlapping columns

    # --- DRAWING ---
    # 1. Base Circle (Returns/Blue) - representing the "whole" if it was 100% returns
    # We use zorder to ensure it sits on top of heatmap
    circle = mpatches.Circle((x_pos, y_pos), radius, facecolor='#1f77b4', edgecolor='white', linewidth=0.5, zorder=10)
    ax.add_patch(circle)

    # 2. Wedge (Rentals/Green)
    if n_rentals > 0:
        ratio = n_rentals / total
        theta = 360 * ratio
        # Draw wedge starting from top (90 deg) going counter-clockwise
        wedge = mpatches.Wedge((x_pos, y_pos), radius, 90, 90 + theta, facecolor='#2ca02c', zorder=11)
        ax.add_patch(wedge)

# --- C. FORMATTING ---
# X-Axis Labels (Dates)
ax.set_xticks(np.arange(len(unique_dates)) + 0.5)
ax.set_xticklabels([d.strftime('%a %d.%m') for d in unique_dates], rotation=45, ha='right')
ax.set_xlabel("")

# Y-Axis Labels (Times)
# Show a tick every 4 hours.
# 4 hours = 16 slots (since 15min bins)
y_ticks = np.arange(0, len(heatmap_data), 16)
y_labels = [f"{int(t):02d}:00" for t in heatmap_data.index[y_ticks]]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, rotation=0)
ax.set_ylabel("Time of Day")
ax.invert_yaxis()  # Ensure 00:00 is at top if desired (Standard heatmap is usually 0 at bottom, check preference)
# Actually, seaborn heatmap 0 index is at TOP by default. So 00:00 is Top.
# Let's NOT invert, otherwise 00:00 goes to bottom.

# Title & Legend
plt.title(f"Commuter Flow Balance: {CITY_KEY}\n(Background: Train Pressure | Pie Charts: Green=Rentals, Blue=Returns)",
          fontsize=16)

legend_elements = [
    mpatches.Patch(facecolor='#2ca02c', edgecolor='white', label='Rentals (Outflow)'),
    mpatches.Patch(facecolor='#1f77b4', edgecolor='white', label='Returns (Inflow)'),
    plt.scatter([], [], s=150, c='gray', alpha=0.5, label='Size = Traffic Volume')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
output_file = f'results/Flow_Balance_Map_{CITY_KEY}.png'
plt.savefig(output_file, dpi=300)
print(f"✅ Saved Flow Balance Map to {output_file}")
plt.show()