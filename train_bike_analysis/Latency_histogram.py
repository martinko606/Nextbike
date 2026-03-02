import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import transport_analytics as ta

# ==========================================
# 1. CONFIGURATION
# ==========================================
CITY_KEY = 'Ostrava-Svinov'
VIS_START = '2025-09-15'
VIS_END = '2025-10-19'

FILE_TRAINS = '../data/Pohyby_Svinov.xlsx'
FILE_BIKES = '../data/nextbike_data_VSB_Vaclavik.xlsx'
SHEET_RENTALS = 'Vypujcky_Ostrava'

STATION_CONFIG = {
    'train_station': 'Ostrava-Svinov',
    'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
    # We will analyze ALL potential connections
    'max_wait_time': 45  # Look at all trains departing within 45 mins of bike return
}

# ==========================================
# 2. DATA LOADING
# ==========================================
print(f"⏱️ Calculating Transfer Probabilities for {CITY_KEY}...")

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
    df_trains = ta.standard_columns(df_trains)
except:
    print("Error loading trains.")
    exit()

# Load Bikes
df_bikes = pd.read_excel(FILE_BIKES, sheet_name=SHEET_RENTALS)
df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]
df_bikes['start_time'] = pd.to_datetime(df_bikes['start_time'])
df_bikes['end_time'] = pd.to_datetime(df_bikes['end_time'])

# Filter Window
zoom_start = pd.to_datetime(VIS_START)
zoom_end = pd.to_datetime(VIS_END)
df_bikes_end = df_bikes[(df_bikes['end_time'] >= zoom_start) & (df_bikes['end_time'] <= zoom_end)].copy()
df_trains = df_trains[
    (df_trains['actual_departure'] >= zoom_start) & (df_trains['actual_departure'] <= zoom_end)].copy()

# ==========================================
# 3. CALCULATE "ALL-TO-ALL" POTENTIAL CONNECTIONS
# ==========================================
# Instead of "nearest", we find ALL trains that departed 0-45 mins after a bike return.
# This avoids the "wrong train" assumption by showing all possibilities.

# To do this efficiently, we use a cross-join or iterative search within window
# Since dataset might be large, we iterate by day or use a smart apply.

print("   -> Finding all potential connections (0-45 min window)...")

connections = []

# Sort for speed
df_trains = df_trains.sort_values('actual_departure')
df_bikes_end = df_bikes_end.sort_values('end_time')

# Iterate through bike returns (this can take a moment for large data, but is accurate)
# Optimization: Limit columns
train_times = df_trains['actual_departure'].values
train_types = df_trains['train_type'].values

# Only look at returns at Svinov
svinov_returns = df_bikes_end[df_bikes_end['end_place'].isin(STATION_CONFIG['bike_stations'])]

for idx, row in svinov_returns.iterrows():
    bike_time = row['end_time']

    # Define window
    window_start = bike_time
    window_end = bike_time + pd.Timedelta(minutes=STATION_CONFIG['max_wait_time'])

    # Find trains in this window
    # Searchsorted finds the index where bike_time fits
    start_idx = df_trains['actual_departure'].searchsorted(window_start)
    end_idx = df_trains['actual_departure'].searchsorted(window_end)

    potential_trains = df_trains.iloc[start_idx:end_idx]

    if not potential_trains.empty:
        for _, train in potential_trains.iterrows():
            wait_time = (train['actual_departure'] - bike_time).total_seconds() / 60.0

            # Categorize Train
            t_type = train['train_type']
            cat = 'Regional' if t_type in ['Os', 'Sp'] else 'LongDist'

            connections.append({
                'bike_time': bike_time,
                'time_of_day': bike_time.hour + bike_time.minute / 60.0,
                'wait_time': wait_time,
                'train_cat': cat
            })

df_conns = pd.DataFrame(connections)

# ==========================================
# 4. PLOTTING THE PROBABILITY HEATMAP
# ==========================================
print(f"   -> Analyzing {len(df_conns)} potential connections.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)


# Define generic plotting function
def plot_density(ax, data, title, cmap):
    # We want a 2D Histogram: X=Time of Day, Y=Wait Time
    # Bins: X=24 hours, Y=45 minutes
    h = ax.hist2d(
        data['time_of_day'],
        data['wait_time'],
        bins=[24, 45],
        range=[[0, 24], [0, 45]],
        cmap=cmap,
        cmin=1  # Don't plot zero-counts
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time of Day (Hour)")
    ax.set_ylabel("Buffer Time (Minutes)")
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xticklabels([f"{h}:00" for h in np.arange(0, 25, 4)])

    # Add Colorbar
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Density of Potential Matches')

    # Add "Hot Zone" boxes for interpretation
    # Commuter Zone (5-10 min)
    ax.axhspan(5, 12, color='green', alpha=0.1, label='Commuter Zone (5-12m)')
    # Traveler Zone (20-30 min)
    ax.axhspan(20, 35, color='blue', alpha=0.1, label='Traveler Zone (20-35m)')

    ax.legend(loc='upper right')


# --- Plot 1: Regional Potential Matches ---
reg_data = df_conns[df_conns['train_cat'] == 'Regional']
plot_density(ax1, reg_data, "Wait Times for REGIONAL Trains\n(Density of Connections)", "Greens")

# --- Plot 2: Long Distance Potential Matches ---
ld_data = df_conns[df_conns['train_cat'] == 'LongDist']
plot_density(ax2, ld_data, "Wait Times for LONG DISTANCE Trains\n(Density of Connections)", "Blues")

plt.suptitle(f"Transfer Probability Analysis: {CITY_KEY}\n(Showing ALL potential connections within 45 mins)",
             fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(f'results/Probability_Heatmap_{CITY_KEY}.png', dpi=300)
print(f"✅ Saved Probability Heatmap to results/Probability_Heatmap_{CITY_KEY}.png")
plt.show()