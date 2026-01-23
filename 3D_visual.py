import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURATION ---
CITY_KEY = 'Ostrava-Svinov'
VIS_DATE = '2025-09-17'
FILE_BIKES = 'data/nextbike_data_VSB_Vaclavik.xlsx'
SHEET_RENTALS = 'Vypujcky_Ostrava'

# Station Location (approximate for Svinov)
STATION_LAT = 49.821
STATION_LON = 18.215
STATION_NAME = "Ostrava-Svinov"

# --- 1. LOAD DATA ---
print(f"🚀 Loading 3D Data for {CITY_KEY} on {VIS_DATE}...")
df_bikes = pd.read_excel(FILE_BIKES, sheet_name=SHEET_RENTALS)
df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]

df_bikes['start_time'] = pd.to_datetime(df_bikes['start_time'])
df_bikes['end_time'] = pd.to_datetime(df_bikes['end_time'])

# Filter for the specific day
subset = df_bikes[df_bikes['start_time'].dt.date == pd.to_datetime(VIS_DATE).date()].copy()

# Calculate "Time of Day" as decimal (Z-axis)
subset['z_start'] = subset['start_time'].dt.hour + subset['start_time'].dt.minute / 60.0
subset['z_end'] = subset['end_time'].dt.hour + subset['end_time'].dt.minute / 60.0

# --- 2. BUILD 3D TRACES ---
fig = go.Figure()

# A. Plot Station Column (Reference)
fig.add_trace(go.Scatter3d(
    x=[STATION_LON, STATION_LON],
    y=[STATION_LAT, STATION_LAT],
    z=[0, 24],
    mode='lines',
    line=dict(color='black', width=5),
    name=f'{STATION_NAME} (Hub)'
))

# --- B. PREPARE BIKE TRIPS (This loop was missing!) ---
x_lines = []
y_lines = []
z_lines = []

for _, row in subset.iterrows():
    # Only map if we have coordinates (columns usually 'start_lng', 'start_lat')
    if 'start_lng' in row and not pd.isna(row['start_lng']):
        # Start Point
        x_lines.append(row['start_lng'])
        y_lines.append(row['start_lat'])
        z_lines.append(row['z_start'])

        # End Point
        x_lines.append(row['end_lng'])
        y_lines.append(row['end_lat'])
        z_lines.append(row['z_end'])

        # None (Break the line so it doesn't connect to the next trip)
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

# Now plot the prepared lines
fig.add_trace(go.Scatter3d(
    x=x_lines,
    y=y_lines,
    z=z_lines,
    mode='lines',
    # FIX: opacity is outside the line dict
    line=dict(color='#1f77b4', width=2),
    opacity=0.4,
    name='Bike Trips'
))

# C. Plot Start Points (Green Dots)
fig.add_trace(go.Scatter3d(
    x=subset['start_lng'],
    y=subset['start_lat'],
    z=subset['z_start'],
    mode='markers',
    marker=dict(size=3, color='#2ca02c', opacity=0.8),
    name='Rentals (Start)'
))

# D. Plot End Points (Red Dots)
fig.add_trace(go.Scatter3d(
    x=subset['end_lng'],
    y=subset['end_lat'],
    z=subset['z_end'],
    mode='markers',
    marker=dict(size=3, color='#d62728', opacity=0.8),
    name='Returns (End)'
))

# --- 3. LAYOUT & FORMATTING ---
fig.update_layout(
    title=f"3D Space-Time Cube: {CITY_KEY} ({VIS_DATE})",
    scene=dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Time of Day (Hour)',
        zaxis=dict(range=[0, 24], tickvals=list(range(0, 25, 2))),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.8)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=800
)

# Save
output_file = f"results/3D_Cube_{CITY_KEY}.html"
fig.write_html(output_file)
print(f"✅ 3D Map saved to {output_file}")