import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import transport_analytics as ta
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. GLOBAL FILE CONFIGURATION
# ==========================================
FILE_BIKE_DATA = 'data/nextbike_data_VSB_Vaclavik.xlsx'
FILE_REBALANCING = 'data/nextbike_data_VSB_Vaclavik2.xlsx'

# Output Directory
RESULTS_DIR = 'results'

# ==========================================
# 2. CITY NETWORK MAP
# ==========================================
NETWORKS = {
    'Ostrava-Svinov': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'sheet_logs': 'Stanice_Ostrava',
        'sheet_rebal': 'Ostrava',
        'train_station': 'Ostrava-Svinov',
        'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
        'params': {
            'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1},
            'peak_arr_reg': 2, 'pulse_arr_reg': 15,
            'peak_arr_ld': 5, 'pulse_arr_ld': 20,
            'peak_dep_reg': 8, 'pulse_dep_reg': 20,
            'peak_dep_ld': 15, 'pulse_dep_ld': 35,
            'rush_am': 2.0, 'rush_pm': 1.8
        },
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    },

    'Ostrava hl.n.': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'sheet_logs': 'Stanice_Ostrava',
        'sheet_rebal': 'Ostrava',
        'train_station': 'Ostrava-Svinov',
        'bike_stations': ['MOAP-Hlavní nádraží'],
        'params': {
            'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1},
            'peak_arr_reg': 5, 'pulse_arr_reg': 25,
            'peak_arr_ld': 10, 'pulse_arr_ld': 30,
            'peak_dep_reg': 10, 'pulse_dep_reg': 25,
            'peak_dep_ld': 15, 'pulse_dep_ld': 35,
            'rush_am': 2.0, 'rush_pm': 1.7
        },
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    },

    'Brno hl.n.': {
        'file_trains': 'data/Pohyby_Brno.xlsx',
        'file_weather': 'data/Brno_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Brno',
        'sheet_logs': 'Stanice_Brno',
        'sheet_rebal': 'Brno',
        'train_station': 'Brno hl.n.',
        'bike_stations': [
            'Hlavní nádraží - Hlavní vstup',
            'Hlavní nádraží - pošta',
            'Bajkazyl 666'
        ],
        'params': {
            'weights': {'Os': 2.5, 'Sp': 2.0, 'R': 1.5, 'Ex': 1},
            'peak_arr_reg': 3.5, 'pulse_arr_reg': 12,
            'peak_arr_ld': 10, 'pulse_arr_ld': 30,
            'peak_dep_reg': 10, 'pulse_dep_reg': 25,
            'peak_dep_ld': 15, 'pulse_dep_ld': 35,
            'rush_am': 2.5, 'rush_pm': 2.0
        },
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    },
    'Přerov': {
        'file_trains': 'data/Pohyby_Prerov.xlsx',
        'file_weather': 'data/Prerov_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Prerov',
        'sheet_logs': 'Stanice_Prerov',
        'sheet_rebal': 'Prerov',
        'train_station': 'Přerov os.n.',
        'bike_stations': ['Nádraží'],
        'params': {
            'weights': {'Os': 1.7, 'Sp': 1.4, 'R': 1.2, 'Ex': 1},
            'peak_arr_reg': 3, 'pulse_arr_reg': 12,
            'peak_arr_ld': 6, 'pulse_arr_ld': 15,
            'peak_dep_reg': 10, 'pulse_dep_reg': 20,
            'peak_dep_ld': 15, 'pulse_dep_ld': 30,
            'rush_am': 1.7, 'rush_pm': 1.4
        },
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    },
    'Valašské Meziříčí': {
        'file_trains': 'data/Pohyby_ValMez.xlsx',
        'file_weather': 'data/ValMez_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_ValMez',
        'sheet_logs': 'Stanice_ValMez',
        'sheet_rebal': 'ValMez',
        'train_station': 'Valašské Meziříčí',
        'bike_stations': ['Vlakové nádraží Valašské Meziříčí (nové umístění)'],
        'params': {
            'weights': {'Os': 1.8, 'Sp': 1.6, 'R': 1.4, 'Ex': 1.6},
            'peak_arr_reg': 3, 'pulse_arr_reg': 12,
            'peak_arr_ld': 8, 'pulse_arr_ld': 20,
            'peak_dep_reg': 10, 'pulse_dep_reg': 20,
            'peak_dep_ld': 20, 'pulse_dep_ld': 25,
            'rush_am': 1.8, 'rush_pm': 1.4
        },
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    }
}

TRAIN_CATS = {
    'Global': None,
    'Regional': ['Os', 'Sp'],
    'LongDist': ['R', 'Ex', 'IC', 'EC', 'SC', 'rj']
}


# ==========================================
# 3. VISUALIZATION FUNCTIONS (The "Activity River" Version)
# ==========================================

# ==========================================
# 3. VISUALIZATION FUNCTIONS (Fixed: Total Pressure & Scaling)
# ==========================================

def save_interactive_heartbeat(city_name, trains, bikes, logs, config):
    """Generates Interactive Flow Plot with Total Pressure."""
    if 'vis_start' not in config: return

    print(f"  > Generating Interactive Plot for {city_name}...")
    start_str, end_str = config['vis_start'], config['vis_end']
    zoom_start, zoom_end = pd.to_datetime(start_str), pd.to_datetime(end_str)
    timeline = pd.date_range(zoom_start, zoom_end, freq='1min')

    # 1. Pressure (Total)
    t_mask = (trains['actual_arrival'] >= zoom_start) & (trains['actual_arrival'] <= zoom_end)
    zoom_trains = trains[t_mask].copy()
    s_arr, s_dep = ta.generate_signals(zoom_trains, timeline, config['params'])
    sig_total = s_arr + s_dep  # <--- The Fix

    # 2. Flows
    starts = bikes[bikes['start_place'].isin(config['bike_stations'])]
    ends = bikes[bikes['end_place'].isin(config['bike_stations'])]

    ts_rentals = starts.set_index('start_time').resample('1min').size().reindex(timeline, fill_value=0)
    ts_returns = ends.set_index('end_time').resample('1min').size().reindex(timeline, fill_value=0) * -1

    # 3. Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=timeline, y=ts_rentals, name="Rentals (Out)", marker_color='#2ca02c', marker_line_width=0),
                  secondary_y=False)
    fig.add_trace(go.Bar(x=timeline, y=ts_returns, name="Returns (In)", marker_color='#1f77b4', marker_line_width=0),
                  secondary_y=False)

    fig.add_trace(
        go.Scatter(x=timeline, y=sig_total, name="Total Pressure", line=dict(color='#d55e00', width=2), hoverinfo='y'),
        secondary_y=True)

    fig.update_layout(
        title=dict(text=f"Interactive Pulse: {city_name}", font=dict(size=20)),
        xaxis=dict(title="Time", rangeslider=dict(visible=True), type="date"),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        template="plotly_white",
        height=600,
        barmode='overlay'
    )

    fig.update_yaxes(title_text="<b>Bike Activity</b> (Qty/min)", secondary_y=False)
    fig.update_yaxes(title_text="<b>Total Passenger Pressure</b>", color="#d55e00", secondary_y=True)

    output_path = os.path.join(RESULTS_DIR, f"Interactive_{city_name}.html")
    fig.write_html(output_path)
    print(f"    -> Saved interactive graph to: {output_path}")

def plot_split_violin(city_name, trains, bikes, config):
    """
    Generates a Split Violin Plot with Enhanced Contrast.
    - Reduces smoothing to show individual train peaks.
    - Applies mild non-linear scaling to accentuate pressure differences.
    """
    if 'vis_start' not in config: return
    print(f"  > Generating Synchronization Violin Plot for {city_name}...")

    start_str, end_str = config['vis_start'], config['vis_end']
    zoom_start, zoom_end = pd.to_datetime(start_str), pd.to_datetime(end_str)

    timeline = pd.date_range(zoom_start, zoom_end, freq='1min')

    # 1. BIKE DATA
    b_mask_start = (bikes['start_time'] >= zoom_start) & (bikes['start_time'] <= zoom_end)
    b_mask_end = (bikes['end_time'] >= zoom_start) & (bikes['end_time'] <= zoom_end)

    bike_times = pd.concat([
        bikes.loc[b_mask_start, 'start_time'],
        bikes.loc[b_mask_end, 'end_time']
    ])

    # 2. TRAIN DATA
    df_trains = ta.standard_columns(trains)
    t_mask = (df_trains['actual_arrival'] >= zoom_start) & (df_trains['actual_arrival'] <= zoom_end)
    subset_trains = df_trains[t_mask].copy()

    if bike_times.empty or subset_trains.empty:
        return

    # Generate signals
    s_arr, s_dep = ta.generate_signals(subset_trains, timeline, config['params'])
    sig_total = s_arr + s_dep

    # 3. BUILD DATAFRAME
    data_rows = []

    # A. Bike Events (Blue)
    for t in bike_times:
        data_rows.append({
            'Day': t.strftime('%a'),
            'DayNum': t.dayofweek,
            'Hour': t.hour + t.minute / 60.0,
            'Type': 'Total Bike Movements'
        })

    # B. Train Events (Orange) - WITH CONTRAST BOOST
    sig_resampled = sig_total.resample('10min').mean()

    # Keep even small signals (night trains)
    sig_resampled = sig_resampled[sig_resampled > 0.0001]

    if not sig_resampled.empty:
        # CONTRAST BOOST: Raise to power 1.5 to make peaks "pop" more
        # This makes the wide parts wider and narrow parts narrower visually
        sig_enhanced = sig_resampled ** 1.5

        # Recalculate scaling based on enhanced signal
        target_count = len(bike_times)
        sum_pressure = sig_enhanced.sum()
        scale_factor = (target_count / sum_pressure) if sum_pressure > 0 else 1

        for t, val in sig_enhanced.items():
            count = int(round(val * scale_factor))

            # Ensure visibility for non-zero pressure
            if count == 0 and val > 0: count = 1

            if count > 0:
                data_rows.extend([{
                    'Day': t.strftime('%a'),
                    'DayNum': t.dayofweek,
                    'Hour': t.hour + t.minute / 60.0,
                    'Type': 'Total Passenger Pressure'
                }] * count)

    df_viz = pd.DataFrame(data_rows)
    df_viz.sort_values('DayNum', inplace=True)

    # 4. PLOTTING
    plt.figure(figsize=(14, 12))
    sns.set_style("whitegrid")

    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    my_pal = {"Total Bike Movements": "#1f77b4", "Total Passenger Pressure": "#ff7f0e"}

    try:
        ax = sns.violinplot(
            data=df_viz,
            x='Day',
            y='Hour',
            hue='Type',
            split=True,
            order=day_order,
            palette=my_pal,
            inner="quartile",
            scale="count",
            cut=0,
            bw_adjust=0.35  # <--- KEY CHANGE: Lower bandwidth = Sharper Details
        )

        plt.title(f"Station Synchronization: {city_name}\n(Left: Bike Rentals+Returns | Right: Train Arr+Dep)",
                  fontsize=16, fontweight='bold')
        plt.ylabel("Time of Day", fontsize=12)
        plt.xlabel("")

        # Axis 00-24
        ax.set_ylim(24, 0)
        ax.set_yticks(range(0, 25, 2))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])

        # Lines
        plt.axhline(y=6.0, color='red', linestyle='--', alpha=0.4, label='Morning Rush (6:00)')
        plt.axhline(y=9.0, color='red', linestyle='--', alpha=0.4, label='Morning Rush (9:00)')
        plt.axhline(y=14.0, color='blue', linestyle='--', alpha=0.4, label='Afternoon Rush (14:00)')
        plt.axhline(y=18.0, color='blue', linestyle='--', alpha=0.4, label='Afternoon Rush (18:00)')

        plt.legend(title="", loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=12)

        output_path = os.path.join(RESULTS_DIR, f"Violin_Sync_{city_name}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
        plt.close()
        print(f"    -> Saved enhanced violin plot to: {output_path}")

    except Exception as e:
        print(f"    [!] Error plotting violin: {e}")

# ==========================================
# 4. EXECUTION
# ==========================================
def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("=== STARTING MULTI-SHEET ANALYSIS ===\n")

    if not os.path.exists(FILE_BIKE_DATA):
        print(f"CRITICAL ERROR: {FILE_BIKE_DATA} not found.")
        return

    all_results = []

    # --- MAIN LOOP ---
    for city_key, config in NETWORKS.items():
        print(f"\n--- Processing {city_key} ---")

        # 1. Load TRAINS
        if not os.path.exists(config['file_trains']):
            print(f"  Skipping: {config['file_trains']} missing.")
            continue

        try:
            df_preview = pd.read_excel(config['file_trains'], header=None, nrows=10)
            header_idx = None
            for idx, row in df_preview.iterrows():
                row_str = row.astype(str).str.lower().values
                if any('čas' in x or 'cas' in x or 'druh' in x for x in row_str):
                    header_idx = idx
                    break

            if header_idx is not None:
                print(f"  [INFO] Found train headers at Row {header_idx + 1}")
                df_trains = pd.read_excel(config['file_trains'], header=header_idx)
            else:
                print(f"  [WARN] Could not auto-detect header. Assuming Row 1.")
                df_trains = pd.read_excel(config['file_trains'], header=0)

        except Exception as e:
            print(f"  [ERROR] Failed to load train file: {e}")
            continue

        # 2. Load WEATHER
        if not os.path.exists(config['file_weather']):
            print(f"  Skipping: {config['file_weather']} missing.")
            continue
        df_weather = pd.read_excel(config['file_weather'], decimal=',')

        # 3. Load RENTALS
        try:
            print(f"  Loading Rentals Sheet: {config['sheet_rentals']}...")
            df_bikes = pd.read_excel(FILE_BIKE_DATA, sheet_name=config['sheet_rentals'])
            df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]
            df_bikes['start_time'] = pd.to_datetime(df_bikes['start_time'])
            df_bikes['end_time'] = pd.to_datetime(df_bikes['end_time'])
        except Exception as e:
            print(f"  Error loading rental sheet '{config['sheet_rentals']}': {e}")
            continue

        # 4. Load LOGS
        try:
            log_sheet = config.get('sheet_logs', config['sheet_rentals'])
            print(f"  Loading Logs Sheet: {log_sheet}...")

            df_logs_raw = pd.read_excel(FILE_BIKE_DATA, sheet_name=log_sheet)
            df_logs_raw.columns = [str(c).strip() for c in df_logs_raw.columns]
            clean_cols = {c.lower(): c for c in df_logs_raw.columns}

            time_col = None
            for candidate in ['čas', 'cas', 'time', 'datum', 'timestamp']:
                if candidate in clean_cols:
                    time_col = clean_cols[candidate]
                    break

            if not time_col:
                print(f"    [!] Log Warning: Time column not found.")
                df_logs = None
            else:
                target_stations = config['bike_stations']
                found_cols = []
                for target in target_stations:
                    match = next((orig for clean, orig in clean_cols.items() if target.lower() in clean), None)
                    if match:
                        found_cols.append(match)

                if found_cols:
                    df_logs = pd.DataFrame()
                    df_logs['timestamp'] = pd.to_datetime(df_logs_raw[time_col])
                    df_logs['bikes'] = df_logs_raw[found_cols].fillna(0).sum(axis=1)
                    print(f"    -> Logs loaded: {len(df_logs)} records.")
                else:
                    print(f"    [!] Log Warning: No matching station columns found.")
                    df_logs = None

        except Exception as e:
            print(f"  Warning: Log processing failed ({e}). Analysis will assume full availability.")
            df_logs = None

        # 5. Load REBALANCING & Filter
        try:
            min_limit = config.get('min_trip_minutes', 0.5)
            initial_count = len(df_bikes)
            df_bikes = df_bikes[(df_bikes['end_time'] - df_bikes['start_time']) > pd.Timedelta(minutes=min_limit)]
            print(f"  Filtered {initial_count - len(df_bikes)} short trips (< {min_limit} min).")
        except Exception as e:
            print(f"  Warning: Filter failed ({e}).")

        # --- RUN ANALYSIS ---
        res = ta.analyze_city_data(city_key, config, df_trains, df_bikes, df_logs, df_weather, TRAIN_CATS)
        all_results.extend(res)

        # --- RUN VISUALIZATION ---
        df_trains = ta.standard_columns(df_trains)

        # 2. Interactive Heartbeat (Now with Stock)
        save_interactive_heartbeat(city_key, df_trains, df_bikes, df_logs, config)  # <--- Added df_logs

        # 3. Split Violin Sync Plot
        plot_split_violin(city_key, df_trains, df_bikes, config)

    # --- SAVE RESULTS ---
    if all_results:
        df_final = pd.DataFrame(all_results)
        excel_path = os.path.join(RESULTS_DIR, "Final_Results.xlsx")
        df_final.to_excel(excel_path, index=False)
        print(f"\nAnalysis Complete. Results saved to {RESULTS_DIR}/")
        print(df_final.round(3).to_string(index=False))


if __name__ == "__main__":
    main()