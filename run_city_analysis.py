import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import transport_analytics as ta
import os

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
            'weights': {
                'Os': 2.0,
                'Sp': 1.5,
                'R': 1.5,
                'Ex': 1
            },
            'peak_arr_reg': 2,
            'pulse_arr_reg': 12,
            'peak_arr_ld': 10,
            'pulse_arr_ld': 35,
            'rush_am': 2.0,
            'rush_pm': 2.0
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
            'weights': {
                'Os': 2.0,
                'Sp': 2.0,
                'R': 2.5,
                'IC': 3.0,
                'rj': 3.5
            },
            'peak_arr_reg': 6,
            'pulse_arr_reg': 20,
            'peak_arr_ld': 10,
            'pulse_arr_ld': 35,
            'rush_am': 2.0,
            'rush_pm': 2.0
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
            'weights': {
                'Os': 2.0,
                'Sp': 2.0,
                'R': 2.5,
                'IC': 3.0,
                'rj': 3.5
            },
            'peak_arr_reg': 6,
            'pulse_arr_reg': 20,
            'peak_arr_ld': 10,
            'pulse_arr_ld': 35,
            'rush_am': 2.0,
            'rush_pm': 2.0
        },
        'vis_start': '2025-09-15 00:00',
        'vis_end': '2025-10-19 23:59'
    },
    'Přerov': {
        'file_trains': 'data/Pohyby_Prerov.xlsx',
        'file_weather': 'data/Prerov_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Prerov',
        'sheet_logs': 'Stanice_Prerov',  # Verify if this is correct sheet name!
        'sheet_rebal': 'Prerov',  # Verify if this is correct sheet name!
        'train_station': 'Přerov os.n.',
        'bike_stations': ['Nádraží'],
        'params': {
            'weights': {
                'Os': 2.0,
                'Sp': 2.0,
                'R': 2.5,
                'IC': 3.0,
                'rj': 3.5
            },
            'peak_arr_reg': 6,
            'pulse_arr_reg': 20,
            'peak_arr_ld': 10,
            'pulse_arr_ld': 35,
            'rush_am': 2.0,
            'rush_pm': 2.0
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
            'weights': {
                'Os': 2.0,
                'Sp': 2.0,
                'R': 2.5,
                'IC': 3.0,
                'rj': 3.5
            },
            'peak_arr_reg': 6,
            'pulse_arr_reg': 20,
            'peak_arr_ld': 10,
            'pulse_arr_ld': 35,
            'rush_am': 2.0,
            'rush_pm': 2.0
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
# 3. VISUALIZATION FUNCTION
# ==========================================
def save_heartbeat_plot(city_name, trains, bikes, config):
    if 'vis_start' not in config: return

    print(f"  > Saving Heartbeat Plot for {city_name}...")
    start_str, end_str = config['vis_start'], config['vis_end']
    zoom_start, zoom_end = pd.to_datetime(start_str), pd.to_datetime(end_str)

    # Filter Data
    t_mask = (trains['actual_arrival'] >= zoom_start) & (trains['actual_arrival'] <= zoom_end)
    zoom_trains = trains[t_mask].copy()

    b_mask = (bikes['start_time'] >= zoom_start) & (bikes['start_time'] <= zoom_end)
    zoom_bikes = bikes[b_mask].copy()

    if zoom_bikes.empty or zoom_trains.empty:
        print(f"    Warning: No data found in window {start_str} - {end_str}")
        return

    timeline = pd.date_range(zoom_start, zoom_end, freq='1min')
    sig_arr, _ = ta.generate_signals(zoom_trains, timeline, config['params'])
    bike_counts = zoom_bikes.set_index('start_time').resample('1min').size().reindex(timeline, fill_value=0)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(timeline, bike_counts, width=0.0007, color='#004c6d', alpha=0.6, label='Actual Rentals')
    ax1.set_ylabel('Bike Rentals', color='#004c6d')
    ax1.tick_params(axis='y', labelcolor='#004c6d')

    ax2 = ax1.twinx()
    ax2.plot(timeline, sig_arr, color='#d55e00', linewidth=2, label='Passenger Pressure')
    ax2.fill_between(timeline, sig_arr, color='#d55e00', alpha=0.1)
    ax2.set_ylabel('Theoretical Pressure', color='#d55e00')
    ax2.tick_params(axis='y', labelcolor='#d55e00')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title(f"Heartbeat: {city_name} ({start_str})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"Heartbeat_{city_name}.png"))
    plt.close()


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

        # run_city_analysis.py (Inside main loop)

        # 1. Load TRAINS (Smart Header Detection)
        if not os.path.exists(config['file_trains']):
            print(f"  Skipping: {config['file_trains']} missing.")
            continue

        try:
            # Step A: Read first 10 rows without header
            df_preview = pd.read_excel(config['file_trains'], header=None, nrows=10)

            # Step B: Find the row index that contains specific keywords
            # We look for 'čas' (time) or 'druh' (type) or 'vlak' (train)
            header_idx = None
            for idx, row in df_preview.iterrows():
                row_str = row.astype(str).str.lower().values
                # Check if this row contains our target column names
                if any('čas' in x or 'cas' in x or 'druh' in x for x in row_str):
                    header_idx = idx
                    break

            # Step C: Load the full file using the found index
            if header_idx is not None:
                print(f"  [INFO] Found train headers at Row {header_idx + 1}")
                df_trains = pd.read_excel(config['file_trains'], header=header_idx)
            else:
                # Fallback to default if no keywords found
                print(f"  [WARN] Could not auto-detect header. Assuming Row 1.")
                df_trains = pd.read_excel(config['file_trains'], header=0)

        except Exception as e:
            print(f"  [ERROR] Failed to load train file: {e}")
            continue

        # 2. Load WEATHER
        if not os.path.exists(config['file_weather']):
            print(f"  Skipping: {config['file_weather']} missing.")
            continue
        df_weather = pd.read_excel(config['file_weather'])

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

            # Normalize Headers
            df_logs_raw.columns = [str(c).strip() for c in df_logs_raw.columns]
            clean_cols = {c.lower(): c for c in df_logs_raw.columns}

            # Find Time Column
            time_col = None
            for candidate in ['čas', 'cas', 'time', 'datum', 'timestamp']:
                if candidate in clean_cols:
                    time_col = clean_cols[candidate]
                    break

            if not time_col:
                print(f"    [!] Log Warning: Time column not found (looked for 'Čas', 'cas'...).")
                df_logs = None
            else:
                # Find Station Columns
                target_stations = config['bike_stations']
                found_cols = []
                for target in target_stations:
                    match = next((orig for clean, orig in clean_cols.items() if target.lower() in clean), None)
                    if match:
                        found_cols.append(match)

                if found_cols:
                    # Consolidate
                    df_logs = pd.DataFrame()
                    df_logs['timestamp'] = pd.to_datetime(df_logs_raw[time_col])
                    df_logs['bikes'] = df_logs_raw[found_cols].fillna(0).sum(axis=1)
                    print(f"    -> Logs loaded: {len(df_logs)} records. Monitoring {len(found_cols)} station columns.")
                else:
                    print(f"    [!] Log Warning: No matching station columns found.")
                    df_logs = None

        except Exception as e:
            print(f"  Warning: Log processing failed ({e}). Analysis will assume full availability.")
            df_logs = None

            # 5. Load REBALANCING & Filter
            try:
                # Get the limit from config (Default to 2 minutes if missing)
                min_limit = config.get('min_trip_minutes', 0.5)

                initial_count = len(df_bikes)

                # Filter trips shorter than the limit
                # Note: We do this REGARDLESS of whether rebalancing file exists
                df_bikes = df_bikes[(df_bikes['end_time'] - df_bikes['start_time']) > pd.Timedelta(minutes=min_limit)]

                print(f"  Filtered {initial_count - len(df_bikes)} short trips (< {min_limit} min).")

                # Load rebalancing file (optional, just for logging/safety)
                if os.path.exists(FILE_REBALANCING):
                    df_rebal = pd.read_excel(FILE_REBALANCING, sheet_name=config['sheet_rebal'])

            except Exception as e:
                print(f"  Warning: Filter/Rebalancing failed ({e}).")

        # --- RUN ANALYSIS ---
        res = ta.analyze_city_data(city_key, config, df_trains, df_bikes, df_logs, df_weather, TRAIN_CATS)
        all_results.extend(res)

        # --- RUN VISUALIZATION ---
        # Note: We re-standardize inside save_heartbeat_plot logic indirectly,
        # but ta.standard_columns modifies in-place, so it's safe to call here.
        df_trains = ta.standard_columns(df_trains)
        save_heartbeat_plot(city_key, df_trains, df_bikes, config)

    # --- SAVE RESULTS ---
    if all_results:
        df_final = pd.DataFrame(all_results)
        excel_path = os.path.join(RESULTS_DIR, "Final_Results.xlsx")
        df_final.to_excel(excel_path, index=False)
        print(f"\nAnalysis Complete. Results saved to {RESULTS_DIR}/")
        print(df_final.round(3).to_string(index=False))


if __name__ == "__main__":
    main()