import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. UNIVERSAL CONFIGURATION
# ==========================================
# Define the parameters for each city here
CITIES = {

    'Ostrava-Svinov': {
        'weather_file': 'data/Ostrava_pocasi.xlsx',
        'bike_file': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_logs': 'Stanice_Ostrava',
        'target_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)']
    },
    'Ostrava hl.n.': {
        'weather_file': 'data/Ostrava_pocasi.xlsx',
        'bike_file': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_logs': 'Stanice_Ostrava',
        'target_stations': ['MOAP-Hlavní nádraží']
    },
    'Brno': {
        'weather_file': 'data/Brno_pocasi.xlsx',
        'bike_file': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_logs': 'Stanice_Brno',
        'target_stations': ['Hlavní nádraží - Hlavní vstup',
            'Hlavní nádraží - pošta',
            'Bajkazyl 666']
    },
    'Prerov': {
        'weather_file': 'data/Prerov_pocasi.xlsx',
        'bike_file': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_logs': 'Stanice_Prerov',
        'target_stations': ['Nádraží']
    },
    'ValMez': {
        'weather_file': 'data/ValMez_pocasi.xlsx',
        'bike_file': 'data/nextbike_data_VSB_Vaclavik.xlsx',
        'sheet_logs': 'Stanice_ValMez',
        'target_stations': ['Vlakové nádraží']
    }
}

# GLOBAL SETTINGS
ZOOM_START = None
ZOOM_END = None
SAVE_FOLDER = 'results/weather_analysis'  # <--- UPDATED FOLDER NAME

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print(f"📂 Created new folder: {SAVE_FOLDER}")


# ==========================================
# 2. PROCESSING FUNCTION
# ==========================================
def analyze_city(city_name, config):
    print(f"\n🔍 Analyzing: {city_name}...")

    # --- A. Load Weather ---
    try:
        # Load with comma decimal support
        df_w = pd.read_excel(config['weather_file'], decimal=',')
        df_w.columns = [str(c).lower().strip() for c in df_w.columns]

        # Smart Time Column Detection
        w_time_col = next((c for c in df_w.columns if c in ['datetime', 'date', 'cas', 'datum']), None)
        if not w_time_col:
            print(f"   ❌ Error: No date column found in weather file for {city_name}")
            return

        df_w['timestamp'] = pd.to_datetime(df_w[w_time_col])

        # Ensure numeric columns (handle any remaining non-numeric issues)
        for col in ['temp', 'precip']:
            # Find column if name varies slightly (e.g. 'precipitation')
            actual_col = next((c for c in df_w.columns if col in c), None)
            if actual_col:
                df_w[actual_col] = pd.to_numeric(df_w[actual_col], errors='coerce')
                # Rename to standard 'temp'/'precip' for easier access
                df_w.rename(columns={actual_col: col}, inplace=True)

        w_res = df_w.set_index('timestamp').resample('1min').ffill()

    except Exception as e:
        print(f"   ❌ Error loading Weather: {e}")
        return

    # --- B. Load Bike Logs ---
    try:
        df_l = pd.read_excel(config['bike_file'], sheet_name=config['sheet_logs'])

        # Detect Time Column
        l_time_col = next((c for c in df_l.columns if str(c).strip().lower() in ['čas', 'cas', 'time', 'date']), None)
        if not l_time_col:
            print(f"   ❌ Error: No time column found in bike logs for {city_name}")
            return

        df_l['timestamp'] = pd.to_datetime(df_l[l_time_col])

        # Find Target Stations (Dynamic Search)
        found_cols = []
        for target in config['target_stations']:
            for col in df_l.columns:
                if target.lower() in str(col).lower():
                    found_cols.append(col)

        found_cols = list(set(found_cols))  # Remove duplicates

        if not found_cols:
            print(f"   ⚠️ Warning: No columns found matching {config['target_stations']}")
            return

        print(f"   -> Found {len(found_cols)} station columns matching target.")

        # Sum stock across all matching columns (e.g. "Entrance A" + "Entrance B")
        log_stock = df_l.set_index('timestamp')[found_cols].fillna(0).sum(axis=1).resample('1min').ffill()

    except Exception as e:
        print(f"   ❌ Error loading Bike Logs: {e}")
        return

    # --- C. Align & Mask ---
    start = max(w_res.index.min(), log_stock.index.min())
    end = min(w_res.index.max(), log_stock.index.max())

    if start >= end:
        print("   ❌ Error: Date ranges do not overlap.")
        return

    timeline = pd.date_range(start, end, freq='1min')

    w_final = w_res.reindex(timeline, method='nearest')
    stock_final = log_stock.reindex(timeline, method='nearest')

    # Logic: Bad Weather
    has_rain = 'precip' in w_final.columns
    has_temp = 'temp' in w_final.columns

    mask_rain = w_final['precip'] > 0.2 if has_rain else pd.Series(False, index=timeline)
    mask_cold = w_final['temp'] < 10 if has_temp else pd.Series(False, index=timeline)
    mask_bad = mask_rain | mask_cold

    # Logic: Empty Station
    mask_empty = stock_final == 0

    # Logic: Overlap (The "Lost Data" Zone)
    mask_nan_cause = mask_bad & mask_empty

    # --- D. Plotting ---
    print(f"   🎨 Generating Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # === TOP PANEL: WEATHER ===
    if has_rain:
        ax1.bar(timeline, w_final['precip'], color='blue', alpha=0.3, label='Rain (mm)', width=0.001)
    ax1.set_ylabel('Precipitation (mm)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')

    if has_temp:
        ax1b = ax1.twinx()
        ax1b.plot(timeline, w_final['temp'], color='red', linewidth=1, label='Temp (°C)')
        ax1b.set_ylabel('Temperature (°C)', color='red', fontsize=12)
        ax1b.tick_params(axis='y', labelcolor='red')

    # Highlight Bad Weather
    ax1.fill_between(timeline, 0, 1, where=mask_bad, color='red', alpha=0.1,
                     transform=ax1.get_xaxis_transform(), label='Bad Weather')

    ax1.set_title(f"Weather Conditions: {city_name}", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # === BOTTOM PANEL: AVAILABILITY ===
    ax2.plot(timeline, stock_final, color='green', linewidth=1.5, label='Available Bikes')
    ax2.set_ylabel('Number of Bikes', fontsize=12)

    # Highlight Empty
    ax2.fill_between(timeline, 0, 1, where=mask_empty, color='grey', alpha=0.3,
                     transform=ax2.get_xaxis_transform(), label='Empty Station')

    # Highlight "Lost Data Cause"
    ax2.fill_between(timeline, 0, 1, where=mask_nan_cause, color='orange', alpha=0.5,
                     transform=ax2.get_xaxis_transform(), label='Rain + Empty (Lost Potential)')

    ax2.set_title(f"Station Availability: {', '.join(config['target_stations'])}", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Formatting
    if ZOOM_START and ZOOM_END:
        ax2.set_xlim(pd.to_datetime(ZOOM_START), pd.to_datetime(ZOOM_END))

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to new folder
    filename = f"{SAVE_FOLDER}/{city_name}_Forensic_Analysis.png"
    plt.savefig(filename)
    print(f"   ✅ Saved to {filename}")
    plt.close()


# ==========================================
# 3. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    for city, cfg in CITIES.items():
        analyze_city(city, cfg)
    print("\n🏁 All cities processed.")