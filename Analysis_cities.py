# run_city_analysis.py
import pandas as pd
import transport_analytics as ta  # Importing your master code

# ==========================================
# 1. SETUP: FILES & WEATHER SETTINGS
# ==========================================
FILE_TRAINS  = 'train_data.xlsx'
FILE_BIKES   = 'nextbike_data.xlsx'     # Contains Sheets: Rides, Logs
FILE_REBAL   = 'rebalancing_data.xlsx'  # New file
FILE_WEATHER = 'weather_data.xlsx'

TEMP_LIMIT = 10.0  # Degrees Celsius
RAIN_LIMIT = 0.0  # mm of rain

# ==========================================
# 2. SETUP: CITY CONFIGURATION
# ==========================================
# Configure your Pulse Parameters here:
# pulse_arr: Duration of influence AFTER arrival (Last Mile)
# pulse_dep: Duration of influence BEFORE departure (First Mile)

NETWORKS = {
    'Ostrava_Svinov': {
        'train_station': 'Ostrava-Svinov',
        'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
        'params': {'pulse_arr': 25, 'pulse_dep': 40, 'rush_am': 1.8, 'rush_pm': 1.8}
    },
    'Ostrava_Main': {
        'train_station': 'Ostrava hl.n.',
        'bike_stations': ['MOAP-Hlavní nádraží'],
        'params': {'pulse_arr': 20, 'pulse_dep': 35, 'rush_am': 1.5, 'rush_pm': 1.5}
    },
    'Brno_Main': {
        'train_station': 'Brno hl.n.',
        'bike_stations': [
            'Hlavní nádraží - Hlavní vstup',
            'Hlavní nádraží - pošta',
            'Bajkazyl 666'
        ],
        'params': {'pulse_arr': 40, 'pulse_dep': 60, 'rush_am': 2.0, 'rush_pm': 2.0}
    },
    'Valmez': {
        'train_station': 'Valašské Meziříčí',
        'bike_stations': ['Vlakové nádraží Valašské Meziříčí (nové umístění)'],
        'params': {'pulse_arr': 15, 'pulse_dep': 25, 'rush_am': 1.2, 'rush_pm': 1.2}
    },
    'Prerov': {
        'train_station': 'Přerov',
        'bike_stations': ['Nádraží'],
        'params': {'pulse_arr': 15, 'pulse_dep': 25, 'rush_am': 1.2, 'rush_pm': 1.2}
    }
}


# ==========================================
# 3. EXECUTION
# ==========================================
def main():
    try:
        # Pass the new file argument
        t, b, l, w = ta.load_data(FILE_TRAINS, FILE_BIKES, FILE_REBAL, FILE_WEATHER)
    except Exception as e:
        print(e)
        return

    all_results = []

    # 2. Iterate through cities
    print("-" * 60)
    for city_key, config in NETWORKS.items():
        # Call the analysis function from Master Module
        city_results = ta.analyze_hub(
            hub_name=city_key,
            config=config,
            trains=t,
            bikes=b,
            logs=l,
            weather=w,
            temp_threshold=TEMP_LIMIT,
            rain_threshold=RAIN_LIMIT
        )
        all_results.extend(city_results)

    # 3. Output Results
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 80)
        print("FINAL SCOREBOARD")
        print("=" * 80)
        print(df.round(3).to_string(index=False))

        # Save
        df.to_excel("Final_Results.xlsx", index=False)
        print("\nResults saved to 'Final_Results.xlsx'")


if __name__ == "__main__":
    main()