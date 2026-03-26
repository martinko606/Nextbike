import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import transport_analytics as ta  # Imports your existing logic
import os
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION (Matches your setup)
# ==========================================
RESULTS_DIR = 'results/correlation_plots'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Define the network (Same as your provided config)
NETWORKS = {
    'Ostrava-Svinov': {
        'file_trains': 'data/Pohyby_Svinov.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava-Svinov',
        'bike_stations': ['SV-Svinov nádraží *(navíc 15min na odjezd)'],
        'params': {'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1}}
    },
    'Ostrava hl.n.': {
        'file_trains': 'data/Pohyby_Ostrava_hl.xlsx',
        'file_weather': 'data/Ostrava_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Ostrava',
        'train_station': 'Ostrava hl.n.',
        'bike_stations': ['MOAP-Hlavní nádraží'],
        'params': {'weights': {'Os': 2.0, 'Sp': 1.7, 'R': 1.5, 'Ex': 1}}
    },
    'Brno': {
        'file_trains': 'data/Pohyby_Brno.xlsx',
        'file_weather': 'data/Brno_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Brno',
        'train_station': 'Brno hl.n.',
        'bike_stations': ['Hlavní nádraží - Hlavní vstup', 'Hlavní nádraží - pošta', 'Bajkazyl 666'],
        'params': {'weights': {'Os': 2.5, 'Sp': 2.0, 'R': 1.5, 'Ex': 1}}
    },
    'Prerov': {
        'file_trains': 'data/Pohyby_Prerov.xlsx',
        'file_weather': 'data/Prerov_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_Prerov',
        'train_station': 'Přerov os.n.',
        'bike_stations': ['Nádraží'],
        'params': {'weights': {'Os': 1.7, 'Sp': 1.4, 'R': 1.2, 'Ex': 1}}
    },
    'ValMez': {
        'file_trains': 'data/Pohyby_ValMez.xlsx',
        'file_weather': 'data/ValMez_pocasi.xlsx',
        'sheet_rentals': 'Vypujcky_ValMez',
        'train_station': 'Valašské Meziříčí',
        'bike_stations': ['Vlakové nádraží Valašské Meziříčí (nové umístění)'],
        'params': {'weights': {'Os': 1.8, 'Sp': 1.6, 'R': 1.4, 'Ex': 1.6}}
    }
}

# We only need 'Global' category for this specific chart
TRAIN_CATS = {'Global': None}

# ==========================================
# 2. DATA PROCESSING LOOP
# ==========================================
print("🚀 Starting 1-Minute High-Precision Analysis...")
all_results = []

for city, cfg in NETWORKS.items():
    print(f"   Processing {city}...")

    # Load Data using standard pandas calls (simplified for this script)
    # Note: Using your ta logic requires dataframes to be passed in.
    # We will implement a quick loader here to feed your ta function.

    try:
        # Load Trains
        df_p = pd.read_excel(cfg['file_trains'], header=None, nrows=20)
        h_idx = 0
        for i, row in df_p.iterrows():
            if any('čas' in str(x).lower() for x in row.values):
                h_idx = i
                break
        df_trains = pd.read_excel(cfg['file_trains'], header=h_idx)

        # Load Bikes
        df_bikes = pd.read_excel('data/nextbike_data_VSB_Vaclavik.xlsx', sheet_name=cfg['sheet_rentals'])
        # Minimal cleaning for ta module compatibility
        df_bikes.columns = [str(c).lower().strip() for c in df_bikes.columns]
        df_bikes['start_time'] = pd.to_datetime(df_bikes['start_time'])
        df_bikes['end_time'] = pd.to_datetime(df_bikes['end_time'])

        # Load Weather (Optional, passing empty if fails)
        try:
            df_weather = pd.read_excel(cfg['file_weather'])
        except:
            df_weather = None

        # Run your Transport Analytics Logic
        # This uses YOUR 1-minute calculation logic
        res = ta.analyze_city_data(city, cfg, df_trains, df_bikes, None, df_weather, TRAIN_CATS)
        all_results.extend(res)

    except Exception as e:
        print(f"   ❌ Error processing {city}: {e}")

# ==========================================
# 3. PLOTTING SPECIFIC METRICS
# ==========================================
if all_results:
    df = pd.DataFrame(all_results)

    # Filter for the specific metrics you requested
    # Your TA script outputs: 'Arr_Global', 'Dep_Global', 'Total_Activity_Corr'
    plot_df = df[['City', 'Arr_Global', 'Dep_Global', 'Total_Activity_Corr']].copy()

    # Rename for chart clarity
    plot_df.rename(columns={
        'Arr_Global': 'Arrivals vs Rentals',
        'Dep_Global': 'Departures vs Returns',
        'Total_Activity_Corr': 'Global Busyness'
    }, inplace=True)

    # Melt for Seaborn
    df_melt = plot_df.melt(id_vars='City', var_name='Interaction Type', value_name='Pearson r (1-min)')

    # Plot
    plt.figure(figsize=(12, 7), dpi=150)
    sns.set_theme(style="whitegrid")

    # Create grouped bar chart
    ax = sns.barplot(
        data=df_melt,
        x='City',
        y='Pearson r (1-min)',
        hue='Interaction Type',
        palette={'Arrivals vs Rentals': '#2ca02c',  # Green for Rentals
                 'Departures vs Returns': '#1f77b4',  # Blue for Returns
                 'Global Busyness': '#d62728'}  # Red for Global
    )

    # Styling
    plt.title('Transport Synchronization Analysis\n(High-Precision 1-Minute Resolution)',
              fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Pearson Correlation (r)', fontsize=12)
    plt.xlabel('')
    plt.legend(title=None, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)

    # Values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

    plt.tight_layout()

    # Save
    out_file = f"{RESULTS_DIR}/Detailed_Correlation_Analysis.png"
    plt.savefig(out_file)
    print(f"\n✅ Plot Saved: {out_file}")

    # Save CSV for paper
    plot_df.to_csv(f"{RESULTS_DIR}/Detailed_Correlation_Table.csv", index=False)
    print(f"✅ Table Saved: {RESULTS_DIR}/Detailed_Correlation_Table.csv")

else:
    print("❌ No results found. Check data paths.")