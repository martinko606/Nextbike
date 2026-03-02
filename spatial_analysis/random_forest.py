import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

ox.settings.log_console = True

warnings.filterwarnings('ignore')


# --- 1. LOAD AND FILTER MULTIPLE SHEETS (TESTING MODE) ---
def load_and_filter_selected_cities(file_path, target_cities, sample_size_per_city=500):
    print(f"Loading selected sheets {target_cities} from {file_path}...")

    # Load only the specified sheets
    all_sheets = pd.read_excel(file_path, sheet_name=target_cities)

    cleaned_dataframes = []

    for city_name, df in all_sheets.items():
        print(f"\nProcessing raw data for sheet: {city_name}")

        # Strip hidden spaces from column names
        df.columns = df.columns.str.strip()

        required_cols = ['start_lat', 'start_lng', 'end_lat', 'end_lng', 'duration']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f" -> SKIPPING SHEET '{city_name}': Missing columns {missing_cols}")
            continue

            # 1. Drop circular trips
        df = df[~((df['start_lat'] == df['end_lat']) & (df['start_lng'] == df['end_lng']))]

        # 2. Drop false starts (< 2 mins) and leisure outliers (> 45 mins)
        df = df[(df['duration'] >= 120) & (df['duration'] <= 2700)]

        # 3. Handle NaNs
        df = df.dropna(subset=required_cols)

        # Add a city label for our records
        df['city_label'] = city_name

        print(f" -> Found {len(df)} valid commuter trips.")

        # Sample the data
        if len(df) > sample_size_per_city:
            print(f" -> Sampling {sample_size_per_city} trips to keep testing time manageable...")
            df = df.sample(n=sample_size_per_city, random_state=42)

        cleaned_dataframes.append(df)

    if not cleaned_dataframes:
        raise ValueError("Critical Error: No valid data found in the selected sheets! Check your column names.")

    # Combine the selected cities into one Master DataFrame
    master_df = pd.concat(cleaned_dataframes, ignore_index=True)
    return master_df


# --- 2. DYNAMIC SPATIAL ENGINEERING ---
def engineer_spatial_features(df):
    distances = []
    elevation_deltas = []

    # Group by the city label so we only download each city's map ONCE
    for city_name, city_df in df.groupby('city_label'):
        print(f"\n--- Downloading and routing network for: {city_name} ---")

        # Calculate a Dynamic Bounding Box based on the exact GPS points
        north = city_df['start_lat'].max() + 0.02
        south = city_df['start_lat'].min() - 0.02
        east = city_df['start_lng'].max() + 0.02
        west = city_df['start_lng'].min() - 0.02

        print(
            f"Downloading map boundary: N:{round(north, 3)}, S:{round(south, 3)}, E:{round(east, 3)}, W:{round(west, 3)}")
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='bike')

        print(f"Calculating network distances for {len(city_df)} trips in {city_name}...")
        for index, row in city_df.iterrows():
            try:
                start_node = ox.distance.nearest_nodes(G, X=row['start_lng'], Y=row['start_lat'])
                end_node = ox.distance.nearest_nodes(G, X=row['end_lng'], Y=row['end_lat'])

                dist = nx.shortest_path_length(G, start_node, end_node, weight='length')
                distances.append((index, dist, 0.0))  # 0.0 is a placeholder for elevation

            except Exception as e:
                distances.append((index, np.nan, np.nan))

    # Map the calculated distances back to the master dataframe
    dist_df = pd.DataFrame(distances, columns=['index', 'network_distance_m', 'elevation_delta_m']).set_index('index')
    df = df.join(dist_df)

    # Drop any trips that failed routing
    df = df.dropna(subset=['network_distance_m'])
    return df


# --- 3. TRAIN THE MACHINE LEARNING MODEL ---
def train_empirical_model(df):
    print("\n==================================================")
    print("--- Training the Empirical ML Commuter Model ---")
    print("==================================================")

    X = df[['network_distance_m', 'elevation_delta_m']]
    y = df['duration']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Global Model Accuracy (R² Score): {round(r2, 3)}")
    print(f"Mean Absolute Error: +/- {round(mae / 60, 1)} minutes")

    return model


if __name__ == "__main__":
    # Your exact file path
    FILE_NAME = r"C:\Users\vaclavikmartin\PycharmProjects\Nextbike\data\nextbike_data_VSB_Vaclavik.xlsx"

    # --- TESTING MODE: CITY SELECTION ---
    # NOTE: Make sure "Ostrava" is the EXACT name of the sheet in your Excel file.
    CITIES_TO_RUN = ['Vypujcky_Ostrava']

    # 1. Load data
    master_trips_df = load_and_filter_selected_cities(FILE_NAME, target_cities=CITIES_TO_RUN, sample_size_per_city=500)

    # 2. Add Network Distance & Elevation
    enhanced_df = engineer_spatial_features(master_trips_df)

    # 3. Train the Model
    trained_model = train_empirical_model(enhanced_df)

    # Test prediction
    test_trip = pd.DataFrame({'network_distance_m': [3000], 'elevation_delta_m': [0]})
    predicted_time = trained_model.predict(test_trip)[0]
    print(
        f"\n[ML Prediction] A flat 3km ride in {CITIES_TO_RUN[0]} takes commuters exactly: {round(predicted_time / 60, 1)} minutes.")