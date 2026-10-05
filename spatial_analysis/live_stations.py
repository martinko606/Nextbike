import requests
import geopandas as gpd
from shapely.geometry import Point
import os

# Define output directory
OUTPUT_DIR = "C:/Users/vaclavikmartin/PycharmProjects/Nextbike/results/spatial_analysis/live_stations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Exact API names for your cities
CITY_API_NAMES = {
    "brno": ["Brno"],
    "ostrava": ["Ostrava", "Ostrava hlavní"],
    "prerov": ["Přerov"],
    "valmez": ["Valašské Meziříčí", "Poličná", "Krhová", "Zašová"]
}


def fetch_and_save_stations():
    print("Fetching live data from Nextbike API...")
    url = "https://api.nextbike.net/maps/nextbike-live.json"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Loop through our target cities
    for city_key, valid_names in CITY_API_NAMES.items():
        station_list = []

        for country in data.get("countries", []):
            for city in country.get("cities", []):
                if city["name"] in valid_names:
                    for p in city.get("places", []):
                        if p.get("spot"):  # Only physical stations
                            station_list.append({
                                "name": p["name"],
                                "uid": p["uid"],
                                "lat": float(p["lat"]),
                                "lon": float(p["lng"])
                            })

        if station_list:
            # Convert to Geopandas
            geom = [Point(s["lon"], s["lat"]) for s in station_list]
            gdf = gpd.GeoDataFrame(station_list, geometry=geom, crs="EPSG:4326")

            # Save to GeoJSON
            out_path = os.path.join(OUTPUT_DIR, f"{city_key}_all_live_stations.geojson")
            gdf.to_file(out_path, driver="GeoJSON")
            print(f"Saved {len(station_list)} stations to {out_path}")


if __name__ == "__main__":
    fetch_and_save_stations()
    print("Done!")