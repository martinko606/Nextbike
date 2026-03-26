import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
import requests
import os
import warnings
from shapely.geometry import Point

# Import your custom ML Engine!
from ml_engine import get_elevations_cached, get_or_train_local_model

warnings.filterwarnings('ignore')


def get_official_nextbike_stations(city_name):
    print(f"  -> Fetching live official Nextbike data for {city_name}...")
    url = "https://api.nextbike.net/maps/nextbike-live.json"
    stations = []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for country in data.get('countries', []):
                for city in country.get('cities', []):
                    if city['name'] == city_name:
                        for p in city.get('places', []):
                            if p.get('spot'):
                                stations.append({'uid': p['uid'], 'name': p['name'], 'lat': float(p['lat']),
                                                 'lon': float(p['lng'])})
                        return stations
    except Exception as e:
        print(f"     [!] API error: {e}")
    return stations


def generate_own_bike_polygon(G_bike, predicted_times_dict, total_mins, output_filename):
    print(f"\n  -> Generating OWN BIKE | {total_mins} mins...")
    total_sec = total_mins * 60

    valid_node_ids = [node for node, time_sec in predicted_times_dict.items() if time_sec <= total_sec]
    sub_graph = G_bike.subgraph(valid_node_ids)
    edges = ox.graph_to_gdfs(sub_graph, nodes=False, edges=True)

    if not edges.empty:
        poly = edges.to_crs(epsg=32633).geometry.buffer(40).unary_union
        final_gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:32633").to_crs(epsg=4326)
        final_gdf.to_file(output_filename, driver="GeoJSON")
        print(f"     Saved: {output_filename}")
        return final_gdf.iloc[0].geometry
    return None


def generate_shared_bike_isochrone(G_bike, G_walk, own_bike_poly, predicted_times_dict, stations, origin_lat,
                                   origin_lon, total_mins, output_filename):
    print(f"\n  -> Generating SHARED BIKE | {total_mins} mins...")
    total_sec = total_mins * 60
    polygons = []
    reached_stations = []

    origin_walk_node = ox.distance.nearest_nodes(G_walk, X=origin_lon, Y=origin_lat)
    walk_sub_graph = nx.ego_graph(G_walk, origin_walk_node, radius=total_sec, distance='travel_time')
    walk_edges = ox.graph_to_gdfs(walk_sub_graph, nodes=False, edges=True)
    if not walk_edges.empty:
        polygons.append(walk_edges.to_crs(epsg=32633).geometry.buffer(20).unary_union)

    stations_df = pd.DataFrame(stations)
    stations_gdf = gpd.GeoDataFrame(stations_df, geometry=gpd.points_from_xy(stations_df.lon, stations_df.lat),
                                    crs="EPSG:4326")

    valid_stations_gdf = stations_gdf[stations_gdf.geometry.within(own_bike_poly)]
    print(f"     Filtered down to {len(valid_stations_gdf)} stations inside the bike zone.")

    for idx, s in valid_stations_gdf.iterrows():
        station_bike_node = ox.distance.nearest_nodes(G_bike, X=s['lon'], Y=s['lat'])

        if station_bike_node in predicted_times_dict:
            predicted_bike_sec = predicted_times_dict[station_bike_node]

            if (predicted_bike_sec + 20) < total_sec:
                remaining_walk_sec = total_sec - (predicted_bike_sec + 20)

                reached_stations.append({
                    'uid': s.get('uid', 'unknown'),
                    'name': s.get('name', 'Nextbike Station'),
                    'geometry': s['geometry'],
                    'bike_travel_time_min': round((predicted_bike_sec + 20) / 60, 1),
                    'remaining_walk_min': round(remaining_walk_sec / 60, 1)
                })

                s_walk_node = ox.distance.nearest_nodes(G_walk, X=s['lon'], Y=s['lat'])
                dock_walk_graph = nx.ego_graph(G_walk, s_walk_node, radius=remaining_walk_sec, distance='travel_time')

                if dock_walk_graph.number_of_edges() > 0:
                    dock_walk_edges = ox.graph_to_gdfs(dock_walk_graph, nodes=False, edges=True)
                    poly = dock_walk_edges.to_crs(epsg=32633).geometry.buffer(20).unary_union
                    polygons.append(poly)

    if polygons:
        final_blob = gpd.GeoSeries(polygons).unary_union
        final_gdf = gpd.GeoDataFrame(geometry=[final_blob], crs="EPSG:32633").to_crs(epsg=4326)
        final_gdf.geometry = final_gdf.geometry.intersection(own_bike_poly)
        final_gdf.to_file(output_filename, driver="GeoJSON")
        print(f"     Saved Area: {output_filename}")

    if reached_stations:
        stations_out_gdf = gpd.GeoDataFrame(reached_stations, crs="EPSG:4326")
        stations_filename = output_filename.replace("isochrone_", "stations_")
        stations_out_gdf.to_file(stations_filename, driver="GeoJSON")
        print(f"     Saved Stations: {stations_filename}")


# ==========================================
# MASTER EXECUTION
# ==========================================
if __name__ == "__main__":

    # --- CONFIGURATION ---
    FILE_NAME = r"C:\Users\vaclavikmartin\PycharmProjects\Nextbike\data\nextbike_data_VSB_Vaclavik.xlsx"
    CITY_SHEET_MAPPING = {
        "Ostrava": "Vypujcky_Ostrava",
        #"Brno": "Vypujcky_Brno",
    }
    MAPPING_CITY_NAME = "Ostrava"
    ORIGIN_LAT = 49.8215
    ORIGIN_LON = 18.2101

    TRAINING_SHEET = CITY_SHEET_MAPPING.get(MAPPING_CITY_NAME)
    if not TRAINING_SHEET:
        raise ValueError(f"City '{MAPPING_CITY_NAME}' not found in the configuration dictionary.")

    # Train (or instantly load) the local AI model
    ai_model = get_or_train_local_model(FILE_NAME, TRAINING_SHEET, MAPPING_CITY_NAME, sample_size=800)

    print("\n--- PHASE 2: LOADING OR DOWNLOADING INFRASTRUCTURE GRAPHS ---")
    city_prefix = MAPPING_CITY_NAME.lower()
    bike_graph_file = f"{city_prefix}_bike_G.graphml"
    walk_graph_file = f"{city_prefix}_walk_G.graphml"

    if os.path.exists(bike_graph_file):
        print(f"  -> SUCCESS: Loading local bike map from {bike_graph_file}")
        G_bike = ox.load_graphml(bike_graph_file)
    else:
        print(f"  -> Downloading bike map for {MAPPING_CITY_NAME} from OpenStreetMap...")
        urban_bike_filter = (
            '["highway"]["area"!~"yes"]["access"!~"private"]'
            '["highway"!~"motorway|motorway_link|trunk|trunk_link|track|steps"]'
            '["surface"!~"unpaved|dirt|sand|grass|gravel|mud"]'
        )
        G_bike = ox.graph_from_point((ORIGIN_LAT, ORIGIN_LON), dist=8500, custom_filter=urban_bike_filter)
        ox.save_graphml(G_bike, bike_graph_file)

    if os.path.exists(walk_graph_file):
        print(f"  -> SUCCESS: Loading local pedestrian map from {walk_graph_file}")
        G_walk = ox.load_graphml(walk_graph_file)
    else:
        print(f"  -> Downloading pedestrian map for {MAPPING_CITY_NAME} from OpenStreetMap...")
        G_walk = ox.graph_from_point((ORIGIN_LAT, ORIGIN_LON), dist=8500, network_type='walk')

        # ---> TOPOLOGY CLEANUP: DESTROYING GHOST ISLANDS <---
        U_walk = G_walk.to_undirected()
        largest_cc = max(nx.connected_components(U_walk), key=len)
        G_walk = G_walk.subgraph(largest_cc).copy()
        print(f"     Topology check: Removed isolated pedestrian islands.")

        G_walk = ox.add_edge_speeds(G_walk, fallback=5.0)
        G_walk = ox.add_edge_travel_times(G_walk)
        ox.save_graphml(G_walk, walk_graph_file)

    print("\n--- PHASE 3: PRE-CALCULATING THE MASTER TIME DICTIONARY ---")
    origin_bike_node = ox.distance.nearest_nodes(G_bike, X=ORIGIN_LON, Y=ORIGIN_LAT)
    distances_dict = nx.single_source_dijkstra_path_length(G_bike, origin_bike_node, weight='length')
    reachable_nodes = {node: dist for node, dist in distances_dict.items() if dist <= 10000}

    coords_to_fetch = [(G_bike.nodes[n]['y'], G_bike.nodes[n]['x']) for n in reachable_nodes.keys()]
    coords_to_fetch.insert(0, (ORIGIN_LAT, ORIGIN_LON))

    stations = get_official_nextbike_stations(MAPPING_CITY_NAME)
    coords_to_fetch += [(s['lat'], s['lon']) for s in stations]
    coords_to_fetch = list(set(coords_to_fetch))

    network_elev_cache = f"{city_prefix}_network_elevations.json"
    elev_dict = get_elevations_cached(coords_to_fetch, cache_file=network_elev_cache)
    origin_elev = elev_dict.get(f"{ORIGIN_LAT},{ORIGIN_LON}", 0.0)

    nodes_data = []
    for n, dist in reachable_nodes.items():
        lat, lon = G_bike.nodes[n]['y'], G_bike.nodes[n]['x']
        elev = elev_dict.get(f"{lat},{lon}", 0.0)
        nodes_data.append([n, dist, elev - origin_elev])

    node_features_df = pd.DataFrame(nodes_data,
                                    columns=['node_id', 'network_distance_m', 'elevation_delta_m']).set_index('node_id')

    print("Asking AI to predict travel times for all city intersections...")
    all_predictions = ai_model.predict(node_features_df[['network_distance_m', 'elevation_delta_m']])
    predicted_times_dict = dict(zip(node_features_df.index, all_predictions))

    print("\n--- PHASE 4: GENERATING 6 DEDICATED ISOCHRONE FILES ---")
    poly_own_15 = generate_own_bike_polygon(G_bike, predicted_times_dict, 15,
                                            f"isochrone_{city_prefix}_own_bike_15min.geojson")
    generate_shared_bike_isochrone(G_bike, G_walk, poly_own_15, predicted_times_dict, stations, ORIGIN_LAT, ORIGIN_LON,
                                   15, f"isochrone_{city_prefix}_shared_bike_15min.geojson")

    #poly_own_30 = generate_own_bike_polygon(G_bike, predicted_times_dict, 30,
      #                                      f"isochrone_{city_prefix}_own_bike_30min.geojson")
    #generate_shared_bike_isochrone(G_bike, G_walk, poly_own_30, predicted_times_dict, stations, ORIGIN_LAT, ORIGIN_LON,
     #                              30, f"isochrone_{city_prefix}_shared_bike_30min.geojson")

    print("\n*** ALL MAPS SUCCESSFULLY COMPLETED! ***")