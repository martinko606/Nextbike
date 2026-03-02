import osmnx as ox
import networkx as nx
import geopandas as gpd
import requests
import math
import warnings
import os

warnings.filterwarnings('ignore')


# --- 1. HELPER: HAVERSINE DISTANCE ---
def calculate_haversine(lat1, lon1, lat2, lon2):
    # Calculates straight-line distance in meters between two GPS points
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# --- 2. FETCH STATIONS ---
def get_nextbike_stations(city_name="Ostrava"):
    print(f"Fetching live Nextbike stations for {city_name}...")
    url = "https://maps.nextbike.net/maps/nextbike-live.json"
    data = requests.get(url).json()

    stations = []
    for country in data['countries']:
        for city in country['cities']:
            if city['name'] == city_name:
                for place in city['places']:
                    if not place['spot']: continue
                    stations.append(
                        {'uid': place['uid'], 'name': place['name'], 'lat': place['lat'], 'lon': place['lng']})
                return stations
    return []


# --- 3. MULTI-MODAL ISOCHRONE GENERATOR ---
def generate_multimodal_isochrone(origin_lat, origin_lon, total_mins=15, base_bike_speed=15.0, walk_speed=5.0):
    total_sec = total_mins * 60

    # At 15 km/h, max straight line distance in 15 mins is 3750m. We buffer to 4500m to capture curvy roads.
    max_network_dist = 4500

    print("Downloading Street Networks...")
    G_bike = ox.graph_from_point((origin_lat, origin_lon), dist=max_network_dist, network_type='bike')
    G_walk = ox.graph_from_point((origin_lat, origin_lon), dist=max_network_dist, network_type='walk')

    # --- TOPOGRAPHY & ELEVATION LOGIC ---
    dem_path = "ostrava_elevation.tif"
    if os.path.exists(dem_path):
        print("DEM found! Applying topographic elevation and slope penalties...")
        G_bike = ox.elevation.add_node_elevations_raster(G_bike, dem_path)
        G_bike = ox.elevation.add_edge_grades(G_bike)

        # Apply custom speeds based on slope
        for u, v, k, data in G_bike.edges(keys=True, data=True):
            grade = data.get('grade', 0.0)
            if grade > 0.05:
                speed = 6.0  # Steep uphill
            elif grade > 0.02:
                speed = 10.0  # Moderate uphill
            elif grade < -0.02:
                speed = 20.0  # Downhill
            else:
                speed = base_bike_speed  # Flat
            data['speed_kph'] = speed
    else:
        print("No DEM file found. Running flat terrain model...")
        G_bike = ox.add_edge_speeds(G_bike, fallback=base_bike_speed)

    G_bike = ox.add_edge_travel_times(G_bike)

    G_walk = ox.add_edge_speeds(G_walk, fallback=walk_speed)
    G_walk = ox.add_edge_travel_times(G_walk)

    origin_bike_node = ox.distance.nearest_nodes(G_bike, X=origin_lon, Y=origin_lat)
    origin_walk_node = ox.distance.nearest_nodes(G_walk, X=origin_lon, Y=origin_lat)

    polygons = []

    print("Calculating baseline walking isochrone from train station...")
    walk_sub_graph = nx.ego_graph(G_walk, origin_walk_node, radius=total_sec, distance='travel_time')
    walk_edges = ox.graph_to_gdfs(walk_sub_graph, nodes=False, edges=True)
    if not walk_edges.empty:
        polygons.append(walk_edges.to_crs(epsg=32633).geometry.buffer(100).unary_union)

    stations = get_nextbike_stations("Ostrava")

    # --- FIX: THE DISTANCE PRE-FILTER ---
    valid_stations = []
    for s in stations:
        dist = calculate_haversine(origin_lat, origin_lon, s['lat'], s['lon'])
        if dist <= max_network_dist:
            valid_stations.append(s)

    print(f"Discarded faraway stations. {len(valid_stations)} stations remain within the local map boundary.")

    print(f"Calculating all reachable nodes by bike within {total_mins} minutes...")
    reachable_bike_nodes = nx.single_source_dijkstra_path_length(G_bike, origin_bike_node, cutoff=total_sec,
                                                                 weight='travel_time')

    reachable_stations = []
    for station in valid_stations:
        station_node = ox.distance.nearest_nodes(G_bike, X=station['lon'], Y=station['lat'])
        if station_node in reachable_bike_nodes:
            bike_time_sec = reachable_bike_nodes[station_node]
            reachable_stations.append((station, bike_time_sec))

    print(f"--> Found exactly {len(reachable_stations)} stations physically reachable by bike!")

    for idx, (station, bike_time_sec) in enumerate(reachable_stations):
        print(
            f"   Processing station {idx + 1}/{len(reachable_stations)}: {station['name']} (Bike time: {round(bike_time_sec / 60, 1)} min)")
        remaining_walk_sec = total_sec - bike_time_sec
        station_walk_node = ox.distance.nearest_nodes(G_walk, X=station['lon'], Y=station['lat'])
        dock_walk_graph = nx.ego_graph(G_walk, station_walk_node, radius=remaining_walk_sec, distance='travel_time')

        try:
            dock_walk_edges = ox.graph_to_gdfs(dock_walk_graph, nodes=False, edges=True)
            if not dock_walk_edges.empty:
                poly = dock_walk_edges.to_crs(epsg=32633).geometry.buffer(100).unary_union
                polygons.append(poly)
        except ValueError:
            pass

    print("Merging all multi-modal catchment areas...")
    final_blob = gpd.GeoSeries(polygons).unary_union

    final_gdf = gpd.GeoDataFrame(geometry=[final_blob], crs="EPSG:32633").to_crs(epsg=4326)
    final_gdf.to_file("svinov_15min_multimodal_topography.geojson", driver="GeoJSON")
    print("Success! Optimized multi-modal isochrone saved.")


if __name__ == "__main__":
    # Exact coordinates for the front of Ostrava-Svinov Nádraží
    SVINOV_LAT = 49.8215
    SVINOV_LON = 18.2101

    generate_multimodal_isochrone(SVINOV_LAT, SVINOV_LON, total_mins=15)