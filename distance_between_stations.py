"""
This script calculates the distances between each station
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import os
import json
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
from heapq import nlargest
import seaborn as sns
import random
import math

stop_id_to_name = {}
line_13 = ["Norsborg", 'Hallunda', 'Alby', 'Fittja', 'Masmo', 'Vårby gård', 'Vårberg', 'Skärholmen', 'Sätra', 'Bredäng',
           'Mälarhöjden', 'Axelsberg', 'Örnsberg', 'Aspudden', 'Liljeholmen', 'Hornstull', 'Zinkensdamm', 'Mariatorget',
           'Slussen', 'Gamla stan', 'T-Centralen', 'Östermalmstorg', 'Karlaplan', 'Gärdet', 'Ropsten']

line_14 = ["Fruängen", "Västertorp", "Hägerstensåsen", "Telefonplan", "Midsommarkransen", 'Liljeholmen', 'Hornstull', 'Zinkensdamm', 'Mariatorget',
           'Slussen', 'Gamla stan', 'T-Centralen', 'Östermalmstorg', "Stadion", "Tekniska högskolan", "Universitetet", "Bergshamra", "Danderyds sjukhus", "Mörby centrum"]

route_id_to_remove = "9011001001400000" #if you want line 13
#route_id_to_remove = "9011001001300000" #if you want line 14

# Starting point
current_point = (59.243739, 17.815538) #if you want line 13
#current_point = (59.286757, 17.964847) #if you want line 14

stop_coordinates = {}
# Read the "stops-r.txt" file
with open("static_data/GTFS-SL-2023-01-01/stops-r.txt", "r") as file:
    # Iterate through each line
    for line in file:
        # Split the line by comma
        stop_data = line.strip().split(',')
        # Check if the stop_name exists in line_13
        if stop_data[1] in line_14:
            # Store the latitude and longitude values for the stop_name
            stop_coordinates[stop_data[1]] = (float(stop_data[2]), float(stop_data[3]))

# Print the stop coordinates
# Read the first .txt file to map trip_id with route_id
trip_id_to_route_id = {}
with open('static_data/GTFS-SL-2023-01-01/trips-r.txt', 'r') as file:
    next(file)  # Skip header
    for line in file:
        route_id, _, trip_id, *_ = line.strip().split(',')
        trip_id_to_route_id[trip_id] = route_id

trip_id_to_route_id_filtered = {}

# Iterate over the original trip_id_to_route_id dictionary
for trip_id, route_id in trip_id_to_route_id.items():
    # Check if the route_id is not equal to route_id_to_remove
    if route_id != route_id_to_remove:
        # Add the (trip_id, route_id) pair to the filtered dictionary
        trip_id_to_route_id_filtered[trip_id] = route_id

# Update trip_id_to_route_id with the filtered dictionary
trip_id_to_route_id = trip_id_to_route_id_filtered
trip_id_list = list(trip_id_to_route_id.keys())

# Directory path where folders '00' to '23' are located
directory = "realtime_vehicle_data/sl-2023-01-01/VehiclePositions/2023/01/01"

# List to store data from JSON files
trip_data = {}

# Loop through folders '07', '08', and '09'
for folder_name in ['07', "08", "09"]:
    folder_path = os.path.join(directory, folder_name)
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Loop through JSON files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                # Read JSON data from the file
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Extract relevant information and store in trip_data dictionary
                    for entry in data:
                        trip_id = entry['vehicle']['trip']['tripId']
                        # Check if trip_id exists in trip_id_list
                        if trip_id in trip_id_list:
                            latitude = entry['vehicle']['position']['latitude']
                            longitude = entry['vehicle']['position']['longitude']
                            timestamp = entry['vehicle']['timestamp']
                            # Check if trip_id already exists in trip_data
                            if trip_id in trip_data:
                                # Append latitude, longitude, and timestamp to existing list
                                trip_data[trip_id]['latitude'].append(latitude)
                                trip_data[trip_id]['longitude'].append(longitude)
                                trip_data[trip_id]['timestamp'].append(timestamp)
                            else:
                                # Create a new list for latitude, longitude, and timestamp
                                trip_data[trip_id] = {'latitude': [latitude], 'longitude': [longitude],
                                                      'timestamp': [timestamp]}

for trip_id, data in trip_data.items():
    # Zip the latitude, longitude, and timestamp lists together
    zipped_data = zip(data['latitude'], data['longitude'], data['timestamp'])
    # Sort the zipped data based on the timestampsimport seaborn as sns
    sorted_data = sorted(zipped_data, key=lambda x: int(x[2]))
    # Unzip the sorted data
    sorted_latitude, sorted_longitude, sorted_timestamp = zip(*sorted_data)
    # Update the trip_data dictionary with sorted latitude, longitude, and timestamp
    trip_data[trip_id]['latitude'] = list(sorted_latitude)
    trip_data[trip_id]['longitude'] = list(sorted_longitude)
    trip_data[trip_id]['timestamp'] = list(sorted_timestamp)

# Extract latitude and longitude values from trip_data
# Extract latitude and longitude values from trip_data
trip_latitude_values = []
trip_longitude_values = []

for trip_id, data in trip_data.items():
    trip_latitude_values.extend(data['latitude'])
    trip_longitude_values.extend(data['longitude'])

# Extract latitude and longitude values from stop_coordinates
stop_latitude_values = []
stop_longitude_values = []
stop_names = []

for stop_name, coordinates in stop_coordinates.items():
    latitude, longitude = coordinates
    stop_latitude_values.append(latitude)
    stop_longitude_values.append(longitude)
    stop_names.append(stop_name)


# Create scatter plot

def remove_90_percent_elements_randomly(list1, list2):
    # Combine the two lists into one list of tuples
    combined_lists = list(zip(list1, list2))
    # Shuffle the combined list randomly
    random.shuffle(combined_lists)

    # Calculate the number of elements to keep (10%)
    num_elements_to_keep = int(len(combined_lists) * 0.95)

    # Slice the combined list to keep only 10% of the elements
    combined_lists_trimmed = combined_lists[:num_elements_to_keep]

    # Unpack the trimmed combined list back into separate lists
    list1_trimmed, list2_trimmed = zip(*combined_lists_trimmed)

    return list(list1_trimmed), list(list2_trimmed)


# Example usage
list1 = trip_longitude_values
list2 = trip_latitude_values

list1_trimmed, list2_trimmed = remove_90_percent_elements_randomly(list1, list2)
print(len(list1_trimmed))
print(len(list2_trimmed))

plt.figure(figsize=(12, 6))
plt.scatter(list1_trimmed, list2_trimmed, color="blue", alpha=0.5, label='Trip Coordinates')
plt.scatter(stop_longitude_values, stop_latitude_values, color='red', label='Stop Coordinates')
print(len(trip_longitude_values))
# Annotate each red dot with its station name
for i, stop_name in enumerate(stop_names):
    plt.annotate(stop_name, (stop_longitude_values[i], stop_latitude_values[i]), textcoords="offset points",
                 xytext=(0, 10), ha='center')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Realtime Trip Coordinates')
plt.legend()
plt.grid(True)
plt.show()


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return r * c


def find_closest_point(current_point, points, used_points, station_points):
    min_distance = float('inf')
    closest_point = None
    station_point = False
    for point in points:
        if point not in used_points:
            distance = haversine(current_point[0], current_point[1], point[0], point[1])
            if distance < min_distance:
                min_distance = distance
                closest_point = point

    if closest_point in station_points:
        station_point = True

    return closest_point, min_distance, station_point


# Sample lists of longitude and latitude
longitudes = list1_trimmed  # Your list of longitudes
latitudes = list2_trimmed  # Your list of latitudes

# Starting point
used_points = set()

total_distance = 0
coordinates = zip(latitudes, longitudes)

# Add station coordinates to the set
station_coordinates = set(zip(stop_latitude_values, stop_longitude_values))

unique_coordinates = set(coordinates).union(station_coordinates)

station_distances = {}


def get_key_from_coordinate(coord, stop_coordinates):
    for key, value in stop_coordinates.items():
        if value == coord:
            return key
    return None


while len(used_points) < len(unique_coordinates):
    closest_point, distance, station = find_closest_point(current_point, unique_coordinates, used_points,
                                                          station_coordinates)
    used_points.add(closest_point)
    current_point = closest_point
    total_distance += distance
    if station == True:
        # Get the key name from coordinate
        station_name = get_key_from_coordinate(closest_point, stop_coordinates)
        station_distances[station_name] = total_distance
    # print(f"Closest point: {closest_point}, Distance: {distance} km")

distances_between_stations = {}
previous_station = None

for station, distance in station_distances.items():
    if previous_station is not None:
        # Calculate the distance between the current station and the previous one
        s = previous_station + "-" + station
        distances_between_stations[s] = distance - station_distances[previous_station]
    previous_station = station
print(distances_between_stations)
print(f"Total distance traveled: {total_distance} km")

