"""
This script plots the coordinates (latitude and longitude) of actual vehicle positions together with
station coordinates and calculates the distance between each station and the total distance.
The script also plots a diagram over actual arrival, departure and dwell times at each station between a time interval.
"""

import numpy as np
import datetime
import os
import json
import matplotlib.pyplot as plt
from scipy import stats
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
import random
import math

stop_id_to_name = {}
line_13 = ["Norsborg", 'Hallunda', 'Alby', 'Fittja', 'Masmo', 'Vårby gård', 'Vårberg', 'Skärholmen', 'Sätra', 'Bredäng', 'Mälarhöjden', 'Axelsberg', 'Örnsberg', 'Aspudden', 'Liljeholmen', 'Hornstull', 'Zinkensdamm', 'Mariatorget', 'Slussen', 'Gamla stan', 'T-Centralen', 'Östermalmstorg', 'Karlaplan', 'Gärdet', 'Ropsten']
#line_13 = ["Fruängen", "Västertorp", "Hägerstensåsen", "Telefonplan", "Midsommarkransen", 'Liljeholmen', 'Hornstull', 'Zinkensdamm', 'Mariatorget','Slussen', 'Gamla stan', 'T-Centralen', 'Östermalmstorg', "Stadion", "Tekniska högskolan", "Universitetet", "Bergshamra", "Danderyds sjukhus", "Mörby centrum"]
stop_coordinates = {}

route_id_to_remove = "9011001001400000"
#route_id_to_remove = "9011001001300000"

with open("static_data/GTFS-SL-2023-01-01/stops-r.txt", "r") as file:
    for line in file:
        # Split the line by comma
        stop_data = line.strip().split(',')
        # Check if the stop_name exists in line_13
        if stop_data[1] in line_13:
            # Store the latitude and longitude values for the stop_name
            stop_coordinates[stop_data[1]] = (float(stop_data[2]), float(stop_data[3]))

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
trip_ids_to_plot = []
# Loop through folders '07', '08', and '09'
for folder_name in ["07", "08", "09", "10"]:
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
                        if trip_id not in trip_ids_to_plot:
                            trip_ids_to_plot.append(trip_id)
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
                                trip_data[trip_id] = {'latitude': [latitude], 'longitude': [longitude], 'timestamp': [timestamp]}

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

trip_latitude_values = []
trip_longitude_values = []
# Extract latitude and longitude values from trip_data
for trip_id, data in trip_data.items():
    trip_latitude_values.extend(data['latitude'])
    trip_longitude_values.extend(data['longitude'])

stop_latitude_values = []
stop_longitude_values = []
stop_names = []
# Extract latitude and longitude values from stop_coordinates
for stop_name, coordinates in stop_coordinates.items():
    latitude, longitude = coordinates
    stop_latitude_values.append(latitude)
    stop_longitude_values.append(longitude)
    stop_names.append(stop_name)

def remove_elements_randomly(list1, list2):
    # Combine the two lists into one list of tuples
    combined_lists = list(zip(list1, list2))
    # Shuffle the combined list randomly
    random.shuffle(combined_lists)

    # Calculate the number of elements to keep (15%)
    num_elements_to_keep = int(len(combined_lists) * 0.15)

    # Slice the combined list to keep only 10% of the elements
    combined_lists_trimmed = combined_lists[:num_elements_to_keep]

    # Unpack the trimmed combined list back into separate lists
    list1_trimmed, list2_trimmed = zip(*combined_lists_trimmed)

    return list(list1_trimmed), list(list2_trimmed)

def haversine(lat1, lon1, lat2, lon2):
    """
        Calculate the great-circle distance between two points on the Earth's surface
        given their latitude and longitude using the Haversine formula.

        Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

        Returns:
        float: The distance between the two points in kilometers.
    """

    R = 6371.0  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Calculate the change in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate distance using Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

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

def get_key_from_coordinate(coord, stop_coordinates):
    for key, value in stop_coordinates.items():
        if value == coord:
            return key
    return None

def calculate_actuall_dwell_time(time_list, station_list, trip_list):
    print(trip_list)
    trip_station_times = []

    # Iterate over each trip
    trip_station_times_dict = {}

    for trip_times, trip_stations, trip_id in zip(time_list, station_list, trip_list):
        # Initialize a dictionary to store station times for the current trip

        trip_station_times_dict[trip_id] = {}
        # Initialize variables to keep track of time and stations for the current trip
        current_station = None
        start_time = None


        # Iterate through each timestamp and station in the current trip
        for timestamp, station in zip(trip_times, trip_stations):
            if station != current_station:
                # If station changes, calculate the time spent at the previous station
                if current_station is not None:
                    # Adjust the end time to be the previous timestamp
                    end_time = previous_timestamp
                    time_spent = end_time - start_time
                    trip_station_times_dict[trip_id][current_station] = time_spent
                # Update current station, start time, and previous timestamp
                current_station = station
                start_time = timestamp
                # Update previous timestamp for the next iteration
            previous_timestamp = timestamp

        # Calculate time spent at the last station of the current trip
        end_time = trip_times[-1]
        time_spent = end_time - start_time
        trip_station_times_dict[trip_id][current_station] = time_spent

        # Append the station times for the current trip to the list
    trip_station_times.append(trip_station_times_dict)

    def plot_station_times(station_times):

        # Plot each station's times
        for station, times in aggregated_times.items():
            plt.figure(figsize=(8, 6))
            plt.plot(times, [station] * len(times), marker='o', linestyle='', color='blue')
            plt.title(f"Time Distribution at {station}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Station")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
        return aggregated_times

    # Example usage
    # Assuming you have already calculated the station times using the previous function

    # Plot station times for each station
    # Create a dictionary to aggregate times for each station
    aggregated_times = {}
    # Aggregate times for each station across all dictionaries

    for trip_data in trip_station_times:
        for trip_id, time in trip_data.items():
            for station in time:
                if station not in aggregated_times:
                    aggregated_times[station] = []
                aggregated_times[station].append({trip_id: time[station].total_seconds()})




    return aggregated_times

def optimal_proximity_range(data, stops_coordinates):
    def find_repeated_coordinates(filtered_trip_data):
        repeated_coordinates = {}

        for trip_id, trip_data in filtered_trip_data.items():
            latitude = trip_data['latitude']
            longitude = trip_data['longitude']
            timestamp = trip_data['timestamp']

            repeated_indexes = []
            for i in range(len(latitude)):
                if latitude.count(latitude[i]) >= 3 and longitude.count(longitude[i]) >= 3:
                    repeated_indexes.append(i)

            if repeated_indexes:
                repeated_coordinates[trip_id] = {
                    'latitude': [latitude[i] for i in repeated_indexes],
                    'longitude': [longitude[i] for i in repeated_indexes],
                    'timestamp': [timestamp[i] for i in repeated_indexes]
                }

        return repeated_coordinates


    def find_closest_station(repeated_coordinates, stops_coordinates):
        closest_stations = {}

        for trip_id, trip_data in repeated_coordinates.items():
            latitudes = trip_data['latitude']
            longitudes = trip_data['longitude']

            for i in range(len(latitudes)):
                min_distance = float('inf')
                closest_station = None

                for station, coords in stops_coordinates.items():
                    station_lat, station_lon = coords
                    distance = haversine(latitudes[i], longitudes[i], station_lat, station_lon)
                    if distance < min_distance:
                        min_distance = distance
                        closest_station = station

                closest_stations.setdefault(trip_id, []).append(
                    (latitudes[i], longitudes[i], closest_station, min_distance))

        return closest_stations



    def count_station_occurrences(closest_stations):
        trip_station_distances = defaultdict(lambda: defaultdict(list))
        lists = []

        for trip_id, coordinates in closest_stations.items():
            for coordinate in coordinates:
                latitude, longitude, station, proximity_range = coordinate
                trip_station_distances[trip_id][(latitude, longitude, station)].append(proximity_range)

        for trip_id, stations in trip_station_distances.items():

            for (latitude, longitude, station), proximity_ranges in stations.items():
                unique_proximity_ranges = set(proximity_ranges)
                for proximity_range in unique_proximity_ranges:
                    count = proximity_ranges.count(proximity_range)
                    lists.append([station, proximity_range, count])
                    #print(f"Station: {station}, Proximity Range: {proximity_range}, Occurrences: {count}")


        # Organize data into a dictionary where keys are station names and values are lists of distances
        station_data = defaultdict(list)
        for station, distance, occurrence in lists:
            station_data[station].append((distance, occurrence))

        # Dictionary to store the output for each station
        output_dict = {}

        # For each station, store the top three unique distances based on occurrence
        for station, distances in station_data.items():
            unique_distances = set()
            sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)
            top_three = []
            for distance, occurrence in sorted_distances:
                if distance not in unique_distances:
                    top_three.append((distance, occurrence))
                    unique_distances.add(distance)
                    if len(top_three) == 3:
                        break
            station_output = []
            for distance, occurrence in top_three:
                station_output.append({'Distance': distance, 'Occurrence': occurrence})
            output_dict[station] = station_output

        return output_dict

    repeated_coordinates = find_repeated_coordinates(data)
    closest_stations = find_closest_station(repeated_coordinates, stops_coordinates)
    proximity_dict = count_station_occurrences(closest_stations)

    print(proximity_dict)
    lowest_distance_dict = {}


    # For each station, find the dictionary with the lowest distance and store it in the new dictionary
    for station, data_list in proximity_dict.items():
        lowest_distance_dict[station] = min(data_list, key=lambda x: x['Distance'])


    lowest_ratio_dict = {}
    # For each station, find the dictionary with the lowest ratio and store it in the new dictionary
    for station, data_list in proximity_dict.items():
        # Calculate ratio (distance/occurrence) for each dictionary in data_list
        ratios = [(data['Distance'] / data['Occurrence']) for data in data_list]
        # Find the index of the dictionary with the lowest ratio
        lowest_ratio_index = ratios.index(min(ratios))
        # Store the dictionary with the lowest ratio in the new dictionary
        lowest_ratio_dict[station] = data_list[lowest_ratio_index]
    return lowest_ratio_dict

def remove_outliers_iqr(data):
    def detect_outliers(data):
        values = [list(item.values())[0] for item in data]
        quartile_1, quartile_3 = np.percentile(values, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (1.5 * iqr)
        upper_bound = quartile_3 + (1.5 * iqr)
        return np.logical_or(values < lower_bound, values > upper_bound)

    # Filter out outliers for each station
    filtered_data = {}
    for station, times in data.items():
        outliers_mask = detect_outliers(times)
        filtered_times = [times[i] for i, outlier in enumerate(outliers_mask) if not outlier]
        filtered_data[station] = filtered_times

    return filtered_data


# Example usage
list1 = trip_longitude_values
list2 = trip_latitude_values

list1_trimmed, list2_trimmed = remove_elements_randomly(list1, list2)
print(len(list1_trimmed))
print(len(list2_trimmed))

plt.figure(figsize=(12, 6))
plt.scatter(list1_trimmed, list2_trimmed, color="blue", alpha=0.5, label='Train Coordinates')
plt.scatter(stop_longitude_values, stop_latitude_values, color='red', label='Station Coordinates')
print(len(trip_longitude_values))
# Annotate each red dot with its station name
for i, stop_name in enumerate(stop_names):
    plt.annotate(stop_name, (stop_longitude_values[i], stop_latitude_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Train and Station Coordinates')
plt.legend()
plt.grid(True)
plt.show()

# Sample lists of longitude and latitude
longitudes = list1_trimmed
latitudes = list2_trimmed

# Starting point
current_point = (59.243739, 17.815538)
used_points = set()

total_distance = 0
coordinates = zip(latitudes, longitudes)

# Add station coordinates to the set
station_coordinates = set(zip(stop_latitude_values, stop_longitude_values))

unique_coordinates = set(coordinates).union(station_coordinates)

station_distances = {}

# Loop until all unique coordinates have been used
while len(used_points) < len(unique_coordinates):
    # Find the closest point to the current point that hasn't been used yet
    closest_point, distance, station = find_closest_point(current_point, unique_coordinates, used_points,
                                                          station_coordinates)

    # Add the closest point to the set of used points
    used_points.add(closest_point)

    # Update the current point to the closest point found
    current_point = closest_point

    # Accumulate the distance traveled so far
    total_distance += distance

    # If the closest point is a station
    if station:
        # Retrieve the station's name using its coordinates
        station_name = get_key_from_coordinate(closest_point, stop_coordinates)

        # Store the total distance traveled up to this station
        station_distances[station_name] = total_distance

distances_between_stations = {}
previous_station = None

# Iterate over each station and its associated distance
for station, distance in station_distances.items():
    if previous_station is not None:
        # Calculate the distance between the current station and the previous one
        s = previous_station + "-" + station
        distances_between_stations[s] = distance - station_distances[previous_station]

    # Update the previous station to the current one
    previous_station = station

# Output the distances between consecutive stations
print(distances_between_stations)

# Output the total distance traveled
print(f"Total distance traveled: {total_distance} km")

proximy_range_dict = optimal_proximity_range(trip_data, stop_coordinates)
filtered_trip_data = {}

# Iterate over trip_data
for trip_id, data in trip_data.items():
    # Initialize lists to store filtered latitude, longitude, and timestamp values
    filtered_latitude = []
    filtered_longitude = []
    filtered_timestamp = []
    station_names = []

    # Iterate over latitude, longitude, and timestamp values in trip_data
    for latitude, longitude, timestamp in zip(data['latitude'], data['longitude'], data['timestamp']):
        # Iterate over stations in stop_coordinates
        for station, (station_latitude, station_longitude) in stop_coordinates.items():
            # Check if the current latitude and longitude are within the specified range of the station
            distance = haversine(latitude, longitude, station_latitude, station_longitude)

            if abs(distance) <= proximy_range_dict[station]['Distance']:
                # Add the latitude, longitude, and timestamp to the filtered lists
                filtered_latitude.append(latitude)
                filtered_longitude.append(longitude)
                filtered_timestamp.append(timestamp)
                station_names.append(station)
                # Break the loop to avoid adding the same coordinates to multiple stations
                break

    # Store the filtered data in the new dictionary
    filtered_trip_data[trip_id] = {'latitude': filtered_latitude, 'longitude': filtered_longitude,
                                   'timestamp': filtered_timestamp, 'station_name': station_names}
    #filtered_trip_data[trip_id] = {'timestamp': filtered_timestamp, 'station_name': station_names}

data_tuples = []

# Print the filtered trip_data
station_to_index = {station: i for i, station in enumerate(line_13)}
plt.figure(figsize=(12, 6))
# Prepare data for plotting
all_timestamps = []
all_station_names = []
all_trip_ids = []
for trip_id, trip_data in filtered_trip_data.items():
    if trip_data:
        timestamps = [int(ts) for ts in trip_data['timestamp']]  # Convert timestamps to integers
        station_names = trip_data['station_name']
        # Convert Unix timestamps to datetime objects
        datetime_timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
        all_timestamps.append(datetime_timestamps)
        all_station_names.append(station_names)
        all_trip_ids.append(trip_id)
        # Plotting each trip separately
        plt.plot(datetime_timestamps, station_names)


dwell_times = calculate_actuall_dwell_time(all_timestamps, all_station_names, all_trip_ids)
actual_dwell_times = remove_outliers_iqr(dwell_times)

plt.xlabel('Timestamp (HH:MM:SS)')
plt.ylabel('Station')
plt.title('Trip Data')
plt.grid(True)
plt.tight_layout()
plt.show()
  