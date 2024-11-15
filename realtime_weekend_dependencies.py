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
from tqdm import tqdm
from json.decoder import JSONDecodeError
import pickle


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


def calculate_actual_running_time(arrival_departure_data):
    """
       Calculate the actual running time between consecutive stations for each train and trip
       based on the provided arrival and departure data.

       Parameters:
       arrival_departure_data (list): A list of dictionaries containing date-specific
       arrival and departure times for multiple trips across various stations.

       Returns:
       dict: A nested dictionary where the top-level keys are dates, and each date maps
       to another dictionary. This second dictionary contains trip IDs as keys and
       dictionaries of station pairs as values. The station pairs are formatted as
       "prev_station-current_station" and map to the time in seconds it took to travel
       between the two stations.
    """

    running_time_dict = {}
    for list in arrival_departure_data:
        for data in list:
            for date, s in data.items():
                # Initialize the nested dictionary for the date if it doesn't exist
                if date not in running_time_dict:
                    running_time_dict[date] = {}

                for trip_id, stations in s[0].items():
                    trip_id_dict = {}
                    prev_station = None
                    for i, station in enumerate(stations):
                        if prev_station is not None:
                            current_station = station
                            previous_station = prev_station
                            previous_departure_time = stations[previous_station]["departure"]
                            current_arrival_time = stations[current_station]["arrival"]
                            time_diff = current_arrival_time - previous_departure_time
                            time_departure_diff_seconds = time_diff.total_seconds()
                            route_path = str(prev_station) + "-" + str(current_station)
                            trip_id_dict[route_path] = time_departure_diff_seconds
                        prev_station = station

                    # Assign the trip_id_dict to the corresponding date and trip_id
                    running_time_dict[date][trip_id] = trip_id_dict

    return running_time_dict


def calculate_actual_dwell_time(time_list, station_list, trip_list):
    """
        Calculate the actual dwell time at each station for the trains, based on
        the provided timestamps and station information. The method based on the
        dwell times also records the arrival and departure times for each station during the trips.

        Parameters:
        time_list (list): A list of lists where each sublist contains the timestamps for
        a particular trip.
        station_list (list): A list of lists where each sublist contains the stations
        visited in sequence during a particular trip.
        trip_list (list): A list of trip identifiers corresponding to each trip's
        time and station list.

        Returns:
        tuple:
            - dwell_time (dict): A dictionary where each key is a station, and the value
              is a list of dictionaries containing trip IDs and the corresponding dwell
              times in seconds for that station across all trips.
            - trip_arrival_departure_times (list): A list of dictionaries where each
              dictionary corresponds to a trip. For each trip, the dictionary contains
              stations as keys and another dictionary as the value, which holds the
              arrival and departure times for that station.
    """

    trip_station_times = []
    trip_arrival_departure_times = []
    trip_arrival_departure_times_dict = {}

    # Iterate over each trip
    trip_station_times_dict = {}
    for trip_times, trip_stations, trip_id in zip(time_list, station_list, trip_list):
        # Initialize a dictionary to store station times for the current trip
        trip_station_times_dict[trip_id] = {}
        trip_arrival_departure_times_dict[trip_id] = {}

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
                    trip_arrival_departure_times_dict[trip_id][current_station] = {"arrival": start_time, "departure": end_time}

                # Update current station, start time, and previous timestamp
                current_station = station
                start_time = timestamp
            # Update previous timestamp for the next iteration
            previous_timestamp = timestamp

        # Calculate time spent at the last station of the current trip
        end_time = trip_times[-1]
        time_spent = end_time - start_time
        trip_station_times_dict[trip_id][current_station] = time_spent
        trip_arrival_departure_times_dict[trip_id][current_station] = {"arrival": start_time, "departure": end_time}

    trip_station_times.append(trip_station_times_dict)
    trip_arrival_departure_times.append(trip_arrival_departure_times_dict)

    # Aggregate times for each station across all dictionaries
    aggregated_times = {}
    for trip_data in trip_station_times:
        for trip_id, time in trip_data.items():
            for station in time:
                if station not in aggregated_times:
                    aggregated_times[station] = []
                aggregated_times[station].append({trip_id: time[station].total_seconds()})

    dwell_time = aggregated_times
    return dwell_time, trip_arrival_departure_times


def calculate_actual_interval_between_trains(both_lines_departure_arrival):
    """
        Organizes the actual departure times of trains for each station on
        two metro lines (Line 13 and Line 14). The method processes the departure and
        arrival data for both lines, organizing it by date and station, and then sorting
        the trains at each station by their departure time in this way for each .


        Parameters:
        both_lines_departure_arrival (list): A list containing two elements, each of
        which is a list of dictionaries. The first element corresponds to Line 13, and
        the second to Line 14. Each dictionary contains date-specific data where keys
        are dates and values are lists of trip data, including station times.

        Returns:
        dict: A nested dictionary where the top-level keys are dates, each mapping to
        another dictionary. The second-level dictionary has stations as keys, which
        further map to a list of dictionaries. Each of these dictionaries contains a
        trip ID and the corresponding departure time at that station. The trips at each
        station are sorted by their departure times.
    """

    line13_actual_departure_arrival = both_lines_departure_arrival[0]
    line14_actual_departure_arrival = both_lines_departure_arrival[1]

    time_interval_between_trains = {}

    # Iterate through the list for line 13
    for element in line13_actual_departure_arrival:
        for date_key, trips_list in element.items():
            if date_key not in time_interval_between_trains:
                time_interval_between_trains[date_key] = {}
            for trip in trips_list:
                for trip_id, stations in trip.items():
                    for station, times in stations.items():
                        if station not in time_interval_between_trains[date_key]:
                            time_interval_between_trains[date_key][station] = []
                        time_interval_between_trains[date_key][station].append(
                            {'trip_id': trip_id, 'departure_time': times['departure']})

    # Iterate through the list for line 14
    for element in line14_actual_departure_arrival:
        for date_key, trips_list in element.items():
            if date_key not in time_interval_between_trains:
                time_interval_between_trains[date_key] = {}
            for trip in trips_list:
                for trip_id, stations in trip.items():
                    for station, times in stations.items():
                        if station not in time_interval_between_trains[date_key]:
                            time_interval_between_trains[date_key][station] = []
                        time_interval_between_trains[date_key][station].append(
                            {'trip_id': trip_id, 'departure_time': times['departure']})

    # Sort the trips for each station with the combine lines
    for date, stations_data in time_interval_between_trains.items():
        for station, trips in stations_data.items():
            sorted_trips = sorted(trips, key=lambda x: x['departure_time'])
            time_interval_between_trains[date][station] = sorted_trips

    return time_interval_between_trains


def optimal_proximity_range(data, stops_coordinates):
    """
        Determine the optimal proximity range of a train to the different stations by analyzing the train's
        location data and calculating the time spent within certain proximity ranges of various stations.

        The function performs several steps:
        1. Identifies repeated coordinates (latitude, longitude) where the train stays for a significant
           duration, suggesting it could be a potential stop at a station.
        2. For each repeated coordinate, finds the closest station from a list of predefined station coordinates.
        3. Counts the occurrences of the train staying within certain proximity ranges of each station.
        4. Calculates the time spent by the train at each station within specific proximity ranges.
        5. Returns the proximity range and the time spent within that range that gives the lowest ratio of
           distance/time for the different stations.

        Parameters:
        data (dict): A dictionary containing the train's location data for multiple trips, including latitude,
                     longitude, and timestamp information.
        stops_coordinates (dict): A dictionary containing the coordinates of the stations, where the keys are
                                  station names and values are tuples of (latitude, longitude).

        Returns:
        dict: A dictionary where the keys are station names and the values are dictionaries that contain the
              optimal proximity range (in terms of distance) and the time spent within that range. This range
              is determined by the lowest distance/time ratio.
    """
    def find_repeated_coordinates(filtered_trip_data):
        """
            Identify coordinates where the train remains for an extended period, suggesting it could be near or at a station.

            The function iterates through the location data of each trip, checking for coordinates (latitude and
            longitude) that appear repeatedly. Coordinates are considered repeated if they appear at least 5 times,
            which likely indicates that the train was stationary or moving very slowly at that location.

            Parameters:
            filtered_trip_data (dict): A dictionary containing location data for multiple trips. Each trip has a
                                       unique trip ID as the key, with the value being another dictionary containing
                                       lists of 'latitude', 'longitude', and 'timestamp' information.

            Returns:
            dict: A dictionary containing the repeated coordinates for each trip. The keys are trip IDs, and the
                  values are dictionaries with lists of repeated 'latitude', 'longitude', and corresponding 'timestamp'
                  data.
        """
        repeated_coordinates = {}

        for trip_id, trip_data in filtered_trip_data.items():
            latitude = trip_data['latitude']
            longitude = trip_data['longitude']
            timestamp = trip_data['timestamp']

            repeated_indexes = []
            for i in range(len(latitude)):
                if latitude.count(latitude[i]) >= 5 and longitude.count(longitude[i]) >= 5:
                    repeated_indexes.append(i)

            if repeated_indexes:
                repeated_coordinates[trip_id] = {
                    'latitude': [latitude[i] for i in repeated_indexes],
                    'longitude': [longitude[i] for i in repeated_indexes],
                    'timestamp': [timestamp[i] for i in repeated_indexes]
                }

        return repeated_coordinates

    def find_closest_station(repeated_coordinates, stops_coordinates):
        """
            Determine the closest station to repeated coordinates and calculate time differences when the train changes location.

            This function processes the coordinates where a train has been detected to linger (identified as repeated coordinates)
            and finds the nearest station for each of these points. It also calculates the time spent at a location before the
            train moves to a new position.

            Parameters:
            repeated_coordinates (dict): A dictionary where each key is a trip ID, and the value is a dictionary containing lists of
                                          'latitude', 'longitude', and 'timestamp' for locations where the train has repeated coordinates.
            stops_coordinates (dict): A dictionary where each key is a station name, and the value is a tuple with the station's latitude
                                      and longitude.

            Returns:
            tuple: A tuple containing:
                   - closest_stations (dict): A dictionary where each key is a trip ID, and the value is a list of tuples. Each tuple
                                              contains a latitude, longitude, closest station, and the distance to that station.
                   - time_diff_dict (dict): A dictionary where each key is a trip ID, and the value is a list of time differences (in
                                            seconds) calculated when the train changes its location.
        """

        closest_stations = {}

        time_diff_dict = {}
        for trip_id, trip_data in repeated_coordinates.items():
            latitudes = trip_data['latitude']
            longitudes = trip_data['longitude']
            timestamps = trip_data["timestamp"]
            start_time = int(timestamps[0])
            start_longitude = longitudes[0]
            start_latitude = latitudes[0]
            time_diff_list = []

            for i in range(len(latitudes)):
                time_diff = 0
                min_distance = float('inf')
                closest_station = None

                for station, coords in stops_coordinates.items():
                    station_lat, station_lon = coords
                    distance = haversine(latitudes[i], longitudes[i], station_lat, station_lon)
                    if distance < min_distance:
                        min_distance = distance
                        closest_station = station

                if start_latitude != latitudes[i] or start_longitude != longitudes[i]:
                    # Update start time
                    start_longitude = longitudes[i]
                    start_latitude = latitudes[i]
                    time_diff = int(timestamps[i - 1]) - int(start_time)
                    start_time = timestamps[i]
                    time_diff_list.append(time_diff)

                closest_stations.setdefault(trip_id, []).append(
                    (latitudes[i], longitudes[i], closest_station, min_distance))
            time_diff_dict[trip_id] = time_diff_list

        return closest_stations, time_diff_dict

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

    def count_station_time(closest_stations, time_diff):
        """
            Calculate and organize the time spent by trains within proximity ranges at each station.

            This function processes the closest stations and their respective proximity ranges for each trip, calculating
            the time spent by the train at different proximity ranges before moving to a new range. It organizes the results
            into a dictionary, where each station's data is further refined to include only significant time durations.

            Parameters:
            closest_stations (dict): A dictionary where each key is a trip ID, and the value is a list of tuples containing
                                     latitude, longitude, station name, and the proximity range for each recorded point.
            time_diff (dict): A dictionary where each key is a trip ID, and the value is a list of time differences (in seconds)
                              calculated when the train changes its location.

            Returns:
            dict: A dictionary where each key is a station name, and the value is a list of dictionaries. Each dictionary contains
                  a 'Distance' and 'Time' key, representing the proximity range and time spent at that range.
        """

        new_dict = {}
        for trip_id, stations in closest_stations.items():
            prev_range = None
            prev_station_name = None
            for station in stations:
                range_value = station[3]  # Extracting the proximity range
                station_name = station[2]  # Extracting the station name

                if prev_range is not None and prev_range != range_value:  # Checking if proximity range changes
                    time_diff_list = time_diff[trip_id]
                    if time_diff_list:
                        time_value = time_diff_list.pop(0)  # Pop the first element

                        if trip_id in new_dict:
                            if prev_station_name in new_dict[trip_id]:
                                new_dict[trip_id][prev_station_name].append((time_value, prev_range))
                            else:
                                new_dict[trip_id][prev_station_name] = [(time_value, prev_range)]
                        else:
                            new_dict[trip_id] = {prev_station_name: [(time_value, prev_range)]}

                prev_station_name = station_name
                prev_range = range_value

            # Handling the last station
            if prev_range is not None:
                time_diff_list = time_diff[trip_id]
                if time_diff_list:
                    time_value = time_diff_list.pop(0)  # Pop the first element

                    if trip_id in new_dict:
                        if prev_station_name in new_dict[trip_id]:
                            new_dict[trip_id][prev_station_name].append((time_value, prev_range))
                        else:
                            new_dict[trip_id][prev_station_name] = [(time_value, prev_range)]
                    else:
                        new_dict[trip_id] = {prev_station_name: [(time_value, prev_range)]}

        all_values = {}
        for trip_id, stations in new_dict.items():
            for station, values in stations.items():
                if station not in all_values:
                    all_values[station] = []
                for value in values:
                    all_values[station].append((value[0], value[1]))  # Append (time, range) tuple

        # Sort and extract the top three shortest range values with their corresponding times for each station
        # Never used
        top_three = {}
        for station, values in all_values.items():
            unique_ranges = sorted(set([x[1] for x in values]))  # Extract unique range values and sort them
            sorted_values = sorted(values, key=lambda x: (
                x[1], -x[0]))  # Sort by range value (second element of tuple) and time in descending order
            filtered_values = [(time, range_val) for time, range_val in sorted_values if time >= 5]
            top_three[station] = filtered_values[:3]

        station_output = {}
        for station, values in all_values.items():
            station_data = []
            for time, distance in values:
                station_data.append({'Distance': distance, 'Time': time})
            station_output[station] = station_data
        station_output_filtered = {}

        for station, data_list in station_output.items():
            filtered_data = [data for data in data_list if data['Time'] >= 5]
            station_output_filtered[station] = filtered_data

        return station_output_filtered

    repeated_coordinates = find_repeated_coordinates(data)
    closest_stations, time_diffs = find_closest_station(repeated_coordinates, stops_coordinates)

    # Based on how many times timestamp is updated
    # proximity_dict = count_station_occurrences(closest_stations)
    # Based on time difference between first and last timestamp at a station (more accurate)
    proximity_dict = count_station_time(closest_stations, time_diffs)

    lowest_distance_dict = {}

    # For each station, find the dictionary with the lowest distance and store it in the new dictionary
    # (Not used in experiments)
    for station, data_list in proximity_dict.items():
        if not data_list:  # If data_list is empty
            lowest_distance_dict[station] = 0.006857780529392969
        else:
            lowest_distance_dict[station] = min(data_list, key=lambda x: x['Distance'])['Distance']
    lowest_ratio_dict = {}

    # For each station, find the dictionary with the lowest ratio and store it in the new dictionary
    ratios = 0
    for station, data_list in proximity_dict.items():
        # Calculate ratio (distance/occurrence) for each dictionary in data_list
        if not data_list:  # Never actually used, only if data_list is empty
            proximity_dict[station] = [{'Distance': 0.01, 'Time': 1}]
            ratios = [0.01]
        else:
            ratios = [(data['Distance'] / data['Time']) for data in data_list]
        # Find the index of the dictionary with the lowest ratio
        lowest_ratio_index = ratios.index(min(ratios))

        # Store the dictionary with the lowest ratio in the new dictionary
        if not data_list:
            lowest_ratio_dict[station] = {'Distance': 0.01, 'Time': 1}
        else:
            lowest_ratio_dict[station] = data_list[lowest_ratio_index]

    return lowest_ratio_dict


line_13 = ["Norsborg", 'Hallunda', 'Alby', 'Fittja', 'Masmo', 'Vårby gård', 'Vårberg', 'Skärholmen', 'Sätra', 'Bredäng',
           'Mälarhöjden', 'Axelsberg', 'Örnsberg', 'Aspudden', 'Liljeholmen', 'Hornstull', 'Zinkensdamm', 'Mariatorget',
           'Slussen', 'Gamla stan', 'T-Centralen', 'Östermalmstorg', 'Karlaplan', 'Gärdet', 'Ropsten']

line_14 = ["Fruängen", "Västertorp", "Hägerstensåsen", "Telefonplan", "Midsommarkransen", 'Liljeholmen', 'Hornstull',
           'Zinkensdamm', 'Mariatorget', 'Slussen', 'Gamla stan', 'T-Centralen', 'Östermalmstorg', "Stadion",
           "Tekniska högskolan", "Universitetet", "Bergshamra", "Danderyds sjukhus", "Mörby centrum"]

both_lines_departure_arrival = []

weekend = None # If weekend = None then both weekdays and weekends are considered
# actual_arrival_departure_list = []
start_date = datetime.date(2023, 1, 1)
#end_date = datetime.date(2023, 2, 25)
end_date = datetime.date(2023, 1, 4)

folder1 = []
folder2 = []
sl_lines = ["13", "14"]

# Loop over the dates within the specified range
current_date = start_date
while current_date <= end_date:
    # Check if the current date is a weekend day (Saturday or Sunday)
    if weekend == True:
        if current_date.weekday() in [5, 6]:  # 5 is Saturday, 6 is Sunday
            # Append the formatted string to the folder lists
            folder1.append(current_date.strftime("%Y-%m-%d"))
            folder2.append(current_date.strftime("%Y/%m/%d"))
    elif weekend == False:
        if current_date.weekday() in [0, 1, 2, 3, 4]:  # 0 is Monday, 4 is Friday
            # Append the formatted string to the folder lists
            folder1.append(current_date.strftime("%Y-%m-%d"))
            folder2.append(current_date.strftime("%Y/%m/%d"))
    else:
        folder1.append(current_date.strftime("%Y-%m-%d"))
        folder2.append(current_date.strftime("%Y/%m/%d"))
    # Move to the next date
    current_date += datetime.timedelta(days=1)
print(folder1)

total_iterations = len(folder2)
i = 0
route_id_to_remove_line_13 = "9011001001400000"  # use this if line 13
route_id_to_remove_line_14 = "9011001001300000"  # use this if line 14
trip_id_list_check = []
route_id_to_remove_lines = [route_id_to_remove_line_13, route_id_to_remove_line_14]
lines = [line_13, line_14]
all_actual_dwell_times = []
all_actual_running_times = []
all_actual_dwell_times_list = []
all_actual_dwell_times_dict = {}

for train_line, route_id_to_remove in zip(lines, route_id_to_remove_lines):
    trip_id_list_print = []
    actual_arrival_departure_list = []
    all_trip_ids_list = []
    all_timestamps = []
    all_station_names = []
    all_trip_ids = []
    for f1, f2 in tqdm(zip(folder1, folder2), total=total_iterations):
        stop_id_to_name = {}
        stop_coordinates = {}
        # Read the "stops-r.txt" file
        with open("static_data/GTFS-SL-" + f1 + "/stops-r.txt", "r") as file:
            # Iterate through each line
            for line in file:
                # Split the line by comma
                stop_data = line.strip().split(',')
                # Check if the stop_name exists in line_13
                if stop_data[1] in train_line:
                    # Store the latitude and longitude values for the stop_name
                    stop_coordinates[stop_data[1]] = (float(stop_data[2]), float(stop_data[3]))

        # Read the first .txt file to map trip_id with route_id
        trip_id_to_route_id = {}
        with open("static_data/GTFS-SL-" + f1 + "/trips-r.txt", 'r') as file:
            next(file)  # Skip header
            for line in file:
                route_id, _, trip_id, *_ = line.strip().split(',')
                trip_id_to_route_id[trip_id] = route_id

        trip_id_to_route_id_filtered = {}
        trip_id_to_route_id_filtered_check = {}

        # Iterate over the original trip_id_to_route_id dictionary
        for trip_id, route_id in trip_id_to_route_id.items():
            # Check if the route_id is not equal to route_id_to_remove
            if route_id != route_id_to_remove:
                # Add the (trip_id, route_id) pair to the filtered dictionary
                trip_id_to_route_id_filtered[trip_id] = route_id
            if route_id != route_id_to_remove_line_14:
                trip_id_to_route_id_filtered_check[trip_id] = route_id

        # Update trip_id_to_route_id with the filtered dictionary
        trip_id_to_route_id = trip_id_to_route_id_filtered
        for tripid in trip_id_to_route_id_filtered_check:
            if tripid not in trip_id_list_check:
                trip_id_list_check.append(tripid)
        trip_id_list = list(trip_id_to_route_id.keys())

        # Directory path where folders '00' to '23' are located
        directory = "realtime_vehicle_data/sl-" + f1 + "/VehiclePositions/" + f2

        # List to store data from JSON files
        trip_data = {}

        # Loop through folders '07', '08', and '09'
        for folder_name in ["04", "05",
                            "06", "07"]:  # "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"
            folder_path = os.path.join(directory, folder_name)
            # Check if the folder exists
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                # Loop through JSON files in the folder
                for filename in os.listdir(folder_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(folder_path, filename)
                        # Read JSON data from the file
                        with open(file_path, 'r') as file:
                            try:
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
                            except JSONDecodeError:
                                pass

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

        proximity_range_dict = optimal_proximity_range(trip_data, stop_coordinates)

        # Define the new dictionary to store filtered data
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

                    if station in proximity_range_dict:
                        if abs(distance) <= proximity_range_dict[station]['Distance']:
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

        station_to_index = {station: i for i, station in enumerate(train_line)}

        for trip_id, trip_data in filtered_trip_data.items():
            if trip_data:
                timestamps = [int(ts) for ts in trip_data['timestamp']]  # Convert timestamps to integers
                station_names = trip_data['station_name']
                # Convert Unix timestamps to datetime objects
                datetime_timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
                all_timestamps.append(datetime_timestamps)
                all_station_names.append(station_names)
                all_trip_ids.append(trip_id)

        i += 1
        print("1" + "/" + str(i))

    indexes = []

    for id in all_trip_ids:
        specific_trip_id_timestamps = []
        specific_trip_id_stations = []
        specific_trip_id = []
        indices_of_id = [index for index, value in enumerate(all_trip_ids) if value == id]
        index = all_trip_ids.index(id)

        if index in indexes:
            for i in indices_of_id:
                if i not in indexes:
                    index = i
                    indexes.append(index)
                    break
        else:
            indexes.append(index)

        specific_trip_id_timestamps.append(all_timestamps[index])
        specific_trip_id_stations.append(all_station_names[index])
        specific_trip_id.append(all_trip_ids[index])

        # Check if trip id have times registered
        if specific_trip_id_timestamps[0]:
            dwell_times, actual_arrival_departure = calculate_actual_dwell_time(specific_trip_id_timestamps,
                                                                                specific_trip_id_stations, specific_trip_id)

            first_trip_data = actual_arrival_departure[0]  # Assuming there's at least one trip data in the list
            first_trip_id = list(first_trip_data.keys())[0]  # Get the first trip ID
            first_station_data = first_trip_data[first_trip_id]  # Get data for the first station

            # Assuming 'arrival' key exists for the first station just to get information about which date it is
            first_station_arrival = first_station_data[list(first_station_data.keys())[0]]['arrival']
            # Creating a datetime object with only year, month, and day
            date_object = datetime.date(first_station_arrival.year, first_station_arrival.month,
                                        first_station_arrival.day)

            # Store the dwell time to the right date
            if date_object not in all_actual_dwell_times_dict:
                all_actual_dwell_times_dict[date_object] = []
            all_actual_dwell_times_dict[date_object].append(dwell_times)
            all_actual_dwell_times.append(dwell_times)

            # Store the actual arrival and departure time to the correct date
            trip_id = list(actual_arrival_departure[0].keys())[0]
            first_station = list(actual_arrival_departure[0][trip_id].keys())[0]
            trip_date = actual_arrival_departure[0][trip_id][first_station]["arrival"].date()
            actual_arrival_departure_list.append({trip_date: actual_arrival_departure})

    both_lines_departure_arrival.append(actual_arrival_departure_list)

actual_interval_between_trains = calculate_actual_interval_between_trains(both_lines_departure_arrival)
actual_running_times = calculate_actual_running_time(both_lines_departure_arrival)

data_to_save = {
    'trip_id_list_check_line14': trip_id_list_check,
}

# Save the dictionary to a file
#with open('trip_id_list_check_line_14_2M.pkl', 'wb') as file:
    #pickle.dump(data_to_save, file)

