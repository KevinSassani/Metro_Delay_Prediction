"""
This script (imported from plot_realtime_map) plots the coordinates (latitude and longitude) of actual vehicle positions
together with station coordinates and calculates the distance between each station and the total distance.
The script also plots a diagram over actual arrival, departure and dwell times at each station between a time interval.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from plot_realtime_map import actual_dwell_times, distances_between_stations, trip_ids_to_plot
import math

def running_time(S1D, S2A, PS, CS, distances):
    max_speed = 90 / 3.6
    station = PS+"-"+CS
    distance_m = distances[station]*1000
    run_time = distance_m/max_speed
    return run_time


def check_allowable_dwell_time(dwell_times_mean, trip_data, station_distances):

    id = "14010000594129751"
    for i, station in enumerate(trip_data[id]):
        if i > 0:
            previous_station = trip_data[id][i - 1]["station"]
            current_station = station["station"]
            previous_departure_time = trip_data[id][i - 1]["departure"]
            arrival_time = station["arrival"]
            current_depart_time = trip_data[id][i]["departure"]

            run_time = running_time(previous_departure_time, arrival_time, previous_station, current_station, station_distances)
            time_diff = current_depart_time - previous_departure_time
            time_departure_diff_seconds = time_diff.total_seconds()
            allowable_dwell_time = time_departure_diff_seconds - run_time


            if dwell_times_mean[station["station"]] > allowable_dwell_time:
                dwell_times_mean[station["station"]] = allowable_dwell_time

    return dwell_times_mean

def calculate_planned_dwell_time(actual_times):

    average_values = {}

    # Iterate over each station
    for station, times in actual_times.items():
        # Extract values for the station
        values = [list(item.values())[0] for item in times]

        # Calculate average value
        average = sum(values) / len(values)

        # Store average value for the station
        average_values[station] = average

    return average_values


stop_id_to_name = {}
line_13 = ["Norsborg", 'Hallunda', 'Alby', 'Fittja', 'Masmo', 'Vårby gård', 'Vårberg', 'Skärholmen', 'Sätra', 'Bredäng',
           'Mälarhöjden', 'Axelsberg', 'Örnsberg', 'Aspudden', 'Liljeholmen', 'Hornstull', 'Zinkensdamm', 'Mariatorget',
           'Slussen', 'Gamla stan', 'T-Centralen', 'Östermalmstorg', 'Karlaplan', 'Gärdet', 'Ropsten']

with open("static_data/GTFS-SL-2023-01-01/stops-r.txt", 'r') as file:
    next(file)  # Skip the header
    for line in file:
        stop_id, stop_name, *_ = line.strip().split(',')
        stop_id_to_name[stop_id] = stop_name

# Read data from the file
file_path = "static_data/GTFS-SL-2023-01-01/stop_times-r.txt"

previous_trip_id = None
departure_time = []
departure_times = []
stop_ids = []
stop_id = []
trip_ids = []
trip_id = []
add = False
with open(file_path, 'r') as file:
    next(file)  # Skip the header
    for line in file:
        data = line.strip().split(',')

        departure_time_str = data[2]
        if '04:30' <= departure_time_str <= '12:30':
            current_trip_id = data[0]
            current_departure_time = datetime.strptime(departure_time_str, '%H:%M:%S')


            # Check if the trip_id has changed
            if previous_trip_id is None:
                previous_trip_id = current_trip_id
                add = True
                departure_time.append(current_departure_time)
                stop_id.append(data[3])
                trip_id.append(data[0])

            elif current_trip_id != previous_trip_id:

                if 7 <= current_departure_time.hour < 10:
                    # Save departure time for trip_id if it's between 7:30 and 9:30
                    trip_ids.append(trip_id[0])
                    departure_times.append(departure_time)
                    stop_ids.append(stop_id)
                    stop_id = []
                    departure_time = []
                    trip_id = []
                    add = True
                    if previous_trip_id is not None:
                        departure_time.append(current_departure_time)
                        stop_id.append(data[3])
                        trip_id.append(data[0])
                else:
                    add = False
                previous_trip_id = current_trip_id
            else:
                if add == True:
                    departure_time.append(current_departure_time)
                    stop_id.append(data[3])
                    trip_id.append(data[0])


planned_dwell_times = calculate_planned_dwell_time(actual_dwell_times)

stations = [[stop_id_to_name[key] for key in sublist if key in stop_id_to_name] for sublist in stop_ids]

line_13_route_id = "9011001001300000"
line_14_route_id = "9011001001400000"

trips_dict = {}
with open("static_data/GTFS-SL-2023-01-01/trips-r.txt", "r") as file:
    next(file)
    for line in file:
        trip_data = line.strip().split(',')
        trips_dict[trip_data[2]] = trip_data[0]


# Filtered lists
filtered_stations = []
filtered_departure_times = []
filtered_trip_ids = []

# Iterate over trip_ids and filter based on route_id
for i, tripID in enumerate(trip_ids):
    route_id = trips_dict.get(tripID)
    if route_id == line_13_route_id and tripID in trip_ids_to_plot:
        filtered_stations.append(stations[i])
        filtered_departure_times.append(departure_times[i])
        filtered_trip_ids.append(tripID)


all_arrival_times = []
for trip_departure_times, trip_station_list in zip(filtered_departure_times, filtered_stations):
    arrival_times = []
    for i, station in enumerate(trip_station_list):
        if i == 0:
            arrival_times.append(trip_departure_times[0])
        else:
            departure = trip_departure_times[i]
            dwell_time = int(math.ceil(planned_dwell_times[station]))
            arrival_time = departure - timedelta(seconds=dwell_time)
            arrival_times.append(arrival_time)
    all_arrival_times.append(arrival_times)

trip_data = {}
# Iterate over each trip ID and corresponding data
for trip_id, stations, departures, arrivals in zip(filtered_trip_ids, filtered_stations, filtered_departure_times, all_arrival_times):
    # Create a list of dictionaries containing station, departure, and arrival
    trip_info = [{'station': station, 'arrival': arrival, 'departure': departure} for station, departure, arrival in zip(stations, departures, arrivals)]
    # Assign the trip_info list to the trip_id in the dictionary
    trip_data[trip_id] = trip_info

dwell_times_after_allowable = check_allowable_dwell_time(planned_dwell_times, trip_data, distances_between_stations)

# Plotting the departure time over time
plt.figure(figsize=(12, 6))

for i, (station_list, departure_list) in enumerate(zip(filtered_stations, filtered_departure_times)):
    station_names = [station_list[i] for i in range(len(station_list))]
    plt.plot(departure_list, station_names, linestyle='-', color='k')  # Set color to black

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('Departure Time')
plt.ylabel('Station')
plt.title('Departure Time Plot for Stations')
plt.grid(True)
plt.tight_layout()
plt.show()


