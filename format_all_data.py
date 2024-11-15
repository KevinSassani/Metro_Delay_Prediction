"""
This script formats the data into desired OF and NF 3D matrices used as input to the models
"""

import numpy as np
import datetime as dt
import math
from datetime import datetime, timedelta
import statistics
from collections import defaultdict
import h5py
import pickle


def find_trip_id_index(trip_list, target_trip_id):
    for index, trip in enumerate(trip_list):
        if trip['trip_id'] == target_trip_id:
            return index
    return -1  # Return -1 if the trip_id is not found


def calculate_average_delay(delay_data):
    station_totals = {}

    for day_data in delay_data:
        for date, trips in day_data.items():
            for trip_id, stations in trips.items():
                for station, delay in stations.items():
                    if station not in station_totals:
                        station_totals[station] = {'total_delay': timedelta(), 'count': 0}
                    station_totals[station]['total_delay'] += delay
                    station_totals[station]['count'] += 1

    # Calculate averages
    station_averages = {
        station: total['total_delay'].total_seconds() / total['count']
        for station, total in station_totals.items()
    }

    return station_averages


def average_time_between_trains(data):
    avg_times = {}
    for day_data in data:
        for date_str, stations in day_data.items():
            for station, departures in stations.items():
                if len(departures) < 2:
                    continue  # Not enough trains to calculate intervals

                # Sort departures by time to ensure correct interval calculation
                sorted_departures = sorted(departures, key=lambda x: x['departure_time'])

                # Calculate intervals in seconds between consecutive departures
                intervals = []
                for i in range(1, len(sorted_departures)):
                    delta = sorted_departures[i]['departure_time'] - sorted_departures[i - 1]['departure_time']
                    intervals.append(delta.total_seconds())

                # Calculate the average interval for the station
                if intervals:
                    avg_times[station] = statistics.mean(intervals)
    return avg_times


with open('saved_realtime_planned_data_2M.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Extract variables from the loaded dictionary
all_actual_dwell_times_dict = loaded_data['all_actual_dwell_times_dict']
actual_interval_between_trains = loaded_data['actual_interval_between_trains']
actual_running_times = loaded_data['actual_running_times']
trip_id_list_check = loaded_data['trip_id_list_check']
all_planned_running_times = loaded_data['all_planned_running_times']
all_arrival_delays = loaded_data['all_arrival_delays']
all_departure_delays = loaded_data['all_departure_delays']
all_planned_interval_between_trains = loaded_data['all_planned_interval_between_trains']
dwell_times_after_allowable = loaded_data['dwell_times_after_allowable']
all_suppliment_buffer = loaded_data['all_suppliment_buffer']
distances_between_stations = loaded_data['distances_between_stations']

with open("trip_id_list_check_line_14_2M.pkl", 'rb') as file:
    loaded_data_line14 = pickle.load(file)
print("Data has been loaded")

# Access the data using the dictionary key
trip_id_list_check_line_14 = loaded_data_line14['trip_id_list_check_line14']

line_13 = ["Norsborg", 'Hallunda', 'Alby', 'Fittja', 'Masmo', 'Vårby gård', 'Vårberg', 'Skärholmen', 'Sätra', 'Bredäng',
           'Mälarhöjden', 'Axelsberg', 'Örnsberg', 'Aspudden', 'Liljeholmen', 'Hornstull', 'Zinkensdamm', 'Mariatorget',
           'Slussen', 'Gamla stan', 'T-Centralen', 'Östermalmstorg', 'Karlaplan', 'Gärdet', 'Ropsten']

line_14 = ["Fruängen", "Västertorp", "Hägerstensåsen", "Telefonplan", "Midsommarkransen", 'Liljeholmen', 'Hornstull',
           'Zinkensdamm', 'Mariatorget',
           'Slussen', 'Gamla stan', 'T-Centralen', 'Östermalmstorg', "Stadion", "Tekniska högskolan", "Universitetet",
           "Bergshamra", "Danderyds sjukhus", "Mörby centrum"]

H = 3  # Number of trains before I considered, including I
Z = 3  # Number of stations before P considered, including P
I = 3  # Train of interest (should be the same as H)
P = 3  # Current station of interest (should be the same as Z)
line13 = False #If False = line14
train_line = None


if line13 == True:
    train_line = line_13
else:
    train_line = line_14
    trip_id_list_check = trip_id_list_check_line_14

station_averages_actual_interval_between_trains = {}

for day, stations in actual_interval_between_trains.items():
    for station, departures in stations.items():
        # Sort departures by departure_time to ensure correct order
        sorted_departures = sorted(departures, key=lambda x: x['departure_time'])

        # Calculate time differences in seconds between consecutive departures
        time_diffs = []
        for i in range(1, len(sorted_departures)):
            diff = (sorted_departures[i]['departure_time'] - sorted_departures[i - 1]['departure_time']).total_seconds()
            time_diffs.append(diff)

        # Calculate the average time difference if there are any differences
        if time_diffs:
            average_diff = sum(time_diffs) / len(time_diffs)
            if station not in station_averages_actual_interval_between_trains:
                station_averages_actual_interval_between_trains[station] = []
            station_averages_actual_interval_between_trains[station].append(average_diff)

# Iterate through each date in the data structure
trip_id_to_departure_time = {}

key_stations_dict = {}
for date, stations in actual_interval_between_trains.items():
    key_stations_dict[date] = {}  # Initialize empty dictionary for each date
    for station, trains in stations.items():
        if station in train_line:
            key_stations_dict[date][station] = trains

ordered_interval_between_trains = {}

# Iterate through each date in the original dictionary
for date, stations in key_stations_dict.items():
    # Initialize a new dictionary for the current date
    ordered_interval_between_trains[date] = {}
    # Iterate through each station in the train line
    for station in train_line:
        # Check if the station exists in the original dictionary
        if station in stations:
            # Add the station and its corresponding value to the ordered dictionary
            ordered_interval_between_trains[date][station] = stations[station]

# Iterate through each date in the data structure
entries = None  # From which station the train order is considered from
# print(ordered_interval_between_trains)
for date in ordered_interval_between_trains:
    if date not in trip_id_to_departure_time:
        trip_id_to_departure_time[date] = {}

    # Check if 'Hornstull' (line 13) or 'Västertorp' station exists for the current date to make sure that
    # the trains are in usage because sometimes the train are just in the first station and no more
    if 'Hallunda' in ordered_interval_between_trains[date] or "Västertorp" in ordered_interval_between_trains[date]:
        if 'Hallunda' in ordered_interval_between_trains[date]:
            entries = ordered_interval_between_trains[date]['Hallunda']
        elif "Västertorp" in ordered_interval_between_trains[date]:
            entries = ordered_interval_between_trains[date]['Västertorp']

        # Extract trip_id and departure_time from entries
        for entry in entries:
            trip_id = entry['trip_id']
            first_departure_time = entry['departure_time']
            if (trip_id not in trip_id_to_departure_time[date] and trip_id in trip_id_list_check and
                    actual_running_times[date].get(trip_id)):
                trip_id_to_departure_time[date][trip_id] = first_departure_time

# Now trip_id_to_departure_time contains unique trip IDs mapped to their corresponding first_departure_time

# Preparation for calculating average actual running times for each station
station_sum = {}
station_count = {}
for date, station_data in actual_running_times.items():
    for _, station_values in station_data.items():
        for station, value in station_values.items():
            if station in station_sum:
                station_sum[station] += value
                station_count[station] += 1
            else:
                station_sum[station] = value
                station_count[station] = 1

# Calculate the average actual running time for each station
station_average_running_time = {station: float(math.ceil(station_sum[station] / station_count[station])) for station in
                                station_sum}

# Preparation for calculating average actual dwell times for each station
station_sums = {}
station_counts = {}
for date_key, entries in all_actual_dwell_times_dict.items():
    for entry in entries:
        for station, values in entry.items():
            for value_dict in values:
                for value_key, value in value_dict.items():
                    if station not in station_sums:
                        station_sums[station] = 0
                        station_counts[station] = 0
                    station_sums[station] += value
                    station_counts[station] += 1

# Calculate the average actual dwell time for each station
station_averages_actual_dwell_times = {station: float(math.ceil(station_sums[station] / station_counts[station]))
                                       for station in station_sums}

# Preparation for calculating average planned running times for each station
station_totals = {}
station_counts = {}
for day_data in all_planned_running_times:
    for date, trips in day_data.items():
        for trip_id, stations in trips.items():
            for station_data in stations:
                for station_pair, travel_time in station_data.items():
                    if station_pair not in station_totals:
                        station_totals[station_pair] = 0
                        station_counts[station_pair] = 0
                    station_totals[station_pair] += travel_time
                    station_counts[station_pair] += 1

# Calculate the planned running time for each station
station_averages_planned_running_times = {station: float(math.ceil(station_totals[station] / station_counts[station]))
                                          for station in station_totals}

# Preparation for calculating average planned dwell times for each station
station_totals = {}
station_counts = {}
for trip_id, stations in dwell_times_after_allowable.items():
    for station, dwell_time in stations.items():
        if station not in station_totals:
            station_totals[station] = 0
            station_counts[station] = 0
        station_totals[station] += dwell_time
        station_counts[station] += 1

# Calculate the average dwell time for each station
station_averages_planned_dwell_times = {}
for station, total_time in station_totals.items():
    count = station_counts[station]
    average_time = total_time / count if count != 0 else 0
    station_averages_planned_dwell_times[station] = average_time
station_totals = defaultdict(lambda: {'sum': 0, 'count': 0})

# Iterate over the data to populate the station_totals dictionary
for day_data in all_suppliment_buffer:
    for date, stations in day_data.items():
        for trip, times in stations.items():
            for station, value in times.items():
                station_totals[station]['sum'] += value
                station_totals[station]['count'] += 1

# Calculate the average running time supplement for each station
station_averages_suppliment_buffer = {station: totals['sum'] / totals['count'] for station, totals in
                                      station_totals.items()}

# Calculate the average planned times between trains
station_average_planned_interval_between_trains = average_time_between_trains(all_planned_interval_between_trains)

# Calculate average arrival delays
station_average_arrival_delays = calculate_average_delay(all_arrival_delays)

# Calculate average departure delays
station_average_departure_delays = calculate_average_delay(all_departure_delays)

all_station_averages_actual_dwell_times = (sum(station_averages_actual_dwell_times.values())
                                           / len(station_averages_actual_dwell_times))

all_station_average_running_time = sum(station_average_running_time.values()) / len(station_average_running_time)

all_values_actual_interval_between_trains = [value for sublist in station_averages_actual_interval_between_trains.values()
                                             for value in sublist]

all_station_averages_actual_interval_between_trains = sum(all_values_actual_interval_between_trains) / len(
    all_values_actual_interval_between_trains)
all_station_average_arrival_delays = float(
    math.ceil(sum(station_average_arrival_delays.values()) / len(station_average_arrival_delays)))
all_station_average_departure_delays = float(
    math.ceil(sum(station_average_departure_delays.values()) / len(station_average_departure_delays)))
all_station_averages_planned_running_times = float(
    math.ceil(sum(station_averages_planned_running_times.values()) / len(station_averages_planned_running_times)))
all_station_averages_planned_dwell_times = float(
    math.ceil(sum(station_averages_planned_dwell_times.values()) / len(station_averages_planned_dwell_times)))
all_station_average_planned_interval_between_trains = float(math.ceil(
    sum(station_average_planned_interval_between_trains.values()) / len(station_average_planned_interval_between_trains)))
all_station_averages_suppliment_buffer = float(
    math.ceil(sum(station_averages_suppliment_buffer.values()) / len(station_averages_suppliment_buffer)))

date = dt.date(2023, 1, 1)
current_I = 0
current_P = 0
OF_matrces = []
NF_matrces = []

sorted_trip_id_to_departure_time = {}

for date, trips in trip_id_to_departure_time.items():
    # Sort the trips by departure time
    sorted_trips = dict(sorted(trips.items(), key=lambda item: item[1]))
    sorted_trip_id_to_departure_time[date] = sorted_trips

for k, date in enumerate(sorted_trip_id_to_departure_time):
    print(date)
    date2 = date.strftime('%Y-%m-%d')
    if date == datetime(2023, 3, 2).date():
        break
    for train in sorted_trip_id_to_departure_time[date]:
        current_I += 1  # Train of interest
        for train_station in train_line:
            current_P += 1  # Current station
            OF = np.zeros((H, 9 * Z))
            NF = np.zeros((H, 2 * Z))
            train_id = None
            a = 0  # Represents the row index which is the train we are currently processing
            b = 0  # Represents the column index which is the station we are currently processing

            # Fill OF matrix with actual running time features
            for i in range(I - H + 1, I + 1):
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    t = 0
                    if current_P - Z >= 0 and current_I - H >= 0:  # Check if there are enough with trains and stations back
                        station = train_line[current_P - Z + b - 1]

                        if b == 1:  # If we are on the first station
                            prev_station = None
                        else:
                            prev_station = train_line[current_P - Z + b - 1 - 1]

                        # Determine the correct train_id
                        train_id = list(sorted_trip_id_to_departure_time[date].keys())[current_I - H + a - 1]

                        if prev_station is not None:
                            running_time_stations = prev_station + "-" + station
                            if running_time_stations in actual_running_times[date][train_id]:
                                t = actual_running_times[date][train_id][running_time_stations]
                            else:
                                t = station_average_running_time[running_time_stations]

                        # Append the calculated actual running time value to the OF matrix
                        OF[a - 1][b - 1] = t
                    else:
                        # There are no actual running time, therefor it is set to 0
                        t = 0
                        OF[a - 1][b - 1] = t
                b = 0

            a = 0
            b = 0
            # Fill OF matrix with actual dwell time features
            for i in range(I - H + 1, I + 1):
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    if current_P - Z >= 0 and current_I - H >= 0:
                        station = train_line[current_P - Z + b - 1]
                        train_id = list(sorted_trip_id_to_departure_time[date].keys())[current_I - H + a - 1]
                        w = station_averages_actual_dwell_times[station]
                        if date in all_actual_dwell_times_dict:
                            station_data = all_actual_dwell_times_dict[date]
                            for station_dict in station_data:
                                if station in station_dict:
                                    id_list = station_dict[station]
                                    for id_dict in id_list:
                                        if train_id in id_dict:
                                            w = id_dict[train_id]
                                            break

                        # Append the calculated t value to the list
                        OF[a - 1][b - 1 + Z] = w
                    else:
                        w = 0
                        OF[a - 1][b - 1 + Z] = w
                b = 0

            prev_train_id = None
            train_id = None
            prev_train_time = 0
            current_train_time = 0
            a = 0
            b = 0
            # Fill OF matrix with actual time interval between two consecutive trains at a specific station features
            for i in range(I - H + 1, I + 1):
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    r = 0
                    if current_P - Z >= 0 and current_I - H >= 0:
                        station = train_line[current_P - Z + b - 1]
                        if station in actual_interval_between_trains[date]:

                            if station == "Norsborg" or station == "Fruängen":  # First station of line13 respectivley line14
                                if current_I - H + a - 1 < len(actual_interval_between_trains[date][station]):
                                    train_id = actual_interval_between_trains[date][station][current_I - H + a - 1]
                                else:
                                    train_id = None
                                # First train
                                if a == 1:
                                    prev_train_id = None
                                # Get the train number index to be able to extract the correct previous train id
                                else:
                                    index = find_trip_id_index(actual_interval_between_trains[date][station], train_id)
                                    if 0 <= index - 1 < len(actual_interval_between_trains[date][station]):
                                        prev_train_id = actual_interval_between_trains[date][station][index - 1][
                                            'trip_id']
                                    else:
                                        prev_train_id = None

                            # If not the first station
                            else:
                                train_id = list(sorted_trip_id_to_departure_time[date].keys())[current_I - H + a - 1]
                                # First train
                                if a == 1:
                                    prev_train_id = None
                                # Get the train number index to be able to extract the correct previous train id
                                else:
                                    index = find_trip_id_index(actual_interval_between_trains[date][station], train_id)
                                    if 0 <= index - 1 < len(actual_interval_between_trains[date][station]):
                                        prev_train_id = actual_interval_between_trains[date][station][index - 1][
                                            'trip_id']
                                    else:
                                        prev_train_id = None
                        else:
                            train_id = None

                        # Initialize current_train_time
                        current_train_time = None

                        if prev_train_id is not None or train_id is not None:
                            if date in actual_interval_between_trains and station in actual_interval_between_trains[
                                date]:
                                trips = actual_interval_between_trains[date][station]
                                time_diff = 0
                                for trip in trips:
                                    if trip['trip_id'] == train_id:
                                        current_train_time = trip['departure_time']
                                    if trip['trip_id'] == prev_train_id:
                                        prev_train_time = trip['departure_time']

                                # Calculate the time difference between current and previous train
                                if (prev_train_time is None or isinstance(prev_train_time, int) or
                                        current_train_time is None or isinstance(current_train_time, int)):
                                    time_diff = float(
                                        math.ceil(station_average_planned_interval_between_trains[station]))
                                else:
                                    time_diff = (current_train_time - prev_train_time).total_seconds()

                                r = time_diff
                        else:
                            r = 0
                        OF[a - 1][b - 1 + (Z * 2)] = r
                    else:
                        r = 0
                        OF[a - 1][b - 1 + (Z * 2)] = r
                b = 0


            def find_value_by_key(data, key):
                for item in data:
                    if key in item:
                        return item[key]
                return None


            arrival_delay = None
            departure_delay = None

            a = 0
            b = 0
            # Fill OF matrix with actual arrival and departure delay features
            for i in range(I - H + 1, I + 1):
                # Iterate over columns
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    d = 0  # Departure delay
                    y = 0  # Arrival delay

                    if current_P - Z >= 0 and current_I - H >= 0:
                        departure_delay = 0
                        arrival_delay = 0

                        station = train_line[current_P - Z + b - 1]

                        train_id = list(sorted_trip_id_to_departure_time[date].keys())[current_I - H + a - 1]
                        data = find_value_by_key(all_departure_delays, date2)  # Get the train data for a specific date

                        if train_id in data:
                            if station in data[train_id]:
                                departure_delay = data[train_id][station]  # Extract the correct departure delay
                                if not isinstance(departure_delay, int):  # The departure delays are saved as strings
                                    d = int(departure_delay.total_seconds())
                            else:
                                # If station data for a specific train does not exist then departure delay is the
                                # average of the specific station
                                d = float(math.ceil(station_average_departure_delays[station]))
                        else:
                            # If the specific train id does not exist in the data then the departure delay is the
                            # average of all the stations
                            d = float(math.ceil(all_station_average_departure_delays))

                        # Now the same thing for arrival delay
                        data = find_value_by_key(all_arrival_delays, date2)
                        # The first station does not have an arrival delay
                        if station == "Norsborg" or station == "Fruängen":
                            arrival_delay = 0
                        elif train_id in data:
                            if station in data[train_id]:
                                arrival_delay = data[train_id][station]
                                if not isinstance(arrival_delay, int):
                                    y = int(arrival_delay.total_seconds())
                            else:
                                y = float(math.ceil(station_average_arrival_delays[station]))
                        else:
                            y = float(math.ceil(all_station_average_arrival_delays))

                        OF[a - 1][b - 1 + (Z * 3)] = d
                        OF[a - 1][b - 1 + (Z * 4)] = y

                    else:
                        d = 0
                        y = 0
                        OF[a - 1][b - 1 + (Z * 3)] = d
                        OF[a - 1][b - 1 + (Z * 4)] = y

                b = 0

            train_id = None
            next_station = None
            a = 0
            b = 0

            # Fill OF matrix with planned running time features
            for i in range(I - H + 1, I + 1):
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    tprime = 0
                    if current_P - Z >= 0 and current_I - H >= 0:
                        station = train_line[current_P - Z + b - 1]

                        # If last station there are no next station
                        if station == "Ropsten" or station == "Mörby centrum":
                            next_station = None
                        else:
                            next_station = train_line[current_P - Z + b]

                        train_id = list(sorted_trip_id_to_departure_time[date].keys())[current_I - H + a - 1]

                        if next_station is not None:
                            running_time_stations = station + "-" + next_station
                            if train_id in all_planned_running_times[k][date2]:
                                if running_time_stations in all_planned_running_times[k][date2][train_id][0]:
                                    tprime = all_planned_running_times[k][date2][train_id][0][
                                        running_time_stations]
                                else:
                                    # If the running time between two stations does not exist for a specific train id
                                    # then take the average of the same stations running time but among other train ids
                                    tprime = station_averages_planned_running_times[running_time_stations]
                            else:
                                # If the specific train id does not exist in the data then the planned running time
                                # is the average planned running time of all the stations
                                tprime = all_station_averages_planned_running_times
                            OF[a - 1][b - 1 + (Z * 5)] = tprime
                        else:
                            OF[a - 1][b - 1 + (Z * 5)] = 0
                    else:

                        tprime = 0
                        OF[a - 1][b - 1 + (Z * 5)] = tprime
                b = 0

            a = 0
            b = 0
            # Fill OF matrix with planned dwell time features
            for i in range(I - H + 1, I + 1):
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    if current_P - Z >= 0 and current_I - H >= 0:
                        station = train_line[current_P - Z + b - 1]

                        train_id = list(sorted_trip_id_to_departure_time[date].keys())[current_I - H + a - 1]
                        planned_dwell_time = 0
                        if station in dwell_times_after_allowable[train_id]:
                            planned_dwell_time = dwell_times_after_allowable[train_id][station]
                        else:
                            planned_dwell_time = station_averages_planned_dwell_times[station]
                        wprime = planned_dwell_time
                        wprime = float(round(wprime))  # We are not using decimal numbers
                        OF[a - 1][b - 1 + (Z * 6)] = wprime
                    else:
                        wprime = 0
                        OF[a - 1][b - 1 + (Z * 6)] = wprime

                b = 0

            next_train_time = 0
            current_train_time = 0
            next_train_id = None
            current_train_time = None
            a = 0
            b = 0
            # Fill OF matrix with planned interval between train features
            for i in range(I - H + 1, I + 1):
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    rprime = 0
                    time_diff = 0
                    if current_P - Z >= 0 and current_I - H >= 0:
                        def find_value_by_key(data, key):
                            for item in data:
                                if key in item:
                                    return item[key]
                            return None


                        time_diff = 0
                        station = train_line[current_P - Z + b - 1]
                        result = find_value_by_key(all_planned_interval_between_trains, date2)

                        train_id = list(sorted_trip_id_to_departure_time[date].keys())[current_I - H + a - 1]
                        index = find_trip_id_index(result[station], train_id)
                        if index != -1:

                            if index < len(result[station]) - 1:
                                next_train_id = result[station][index + 1]["trip_id"]
                            else:
                                # Handle the case when next index doesn't exist
                                next_train_id = None

                            if next_train_id is not None:
                                trips = result[station]

                                for trip in trips:
                                    if trip['trip_id'] == train_id:
                                        current_train_time = trip['departure_time']
                                    if trip['trip_id'] == next_train_id:
                                        next_train_time = trip['departure_time']

                                if (next_train_time is None or isinstance(next_train_time, int) or current_train_time is
                                        None or isinstance(current_train_time, int)):
                                    time_diff = float(
                                        math.ceil(station_average_planned_interval_between_trains[station]))
                                else:
                                    time_diff = (next_train_time - current_train_time).total_seconds()
                                rprime = time_diff

                                OF[a - 1][b - 1 + (Z * 7)] = rprime
                    else:
                        rprime = 0
                        OF[a - 1][b - 1 + (Z * 7)] = rprime
                b = 0

            suppliment_time = None
            a = 0
            b = 0
            # Fill OF matrix with running time supplement
            for i in range(I - H + 1, I + 1):
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    if current_P - Z >= 0 and current_I - H >= 0:
                        station = train_line[current_P - Z + b - 1]
                        train_id = list(sorted_trip_id_to_departure_time[date].keys())[current_I - H + a - 1]
                        # The first station does not have running time supplement
                        if station == "Norsborg" or station == "Fruängen":
                            suppliment_time = 0
                        else:
                            # Use average running suppliment time
                            if train_id in all_suppliment_buffer[k][date2]:
                                if station in all_suppliment_buffer[k][date2][train_id]:
                                    suppliment_time = all_suppliment_buffer[k][date2][train_id][station]
                                else:
                                    suppliment_time = station_averages_suppliment_buffer[station]
                            else:
                                suppliment_time = all_station_averages_suppliment_buffer
                        sprime = suppliment_time

                        OF[a - 1][b - 1 + (Z * 8)] = sprime
                    else:
                        sprime = 0
                        OF[a - 1][b - 1 + (Z * 8)] = sprime
                b = 0

            station_length = None
            next_station = None
            a = 0
            b = 0
            # Fill NF matrix with length between two consecutive stations
            for i in range(I - H + 1, I + 1):
                # Iterate over columns
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    if current_P - Z >= 0 and current_I - H >= 0:
                        station = train_line[current_P - Z + b - 1]
                        # Last stations do not have a next stations
                        if station == "Ropsten" or station == "Mörby centrum":
                            next_station = None
                        else:
                            next_station = train_line[current_P - Z + b]

                        if next_station is not None:
                            stations_length_name = station + "-" + next_station
                            if stations_length_name in distances_between_stations:
                                station_length = distances_between_stations[
                                                     stations_length_name] * 1000  # get the distsnce in meter
                                l = station_length
                                NF[a - 1][b - 1] = l
                        else:
                            NF[a - 1][b - 1] = 0
                    else:
                        l = 0
                        NF[a - 1][b - 1] = l
                b = 0

            shared_line = None
            a = 0
            b = 0
            # Fill NF matrix with the number of shared tracks of a specific station features
            for i in range(I - H + 1, I + 1):
                # Iterate over columns
                a += 1
                for j in range(P - Z + 1, P + 1):
                    b += 1
                    if current_P - Z >= 0 and current_I - H >= 0:
                        station = train_line[current_P - Z + b - 1]
                        if station in line_13 and station in line_14:
                            shared_line = 2
                        elif station in line_13 or station in line_14:
                            shared_line = 1
                        else:
                            shared_line = 0
                        m = shared_line
                        NF[a - 1][b - 1] = m

                    else:
                        m = 0
                        NF[a - 1][b - 1] = m
                b = 0

            OF_matrces.append(OF)
            NF_matrces.append(NF)
        current_P = 0
    current_I = 0
# Stack the matrices to create 3D matrix model input
stacked_matrices = np.stack(OF_matrces, axis=0)
stacked_matrices_NF = np.stack(NF_matrces, axis=0)

# Save the formatted data
file_path = "stacked_matrices_OF_Z3_2M.h5"
file_path_NF = "stacked_matrices_NF_Z3_2M.h5"
with h5py.File(file_path, 'w') as f:
    f.create_dataset('stacked_matrices', data=stacked_matrices)
with h5py.File(file_path_NF, 'w') as f:
    f.create_dataset('stacked_matrices_NF', data=stacked_matrices_NF)
