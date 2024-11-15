import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import datetime, timedelta
import datetime
from realtime_weekend_dependencies import all_actual_dwell_times, weekend, start_date, end_date, \
    both_lines_departure_arrival
from collections import Counter
import math
from tqdm import tqdm
import numpy as np

distances_between_stations = {'Norsborg-Hallunda': 0.5608518352957749, 'Hallunda-Alby': 1.2315301701677526,
                              'Alby-Fittja': 1.4126735335704512, 'Fittja-Masmo': 1.2543678989485225,
                              'Masmo-Vårby gård': 1.8435275914588267, 'Vårby gård-Vårberg': 1.680549008842604,
                              'Vårberg-Skärholmen': 0.9768545535692148, 'Skärholmen-Sätra': 1.2693274997645378,
                              'Sätra-Bredäng': 1.596231549419274, 'Bredäng-Mälarhöjden': 1.45777157183913,
                              'Mälarhöjden-Axelsberg': 1.0812680602269094, 'Axelsberg-Örnsberg': 0.7845559600354761,
                              'Örnsberg-Aspudden': 0.7454608126073872, 'Aspudden-Liljeholmen': 1.3413378607698725,
                              'Liljeholmen-Hornstull': 0.9283123804473377, 'Hornstull-Zinkensdamm': 0.8289455429560384,
                              'Zinkensdamm-Mariatorget': 0.7310200075195858, 'Mariatorget-Slussen': 0.9103131908914186,
                              'Slussen-Gamla stan': 0.5001863954622259, 'Gamla stan-T-Centralen': 1.0837572936539317,
                              'T-Centralen-Östermalmstorg': 1.0563821089045788,
                              'Östermalmstorg-Karlaplan': 1.0157615312474064, 'Karlaplan-Gärdet': 0.7934517494988889,
                              'Gärdet-Ropsten': 1.577821810033754, 'Fruängen-Västertorp': 0.5485091376192011,
                              'Västertorp-Hägerstensåsen': 0.7944159547301006,
                              'Hägerstensåsen-Telefonplan': 1.1328626694305175,
                              'Telefonplan-Midsommarkransen': 0.9475213021057707,
                              'Midsommarkransen-Liljeholmen': 1.1962212823302503,
                              'Liljeholmen-Hornstull': 0.9287739857735806, 'Hornstull-Zinkensdamm': 0.8289478434588036,
                              'Zinkensdamm-Mariatorget': 0.7448139103345772,
                              'Mariatorget-Slussen': 0.9074999076358843, 'Slussen-Gamla stan': 0.5001863954622241,
                              'Gamla stan-T-Centralen': 1.083560977522609,
                              'T-Centralen-Östermalmstorg': 1.06315518660276,
                              'Östermalmstorg-Stadion': 0.8210245125246693,
                              'Stadion-Tekniska högskolan': 1.143955054689398,
                              'Tekniska högskolan-Universitetet': 2.4181849604455667,
                              'Universitetet-Bergshamra': 2.0797581987896177,
                              'Bergshamra-Danderyds sjukhus': 1.186395498170107,
                              'Danderyds sjukhus-Mörby centrum': 0.9321432525935123}


def calculate_departure_arrival_delays(planned_data, actual_data, date):
    """
        Calculate the departure and arrival delays for train trips on a specific date.

        This function compares the planned and actual arrival/departure times of train trips on a given date
        and computes the delays. The results are returned in two dictionaries: one for arrival delays and another
        for departure delays, organized by trip ID and station.

        Parameters:
        planned_data (dict): A dictionary where each key is a trip ID, and the value is a list of planned stations data,
                             including the planned arrival and departure times.
        actual_data (list): A list of dictionaries containing actual recorded data for trips, including actual arrival
                            and departure times, organized by date.
        date (str): The specific date for which to calculate delays, in the format "YYYY-MM-DD".

        Returns:
        tuple: Two dictionaries - one for arrival delays and another for departure delays. Both are organized
               by date, trip ID, and station.
    """

    # Extract the actual data of a specific date
    actual_data_specific_date = []
    target_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    for day_data in actual_data:
        for date_dict in day_data:
            for date_key, trips_list in date_dict.items():
                if date_key == target_date:
                    actual_data_specific_date.append(trips_list)

    # Iterate through actual and planned data for the specific date and calculates the arrival and departure delays
    arrival_delay_dict = {}
    departure_delay_dict = {}
    for trip_data in actual_data_specific_date:
        for trip_id, stations in trip_data[0].items():
            if trip_id in planned_data:
                planned_stations_data = planned_data[trip_id]
                for station in stations:
                    # Initialize nested dictionaries if they don't exist for arrival delay
                    if date not in arrival_delay_dict:
                        arrival_delay_dict[date] = {}
                    if trip_id not in arrival_delay_dict[date]:
                        arrival_delay_dict[date][trip_id] = {}
                    # Initialize nested dictionaries if they don't exist for departure delay
                    if date not in departure_delay_dict:
                        departure_delay_dict[date] = {}
                    if trip_id not in departure_delay_dict[date]:
                        departure_delay_dict[date][trip_id] = {}

                    # Extract the actual arrival and departure time of a specific train at a specific station
                    actual_arrival_time = stations[station]["arrival"]
                    actual_departure_time = stations[station]["departure"]
                    planned_station = next((item for item in planned_stations_data if item['station'] == station), None)

                    if planned_station:
                        threshold = timedelta(hours=6)
                        fixed_time = timedelta(hours=24)
                        delay_departure = 0
                        delay_arrival = 0
                        planned_arrival_time = planned_station['arrival']
                        planned_departure_time = planned_station['departure']
                        arrival_delay = actual_arrival_time - planned_arrival_time

                        # Only calculate the arrival delay for the last stations
                        if station == "Mörby centrum" or station == "Ropsten":
                            departure_delay = datetime.timedelta(seconds=0)
                            delay_departure = departure_delay  # not used

                            delay_arrival = arrival_delay - timedelta(days=arrival_delay.days)

                            # If the arrival delay is greater than 6 hours then it indicates that the hour mark
                            # is greater than 24h which indicates that the time is on the next day
                            if delay_arrival > threshold:
                                delay_arrival = fixed_time - delay_arrival

                        # Only calculate the departure delay for the first stations
                        elif station == "Norsborg" or station == "Fruängen":
                            arrival_delay = datetime.timedelta(seconds=0)
                            delay_arrival = arrival_delay

                            departure_delay = actual_departure_time - planned_departure_time
                            delay_departure = departure_delay - timedelta(days=departure_delay.days)

                            if delay_departure > threshold:
                                delay_departure = fixed_time - delay_departure

                        # Calculate the departure and arrival delays for the other stations
                        else:
                            departure_delay = actual_departure_time - planned_departure_time

                            delay_departure = departure_delay - timedelta(days=departure_delay.days)
                            delay_arrival = arrival_delay - timedelta(days=arrival_delay.days)

                            threshold = timedelta(hours=6)
                            fixed_time = timedelta(hours=24)

                            if delay_departure > threshold:
                                delay_departure = fixed_time - delay_departure

                            if delay_arrival > threshold:
                                delay_arrival = fixed_time - delay_arrival

                        arrival_delay_dict[date][trip_id][station] = delay_arrival
                        departure_delay_dict[date][trip_id][station] = delay_departure

    return arrival_delay_dict, departure_delay_dict


def calculate_planned_running_time(arrival_departure_data, date):
    """
        Calculate the planned running time between stations for each trip on a specific date.

        This function computes the time differences between the departure from one station and the
        arrival at the next station for each trip. It organizes the results by trip ID and stores them
        in a dictionary keyed by the provided date.

        Parameters:
        arrival_departure_data (dict): A dictionary where each key is a trip ID, and the value is a list of
                                       station data, including planned arrival and departure times.
        date (str): The specific date for which to calculate running times, in the format "YYYY-MM-DD".

        Returns:
        dict: A dictionary containing the running times between stations for each trip, organized by date
              and trip ID.
    """
    list = []
    running_time_dict = {}  # Dictionary to store the final running time data organized by date.
    date_dict = {}  # Dictionary to store running time data for each trip on the given date.

    # Iterate over each trip in the input data
    for trip_id, stations in arrival_departure_data.items():
        trip_id_dict = {}  # Dictionary to store running time data for the current trip.
        prev_departure_time = None
        prev_station = None

        # Iterate over each station's data within the trip
        for station_data in stations:
            station = station_data['station']  # Current station name
            arrival_time = station_data['arrival']  # Arrival time at the current station
            departure_time = station_data['departure']  # Departure time from the current station

            # If there's a previous departure time, calculate the running time between stations
            if prev_departure_time is not None:
                time_diff = arrival_time - prev_departure_time  # Calculate the time difference
                time_departure_diff_seconds = time_diff.total_seconds()  # Convert to seconds
                route_path = f"{prev_station}-{station}"  # Create the route path name
                trip_id_dict[route_path] = time_departure_diff_seconds

            prev_departure_time = departure_time
            prev_station = station

        date_dict[trip_id] = [trip_id_dict]

    running_time_dict[date] = date_dict


def calculate_planned_interval_between_trains(both_lines_departure_arrival, date):
    """
    Organize and sort the planned departure times of trains for a specific date.

    This function organizes the departure times of trains by station and sorts them
    in chronological order for each station. It processes the planned departure and
    arrival times for two train lines (e.g., line 13 and line 14) on a specific date.

    Parameters:
    both_lines_departure_arrival (dict): A dictionary where each key is a trip ID, and
                                         the value is a list of dictionaries containing
                                         station, departure, and arrival times.
    date (str): The specific date for which to organize and sort the departure times,
                in the format "YYYY-MM-DD".

    Returns:
    dict: A dictionary that organizes the departure times by station for the specified date,
          with each station's departures sorted chronologically.
    """

    organized_data = {}

    # Iterate through the planned departure and arrival times for each trip
    for trip_id, trips_list in both_lines_departure_arrival.items():
        if date not in organized_data:
            organized_data[date] = {}

        # Process each station's data within the trip
        for trip in trips_list:
            station = trip["station"]
            departure = trip["departure"]

            # Initialize the station's list if it doesn't exist in the dictionary
            if station not in organized_data[date]:
                organized_data[date][station] = []

            # Append the trip ID and departure time to the station's list
            organized_data[date][station].append(
                {'trip_id': trip_id, 'departure_time': departure}
            )

    # Sort the trips for each station by departure time
    for date, stations_data in organized_data.items():
        for station, trips in stations_data.items():
            sorted_trips = sorted(trips, key=lambda x: x['departure_time'])
            organized_data[date][station] = sorted_trips

    return organized_data


def running_time(S1D, S2A, PS, CS, distances):
    """
    Calculate the maximum running time between two stations based on the maximum speed and distance.

    This function calculates the running time required for a train to travel from a previous station (PS)
    to a current station (CS), assuming constant maximum speed and acceleration.

    Parameters:
    S1D (datetime): Departure time from the previous station (not used in this calculation).
    S2A (datetime): Arrival time at the current station (not used in this calculation).
    PS (str): Previous station identifier (e.g., station code or name).
    CS (str): Current station identifier (e.g., station code or name).
    distances (dict): A dictionary containing distances between station pairs, where the key is a
                      string in the format "PS-CS" and the value is the distance in kilometers.

    Returns:
    float: The running time in seconds required to travel between the two stations.
    """

    # Constants
    max_speed = 80 / 3.6  # Convert speed from km/h to m/s (80 km/h)

    # Construct the station pair key and calculate the distance in meters
    station = PS + "-" + CS
    distance_m = distances[station] * 1000  # Convert distance from kilometers to meters

    # Calculate the running time assuming constant maximum speed
    run_time = distance_m / max_speed

    return run_time


def calculate_running_time_supplement(planned_arrival_departure, distances, date):
    """
    Calculate and update the running time supplement for each station in planned trips.

    This function calculates the running timr supplement for each station on a planned trip,
    comparing the planned arrival times with the calculated arrival times based on distance and speed.

    Parameters:
    planned_arrival_departure (dict): A dictionary where keys are trip IDs, and values are lists of station data dictionaries.
                                      Each station data dictionary contains 'station', 'arrival', and 'departure' times.
    distances (dict): A dictionary containing distances between station pairs, where the key is a string in the format "PS-CS",
                      and the value is the distance in kilometers.
    date (str): The date for which the supplement buffer is being calculated.

    Returns:
    dict: A dictionary containing the running time supplement for each station on each trip for the specified date.
    """

    updated_arrival_departure = {}
    time_differences = {}
    running_time_supplement_dict = {}

    # Iterate through each trip in the planned arrival and departure data
    for trip_id, stations_data in planned_arrival_departure.items():
        updated_stations_data = []
        time_differences[trip_id] = {}

        prev_station_data = None
        prev_departure_time = None
        prev_station_name = None

        # Iterate through the stations for the current trip
        for station_data in stations_data:
            station = station_data['station']  # Current station name
            departure_time = station_data['departure']  # Current departure time
            updated_station_data = station_data.copy()
            planned_arrival_time = station_data['arrival']  # Planned arrival time for the current station

            # Calculate and update the arrival time based on the previous station's data
            if prev_station_data and prev_departure_time:
                n = prev_station_name + "-" + station  # Which route path between stations

                if n in distances:
                    # Calculate travel time based on the distance and maximum speed
                    travel_time = running_time(None, None, prev_station_name, station, distances)
                    arrival_time = prev_departure_time + timedelta(seconds=travel_time)
                    arrival_time = arrival_time.replace(microsecond=0)  # Remove microseconds for consistency
                    updated_station_data['arrival'] = arrival_time  # Update the arrival time in the station data

                    # Calculate the time difference between planned and calculated arrival times
                    time_diff = planned_arrival_time - arrival_time
                    time_differences[trip_id][station] = time_diff.total_seconds()

            updated_stations_data.append(updated_station_data)  # Add the updated station data to the list

            # Update variables for the next iteration
            prev_station_name = station
            prev_station_data = station_data
            prev_departure_time = departure_time

        # Store the updated station data for the current trip
        updated_arrival_departure[trip_id] = updated_stations_data

    # Store the time differences for all trips under the specified date
    running_time_supplement_dict[date] = time_differences

    return running_time_supplement_dict


def check_allowable_dwell_time(dwell_times_mean, trip_data, station_distances):
    """
    Check dwell time if allowable and if not adjust it based on the calculated running time between stations.

    This function compares the dwell times at each station with the allowable dwell time, which is the time
    difference between departure from the previous station and the running time. If the current dwell time
    exceeds the allowable dwell time, it is adjusted.

    Parameters:
    dwell_times_mean (dict): A dictionary where keys are trip IDs and values are dictionaries containing the mean dwell
                             time for each station.
    trip_data (dict): A dictionary where keys are trip IDs and values are lists of station data dictionaries.
                      Each station data dictionary contains 'station', 'arrival', and 'departure' times.
    station_distances (dict): A dictionary containing distances between station pairs, where the key is a string
                              in the format "PS-CS" (Previous Station - Current Station), and the value is the distance in kilometers.

    Returns:
    dict: Updated dictionary of dwell times (including allowable check) for each station.
    """

    # Iterate through each trip in the trip_data
    for trip_id, stations in trip_data.items():
        # Iterate through stations in the trip, starting from the second station
        # since the first station does not have dwell time
        for i, station in enumerate(stations):
            if i > 0:
                previous_station = stations[i - 1]["station"]
                current_station = station["station"]

                previous_departure_time = stations[i - 1]["departure"]
                arrival_time = station["arrival"]
                current_depart_time = station["departure"]

                # Check if the station pair exists in station_distances
                if previous_station + "-" + current_station in station_distances:
                    # Calculate running time between the previous station and the current station
                    run_time = running_time(previous_departure_time, arrival_time, previous_station, current_station,
                                            station_distances)

                    # Calculate the total time difference between departure from the previous and the current stations
                    time_diff = current_depart_time - previous_departure_time
                    time_departure_diff_seconds = time_diff.total_seconds()

                    # Calculate the allowable dwell time by subtracting max running time from the total time difference
                    allowable_dwell_time = time_departure_diff_seconds - run_time

                    # Update the dwell time if it exceeds the allowable dwell time
                    if station["station"] in dwell_times_mean[trip_id]:
                        if dwell_times_mean[trip_id][station["station"]] > allowable_dwell_time:
                            dwell_times_mean[trip_id][station["station"]] = allowable_dwell_time

    return dwell_times_mean


def remove_outliers(times):
    """
    Remove outliers from a list of times using the interquartile range (IQR) method.

    This function identifies and removes outliers from a list of numerical values (times) based on the
    interquartile range (IQR). Outliers are defined as values that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

    Parameters:
    times (list): A list of numerical values (times) from which outliers are to be removed.

    Returns:
    list: A filtered list of times with outliers removed.
    """

    times_array = np.array(times)

    Q1 = np.percentile(times_array, 25)  # Calculate the first quartile
    Q3 = np.percentile(times_array, 75)  # Calculate the third quartile
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out values outside the bounds defined by the IQR
    filtered_times = [time for time in times if (time >= lower_bound) and (time <= upper_bound)]

    return filtered_times


def remove_outliers_for_average_times(average_times):
    """
    # Remove outliers for each station in each trip ID for average times
    :param
    average_times (dict): Average actual dwell times
    :return:
    dict: A filtered dictionary of times with outliers removed.
    """
    for trip_id, station_times in average_times.items():
        for station, times in station_times.items():
            average_times[trip_id][station] = remove_outliers(times)
    return average_times


def calculate_planned_dwell_time(actual_dwell_times):
    """
    Calculate the average planned dwell time at each station for each trip ID from the provided actual dwell times.

    This function processes a list of actual dwell times to calculate the average dwell time at each station for each
    trip ID. It first organizes the times by trip ID and station, then removes outliers, and finally computes the
    average dwell time for each station which becomes the planned running time.

    Parameters:
    actual_dwell_times (list): A list of dictionaries containing actual dwell times for different trips at various stations.
                         Each dictionary in the list represents a trip and contains station names as keys and a list
                         of dictionaries with trip IDs and their corresponding dwell times as values.

    Returns:
    dict: A dictionary containing the average dwell times at each station for each trip ID which represents the planned dwell times.
          The structure is {trip_id: {station: average_dwell_time}}.
    """

    average_times = {}
    for trip_data in actual_dwell_times:
        # Iterate through each station and its associated times in the current trip's data
        for station, times_list in trip_data.items():
            # Extract the trip ID from the first dictionary in the times list
            trip_id = list(times_list[0].keys())[0]

            # Extract the dwell time value for this station and trip ID
            time = list(times_list[0].values())[0]

            # If this trip ID is not already in the average_times dictionary, add it
            if trip_id not in average_times:
                average_times[trip_id] = {}

            # If this station is not already in the dictionary for this trip ID, add it
            if station not in average_times[trip_id]:
                average_times[trip_id][station] = []

            # Append the dwell time to the list of times for this station and trip ID
            average_times[trip_id][station].append(time)

    # Remove outliers from the collected times to ensure accurate average calculation
    average_times_after_outliers = remove_outliers_for_average_times(average_times)

    # Initialize a dictionary to store the calculated average dwell times for each trip ID
    average_times_per_trip = {}

    # Iterate through each trip ID and its associated station times after outlier removal
    for trip_id, station_times in average_times_after_outliers.items():
        # Initialize a dictionary to store average dwell times for this trip ID
        average_times_per_trip[trip_id] = {}

        # Iterate through each station and its list of dwell times
        for station, times in station_times.items():
            # Calculate the average dwell time for this station by summing and dividing by the number of times
            average_time = sum(times) / len(times)

            # Store the calculated average dwell time for this station under the current trip ID
            average_times_per_trip[trip_id][station] = average_time

    return average_times_per_trip


def calculate_arrival_and_dwell_times(filtered_departure_times, filtered_stations, filtered_trip_ids, dwell_times):
    """
    Calculate the arrival and dwell times for each trip based on the filtered departure times, stations,
    and trip IDs. The method calculates when each train is expected to arrive at each station and how long
    it will dwell there.

    Parameters:
    filtered_departure_times (list): A list of lists where each inner list contains the departure times
                                     for a specific trip.
    filtered_stations (list): A list of lists where each inner list contains the stations for a specific trip.
    filtered_trip_ids (list): A list of trip IDs corresponding to the trips in filtered_departure_times and
                              filtered_stations.
    dwell_times (dict): A dictionary containing the dwell times for each station and trip ID.

    Returns:
    dict: A dictionary with trip IDs as keys and lists of dictionaries as values. Each inner dictionary
          contains the station name, arrival time, and departure time for that station.
    """

    # Iterate through each trip's departure times, stations, and trip IDs simultaneously
    all_arrival_times = []
    all_arrival_times_dict = {}
    for trip_departure_times, trip_station_list, trip_trip_id in zip(filtered_departure_times, filtered_stations,
                                                                     filtered_trip_ids):

        arrival_times = []
        arrival_times_dict = {}

        # Iterate through each station in the current trip's station list
        for i, station in enumerate(trip_station_list):
            if i == 0 and (station in dwell_times[trip_trip_id]):
                # For the first station, the arrival time is the same as the first departure time
                arrival_times.append(trip_departure_times[0])
                arrival_times_dict[station] = trip_departure_times[0]
            else:
                # For subsequent stations, calculate arrival time based on dwell time
                departure = trip_departure_times[i]
                if station in dwell_times[trip_trip_id]:
                    dwell_time = int(math.ceil(planned_dwell_times[trip_trip_id][station]))
                    arrival_time = departure - timedelta(seconds=dwell_time)
                    arrival_times.append(arrival_time)
                    arrival_times_dict[station] = arrival_time

        # Store arrival times for the current trip
        all_arrival_times.append(arrival_times)
        all_arrival_times_dict[trip_trip_id] = arrival_times_dict

    trip_datas = {}
    # Iterate over each trip ID and corresponding data
    for trip_id, stations, departures, arrivals in zip(filtered_trip_ids, filtered_stations, filtered_departure_times,
                                                       all_arrival_times):

        trip_info = []
        # Iterate over the zipped lists (stations, departures, arrivals) for the current trip
        for station, departure, arrival in zip(stations, departures, arrivals):
            # Create a dictionary for each station's data and append to trip_info list
            if station in all_arrival_times_dict[trip_id]:
                trip = {'station': station, 'arrival': all_arrival_times_dict[trip_id][station], 'departure': departure}
                trip_info.append(trip)

        trip_datas[trip_id] = trip_info

    return trip_datas


stop_id_to_name = {}
folder1 = []
folder2 = []
start_time_hour = 0
end_time_hour = 24

current_date = start_date
all_arrival_delays = []
all_departure_delays = []
all_planned_running_times = []
all_planned_interval_between_trains = []
all_running_time_supplement = []

line_13_route_id = "9011001001300000"
line_14_route_id = "9011001001400000"

# Loop over the dates within the specified range
while current_date <= end_date:
    # Check if the current date is a weekend day (Saturday or Sunday)
    if weekend == True:
        if current_date.weekday() in [5, 6]:  # 5 is Saturday, 6 is Sunday
            # Append the formatted string to the folder lists
            folder1.append(current_date.strftime("%Y-%m-%d"))
    elif weekend == False:
        if current_date.weekday() in [0, 1, 2, 3, 4]:  # 0 is Monday, 4 is Friday
            # Append the formatted string to the folder lists
            folder1.append(current_date.strftime("%Y-%m-%d"))
    else:
        folder1.append(current_date.strftime("%Y-%m-%d"))

    # Move to the next date
    current_date += datetime.timedelta(days=1)

print(folder1)
total_iterations = len(folder2)
i = 0
for f1 in tqdm(folder1, total=total_iterations):
    # Extract and Store the stop id and stop names
    with open("static_data/GTFS-SL-" + f1 + "/stops-r.txt", 'r') as file:
        next(file)  # Skip the header
        for line in file:
            stop_id, stop_name, *_ = line.strip().split(',')
            stop_id_to_name[stop_id] = stop_name

    previous_trip_id = None
    departure_time = []
    departure_times = []
    stop_ids = []
    stop_id = []
    trip_ids = []
    trip_id = []
    add = False

    with open("static_data/GTFS-SL-" + f1 + "/stop_times-r.txt", 'r') as file:
        next(file)  # Skip the header
        for line in file:

            # Extract the planned departure times
            data = line.strip().split(',')
            departure_time_str = data[2]

            # Handel cases when the departure time is e.g. 24:10:00'
            if departure_time_str > '23:59:59':
                # Split the time string into hours, minutes, and seconds
                hours, minutes, seconds = map(int, departure_time_str.split(':'))

                # Adjust hours if it's greater than 23 and subtract it with 24 to get the correct hour
                if hours >= 24:
                    hours -= 24

                # Format the time with leading zero if necessary
                departure_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            if '00:00' <= departure_time_str < '24:00':
                current_trip_id = data[0]
                current_departure_time = datetime.datetime.strptime(departure_time_str, '%H:%M:%S')

                # Check if the trip_id has changed
                if previous_trip_id is None:
                    previous_trip_id = current_trip_id
                    add = True
                    departure_time.append(current_departure_time)
                    stop_id.append(data[3])
                    trip_id.append(data[0])

                # If the trip_id has not changed
                elif current_trip_id != previous_trip_id:
                    if start_time_hour <= current_departure_time.hour < end_time_hour:
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

    planned_dwell_times = calculate_planned_dwell_time(all_actual_dwell_times)

    stations = [[stop_id_to_name[key] for key in sublist if key in stop_id_to_name] for sublist in stop_ids]

    trips_dict = {}
    with open("static_data/GTFS-SL-" + f1 + "/trips-r.txt", "r") as file:
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
        if route_id == line_13_route_id or route_id == line_14_route_id:
            if tripID in planned_dwell_times:
                filtered_stations.append(stations[i])
                filtered_departure_times.append(departure_times[i])
                filtered_trip_ids.append(tripID)

    # Calculate initial arrival and dwell times using the filtered data
    trip_datas = calculate_arrival_and_dwell_times(filtered_departure_times, filtered_stations, filtered_trip_ids, planned_dwell_times)

    # Check and adjust dwell times based on allowable limits and calculate dwell times again
    dwell_times_after_allowable = check_allowable_dwell_time(planned_dwell_times, trip_datas, distances_between_stations)

    # (Optional) Calculate planned running times based on initial trip data (commented out)
    # planned_running_times = calculate_planned_running_time(trip_datas, f1)

    # Recalculate trip data with the adjusted dwell times after checking allowable limits
    recalculated_trip_datas = calculate_arrival_and_dwell_times(filtered_departure_times, filtered_stations,
                                                                filtered_trip_ids, dwell_times_after_allowable)

    # Calculate the planned running times using the recalculated trip data
    planned_running_times = calculate_planned_running_time(recalculated_trip_datas, f1)
    all_planned_running_times.append(planned_running_times)  # Store the planned running times in a list

    # Calculate the arrival and departure delays based on the recalculated trip data
    arrival_delays, departure_delays = calculate_departure_arrival_delays(recalculated_trip_datas, both_lines_departure_arrival, f1)

    # Calculate the running time supplement needed between stations
    running_time_supplement = calculate_running_time_supplement(recalculated_trip_datas, distances_between_stations, f1)
    all_running_time_supplement.append(running_time_supplement)  # Store the running time supplements in a list

    # Store the calculated arrival and departure delays in their respective lists
    all_arrival_delays.append(arrival_delays)
    all_departure_delays.append(departure_delays)

    # Calculate the planned interval between trains based on the recalculated trip data
    planned_interval_between_trains = calculate_planned_interval_between_trains(recalculated_trip_datas, f1)
    all_planned_interval_between_trains.append(planned_interval_between_trains)

