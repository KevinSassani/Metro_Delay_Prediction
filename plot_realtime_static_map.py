"""
This script plots the plots from "plot_realtime_map" and "plot_map" but also a diagram over actual and planned arrival,
departure and dwell times at each station between a time interval.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from plot_realtime_map import all_station_names, all_timestamps
from plot_map import filtered_stations, filtered_departure_times
import datetime

# Define the desired year (e.g., 2023)
desired_year = 2023

# Define the time range boundaries
start_time = datetime.time(7, 10, 0)
end_time = datetime.time(10, 0, 0)

indices = []
filtered_timestamps = []
for index, timestamps in enumerate(all_timestamps):
    if start_time <= timestamps[0].time() <= end_time:
        filtered_timestamps.append(timestamps)
        indices.append(index)

filtered_station_names = [all_station_names[i] for i in indices]

# Function to change the year of datetime objects
def change_year(dt, year):
    return dt.replace(year=year)

# Change the year for each datetime object in filtered_departure_times
filtered_departure_times_updated = [[change_year(dt, desired_year) for dt in trip] for trip in filtered_departure_times]

plt.figure(figsize=(12, 6))

# Plotting trip data from plot_map
for station_list, departure_list in zip(filtered_stations, filtered_departure_times_updated):
    station_names = [station_list[i] for i in range(len(station_list))]
    plt.plot(departure_list, station_names, linestyle='-', color='k')

# Plotting each trip separately from plot_realtime_map
for station_names, datetime_timestamps in zip(filtered_station_names, filtered_timestamps):
    plt.plot(datetime_timestamps, station_names)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('Time')
plt.ylabel('Station')
plt.grid(True)
plt.tight_layout()
plt.show()