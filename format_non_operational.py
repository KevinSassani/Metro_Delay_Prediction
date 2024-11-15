"""
This script runs and gets realtime and planned data from the scripts realtime_weekend_dependencies and
planned_weekend_dependencies and saves it in a .pkl file
"""

import pickle

from realtime_weekend_dependencies import (
    all_actual_dwell_times_dict, actual_interval_between_trains,
    actual_running_times, trip_id_list_check
)
from planned_weekend_dependencies import (
    all_planned_running_times, all_arrival_delays, all_departure_delays,
    all_planned_interval_between_trains, dwell_times_after_allowable,
    all_suppliment_buffer, distances_between_stations
)

# Create a dictionary to store all variables
data_to_save = {
    'all_actual_dwell_times_dict': all_actual_dwell_times_dict,
    'actual_interval_between_trains': actual_interval_between_trains,
    'actual_running_times': actual_running_times,
    'trip_id_list_check': trip_id_list_check,
    'all_planned_running_times': all_planned_running_times,
    'all_arrival_delays': all_arrival_delays,
    'all_departure_delays': all_departure_delays,
    'all_planned_interval_between_trains': all_planned_interval_between_trains,
    'dwell_times_after_allowable': dwell_times_after_allowable,
    'all_suppliment_buffer': all_suppliment_buffer,
    'distances_between_stations': distances_between_stations
}

# Save the dictionary to a .pkl file
with open('Junk/saved_realtime_planned_data_1D_test.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

print("Data has been saved")
