"""
This script converts the vehicle position data in protobuf format into JSON format
also removes unnecessary files to save storage space
"""

import gtfs_realtime_pb2
from google.protobuf.json_format import MessageToJson
import json
import datetime
import os

feed = gtfs_realtime_pb2.FeedMessage()

start_date = datetime.date(2023, 2, 5)  # Defines which date to start the process
end_date = datetime.date(2023, 2, 5) # Defines which date to end the process
dwell_times_list = []
folder1 = []
folder2 = []
trip_ids = []
route_ids = []

# The date defined in the files is formatted in two different ways, this is to handle this issue
current_date = start_date
while current_date <= end_date:
    # Append the formatted string to the folder list
    folder1.append(current_date.strftime("%Y-%m-%d")) # Used for static data
    folder2.append(current_date.strftime("%Y/%m/%d")) # Used for realtime data
    # Move to the next date
    current_date += datetime.timedelta(days=1)

for f1, f2 in zip(folder1, folder2):
    # Open the file
    with open("static_data/GTFS-SL-" + f1 + "/routes-r.txt", 'r') as file:
        # Skip the first line (header)
        next(file)
        # Iterate over each line in the file
        for line in file:
            # Split the line by commas
            parts = line.strip().split(',')
            # Extract the route_id (first element)
            route_id = parts[0].strip()
            # Append the route_id to the list
            route_ids.append(route_id)

    with open("static_data/GTFS-SL-" + f1 + "/trips-r.txt", 'r') as file:
        for line in file:
            # Split the line by commas
            parts = line.strip().split(',')
            # Check if the route_id and direction_id is correct
            if parts[0] in route_ids:
                # If it does, write the line to the output file
                trip_ids.append(parts[2])
    print(trip_ids)

    # Assuming 'TripUpdates.pb' contains the serialized GTFS-realtime data
    main_folder = "realtime_vehicle_data/sl-" + f1 + "/VehiclePositions/" + f2
    folder_names = sorted(os.listdir(main_folder))
    print(folder_names)
    end_index = folder_names.index('16')
    start_index = folder_names.index('00')
    print(f2)

    # Iterate through each folder starting from index 'start_index'
    for folder_name in folder_names[start_index:end_index + 1]:
        # for folder_name in "1":
        # folder_name = "23"
        folder_path = os.path.join(main_folder, folder_name)
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        # Iterate through each file in the folder
        for file_name in sorted(os.listdir(folder_path)):
            print(file_name)
            # Skip if not a .pb file
            if not file_name.endswith('.pb'):
                continue

            # Read the Protocol Buffer file
            with open(
                    "realtime_vehicle_data/sl-" + f1 + "/VehiclePositions/" + f2 + "/" + folder_name + "/" + file_name,
                    'rb') as file:
                feed.ParseFromString(file.read())

            train_data = []

            # Save to JSON file
            for train_entity in feed.entity:
                if train_entity.vehicle.trip.trip_id in trip_ids:
                    train_json = MessageToJson(train_entity)
                    train_data.append(json.loads(train_json))

            file_path = os.path.join(folder_path, file_name)
            # Construct the JSON file path
            json_file_path = os.path.splitext(file_path)[0] + '.json'

            # Writing the list as a single JSON object
            with open(json_file_path, 'w') as json_file:
                json.dump(train_data, json_file)

            print("Data saved to ", json_file_path)

    # Just for test purposes
    # Find the index of folder '15' in the sorted list
    start_index = folder_names.index('00')
    end_index = folder_names.index('16')

    # Loop through folder names starting from '15' and remove pb files
    for folder_name in folder_names[start_index:end_index + 1]:
        # for folder_name in folder_names[start_index]:
        folder_path = os.path.join(main_folder, folder_name)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pb"):
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"Removed: {file_path}")

    """
    # Read the JSON file
    trip_id_to_find = "14010000649252264"
    trip_data = []
    with open('vehicle_data2.json', 'r') as json_file:
        for line in json_file:
            # Parse each line as JSON
            train_entities = json.loads(line)
            # Iterate over each entity in the list
            for train_entity in train_entities:
                # Accessing the correct keys at each level
                trip_id = train_entity.get("vehicle", {}).get("trip", {}).get("tripId", "")
                # Convert trip_id to string before comparison
                if str(trip_id) == trip_id_to_find:
                    trip_data.append(train_entity)
                    break
    
    
    """

    """
    #Remove json files
    main_folder = "realtime_vehicle_data/sl-"+f1+"/VehiclePositions/2023/01/04"
    folder_names = sorted(os.listdir(main_folder))
    print(folder_names)
    # Find the index of folder '15' in the sorted list
    start_index = folder_names.index('00')
    
    # Loop through folder names starting from '15' and remove JSON files
    for folder_name in folder_names[start_index:]:
        folder_path = os.path.join(main_folder, folder_name)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"Removed: {file_path}")
    """
