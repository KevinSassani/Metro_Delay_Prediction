import os
import datetime


start_date = datetime.date(2023, 2, 28)
end_date = datetime.date(2023, 2, 28)

folder = []
sl_lines = ["13", "14"]

# Loop over the dates within the specified range
current_date = start_date
while current_date <= end_date:
    # Append the formatted string to the folder list
    folder.append("GTFS-SL-" + current_date.strftime("%Y-%m-%d"))
    # Move to the next date
    current_date += datetime.timedelta(days=1)

# Print the contents of the folder list

for f in folder:
    files_to_filter = ["static_data/"+f+'/routes', "static_data/"+f+"/trips", "static_data/"+f+'/stop_times', "static_data/"+f+'/stops']

    with open("static_data/"+f+"/routes.txt", 'r') as f_in, open(files_to_filter[0]+"-r.txt", 'w') as f_out:
        # Read the first line and write it to the output file
        first_line = f_in.readline()
        f_out.write(first_line)

        # Read and process each subsequent line from the input file
        for line in f_in:
            # Split the line by commas
            parts = line.split(',')
            # Check if the 5th element equals
            if parts[4].strip() == "401" and parts[2].strip() in sl_lines:
                # If it does, write the line to the output file
                f_out.write(line)

    if os.path.exists(files_to_filter[0]+".txt"):
        # Remove the file
        os.remove(files_to_filter[0]+".txt")
        print(f"File '{files_to_filter[0] + ".txt"}' removed")
    else:
        print(f"File '{files_to_filter[0]+".txt"}' does not exist.")


    route_ids = []

    # Open the file
    with open(files_to_filter[0]+"-r.txt", 'r') as file:
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



    with open("static_data/"+f+"/trips.txt", 'r') as f_in, open(files_to_filter[1]+"-r.txt", 'w') as f_out:
        first_line = f_in.readline()
        f_out.write(first_line)
        trip_ids = []

        for line in f_in:
            # Split the line by commas
            parts = line.strip().split(',')
            # Check if the route_id is "9011001004300000" and direction_id is "1"
            if parts[0] in route_ids and parts[4] == "1":
                # If it does, write the line to the output file
                f_out.write(line)
                trip_ids.append(parts[2])
    if os.path.exists(files_to_filter[1]+".txt"):
        # Remove the file
        os.remove(files_to_filter[1]+".txt")
        print(f"File '{files_to_filter[1] + ".txt"}' removed")
    else:
        print(f"File '{files_to_filter[1]+".txt"}' does not exist.")


    with open("static_data/"+f+"/stop_times.txt", 'r') as f_in, open(files_to_filter[2]+"-r.txt", 'w') as f_out:
        first_line = f_in.readline()
        f_out.write(first_line)
        stop_ids = []

        for line in f_in:
            # Split the line by commas
            parts = line.strip().split(',')
            # Check if the route_id is "9011001004300000" and direction_id is "1"
            if parts[0] in trip_ids:
                # If it does, write the line to the output file
                f_out.write(line)
                stop_ids.append(parts[3])

    if os.path.exists(files_to_filter[2]+".txt"):
        # Remove the file
        os.remove(files_to_filter[2]+".txt")
        print(f"File '{files_to_filter[2] + ".txt"}' removed")
    else:
        print(f"File '{files_to_filter[2]+".txt"}' does not exist.")

    with open("static_data/"+f+"/stops.txt", 'r') as f_in, open(files_to_filter[3]+"-r.txt", 'w') as f_out:
        first_line = f_in.readline()
        f_out.write(first_line)

        for line in f_in:
            # Split the line by commas
            parts = line.strip().split(',')
            # Check if the route_id is "9011001004300000" and direction_id is "1"
            if parts[0] in stop_ids:
                # If it does, write the line to the output file
                f_out.write(line)

    if os.path.exists(files_to_filter[3]+".txt"):
        # Remove the file
        os.remove(files_to_filter[3]+".txt")
        print(f"File '{files_to_filter[3] + ".txt"}' removed")
    else:
        print(f"File '{files_to_filter[3]+".txt"}' does not exist.")



