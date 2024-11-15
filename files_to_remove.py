import datetime
import os

start_date = datetime.date(2023, 2, 28)
end_date = datetime.date(2023, 2, 28)

folder = []

# Loop over the dates within the specified range
current_date = start_date
while current_date <= end_date:
    # Append the formatted string to the folder list
    folder.append("GTFS-SL-" + current_date.strftime("%Y-%m-%d"))
    # Move to the next date
    current_date += datetime.timedelta(days=1)

# Print the contents of the folder list

for f in folder:
    files_to_remove = ["static_data/"+f+'/agency.txt', "static_data/"+f+'/attributions.txt', "static_data/"+f+'/calendar.txt', "static_data/"+f+"/calendar_dates.txt",
                       "static_data/"+f+"/feed_info.txt", "static_data/"+f+"/shapes.txt", "static_data/"+f+"/transfers.txt"]

    for filename in files_to_remove:
        # Check if the file exists before attempting to remove it
        if os.path.exists(filename):
            # Remove the file
            os.remove(filename)
            print(f"File '{filename}' has been removed.")
        else:
            print(f"File '{filename}' does not exist.")

