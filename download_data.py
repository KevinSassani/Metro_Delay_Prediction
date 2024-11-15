"""
This script downloads the actual and static data from the API provided by Trafiklab
"""

import requests
import datetime
import os
import webbrowser

def download_gtfs_realtime_data(operator, feed, start_date, end_date, api_key, static):
    if static:
        base_url = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/{}?date={}&key={}"  # Static
    else:
        base_url = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/{}/{}?date={}&key={}" #Real-time

    start_date_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    dates = generate_dates(start_date_dt, end_date_dt)

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        if static:
            api_url = base_url.format(operator, date_str, api_key)
        else:
            api_url = base_url.format(operator, feed, date_str, api_key)
        print("Testing:", api_url)

        # Make API request
        response = requests.get(api_url)

        if response.status_code == 200:
            print(f"Data downloaded and saved for date: {date_str}")
            webbrowser.open(api_url)
        else:
            print("Error downloading data for date:", date_str)


def generate_dates(start_date, end_date):
    # Generate list of dates between start_date and end_date
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += datetime.timedelta(days=1)
    return dates


# Configuration what to be downloaded
start_date = "2023-02-05"
end_date = "2023-02-05"
operator = "sl"
#feed = "TripUpdates"
feed = "VehiclePositions"
#feed = "ServiceAlerts"
api_key = ""
static = False

download_gtfs_realtime_data(operator, feed, start_date, end_date, api_key, static)


#https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/sl/TripUpdates/?date=2024-03-07&key=AecIuOCU0vxr4sGZAo2Yj7sEBzNFn7yRfX8tPASIcjI
#https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/sl/TripUpdates/?date=2021-06-01&key=TL9nJNMW84GOdYHQSBco8YGJ6t20EaaR4uoKaHvbQU0
#https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/sl/TripUpdates?date={date}&key={api_key}