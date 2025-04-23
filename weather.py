import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from datetime import datetime

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Close coordinates of the top 5 biggest solar power plants in Belgium
coordinates = [[51.2307, 5.3135],
               [51.2194, 4.4025],
               [50.5958, 3.8536],
               [51.1896, 3.8078],
               [50.5836, 5.5012]]

# Define the API request parameters
url = "https://archive-api.open-meteo.com/v1/archive"

# Final df
final_df = None

for i, coordinate in enumerate(coordinates):
    latitude = coordinate[0]
    longitude = coordinate[1]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2019-01-01",  # Adjust to your desired date range
        "end_date": "2025-03-31",
        "hourly": ["temperature_2m", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "precipitation_probability", "weather_code"],
        "timezone": "auto"
    }

    # Make the API request
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    hourly_cloud_cover = hourly.Variables(0).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(1).ValuesAsNumpy()
    hourly_cloud_cover_mid = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover_high = hourly.Variables(3).ValuesAsNumpy()
    hourly_temperature_2m = hourly.Variables(4).ValuesAsNumpy()
    hourly_precipitation_probability = hourly.Variables(5).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
      start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
      end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
      freq = pd.Timedelta(seconds = hourly.Interval()),
      inclusive = "left"
    )}

    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
    hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
    hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["precipitation_probability"] = hourly_precipitation_probability
    hourly_data["weather_code"] = hourly_weather_code

    hourly_dataframe = pd.DataFrame(data = hourly_data)

    for col in hourly_dataframe.columns:
        if col != 'date':
            hourly_dataframe = hourly_dataframe.rename(columns={col: f"{col}_{i+1}"})

    if final_df is None:
        final_df = hourly_dataframe
    else:
        final_df = final_df.merge(hourly_dataframe, on='date', how='right')

## SAVING THE DATA
      
# If working in google colab
# from google.colab import drive 
# drive.mount('/content/drive')
# display(final_df)
# final_df.to_csv('/content/drive/My Drive/weather_df_5_coordinates.csv', index=False) # Substitue with actual path

# If working on regular pc
# final_df.to_csv("weather_df_5_coordinates.csv", index=False) # Or prefered path
