import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pysolar.solar import get_altitude, get_azimuth

def get_sun_angles_for_day(latitude, longitude, date=None, local_timezone=None, interval_minutes=15):
    """
    Calculate sun angles for an entire day at specified intervals using Pysolar.
    
    Parameters:
    -----------
    latitude : float
        Latitude in degrees (positive for North, negative for South)
    longitude : float
        Longitude in degrees (positive for East, negative for West)
    date : datetime, optional
        Base date for calculations (time component will be ignored)
    local_timezone : str, optional
        Timezone name (e.g., 'America/New_York')
    interval_minutes : int, optional
        Interval between calculations in minutes. Default is 15.
        
    Returns:
    --------
    times : list
        List of datetime objects
    altitudes : numpy array
        Sun altitude angles in degrees
    azimuths : numpy array
        Sun azimuth angles in degrees
    """
    # Set up timezone
    if local_timezone:
        tz = pytz.timezone(local_timezone)
    else:
        tz = pytz.UTC
    
    # If no date provided, use today
    if date is None:
        date = datetime.now(tz)
    elif not date.tzinfo:
        date = tz.localize(date)
    
    # Strip time component to get just the date
    date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Calculate number of intervals in a day
    intervals_per_day = 24 * 60 // interval_minutes
    
    # Create time points for the entire day
    times = [date + timedelta(minutes=i * interval_minutes) for i in range(intervals_per_day)]
    
    # Convert times to UTC for Pysolar calculations
    utc_times = [t.astimezone(pytz.UTC) for t in times]
    
    # Calculate sun position for each time point
    altitudes = []
    azimuths = []
    
    for t in utc_times:
        # Pysolar expects UTC time
        alt = get_altitude(latitude, longitude, t)
        az = get_azimuth(latitude, longitude, t)
        altitudes.append(alt)
        azimuths.append(az)
    
    return times, np.array(altitudes), np.array(azimuths)

def main():
    # We are choosing center of Belgium as location
    latitude = 50.6403
    longitude = 4.6667
    timezone_str = 'Europe/Madrid'  # Same timezone as Belgium

    tempos, altituds, azmuths = [], [], []
    
    # From date_1 to date_2
    date_1 = datetime(2019, 1, 1)
    date_2 = datetime(2025, 3, 31)

    d_day = date_1
    while d_day <= date_2:
        # Calculate sun angles for the day
        times, altitudes, azimuths = get_sun_angles_for_day(
            latitude, longitude, date=d_day, local_timezone=timezone_str)
        d_day += timedelta(days=1)

        tempos.extend(times)
        altituds.extend(altitudes)
        azmuths.extend(azimuths)
    
    # Create a pandas DataFrame with the results
    results = pd.DataFrame({
        'Time': tempos,
        'Altitude (degrees)': altituds,
        'Azimuths': azmuths
    })
    
    # Return the vectors for altitude and azimuth
    return results

if __name__ == "__main__":
    data = main()
    x = data
    x['Altitude (degrees)'] = [0 if alt < 0 else alt for alt in data['Altitude (degrees)']]
    x.head(-20)
    x.to_csv('Solar_angle.csv')
