import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather_data():
    """
    Fetch weather data from Open-Meteo API for Iceland
    Returns: pandas DataFrame with weather data
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Calculate dates
    end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=35)).strftime('%Y-%m-%d')  # Get last 30 days
    
    # API parameters for Iceland
    params = {
        "latitude": 64.9631,
        "longitude": -19.0208,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "weather_code,temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        daily_data = data.get("daily", {})
        
        if daily_data:
            df = pd.DataFrame(daily_data)
            
            # Rename the time column to date
            df = df.rename(columns={"time": "date"})
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            return df
        else:
            raise ValueError("No daily data found in response")
            
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return None