# tests/test_weather.py

import pytest
import pandas as pd
from datetime import datetime
from src.data.weather import fetch_weather_data

def test_fetch_weather_data():
    """Test weather data fetching function"""
    # Fetch data
    df = fetch_weather_data()
    
    # Check if DataFrame is returned
    assert isinstance(df, pd.DataFrame)
    
    # Check if required columns are present
    required_columns = [
        'date', 'temperature_2m_mean', 'precipitation_sum',
        'wind_speed_10m_max', 'weather_code'
    ]
    assert all(col in df.columns for col in required_columns)
    
    # Check if data types are correct
    assert pd.api.types.is_datetime64_dtype(df['date']) or isinstance(df['date'].iloc[0], datetime)
    assert pd.api.types.is_float_dtype(df['temperature_2m_mean'])
    assert pd.api.types.is_float_dtype(df['precipitation_sum'])
    assert pd.api.types.is_float_dtype(df['wind_speed_10m_max'])
    assert pd.api.types.is_integer_dtype(df['weather_code'])
    
    # Check if data is not empty
    assert not df.empty
    
    # Check if dates are in ascending order
    assert df['date'].is_monotonic_increasing
    
    # Check if weather codes are in valid range
    assert df['weather_code'].between(0, 99).all()