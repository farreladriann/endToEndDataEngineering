# tests/test_currency.py

import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data.currency import fetch_currency_data

def test_fetch_currency_data():
    """Test currency data fetching function"""
    # Fetch data
    df = fetch_currency_data()
    
    # Check if DataFrame is returned
    assert isinstance(df, pd.DataFrame)
    
    # Check if required columns are present
    assert all(col in df.columns for col in ['date', 'rate'])
    
    # Check if data types are correct
    assert pd.api.types.is_datetime64_dtype(df['date']) or isinstance(df['date'].iloc[0], datetime)
    assert pd.api.types.is_float_dtype(df['rate'])
    
    # Check if data is not empty
    assert not df.empty
    
    # Check if dates are in ascending order
    assert df['date'].is_monotonic_increasing