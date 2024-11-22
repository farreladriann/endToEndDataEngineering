import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_currency_data():
    """
    Fetch USD/ISK currency data from Yahoo Finance
    Returns: pandas DataFrame with date and rate columns
    """
    # Get current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download data
    symbol = "USDISK=X"
    data = yf.download(symbol, start="2020-01-01", end=current_date, interval="1mo")
    
    # Process data
    data = data.reset_index()
    data = data[['Date', 'Close']]  # We only need the closing price
    data.columns = ['date', 'rate']
    
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date']).dt.date
    
    return data