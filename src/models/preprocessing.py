# src/models/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged data from database"""
    logger.info("Starting data cleaning process...")
    
    # Remove missing values
    df = df.dropna()
    
    # Convert date if needed
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])
    
    # Handle outliers for numeric columns
    numeric_cols = ['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max', 'rate']
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        df = df.loc[(df[col] > fence_low) & (df[col] < fence_high)]
    
    logger.info(f"Data cleaning completed. Rows remaining: {len(df)}")
    return df

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Prepare features for modeling"""
    logger.info("Preparing features...")
    
    # Create features
    X = df.drop(['date', 'rate'], axis=1)
    y = df['rate']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    logger.info(f"Feature preparation completed. Features: {', '.join(X.columns)}")
    return X_scaled, y, scaler

def split_data(X: pd.DataFrame, y: pd.Series, train_size: float = 0.8) -> Dict:
    """Split data chronologically"""
    logger.info("Splitting data...")
    
    split_idx = int(len(X) * train_size)
    
    train_data = {
        'X': X[:split_idx],
        'y': y[:split_idx]
    }
    
    test_data = {
        'X': X[split_idx:],
        'y': y[split_idx:]
    }
    
    logger.info(f"Data split completed. Train size: {len(train_data['X'])}, Test size: {len(test_data['X'])}")
    return {'train': train_data, 'test': test_data}