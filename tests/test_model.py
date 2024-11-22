# tests/test_model.py

import pytest
import pandas as pd
import numpy as np
from src.models.preprocessing import clean_data, create_features, prepare_data
from src.models.train import train_model, predict_rate

def create_sample_data():
    """Create sample data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date': dates,
        'rate': np.random.normal(140, 5, len(dates)),
        'temperature_2m_mean': np.random.normal(10, 5, len(dates)),
        'precipitation_sum': np.random.exponential(2, len(dates)),
        'wind_speed_10m_max': np.random.normal(8, 2, len(dates)),
        'weather_code': np.random.randint(0, 99, len(dates))
    })
    
    return df

def test_data_preprocessing():
    """Test data preprocessing functions"""
    # Create sample data
    df = create_sample_data()
    
    # Test clean_data
    cleaned_df = clean_data(df)
    assert not cleaned_df.isna().any().any()
    assert cleaned_df['date'].is_monotonic_increasing
    
    # Test create_features
    featured_df = create_features(cleaned_df)
    assert 'month' in featured_df.columns
    assert 'day_of_week' in featured_df.columns
    assert 'rate_lag1' in featured_df.columns
    assert 'rate_rolling_mean_7' in featured_df.columns
    
    # Test prepare_data
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert len(X_train) + len(X_test) == len(df) - 30  # Account for rolling features
    assert all(col in feature_cols for col in ['temperature_2m_mean', 'precipitation_sum'])

def test_model_training():
    """Test model training and prediction"""
    # Create sample data
    df = create_sample_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)
    
    # Train model
    model, train_score, test_score = train_model(X_train, X_test, y_train, y_test)
    
    # Check if scores are reasonable
    assert 0 <= train_score <= 1
    assert 0 <= test_score <= 1
    
    # Test prediction
    predictions = predict_rate(X_test, model)
    assert len(predictions) == len(X_test)
    assert isinstance(predictions, np.ndarray)