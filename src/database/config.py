# src/database/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': "aws-0-ap-southeast-1.pooler.supabase.com",
    'database': "postgres",
    'user': "postgres.nvssrkzspnbjzmhclcyf",
    'password': "kambing12345677778"
}

# SQL Queries
CREATE_CURRENCY_TABLE = """
CREATE TABLE IF NOT EXISTS currency_data (
    date DATE PRIMARY KEY,
    rate FLOAT NOT NULL
);
"""

CREATE_WEATHER_TABLE = """
CREATE TABLE IF NOT EXISTS weather_data (
    date DATE PRIMARY KEY,
    temperature_2m_mean FLOAT,
    precipitation_sum FLOAT,
    wind_speed_10m_max FLOAT,
    weather_code INTEGER
);
"""

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS rate_predictions (
    prediction_date DATE PRIMARY KEY,
    predicted_rate FLOAT NOT NULL,
    actual_rate FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Query to get merged data for model training
GET_TRAINING_DATA = """
SELECT 
    c.date,
    c.rate,
    w.temperature_2m_mean,
    w.precipitation_sum,
    w.wind_speed_10m_max,
    w.weather_code
FROM currency_data c
LEFT JOIN weather_data w ON c.date = w.date
ORDER BY c.date;
"""