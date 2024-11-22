# src/database/connection.py

import os
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    return psycopg2.connect(
        host="aws-0-ap-southeast-1.pooler.supabase.com",
        database="postgres",
        user="postgres.nvssrkzspnbjzmhclcyf",
        password="kambing12345677778"
    )

def insert_currency_data(df):
    """Insert currency data into the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS currency_data (
            date DATE PRIMARY KEY,
            rate FLOAT
        )
    """)
    
    # Convert DataFrame to list of tuples
    values = [tuple(x) for x in df[['date', 'rate']].values]
    
    # Insert data
    execute_values(
        cur,
        "INSERT INTO currency_data (date, rate) VALUES %s ON CONFLICT (date) DO UPDATE SET rate = EXCLUDED.rate",
        values
    )
    
    conn.commit()
    cur.close()
    conn.close()

def insert_weather_data(df):
    """Insert weather data into the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            date DATE PRIMARY KEY,
            temperature_2m_mean FLOAT,
            precipitation_sum FLOAT,
            wind_speed_10m_max FLOAT,
            weather_code INTEGER
        )
    """)
    
    # Convert DataFrame to list of tuples
    values = [tuple(x) for x in df[['date', 'temperature_2m_mean', 'precipitation_sum', 
                                  'wind_speed_10m_max', 'weather_code']].values]
    
    # Insert data
    execute_values(
        cur,
        """
        INSERT INTO weather_data 
        (date, temperature_2m_mean, precipitation_sum, wind_speed_10m_max, weather_code) 
        VALUES %s 
        ON CONFLICT (date) DO UPDATE SET 
            temperature_2m_mean = EXCLUDED.temperature_2m_mean,
            precipitation_sum = EXCLUDED.precipitation_sum,
            wind_speed_10m_max = EXCLUDED.wind_speed_10m_max,
            weather_code = EXCLUDED.weather_code
        """,
        values
    )
    
    conn.commit()
    cur.close()
    conn.close()

def insert_predictions(dates, actual_values, predicted_values, model_name):
    """Insert model predictions into the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rate_predictions (
            prediction_date DATE PRIMARY KEY,
            predicted_rate FLOAT NOT NULL,
            actual_rate FLOAT,
            model_name VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Prepare data for insertion
    values = []
    for date, actual, pred in zip(dates, actual_values, predicted_values):
        values.append((date, pred, actual, model_name))
    
    # Insert predictions
    execute_values(
        cur,
        """
        INSERT INTO rate_predictions 
        (prediction_date, predicted_rate, actual_rate, model_name)
        VALUES %s
        ON CONFLICT (prediction_date) DO UPDATE SET 
            predicted_rate = EXCLUDED.predicted_rate,
            actual_rate = EXCLUDED.actual_rate,
            model_name = EXCLUDED.model_name,
            created_at = CURRENT_TIMESTAMP
        """,
        values
    )
    
    conn.commit()
    cur.close()
    conn.close()

def get_merged_data():
    """Get merged data from database for model training"""
    conn = get_db_connection()
    
    query = """
        SELECT 
            w.date,
            c.rate,
            w.temperature_2m_mean,
            w.precipitation_sum,
            w.wind_speed_10m_max,
            w.weather_code
        FROM weather_data w
        LEFT JOIN (
            SELECT
                DATE_TRUNC('month', date) AS month,
                AVG(rate) AS rate
            FROM currency_data
            GROUP BY month
        ) c
        ON DATE_TRUNC('month', w.date) = c.month
        ORDER BY w.date
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df