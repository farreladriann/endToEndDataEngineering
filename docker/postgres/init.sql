-- docker/postgres/init.sql

-- Create currency data table
CREATE TABLE IF NOT EXISTS currency_dataa (
    date DATE PRIMARY KEY,
    rate FLOAT NOT NULL
);

-- Create weather data table
CREATE TABLE IF NOT EXISTS weather_dataa (
    date DATE PRIMARY KEY,
    temperature_2m_mean FLOAT,
    precipitation_sum FLOAT,
    wind_speed_10m_max FLOAT,
    weather_code INTEGER
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS rate_predictions (
    prediction_date DATE PRIMARY KEY,
    predicted_rate FLOAT NOT NULL,
    actual_rate FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create view for model training
CREATE OR REPLACE VIEW training_data AS
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