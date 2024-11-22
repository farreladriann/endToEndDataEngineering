from datetime import datetime, timedelta
import yfinance as yf
import requests
import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@task()
def extract_currency_data():
    """
    Fetch USD/ISK currency data from Yahoo Finance.
    """
    current_date = datetime.now().strftime('%Y-%m-%d')
    symbol = "USDISK=X"

    # Fetch data from Yahoo Finance
    try:
        data = yf.download(symbol, start="2020-01-01", end=current_date, interval="1mo")
        if data.empty:
            print("No data fetched. Please check the symbol or date range.")
            return None
        data = data.reset_index()
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')  # Convert Timestamp to string
        print(data)
        return data.to_dict('records')  # Convert DataFrame to JSON-serializable list of dictionaries
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

@task()
def extract_weather_data():
    """Fetch weather data from Open-Meteo API for Iceland"""
    end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 64.9631,
        "longitude": -19.0208,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "weather_code,temperature_2m_mean,precipitation_sum,wind_speed_10m_max,rain_sum",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        daily_data = data.get("daily", {})
        if daily_data:
            df = pd.DataFrame(daily_data)
            print(df)
            return df.to_dict('records')
        else:
            raise ValueError("No daily data found in response")
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return None

@task()
def transform_currency_data(data):
    """Transform currency data by dropping unnecessary columns and renaming."""
    df = pd.DataFrame(data)
    # Drop unnecessary columns
    df = df.drop(['Adj Close', 'High', 'Low', 'Open', 'Volume'], axis=1)
    
    # Rename columns for consistency
    df.columns = ["date", "rate"]

    # Remove rows with NaN in 'date' column
    df = df.dropna(subset=["date"])
    
    # Convert 'date' column to datetime and then to string format
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Filter data for years 2022-2024
    df = df[df['date'].str[:4].astype(int).isin([2022, 2023, 2024])]
    
    # Sort data by date
    return df.sort_values(by='date', ascending=True).to_dict('records')

@task()
def transform_weather_data(data):
    """Transform weather data, drop unnecessary columns, and rename remaining columns."""
    df = pd.DataFrame(data)
    # Convert 'time' column to datetime and then to string format
    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
    
    # Drop columns that are not needed
    df = df.drop([
        'temperature_2m_min', 
        'temperature_2m_max', 
        'precipitation_hours', 
        'wind_gusts_10m_max', 
        'wind_direction_10m_dominant',
        'snowfall_sum',
        'weather_description'
    ], axis=1, errors='ignore')
    
    # Rename columns to match database schema
    df.rename(columns={
        'time': 'date',
        'temperature_2m_mean': 'temperature',
        'precipitation_sum': 'precipitation',
        'wind_speed_10m_max': 'wind_speed',
        'rain_sum': 'rain'
    }, inplace=True)
    
    # Handle outliers for numeric columns
    numeric_columns = ['temperature', 'wind_speed', 'precipitation', 'rain']
    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        df = df.loc[(df[column] > fence_low) & (df[column] < fence_high)]
    
    # Convert all date columns to string
    df['date'] = df['date'].astype(str)
    
    return df.sort_values(by='date', ascending=True).to_dict('records')

@task()
def load_to_postgres(data, table_name):
    """Load transformed data to PostgreSQL database."""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    df = pd.DataFrame(data)

    # Convert DataFrame to list of tuples
    rows = [tuple(row) for row in df.to_numpy()]
    
    # Generate the SQL INSERT statement
    fields = ', '.join(df.columns)
    values_template = ', '.join(['%s'] * len(df.columns))
    insert_query = f"""
        INSERT INTO {table_name} ({fields})
        VALUES ({values_template})
        ON CONFLICT (date) 
        DO UPDATE SET 
        {', '.join([f'{col} = EXCLUDED.{col}' for col in df.columns if col != 'date'])};
    """

    try:
        # Execute the query for each record
        with pg_hook.get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_query, rows)
            conn.commit()
    except Exception as e:
        print(f"Error inserting data into table {table_name}: {e}")
        raise


with DAG(
    'currency_weather_pipeline',
    default_args=default_args,
    description='ETL pipeline for currency and weather data',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    # Create tables - schema specified in SQL
    create_currency_table = PostgresOperator(
        task_id='create_currency_table',
        postgres_conn_id='postgres_default',
        sql="""
            CREATE TABLE IF NOT EXISTS public.currency_data (
                date TIMESTAMP PRIMARY KEY,
                rate FLOAT
            );

            CREATE INDEX IF NOT EXISTS idx_currency_date ON public.currency_data(date);
        """
    )

    create_weather_table = PostgresOperator(
        task_id='create_weather_table',
        postgres_conn_id='postgres_default',
        sql="""
            CREATE TABLE IF NOT EXISTS public.weather_data (
                date TIMESTAMP PRIMARY KEY,
                weather_code INTEGER,
                temperature FLOAT,
                precipitation FLOAT,
                wind_speed FLOAT,
                rain FLOAT
            );

            CREATE INDEX IF NOT EXISTS idx_weather_date ON public.weather_data(date);
        """
    )

    # Extract tasks
    currency_data = extract_currency_data()
    weather_data = extract_weather_data()

    # Transform tasks
    transformed_currency = transform_currency_data(currency_data)
    transformed_weather = transform_weather_data(weather_data)

    # Load tasks
    load_currency = load_to_postgres(transformed_currency, 'currency_data')
    load_weather = load_to_postgres(transformed_weather, 'weather_data')

    # Define the task dependencies
    create_currency_table >> currency_data
    create_weather_table >> weather_data

    currency_data >> transformed_currency >> load_currency
    weather_data >> transformed_weather >> load_weather

