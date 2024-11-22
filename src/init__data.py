from data.currency import fetch_currency_data
from data.weather import fetch_weather_data
from database.connection import insert_currency_data, insert_weather_data, get_merged_data, insert_training_data
from models.preprocessing import prepare_data

def main():
    """Main function to initialize and preprocess data"""
    print("Starting data initialization...")
    
    # Fetch data
    currency_data = fetch_currency_data()
    weather_data = fetch_weather_data()
    
    # Insert into database
    if currency_data is not None and weather_data is not None:
        insert_currency_data(currency_data)
        insert_weather_data(weather_data)
    else:
        print("Failed to fetch data, database insertion skipped")
        return
    
    print("Data fetched and inserted successfully.")
    
    # Process and store data
    print("Starting data preprocessing...")
    df = get_merged_data()
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)
    
    # Combine X_train and y_train for storage
    training_data = X_train.copy()
    training_data['date'] = df['date']
    training_data['rate'] = y_train
    training_data['model_version'] = 'v1.0'  # Example model version
    
    insert_training_data(training_data)
    print("Data preprocessing and storage completed.")

if __name__ == "__main__":
    main()