pip install pandas matplotlib seaborn numpy scikit-learn xgboost pickle-mixin optuna tqdm


import yfinance as yf
import pandas as pd
import numpy as np
import requests
import csv
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import psycopg2
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle
import optuna
import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Get the current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Download the historical exchange rate data for USD/ISK from current date to 2020
symbol = "USDISK=X"  # USD to Icelandic Krona
data = yf.download(symbol, start="2020-01-01", end=current_date, interval="1mo")

# Filter the data to get the first day of each month
data_first_day_of_month = data.resample('MS').first()  # 'MS' stands for Month Start

# Save to CSV with the name 'currency.csv'
data_first_day_of_month.to_csv('currency.csv')


data1 = pd.read_csv('currency.csv')
data1.info()

data1


data1['Price'].value_counts()

data1['Adj Close'].value_counts()

data1['Close'].value_counts()

data1['High'].value_counts()

data1['Low'].value_counts()

data1['Open'].value_counts()

data1['Volume'].value_counts()

data1 = data1.drop(['Adj Close', 'High', 'Low', 'Open', 'Volume'], axis = 1)


data1.columns = ["date", "rate"]


# Hapus baris yang mengandung nilai tertentu di kolom 'date' atau 'rate'
data1 = data1[~data1["date"].astype(str).str.contains("Ticker|Date", na=False)]
data1 = data1.dropna(subset=["date"])  # Menghapus baris dengan NaN di kolom 'date'


data1.info()

data1["date"] = pd.to_datetime(data1["date"]).dt.strftime('%Y-%m-%d')


data1['date'] = pd.to_datetime(data1['date'])

data1 = data1[data1['date'].dt.year.isin([2022, 2023, 2024])]

data1 = data1.sort_values(by='date', ascending=True, ignore_index=True)

data1

url = "https://archive-api.open-meteo.com/v1/archive"

# Menghitung end_date sesuai dengan permintaan
end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

# Mengatur start_date menjadi 1 Januari 2022
start_date = datetime(2022, 1, 1).strftime('%Y-%m-%d')

# Mengupdate parameter URL
params = {
    "latitude": 64.9631,
    "longitude": -19.0208,
    "start_date": start_date,
    "end_date": end_date,
    "daily": "weather_code,temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant",
    "timezone": "auto"
}

start_date, end_date, params


# WMO Weather Code Descriptions
weather_code_descriptions = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Drizzle: Light",
    53: "Drizzle: Moderate",
    55: "Drizzle: Dense intensity",
    56: "Freezing Drizzle: Light",
    57: "Freezing Drizzle: Dense intensity",
    61: "Rain: Slight",
    63: "Rain: Moderate",
    65: "Rain: Heavy intensity",
    66: "Freezing Rain: Light",
    67: "Freezing Rain: Heavy",
    71: "Snow fall: Slight",
    73: "Snow fall: Moderate",
    75: "Snow fall: Heavy",
    77: "Snow grains",
    80: "Rain showers: Slight",
    81: "Rain showers: Moderate",
    82: "Rain showers: Violent",
    85: "Snow showers: Slight",
    86: "Snow showers: Heavy",
    95: "Thunderstorm: Slight or moderate",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# Mengirim permintaan GET ke API
response = requests.get(url, params=params)

# Memeriksa status permintaan
if response.status_code == 200:
    data = response.json()

    # Ambil data daily
    daily_data = data.get("daily", {})

    if daily_data:
        # Convert to DataFrame
        df = pd.DataFrame(daily_data)

        # Add weather code descriptions
        df["weather_description"] = df["weather_code"].map(weather_code_descriptions)

        # Save to CSV
        df.to_csv("weather.csv", index=False)
        print("Data has been saved to weather.csv")
    else:
        print("No daily data found in response.")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")

data2 = pd.read_csv('weather.csv')
data2.info()

data2

data2['time'].value_counts()

data2['temperature_2m_max'].value_counts()

data2['temperature_2m_min'].value_counts()

data2['temperature_2m_mean'].value_counts()

data2['precipitation_sum'].value_counts()

data2['rain_sum'].value_counts()

data2['snowfall_sum'].value_counts()

data2['precipitation_hours'].value_counts()

data2['wind_speed_10m_max'].value_counts()

data2['wind_gusts_10m_max'].value_counts()

data2['wind_direction_10m_dominant'].value_counts()

data2['weather_description'].value_counts()

data2 = data2.drop(['temperature_2m_min', 'temperature_2m_max', 'precipitation_hours', 'wind_gusts_10m_max', 'wind_direction_10m_dominant','rain_sum', 'snowfall_sum' ], axis=1)

data2.info()

data2["time"] = pd.to_datetime(data2["time"]).dt.strftime('%Y-%m-%d')


data2['time'] = pd.to_datetime(data2['time'])

data2

data1.to_csv('currency_new.csv', index=False)
data2.to_csv('weather_new.csv', index=False)

data_new1 = pd.read_csv('currency_new.csv')
data_new2 = pd.read_csv('weather_new.csv')

# Ubah kolom ke format datetime
data_new1['date'] = pd.to_datetime(data_new1['date'])
data_new2['time'] = pd.to_datetime(data_new2['time'])

# Tambahkan kolom 'year' dan 'month' di kedua DataFrame
data_new1['year'] = data_new1['date'].dt.year
data_new1['month'] = data_new1['date'].dt.month

data_new2['year'] = data_new2['time'].dt.year
data_new2['month'] = data_new2['time'].dt.month

# Drop duplikasi bulan di data1 (jika ada)
data_new1_unique = data_new1.drop_duplicates(subset=['year', 'month'])

# Merge data berdasarkan tahun dan bulan
data_new = pd.merge(data_new2, data_new1_unique[['year', 'month', 'rate']], on=['year', 'month'], how='left')

# Hasil
data_new

# drop unnecessary columns
data_new = data_new.drop(['month', 'year'], axis = 1)

data_new.info()

data_new.isna().sum()

print(f'The oldest time is {df.time.min()}\nThe latest time is {df.time.max()}')

plt.figure(figsize=(20, 10))
sns.lineplot(x=data_new['time'], y=data_new['rate'], color='green')
plt.xlabel('time')
plt.ylabel('rate')
plt.title('Lineplot of Time vs rate')
plt.show()


# Convert time to numerical rate for regression
data_new['numeric_datetime'] = pd.to_numeric(pd.to_datetime(data_new['time']))

# Fit a linear regression line
coefficients = np.polyfit(data_new['numeric_datetime'], data_new['rate'], 1)
polynomial = np.poly1d(coefficients)
trend_line = polynomial(data_new['numeric_datetime'])

plt.figure(figsize=(20, 10))
# Changed 'datetime' to 'Date time' in line 11 and 14
plt.plot(data_new['time'], data_new['rate'], label='Value', color = 'green') # Changed 'Date time' to 'datetime'
plt.plot(data_new['time'], trend_line, label='Trend Line', color='red', linestyle='--') # Changed 'Date time' to 'datetime'
plt.xlabel('Datetime')
plt.ylabel('Value')
plt.title('Distribution of Datetime and Value')

# Filter data to include only the first day of each month
# Changed 'datetime' to 'Date time' in line 19
monthly_data = data_new[pd.to_datetime(data_new['time']).dt.day == 1] # Changed 'Date time' to 'datetime'


# Adding text annotations on top of the line for the first day of each month
# Changed 'datetime' to 'Date time' in line 22
for i, (dt, val) in enumerate(zip(monthly_data['time'], monthly_data['rate'])): # Changed 'Date time' to 'datetime'
    plt.text(dt, val, f'{val:.2f}', ha='center', va='bottom', fontsize=12, rotation=0, color='blue')

plt.legend()
plt.show()

data_new.drop(['numeric_datetime'], axis = 1, inplace=True)

data_new

# @title Temperature vs. Wind Speed

import matplotlib.pyplot as plt

# Assuming your data is in a pandas DataFrame called 'df'

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['temperature_2m_mean'], df['wind_speed_10m_max'])

plt.xlabel('Temperature (℃)')
plt.ylabel('Wind Speed (m/s)')
_ = plt.title('Temperature vs. Wind Speed')


# @title precipitation_sum vs wind_speed_10m_max

from matplotlib import pyplot as plt
data_new.plot(kind='scatter', x='precipitation_sum', y='wind_speed_10m_max', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

data_encoded = data_new.drop(['weather_description'], axis = 1)

plt.figure(figsize=(20, 10))
sns.heatmap(data_encoded.corr(), annot=True, cmap='Greens')
plt.show()

# relocate the value column to the end

col_value = data_encoded.pop('rate')
data_encoded.insert(len(data_encoded.columns), 'rate', col_value)

# data_encoded to csv

data_encoded.to_csv('data_final.csv', index=False)

df = pd.read_csv('data_final.csv')
df.info()

# Identify numeric columns excluding datetime and value columns
features_col = df.select_dtypes(include=['number']).columns
features_col = [col for col in features_col if col not in ['datetime', 'rate', 'weather_code']]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(30, 5))
axes = axes.flatten()

# Iterate over numeric columns
for i, column in enumerate(features_col):
    # Create boxplot only if the column is numeric
    sns.boxplot(x=df[column], ax=axes[i])

# Display the plots
plt.show()

# Remove outliers
for column in features_col:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df = df.loc[(df[column] > fence_low) & (df[column] < fence_high)]

# Display the modified data
df

# Identify numeric columns excluding datetime and value columns
features_col = df.select_dtypes(include=['number']).columns
features_col = [col for col in features_col if col not in ['datetime', 'rate', 'weather_code']]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(30, 5))
axes = axes.flatten()

# Iterate over numeric columns
for i, column in enumerate(features_col):
    # Create boxplot only if the column is numeric
    sns.boxplot(x=df[column], ax=axes[i])

# Display the plots
plt.show()

# Data preprocessing
X = df.drop(['time', 'rate'], axis=1)  # Features (excluding 'time' dan 'rate')
y = df['rate']  # Target (kolom 'rate')

# Train-test split berdasarkan waktu
train_size = int(0.8 * len(df))  # Menghitung ukuran data training (80%)
train, test = df[:train_size], df[train_size:]  # Membagi data menjadi train dan test


# Splitting into X_train, X_test, y_train, y_test
X_train, y_train = train.drop(['time', 'rate'], axis=1), train['rate']  # Fitur dan target untuk data training
X_test, y_test = test.drop(['time', 'rate'], axis=1), test['rate']  # Fitur dan target untuk data testing


model = LinearRegression()
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')

# Add a line for perfect prediction (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect prediction')

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - Linear Regression')
plt.legend()

# Show plot
plt.show()



# Fit XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')

# Add a line for perfect prediction (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect prediction')

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - XGBoostRegressor')
plt.legend()

# Show plot
plt.show()



# Fit RandomForestRegressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')

# Add a line for perfect prediction (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect prediction')

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - RandomForestRegressor')
plt.legend()

# Show plot
plt.show()


# Fit DecisionTreeRegressor model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')

# Add a line for perfect prediction (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect prediction')

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - DecisionTreeRegressor')
plt.legend()

# Show plot
plt.show()


