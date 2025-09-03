import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Script started")


# Parameters
ticker = "AAPL"  # change to any valid stock ticker

# Calculate the dates for last 10 years dynamically
end_date = datetime.today()
start_date = end_date - timedelta(days=365*10)  # approx 10 years

# Format dates as strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Download historical stock data
stock_data = yf.download(ticker, start=start_date_str, end=end_date_str)
print("Data downloaded")


# Save data to CSV
stock_data.to_csv("stock_data.csv")

print(stock_data.head())
print(f"Data collected and saved as 'stock_data.csv' for the period {start_date_str} to {end_date_str}")
print("Script finished")


