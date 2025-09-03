import pandas as pd

# Load full stock data with date parsing
stock_data = pd.read_csv('stock_data.csv', parse_dates=['Date'], dayfirst=True)
stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')

# Forecast dates: 10 business days ahead after last available date
last_date = stock_data['Date'].max()
forecast_dates = pd.date_range(last_date, periods=11, freq='B')[1:]

# Actual closes for forecast dates (might be empty if those are real future days)
actuals = stock_data[stock_data['Date'].isin(forecast_dates)][['Date', 'Close']]
actuals.to_csv('actual_closing_prices.csv', index=False)

print("Saved actual closing prices for forecast dates as 'actual_closing_prices.csv'.")
