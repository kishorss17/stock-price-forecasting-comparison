import pandas as pd
import matplotlib.pyplot as plt

# Load collected data with correct parsing
stock_data = pd.read_csv('stock_data.csv', parse_dates=['Date'], dayfirst=True, index_col='Date')
# Explanation:
# - parse_dates on 'Date' column (not index_col=0), then set it as index by index_col='Date'
# - dayfirst=True because your dates are in dd-mm-yyyy format

stock_data.index = pd.to_datetime(stock_data.index, errors='coerce', dayfirst=True)

# Check for any NaT values in index (dates)
if stock_data.index.isnull().any():
    print("WARNING: Some index entries could not be converted to dates and are NaT (Not-a-Time).")
    stock_data = stock_data[~stock_data.index.isnull()]

# Plot closing prices over time
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Close Price')
plt.title('Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.grid(True)
plt.legend()
plt.show()

# Check for missing values in each column
print("Missing values in each column:")
print(stock_data.isnull().sum())
