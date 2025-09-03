import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Read the CSV with proper date parsing for dd-mm-yyyy format
stock_data = pd.read_csv('stock_data.csv', parse_dates=['Date'], dayfirst=True)

# Rename columns to Prophet expected format: ‘ds’ and ‘y’
prophet_data = stock_data.rename(columns={'Date': 'ds', 'Close': 'y'})[['ds', 'y']]

# Optional: Check for missing or invalid dates
prophet_data = prophet_data.dropna(subset=['ds', 'y'])
prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], errors='coerce')
prophet_data = prophet_data[~prophet_data['ds'].isnull()]

# Instantiate and fit Prophet model
model = Prophet()
model.fit(prophet_data)

# Forecast for next 10 days (business days included, but Prophet creates daily by default)
future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)
plt.title('Prophet Stock Price Forecast - Next 10 Days')
plt.show()

# Plot trend and seasonal components
model.plot_components(forecast)
plt.show()

# Export LAST 10 forecast days for plotting
prophet_forecast_10 = forecast[['ds', 'yhat']].tail(10)
prophet_forecast_10 = prophet_forecast_10.rename(columns={'ds': 'Date', 'yhat': 'Prophet_Prediction'})
prophet_forecast_10.to_csv('prophet_predictions.csv', index=False)

print("Prophet predictions exported as CSV.")
