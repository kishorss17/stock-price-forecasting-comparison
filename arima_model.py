import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load your stock data with correct parsing
stock_data = pd.read_csv('stock_data.csv', parse_dates=['Date'], dayfirst=True, index_col='Date')

# Properly convert Close to numeric and index to datetime
stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data.index = pd.to_datetime(stock_data.index, errors='coerce', dayfirst=True)

# Remove rows with invalid dates or Close prices
stock_data = stock_data[~stock_data.index.isnull()]
stock_data = stock_data[stock_data['Close'].notnull()]

# Visual diagnostics: Plot ACF and PACF of differenced Close price
plt.figure(figsize=(10, 4))
plot_acf(stock_data['Close'].diff().dropna(), lags=20)
plt.title("Autocorrelation (ACF) of Differenced 'Close' Price")
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(stock_data['Close'].diff().dropna(), lags=20)
plt.title("Partial Autocorrelation (PACF) of Differenced 'Close' Price")
plt.show()

# Fit ARIMA model (order=(1,1,1) starting point)
model = ARIMA(stock_data['Close'], order=(1, 1, 1))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast next 10 closing prices using ARIMA model
forecast = fitted_model.forecast(steps=10)
print("Next 10 predicted closing prices (ARIMA):")
print(forecast)

# Plot actual Close prices and ARIMA forecasts
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Close'], label='Actual')

forecast_index = pd.date_range(stock_data.index[-1], periods=11, freq='B')[1:]  # business days
plt.plot(forecast_index, forecast, label='ARIMA Forecast', color='red', marker='o')

plt.title('ARIMA Forecast of Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# SARIMA Model Implementation
sarima_model = SARIMAX(stock_data['Close'],
                      order=(1, 1, 1),  # (p,d,q)
                      seasonal_order=(1, 1, 1, 12))  # (P,D,Q,s)
sarima_fitted = sarima_model.fit()

print("SARIMA Model Summary:")
print(sarima_fitted.summary())

# Forecast next 10 periods using SARIMA
sarima_forecast = sarima_fitted.forecast(steps=10)
print("Next 10 predicted closing prices (SARIMA):")
print(sarima_forecast)

# Plot actual data and SARIMA forecast
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Close'], label='Actual')
future_index = pd.date_range(stock_data.index[-1], periods=11, freq='B')[1:]
plt.plot(future_index, sarima_forecast, label='SARIMA Forecast', color='green', marker='o')
plt.title('SARIMA Forecast of Closing Price')
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Save ARIMA and SARIMA forecasts for Plotly visualization
import pandas as pd

# Build future dates index
future_index = pd.date_range(stock_data.index[-1], periods=11, freq='B')[1:]

# Save ARIMA predictions
arima_forecast_df = pd.DataFrame({'Date': future_index, 'ARIMA_Prediction': forecast.values})
arima_forecast_df.to_csv('arima_predictions.csv', index=False)

# Save SARIMA predictions
sarima_forecast_df = pd.DataFrame({'Date': future_index, 'SARIMA_Prediction': sarima_forecast.values})
sarima_forecast_df.to_csv('sarima_predictions.csv', index=False)

print("ARIMA and SARIMA predictions exported as CSV.")
