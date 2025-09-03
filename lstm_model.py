import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------------
# 1. Load and prepare data
# ------------------------

# Read with date parsing
stock_data = pd.read_csv('stock_data.csv', parse_dates=['Date'], dayfirst=True, index_col='Date')
stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data = stock_data[stock_data['Close'].notnull()]

# Sort by date, if not sorted
stock_data.sort_index(inplace=True)

# Visualize the closing prices
plt.figure(figsize=(12,6))
plt.plot(stock_data['Close'], label='Closing Price')
plt.title('Stock Closing Price History')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()
plt.show()

# ---------------------------
# 2. Preprocess for LSTM
# ---------------------------

# Use only 'Close' column, scale to [0, 1]
close_prices = stock_data[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_prices)

# Sequence length (lookback window)
seq_size = 60  # use past 60 days to predict next day

X = []
y = []

for i in range(seq_size, len(scaled_close)):
    X.append(scaled_close[i-seq_size:i, 0])
    y.append(scaled_close[i, 0])

X, y = np.array(X), np.array(y)

# LSTM expects input shape: (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets (e.g., last 10% as test)
split = int(0.9 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# ---------------------------------
# 3. Define and train LSTM model
# ---------------------------------

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# ---------------------------------
# 4. Predict and visualize results
# ---------------------------------

# Predict on test set
y_pred = model.predict(X_test)

# Reverse the scaling
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_actual = scaler.inverse_transform(y_pred).flatten()

# Plot actual vs. predicted prices
plt.figure(figsize=(12,6))
plt.plot(stock_data.index[-len(y_test_actual):], y_test_actual, color='blue', label='Actual Price')
plt.plot(stock_data.index[-len(y_test_actual):], y_pred_actual, color='red', label='Predicted Price')
plt.title('LSTM Model: Actual vs Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# ---------------------------------
# 5. Forecast future prices
# ---------------------------------

# Forecast next 10 days using last 'seq_size' days
last_sequence = scaled_close[-seq_size:]
future_preds = []

current_seq = last_sequence.copy()
for _ in range(10):
    pred = model.predict(current_seq.reshape(1, seq_size, 1))[0][0]
    future_preds.append(pred)
    current_seq = np.append(current_seq[1:], pred).reshape(seq_size, 1)

# Inverse scaling
future_preds_actual = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

# Generate future dates
last_date = stock_data.index[-1]
future_dates = pd.date_range(last_date, periods=11, freq='B')[1:]

# Plot future forecast
plt.figure(figsize=(10,6))
plt.plot(stock_data.index[-60:], scaler.inverse_transform(scaled_close[-60:]), label='Recent Close Prices')
plt.plot(future_dates, future_preds_actual, marker='o', linestyle='--', color='orange', label='LSTM Forecast (Next 10 Days)')
plt.title('LSTM Forecast for Next 10 Business Days')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

# Save LSTM 10-day forecast for Plotly
lstm_forecast_df = pd.DataFrame({'Date': future_dates, 'LSTM_Prediction': future_preds_actual})
lstm_forecast_df.to_csv('lstm_predictions.csv', index=False)

print("LSTM predictions exported as CSV.")

