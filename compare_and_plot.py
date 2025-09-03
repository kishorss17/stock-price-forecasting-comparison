import pandas as pd
import plotly.graph_objects as go

# Load model predictions
arima_preds = pd.read_csv('arima_predictions.csv', parse_dates=['Date'])
sarima_preds = pd.read_csv('sarima_predictions.csv', parse_dates=['Date'])
prophet_preds = pd.read_csv('prophet_predictions.csv', parse_dates=['Date'])
lstm_preds = pd.read_csv('lstm_predictions.csv', parse_dates=['Date'])

# Merge predictions on Date
all_preds = arima_preds.merge(sarima_preds, on='Date') \
                       .merge(prophet_preds, on='Date') \
                       .merge(lstm_preds, on='Date')

# Load actuals
try:
    actuals = pd.read_csv('actual_closing_prices.csv', parse_dates=['Date'])
    all_data = all_preds.merge(actuals, on='Date', how='left')
except FileNotFoundError:
    print("No actual closing prices found for forecast dates.")
    all_data = all_preds.copy()
    all_data['Close'] = None

# Prepare series for plotting
dates = all_data['Date']
test_actual = all_data['Close']
arima_predictions = all_data['ARIMA_Prediction']
sarima_predictions = all_data['SARIMA_Prediction']
prophet_predictions = all_data['Prophet_Prediction']
lstm_predictions = all_data['LSTM_Prediction']

fig = go.Figure()
if test_actual.notnull().any():
    fig.add_trace(go.Scatter(x=dates, y=test_actual, mode='lines+markers', name='Actual'))
fig.add_trace(go.Scatter(x=dates, y=arima_predictions, mode='lines+markers', name='ARIMA'))
fig.add_trace(go.Scatter(x=dates, y=sarima_predictions, mode='lines+markers', name='SARIMA'))
fig.add_trace(go.Scatter(x=dates, y=prophet_predictions, mode='lines+markers', name='Prophet'))
fig.add_trace(go.Scatter(x=dates, y=lstm_predictions, mode='lines+markers', name='LSTM'))

fig.update_layout(
    title='Stock Price Predictions Comparison',
    xaxis_title='Date',
    yaxis_title='Price ($)',
    width=900,
    height=500
)
fig.write_html("plotly_test.html")
print("Wrote plotly_test.html to current folder, open it in your browser.")

