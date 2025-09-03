# Stock Price Forecasting Comparison

This project demonstrates the forecasting of stock prices using various time series models including ARIMA, SARIMA, Prophet, and LSTM. It compares the predictive performance of these models on historical stock data and visualizes the forecasts with an interactive dashboard built using Streamlit and Plotly.

## Features

- Data preprocessing including date parsing and cleaning
- ARIMA and SARIMA modeling for classical time series forecasting
- Prophet modeling for additive seasonal trend forecasting
- LSTM neural network modeling for deep learning-based forecasting
- Generation of prediction CSV files for each model over a 10-day horizon
- Actual historical closing prices loading and comparison
- Interactive dashboard for comparison of actuals vs. predicted values across models
- Visualizations including forecast plots and model performance insights

## Project Structure

- `stock_data.csv` - Historical stock price data including Open, High, Low, Close, and Volume
- `arima_model.py` - Code for training and predicting with ARIMA model
- `sarima_model.py` - Code for training and predicting with SARIMA model
- `prophet_model.py` - Code for training and predicting with Prophet model
- `lstm_model.py` - Code for training and predicting with LSTM deep learning model
- `prepare_actuals.py` - Extracts actual closing prices for forecast dates
- `streamlit_app.py` - Streamlit dashboard application showing all model predictions vs actuals
- `arima_predictions.csv`, `sarima_predictions.csv`, `prophet_predictions.csv`, `lstm_predictions.csv` - Model forecast CSV files

## Installation

Make sure you have Python 3.8+ installed along with the dependencies below:


## Usage

1. Run model training and prediction scripts to generate forecast CSVs:

2. Launch the Streamlit dashboard:

3. Use the dashboard to visualize and compare forecasts from different models with actual stock closing prices.

## Notes

- The Prophet model expects dates in day-first format (dd-mm-yyyy).
- LSTM model uses the past 60 days of closing prices to predict future prices.
- Forecasts cover the next 10 business days beyond the last date of the historical data.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or additional features.

## License

This project is licensed under the MIT License.

