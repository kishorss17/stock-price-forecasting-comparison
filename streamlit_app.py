import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Sidebar ---
st.sidebar.title("Stock Prediction Models Demo")
st.sidebar.write("Choose what to display:")

# In a real scenario, you may let the user select stock, forecast period, etc.

# --- Load Data Helper Functions ---

@st.cache_data
def load_preds():
    arima = pd.read_csv('arima_predictions.csv', parse_dates=['Date'])
    sarima = pd.read_csv('sarima_predictions.csv', parse_dates=['Date'])
    prophet = pd.read_csv('prophet_predictions.csv', parse_dates=['Date'])
    lstm = pd.read_csv('lstm_predictions.csv', parse_dates=['Date'])
    return arima, sarima, prophet, lstm

@st.cache_data
def load_actuals():
    try:
        actuals = pd.read_csv('actual_closing_prices.csv', parse_dates=['Date'])
        return actuals
    except:
        return None

# --- Core Functions ---

def run_model():
    """You'd implement model training/prediction if you want live runs. For pre-run output, just load predictions."""
    arima, sarima, prophet, lstm = load_preds()
    actuals = load_actuals()
    # Merge as before for clean plotting
    all_data = arima.merge(sarima, on='Date') \
            .merge(prophet, on='Date') \
            .merge(lstm, on='Date')
    if actuals is not None and not actuals.empty:
        all_data = all_data.merge(actuals, on='Date', how='left')
    return all_data

def create_forecast_plot(all_data):
    """Creates the interactive Plotly figure for display in Streamlit"""
    fig = go.Figure()

    # Plot actuals if available
    if 'Close' in all_data.columns and all_data['Close'].notnull().any():
        fig.add_trace(go.Scatter(x=all_data['Date'], y=all_data['Close'], mode='lines+markers', name='Actual', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=all_data['Date'], y=all_data['ARIMA_Prediction'], mode='lines+markers', name='ARIMA'))
    fig.add_trace(go.Scatter(x=all_data['Date'], y=all_data['SARIMA_Prediction'], mode='lines+markers', name='SARIMA'))
    fig.add_trace(go.Scatter(x=all_data['Date'], y=all_data['Prophet_Prediction'], mode='lines+markers', name='Prophet'))
    fig.add_trace(go.Scatter(x=all_data['Date'], y=all_data['LSTM_Prediction'], mode='lines+markers', name='LSTM'))

    fig.update_layout(
        title='Stock Price Predictions Comparison',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        width=900,
        height=500
    )
    return fig

# --- Streamlit Main UI ---

st.title("Stock Price Model Predictions Dashboard")

if st.button('Load and Display Model Forecasts'):
    all_data = run_model()
    st.subheader("Forecast Results")
    st.write("Table Preview:")
    st.write(all_data)
    st.plotly_chart(create_forecast_plot(all_data), use_container_width=True)
else:
    st.info("Click the button above to view predictions and plot.")

st.caption("Built with Streamlit, Plotly, and data science models for ARIMA, SARIMA, Prophet, LSTM.")
