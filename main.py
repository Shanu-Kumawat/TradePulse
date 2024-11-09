import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    

# Constants
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Main app title
st.title("Advanced Stock Prediction and Comparison App")

# Sidebar configuration
st.sidebar.title("Settings")
selected_stock = st.sidebar.text_input("Primary stock ticker (e.g., AAPL, MSFT)", "AAPL")
n_years = st.sidebar.slider("Forecast years", 1, 4)
period = n_years * 365
candlestick_interval = st.sidebar.selectbox("Candlestick Interval", ["1d", "1wk", "1mo"])

# Comparison mode configuration
enable_comparison = st.sidebar.checkbox("Enable Comparison Mode")
if enable_comparison:
    stock2 = st.sidebar.text_input("Second stock ticker", "")
    stock3 = st.sidebar.text_input("Third stock ticker", "")

@st.cache_data
def load_data(ticker, interval):
    try:
        data = yf.download(ticker, START, TODAY, interval=interval)
        if data.empty:
            st.error(f"No data found for {ticker}.")
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def process_data(data):
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Date'])
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data.dropna()

def plot_candlestick(data, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
        increasing_line_color='lime', decreasing_line_color='crimson'
    )])
    fig.update_layout(title=f'{ticker} - Candlestick Chart', template='plotly_dark')
    return fig

def plot_moving_average(data, ticker, show_moving_avg=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name=f'{ticker} Stock Price'))
    if show_moving_avg:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'].rolling(window=20).mean(),
                                 mode='lines', name='20-day Moving Average'))
    fig.update_layout(title=f'{ticker} - Stock Price with Moving Average', xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_volume(data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume'))
    fig.update_layout(title=f'{ticker} - Stock Volume', xaxis_title='Date', yaxis_title='Volume')
    return fig

def prophet_forecast(data):
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = df_train['ds'].dt.tz_convert(None)
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    return model, forecast

def lstm_forecast(data):
    df = data[['Date', 'Close']].copy()
    df.index = df['Date']
    df = df.drop(columns=['Date'])

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Create training data
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=2)

    # Prepare testing data
    test_data = scaled_data[train_size - 60:]
    x_test, y_test = [], scaled_data[train_size:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Forecast future prices
    last_60_days = scaled_data[-60:]
    future_input = last_60_days.reshape((1, 60, 1))
    future_pred = model.predict(future_input)
    future_pred = scaler.inverse_transform(future_pred)

    # Return predictions and future forecast
    return predictions, future_pred

# Load and process primary stock data
data_load_state = st.text("Loading primary stock data...")
data = load_data(selected_stock, candlestick_interval)
data_load_state.text("Loading primary stock data...Done!")

# Display primary stock data and features
if data is not None:
    data = process_data(data)
    st.subheader(f"Primary Stock Data for {selected_stock}")
    st.write(data.tail())

    # Tabs for organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Candlestick Chart", "Moving Average", "Volume", "Forecast", "LSTM Model"])

    with tab1:
        st.plotly_chart(plot_candlestick(data, selected_stock))

    with tab2:
        show_moving_avg = st.checkbox("Show Moving Average", value=True)
        st.plotly_chart(plot_moving_average(data, selected_stock, show_moving_avg))

    with tab3:
        st.plotly_chart(plot_volume(data, selected_stock))

    with tab4:
        model, forecast = prophet_forecast(data)
        st.subheader("Forecast Data")
        st.write(forecast.tail())
        st.plotly_chart(plot_plotly(model, forecast))
        st.write(model.plot_components(forecast))

    
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

with tab5:
    st.subheader("LSTM Model Prediction")

    # Run LSTM forecasting
    lstm_pred, future_pred = lstm_forecast(data)

    # Prepare data for plotting
    data['LSTM Prediction'] = np.nan  # Initialize with NaNs
    data['LSTM Prediction'].iloc[-len(lstm_pred):] = lstm_pred.flatten()  # Fill LSTM predictions in the last part of the series

    # Calculate evaluation metrics
    y_test = data['Close'].iloc[-len(lstm_pred):].values  # Actual values for the prediction period
    mae = mean_absolute_error(y_test, lstm_pred)
    mse = mean_squared_error(y_test, lstm_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, lstm_pred)

    # Plot LSTM predictions using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name="Actual Close Price", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['LSTM Prediction'], mode='lines', name="LSTM Prediction", line=dict(color="orange")))

    # Mark the next day predicted price
    future_date = data['Date'].iloc[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[future_pred[0][0]], mode='markers+text', text=f"{future_pred[0][0]:.2f}",
                             name="Next Day Prediction", marker=dict(color="red", size=10),
                             textposition="top center"))

    # Customize the layout
    fig.update_layout(
        title=f"{selected_stock} - LSTM Prediction vs Actual Prices",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_dark",xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig)  # Display the plot in Streamlit

    # Display LSTM metrics
    st.write("### LSTM Model Evaluation Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
    st.write(f"Predicted Price for Next Day: {future_pred[0][0]:.2f}")
