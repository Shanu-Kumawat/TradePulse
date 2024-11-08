import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

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
    tab1, tab2, tab3, tab4 = st.tabs(["Candlestick Chart", "Moving Average", "Volume", "Forecast"])

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

# Comparison mode - Load and display data for additional stocks
if enable_comparison:
    for ticker in [stock2, stock3]:
        if ticker:
            comp_data = load_data(ticker, candlestick_interval)
            if comp_data is not None:
                comp_data = process_data(comp_data)
                st.subheader(f"Comparison Stock Data for {ticker}")

                # Display each feature for comparison stocks
                tab1, tab2, tab3, tab4 = st.tabs([f"{ticker} Candlestick", f"{ticker} Moving Avg",
                                                  f"{ticker} Volume", f"{ticker} Forecast"])

                with tab1:
                    st.plotly_chart(plot_candlestick(comp_data, ticker))

                with tab2:
                    show_moving_avg = st.checkbox(f"Show Moving Average for {ticker}", value=True)
                    st.plotly_chart(plot_moving_average(comp_data, ticker, show_moving_avg))

                with tab3:
                    st.plotly_chart(plot_volume(comp_data, ticker))

                with tab4:
                    comp_model, comp_forecast = prophet_forecast(comp_data)
                    st.subheader(f"Forecast Data for {ticker}")
                    st.write(comp_forecast.tail())
                    st.plotly_chart(plot_plotly(comp_model, comp_forecast))
                    st.write(comp_model.plot_components(comp_forecast))

