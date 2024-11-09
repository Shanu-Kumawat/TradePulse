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
stock2 = st.sidebar.text_input("Second stock ticker") if enable_comparison else None

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

def plot_combined_candlestick(data1, ticker1, data2=None, ticker2=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data1['Date'], open=data1['Open'], high=data1['High'], low=data1['Low'], close=data1['Close'],
        name=f"{ticker1}", increasing_line_color='lime', decreasing_line_color='crimson'
    ))
    if data2 is not None:
        fig.add_trace(go.Candlestick(
            x=data2['Date'], open=data2['Open'], high=data2['High'], low=data2['Low'], close=data2['Close'],
            name=f"{ticker2}", increasing_line_color='blue', decreasing_line_color='orange'
        ))
    fig.update_layout(title="Candlestick Comparison", template="plotly_dark")
    return fig

def plot_combined_moving_average(data1, ticker1, data2=None, ticker2=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['Close'], mode='lines', name=f"{ticker1} Stock Price"))
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['Close'].rolling(window=20).mean(),
                             mode='lines', name=f"{ticker1} 20-day Moving Avg", line=dict(dash="dash")))
    if data2 is not None:
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Close'], mode='lines', name=f"{ticker2} Stock Price"))
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Close'].rolling(window=20).mean(),
                                 mode='lines', name=f"{ticker2} 20-day Moving Avg", line=dict(dash="dash")))
    fig.update_layout(title="Stock Price and Moving Average Comparison", xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_combined_volume(data1, ticker1, data2=None, ticker2=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data1['Date'], y=data1['Volume'], name=f"{ticker1} Volume"))
    if data2 is not None:
        fig.add_trace(go.Bar(x=data2['Date'], y=data2['Volume'], name=f"{ticker2} Volume"))
    fig.update_layout(title="Volume Comparison", xaxis_title="Date", yaxis_title="Volume")
    return fig

def plot_combined_forecast(forecast1, ticker1, forecast2=None, ticker2=None):
    fig = go.Figure()
    
    # Plot the first forecast with confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast1['ds'], y=forecast1['yhat'], mode='lines', name=f"{ticker1} Forecast",
        line=dict(color="lime")
    ))
    fig.add_trace(go.Scatter(
        x=forecast1['ds'], y=forecast1['yhat_upper'], fill=None, mode='lines',
        line=dict(color="lightgreen", dash="dash"), name=f"{ticker1} Upper Confidence Interval"
    ))
    fig.add_trace(go.Scatter(
        x=forecast1['ds'], y=forecast1['yhat_lower'], fill='tonexty', mode='lines',
        line=dict(color="lightgreen", dash="dash"), name=f"{ticker1} Lower Confidence Interval"
    ))

    # Plot the second forecast if available
    if forecast2 is not None and ticker2:
        fig.add_trace(go.Scatter(
            x=forecast2['ds'], y=forecast2['yhat'], mode='lines', name=f"{ticker2} Forecast",
            line=dict(color="orange")
        ))
        fig.add_trace(go.Scatter(
            x=forecast2['ds'], y=forecast2['yhat_upper'], fill=None, mode='lines',
            line=dict(color="salmon", dash="dash"), name=f"{ticker2} Upper Confidence Interval"
        ))
        fig.add_trace(go.Scatter(
            x=forecast2['ds'], y=forecast2['yhat_lower'], fill='tonexty', mode='lines',
            line=dict(color="salmon", dash="dash"), name=f"{ticker2} Lower Confidence Interval"
        ))

    fig.update_layout(
        title="Combined Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )
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
data1 = load_data(selected_stock, candlestick_interval)
data_load_state.text("Loading primary stock data...Done!")

# Display primary stock data and features
if data1 is not None:
    data1 = process_data(data1)
    data2 = None
    if enable_comparison and stock2:
        data2 = load_data(stock2, candlestick_interval)
        if data2 is not None:
            data2 = process_data(data2)
    
    st.subheader(f"Primary Stock Data for {selected_stock}")
    st.write(data1.tail())

    # Tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["Candlestick Chart", "Moving Average", "Volume", "Forecast"])

    with tab1:
        st.plotly_chart(plot_combined_candlestick(data1, selected_stock, data2, stock2 if enable_comparison else None))

    with tab2:
        show_moving_avg = st.checkbox("Show Moving Average", value=True)
        st.plotly_chart(plot_combined_moving_average(data1, selected_stock, data2, stock2 if enable_comparison else None))

    with tab3:
        st.plotly_chart(plot_combined_volume(data1, selected_stock, data2, stock2 if enable_comparison else None))

# Main section for forecast display in tabs
with tab4:
    model1, forecast1 = prophet_forecast(data1)
    st.subheader("Forecast Data")
    st.write(forecast1.tail())
    
    # Generate and show the combined forecast plot
    if enable_comparison and data2 is not None:
        model2, forecast2 = prophet_forecast(data2)
        combined_forecast_fig = plot_combined_forecast(forecast1, selected_stock, forecast2, stock2)
        st.plotly_chart(combined_forecast_fig)
    else:
        # Show forecast for primary stock only
        st.plotly_chart(plot_plotly(model1, forecast1))
        
    st.write(model1.plot_components(forecast1))

    # Show the components for the second forecast if in comparison mode
    if enable_comparison and data2 is not None:
        st.write(model2.plot_components(forecast2))
