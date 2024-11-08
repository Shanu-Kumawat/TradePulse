import streamlit as st
import pandas as pd
from datetime import date

import yfinance as yf
from prophet import Prophet

from prophet.plot import plot_plotly
from plotly import graph_objs as go


START="2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks=("GOOG","MSFT","GME","SONY")

selected_stock = st.selectbox("Select dataset for prediction", options=stocks)

n_years=st.slider("Years of prediction",1,4)
period=n_years*365

@st.cache_data 
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("Load data...")
data=load_data(selected_stock)
data_load_state.text("Loading data....Done!")

data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
st.subheader("Raw data")
st.write(data.tail())

# Check for missing values
st.subheader("Missing Values")
st.write(data.isnull().sum())

# Fill missing values
data = data.fillna(method='ffill')

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Convert numeric columns to numeric data type
for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
    data[col] = pd.to_numeric(data[col])


# Detect and remove outliers
Q1 = data['Close'].quantile(0.25)
Q3 = data['Close'].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (data['Close'] < Q1 - 1.5 * IQR) | (data['Close'] > Q3 + 1.5 * IQR)
data = data[~outlier_mask]

st.subheader("Data Summary")
st.write(data.describe())


st.subheader("Candlestick Chart - Last 10 Days")

fig = go.Figure(data=[go.Candlestick(
    x=data['Date'][-10:],
    open=data['Open'][-10:],
    high=data['High'][-10:],
    low=data['Low'][-10:],
    close=data['Close'][-10:],
    increasing_line_color='lime',  # Bright green for bullish candles
    decreasing_line_color='crimson' # Bright red for bearish candles
)])

fig.update_layout(
    title={
        'text': 'Stock Price Candlestick Chart - Last 10 Days',
        'font': {'size': 24, 'color': 'white'}
    },
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True,
    template='plotly_dark',           # Dark theme for improved contrast
    xaxis=dict(
        showgrid=True,                # Gridlines for readability
        gridcolor='gray'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='gray'
    )
)

st.plotly_chart(fig)

st.subheader("Stock Price with Moving Average")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Stock Price'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'].rolling(window=20).mean(), mode='lines', name='20-day Moving Average'))
fig.update_layout(
    title='Stock Price with Moving Average',
    xaxis_title='Date',
    yaxis_title='Price',xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

st.subheader("Stock Volume")
fig = go.Figure()
fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume'))
fig.update_layout(
    title='Stock Volume',
    xaxis_title='Date',
    yaxis_title='Volume',xaxis_rangeslider_visible=True)
st.plotly_chart(fig)


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']].copy()  # Select Date and Close columns
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # Rename for Prophet

# Ensure the date column is in datetime format and remove timezone
df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None)  # Remove timezone if present
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')  # Convert y to numeric, handling errors

# Initialize and fit the Prophet model
m = Prophet()
m.fit(df_train)

# Create future dates and predict
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data in Streamlit
st.subheader("Forecast data")
st.write(forecast.tail())

st.write('forecast data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2=m.plot_components(forecast)
st.write(fig2)