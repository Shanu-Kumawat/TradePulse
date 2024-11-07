import streamlit as st
import pandas as pd
from datetime import date

import yfinance as yf
from prophet import Prophet

from prophet.plot import plot_plotly
from plotly import graph_objs as go


START="2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks=("AAPL","GOOG","MSFT","GME")

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

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# # Display column names
# st.subheader("Column Names")
# st.write(data.columns)

# Prepare data for Prophet
# Prepare data for Prophet
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
