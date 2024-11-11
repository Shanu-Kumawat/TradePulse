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
from sklearn.ensemble import RandomForestRegressor
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
from datetime import datetime

def prophet_forecast(data):
    # Prepare data for Prophet
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = df_train['ds'].dt.tz_convert(None)
    
    # Initialize and fit the model
    model = Prophet()
    model.fit(df_train)
    
    # Calculate the number of days between the last date in data and today
    last_date = df_train['ds'].max()
    end_date = datetime.today()
    period = max(0, (end_date - last_date).days)  # Ensure non-negative
    
    # Generate future dates up to the present date
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    
    # Filter the forecast to include only dates up to today
    forecast = forecast[forecast['ds'] <= end_date]

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

def random_forest_forecast(data):
    df = data[['Date', 'Close']].copy()
    df.index = df['Date']
    df = df.drop(columns=['Date'])
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Create features and target
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Reshape features for Random Forest
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # Train the model
    model.fit(X_train_2d, y_train)
    
    # Make predictions on test set
    predictions = model.predict(X_test_2d)
    
    # Reshape predictions for inverse transform
    predictions = predictions.reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    # Prepare data for future prediction
    last_60_days = scaled_data[-60:].reshape(1, -1)
    future_pred = model.predict(last_60_days)
    future_pred = future_pred.reshape(-1, 1)
    future_pred = scaler.inverse_transform(future_pred)
    
    return predictions, future_pred, model

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
    tab1, tab2, tab3, tab4, tab5,tab6,tab7 = st.tabs(["Candlestick Chart", "Moving Average", "Volume", "Forecast", "LSTM Model","Random Forest Model","Comaprison between Models"])

    with tab1:
        st.plotly_chart(plot_combined_candlestick(data1, selected_stock, data2, stock2 if enable_comparison else None))

    with tab2:
        show_moving_avg = st.checkbox("Show Moving Average", value=True)
        st.plotly_chart(plot_combined_moving_average(data1, selected_stock, data2, stock2 if enable_comparison else None))

    with tab3:
        st.plotly_chart(plot_combined_volume(data1, selected_stock, data2, stock2 if enable_comparison else None))

# Main section for forecast display in tabs
with tab4:
    # Forecast with Prophet model for primary stock
    model1, forecast1 = prophet_forecast(data1)
    st.subheader(f"{selected_stock} Prophet Model Forecast")

    # Display forecast data
    st.write("### Forecast Data for Primary Stock")
    st.write(forecast1.tail())

    # Plot actual vs predicted for primary stock
    actual_fig1 = go.Figure()
    actual_fig1.add_trace(go.Scatter(x=data1['Date'], y=data1['Close'], 
                                     mode='lines', name=f"{selected_stock} Actual Price",
                                     line=dict(color="blue")))
    actual_fig1.add_trace(go.Scatter(x=forecast1['ds'], y=forecast1['yhat'], 
                                     mode='lines', name=f"{selected_stock} Prophet Prediction", 
                                     line=dict(color="orange")))
        # Customize the layout
    actual_fig1.update_layout(
        title="Forecast Prediction Comparison",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_dark", xaxis_rangeslider_visible=True
    )

    # Run Prophet model for secondary stock if comparison is enabled
    if enable_comparison and data2 is not None:
        model2, forecast2 = prophet_forecast(data2)
        st.write("### Forecast Data for Secondary Stock")
        st.write(forecast2.tail())
        
        # Plot actual vs predicted for secondary stock
        actual_fig2 = go.Figure()
        actual_fig2.add_trace(go.Scatter(x=data2['Date'], y=data2['Close'], 
                                         mode='lines', name=f"{stock2} Actual Price", 
                                         line=dict(color="purple")))
        actual_fig2.add_trace(go.Scatter(x=forecast2['ds'], y=forecast2['yhat'], 
                                         mode='lines', name=f"{stock2} Prophet Prediction", 
                                         line=dict(color="green")))

        # Display combined forecast plot for both stocks
        combined_forecast_fig = plot_combined_forecast(forecast1, selected_stock, forecast2, stock2)
        st.plotly_chart(combined_forecast_fig)

        # Display individual plots for actual vs. predicted data
        st.plotly_chart(actual_fig1)
        st.plotly_chart(actual_fig2)

        # Show components for secondary stock if in comparison mode
        st.write("### Prophet Model Components for Secondary Stock")
        st.write(model2.plot_components(forecast2))
    else:
        # Show forecast for primary stock only
        st.plotly_chart(actual_fig1)
        
    # Display Prophet components for primary stock
    st.write("### Prophet Model Components for Primary Stock")
    st.write(model1.plot_components(forecast1))


    
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

with tab5:
    st.subheader("LSTM Model Prediction")

    # Run LSTM forecasting for primary stock
    lstm_pred1, future_pred1 = lstm_forecast(data1)

    # Prepare data for plotting
    data1['LSTM Prediction'] = np.nan  # Initialize with NaNs
    data1['LSTM Prediction'].iloc[-len(lstm_pred1):] = lstm_pred1.flatten()  # Fill LSTM predictions

    # Calculate evaluation metrics
    y_test1 = data1['Close'].iloc[-len(lstm_pred1):].values  # Actual values for the prediction period
    mae1 = mean_absolute_error(y_test1, lstm_pred1)
    mse1 = mean_squared_error(y_test1, lstm_pred1)
    rmse1 = np.sqrt(mse1)
    r2_1 = r2_score(y_test1, lstm_pred1)

    # Plot LSTM predictions using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['Close'], mode='lines', name=f"{selected_stock} Actual Price", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['LSTM Prediction'], mode='lines', name=f"{selected_stock} LSTM Prediction", line=dict(color="orange")))

    # Mark the next day predicted price for primary stock
    future_date = data1['Date'].iloc[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[future_pred1[0][0]], mode='markers+text', text=f"{future_pred1[0][0]:.2f}",
                             name=f"{selected_stock} Next Day Prediction", marker=dict(color="red", size=10),
                             textposition="top center"))

    # Run LSTM forecasting for secondary stock if in comparison mode
    if enable_comparison and data2 is not None:
        lstm_pred2, future_pred2 = lstm_forecast(data2)

        # Prepare data for plotting
        data2['LSTM Prediction'] = np.nan  # Initialize with NaNs
        data2['LSTM Prediction'].iloc[-len(lstm_pred2):] = lstm_pred2.flatten()  # Fill LSTM predictions for secondary stock

        # Calculate evaluation metrics for secondary stock
        y_test2 = data2['Close'].iloc[-len(lstm_pred2):].values  # Actual values for the prediction period
        mae2 = mean_absolute_error(y_test2, lstm_pred2)
        mse2 = mean_squared_error(y_test2, lstm_pred2)
        rmse2 = np.sqrt(mse2)
        r2_2 = r2_score(y_test2, lstm_pred2)

        # Plot LSTM predictions for secondary stock
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Close'], mode='lines', name=f"{stock2} Actual Price", line=dict(color="purple")))
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['LSTM Prediction'], mode='lines', name=f"{stock2} LSTM Prediction", line=dict(color="green")))

        # Mark the next day predicted price for secondary stock
        future_date2 = data2['Date'].iloc[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(x=[future_date2], y=[future_pred2[0][0]], mode='markers+text', text=f"{future_pred2[0][0]:.2f}",
                                 name=f"{stock2} Next Day Prediction", marker=dict(color="pink", size=10),
                                 textposition="top center"))

        # Display metrics for secondary stock
        st.write(f"### {stock2} LSTM Model Evaluation Metrics")
        st.write(f"Mean Absolute Error (MAE): {mae2:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse2:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse2:.2f}")
        st.write(f"R-squared: {r2_2:.2f}")
        st.write(f"Predicted Price for Next Day: {future_pred2[0][0]:.2f}")

    # Customize the layout
    fig.update_layout(
        title="LSTM Prediction Comparison",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_dark", xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig)  # Display the plot in Streamlit

    # Display metrics for primary stock
    st.write(f"### {selected_stock} LSTM Model Evaluation Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae1:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse1:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse1:.2f}")
    st.write(f"R-squared: {r2_1:.2f}")
    st.write(f"Predicted Price for Next Day: {future_pred1[0][0]:.2f}")

with tab6:
    st.subheader("Random Forest Model Prediction")

    # Run Random Forest forecasting for primary stock
    rf_pred1, rf_future_pred1, rf_model1 = random_forest_forecast(data1)  # Note the additional return value

    # Prepare data for plotting
    data1['RF Prediction'] = np.nan  # Initialize with NaNs
    data1['RF Prediction'].iloc[-len(rf_pred1):] = rf_pred1.flatten()  # Fill RF predictions

    # Calculate evaluation metrics
    y_test1 = data1['Close'].iloc[-len(rf_pred1):].values  # Actual values for the prediction period
    rf_mae1 = mean_absolute_error(y_test1, rf_pred1)
    rf_mse1 = mean_squared_error(y_test1, rf_pred1)
    rf_rmse1 = np.sqrt(rf_mse1)
    rf_r2_1 = r2_score(y_test1, rf_pred1)

    # Plot Random Forest predictions using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['Close'], 
                            mode='lines', name=f"{selected_stock} Actual Price", 
                            line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['RF Prediction'], 
                            mode='lines', name=f"{selected_stock} RF Prediction", 
                            line=dict(color="orange")))

    # Mark the next day predicted price for primary stock
    future_date = data1['Date'].iloc[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[rf_future_pred1[0][0]], 
                            mode='markers+text', 
                            text=f"{rf_future_pred1[0][0]:.2f}",
                            name=f"{selected_stock} Next Day Prediction", 
                            marker=dict(color="red", size=10),
                            textposition="top center"))

    # Run Random Forest forecasting for secondary stock if in comparison mode
    if enable_comparison and data2 is not None:
        rf_pred2, rf_future_pred2, rf_model2 = random_forest_forecast(data2)  # Note the additional return value

        # Prepare data for plotting
        data2['RF Prediction'] = np.nan  # Initialize with NaNs
        data2['RF Prediction'].iloc[-len(rf_pred2):] = rf_pred2.flatten()  # Fill RF predictions for secondary stock

        # Calculate evaluation metrics for secondary stock
        y_test2 = data2['Close'].iloc[-len(rf_pred2):].values
        rf_mae2 = mean_absolute_error(y_test2, rf_pred2)
        rf_mse2 = mean_squared_error(y_test2, rf_pred2)
        rf_rmse2 = np.sqrt(rf_mse2)
        rf_r2_2 = r2_score(y_test2, rf_pred2)

        # Plot Random Forest predictions for secondary stock
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Close'], 
                                mode='lines', name=f"{stock2} Actual Price", 
                                line=dict(color="purple")))
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['RF Prediction'], 
                                mode='lines', name=f"{stock2} RF Prediction", 
                                line=dict(color="green")))

        # Mark the next day predicted price for secondary stock
        future_date2 = data2['Date'].iloc[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(x=[future_date2], y=[rf_future_pred2[0][0]], 
                                mode='markers+text', 
                                text=f"{rf_future_pred2[0][0]:.2f}",
                                name=f"{stock2} Next Day Prediction", 
                                marker=dict(color="pink", size=10),
                                textposition="top center"))

        # Display metrics for secondary stock
        st.write(f"### {stock2} Random Forest Model Evaluation Metrics")
        st.write(f"Mean Absolute Error (MAE): {rf_mae2:.2f}")
        st.write(f"Mean Squared Error (MSE): {rf_mse2:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rf_rmse2:.2f}")
        st.write(f"R-squared: {rf_r2_2:.2f}")
        st.write(f"Predicted Price for Next Day: {rf_future_pred2[0][0]:.2f}")

    # Customize the layout
    fig.update_layout(
        title="Random Forest Prediction Comparison",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_dark",
        xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig)  # Display the plot in Streamlit

    # Display metrics for primary stock
    st.write(f"### {selected_stock} Random Forest Model Evaluation Metrics")
    st.write(f"Mean Absolute Error (MAE): {rf_mae1:.2f}")
    st.write(f"Mean Squared Error (MSE): {rf_mse1:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rf_rmse1:.2f}")
    st.write(f"R-squared: {rf_r2_1:.2f}")
    st.write(f"Predicted Price for Next Day: {rf_future_pred1[0][0]:.2f}")


with tab7:
    st.subheader("Model Comparison Analysis")

    # Get predictions from all models for the primary stock
    prophet_model1, prophet_forecast1 = prophet_forecast(data1)
    lstm_pred1, lstm_future1 = lstm_forecast(data1)
    rf_pred1, rf_future1, rf_model1 = random_forest_forecast(data1)

    # Calculate actual values for comparison
    # For Prophet
    prophet_pred1 = prophet_forecast1['yhat'][-len(data1.index):]
    prophet_actual1 = data1['Close']
    prophet_mae1 = mean_absolute_error(prophet_actual1, prophet_pred1)
    prophet_rmse1 = np.sqrt(mean_squared_error(prophet_actual1, prophet_pred1))
    prophet_r2_1 = r2_score(prophet_actual1, prophet_pred1)

    # For LSTM
    lstm_actual1 = data1['Close'].iloc[-len(lstm_pred1):]
    lstm_mae1 = mean_absolute_error(lstm_actual1, lstm_pred1)
    lstm_rmse1 = np.sqrt(mean_squared_error(lstm_actual1, lstm_pred1))
    lstm_r2_1 = r2_score(lstm_actual1, lstm_pred1)

    # For Random Forest
    rf_actual1 = data1['Close'].iloc[-len(rf_pred1):]
    rf_mae1 = mean_absolute_error(rf_actual1, rf_pred1)
    rf_rmse1 = np.sqrt(mean_squared_error(rf_actual1, rf_pred1))
    rf_r2_1 = r2_score(rf_actual1, rf_pred1)

    # Plot predictions from all models
    fig_comparison = go.Figure()
    
    # Add actual price
    fig_comparison.add_trace(go.Scatter(
        x=data1['Date'],
        y=data1['Close'],
        mode='lines',
        name=f'{selected_stock} Actual Price',
        line=dict(color='blue', width=2)
    ))

    # Add Prophet predictions
    fig_comparison.add_trace(go.Scatter(
        x=data1['Date'][-len(prophet_pred1):],
        y=prophet_pred1,
        mode='lines',
        name=f'{selected_stock} Prophet Prediction',
        line=dict(color='red', width=2)
    ))

    # Add LSTM predictions
    fig_comparison.add_trace(go.Scatter(
        x=data1['Date'][-len(lstm_pred1):],
        y=lstm_pred1.flatten(),
        mode='lines',
        name=f'{selected_stock} LSTM Prediction',
        line=dict(color='green', width=2)
    ))

    # Add Random Forest predictions
    fig_comparison.add_trace(go.Scatter(
        x=data1['Date'][-len(rf_pred1):],
        y=rf_pred1.flatten(),
        mode='lines',
        name=f'{selected_stock} Random Forest Prediction',
        line=dict(color='purple', width=2)
    ))

    # Add future predictions as markers
    future_date = data1['Date'].iloc[-1] + pd.Timedelta(days=1)
    
    fig_comparison.add_trace(go.Scatter(
        x=[future_date],
        y=[prophet_forecast1['yhat'].iloc[-1]],
        mode='markers+text',
        name=f'{selected_stock} Prophet Future',
        marker=dict(color='red', size=10),
        text=[f"P: {prophet_forecast1['yhat'].iloc[-1]:.2f}"],
        textposition="top center"
    ))

    fig_comparison.add_trace(go.Scatter(
        x=[future_date],
        y=[lstm_future1[0][0]],
        mode='markers+text',
        name=f'{selected_stock} LSTM Future',
        marker=dict(color='green', size=10),
        text=[f"L: {lstm_future1[0][0]:.2f}"],
        textposition="top center"
    ))

    fig_comparison.add_trace(go.Scatter(
        x=[future_date],
        y=[rf_future1[0][0]],
        mode='markers+text',
        name=f'{selected_stock} RF Future',
        marker=dict(color='purple', size=10),
        text=[f"RF: {rf_future1[0][0]:.2f}"],
        textposition="top center"
    ))

    if enable_comparison and data2 is not None:
        # Get predictions for second stock
        prophet_model2, prophet_forecast2 = prophet_forecast(data2)
        lstm_pred2, lstm_future2 = lstm_forecast(data2)
        rf_pred2, rf_future2, rf_model2 = random_forest_forecast(data2)

        # Calculate metrics for second stock
        # Prophet
        prophet_pred2 = prophet_forecast2['yhat'][-len(data2.index):]
        prophet_actual2 = data2['Close']
        prophet_mae2 = mean_absolute_error(prophet_actual2, prophet_pred2)
        prophet_rmse2 = np.sqrt(mean_squared_error(prophet_actual2, prophet_pred2))
        prophet_r2_2 = r2_score(prophet_actual2, prophet_pred2)

        # LSTM
        lstm_actual2 = data2['Close'].iloc[-len(lstm_pred2):]
        lstm_mae2 = mean_absolute_error(lstm_actual2, lstm_pred2)
        lstm_rmse2 = np.sqrt(mean_squared_error(lstm_actual2, lstm_pred2))
        lstm_r2_2 = r2_score(lstm_actual2, lstm_pred2)

        # Random Forest
        rf_actual2 = data2['Close'].iloc[-len(rf_pred2):]
        rf_mae2 = mean_absolute_error(rf_actual2, rf_pred2)
        rf_rmse2 = np.sqrt(mean_squared_error(rf_actual2, rf_pred2))
        rf_r2_2 = r2_score(rf_actual2, rf_pred2)

        # Add plots for second stock
        # Actual price
        fig_comparison.add_trace(go.Scatter(
            x=data2['Date'],
            y=data2['Close'],
            mode='lines',
            name=f'{stock2} Actual Price',
            line=dict(color='lightblue', width=2)
        ))

        # Prophet predictions
        fig_comparison.add_trace(go.Scatter(
            x=data2['Date'][-len(prophet_pred2):],
            y=prophet_pred2,
            mode='lines',
            name=f'{stock2} Prophet Prediction',
            line=dict(color='pink', width=2)
        ))

        # LSTM predictions
        fig_comparison.add_trace(go.Scatter(
            x=data2['Date'][-len(lstm_pred2):],
            y=lstm_pred2.flatten(),
            mode='lines',
            name=f'{stock2} LSTM Prediction',
            line=dict(color='yellow', width=2)
        ))

        # Random Forest predictions
        fig_comparison.add_trace(go.Scatter(
            x=data2['Date'][-len(rf_pred2):],
            y=rf_pred2.flatten(),
            mode='lines',
            name=f'{stock2} Random Forest Prediction',
            line=dict(color='orange', width=2)
        ))

        # Add future predictions for second stock
        future_date2 = data2['Date'].iloc[-1] + pd.Timedelta(days=1)
        
        fig_comparison.add_trace(go.Scatter(
            x=[future_date2],
            y=[prophet_forecast2['yhat'].iloc[-1]],
            mode='markers+text',
            name=f'{stock2} Prophet Future',
            marker=dict(color='pink', size=10),
            text=[f"P: {prophet_forecast2['yhat'].iloc[-1]:.2f}"],
            textposition="top center"
        ))

        fig_comparison.add_trace(go.Scatter(
            x=[future_date2],
            y=[lstm_future2[0][0]],
            mode='markers+text',
            name=f'{stock2} LSTM Future',
            marker=dict(color='yellow', size=10),
            text=[f"L: {lstm_future2[0][0]:.2f}"],
            textposition="top center"
        ))

        fig_comparison.add_trace(go.Scatter(
            x=[future_date2],
            y=[rf_future2[0][0]],
            mode='markers+text',
            name=f'{stock2} RF Future',
            marker=dict(color='orange', size=10),
            text=[f"RF: {rf_future2[0][0]:.2f}"],
            textposition="top center"
        ))

    # Update layout
    fig_comparison.update_layout(
        title="Model Predictions Comparison",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig_comparison, use_container_width=True)



    # Comparison Tables
    st.subheader("Detailed Metrics Comparison")
    
    # Table for first stock
    comparison_data1 = {
        'Model': ['Prophet', 'LSTM', 'Random Forest'],
        'MAE': [prophet_mae1, lstm_mae1, rf_mae1],
        'RMSE': [prophet_rmse1, lstm_rmse1, rf_rmse1],
        'R²': [prophet_r2_1, lstm_r2_1, rf_r2_1]
    }
    comparison_df1 = pd.DataFrame(comparison_data1)
    comparison_df1['MAE'] = comparison_df1['MAE'].round(2)
    comparison_df1['RMSE'] = comparison_df1['RMSE'].round(2)
    comparison_df1['R²'] = comparison_df1['R²'].round(4)
    
    st.write(f"### {selected_stock} Metrics")
    st.dataframe(comparison_df1)

    if enable_comparison and data2 is not None:
        # Table for second stock
        comparison_data2 = {
            'Model': ['Prophet', 'LSTM', 'Random Forest'],
            'MAE': [prophet_mae2, lstm_mae2, rf_mae2],
            'RMSE': [prophet_rmse2, lstm_rmse2, rf_rmse2],
            'R²': [prophet_r2_2, lstm_r2_2, rf_r2_2]
        }
        comparison_df2 = pd.DataFrame(comparison_data2)
        comparison_df2['MAE'] = comparison_df2['MAE'].round(2)
        comparison_df2['RMSE'] = comparison_df2['RMSE'].round(2)
        comparison_df2['R²'] = comparison_df2['R²'].round(4)
        
        st.write(f"### {stock2} Metrics")
        st.dataframe(comparison_df2)
