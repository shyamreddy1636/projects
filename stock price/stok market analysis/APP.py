import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Title of the App
st.title("Apple Stock Price Prediction using Holt-Winter's Method")

# Sidebar for User Input
st.sidebar.header("Input Parameters")
forecast_days = st.sidebar.slider("Number of Days to Forecast", min_value=1, max_value=365, value=30)

# Load data function
@st.cache_data
def load_data():
    data = pd.read_csv("B:/stok market analysis/APPLE_Data.csv", parse_dates=["Date"], index_col="Date")
    data['Close_Diff'] = data['Close'].diff()
    return data

data = load_data()

# Display Historical Data
st.write("### Historical Stock Prices")
st.line_chart(data['Close'])

# Train the model
if st.button("Train and Forecast"):
    try:
        # Train Holt-Winters Model
        model = ExponentialSmoothing(
            data['Close_Diff'].dropna(),
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )
        model_fit = model.fit()

        with open("stok_market.pkl", 'wb') as file:
            pickle.dump(model_fit, file)

        # Forecast Future Prices
        forecast_values = model_fit.forecast(steps=forecast_days)

        # Reverse Differencing to Original S
        forecast_original_scale = forecast_values.cumsum() + data['Close'].iloc[-1]

        # Create a dataframe with forecasted values
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_original = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_original_scale
        }).set_index('Date')

        # Display forecast
        st.write("### Forecasted Prices")
        st.write(forecast_original)

        # Plot historical and forecasted data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Close'], label="Historical data", color="blue")
        ax.plot(forecast_original.index, forecast_original['Forecast'], label="Forecasted Data", color="orange")
        ax.set_title("Apple Stock Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        st.pyplot(fig)

        # Evaluate the Model
        mae = mean_absolute_error(data['Close_Diff'].dropna(), model_fit.fittedvalues)
        mse = mean_squared_error(data['Close_Diff'].dropna(), model_fit.fittedvalues)
        rmse = np.sqrt(mse)

        st.write("### Model Evaluation Metrics")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    except Exception as e:
        st.error(f"Error: {e}")
