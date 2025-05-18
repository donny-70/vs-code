# pip install streamlit prophet yfinance plotly
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from datetime import date
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

# Set page config for better appearance
st.set_page_config(page_title="Stock Forecast App", layout="wide")

# Define date range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title
st.title('Stock Forecast App')

# Stock selection
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Prediction period slider
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Cache data loading
@st.cache_data
def load_data(ticker):
    try:
        # Download data with error handling
        data = yf.download(ticker, START, TODAY, progress=False)
        if data.empty:
            raise ValueError(f"No data retrieved for ticker {ticker}.")
        
        # Handle multi-level columns (yfinance sometimes returns these)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)  # Flatten to top level
            # Rename columns to ensure consistency
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # Reset index to make 'Date' a column
        data.reset_index(inplace=True)
        
        # Ensure 'Date' is datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # Check if 'close' column exists
        if 'close' not in data.columns:
            raise ValueError(f"'close' column not found in data. Available columns: {list(data.columns)}")
        
        # Inspect 'close' column before conversion
        if data['close'].dtype == object:
            st.warning(f"'close' column contains object type. Sample: {data['close'].head().to_list()}")
        
        # Convert 'close' to numeric
        data['close'] = pd.to_numeric(data['close'], errors='coerce')
        
        # Check for all NaN in 'close'
        if data['close'].isna().all():
            raise ValueError(f"All 'close' values are invalid for ticker {ticker}.")
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Data sample (if available):", data.head() if 'data' in locals() else "No data loaded")
        return None

# Load data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)

if data is None:
    data_load_state.text("Failed to load data!")
    st.stop()

data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['open'], name="stock_open", line=dict(color="#00ff00")))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['close'], name="stock_close", line=dict(color="#ff0000")))
    fig.update_layout(
        title_text='Time Series Data with Rangeslider',
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# Prepare data for Prophet
df_train = data[['Date', 'close']].rename(columns={"Date": "ds", "close": "y"})

# Validate data before processing
if df_train.empty:
    st.error("Error: Training data is empty.")
    st.stop()

# Ensure 'ds' is datetime
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')

# Ensure 'y' is numeric (should already be handled in load_data)
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

# Check for invalid or missing data
if df_train['ds'].isnull().any():
    st.error("Error: Some dates could not be converted to datetime.")
    st.write("Invalid rows:", df_train[df_train['ds'].isnull()])
    st.stop()
if df_train['y'].isnull().any():
    st.warning("Warning: Some 'close' prices are missing or invalid. Dropping invalid rows.")
    df_train = df_train.dropna(subset=['y'])
if len(df_train) < 2:
    st.error("Error: Not enough valid data points to fit the model.")
    st.stop()

# Log data types for debugging
st.write("Data types:", df_train.dtypes)

# Train Prophet model
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
try:
    m.fit(df_train)
except Exception as e:
    st.error(f"Error fitting Prophet model: {str(e)}")
    st.stop()

# Make future predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast Data')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot forecast
st.write(f'Forecast Plot for {n_years} Years')
fig1 = plot_plotly(m, forecast)
fig1.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig1, use_container_width=True)

# Plot forecast components
st.subheader('Forecast Components')
fig2 = plot_components_plotly(m, forecast)
fig2.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig2, use_container_width=True)