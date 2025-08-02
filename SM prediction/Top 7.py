# stock_predictor.py

import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Configuration ---
# Your API Key placeholder:
# For yfinance, an API key is generally NOT required for historical data.
# However, if you switch to other providers like Alpha Vantage, Finnhub, or Twelve Data
# for more extensive or real-time data, you would insert your key here.
API_KEY = "FGPN4DT5XBKSV94Z"  # Replace with your actual API key if using a different API

# List of top NSE Stock Tickers to analyze (based on general market cap/prominence)
# You can customize this list with any valid NSE tickers (e.g., "TCS.NS", "HDFCBANK.NS")
TOP_NSE_TICKERS = [
    "RELIANCE.NS",  # Reliance Industries
    "TCS.NS",  # Tata Consultancy Services
    "HDFCBANK.NS",  # HDFC Bank
    "INFY.NS",  # Infosys
    "ICICIBANK.NS",  # ICICI Bank
    "BHARTIARTL.NS",  # Bharti Airtel
    "ITC.NS",  # ITC
    "LT.NS",  # Larsen & Toubro
    "HINDUNILVR.NS"  # Hindustan Unilever
]

# Number of historical days to fetch for analysis
HISTORY_DAYS = 365  # Increased for better ML training data

# Number of past days to use as features for prediction
LAG_DAYS = 5


# --- Functions for Technical Indicators ---

def calculate_sma(data, period):
    """Calculates Simple Moving Average (SMA) for a given period."""
    return data['Close'].rolling(window=period).mean()


def calculate_ema(data, period):
    """Calculates Exponential Moving Average (EMA) for a given period."""
    return data['Close'].ewm(span=period, adjust=False).mean()


def calculate_rsi(data, period=14):
    """Calculates Relative Strength Index (RSI) using EMA for smoothing."""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use EMA for smoothing average gain and loss, as is standard for RSI
    # Corrected 'adjust-False' to 'adjust=False'
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --- Main Prediction Logic for a Single Stock ---

def get_stock_prediction(stock_ticker):
    """
    Fetches data, trains a model, and returns prediction details for a single stock.
    Returns None if an error occurs.
    """
    print(f"\n--- Processing {stock_ticker} ---")

    end_date = datetime.date.today()
    # Fetch extra data for indicator warm-up and lagged features
    start_date = end_date - datetime.timedelta(days=HISTORY_DAYS + max(20, 14) + LAG_DAYS + 30)

    try:
        df = yf.download(stock_ticker, start=start_date, end=end_date, progress=False)  # Suppress progress bar

        if df.empty:
            print(f"Error: No data fetched for {stock_ticker}. Check ticker symbol or date range.")
            return None

        # Ensure 'Close' column is float before calculating indicators
        df['Close'] = df['Close'].astype(float)

        # Calculate Indicators
        df['SMA_20'] = calculate_sma(df, 20)
        df['EMA_10'] = calculate_ema(df, 10)
        df['RSI_14'] = calculate_rsi(df, 14)

        # --- Feature Engineering for Machine Learning ---
        for i in range(1, LAG_DAYS + 1):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df['Target'] = df['Close'].shift(-1)
        df_ml = df.dropna()

        if df_ml.empty:
            print(
                f"Error for {stock_ticker}: Not enough data after dropping NaN values for machine learning. Try increasing HISTORY_DAYS.")
            return None

        features = [f'Close_Lag_{i}' for i in range(1, LAG_DAYS + 1)] + ['SMA_20', 'EMA_10', 'RSI_14']
        X = df_ml[features]
        y = df_ml['Target']

        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        X_predict = X.iloc[-1:].copy()

        if X_train.empty:
            print(
                f"Error for {stock_ticker}: Not enough data to train the model. Ensure enough historical data is fetched.")
            return None

        # --- Train the Linear Regression Model ---
        model = LinearRegression()
        model.fit(X_train, y_train)

        # --- Make a Prediction for the Next Day ---
        predicted_next_close_array = model.predict(X_predict)
        predicted_next_close = round(predicted_next_close_array[0], 2)

        latest_actual_close = df.iloc[-1]['Close'].item()

        predicted_change_percentage = ((predicted_next_close - latest_actual_close) / latest_actual_close) * 100

        # Get full name of the stock
        try:
            info = yf.Ticker(stock_ticker).info
            stock_name = info.get('longName', stock_ticker)
        except Exception:
            stock_name = stock_ticker  # Fallback to ticker if name not found

        return {
            "ticker": stock_ticker,
            "name": stock_name,
            "latest_close": latest_actual_close,
            "predicted_next_close": predicted_next_close,
            "predicted_change_percentage": predicted_change_percentage
        }

    except Exception as e:
        print(f"\nAn error occurred for {stock_ticker}: {e}")
        print("Detailed error:", e)
        print(
            f"Please ensure internet connection, correct ticker symbol, and installed packages (yfinance, pandas, numpy, scikit-learn).")
        print("You might need to increase HISTORY_DAYS.")
        return None


# --- Overall Program Execution ---

if __name__ == "__main__":
    all_predictions = []
    print("Starting prediction process for top NSE companies...\n")

    for ticker in TOP_NSE_TICKERS:
        prediction_result = get_stock_prediction(ticker)
        if prediction_result:
            all_predictions.append(prediction_result)

    print("\n" + "=" * 70)
    print("                  TOP 7 NSE STOCK PREDICTIONS (SIMULATED)")  # Updated title
    print("=" * 70)

    if not all_predictions:
        print("No predictions could be generated. Please check for errors above.")
    else:
        # Sort predictions by predicted change percentage in descending order
        all_predictions.sort(key=lambda x: x['predicted_change_percentage'], reverse=True)

        # Print the top 7
        for i, pred in enumerate(all_predictions[:7]):  # Changed from [:5] to [:7]
            arrow = '▲' if pred['predicted_change_percentage'] >= 0 else '▼'
            print(f"\n{i + 1}. {pred['name']} ({pred['ticker']})")
            print(f"   Latest Close: ₹{pred['latest_close']:.2f}")
            print(f"   Predicted Next Close: ₹{pred['predicted_next_close']:.2f}")
            print(f"   Predicted Change: {pred['predicted_change_percentage']:.2f}% {arrow}")
            print("-" * 30)

    print("\n" + "=" * 70)
    print("Disclaimer: These are simulated predictions for demonstration purposes.")
    print("            Do not use for actual financial investment decisions.")
    print("=" * 70)
