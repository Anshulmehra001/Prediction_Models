# stock_predictor.py

import yfinance as yf
import pandas as pd
import datetime
import numpy as np  # Import numpy explicitly for random operations

# --- Configuration ---
# Your API Key placeholder:
# For yfinance, an API key is generally NOT required for historical data.
# However, if you switch to other providers like Alpha Vantage, Finnhub, or Twelve Data
# for more extensive or real-time data, you would insert your key here.
API_KEY = "FGPN4DT5XBKSV94Z"  # Replace with your actual API key if using a different API

# NSE Stock Ticker to analyze (e.g., Reliance Industries on NSE)
# Common suffixes for NSE stocks on Yahoo Finance are ".NS" or ".BO" for BSE.
# You can search for a stock on finance.yahoo.com and find its ticker.
STOCK_TICKER = "RELIANCE.NS"  # Example: Reliance Industries on NSE

# Number of historical days to fetch for analysis
HISTORY_DAYS = 100


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
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    # Handle division by zero for rs where avg_loss is 0 (no losses in the period)
    # If avg_loss is 0, rs becomes infinity, then rsi becomes 100.
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --- Prediction Logic (Illustrative - NOT for financial advice) ---

def predict_next_day_close(last_close, sma_20, ema_10, rsi_14):
    """
    Simulates a next-day closing price prediction based on indicators.
    This is a very basic, rule-based approach for demonstration.
    DO NOT use for actual trading.
    """
    prediction_bias = 0.0  # Initialize bias

    # Defensive conversion: Ensure indicators are scalar floats or np.nan
    # This prevents 'truth value of a Series is ambiguous' if a Series slips through
    if isinstance(rsi_14, pd.Series) or isinstance(rsi_14, np.ndarray):
        rsi_14 = rsi_14.item() if len(rsi_14) == 1 else np.nan # Get scalar, or nan if multiple/empty
    if isinstance(ema_10, pd.Series) or isinstance(ema_10, np.ndarray):
        ema_10 = ema_10.item() if len(ema_10) == 1 else np.nan
    if isinstance(sma_20, pd.Series) or isinstance(sma_20, np.ndarray):
        sma_20 = sma_20.item() if len(sma_20) == 1 else np.nan


    # Rule 1: RSI signals
    if pd.notna(rsi_14):  # rsi_14 is now guaranteed to be a scalar or np.nan
        if rsi_14 < 30:
            prediction_bias += 0.010
        elif rsi_14 > 70:
            prediction_bias -= 0.008

    # Rule 2: Moving Average Crossover (Bullish/Bearish Signal)
    if pd.notna(ema_10) and pd.notna(sma_20):  # ema_10 and sma_20 are now guaranteed scalars or np.nan
        if ema_10 > sma_20 and last_close > ema_10:  # Compare scalar values
            prediction_bias += 0.005
        elif ema_10 < sma_20 and last_close < ema_10:  # Compare scalar values
            prediction_bias -= 0.003

    # Add a small random component to simulate market noise
    prediction_bias += (0.005 * (2 * (np.random.random()) - 1))  # Use numpy's random

    predicted_close = last_close * (1 + prediction_bias)
    return round(predicted_close, 2)


# --- Main Program Execution ---

def run_prediction_project():
    print(f"Fetching historical data for {STOCK_TICKER}...\n")

    end_date = datetime.date.today()
    # Fetch extra data for indicator warm-up. RSI/EMA needs prior data to be meaningful.
    # A 20-day SMA/EMA needs 20 days. A 14-day RSI needs 14 days,
    # plus previous data for the EMA smoothing within RSI.
    # Let's ensure we have at least 30 days more than the largest period (20 for SMA/EMA).
    start_date = end_date - datetime.timedelta(days=HISTORY_DAYS + max(20, 14) + 10)

    try:
        # Fetch data using yfinance
        # auto_adjust=False is good for explicit control, but can be left as default True
        df = yf.download(STOCK_TICKER, start=start_date, end=end_date)

        if df.empty:
            print(f"Error: No data fetched for {STOCK_TICKER}. Check ticker symbol or date range.")
            return

        print(f"Successfully fetched {len(df)} days of historical data.")
        print("Last 5 days of historical data:")
        # Select relevant columns, and ensure they are floats if not already
        print(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail().astype(float))
        print("-" * 50)

        # Calculate Indicators
        # Ensure 'Close' column is float before calculating indicators
        df['Close'] = df['Close'].astype(float)
        df['SMA_20'] = calculate_sma(df, 20)
        df['EMA_10'] = calculate_ema(df, 10)
        df['RSI_14'] = calculate_rsi(df, 14)

        # Get the latest data point with calculated indicators
        latest_data_with_indicators = df.iloc[-1]

        # Extract latest indicator values and ensure they are scalar for formatting and comparison
        latest_close = latest_data_with_indicators['Close'].item()  # Use .item() to get scalar

        # Using .item() to extract scalar values. If NaN, it will remain np.nan.
        # This handles cases where insufficient data for indicator calculation results in NaN.
        latest_sma_20 = latest_data_with_indicators['SMA_20'].item() if pd.notna(
            latest_data_with_indicators['SMA_20']) else np.nan
        latest_ema_10 = latest_data_with_indicators['EMA_10'].item() if pd.notna(
            latest_data_with_indicators['EMA_10']) else np.nan
        latest_rsi_14 = latest_data_with_indicators['RSI_14'].item() if pd.notna(
            latest_data_with_indicators['RSI_14']) else np.nan

        print(f"\nLatest Data for {STOCK_TICKER} ({latest_data_with_indicators.name.strftime('%Y-%m-%d')}):")
        # Ensure formatting only happens if the value is not NaN
        print(f"  Last Close: ₹{latest_close:.2f}")
        print(f"  SMA(20): ₹{latest_sma_20:.2f}" if pd.notna(
            latest_sma_20) else "  SMA(20): N/A (not enough data for period)")
        print(f"  EMA(10): ₹{latest_ema_10:.2f}" if pd.notna(
            latest_ema_10) else "  EMA(10): N/A (not enough data for period)")
        print(f"  RSI(14): {latest_rsi_14:.2f}" if pd.notna(
            latest_rsi_14) else "  RSI(14): N/A (not enough data for period)")
        print("-" * 50)

        # Perform the "prediction"
        # Check if all relevant latest indicators are NOT NaN before predicting
        if pd.notna(latest_sma_20) and pd.notna(latest_ema_10) and pd.notna(latest_rsi_14):
            predicted_next_close = predict_next_day_close(
                latest_close,  # Pass the scalar directly
                latest_sma_20,  # These are now guaranteed scalars or np.nan
                latest_ema_10,  # These are now guaranteed scalars or np.nan
                latest_rsi_14  # These are now guaranteed scalars or np.nan
            )
            predicted_date = (latest_data_with_indicators.name + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"\nSimulated Prediction for {predicted_date}:")
            print(f"  Predicted Next Day Close: ₹{predicted_next_close:.2f}")
            predicted_change_percentage = ((predicted_next_close - latest_close) / latest_close) * 100
            print(
                f"  Predicted Change: {predicted_change_percentage:.2f}% {'▲' if predicted_change_percentage >= 0 else '▼'}")
        else:
            print("\nSkipping prediction: Not enough historical data to calculate all indicators meaningfully.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Detailed error:", e)  # Print the full error for more context
        print("Please ensure you have an active internet connection and the ticker symbol is correct.")
        print("Also check if the 'Close' column in the fetched data is numeric.")


if __name__ == "__main__":
    run_prediction_project()
