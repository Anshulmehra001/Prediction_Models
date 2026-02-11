
import os

import warnings
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import mplfinance as mpf  # New import for candlestick charts

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import models and tools
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from skopt import gp_minimize
from skopt.space import Integer, Real
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

# Attempt to import Attention layer, providing a fallback
try:
    from tensorflow.keras.layers import Attention
except ImportError:
    from tensorflow.keras.layers import AdditiveAttention as Attention

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration for NewsAPI (Loaded from .env file) ---
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')  # Load from environment
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

# --- GLOBAL CONSTANTS ---
REQUIRED_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']


# --- Data Loading and Preprocessing ---
def load_data_ohlcv(symbol: str, days: int = 720) -> tuple[pd.DataFrame, ARIMA]:
    """
    Loads historical stock OHLCV data, calculates technical indicators,
    and fits an ARIMA model to the 'Close' price.
    It also calculates daily changes for OHLCV to be used as targets.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL', 'TCS.NS').
        days (int): The number of historical days to download.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume',
                                 'Close_Res' (ARIMA residual), 'EMA10', 'RSI', 'MACD',
                                 and daily change columns for OHLCV.
            - arima (statsmodels.tsa.arima.model.ARIMAResultsWrapper): Fitted ARIMA model for Close.

    Raises:
        ValueError: If no data is found, required columns are missing, or data is all null.
    """
    print(f"Attempting to download data for {symbol} for {days} days...")
    df = yf.download(symbol, period=f"{days}d", auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data found for '{symbol}'. Please verify the ticker or period.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
    df.columns = [col.replace(' ', '_').title() for col in df.columns]

    df = df.rename(columns={'Adj_Close': 'Close'}, errors='ignore')

    if not all(col in df.columns for col in REQUIRED_OHLCV_COLS):
        missing = [col for col in REQUIRED_OHLCV_COLS if col not in df.columns]
        raise ValueError(f"Missing required columns for '{symbol}': {missing}. Available: {df.columns.tolist()}")

    df.index = pd.to_datetime(df.index)
    df = df.asfreq('B', method='pad')

    for col in REQUIRED_OHLCV_COLS:
        if df[col].isnull().all():
            raise ValueError(f"No valid '{col}' data for '{symbol}' after reindexing (all are null). "
                             "Try a longer period or a different symbol.")
        if df[col].isnull().any():
            initial_nan_count = df[col].isnull().sum()
            df[col] = df[col].fillna(method='ffill')
            df.dropna(subset=[col], inplace=True)
            if initial_nan_count > 0 and df[col].isnull().sum() == 0:
                print(f"Filled {initial_nan_count} NaNs in '{col}'.")
            elif df[col].isnull().sum() > 0:
                print(f"Warning: Still {df[col].isnull().sum()} NaNs in '{col}' after ffill and dropna.")

    try:
        if len(df) < 10:
            raise ValueError("Insufficient data points for ARIMA fitting after preprocessing.")
        arima = ARIMA(df["Close"], order=(5, 1, 0)).fit()
        df["Close_Res"] = arima.resid
        print("ARIMA model for Close price fitted successfully.")
    except Exception as e:
        print(f"Warning: ARIMA model fitting for Close price failed ({e}). "
              "Using simple difference for Close residual as fallback.")
        df["Close_Res"] = df["Close"].diff().fillna(0)

    for col in REQUIRED_OHLCV_COLS:
        df[f"{col}_Change"] = df[col] - df[col].shift(1)

    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    initial_rows = len(df)
    df.dropna(inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows due to NaN values after calculations (common).")
    if df.empty:
        raise ValueError(f"No valid data remaining for '{symbol}' after dropping NaNs. Try a longer period.")

    return df, arima


# --- Sentiment Analysis (Automated) ---
nlp_sent = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")


def get_automated_sentiment(symbol: str, query_days: int = 7) -> float:
    """
    Fetches recent news headlines for a given stock symbol using NewsAPI.org
    and calculates an aggregated sentiment score.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        print("Warning: NEWS_API_KEY not set or is default. Cannot fetch automated news. Defaulting sentiment to 0.0.")
        print("Please obtain a free API key from https://newsapi.org/ and replace 'YOUR_NEWS_API_KEY' in the code.")
        return 0.0

    print(f"Fetching recent news headlines for '{symbol}' from the last {query_days} days...")

    from_date = (datetime.now() - timedelta(days=query_days)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    params = {
        'q': f'"{symbol}" stock OR "{symbol}" company',
        'language': 'en',
        'sortBy': 'relevancy',
        'from': from_date,
        'to': to_date,
        'pageSize': 10,
        'apiKey': NEWS_API_KEY
    }

    headers = {'User-Agent': 'Stock-Forecaster/1.0'}

    try:
        response = requests.get(NEWS_API_BASE_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        news_data = response.json()

        headlines = [article['title'] for article in news_data.get('articles', []) if article.get('title')]

        if not headlines:
            print(f"No relevant news headlines found for '{symbol}' in the last {query_days} days.")
            return 0.0

        sentiment_scores = []
        for headline in headlines:
            res = nlp_sent(headline)[0]
            if res["label"].upper() == "POSITIVE":
                sentiment_scores.append(res["score"])
            elif res["label"].upper() == "NEGATIVE":
                sentiment_scores.append(-res["score"])
            else:
                sentiment_scores.append(0.0)

        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        print(f"Aggregated sentiment from {len(headlines)} headlines: {avg_sentiment:.4f}")
        return avg_sentiment

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching news: {http_err}")
        print(f"NewsAPI response: {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred while fetching news: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout occurred while fetching news: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected request error occurred while fetching news: {req_err}")
    except Exception as e:
        print(f"An error occurred during automated news sentiment analysis: {e}")

    return 0.0


# --- XGBoost Model Tuning ---
def tune_xgb(X: np.ndarray, y: np.ndarray, n_outputs: int) -> tuple:
    """
    Tunes XGBoost Regressor hyperparameters using Bayesian optimization and TimeSeriesSplit.
    Modified to handle multiple outputs.
    """

    def objective(params: list) -> float:
        ne, md, lr = params
        model = XGBRegressor(n_estimators=int(ne), max_depth=int(md), learning_rate=lr,
                             random_state=42, n_jobs=-1, verbosity=0)
        errs = []
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if len(X_train) == 0 or len(X_val) == 0:
                continue

            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            errs.append(mean_absolute_error(y_val, pred))

        if not errs:
            return 1e10

        return np.mean(errs)

    space = [
        Integer(50, 500, name="ne"),
        Integer(3, 15, name="md"),
        Real(0.005, 0.5, prior="log-uniform", name="lr")
    ]

    print("Tuning XGBoost hyperparameters (this may take a moment, ~15 calls)...")
    sol = gp_minimize(objective, space, n_calls=15, random_state=42, verbose=False)

    optimal_n_estimators, optimal_max_depth, optimal_learning_rate = sol.x
    print(f"XGBoost optimal hyperparameters: "
          f"n_estimators={optimal_n_estimators}, "
          f"max_depth={optimal_max_depth}, "
          f"learning_rate={optimal_learning_rate:.4f}")
    return optimal_n_estimators, optimal_max_depth, optimal_learning_rate


# --- Deep Learning Model ---
def build_model_ohlcv(seq_len: int, feat_dim: int, output_dim: int) -> Model:
    """
    Builds a CNN-LSTM model with an Attention mechanism for multivariate sequence prediction.
    """
    i = Input(shape=(seq_len, feat_dim))

    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="causal")(i)
    x = Dropout(0.2)(x)

    x = LSTM(64, return_sequences=True)(x)

    a = Attention()([x, x])

    x = LSTM(32)(a)
    out = Dense(output_dim)(x)

    model = Model(inputs=i, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    print("CNN-LSTM-Attention model built successfully for OHLCV prediction.")
    return model


# --- Hybrid Forecasting Logic ---
def hybrid_forecast_ohlcv(seq_len: int = 60, future: int = 5, epochs: int = 20, batch_size: int = 16) -> tuple[
    pd.DataFrame, pd.DataFrame, str]:
    """
    Performs hybrid stock OHLCV forecasting.
    Returns: forecasted_df, historical_df_for_plotting, chosen_symbol
    """
    df_loaded = None
    arima_loaded = None
    sym_chosen = None

    while True:
        sym_chosen = input("Enter stock ticker (e.g., TCS.NS for NSE, AAPL for NASDAQ): ").strip().upper()
        try:
            df_loaded, arima_loaded = load_data_ohlcv(sym_chosen)
            break
        except ValueError as e:
            print(f"Error loading data: {e}. Please check the ticker and try again.")
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}. Please try again.")

    sentiment = get_automated_sentiment(sym_chosen)
    print(f"Automated sentiment score for {sym_chosen}: {sentiment:.4f}")

    feats = ["Open", "High", "Low", "Close", "Volume", "EMA10", "RSI", "MACD", "Close_Res"]
    target_change_cols = ["Open_Change", "High_Change", "Low_Change", "Close_Change", "Volume_Change"]

    if len(df_loaded) < (seq_len + 2):
        raise ValueError(f"Insufficient data ({len(df_loaded)} points) after preprocessing to run the model with "
                         f"seq_len={seq_len}. Try increasing `days` in `load_data_ohlcv`.")

    X_features_raw = df_loaded[feats].iloc[:-1].values
    y_target_changes_raw = df_loaded[target_change_cols].iloc[1:].values

    min_len = min(len(X_features_raw), len(y_target_changes_raw))
    X_features_aligned = X_features_raw[:min_len]
    y_target_changes_aligned = y_target_changes_raw[:min_len]

    if len(X_features_aligned) == 0:
        raise ValueError("Not enough processed data to create aligned features and targets for forecasting models. "
                         "Check data length after preprocessing and shifting.")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    Xs_scaled = scaler_X.fit_transform(X_features_aligned)
    ys_scaled = scaler_y.fit_transform(y_target_changes_aligned)

    if len(Xs_scaled) < seq_len:
        raise ValueError(f"Not enough scaled data ({len(Xs_scaled)} points) to create sequences with "
                         f"seq_len={seq_len}. Try reducing `seq_len` or increasing `days`.")

    X_seq = np.array([Xs_scaled[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(Xs_scaled))])
    y_seq = ys_scaled[seq_len - 1:]

    split_point = int(0.8 * len(X_seq))
    if split_point == 0 or split_point >= len(X_seq):
        raise ValueError("Not enough data to create valid train/validation split for deep learning model. "
                         "Adjust `seq_len` or `days` to get more data points.")

    X_train_dl, X_val_dl = X_seq[:split_point], X_seq[split_point:]
    y_train_dl, y_val_dl = y_seq[:split_point], y_seq[split_point:]

    model_dl = build_model_ohlcv(seq_len, Xs_scaled.shape[1], len(target_change_cols))
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)

    print(f"Training CNN-LSTM-Attention model for {epochs} epochs (batch size: {batch_size})...")
    model_dl.fit(X_train_dl, y_train_dl,
                 validation_data=(X_val_dl, y_val_dl),
                 epochs=epochs,
                 batch_size=batch_size,
                 callbacks=[early_stopping],
                 verbose=0)
    print("CNN-LSTM-Attention model training complete.")

    optimal_ne, optimal_md, optimal_lr = tune_xgb(Xs_scaled, ys_scaled, len(target_change_cols))

    xgb_model = XGBRegressor(n_estimators=int(optimal_ne), max_depth=int(optimal_md),
                             learning_rate=optimal_lr, random_state=42, n_jobs=-1, verbosity=0)

    print("Training XGBoost model...")
    xgb_model.fit(Xs_scaled, ys_scaled)
    print("XGBoost model training complete.")

    print(f"Generating {future}-day OHLCV forecast for {sym_chosen}...")

    last_ohlcv_actual = df_loaded[REQUIRED_OHLCV_COLS].iloc[-1].values
    last_feature_vector_actual = df_loaded[feats].iloc[-1].values

    current_seq_input = Xs_scaled[-seq_len:].reshape(1, seq_len, Xs_scaled.shape[1])

    forecasted_ohlcv_data = []

    feature_indices = {feat: feats.index(feat) for feat in feats}
    change_indices = {col: target_change_cols.index(col) for col in target_change_cols}

    current_ohlcv_base = last_ohlcv_actual.copy()

    for i in range(future):
        dl_pred_scaled_changes = model_dl.predict(current_seq_input, verbose=0)[0]
        xgb_pred_scaled_changes = xgb_model.predict(current_seq_input[0, -1].reshape(1, -1))[0]

        combined_scaled_changes = (0.5 * dl_pred_scaled_changes + 0.5 * xgb_pred_scaled_changes)

        close_change_idx = target_change_cols.index("Close_Change")
        combined_scaled_changes[close_change_idx] += (0.1 * sentiment)

        predicted_inv_changes = scaler_y.inverse_transform(combined_scaled_changes.reshape(1, -1))[0]

        next_open = current_ohlcv_base[0] + predicted_inv_changes[change_indices["Open_Change"]]
        next_high = current_ohlcv_base[1] + predicted_inv_changes[change_indices["High_Change"]]
        next_low = current_ohlcv_base[2] + predicted_inv_changes[change_indices["Low_Change"]]
        next_close = current_ohlcv_base[3] + predicted_inv_changes[change_indices["Close_Change"]]
        next_volume = current_ohlcv_base[4] + predicted_inv_changes[change_indices["Volume_Change"]]

        next_volume = max(0, next_volume)

        # Ensure OHLC relations (High >= Open, Close, Low; Low <= Open, Close, High)
        # Sort the four main price points to determine the min/max
        temp_prices = sorted([next_open, next_high, next_low, next_close])
        next_low = temp_prices[0]
        next_high = temp_prices[-1]

        # Re-assign Open and Close if they fall outside the new High/Low bounds
        next_open = np.clip(next_open, next_low, next_high)
        next_close = np.clip(next_close, next_low, next_high)

        current_day_forecast = [next_open, next_high, next_low, next_close, next_volume]
        forecasted_ohlcv_data.append(current_day_forecast)

        current_ohlcv_base = np.array(current_day_forecast)

        new_feature_vector = np.zeros(len(feats))

        new_feature_vector[feature_indices["Open"]] = current_ohlcv_base[0]
        new_feature_vector[feature_indices["High"]] = current_ohlcv_base[1]
        new_feature_vector[feature_indices["Low"]] = current_ohlcv_base[2]  # Corrected index
        new_feature_vector[feature_indices["Close"]] = current_ohlcv_base[3]  # Corrected index
        new_feature_vector[feature_indices["Volume"]] = current_ohlcv_base[4]

        # Simplification for technical indicators and Close_Res for future steps:
        # For EMA10, RSI, MACD in future steps, a simple approximation is used:
        # The model is trained on these features, so it needs *some* value.
        # For predictions, it's a simplification to carry forward the last actual value
        # or calculate a very basic moving average on the predicted data.
        # Here, we will use the predicted close to update a running EMA, and set others to last known.
        # This is still a simplification but slightly better than just using last actual.

        # A simple running EMA update based on predicted close. This is a very rough approximation.
        # Better: predict these indicators directly or develop a more complex recursion.

        # If this is the first forecast day, use last actual EMA.
        # For subsequent days, use a simple update based on the predicted close.
        if i == 0:
            new_feature_vector[feature_indices["EMA10"]] = df_loaded["EMA10"].iloc[-1]
            new_feature_vector[feature_indices["RSI"]] = df_loaded["RSI"].iloc[-1]
            new_feature_vector[feature_indices["MACD"]] = df_loaded["MACD"].iloc[-1]
        else:
            # Re-calculate simple EMA, RSI, MACD using the last predicted Close and Volume
            # This is still a very rudimentary approximation!
            prev_close = forecasted_ohlcv_data[i - 1][3] if i > 0 else df_loaded["Close"].iloc[-1]
            # EMA10 calculation (simple, not proper EWM for a single point)
            new_feature_vector[feature_indices["EMA10"]] = (new_feature_vector[feature_indices["Close"]] * (2 / 11)) + (
                        prev_close * (1 - (2 / 11)))
            # RSI and MACD are too complex to update with just one new data point and would require a history.
            # Best to either predict them as part of the output or rely on the model to learn the patterns
            # without perfectly updated indicators in the input.
            # For now, keep last known for RSI/MACD or use a constant. Let's keep last known.
            new_feature_vector[feature_indices["RSI"]] = df_loaded["RSI"].iloc[-1]
            new_feature_vector[feature_indices["MACD"]] = df_loaded["MACD"].iloc[-1]

        new_feature_vector[feature_indices["Close_Res"]] = 0  # Future ARIMA residual assumed 0

        scaled_new_feature_vector = scaler_X.transform(new_feature_vector.reshape(1, -1))

        current_seq_input = np.concatenate(
            [current_seq_input[:, 1:, :], scaled_new_feature_vector.reshape(1, 1, -1)],
            axis=1
        )

    last_known_date = df_loaded.index[-1]
    forecast_dates = pd.bdate_range(start=last_known_date + pd.Timedelta(days=1), periods=future)

    forecast_df = pd.DataFrame(
        forecasted_ohlcv_data,
        index=forecast_dates,
        columns=REQUIRED_OHLCV_COLS
    )
    forecast_df.index.name = "Date"
    forecast_df.name = sym_chosen

    return forecast_df, df_loaded, sym_chosen


# --- Visualization ---
def plot_two_charts_ohlcv(forecast_df: pd.DataFrame, historical_df: pd.DataFrame, symbol: str,
                          hist_days_to_show: int = 90):
    """
    Plots two separate charts: one for historical OHLCV and one for forecasted OHLCV.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing the forecasted OHLCV prices.
        historical_df (pd.DataFrame): Historical DataFrame (used to plot actual OHLCV prices).
        symbol (str): The stock ticker symbol.
        hist_days_to_show (int): Number of recent historical days to show on the plot.
    """
    # Ensure historical_df only includes the relevant columns and period for plotting
    historical_df_plot = historical_df[REQUIRED_OHLCV_COLS].tail(hist_days_to_show).copy()

    # --- Plot Historical Chart ---
    # Convert index to datetime if not already (mplfinance requires this)
    historical_df_plot.index = pd.to_datetime(historical_df_plot.index)

    print(f"\n--- Displaying Historical OHLCV Chart for {symbol} (Last {hist_days_to_show} Days) ---")
    mc_historical = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s_historical = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc_historical)

    fig, axes = mpf.plot(historical_df_plot,
                         type='candle',
                         style=s_historical,
                         title=f"Historical OHLCV for {symbol} (Last {hist_days_to_show} Days)",
                         ylabel='Price',
                         ylabel_lower='Volume',
                         volume=True,
                         figscale=1.5,
                         returnfig=True)

    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[2].grid(True, linestyle='--', alpha=0.6)  # Volume subplot
    plt.tight_layout()
    plt.show()

    # --- Plot Forecasted Chart ---
    # Convert index to datetime if not already
    forecast_df.index = pd.to_datetime(forecast_df.index)

    print(f"\n--- Displaying Forecasted OHLCV Chart for {symbol} (Next {len(forecast_df)} Days) ---")
    mc_forecast = mpf.make_marketcolors(up='cyan', down='magenta', inherit=True)  # Use distinct colors
    s_forecast = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc_forecast)

    fig_forecast, axes_forecast = mpf.plot(forecast_df,
                                           type='candle',
                                           style=s_forecast,
                                           title=f"Forecasted OHLCV for {symbol} (Next {len(forecast_df)} Days)",
                                           ylabel='Price',
                                           ylabel_lower='Volume',
                                           volume=True,
                                           figscale=1.5,
                                           returnfig=True)

    axes_forecast[0].grid(True, linestyle='--', alpha=0.6)
    axes_forecast[2].grid(True, linestyle='--', alpha=0.6)  # Volume subplot
    plt.tight_layout()
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Hybrid Stock OHLCV Chart Forecasting Model...")
    try:
        forecast_result_df, historical_df_for_plot, symbol_for_plot = hybrid_forecast_ohlcv(future=5)  # Forecast 5 days
        print("\n🎯 Final OHLCV Forecast:\n", forecast_result_df)

        plot_two_charts_ohlcv(forecast_result_df, historical_df_for_plot, symbol_for_plot,
                              hist_days_to_show=60)  # Show last 60 historical days


    except Exception as main_error:
        print(f"\nAn unrecoverable error occurred during forecasting: {main_error}")
        print("Please review the error message and ensure all inputs are valid. "
              "Common issues include incorrect ticker symbols, unstable internet connection, "
              "or insufficient historical data. "
              "Also, ensure your NewsAPI key is correctly set if using automated sentiment.")