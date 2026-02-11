# import warnings
# warnings.filterwarnings("ignore")
#
# import yfinance as yf
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
# from transformers import pipeline
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import TimeSeriesSplit
# from xgboost import XGBRegressor
# from skopt import gp_minimize
# from skopt.space import Integer, Real
# from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import mean_absolute_error
# import matplotlib.pyplot as plt
#
# try:
#     from tensorflow.keras.layers import Attention
# except ImportError:
#     from tensorflow.keras.layers import AdditiveAttention as Attention
#
# def load_data(symbol, days=720):
#     """
#     Loads historical stock data, calculates technical indicators,
#     and fits an ARIMA model to the 'Close' price.
#
#     Args:
#         symbol (str): The stock ticker symbol.
#         days (int): The number of historical days to download.
#
#     Returns:
#         tuple: A tuple containing:
#             - df (pd.DataFrame): DataFrame with 'Close', 'Res', 'EMA10', 'RSI', 'MACD'.
#             - arima (statsmodels.tsa.arima.model.ARIMAResultsWrapper): Fitted ARIMA model.
#
#     Raises:
#         ValueError: If no data is found, 'Close' column is missing, or 'Close' data is all null.
#     """
#     print(f"Attempting to download data for {symbol} for {days} days...")
#     df = yf.download(symbol, period=f"{days}d", auto_adjust=True, progress=False)
#
#     # --- MODIFIED SECTION START ---
#     # Flatten MultiIndex columns if present and rename them to standard names
#     if isinstance(df.columns, pd.MultiIndex):
#         original_cols = df.columns.tolist()
#         df.columns = [col[0] for col in original_cols] # Take the first level (e.g., 'Close')
#         print("Flattened MultiIndex columns to:", df.columns.tolist())
#     else:
#         # If not MultiIndex, ensure common names are used
#         df.columns = [col.replace(' ', '_') for col in df.columns]
#         print("Downloaded columns:", df.columns.tolist())
#
#     # Ensure standard column names regardless of initial format
#     # This explicit renaming handles cases where yfinance might return 'Adj Close'
#     # or other variations, ensuring we always have 'Close', 'Open', etc.
#     df = df.rename(columns={
#         'Adj Close': 'Close', # Common alternative name for Close
#         'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'
#     }, errors='ignore') # errors='ignore' prevents error if key not found
#     # --- MODIFIED SECTION END ---
#
#
#     if df.empty:
#         raise ValueError(f"No data found for '{symbol}'. Please verify the ticker.")
#     if 'Close' not in df.columns:
#         raise ValueError(f"'Close' column missing for '{symbol}'. Available columns: {df.columns.tolist()}")
#
#     # Convert index to datetime and reindex to business days
#     df.index = pd.to_datetime(df.index)
#     # Reindex to ensure continuous business days, filling missing with previous valid data
#     df = df.asfreq('B', method='pad')
#
#     # Check for nulls in 'Close' after reindexing
#     if df['Close'].isnull().all():
#         raise ValueError(f"No valid 'Close' data for '{symbol}' after reindexing (all are null).")
#     if df['Close'].isnull().any():
#         print("Warning: Some 'Close' price NaNs present after reindexing. Forward filling.")
#         df['Close'] = df['Close'].fillna(method='ffill')
#         # If NaNs at the very beginning, drop them if ffill can't fill
#         df.dropna(subset=['Close'], inplace=True)
#
#
#     # Fit ARIMA model to the 'Close' price
#     try:
#         arima = ARIMA(df["Close"], order=(5, 1, 0)).fit()
#         df["Res"] = arima.resid # ARIMA residuals
#     except Exception as e:
#         print(f"Warning: ARIMA model fitting failed, using simple difference for residuals. Error: {e}")
#         df["Res"] = df["Close"].diff().fillna(0) # Fallback for residuals
#
#     # Calculate technical indicators
#     df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
#
#     # RSI Calculation
#     delta = df["Close"].diff()
#     up = delta.clip(lower=0)
#     down = -1 * delta.clip(upper=0)
#     ma_up = up.ewm(com=13, adjust=False).mean() # Equivalent to 14-period SMA for RSI
#     ma_down = down.ewm(com=13, adjust=False).mean()
#     # Avoid division by zero
#     rs = ma_up / (ma_down + 1e-10)
#     df["RSI"] = 100 - (100 / (1 + rs))
#
#     # MACD Calculation
#     exp1 = df["Close"].ewm(span=12, adjust=False).mean()
#     exp2 = df["Close"].ewm(span=26, adjust=False).mean()
#     df["MACD"] = exp1 - exp2
#
#     # Drop any remaining NaN values that result from indicator calculations
#     initial_rows = len(df)
#     df.dropna(inplace=True)
#     if len(df) < initial_rows:
#         print(f"Dropped {initial_rows - len(df)} rows due to NaN values after indicator calculation.")
#     if df.empty:
#         raise ValueError(f"No valid data remaining for '{symbol}' after dropping NaNs. Try a longer period.")
#
#     return df, arima
#
# # The rest of your code (nlp_sent, get_sentiment, tune_xgb, build_model, hybrid_forecast, plot_forecast, __main__) remains the same.
# # Just replace your load_data function with the one above.
#
# nlp_sent = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")
#
# def get_sentiment():
#     txt = input("Enter a recent news headline for sentiment analysis (e.g., 'Company reports strong earnings'):\n")
#     if not txt.strip():
#         print("No headline entered. Defaulting to neutral sentiment (0.0).")
#         return 0.0
#     res = nlp_sent(txt)[0]
#     return res["score"] if res["label"].upper() == "POSITIVE" else -res["score"]
#
# def tune_xgb(X, y):
#     def objective(params):
#         ne, md, lr = params
#         m = XGBRegressor(n_estimators=ne, max_depth=md, learning_rate=lr,
#                          random_state=42, n_jobs=-1, verbosity=0)
#         errs = []
#         tscv = TimeSeriesSplit(n_splits=3)
#         for tr, va in tscv.split(X):
#             m.fit(X[tr], y[tr])
#             pred = m.predict(X[va])
#             errs.append(mean_absolute_error(y[va], pred))
#         return np.mean(errs)
#
#     space = [
#         Integer(50, 300, name="ne"),
#         Integer(3, 10, name="md"),
#         Real(0.01, 0.3, prior="log-uniform", name="lr")
#     ]
#     print("Tuning XGBoost hyperparameters (this may take a moment)...")
#     sol = gp_minimize(objective, space, n_calls=15, random_state=42, verbose=False)
#     print(f"XGBoost optimal hyperparameters: n_estimators={sol.x[0]}, max_depth={sol.x[1]}, learning_rate={sol.x[2]:.4f}")
#     return sol.x
#
# def build_model(seq_len, feat_dim):
#     i = Input((seq_len, feat_dim))
#     x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="causal")(i)
#     x = Dropout(0.2)(x)
#     x = LSTM(64, return_sequences=True)(x)
#     a = Attention()([x, x])
#     x = LSTM(32)(a)
#     out = Dense(1)(x)
#     m = Model(i, out)
#     m.compile(optimizer="adam", loss="mse")
#     return m
#
# def hybrid_forecast(seq_len=60, future=5, epochs=15, batch_size=16):
#     while True:
#         sym = input("Enter stock ticker (e.g., TCS.NS for NSE, AAPL for NASDAQ): ").strip().upper()
#         try:
#             df, arima = load_data(sym)
#             break
#         except ValueError as e:
#             print(f"Error: {e}. Please try again.")
#
#     sentiment = get_sentiment()
#     print(f"Sentiment score: {sentiment:.4f}")
#
#     feats = ["EMA10", "RSI", "MACD", "Res"]
#     X = df[feats].values[:-1]
#     y = df["Res"].shift(-1).dropna().values
#
#     try:
#         res_idx = feats.index("Res")
#     except ValueError:
#         raise ValueError("'Res' feature not found in the list of features. Please ensure 'Res' is in 'feats'.")
#
#     scaler_X = MinMaxScaler()
#     scaler_y = MinMaxScaler()
#     Xs = scaler_X.fit_transform(X)
#     ys = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
#
#     if len(Xs) < seq_len:
#         raise ValueError(f"Not enough data to create sequences with seq_len={seq_len}. "
#                          f"Downloaded data length: {len(Xs)}. Try reducing seq_len or increasing 'days'.")
#
#     X_seq = np.array([Xs[i-seq_len+1:i+1] for i in range(seq_len-1, len(Xs))])
#     y_seq = ys[seq_len-1:]
#
#     split = int(0.8 * len(X_seq))
#     if split == 0 or split >= len(X_seq):
#         raise ValueError("Not enough data to create valid train/validation split for deep learning model. "
#                          "Adjust seq_len or 'days'.")
#
#     print("Building and training CNN-LSTM-Attention model...")
#     model = build_model(seq_len, Xs.shape[1])
#     early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
#     model.fit(X_seq[:split], y_seq[:split],
#               validation_data=(X_seq[split:], y_seq[split:]),
#               epochs=epochs,
#               batch_size=batch_size,
#               callbacks=[early_stopping],
#               verbose=0)
#     print("CNN-LSTM-Attention model training complete.")
#
#     ne, md, lr = tune_xgb(Xs, ys)
#     xgb = XGBRegressor(n_estimators=int(ne), max_depth=int(md), learning_rate=lr,
#                        random_state=42, n_jobs=-1, verbosity=0)
#     xgb.fit(Xs, ys)
#
#     print(f"Generating {future}-day forecast...")
#     seq = Xs[-seq_len:].reshape(1, seq_len, Xs.shape[1])
#     preds_scaled_residuals = []
#
#     for _ in range(future):
#         p1_scaled = float(model.predict(seq, verbose=0)[0, 0])
#         p2_scaled = float(xgb.predict(seq[0, -1].reshape(1, -1))[0])
#
#         combined_scaled_residual = 0.5 * p1_scaled + 0.5 * p2_scaled + 0.1 * sentiment
#         preds_scaled_residuals.append(combined_scaled_residual)
#
#         nxt_feature_vector = seq[0, -1].copy()
#         nxt_feature_vector[res_idx] = combined_scaled_residual
#
#         seq = np.concatenate([seq[:, 1:, :], nxt_feature_vector.reshape(1, 1, -1)], axis=1)
#
#     preds_inv_residuals = scaler_y.inverse_transform(np.array(preds_scaled_residuals).reshape(-1, 1)).flatten()
#     arima_pred_base = arima.forecast(steps=future)
#     final_forecast_prices = arima_pred_base + preds_inv_residuals
#
#     last_known_date = df.index[-1]
#     forecast_dates = pd.bdate_range(start=last_known_date + pd.Timedelta(days=1), periods=future)
#
#     forecast_series = pd.Series(final_forecast_prices, index=forecast_dates, name=sym)
#     return forecast_series
#
# def plot_forecast(f):
#     plt.figure(figsize=(12, 6))
#     plt.plot(f.index, f.values, marker="o", linestyle='-', color="purple", linewidth=2)
#     plt.title(f"{f.name} {len(f)}-Day Hybrid Price Forecast", fontsize=16)
#     plt.xlabel("Date", fontsize=12)
#     plt.ylabel("Price", fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()
#
# if __name__ == "__main__":
#     print("Starting Hybrid Stock Price Forecasting Model...")
#     try:
#         fc = hybrid_forecast()
#         print("\n🎯 Final Forecast:\n", fc)
#         plot_forecast(fc)
#     except Exception as e:
#         print(f"\nAn error occurred during forecasting: {e}")
#         print("Please ensure your internet connection is stable and the ticker symbol is correct.")


import os

import warnings
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests  # New import for making HTTP requests to news API

# Suppress warnings for cleaner output, but be aware of what's being suppressed
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

# Attempt to import Attention layer, providing a fallback for older TensorFlow versions
try:
    from tensorflow.keras.layers import Attention
except ImportError:
    from tensorflow.keras.layers import AdditiveAttention as Attention

# Set random seeds for reproducibility across runs
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration for NewsAPI (Loaded from .env file) ---
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')  # Load from environment
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"


# --- Data Loading and Preprocessing ---
def load_data(symbol: str, days: int = 720) -> tuple[pd.DataFrame, ARIMA]:
    """
    Loads historical stock data, calculates technical indicators,
    and fits an ARIMA model to the 'Close' price.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL', 'TCS.NS').
        days (int): The number of historical days to download.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): DataFrame with 'Close', 'Res', 'EMA10', 'RSI', 'MACD'.
            - arima (statsmodels.tsa.arima.model.ARIMAResultsWrapper): Fitted ARIMA model.

    Raises:
        ValueError: If no data is found, 'Close' column is missing, or 'Close' data is all null.
    """
    print(f"Attempting to download data for {symbol} for {days} days...")
    df = yf.download(symbol, period=f"{days}d", auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data found for '{symbol}'. Please verify the ticker or period.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
        print(f"Flattened MultiIndex columns to: {df.columns.tolist()}")
    else:
        df.columns = [col.replace(' ', '_').title() for col in df.columns]
        print(f"Downloaded columns (standardized): {df.columns.tolist()}")

    df = df.rename(columns={
        'Adj Close': 'Close',
        'Gmt Offset': 'Gmt_Offset'
    }, errors='ignore')

    if 'Close' not in df.columns:
        raise ValueError(f"'Close' column missing for '{symbol}'. Available columns: {df.columns.tolist()}")

    df.index = pd.to_datetime(df.index)
    df = df.asfreq('B', method='pad')

    if df['Close'].isnull().all():
        raise ValueError(f"No valid 'Close' data for '{symbol}' after reindexing (all are null). "
                         "Try a longer period or a different symbol.")
    if df['Close'].isnull().any():
        initial_close_nan_count = df['Close'].isnull().sum()
        df['Close'] = df['Close'].fillna(method='ffill')
        df.dropna(subset=['Close'], inplace=True)
        if initial_close_nan_count > 0 and df['Close'].isnull().sum() == 0:
            print(f"Filled {initial_close_nan_count} NaNs in 'Close' and dropped any leading NaNs.")
        elif df['Close'].isnull().sum() > 0:
            print(f"Warning: Still {df['Close'].isnull().sum()} NaNs in 'Close' after ffill and dropna. "
                  "This might indicate insufficient historical data.")

    try:
        if len(df) < 10:
            raise ValueError("Insufficient data points for ARIMA fitting after preprocessing.")
        arima = ARIMA(df["Close"], order=(5, 1, 0)).fit()
        df["Res"] = arima.resid
        print("ARIMA model fitted successfully.")
    except Exception as e:
        print(f"Warning: ARIMA model fitting failed ({e}). Using simple difference for residuals as fallback.")
        df["Res"] = df["Close"].diff().fillna(0)

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
        print(f"Dropped {initial_rows - len(df)} rows due to NaN values after indicator calculation (common).")
    if df.empty:
        raise ValueError(f"No valid data remaining for '{symbol}' after dropping NaNs. Try a longer period.")

    return df, arima


# --- Sentiment Analysis (Automated) ---
nlp_sent = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")


def get_automated_sentiment(symbol: str, query_days: int = 7) -> float:
    """
    Fetches recent news headlines for a given stock symbol using NewsAPI.org
    and calculates an aggregated sentiment score.

    Args:
        symbol (str): The stock ticker symbol.
        query_days (int): How many days back to search for news.

    Returns:
        float: Aggregated sentiment score (average of individual headline scores).
               Returns 0.0 if no news is found or API key is missing/invalid.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        print("Warning: NEWS_API_KEY not set or is default. Cannot fetch automated news. Defaulting sentiment to 0.0.")
        print("Please obtain a free API key from https://newsapi.org/ and replace 'YOUR_NEWS_API_KEY' in the code.")
        return 0.0

    print(f"Fetching recent news headlines for '{symbol}' from the last {query_days} days...")

    # Calculate date range for news query
    from_date = (pd.Timestamp.now() - pd.Timedelta(days=query_days)).strftime('%Y-%m-%d')
    to_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    params = {
        'q': f'"{symbol}" stock OR "{symbol}" company',  # Search for the symbol or company name
        'language': 'en',
        'sortBy': 'relevancy',
        'from': from_date,
        'to': to_date,
        'pageSize': 10,  # Get up to 10 recent headlines
        'apiKey': NEWS_API_KEY
    }

    headers = {
        'User-Agent': 'Stock-Forecaster/1.0'  # Good practice to identify your application
    }

    try:
        response = requests.get(NEWS_API_BASE_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
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
            # Neutral and other tones (analytical, confident, tentative, joy, fear, anger, sadness)
            # are treated as 0 for simplicity, focusing on positive/negative impact.
            else:
                sentiment_scores.append(0.0)

        # Return the average sentiment score
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

    return 0.0  # Return 0.0 in case of any error


# --- XGBoost Model Tuning ---
def tune_xgb(X: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    """
    Tunes XGBoost Regressor hyperparameters using Bayesian optimization and TimeSeriesSplit.

    Args:
        X (np.ndarray): Features for XGBoost.
        y (np.ndarray): Target for XGBoost (residuals).

    Returns:
        tuple: Optimal hyperparameters (n_estimators, max_depth, learning_rate).
    """

    def objective(params: list) -> float:
        """Objective function to minimize for Bayesian optimization."""
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
def build_model(seq_len: int, feat_dim: int) -> Model:
    """
    Builds a CNN-LSTM model with an Attention mechanism for sequence prediction.

    Args:
        seq_len (int): Length of the input sequences (time steps).
        feat_dim (int): Number of features per time step.

    Returns:
        tensorflow.keras.models.Model: Compiled Keras model.
    """
    i = Input(shape=(seq_len, feat_dim))

    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="causal")(i)
    x = Dropout(0.2)(x)

    x = LSTM(64, return_sequences=True)(x)

    a = Attention()([x, x])

    x = LSTM(32)(a)
    out = Dense(1)(x)

    model = Model(inputs=i, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    print("CNN-LSTM-Attention model built successfully.")
    # model.summary() # Uncomment to see model summary during execution
    return model


# --- Hybrid Forecasting Logic ---
def hybrid_forecast(seq_len: int = 60, future: int = 5, epochs: int = 20, batch_size: int = 16) -> pd.Series:
    """
    Performs hybrid stock price forecasting combining ARIMA, CNN-LSTM with Attention,
    XGBoost, and sentiment analysis.

    Args:
        seq_len (int): Length of historical sequence to consider for deep learning.
        future (int): Number of days to forecast into the future.
        epochs (int): Number of epochs for deep learning model training.
        batch_size (int): Batch size for deep learning model training.

    Returns:
        pd.Series: A pandas Series containing the forecasted stock prices with future dates as index.
    """
    # Loop to allow user to retry if data loading fails
    while True:
        sym = input("Enter stock ticker (e.g., TCS.NS for NSE, AAPL for NASDAQ): ").strip().upper()
        try:
            df, arima = load_data(sym)
            break
        except ValueError as e:
            print(f"Error loading data: {e}. Please check the ticker and try again.")
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}. Please try again.")

    # Get sentiment automatically for the chosen symbol
    sentiment = get_automated_sentiment(sym)
    print(f"Automated sentiment score for {sym}: {sentiment:.4f}")

    # Define features for the hybrid model. 'Res' is the ARIMA residual, which is our target.
    feats = ["EMA10", "RSI", "MACD", "Res"]

    X_features = df[feats].values[:-1]
    y_target_residuals = df["Res"].shift(-1).dropna().values

    min_len = min(len(X_features), len(y_target_residuals))
    X_features = X_features[:min_len]
    y_target_residuals = y_target_residuals[:min_len]

    if len(X_features) == 0:
        raise ValueError("Not enough processed data to create features and targets for forecasting models.")

    try:
        res_idx = feats.index("Res")
    except ValueError:
        raise ValueError("Critical Error: 'Res' feature not found in the defined 'feats' list. "
                         "Please ensure 'Res' is part of `feats` for residual forecasting.")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    Xs_scaled = scaler_X.fit_transform(X_features)
    ys_scaled = scaler_y.fit_transform(y_target_residuals.reshape(-1, 1)).flatten()

    if len(Xs_scaled) < seq_len:
        raise ValueError(f"Not enough scaled data ({len(Xs_scaled)} points) to create sequences with "
                         f"seq_len={seq_len}. Try reducing `seq_len` or increasing `days` in `load_data`.")

    X_seq = np.array([Xs_scaled[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(Xs_scaled))])
    y_seq = ys_scaled[seq_len - 1:]

    split_point = int(0.8 * len(X_seq))
    if split_point == 0 or split_point >= len(X_seq):
        raise ValueError("Not enough data to create valid train/validation split for deep learning model. "
                         "Adjust `seq_len` or `days` to get more data points.")

    X_train_dl, X_val_dl = X_seq[:split_point], X_seq[split_point:]
    y_train_dl, y_val_dl = y_seq[:split_point], y_seq[split_point:]

    model_dl = build_model(seq_len, Xs_scaled.shape[1])
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)

    print(f"Training CNN-LSTM-Attention model for {epochs} epochs (batch size: {batch_size})...")
    model_dl.fit(X_train_dl, y_train_dl,
                 validation_data=(X_val_dl, y_val_dl),
                 epochs=epochs,
                 batch_size=batch_size,
                 callbacks=[early_stopping],
                 verbose=0)
    print("CNN-LSTM-Attention model training complete.")

    optimal_ne, optimal_md, optimal_lr = tune_xgb(Xs_scaled, ys_scaled)

    xgb_model = XGBRegressor(n_estimators=int(optimal_ne), max_depth=int(optimal_md),
                             learning_rate=optimal_lr, random_state=42, n_jobs=-1, verbosity=0)

    print("Training XGBoost model...")
    xgb_model.fit(Xs_scaled, ys_scaled)
    print("XGBoost model training complete.")

    # --- Multi-step Hybrid Forecasting ---
    print(f"Generating {future}-day forecast for {sym}...")

    current_seq_input = Xs_scaled[-seq_len:].reshape(1, seq_len, Xs_scaled.shape[1])

    predicted_scaled_residuals = []

    for i in range(future):
        dl_pred_scaled = float(model_dl.predict(current_seq_input, verbose=0)[0, 0])
        xgb_pred_scaled = float(xgb_model.predict(current_seq_input[0, -1].reshape(1, -1))[0])

        combined_scaled_residual = 0.5 * dl_pred_scaled + 0.5 * xgb_pred_scaled + 0.1 * sentiment
        predicted_scaled_residuals.append(combined_scaled_residual)

        next_feature_vector = current_seq_input[0, -1].copy()
        next_feature_vector[res_idx] = combined_scaled_residual

        current_seq_input = np.concatenate(
            [current_seq_input[:, 1:, :], next_feature_vector.reshape(1, 1, -1)],
            axis=1
        )

    predicted_inv_residuals = scaler_y.inverse_transform(
        np.array(predicted_scaled_residuals).reshape(-1, 1)
    ).flatten()

    arima_pred_base = arima.forecast(steps=future)

    final_forecast_prices = arima_pred_base + predicted_inv_residuals

    last_known_date = df.index[-1]
    forecast_dates = pd.bdate_range(start=last_known_date + pd.Timedelta(days=1), periods=future)

    forecast_series = pd.Series(final_forecast_prices, index=forecast_dates, name=sym)
    return forecast_series


# --- Visualization ---
def plot_forecast(forecast_series: pd.Series):
    """
    Plots the forecasted stock prices.

    Args:
        forecast_series (pd.Series): Pandas Series containing the forecasted prices with dates as index.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(forecast_series.index, forecast_series.values, marker="o", linestyle='-', color="purple", linewidth=2,
             label="Forecasted Price")

    plt.title(f"{forecast_series.name} {len(forecast_series)}-Day Hybrid Price Forecast", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Hybrid Stock Price Forecasting Model...")
    try:
        forecast_result = hybrid_forecast()
        print("\n🎯 Final Forecast:\n", forecast_result)
        plot_forecast(forecast_result)
    except Exception as main_error:
        print(f"\nAn unrecoverable error occurred during forecasting: {main_error}")
        print("Please review the error message and ensure all inputs are valid. "
              "Common issues include incorrect ticker symbols, unstable internet connection, "
              "or insufficient historical data. "
              "Also, ensure your NewsAPI key is correctly set if using automated sentiment.")