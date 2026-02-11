import warnings
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import mplfinance as mpf

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import models and tools
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

# --- Configuration for NewsAPI (ADD YOUR API KEY HERE) ---
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # <--- REPLACE WITH YOUR ACTUAL NEWSAPI.ORG API KEY
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

# --- GLOBAL CONSTANTS ---
REQUIRED_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
EPOCHS = 50  # Reduced for demonstration, increase for better training
BATCH_SIZE = 32


# --- Data Loading and Preprocessing (ADJUSTED FOR DAILY) ---
def load_data_daily(symbol: str, period: str = '5y') -> pd.DataFrame:
    """
    Loads historical daily stock OHLCV data.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL', 'TCS.NS').
        period (str): The historical period (e.g., '5y', '10y', 'max').

    Returns:
        pd.DataFrame: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'.

    Raises:
        ValueError: If no data is found or required columns are missing.
    """
    print(f"Downloading daily data for {symbol} for {period} period...")
    df = yf.download(symbol, period=period, interval='1d', auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(
            f"No daily data found for '{symbol}' for period '{period}'. Please verify the ticker or period.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
    df.columns = [col.replace(' ', '_').title() for col in df.columns]
    df = df.rename(columns={'Adj_Close': 'Close'}, errors='ignore')

    if not all(col in df.columns for col in REQUIRED_OHLCV_COLS):
        missing = [col for col in REQUIRED_OHLCV_COLS if col not in df.columns]
        raise ValueError(f"Missing required columns for '{symbol}': {missing}. Available: {df.columns.tolist()}")

    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(f"No valid data remaining for '{symbol}' after dropping NaNs. Try a longer period.")

    return df


# --- Sentiment Analysis (Automated - unchanged) ---
nlp_sent = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")


def get_automated_sentiment(symbol: str, query_days: int = 7) -> float:
    """
    Fetches recent news headlines for a given stock symbol using NewsAPI.org
    and calculates an aggregated sentiment score.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        print("Warning: NEWS_API_KEY not set or is default. Cannot fetch automated news. Defaulting sentiment to 0.0.")
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


# --- XGBoost Model Tuning (Unchanged, but target will be 2 outputs) ---
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
            return 1e10  # Return a very high error if no splits could be evaluated

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
def build_model_high_low(seq_len: int, feat_dim: int) -> Model:
    """
    Builds a CNN-LSTM model with an Attention mechanism for multivariate sequence prediction.
    Output dimension is 2 for High and Low.
    """
    i = Input(shape=(seq_len, feat_dim))

    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="causal")(i)
    x = Dropout(0.2)(x)

    x = LSTM(64, return_sequences=True)(x)

    a = Attention()([x, x])

    x = LSTM(32)(a)
    out = Dense(2)(x)  # Output 2 values: High and Low

    model = Model(inputs=i, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    print("CNN-LSTM-Attention model built successfully for High/Low prediction.")
    return model


# --- Hybrid Forecasting Logic (ADJUSTED FOR NEXT DAY HIGH/LOW) ---
def hybrid_forecast_high_low(seq_len: int = 60, forecast_days: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Performs hybrid stock High/Low forecasting for the next trading day.
    Returns: forecasted_df (with High, Low), historical_df_for_plotting, chosen_symbol
    """
    df_loaded = None
    sym_chosen = None
    period_chosen = '5y'  # Good period for daily data

    while True:
        sym_chosen = input("Enter stock ticker (e.g., TCS.NS for NSE, AAPL for NASDAQ): ").strip().upper()
        try:
            df_loaded = load_data_daily(sym_chosen, period=period_chosen)

            # Recalculate indicators on the daily data
            df_loaded["EMA10"] = df_loaded["Close"].ewm(span=10, adjust=False).mean()
            delta = df_loaded["Close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.ewm(com=13, adjust=False).mean()
            ma_down = down.ewm(com=13, adjust=False).mean()
            rs = ma_up / (ma_down + 1e-10)
            df_loaded["RSI"] = 100 - (100 / (1 + rs))
            exp1 = df_loaded["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df_loaded["Close"].ewm(span=26, adjust=False).mean()
            df_loaded["MACD"] = exp1 - exp2

            # No 'Close_Res' for High/Low prediction, use Close directly or its change as feature
            # For simplicity, we will use OHLCV, indicators, and sentiment directly as features.

            # Target variables: Next Day's High and Low
            df_loaded['Next_Day_High'] = df_loaded['High'].shift(-1)
            df_loaded['Next_Day_Low'] = df_loaded['Low'].shift(-1)

            initial_rows = len(df_loaded)
            df_loaded.dropna(inplace=True)  # Drop NaNs introduced by indicators and shifting target
            if len(df_loaded) < initial_rows:
                print(f"Dropped {initial_rows - len(df_loaded)} rows due to NaN values after calculations (common).")
            if df_loaded.empty:
                raise ValueError(
                    f"No valid data remaining for '{sym_chosen}' after dropping NaNs. Try a longer period.")

            break
        except ValueError as e:
            print(f"Error loading data: {e}. Please check the ticker and try again.")
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}. Please try again.")

    sentiment = get_automated_sentiment(sym_chosen)
    print(f"Automated sentiment score for {sym_chosen}: {sentiment:.4f}")

    feats = ["Open", "High", "Low", "Close", "Volume", "EMA10", "RSI", "MACD",
             "Sentiment"]  # Add sentiment as a feature
    df_loaded['Sentiment'] = sentiment  # Assign the daily sentiment to all rows or last row

    target_cols = ["Next_Day_High", "Next_Day_Low"]

    if len(df_loaded) < (seq_len + 2):
        raise ValueError(f"Insufficient data ({len(df_loaded)} points) after preprocessing to run the model with "
                         f"seq_len={seq_len}. Try increasing `period` or reducing `seq_len`.")

    X_features_raw = df_loaded[feats].iloc[:-1].values
    y_target_raw = df_loaded[target_cols].iloc[
                   :-1].values  # Target is next day, so last row is excluded as it has no next day

    # Ensure X_features_raw and y_target_raw have same number of rows for proper alignment
    min_len = min(len(X_features_raw), len(y_target_raw))
    X_features_aligned = X_features_raw[:min_len]
    y_target_aligned = y_target_raw[:min_len]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()  # Scaler for High and Low targets

    Xs_scaled = scaler_X.fit_transform(X_features_aligned)
    ys_scaled = scaler_y.fit_transform(y_target_aligned)

    if len(Xs_scaled) < seq_len:
        raise ValueError(f"Not enough scaled data ({len(Xs_scaled)} points) to create sequences with "
                         f"seq_len={seq_len}. Try reducing `seq_len` or increasing `period`.")

    # Create sequences for deep learning
    X_seq = np.array([Xs_scaled[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(Xs_scaled))])
    y_seq = ys_scaled[seq_len - 1:]

    split_point = int(0.8 * len(X_seq))
    if split_point == 0 or split_point >= len(X_seq):
        raise ValueError("Not enough data to create valid train/validation split for deep learning model. "
                         "Adjust `seq_len` or `period` to get more data points.")

    X_train_dl, X_val_dl = X_seq[:split_point], X_seq[split_point:]
    y_train_dl, y_val_dl = y_seq[:split_point], y_seq[split_point:]

    model_dl = build_model_high_low(seq_len, Xs_scaled.shape[1])
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)

    print(f"Training CNN-LSTM-Attention model for {EPOCHS} epochs (batch size: {BATCH_SIZE})...")
    model_dl.fit(X_train_dl, y_train_dl,
                 validation_data=(X_val_dl, y_val_dl),
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 callbacks=[early_stopping],
                 verbose=0)
    print("CNN-LSTM-Attention model training complete.")

    optimal_ne, optimal_md, optimal_lr = tune_xgb(Xs_scaled, ys_scaled, len(target_cols))  # Target is 2 outputs

    xgb_model = XGBRegressor(n_estimators=int(optimal_ne), max_depth=int(optimal_md),
                             learning_rate=optimal_lr, random_state=42, n_jobs=-1, verbosity=0)

    print("Training XGBoost model...")
    xgb_model.fit(Xs_scaled, ys_scaled)
    print("XGBoost model training complete.")

    print(f"Generating next {forecast_days} day(s) High and Low forecast for {sym_chosen}...")

    last_feature_vector_actual = df_loaded[feats].iloc[-1].values  # Last complete feature vector

    # Use the last 'seq_len' scaled feature vectors as input for prediction
    current_seq_input = Xs_scaled[-seq_len:].reshape(1, seq_len, Xs_scaled.shape[1])

    dl_pred_scaled = model_dl.predict(current_seq_input, verbose=0)[0]
    xgb_pred_scaled = xgb_model.predict(current_seq_input[0, -1].reshape(1, -1))[0]  # Use the last vector for XGBoost

    # Combine predictions
    combined_scaled_pred = (0.5 * dl_pred_scaled + 0.5 * xgb_pred_scaled)

    # Inverse transform to get actual High and Low values
    predicted_high_low = scaler_y.inverse_transform(combined_scaled_pred.reshape(1, -1))[0]

    forecasted_high = predicted_high_low[0]
    forecasted_low = predicted_high_low[1]

    # Basic sanity checks for High and Low
    # High should be >= Low
    if forecasted_high < forecasted_low:
        forecasted_high, forecasted_low = forecasted_low, forecasted_high  # Swap if inverted

    # High and Low should be around current Close for next day (prevent extreme values)
    current_close = df_loaded['Close'].iloc[-1]

    # Ensure High is not much lower than current close, and Low not much higher
    # These are heuristic checks, actual bounds might be derived from volatility
    # For now, let's just make sure Low is <= Close and High is >= Close
    forecasted_low = min(forecasted_low, current_close)
    forecasted_high = max(forecasted_high, current_close)

    # Also ensure High is strictly greater than Low unless it's a flat market (unlikely for high/low range)
    if forecasted_high <= forecasted_low:
        forecasted_high = forecasted_low * 1.001  # Add a small buffer

    # Get the next trading day's date
    last_known_date = df_loaded.index[-1]
    next_trading_date = last_known_date + pd.Timedelta(days=1)
    # Handle weekends and holidays (basic: skip to Monday if next day is Sat/Sun)
    while next_trading_date.weekday() > 4:  # Monday=0, Sunday=6
        next_trading_date += pd.Timedelta(days=1)

    forecast_df = pd.DataFrame(
        [[forecasted_high, forecasted_low]],
        index=[next_trading_date],
        columns=["Predicted High", "Predicted Low"]
    )
    forecast_df.index.name = "Date"
    forecast_df.name = sym_chosen

    # For plotting, we'll return historical data up to the last known day
    historical_df_for_plotting = df_loaded[REQUIRED_OHLCV_COLS].copy()

    return forecast_df, historical_df_for_plotting, sym_chosen


# --- Visualization (ADJUSTED FOR NEXT DAY HIGH/LOW - FIXING LAYOUT) ---
def plot_forecast_chart_daily(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, symbol: str):
    """
    Plots historical OHLCV data and the next day's predicted High/Low.
    """
    if historical_df.empty:
        print("No historical data to plot.")
        return
    if forecast_df.empty:
        print("No forecast data to plot.")
        return

    # Combine historical and forecast for plotting.
    last_historical_close = historical_df['Close'].iloc[-1]
    forecast_date = forecast_df.index[0]
    predicted_high = forecast_df['Predicted High'].iloc[0]
    predicted_low = forecast_df['Predicted Low'].iloc[0]

    dummy_forecast_row = pd.DataFrame(
        [[last_historical_close, predicted_high, predicted_low, last_historical_close, historical_df['Volume'].mean()]],
        index=[forecast_date],
        columns=REQUIRED_OHLCV_COLS
    )

    plot_df = pd.concat([historical_df.tail(60), dummy_forecast_row])

    print(f"\n--- Displaying Historical & Predicted Chart for {symbol} ---")

    mc_historical = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s_historical = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc_historical)

    high_series_for_plot = pd.Series(np.nan, index=plot_df.index)
    high_series_for_plot.loc[forecast_date] = predicted_high

    low_series_for_plot = pd.Series(np.nan, index=plot_df.index)
    low_series_for_plot.loc[forecast_date] = predicted_low

    apds = [
        mpf.make_addplot(high_series_for_plot, type='scatter', marker='^', markersize=200, color='blue', panel=0,
                         label='Predicted High'),
        mpf.make_addplot(low_series_for_plot, type='scatter', marker='v', markersize=200, color='orange', panel=0,
                         label='Predicted Low'),
        # FIX: Changed `pd.Series([])` to `pd.Series(np.nan, index=plot_df.index)`
        mpf.make_addplot(pd.Series(np.nan, index=plot_df.index), panel=0, type='line', secondary_y=False, label=' ',
                         ylabel=' '),
    ]

    fig, axes = mpf.plot(plot_df,
                         type='candle',
                         style=s_historical,
                         title=f"Historical & Next Day Predicted High/Low for {symbol}",
                         ylabel='Price',
                         ylabel_lower='Volume',
                         volume=True,
                         figscale=1.8,  # Increased figscale for more overall space
                         addplot=apds,
                         show_nontrading=False,
                         xrotation=15,  # Slightly rotate for better readability if dates overlap
                         returnfig=True  # Return the figure and axes objects
                         )

    # Adjust tight_layout to give a bit more padding
    # This specifically addresses the legend cutoff and potential label overlaps
    fig.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust right boundary of the layout (0.95 means 95% of figure width)
    # The 0.95 will give more space on the right for the legend

    plt.show()  # Make sure to show the plot

    print(f"\nNext Trading Day ({forecast_date.strftime('%Y-%m-%d')}):\n"
          f"  Predicted High: {predicted_high:.2f}\n"
          f"  Predicted Low: {predicted_low:.2f}")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Hybrid Stock Next Day High/Low Forecasting Model...")
    try:
        # We now forecast 1 day's High and Low
        forecast_result_df, historical_data_for_plot, symbol_for_plot = hybrid_forecast_high_low(
            forecast_days=1  # Predict only the next day
        )
        print("\n🎯 Next Day High/Low Forecast:\n", forecast_result_df)

        # Plot the historical data with the predicted High/Low point
        plot_forecast_chart_daily(historical_data_for_plot, forecast_result_df, symbol_for_plot)

    except ValueError as val_error:
        print(f"\nError: {val_error}")
        print("This often means insufficient data from yfinance for the chosen period, or a bad ticker.")
    except Exception as main_error:
        print(f"\nAn unrecoverable error occurred during forecasting: {main_error}")
        print("Please review the error message and ensure all inputs are valid. "
              "Common issues include incorrect ticker symbols, unstable internet connection, "
              "or insufficient historical data. "
              "Also, ensure your NewsAPI key is correctly set if using automated sentiment.")
