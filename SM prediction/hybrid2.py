import warnings
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
import tensorflow as tf  # Import tensorflow for setting random seed

# Attempt to import Attention layer, providing a fallback for older TensorFlow versions
try:
    from tensorflow.keras.layers import Attention
except ImportError:
    from tensorflow.keras.layers import AdditiveAttention as Attention

# Set random seeds for reproducibility across runs
# Note: Complete reproducibility with TensorFlow can be challenging due to
# multi-threading and GPU operations.
np.random.seed(42)
tf.random.set_seed(42)


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
    # Use auto_adjust=True to get adjusted closing prices, which are generally preferred.
    df = yf.download(symbol, period=f"{days}d", auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data found for '{symbol}'. Please verify the ticker or period.")

    # Handle yfinance MultiIndex columns and ensure standard column names
    if isinstance(df.columns, pd.MultiIndex):
        # If MultiIndex (e.g., for multiple tickers or older yfinance versions),
        # extract the first level which typically contains 'Open', 'High', 'Low', 'Close', 'Volume'
        df.columns = [col[0] for col in df.columns.values]
        print(f"Flattened MultiIndex columns to: {df.columns.tolist()}")
    else:
        # If single-level columns, just ensure they are capitalized and don't have spaces
        df.columns = [col.replace(' ', '_').title() for col in df.columns]
        print(f"Downloaded columns (standardized): {df.columns.tolist()}")

    # Ensure 'Close' column exists and is in the correct casing after standardizing
    # Use a dictionary for robust renaming, handling potential 'Adj Close'
    df = df.rename(columns={
        'Adj Close': 'Close',  # Rename 'Adj Close' to 'Close' if it exists
        'Gmt Offset': 'Gmt_Offset'  # Example: clean other potential non-standard names
    }, errors='ignore')  # 'errors=ignore' prevents key errors if column doesn't exist

    if 'Close' not in df.columns:
        raise ValueError(f"'Close' column missing for '{symbol}'. Available columns: {df.columns.tolist()}")

    # Convert index to datetime and reindex to business days frequency
    df.index = pd.to_datetime(df.index)
    # 'B' for Business Day frequency. 'pad' fills missing dates with previous valid observation.
    df = df.asfreq('B', method='pad')

    # Handle potential NaNs introduced by `asfreq` or initial download (e.g., holidays)
    if df['Close'].isnull().all():
        raise ValueError(f"No valid 'Close' data for '{symbol}' after reindexing (all are null). "
                         "Try a longer period or a different symbol.")
    if df['Close'].isnull().any():
        initial_close_nan_count = df['Close'].isnull().sum()
        df['Close'] = df['Close'].fillna(method='ffill')  # Forward fill remaining NaNs
        # Drop any leading NaNs that couldn't be filled by ffill
        df.dropna(subset=['Close'], inplace=True)
        if initial_close_nan_count > 0 and df['Close'].isnull().sum() == 0:
            print(f"Filled {initial_close_nan_count} NaNs in 'Close' and dropped any leading NaNs.")
        elif df['Close'].isnull().sum() > 0:
            print(f"Warning: Still {df['Close'].isnull().sum()} NaNs in 'Close' after ffill and dropna. "
                  "This might indicate insufficient historical data.")

    # Fit ARIMA model to the 'Close' price
    # ARIMA order (5, 1, 0) is common: (AR_order, differencing_order, MA_order)
    # differencing_order=1 means it models the difference of prices, assuming stationarity after one differencing.
    try:
        # Ensure enough data for ARIMA after dropping NaNs
        if len(df) < 10:  # Arbitrary minimum for ARIMA to make sense
            raise ValueError("Insufficient data points for ARIMA fitting after preprocessing.")
        arima = ARIMA(df["Close"], order=(5, 1, 0)).fit()
        df["Res"] = arima.resid  # Store ARIMA residuals
        print("ARIMA model fitted successfully.")
    except Exception as e:
        print(f"Warning: ARIMA model fitting failed ({e}). Using simple difference for residuals as fallback.")
        # Fallback if ARIMA fails (e.g., non-stationary data, too few points)
        df["Res"] = df["Close"].diff().fillna(0)  # Simple difference as a robust fallback

    # Calculate technical indicators
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # RSI Calculation (Standard 14-period RSI, EWM for smoothing)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Use 13 for com to get 14-period EMA-like smoothing for RSI
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    # Add a small epsilon to prevent division by zero in case ma_down is 0
    rs = ma_up / (ma_down + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD Calculation (Standard 12-day EMA, 26-day EMA)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    # Signal line (9-day EMA of MACD) could also be added as a feature if desired
    # df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Drop any remaining NaN values that result from indicator calculations
    # These typically occur at the beginning of the DataFrame due to rolling windows.
    initial_rows = len(df)
    df.dropna(inplace=True)
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows due to NaN values after indicator calculation (common).")
    if df.empty:
        raise ValueError(f"No valid data remaining for '{symbol}' after dropping NaNs. Try a longer period.")

    return df, arima


# --- Sentiment Analysis ---
# Initialize sentiment analysis pipeline once to avoid reloading model for each call
# Model: 'yiyanghkust/finbert-tone' is designed for financial sentiment.
nlp_sent = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")


def get_sentiment() -> float:
    """
    Prompts the user for a news headline and returns its sentiment score.
    The 'finbert-tone' model outputs scores for 'positive', 'negative', 'neutral' etc.
    This function simplifies it to a single positive (score) or negative (-score) value.
    """
    txt = input("Enter a recent news headline for sentiment analysis "
                "(e.g., 'Company reports strong earnings', 'Market faces downturn'):\n")
    if not txt.strip():
        print("No headline entered. Defaulting to neutral sentiment (0.0).")
        return 0.0

    # Process the text with the sentiment pipeline
    res = nlp_sent(txt)[0]

    # Extract score based on predicted label
    # 'finbert-tone' specifically outputs 'positive', 'negative', 'neutral', 'analytical', 'confident', 'joy', 'fear', 'anger', 'sadness'
    # We are simplifying to a binary positive/negative based on the primary labels.
    if res["label"].upper() == "POSITIVE":
        return res["score"]
    elif res["label"].upper() == "NEGATIVE":
        return -res["score"]
    else:  # Neutral or other tones will be treated as 0 for simplicity in this context
        return 0.0


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
                             random_state=42, n_jobs=-1, verbosity=0)  # verbosity=0 for silent training
        errs = []
        # TimeSeriesSplit ensures that training data always precedes validation data
        # n_splits determines how many splits; 3-5 is common.
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Ensure enough samples in train/val splits for meaningful training
            if len(X_train) == 0 or len(X_val) == 0:
                continue

            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            errs.append(mean_absolute_error(y_val, pred))

        # Return a high error if no valid splits were processed
        if not errs:
            return 1e10  # A very large number to indicate failure

        return np.mean(errs)

    # Define the search space for hyperparameters for Bayesian optimization
    space = [
        Integer(50, 500, name="ne"),  # n_estimators: Number of boosting rounds
        Integer(3, 15, name="md"),  # max_depth: Maximum depth of a tree
        Real(0.005, 0.5, prior="log-uniform", name="lr")  # learning_rate: Step size shrinkage
    ]

    print("Tuning XGBoost hyperparameters (this may take a moment, ~15 calls)...")
    # gp_minimize performs Bayesian optimization
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

    # Conv1D layer: Applies 1D convolution, 'causal' padding ensures no future data leakage.
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="causal")(i)
    x = Dropout(0.2)(x)  # Dropout for regularization

    # LSTM layer: return_sequences=True is crucial for passing a sequence to the Attention layer
    x = LSTM(64, return_sequences=True)(x)

    # Attention layer: Self-attention (queries, keys, and values are all from x)
    # This helps the model focus on important parts of the input sequence.
    a = Attention()([x, x])

    # Another LSTM layer to process the output of the attention mechanism
    x = LSTM(32)(a)
    out = Dense(1)(x)  # Output layer for predicting a single residual value

    model = Model(inputs=i, outputs=out)
    model.compile(optimizer="adam", loss="mse")  # Adam optimizer, Mean Squared Error loss
    print("CNN-LSTM-Attention model built successfully.")
    model.summary()  # Print model summary for inspection
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
            break  # Exit loop if data loads successfully
        except ValueError as e:
            print(f"Error loading data: {e}. Please check the ticker and try again.")
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}. Please try again.")

    # Get sentiment from user input
    sentiment = get_sentiment()
    print(f"Sentiment score: {sentiment:.4f}")

    # Define features for the hybrid model. 'Res' is the ARIMA residual, which is our target.
    # The order of features here must be consistent with how `nxt_feature_vector` is constructed.
    feats = ["EMA10", "RSI", "MACD", "Res"]

    # Prepare data for models: X will contain features, y will contain the *next* day's residual.
    # We use df[feats].values[:-1] because the target 'Res' for a given row is what we want to predict
    # for the *next* time step based on the features of the current row.
    X_features = df[feats].values[:-1]
    y_target_residuals = df["Res"].shift(-1).dropna().values  # Target: residual of the next day

    # Ensure y_target_residuals has enough values corresponding to X_features
    # This might happen if the last few rows of df didn't have enough look-ahead for shift(-1).
    min_len = min(len(X_features), len(y_target_residuals))
    X_features = X_features[:min_len]
    y_target_residuals = y_target_residuals[:min_len]

    if len(X_features) == 0:
        raise ValueError("Not enough processed data to create features and targets for forecasting models.")

    # Determine the index of 'Res' within the features list
    # This is crucial for correctly updating the sequential input during prediction.
    try:
        res_idx = feats.index("Res")
    except ValueError:
        raise ValueError("Critical Error: 'Res' feature not found in the defined 'feats' list. "
                         "Please ensure 'Res' is part of `feats` for residual forecasting.")

    # Scale features (X) and target (y) separately
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    Xs_scaled = scaler_X.fit_transform(X_features)
    ys_scaled = scaler_y.fit_transform(y_target_residuals.reshape(-1, 1)).flatten()

    # Create sequences for the deep learning model (CNN-LSTM-Attention)
    # Each sequence `X_seq` will be `seq_len` long, predicting the `y_seq` at its end.
    if len(Xs_scaled) < seq_len:
        raise ValueError(f"Not enough scaled data ({len(Xs_scaled)} points) to create sequences with "
                         f"seq_len={seq_len}. Try reducing `seq_len` or increasing `days` in `load_data`.")

    # X_seq will be (num_samples, seq_len, num_features)
    X_seq = np.array([Xs_scaled[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(Xs_scaled))])
    # y_seq corresponds to the target residual for the last step of each X_seq
    y_seq = ys_scaled[seq_len - 1:]

    # Split data for training and validation (time-series split)
    split_point = int(0.8 * len(X_seq))  # 80% for training, 20% for validation
    if split_point == 0 or split_point >= len(X_seq):
        raise ValueError("Not enough data to create valid train/validation split for deep learning model. "
                         "Adjust `seq_len` or `days` to get more data points.")

    X_train_dl, X_val_dl = X_seq[:split_point], X_seq[split_point:]
    y_train_dl, y_val_dl = y_seq[:split_point], y_seq[split_point:]

    # Build and train the deep learning model
    model_dl = build_model(seq_len, Xs_scaled.shape[1])
    # EarlyStopping monitors validation loss and stops if it doesn't improve, restoring best weights.
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)

    print(f"Training CNN-LSTM-Attention model for {epochs} epochs (batch size: {batch_size})...")
    model_dl.fit(X_train_dl, y_train_dl,
                 validation_data=(X_val_dl, y_val_dl),
                 epochs=epochs,
                 batch_size=batch_size,
                 callbacks=[early_stopping],
                 verbose=0)  # Set verbose=1 to see training progress
    print("CNN-LSTM-Attention model training complete.")

    # Tune and train the XGBoost model
    # Use the full scaled features and target for tuning and final training of XGBoost.
    # XGBoost does not inherently handle sequences, so it's trained on individual (feature, target) pairs.
    optimal_ne, optimal_md, optimal_lr = tune_xgb(Xs_scaled, ys_scaled)

    xgb_model = XGBRegressor(n_estimators=int(optimal_ne), max_depth=int(optimal_md),
                             learning_rate=optimal_lr, random_state=42, n_jobs=-1, verbosity=0)

    print("Training XGBoost model...")
    xgb_model.fit(Xs_scaled, ys_scaled)  # Train on the full dataset
    print("XGBoost model training complete.")

    # --- Multi-step Hybrid Forecasting ---
    print(f"Generating {future}-day forecast for {sym}...")

    # Initialize the input sequence for deep learning prediction with the last `seq_len` observations
    # This `seq` will be updated iteratively for multi-step forecasting.
    # Shape must be (1, seq_len, num_features) for Keras model.predict.
    current_seq_input = Xs_scaled[-seq_len:].reshape(1, seq_len, Xs_scaled.shape[1])

    predicted_scaled_residuals = []

    for i in range(future):
        # 1. Predict residual using the deep learning model (CNN-LSTM-Attention)
        dl_pred_scaled = float(model_dl.predict(current_seq_input, verbose=0)[0, 0])

        # 2. Predict residual using XGBoost (using the *last time step's features* from the sequence)
        # XGBoost expects a 2D array: (num_samples, num_features), so reshape `current_seq_input[0, -1]`
        xgb_pred_scaled = float(xgb_model.predict(current_seq_input[0, -1].reshape(1, -1))[0])

        # 3. Combine predictions and incorporate sentiment
        # The weights (0.5 for each model, 0.1 for sentiment) are heuristic.
        # This is where the 'hybrid' part really comes in.
        combined_scaled_residual = 0.5 * dl_pred_scaled + 0.5 * xgb_pred_scaled + 0.1 * sentiment
        predicted_scaled_residuals.append(combined_scaled_residual)

        # 4. Prepare the next input sequence for the deep learning model (recursive prediction)
        # We need to construct the feature vector for the next time step.
        # It's based on the last observed features, but the 'Res' component is replaced
        # by our newly predicted residual, as this is what we're forecasting.
        next_feature_vector = current_seq_input[0, -1].copy()  # Copy the last feature vector from the current sequence
        next_feature_vector[res_idx] = combined_scaled_residual  # Replace the 'Res' component at its known index

        # Shift the sequence left by one time step and append the new feature vector
        # This simulates rolling the window forward with the new prediction.
        current_seq_input = np.concatenate(
            [current_seq_input[:, 1:, :], next_feature_vector.reshape(1, 1, -1)],
            axis=1
        )

    # Inverse transform the scaled residual predictions back to their original scale
    # This gives us the predicted residual values in terms of actual price units.
    predicted_inv_residuals = scaler_y.inverse_transform(
        np.array(predicted_scaled_residuals).reshape(-1, 1)
    ).flatten()

    # Get ARIMA's base forecast for the stock price
    # ARIMA predicts the direct price based on its linear model.
    arima_pred_base = arima.forecast(steps=future)

    # Combine ARIMA base forecast with the predicted residuals from the hybrid model
    # The final forecast is the ARIMA's linear prediction plus the hybrid model's non-linear residual prediction.
    final_forecast_prices = arima_pred_base + predicted_inv_residuals

    # Generate future business dates for the forecast
    last_known_date = df.index[-1]
    # Start forecasting from the next business day after the last known data point
    forecast_dates = pd.bdate_range(start=last_known_date + pd.Timedelta(days=1), periods=future)

    # Create a pandas Series for the final forecast with proper dates and name
    forecast_series = pd.Series(final_forecast_prices, index=forecast_dates, name=sym)
    return forecast_series


# --- Visualization ---
def plot_forecast(forecast_series: pd.Series):
    """
    Plots the forecasted stock prices.

    Args:
        forecast_series (pd.Series): Pandas Series containing the forecasted prices with dates as index.
    """
    plt.figure(figsize=(14, 7))  # Increased figure size for better readability
    plt.plot(forecast_series.index, forecast_series.values, marker="o", linestyle='-', color="purple", linewidth=2,
             label="Forecasted Price")

    plt.title(f"{forecast_series.name} {len(forecast_series)}-Day Hybrid Price Forecast", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
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
              "or insufficient historical data.")