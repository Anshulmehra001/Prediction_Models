import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')


# ================== DATA DOWNLOAD ==================
def download_data(ticker, days):
    """Download stock data and return full OHLCV DataFrame"""
    ticker_str = str(ticker).strip().upper()
    print(f"Downloading {days} days of data for {ticker_str}...")
    try:
        data = yf.download(ticker_str, period=f"{days}d")
        if data.empty:
            print(f"No data found for {ticker_str}")
            return None
        return data
    except Exception as e:
        print(f"Download error: {str(e)}")
        return None


# ================== FEATURE ENGINEERING ==================
def add_features(data):
    """Add technical indicators and oil/gas sector-specific features"""
    if data is None or data.empty:
        print("No data provided for feature engineering")
        return None

    try:
        # Make a copy to preserve original data
        df = data.copy()

        # Technical indicators with individual error handling
        try:
            df['RSI'] = ta.rsi(df['Close'], length=14)
        except:
            df['RSI'] = 50  # Neutral value if fails

        try:
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            df['MACD'] = macd['MACD_12_26_9'] if macd is not None else 0
        except:
            df['MACD'] = 0

        try:
            df['SMA_20'] = ta.sma(df['Close'], length=20)
        except:
            df['SMA_20'] = df['Close']

        try:
            df['SMA_50'] = ta.sma(df['Close'], length=50)
        except:
            df['SMA_50'] = df['Close']

        try:
            df['EMA_12'] = ta.ema(df['Close'], length=12)
        except:
            df['EMA_12'] = df['Close']

        # Handle volume-based indicators safely
        try:
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            else:
                df['VWAP'] = df['Close']
        except:
            df['VWAP'] = df['Close']

        # Volatility measures
        try:
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        except:
            df['ATR'] = 0

        try:
            bbands = ta.bbands(df['Close'], length=20)
            if bbands is not None:
                df['BOLL_Upper'] = bbands['BBU_20_2.0']
                df['BOLL_Lower'] = bbands['BBL_20_2.0']
            else:
                df['BOLL_Upper'] = df['Close']
                df['BOLL_Lower'] = df['Close']
        except:
            df['BOLL_Upper'] = df['Close']
            df['BOLL_Lower'] = df['Close']

        # Oil/gas sector specific features
        try:
            oil_data = yf.download('CL=F', start=df.index[0], end=df.index[-1], progress=False)
            if not oil_data.empty:
                oil_prices = oil_data['Close']
                oil_prices.name = 'Crude_Oil'
                df = df.join(oil_prices, how='left')
                df['Crude_Oil'].ffill(inplace=True)
            else:
                df['Crude_Oil'] = df['Close']
        except:
            df['Crude_Oil'] = df['Close']

        # Forward fill missing values
        df.ffill(inplace=True)
        return df.dropna()

    except Exception as e:
        print(f"Feature engineering error: {str(e)}")
        # Return only close price if all else fails
        return data[['Close']]

# ================== DATA PROCESSING ==================
def prepare_data(data, time_step=60):
    """Create sequences with multiple features"""
    if data is None or data.empty:
        print("No data provided for processing")
        return None, None, None

    try:
        # Use only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        X, Y = [], []
        for i in range(len(scaled_data) - time_step - 1):
            X.append(scaled_data[i:(i + time_step)])
            Y.append(scaled_data[i + time_step, 0])  # 0 is Close price position

        X = np.array(X)
        Y = np.array(Y)
        return X, Y, scaler
    except Exception as e:
        print(f"Data preparation error: {str(e)}")
        return None, None, None


# ================== HYBRID MODEL ==================
def hybrid_forecast(data, forecast_days):
    """Enhanced hybrid model with feature engineering and XGBoost"""
    if data is None or data.empty or len(data) < 100:
        print("Insufficient data for forecasting")
        return [data['Close'].iloc[-1]] * forecast_days if not data.empty else [0] * forecast_days

    try:
        # Save original data for fallback
        original_data = data.copy()

        # Add features
        feature_data = add_features(data)

        # If feature engineering failed, use close price only
        if feature_data is None or feature_data.empty:
            print("Using basic close price for forecasting")
            feature_data = original_data[['Close']]

        # Prepare data
        X, Y, scaler = prepare_data(feature_data)

        if X is None or len(X) == 0:
            print("Falling back to simple moving average")
            return [original_data['Close'].iloc[-1]] * forecast_days

        # Time-series cross validation
        tscv = TimeSeriesSplit(n_splits=3)
        best_val_loss = float('inf')
        best_model = None

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = Y[train_index], Y[val_index]

            # Build simpler LSTM model
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(32),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            # Train with early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=32,
                verbose=0,
                callbacks=[early_stop]
            )

            # Track best model
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

        # Fallback if no model created
        if best_model is None:
            print("LSTM training failed - using simple average")
            return [original_data['Close'].iloc[-1]] * forecast_days

        # XGBoost for residual correction
        print("Training XGBoost residual model...")
        lstm_train_pred = best_model.predict(X_train, verbose=0).flatten()
        residuals = y_train - lstm_train_pred

        xgb = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8
        )
        xgb.fit(X_train.reshape(X_train.shape[0], -1), residuals)

        # Forecast future
        last_sequence = X[-1]
        forecast = []

        for _ in range(forecast_days):
            # LSTM prediction
            lstm_pred = best_model.predict(last_sequence.reshape(1, X.shape[1], X.shape[2]), verbose=0)[0][0]

            # XGBoost residual prediction
            xgb_res = xgb.predict(last_sequence.reshape(1, -1))[0]

            # Hybrid prediction
            hybrid_value = lstm_pred + xgb_res
            forecast.append(hybrid_value)

            # Update sequence with last known values
            new_row = feature_data.iloc[-1].copy()
            new_row[0] = hybrid_value  # Update Close price

            # Create new scaled features
            new_scaled = scaler.transform([new_row])[0]
            last_sequence = np.vstack([last_sequence[1:], new_scaled])

        # Inverse transform
        dummy_features = np.zeros((len(forecast), feature_data.shape[1] - 1))
        forecast_array = np.array(forecast).reshape(-1, 1)
        forecast_matrix = np.hstack([forecast_array, dummy_features])
        forecast = scaler.inverse_transform(forecast_matrix)[:, 0]

        return forecast

    except Exception as e:
        print(f"Forecasting error: {str(e)}")
        # Fallback to simple moving average
        return [original_data['Close'].iloc[-1]] * forecast_days


# ================== BACKTESTING ==================
def backtest_model(ticker):
    try:
        print(f"Backtesting {ticker} with enhanced model...")
        data = download_data(ticker, 720)

        if data is None or len(data) < 100:
            print("Insufficient data for backtesting")
            return

        # Prepare test period (last 30 days)
        test_data = data.iloc[-30:]
        actual = test_data['Close'].values

        print("Running backtest...")
        train_data = data.iloc[:-30]
        forecast = hybrid_forecast(train_data, 30)

        # Handle forecast errors
        if forecast is None or len(forecast) != 30:
            print("Forecasting failed during backtest")
            return

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mae = mean_absolute_error(actual, forecast)
        r2 = r2_score(actual, forecast)

        # Plot results
        plt.figure(figsize=(14, 7))
        plt.plot(test_data.index, actual, 'b-o', linewidth=2, markersize=5, label='Actual Price')
        plt.plot(test_data.index, forecast, 'r--s', linewidth=2, markersize=5, label='Forecasted Price')

        # Formatting
        plt.title(f"{ticker} Backtesting Results", fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # Format dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        plt.gcf().autofmt_xdate()

        # Add metrics
        textstr = '\n'.join((
            f'RMSE: {rmse:.2f}',
            f'MAE: {mae:.2f}',
            f'R²: {r2:.4f}'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                       fontsize=10, verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.show()

        print("\n===== Backtest Results =====")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")

        # Interpretation
        if r2 > 0.6:
            print("✅ Excellent predictive power (R² > 0.6)")
        elif r2 > 0.4:
            print("⚠️ Moderate predictive power (R² > 0.4)")
        else:
            print("❌ Low predictive power - Consider model tuning")

    except Exception as e:
        print(f"Backtesting failed: {str(e)}")


# ================== FORECAST GENERATION ==================
def generate_forecast():
    horizon = input("Enter forecast horizon (1, 5, 10): ").strip()
    if horizon not in ['1', '5', '10']:
        print("Invalid horizon. Using default 5 days.")
        horizon = 5
    else:
        horizon = int(horizon)

    ticker = input("Enter stock ticker (e.g., AAPL, TCS.NS): ").strip()
    if not ticker:
        print("Invalid ticker")
        return

    try:
        print()
        data = download_data(ticker, 720)

        if data is None or len(data) < 100:
            print("Insufficient data for forecasting")
            return

        print("Generating forecast...")
        forecast = hybrid_forecast(data, horizon)

        if forecast is None:
            print("Forecasting failed")
            return

        print("\n===== Forecast Results =====")
        for i, price in enumerate(forecast, 1):
            print(f"Day {i}: ₹{price:.2f}")

    except Exception as e:
        print(f"Error: {str(e)}. Please try again.")


# ================== MAIN SYSTEM ==================
def main():
    print("\n===== Robust Stock Forecasting System =====")
    print("TensorFlow optimization warnings suppressed\n")

    while True:
        print("\nOptions:")
        print("1. Generate Price Forecast (1, 5, or 10 days)")
        print("2. Backtest Model Accuracy")
        print("3. Exit")

        choice = input("Select option (1-3): ").strip()

        if choice == '1':
            generate_forecast()
        elif choice == '2':
            ticker = input("Enter stock ticker for backtesting: ").strip()
            if ticker:
                backtest_model(ticker)
            else:
                print("Invalid ticker")
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid option. Please choose 1-3.")


if __name__ == "__main__":
    main()