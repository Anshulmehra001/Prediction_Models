import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow optimization warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')


def download_data(ticker, days):
    """Download stock data and return ONLY DataFrame"""
    # Ensure ticker is string
    ticker_str = str(ticker).strip().upper()
    print(f"Downloading {days} days of data for {ticker_str}...")
    data = yf.download(ticker_str, period=f"{days}d")
    return data


def create_dataset(data, time_step=1):
    """Create LSTM dataset from time series"""
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


def hybrid_forecast(data, forecast_days):
    """Hybrid LSTM + Random Forest forecasting"""
    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)

    # LSTM Model
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/test split (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM
    print("Building LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

    # LSTM Predictions
    print("Generating predictions...")
    lstm_train_pred = model.predict(X_train, verbose=0)
    lstm_test_pred = model.predict(X_test, verbose=0)

    # Random Forest on LSTM residuals
    print("Training Random Forest model...")
    rf = RandomForestRegressor(n_estimators=100)
    residuals = y_train - lstm_train_pred.flatten()
    rf.fit(X_train.reshape(X_train.shape[0], -1), residuals)

    # Hybrid predictions
    rf_residuals = rf.predict(X_test.reshape(X_test.shape[0], -1))
    hybrid_pred = lstm_test_pred.flatten() + rf_residuals

    # Forecast future
    last_sequence = scaled_data[-time_step:]
    forecast = []
    for _ in range(forecast_days):
        x_input = last_sequence[-time_step:].reshape(1, time_step, 1)
        lstm_pred = model.predict(x_input, verbose=0)[0][0]
        rf_res = rf.predict(x_input.reshape(1, -1))[0]
        hybrid_value = lstm_pred + rf_res
        forecast.append(hybrid_value)
        last_sequence = np.append(last_sequence, hybrid_value)
        last_sequence = last_sequence[1:]

    # Inverse transform
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast.flatten()


def backtest_model(ticker):
    """Backtest hybrid model accuracy"""
    try:
        # Download data (only DataFrame returned)
        data = download_data(ticker, 365)

        if len(data) < 100:
            print("Insufficient data for backtesting")
            return

        # Prepare test period (last 30 days)
        test_data = data.iloc[-30:]
        actual = test_data['Close'].values

        print("Backtesting model...")
        forecast = hybrid_forecast(data.iloc[:-30], 30)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mae = mean_absolute_error(actual, forecast)
        r2 = r2_score(actual, forecast)

        # Plot results with professional styling
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, actual, 'b-', linewidth=2, label='Actual Price')
        plt.plot(test_data.index, forecast, 'r--', linewidth=2, label='Forecasted Price')

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

        # Add metrics to plot
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

    except Exception as e:
        print(f"Backtesting failed: {str(e)}")


def main():
    print("\n===== Hybrid Stock Forecasting System =====")
    print("TensorFlow optimization warnings suppressed\n")

    while True:
        print("\nOptions:")
        print("1. Generate Price Forecast (1, 5, or 10 days)")
        print("2. Backtest Model Accuracy")
        print("3. Exit")

        choice = input("Select option (1-3): ").strip()

        if choice == '1':
            horizon = input("Enter forecast horizon (1, 5, 10): ").strip()
            if horizon not in ['1', '5', '10']:
                print("Invalid horizon. Using default 5 days.")
                horizon = 5
            else:
                horizon = int(horizon)

            ticker = input("Enter stock ticker (e.g., AAPL, TCS.NS): ").strip()
            if not ticker:
                print("Invalid ticker")
                continue

            try:
                # Download data (only DataFrame returned)
                print()
                data = download_data(ticker, 720)  # 720 days = ~2 years

                if len(data) < 100:
                    print("Insufficient data for forecasting")
                    continue

                print("Generating forecast...")
                forecast = hybrid_forecast(data, horizon)

                print("\n===== Forecast Results =====")
                for i, price in enumerate(forecast, 1):
                    print(f"Day {i}: ${price:.2f}")

            except Exception as e:
                print(f"Error: {str(e)}. Please try again.")

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