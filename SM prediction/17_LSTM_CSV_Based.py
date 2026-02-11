
import sys

try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import datetime
except ImportError as e:
    print(f"Error: {e}. Please install required libraries with: pip install pandas numpy tensorflow scikit-learn")
    sys.exit(1)


# Step 1: Fetch historical stock data from CSV
def fetch_stock_data(ticker, start_date, end_date):
    """
    Load daily close prices from a CSV file.
    """
    try:
        data = pd.read_csv("C:\\Users\\Aniket Mehra\\3\\SM prediction\\aapl_data.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        if data.empty:
            raise ValueError(f"No data found in CSV for {ticker}")
        print(f"Fetched {len(data)} data points for {ticker} from {start_date} to {end_date}")
        return data['Close'].values
    except Exception as e:
        print(f"Error loading CSV data for {ticker}: {e}")
        sys.exit(1)


# Step 2: Prepare data for LSTM
def prepare_data(data, look_back=60):
    """
    Prepare data by scaling and creating sequences for LSTM input.
    """
    if len(data) < look_back:
        raise ValueError(f"Insufficient data: {len(data)} samples, minimum {look_back} required.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler


# Step 3: Build LSTM model
def build_model(look_back):
    """
    Build and compile an LSTM model for stock price prediction.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Step 4: Train and predict
def predict_stock_price(ticker, start_date, end_date, look_back=60):
    """
    Load data, train LSTM model, and predict stock prices.
    """
    try:
        # Fetch data
        data = fetch_stock_data(ticker, start_date, end_date)

        # Prepare data
        X, y, scaler = prepare_data(data, look_back)

        # Split data into training and testing
        train_size = int(len(X) * 0.8)
        if train_size < 1:
            raise ValueError("Not enough data to train the model after splitting.")
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Reshape data for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build and train model
        model = build_model(look_back)
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform([y_test])

        return predictions, y_test
    except Exception as e:
        print(f"Error during model training or prediction: {e}")
        sys.exit(1)


# Example usage
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-06-23"
    predictions, actual = predict_stock_price(ticker, start_date, end_date)

    # Print last few predictions vs actual
    print("\nPredicted vs Actual Prices (AAPL Closing Prices):")
    for i in range(min(5, len(predictions))):
        print(f"Day {i + 1}: Predicted = {predictions[i][0]:.2f}, Actual = {actual[0][i]:.2f}")
