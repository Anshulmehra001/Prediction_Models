import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

def train_predict_one_day(ticker: str, seq_len=60):
    # Fetch historical data
    df = yf.download(ticker, period="2y", auto_adjust=True)
    data = df[['Close']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Build sequences
    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len])
    X, y = np.array(X), np.array(y)

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=20, batch_size=16, verbose=0)

    # Predict next day
    last_seq = data_scaled[-seq_len:].reshape(1, seq_len, 1)
    next_scaled = model.predict(last_seq, verbose=0)[0, 0]
    next_price = scaler.inverse_transform([[next_scaled]])[0, 0]

    # Return today’s data and tomorrow's prediction
    return df, next_price

def plot_intraday_with_prediction(ticker: str, predicted_price: float, interval="5m", period="1d"):
    df_intraday = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df_intraday.empty:
        print("⚠️ No intraday data—market closed or wrong ticker.")
        return

    next_date = pd.bdate_range(df_intraday.index[-1] + pd.Timedelta(days=1), periods=1)[0]

    plt.figure(figsize=(10, 4), facecolor="white")
    plt.plot(df_intraday.index, df_intraday["Close"], color="navy", linewidth=1)
    plt.scatter(next_date, predicted_price, color="red", s=80, label=f'Predicted Next Close: {predicted_price:.2f}')
    plt.title(f"{ticker} Intraday + Next-Day Forecast", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45, fontsize=9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ticker = input("ENter the ticker :")  # You can change to any ticker, e.g., "TCS.NS"
    hist_df, pred = train_predict_one_day(ticker)
    plot_intraday_with_prediction(ticker, pred, interval="5m", period="1d")
    print(f"🔍 Predicted next-day close for {ticker}: {pred:.2f}")
