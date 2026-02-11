import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")
HISTORY_DAYS = 750
SEQ_LEN = 60
TRAIN_RATIO = 0.7

def fetch_and_prepare(ticker):
    print(f"Fetching {ticker}…")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=int(HISTORY_DAYS*1.5)+SEQ_LEN)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # Flatten multi-index :contentReference[oaicite:6]{index=6}

    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️ Missing columns: {missing}")
        return None

    # Clean data
    df = df[required].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    if df.empty:
        print("⚠️ No data after cleaning.")
        return None
    return df

def calc_features(df):
    df = df.copy()
    df['EMA10'] = df['Close'].ewm(span=10).mean()
    df['RSI14'] = 100 - 100 / (1 + df['Close'].diff().clip(lower=0).ewm(com=13).mean() / df['Close'].diff().clip(upper=0).mul(-1).ewm(com=13).mean())
    for i in range(1, SEQ_LEN+1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df['target'] = df['Close'].shift(-1)
    return df.dropna()

def create_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len, :-1])
        y.append(arr[i+seq_len, -1])
    return np.array(X), np.array(y)

def build_model(n_features):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def run(ticker):
    df = fetch_and_prepare(ticker)
    if df is None:
        return

    df_feat = calc_features(df)
    Xy = df_feat.drop(columns=['target']), df_feat['target']
    arr = pd.concat(Xy, axis=1).values

    X, y = create_sequences(arr, SEQ_LEN)
    if len(X) < 2:
        print("⚠️ Not enough data for sequences.")
        return

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    ns = X_train.shape[2]
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, ns)).reshape(-1, SEQ_LEN, ns)
    X_test_s = scaler.transform(X_test.reshape(-1, ns)).reshape(-1, SEQ_LEN, ns)

    y_scaler = MinMaxScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()

    model = build_model(ns)
    model.fit(X_train_s, y_train_s, validation_split=0.1,
              epochs=50, batch_size=16,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5)], verbose=0)

    pred_s = model.predict(X_test_s, verbose=0).flatten()
    pred = y_scaler.inverse_transform(pred_s.reshape(-1,1)).flatten()
    true = y[split:]

    rmse = np.sqrt(np.mean((true - pred)**2))
    mae = np.mean(np.abs(true - pred))

    last_seq = X_test_s[-1:]
    next_s = model.predict(last_seq)[0][0]
    next_price = y_scaler.inverse_transform([[next_s]])[0][0]

    print(f"\n{ticker}: Last Close ₹{df['Close'].iloc[-1]:.2f}")
    print(f"→ Predicted Next Close ₹{next_price:.2f}")
    print(f"📏 RMSE={rmse:.2f}, MAE={mae:.2f}\n")

if __name__ == "__main__":
    while True:
        t = input("Ticker (or exit): ").strip().upper()
        if t.lower() == 'exit':
            break
        run(t)



