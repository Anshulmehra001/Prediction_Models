import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
import warnings

warnings.filterwarnings("ignore", module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning)

HISTORY_TRADING_DAYS = 750
SEQ_LEN = 60
TRAIN_RATIO = 0.70

def calc_indicators(df):
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['EMA10'] = df['Close'].ewm(span=10).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain, avg_loss = gain.ewm(com=13).mean(), loss.ewm(com=13).mean()
    rs = (avg_gain / avg_loss.replace(0, np.nan)).fillna(0)
    df['RSI14'] = 100 - 100/(1 + rs)
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACDsig'] = df['MACD'].ewm(span=9).mean()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    for i in range(1, SEQ_LEN+1):
        df[f'Lag{i}'] = df['Close'].shift(i)
    return df.dropna()

def create_seqs(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(50, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_ticker(ticker):
    print(f"\n📈 Fetching {ticker}")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=int(HISTORY_TRADING_DAYS*1.5)+SEQ_LEN)

    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    required = ['Open','High','Low','Close','Volume']
    df.dropna(subset=['Close'], inplace=True)
    if df.empty or not all(col in df.columns for col in required):
        print("⚠️ Data not sufficient."); return

    df = calc_indicators(df)
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    feats = [c for c in df.columns if c not in ['Open','High','Low','Volume','Target']]
    arr = df[feats + ['Target']].values

    scaler = MinMaxScaler()
    arr_s = scaler.fit_transform(arr)
    X, y = create_seqs(arr_s, SEQ_LEN)

    split = int(len(X)*TRAIN_RATIO)
    X_train, y_train = X[:split], y[:split]
    X_temp, y_temp = X[split:], y[split:]
    val_split = int(len(X_temp)*0.5)
    X_val, y_val = X_temp[:val_split], y_temp[:val_split]
    X_test, y_test = X_temp[val_split:], y_temp[val_split:]

    model = build_model((SEQ_LEN, X.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=16,
        callbacks=[es], verbose=0
    )

    test_loss = model.evaluate(X_test, y_test, verbose=0)
    test_pred = model.predict(X_test, verbose=0).flatten()
    # Compute unscaled RMSE, MAE
    y_inv = scaler.inverse_transform(np.hstack([np.zeros((len(test_pred), arr.shape[1]-1)), test_pred.reshape(-1,1)]))[:, -1]
    y_true_inv = scaler.inverse_transform(np.hstack([np.zeros((len(y_test), arr.shape[1]-1)), y_test.reshape(-1,1)]))[:, -1]

    rmse = np.sqrt(np.mean((y_true_inv - y_inv)**2))
    mae  = np.mean(np.abs(y_true_inv - y_inv))

    last_seq = arr_s[-SEQ_LEN:, :-1].reshape(1, SEQ_LEN, -1)
    pred_scaled = model.predict(last_seq)[0][0]
    inv_full = scaler.inverse_transform(np.hstack([np.zeros((1, arr.shape[1]-1)), [[pred_scaled]]]))
    next_pred = inv_full[0, -1]

    return {
        'ticker': ticker,
        'last': round(df.iloc[-1]['Close'], 2),
        'predicted': round(next_pred, 2),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2)
    }

if __name__ == "__main__":
    print("🔍 Stacked Bi-LSTM Stock Predictor")
    while True:
        t = input("Ticker (or exit): ").strip()
        if t.lower() == 'exit':
            print("👋 Goodbye!")
            break
        res = predict_ticker(t.upper())
        if res:
            print(f"{res['ticker']}: Last ₹{res['last']} → Predicted ₹{res['predicted']}")
            print(f"   📏 Test RMSE: {res['RMSE']}, MAE: {res['MAE']}\n")
