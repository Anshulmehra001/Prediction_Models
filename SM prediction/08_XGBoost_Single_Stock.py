import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning)

TARGET_STOCK_TICKER = "ANGELONE.NS"
HISTORY_TRADING_DAYS = 750
LAG_FEATURES_COUNT = 15
TRAIN_SPLIT_RATIO = 0.8

def calculate_all_indicators(df):
    # ... same as before: SMA, EMA, RSI, MACD, Bollinger, OBV, Volatility, lagged features ...
    return df

def get_stock_prediction_xgboost(stock_ticker):
    print(f"--- DEBUG XGBoost Single Stock: {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ---")
    start = datetime.date.today() - datetime.timedelta(days=int(HISTORY_TRADING_DAYS * 1.5 + LAG_FEATURES_COUNT + 50))
    df = yf.download(stock_ticker, start=start, end=datetime.date.today(), progress=False)

    if df.empty:
        print("Error: no data fetched.")
        return None

    # 🔧 Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    numeric = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    if not numeric:
        print("Error: no OHLCV columns found after flattening.")
        return None

    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric, inplace=True)
    if df.empty:
        print("Error: no valid OHLCV data.")
        return None

    df = calculate_all_indicators(df)
    df['Target_Next_Close'] = df['Close'].shift(-1)

    features = [c for c in df.columns if c != 'Target_Next_Close']
    numeric_feats = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    # Optionally filter for variance, non-null, etc...

    data = df[numeric_feats + ['Target_Next_Close']].dropna()
    if len(data) < 2:
        print("Error: insufficient clean data.")
        return None

    X = data[numeric_feats].to_numpy()
    y = np.ravel(data['Target_Next_Close'].to_numpy())
    assert y.ndim == 1

    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    Xs = scaler_X.fit_transform(X)
    ys = np.ravel(scaler_y.fit_transform(y.reshape(-1, 1)))

    split = int(len(Xs) * TRAIN_SPLIT_RATIO)
    X_tr, y_tr = Xs[:split], ys[:split]
    assert y_tr.ndim == 1

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)

    Xin = Xs[-1].reshape(1, -1)
    ps = model.predict(Xin)[0]
    p = round(scaler_y.inverse_transform([[ps]])[0][0], 2)

    last_price = df['Close'].iloc[-1]
    change_pct = (p - last_price) / last_price * 100
    pred_date = (df.index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    name = yf.Ticker(stock_ticker).info.get('longName', stock_ticker)

    return {
        "ticker": stock_ticker,
        "name": name,
        "latest_close": last_price,
        "latest_close_date": df.index[-1].strftime('%Y-%m-%d'),
        "predicted_next_close": p,
        "predicted_date": pred_date,
        "predicted_change_percentage": change_pct
    }

if __name__ == "__main__":
    print("🛠️ Starting prediction...\n")
    res = get_stock_prediction_xgboost(TARGET_STOCK_TICKER)
    print("\n" + "="*60)
    if res:
        arrow = '▲' if res['predicted_change_percentage'] >= 0 else '▼'
        print(f"{res['name']} ({res['ticker']}):")
        print(f"  Latest Close ({res['latest_close_date']}): ₹{res['latest_close']:.2f}")
        print(f"  Predicted Close ({res['predicted_date']}): ₹{res['predicted_next_close']:.2f} {arrow}")
        print(f"  Change: {res['predicted_change_percentage']:.2f}%")
    else:
        print("No prediction could be generated.")
    print("\nDisclaimer: educational/demo only.")
