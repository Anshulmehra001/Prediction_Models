import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import yfinance as yf
import joblib, os

class SimpleStockPredictor:
    def __init__(self, symbol="TCS.NS"):
        self.symbol = symbol
        self.model = None
        self.load_model()

    def get_data(self):
        df = yf.download(self.symbol, period="3mo", interval="1d")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
        return df

    def load_model(self):
        if os.path.exists("simple_model.pkl"):
            self.model = joblib.load("simple_model.pkl")
        else:
            self.train_model()

    def train_model(self):
        df = self.get_data()
        X = df[['SMA_5', 'SMA_20', 'Daily_Return', 'Volume']]
        y = df['Target']

        imp = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, "simple_model.pkl")

    def predict_next_day(self):
        df = self.get_data()
        if df.empty:
            print("Prediction failed: no data available.")
            return None
        if self.model is None:
            print("Prediction failed: model is not loaded.")
            return None

        latest = df.iloc[-1][['SMA_5', 'SMA_20', 'Daily_Return', 'Volume']].values.reshape(1, -1)
        pred_arr = self.model.predict(latest)

        # Ensure scalar floats for comparisons
        pred = float(pred_arr[0]) if hasattr(pred_arr, "__len__") else float(pred_arr)
        current = float(df.iloc[-1]['Close'])

        return {
            'symbol': self.symbol,
            'current_price': round(current, 2),
            'predicted_price': round(pred, 2),
            'action': 'BUY' if pred > current else 'SELL',
            'confidence': round(abs(pred - current) / current, 2)
        }

if __name__ == "__main__":
    print("Simple Stock Predictor")
    predictor = SimpleStockPredictor()
    result = predictor.predict_next_day()

    if result:
        print("\nPrediction Results:")
        for k, v in result.items():
            print(f"{k.replace('_', ' ').title():<20}: {v}")
    else:
        print("\nERROR: Could not generate prediction")
        print("SOLUTION: Try a different symbol or check data/model availability")
