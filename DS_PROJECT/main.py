import pandas as pd
import tensorflow as tf
import pandas_ta as ta
from utils.news_scraper import IndianNewsScraper
from models.train_lstm import create_lstm_model
from models.train_xgboost import train_xgboost
import joblib
import numpy as np
from typing import Dict, Any


class IndianStockPredictor:
    def __init__(self, symbol: str = "RELIANCE") -> None:
        self.symbol = symbol
        self.news_scraper = IndianNewsScraper()
        self.lstm_model = None
        self.xgb_model = None
        self.load_models()

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess stock data with technical indicators"""
        try:
            df = pd.read_csv(f"data/{self.symbol}_intraday.csv", parse_dates=['Date'])
            df = self._calculate_indicators(df)
            return df.dropna()
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file for {self.symbol} not found")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using pandas-ta"""
        # Ensure consistent column naming
        df = df.rename(columns={
            'close': 'Close',
            'volume': 'Volume',
            'open': 'Open',
            'high': 'High',
            'low': 'Low'
        })

        # Calculate indicators
        df.ta.rsi(close='Close', length=14, append=True, col_names=('RSI_14',))
        df.ta.macd(close='Close', append=True, col_names=('MACD', 'MACD_Signal', 'MACD_Hist'))
        df.ta.vwap(high='High', low='Low', close='Close', volume='Volume', append=True, col_names=('VWAP',))

        # Add any custom indicators here
        df['FII_DII_Impact'] = df['Close'].pct_change() * 100  # Example placeholder

        return df

    def load_models(self) -> None:
        """Load trained models or train new ones if not available"""
        try:
            self.lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
            self.xgb_model = joblib.load("models/xgboost_model.pkl")
        except (OSError, Exception) as e:
            print(f"Model loading failed: {str(e)}. Training new models...")
            self.train_models()

    def train_models(self) -> None:
        """Train and save LSTM and XGBoost models"""
        df = self.load_data()
        features = ['RSI_14', 'MACD', 'VWAP', 'FII_DII_Impact']
        target = 'Close'

        X = df[features].values
        y = df[target].shift(-1).dropna().values  # Predict next day's close

        # Train LSTM
        X_3d = X[:-1].reshape(X.shape[0] - 1, 1, X.shape[1])
        self.lstm_model = create_lstm_model(input_shape=(1, X.shape[1]))
        self.lstm_model.fit(X_3d, y, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

        # Train XGBoost
        self.xgb_model = train_xgboost(X[:-1], y)

        # Save models
        self.lstm_model.save("models/lstm_model.h5")
        joblib.dump(self.xgb_model, "models/xgboost_model.pkl")

    def predict_next_move(self) -> Dict[str, Any]:
        """Generate trading prediction using ensemble approach"""
        df = self.load_data()
        features = ['RSI_14', 'MACD', 'VWAP', 'FII_DII_Impact']
        latest = df[features].iloc[-1].values
        current_price = df['Close'].iloc[-1]

        # LSTM prediction
        lstm_pred = self.lstm_model.predict(latest.reshape(1, 1, len(latest)), verbose=0)[0][0]

        # XGBoost prediction
        xgb_pred = self.xgb_model.predict([latest])[0]

        # News sentiment analysis
        sentiment = self.news_scraper.get_moneycontrol_news()  # Normalized between -1 and 1

        # Ensemble prediction with weighted average
        weights = {'lstm': 0.5, 'xgb': 0.3, 'sentiment': 0.2}
        final_pred = (
                weights['lstm'] * lstm_pred +
                weights['xgb'] * xgb_pred +
                weights['sentiment'] * sentiment * current_price
        )

        return {
            "symbol": self.symbol,
            "current_price": round(float(current_price), 2),
            "lstm_prediction": round(float(lstm_pred), 2),
            "xgboost_prediction": round(float(xgb_pred), 2),
            "sentiment_score": round(float(sentiment), 4),
            "final_prediction": round(float(final_pred), 2),
            "recommendation": "BUY" if final_pred > current_price else "SELL",
            "confidence": abs(final_pred - current_price) / current_price  # Relative difference
        }


if __name__ == "__main__":
    try:
        predictor = IndianStockPredictor("RELIANCE")
        prediction = predictor.predict_next_move()
        print("\nStock Prediction Analysis:")
        for key, value in prediction.items():
            print(f"{key.replace('_', ' ').title():<20}: {value}")
    except Exception as e:
        print(f"Error in prediction pipeline: {str(e)}")