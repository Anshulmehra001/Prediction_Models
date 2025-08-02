import os
import warnings
import uuid
import threading
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import yfinance as yf
import pandas_ta as ta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import RobustScaler
from dotenv import load_dotenv
import functools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D
from transformers import pipeline
import requests
from datetime import datetime

# --- Import Flask and related components ---
from flask import Flask, render_template_string, jsonify, request

# --- Configuration & Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY_HERE")

# --- Global NLP Model (Initialize once) ---
NLP_SENTIMENT_PIPELINE = None
try:
    NLP_SENTIMENT_PIPELINE = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone",
                                      tokenizer="yiyanghkust/finbert-tone")
    print("AI Sentiment Model loaded successfully.")
except Exception as e:
    print(f"Could not initialize NLP pipeline. News sentiment will be disabled. Error: {e}")

# --- Global Task Management ---
RESULTS = {}
task_lock = threading.Lock()


class PhoenixForecaster:
    def __init__(self, symbol, period="3y", task_id=None):
        self.symbol = symbol
        self.period = period
        self.task_id = task_id
        self.raw_df = self._fetch_data()
        self.TARGET = 'Close'
        self.SEQ_LEN = 60
        self.models = {
            "LGBM": lgb.LGBMRegressor(random_state=42, verbose=-1),
            "XGB": xgb.XGBRegressor(random_state=42, verbosity=0)
        }

    def _update_status(self, message, progress=None):
        if self.task_id:
            with task_lock:
                if self.task_id not in RESULTS: RESULTS[self.task_id] = {}
                RESULTS[self.task_id]['status'] = 'running'
                RESULTS[self.task_id]['message'] = message
                if progress is not None: RESULTS[self.task_id]['progress'] = progress

    @functools.lru_cache(maxsize=2)
    def _fetch_data(self):
        self._update_status(f"Fetching {self.period} of data for {self.symbol}...")
        df = yf.download(self.symbol, period=self.period, auto_adjust=True, progress=False)
        if df.empty: raise ValueError(f"No data found for '{self.symbol}'.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(col).title().replace(' ', '_') for col in df.columns]

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.asfreq('B', method='pad')
        return df

    @functools.lru_cache(maxsize=1024)
    def _get_news_sentiment(self, query):
        if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE" or not NLP_SENTIMENT_PIPELINE: return 0.0
        try:
            url = "https://newsapi.org/v2/everything"
            params = {'q': query, 'language': 'en', 'sortBy': 'relevancy', 'pageSize': 10, 'apiKey': NEWS_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            if not articles: return 0.0
            scores = [NLP_SENTIMENT_PIPELINE(a['title'])[0] for a in articles]
            sentiment_values = [(s['score'] if s['label'].lower() == 'positive' else -s['score']) for s in scores if
                                s['label'].lower() != 'neutral']
            return np.mean(sentiment_values) if sentiment_values else 0.0
        except Exception:
            return 0.0

    def _engineer_features(self, df_in):
        df = df_in.copy()
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        sentiment_score = self._get_news_sentiment(self.symbol.split('.')[0])
        df['Sentiment'] = sentiment_score
        return df.dropna()

    def _build_cnn_lstm_model(self, n_features):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal',
                   input_shape=(self.SEQ_LEN, n_features)),
            LSTM(64, return_sequences=True), Dropout(0.2),
            LSTM(32), Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _generate_suggestion(self, prediction, last_close, atr):
        change_pct = (prediction / last_close - 1) * 100
        atr_pct = (atr / last_close) * 100

        suggestion = {"text": "", "signal": "hold", "subtext": f"Recent daily volatility (ATR) is ~{atr_pct:.2f}%."}
        if change_pct > atr_pct / 1.5:
            suggestion.update({
                                  "text": f"Strong Bullish signal. Prediction ({prediction:.2f}) indicates a significant {change_pct:.2f}% potential increase.",
                                  "signal": "buy"})
        elif change_pct > 0.2:
            suggestion.update({
                                  "text": f"Slightly Bullish. Prediction ({prediction:.2f}) suggests a possible {change_pct:.2f}% upside.",
                                  "signal": "buy"})
        elif change_pct < -(atr_pct / 1.5):
            suggestion.update({
                                  "text": f"Strong Bearish signal. Prediction ({prediction:.2f}) indicates a significant {change_pct:.2f}% potential decrease.",
                                  "signal": "sell"})
        elif change_pct < -0.2:
            suggestion.update({
                                  "text": f"Slightly Bearish. Prediction ({prediction:.2f}) suggests a possible {change_pct:.2f}% downside.",
                                  "signal": "sell"})
        else:
            suggestion[
                "text"] = f"Neutral / Hold. Predicted change of {change_pct:.2f}% is within the normal market noise."

        return suggestion

    def predict(self):
        self._update_status("Starting prediction process...")
        full_df = self.raw_df.copy()

        try:
            arima_model = ARIMA(full_df['Close'], order=(5, 1, 0)).fit()
            arima_forecast_val = arima_model.forecast(steps=1).iloc[0]
            residuals = arima_model.resid
        except Exception:
            arima_forecast_val = full_df['Close'].iloc[-1]
            residuals = full_df['Close'].diff().fillna(0)

        self._update_status("Engineering features for full history...")
        featured_df = self._engineer_features(full_df)
        featured_df['Residuals'] = residuals
        featured_df = featured_df.dropna()
        if len(featured_df) < self.SEQ_LEN: raise ValueError("Not enough data to make a prediction.")

        features = [col for col in featured_df.columns if col != 'Residuals']
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(featured_df[features + ['Residuals']])

        self._update_status("Training deep learning model...")
        X_seq, y_res_seq = [], []
        for j in range(self.SEQ_LEN, len(scaled_data)):
            X_seq.append(scaled_data[j - self.SEQ_LEN:j, :-1])
            y_res_seq.append(scaled_data[j, -1])
        X_seq, y_res_seq = np.array(X_seq), np.array(y_res_seq)

        cnn_lstm_model = self._build_cnn_lstm_model(X_seq.shape[2])
        cnn_lstm_model.fit(X_seq, y_res_seq, epochs=15, batch_size=16, verbose=0, shuffle=False)

        self._update_status("Training ensemble models...")
        X_tab, y_res_tab = scaled_data[self.SEQ_LEN - 1:-1, :-1], scaled_data[self.SEQ_LEN:, -1]
        final_model = None
        for name, model in self.models.items():
            model.fit(X_tab, y_res_tab)
            if name == "LGBM": final_model = model

        self._update_status("Generating final forecast...")
        test_sequence_scaled = scaled_data[-self.SEQ_LEN:, :-1].reshape(1, self.SEQ_LEN, X_seq.shape[2])
        test_vector_scaled = scaled_data[-1, :-1].reshape(1, -1)

        cnn_lstm_res_pred_scaled = cnn_lstm_model.predict(test_sequence_scaled, verbose=0)[0, 0]
        xgb_res_pred_scaled = self.models['XGB'].predict(test_vector_scaled)[0]
        lgbm_res_pred_scaled = self.models['LGBM'].predict(test_vector_scaled)[0]

        component_preds = {
            "CNN-LSTM": cnn_lstm_res_pred_scaled,
            "XGBoost": xgb_res_pred_scaled,
            "LightGBM": lgbm_res_pred_scaled
        }

        ensemble_res_pred_scaled = np.mean(list(component_preds.values()))

        dummy_for_inverse = np.zeros((1, len(features) + 1))
        dummy_for_inverse[0, -1] = ensemble_res_pred_scaled
        ensemble_res_pred = scaler.inverse_transform(dummy_for_inverse)[0, -1]

        final_prediction = arima_forecast_val + ensemble_res_pred

        last_close = full_df['Close'].iloc[-1]
        atr_val = featured_df['ATRr_14'].iloc[-1]
        suggestion = self._generate_suggestion(final_prediction, last_close, atr_val)

        prediction_analysis = {
            'last_close': f"{last_close:.2f}",
            'arima_baseline': f"{arima_forecast_val:.2f}",
            'ensemble_residual': f"{ensemble_res_pred:+.2f}",
            'final_prediction': f"{final_prediction:.2f}",
            'component_predictions': {k: f"{v:+.3f}" for k, v in component_preds.items()},
            'suggestion': suggestion
        }

        # Chart data for prediction
        hist_df = self.raw_df.iloc[-self.SEQ_LEN:]
        pred_date = hist_df.index[-1] + pd.Timedelta(days=1)
        chart_data = {
            'dates': hist_df.index.strftime('%Y-%m-%d').tolist() + [pred_date.strftime('%Y-%m-%d')],
            'actuals': hist_df['Close'].tolist() + [None],
            'predictions': [None] * self.SEQ_LEN + [final_prediction]
        }

        feature_importance = sorted(zip(final_model.feature_name_, final_model.feature_importances_),
                                    key=lambda x: x[1], reverse=True)

        with task_lock:
            RESULTS[self.task_id] = {
                'status': 'complete', 'type': 'forecast', 'symbol': self.symbol,
                'analysis': prediction_analysis,
                'chart_data': chart_data,
                'feature_importance': [{'feature': f, 'importance': int(i)} for f, i in feature_importance]
            }

    def backtest(self, backtest_days=90):
        if len(self.raw_df) < backtest_days + self.SEQ_LEN + 50:
            raise ValueError("Not enough historical data for the requested backtest period.")

        all_daily_results = []
        final_model_for_importance = None
        start_index = len(self.raw_df) - backtest_days

        for i in range(start_index, len(self.raw_df)):
            progress = ((i - start_index + 1) / backtest_days) * 100
            current_date_str = self.raw_df.index[i].strftime('%Y-%m-%d')
            self._update_status(
                f"Day {i - start_index + 1}/{backtest_days} ({current_date_str}) - Retraining models...", progress)

            train_df = self.raw_df.iloc[:i]

            try:
                arima_model = ARIMA(train_df['Close'], order=(5, 1, 0)).fit()
                arima_forecast_val = arima_model.forecast(steps=1).iloc[0]
                residuals = arima_model.resid
            except Exception:
                arima_forecast_val = train_df['Close'].iloc[-1]
                residuals = train_df['Close'].diff().fillna(0)

            train_featured_df = self._engineer_features(train_df)
            train_featured_df['Residuals'] = residuals
            train_featured_df = train_featured_df.dropna()

            if len(train_featured_df) < self.SEQ_LEN: continue

            features = [col for col in train_featured_df.columns if col != 'Residuals']
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(train_featured_df[features + ['Residuals']])

            X_seq, y_res_seq = [], []
            for j in range(self.SEQ_LEN, len(scaled_data)):
                X_seq.append(scaled_data[j - self.SEQ_LEN:j, :-1])
                y_res_seq.append(scaled_data[j, -1])
            X_seq, y_res_seq = np.array(X_seq), np.array(y_res_seq)

            if len(X_seq) == 0: continue

            cnn_lstm_model = self._build_cnn_lstm_model(X_seq.shape[2])
            cnn_lstm_model.fit(X_seq, y_res_seq, epochs=10, batch_size=16, verbose=0, shuffle=False)

            X_tab, y_res_tab = scaled_data[self.SEQ_LEN - 1:-1, :-1], scaled_data[self.SEQ_LEN:, -1]
            for name, model in self.models.items():
                model.fit(X_tab, y_res_tab)
                if name == "LGBM": final_model_for_importance = model

            test_sequence_scaled = scaled_data[-self.SEQ_LEN:, :-1].reshape(1, self.SEQ_LEN, X_seq.shape[2])
            test_vector_scaled = scaled_data[-1, :-1].reshape(1, -1)

            cnn_lstm_res_pred_scaled = cnn_lstm_model.predict(test_sequence_scaled, verbose=0)[0, 0]
            xgb_res_pred_scaled = self.models['XGB'].predict(test_vector_scaled)[0]
            lgbm_res_pred_scaled = self.models['LGBM'].predict(test_vector_scaled)[0]

            ensemble_res_pred_scaled = np.mean([cnn_lstm_res_pred_scaled, xgb_res_pred_scaled, lgbm_res_pred_scaled])

            dummy_for_inverse = np.zeros((1, len(features) + 1))
            dummy_for_inverse[0, -1] = ensemble_res_pred_scaled
            ensemble_res_pred = scaler.inverse_transform(dummy_for_inverse)[0, -1]

            final_prediction = arima_forecast_val + ensemble_res_pred

            day_result = {
                'Date': self.raw_df.index[i],
                'ARIMA_Baseline': arima_forecast_val,
                'Ensemble_Residual': ensemble_res_pred,
                'Final_Forecast': final_prediction
            }
            all_daily_results.append(day_result)

        if not all_daily_results:
            raise ValueError("Backtest failed to produce any results.")

        results_df = pd.DataFrame(all_daily_results).set_index('Date')
        self._generate_detailed_report(results_df, final_model_for_importance)

    def _generate_detailed_report(self, results_df, model):
        self._update_status("Generating final forensic report...", 100)

        trade_log = self.raw_df.join(results_df, how='inner')

        signal = np.where(trade_log['Final_Forecast'] > trade_log['Final_Forecast'].shift(1), 1, -1)
        trade_log['signal'] = pd.Series(signal, index=trade_log.index).shift(1)
        trade_log['Action'] = trade_log['signal'].map({1.0: 'BUY', -1.0: 'SELL'}).fillna('HOLD')

        trade_log.rename(columns={'Close': 'Actual'}, inplace=True)
        trade_log['Prediction_Error_%'] = ((trade_log['Actual'] - trade_log['Final_Forecast']) / trade_log[
            'Actual']) * 100
        trade_log['Daily_PnL_%'] = trade_log['Actual'].pct_change() * 100 * trade_log['signal']
        trade_log.fillna(0, inplace=True)
        trade_log['Equity_Curve'] = (1 + trade_log['Daily_PnL_%'] / 100).cumprod()

        report = {}
        strategy_returns = trade_log['Daily_PnL_%'] / 100
        report['total_return'] = f"{(trade_log['Equity_Curve'].iloc[-1] - 1):.2%}"
        report['sharpe_ratio'] = f"{(strategy_returns.mean() / (strategy_returns.std() + 1e-10)) * np.sqrt(252):.2f}"
        report['max_drawdown'] = f"{(trade_log['Equity_Curve'] / trade_log['Equity_Curve'].cummax() - 1).min():.2%}"

        first_price = trade_log['Actual'].iloc[0]
        last_price = trade_log['Actual'].iloc[-1]
        report['benchmark_return'] = f"{(last_price / first_price - 1):.2%}"

        trades = trade_log[trade_log['Action'] != 'HOLD']
        if not trades.empty:
            wins = trades[trades['Daily_PnL_%'] > 0]
            report['win_rate'] = f"{len(wins) / len(trades):.2%}"
        else:
            report['win_rate'] = "N/A"

        chart_data = {
            'vs_actual': {
                'dates': trade_log.index.strftime('%Y-%m-%d').tolist(),
                'actuals': trade_log['Actual'].tolist(),
                'predictions': trade_log['Final_Forecast'].tolist(),
                'arima': trade_log['ARIMA_Baseline'].tolist()
            },
            'equity_curve': {'dates': trade_log.index.strftime('%Y-%m-%d').tolist(),
                             'values': trade_log['Equity_Curve'].tolist()},
            'drawdown': {'dates': trade_log.index.strftime('%Y-%m-%d').tolist(),
                         'values': (trade_log['Equity_Curve'] / trade_log['Equity_Curve'].cummax() - 1).tolist()}
        }

        display_cols = {
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Actual': 'Actual',
            'ARIMA_Baseline': 'ARIMA', 'Ensemble_Residual': 'Residual', 'Final_Forecast': 'Forecast',
            'Prediction_Error_%': 'Error %', 'Action': 'Action', 'Daily_PnL_%': 'PnL %', 'Equity_Curve': 'Equity'
        }
        display_log = trade_log[list(display_cols.keys())].copy()
        display_log.rename(columns=display_cols, inplace=True)

        formatters = {col: "{:,.2f}".format for col in display_log.columns if
                      display_log[col].dtype == 'float64' and col not in ['Error %', 'PnL %']}
        formatters['Error %'] = "{:,.2f}%".format
        formatters['PnL %'] = "{:+.2f}%".format

        styler = display_log.style.format(formatters).set_properties(**{
            'text-align': 'right', 'padding': '6px', 'font-size': '0.9em'
        }).set_properties(subset=['Action'], **{
            'text-align': 'center', 'font-weight': 'bold'
        })

        styler = styler.applymap(lambda x: 'color: #20c997' if x > 0 else 'color: #dc3545' if x < 0 else '',
                                 subset=['PnL %'])

        trade_log_html = styler.to_html(classes='table table-sm table-striped table-hover table-dark', border=0)

        feature_importance = sorted(zip(model.feature_name_, model.feature_importances_), key=lambda x: x[1],
                                    reverse=True)

        with task_lock:
            RESULTS[self.task_id] = {
                'status': 'complete', 'type': 'backtest', 'symbol': self.symbol,
                'report': report, 'trade_log': trade_log_html,
                'chart_data': chart_data,
                'feature_importance': [{'feature': f, 'importance': int(i)} for f, i in feature_importance]
            }


# --- Flask App ---
app = Flask(__name__)

PHOENIX_LOGO_SVG = """
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="d-inline-block align-text-top me-2">
<path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2Z" stroke="#0dcaf0" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M15.48 12.63C15.48 12.63 15.11 14.82 12 16.32C8.89 14.82 8.52 12.63 8.52 12.63C8.52 12.63 9.47 10.97 12 7.68C14.53 10.97 15.48 12.63 15.48 12.63Z" stroke="#ffc107" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M12 16.32V19.32" stroke="#ffc107" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M10.13 18.32L12 19.32L13.87 18.32" stroke="#ffc107" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M7.5 12.32L5 11.32" stroke="#ffc107" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M16.5 12.32L19 11.32" stroke="#ffc107" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <title>Phoenix Forecaster</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', system-ui, sans-serif; background-color: #0a0a0a; }
        .card { background-color: #1a1a1a; border: 1px solid #333; }
        .kpi-card .value { font-size: 2rem; font-weight: 700; }
        .kpi-card .label { font-size: 0.8rem; color: #adb5bd; text-transform: uppercase; letter-spacing: 0.5px;}
        .text-success { color: #20c997 !important; }
        .text-danger { color: #dc3545 !important; }
        .chart-container { position: relative; height: 400px; width: 100%; }
        #suggestion-card .signal-icon { font-size: 2.5rem; }
        .btn-info { background-color: #0dcaf0; border-color: #0dcaf0; }
        .form-control:focus, .btn:focus { box-shadow: 0 0 0 0.25rem rgba(13, 202, 240, 0.5); }
        .table-responsive { max-height: 600px; }
        .table th { position: sticky; top: 0; background-color: #2c2c2c; z-index: 1; }
        .list-group-item { background-color: #2a2a2a; border-color: #333; }
    </style>
</head>
<body>
    <div class="container-fluid my-4">
        <header class="text-center mb-4">
            <h1 class="display-4">{logo_placeholder} Phoenix Forecaster</h1>
            <p class="lead text-muted">Grand Ensemble AI Forecasting & Analytics</p>
        </header>

        <div class="row">
            <div class="col-lg-10 mx-auto">
                <div class="card shadow-sm mb-4">
                    <div class="card-body p-4">
                        <div class="row">
                            <div class="col-md-12 mb-3">
                                <label for="ticker" class="form-label">Stock/Crypto Ticker</label>
                                <input type="text" class="form-control form-control-lg" id="ticker" value="TCS.NS">
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-6 mb-3 mb-md-0">
                                <div class="card h-100">
                                    <div class="card-body d-flex flex-column">
                                        <h5 class="card-title"><i class="bi bi-graph-up-arrow"></i> Forensic Backtesting</h5>
                                        <p class="card-text text-muted small">Simulate the model's historical performance with a genuine walk-forward analysis.</p>
                                        <div class="mt-auto">
                                            <div class="mb-3">
                                                <label for="days" class="form-label">Period to Backtest (Days)</label>
                                                <input type="number" class="form-control" id="days" value="30">
                                            </div>
                                            <div class="d-grid">
                                                <button class="btn btn-primary" id="runBtn" onclick="startTask('backtest')">Run Genuine Backtest</button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card h-100">
                                    <div class="card-body d-flex flex-column">
                                        <h5 class="card-title"><i class="bi bi-binoculars-fill"></i> Future Prediction</h5>
                                        <p class="card-text text-muted small">Train the ensemble on all available data to generate the most accurate forecast for the next trading day.</p>
                                        <div class="d-grid mt-auto">
                                            <button class="btn btn-info text-dark" id="forecastBtn" onclick="startTask('forecast')">Predict Next Day</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="progress-container" class="my-4" style="display: none;">
             <div class="card shadow-sm"><div class="card-body">
                <h4 id="progress-title" class="text-center"></h4>
                <div class="progress" style="height: 30px;"><div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated bg-info" role="progressbar" style="width: 0%;"></div></div>
                <p id="progress-message" class="text-center mt-2 small"></p>
            </div></div>
        </div>
        <div id="error-container" class="my-4" style="display: none;"><div class="alert alert-danger"><h4>An Error Occurred</h4><p id="error-message"></p></div></div>

        <div id="results-container" class="mt-4" style="display:none;">
            <h2 id="results-title" class="text-center mb-4"></h2>
            <!-- Main Row -->
            <div class="row g-4">
                <!-- Left Column -->
                <div class="col-xl-4">
                    <div id="kpi-card" class="card"><div class="card-body" id="kpi-container"></div></div>
                    <div id="prediction-analysis-card" class="card mt-4" style="display:none;"><div class="card-body" id="prediction-analysis-container"></div></div>
                </div>
                <!-- Right Column -->
                <div class="col-xl-8">
                    <div class="card"><div class="card-body">
                        <h5 class="card-title" id="main-chart-title"></h5>
                        <div class="chart-container"><canvas id="main-chart"></canvas></div>
                    </div></div>
                    <div class="card mt-4"><div class="card-body">
                        <h5 class="card-title" id="secondary-chart-title"></h5>
                        <div class="chart-container"><canvas id="secondary-chart"></canvas></div>
                    </div></div>
                </div>
            </div>
            <!-- Log Row -->
            <div id="log-wrapper" class="row mt-4" style="display: none;"><div class="col-12"><div class="card"><div class="card-body"><h5 class="card-title">Forensic Trade Log</h5><div id="trade-log-container" class="table-responsive border rounded"></div></div></div></div></div>
        </div>
    </div>
<script>
    let charts = {};
    function createOrUpdateChart(ctx, type, data, options) {
        if (charts[ctx.canvas.id]) { charts[ctx.canvas.id].destroy(); }
        const defaultOptions = {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#ccc' } } },
            scales: { y: { ticks: { color: '#ccc' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } }, x: { ticks: { color: '#ccc' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } } }
        };
        charts[ctx.canvas.id] = new Chart(ctx, { type, data, options: {...defaultOptions, ...options} });
    }

    async function startTask(taskType) {
        const ticker = document.getElementById('ticker').value.toUpperCase();
        if (!ticker) { alert('Please enter a ticker.'); return; }
        const days = parseInt(document.getElementById('days').value);

        document.getElementById('runBtn').disabled = true;
        document.getElementById('forecastBtn').disabled = true;

        document.getElementById('progress-container').style.display = 'block';
        document.getElementById('results-container').style.display = 'none';
        document.getElementById('error-container').style.display = 'none';

        let endpoint = '';
        let payload = {};
        if (taskType === 'backtest') {
            document.getElementById('progress-title').textContent = 'Backtest in Progress...';
            endpoint = '/start_backtest';
            payload = {symbol: ticker, days: days};
        } else {
            document.getElementById('progress-title').textContent = 'Prediction in Progress...';
            endpoint = '/start_predict';
            payload = {symbol: ticker};
        }

        try {
            const response = await fetch(endpoint, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
            const data = await response.json();
            if (!response.ok) { throw new Error(data.message || `HTTP error!`); }
            const intervalId = setInterval(() => checkStatus(data.task_id, intervalId, taskType), 2000);
        } catch (error) {
            showError(`Failed to start task: ${error.message}`);
        }
    }

    async function checkStatus(taskId, intervalId, taskType) {
        try {
            const response = await fetch(`/task_status/${taskId}`);
            const data = await response.json();
            if (!response.ok) { throw new Error(data.message || `HTTP error!`); }

            document.getElementById('progress-message').textContent = data.message || '';
            if (data.progress) { document.getElementById('progress-bar').style.width = `${data.progress}%`; }

            if (data.status === 'complete') {
                clearInterval(intervalId);
                document.getElementById('progress-container').style.display = 'none';
                displayResults(data);
                resetButtons();
            } else if (data.status === 'error') {
                showError(data.message);
                clearInterval(intervalId);
            }
        } catch (error) {
            showError(`Failed to get status: ${error.message}`);
            clearInterval(intervalId);
        }
    }

    function displayResults(data) {
        document.getElementById('results-container').style.display = 'block';
        document.getElementById('results-title').textContent = `${data.type.charAt(0).toUpperCase() + data.type.slice(1)} Results for ${data.symbol}`;

        // Hide all optional components first
        document.getElementById('prediction-analysis-card').style.display = 'none';
        document.getElementById('log-wrapper').style.display = 'none';

        if (data.type === 'forecast') {
            displayPredictionResult(data);
        } else { // backtest
            displayBacktestResults(data);
        }
    }

    function displayPredictionResult(data) {
        const pa = data.analysis;
        const s = pa.suggestion;

        document.getElementById('kpi-container').innerHTML = `
            <div class="kpi-card p-3 mb-3"><div class="label">Last Close</div><div class="value">${pa.last_close}</div></div>
            <div class="kpi-card p-3"><div class="label">AI Price Forecast</div><div class="value">${pa.final_prediction}</div></div>
        `;

        document.getElementById('prediction-analysis-card').style.display = 'block';

        let residual_val = parseFloat(pa.ensemble_residual);
        let final_val = parseFloat(pa.final_prediction);
        let last_close_val = parseFloat(pa.last_close);

        let residual_class = residual_val >= 0 ? 'bg-success' : 'bg-danger';
        let final_class = final_val >= last_close_val ? 'text-success' : 'text-danger';
        let final_badge_class = final_val >= last_close_val ? 'bg-success' : 'bg-danger';

        let analysisHTML = `
            <h5 class="card-title">Prediction Analysis</h5>
            <ul class="list-group">
                <li class="list-group-item d-flex justify-content-between align-items-center"><i class="bi bi-bricks me-2"></i> ARIMA Trend <span class="badge bg-primary">${pa.arima_baseline}</span></li>
                <li class="list-group-item d-flex justify-content-between align-items-center"><i class="bi bi-motherboard me-2"></i> ML Residual <span class="badge ${residual_class}">${pa.ensemble_residual}</span></li>
            </ul>
            <hr>
            <p class="text-muted small">Component Model Forecasts (for Residual)</p>
            <ul class="list-group list-group-flush">
        `;
        for (const [name, value] of Object.entries(pa.component_predictions)) {
            let comp_val = parseFloat(value);
            let comp_class = comp_val >= 0 ? 'text-success' : 'text-danger';
            analysisHTML += `<li class="list-group-item d-flex justify-content-between align-items-center">${name} <span class="${comp_class}">${value}</span></li>`;
        }
        analysisHTML += '</ul>';
        document.getElementById('prediction-analysis-container').innerHTML = analysisHTML;

        let signalIcon = s.signal === 'buy' ? 'bi-arrow-up-circle-fill' : s.signal === 'sell' ? 'bi-arrow-down-circle-fill' : 'bi-pause-circle-fill';
        document.getElementById('kpi-container').innerHTML += `
            <div class="card mt-4" id="suggestion-card">
                <div class="card-body text-center">
                    <div class="signal-icon ${final_class}"><i class="${signalIcon}"></i></div>
                    <h4 class="card-title mt-2">${s.signal.toUpperCase()}</h4>
                    <p class="card-text">${s.text}</p>
                    <p class="card-text small text-muted">${s.subtext}</p>
                </div>
            </div>
        `;

        // Render Prediction charts
        document.getElementById('main-chart-title').textContent = 'Prediction vs. Recent History';
        renderPredictionChart(data.chart_data);

        document.getElementById('secondary-chart-title').textContent = 'Feature Importance for Prediction';
        renderFeatureImportanceChart(data.feature_importance);
    }

    function displayBacktestResults(data) {
        const r = data.report;
        document.getElementById('kpi-container').innerHTML = `
            <div class="kpi-card p-3 mb-2"><div class="label">Strategy Return</div><div class="value">${r.total_return}</div></div>
            <div class="kpi-card p-3 mb-2"><div class="label">Buy & Hold Return</div><div class="value">${r.benchmark_return}</div></div>
            <div class="kpi-card p-3 mb-2"><div class="label">Win Rate</div><div class="value text-success">${r.win_rate}</div></div>
            <div class="kpi-card p-3 mb-2"><div class="label">Sharpe Ratio</div><div class="value">${r.sharpe_ratio}</div></div>
            <div class="kpi-card p-3"><div class="label">Max Drawdown</div><div class="value text-danger">${r.max_drawdown}</div></div>
        `;
        document.getElementById('log-wrapper').style.display = 'block';
        document.getElementById('trade-log-container').innerHTML = data.trade_log;

        // Render Backtest charts
        document.getElementById('main-chart-title').textContent = 'Forecast vs. Actual Price';
        renderBacktestChart(data.chart_data.vs_actual);

        document.getElementById('secondary-chart-title').textContent = 'Strategy Equity Curve';
        renderEquityCurveChart(data.chart_data.equity_curve);
    }

    function renderPredictionChart(chartData) {
        const ctx = document.getElementById('main-chart').getContext('2d');
        createOrUpdateChart(ctx, 'line', {
            labels: chartData.dates,
            datasets: [ 
                { label: 'Historical Price', data: chartData.actuals, borderColor: 'rgba(255, 255, 255, 0.9)', borderWidth: 3, pointRadius: 0, tension: 0.1 }, 
                { label: 'Forecasted Price', data: chartData.predictions, borderColor: '#0dcaf0', pointBackgroundColor: '#0dcaf0', pointRadius: 6, pointHoverRadius: 8, tension: 0.1 }
            ]
        });
    }

    function renderFeatureImportanceChart(features) {
        const ctx = document.getElementById('secondary-chart').getContext('2d');
        const data = {
            labels: features.slice(0, 15).map(f => f.feature.replace('feat_', '').replace(/_/g, ' ')).reverse(),
            datasets: [{ label: 'Importance', data: features.slice(0, 15).map(f => f.importance).reverse(), backgroundColor: 'rgba(13, 202, 240, 0.6)' }]
        };
        createOrUpdateChart(ctx, 'bar', data, { indexAxis: 'y', plugins: { legend: { display: false } } });
    }

    function renderBacktestChart(chartData) {
        const ctx = document.getElementById('main-chart').getContext('2d');
        createOrUpdateChart(ctx, 'line', {
            labels: chartData.dates,
            datasets: [ 
                { label: 'Actual Price', data: chartData.actuals, borderColor: 'rgba(255, 255, 255, 0.9)', borderWidth: 3, pointRadius: 0, tension: 0.1 }, 
                { label: 'Final Forecast', data: chartData.predictions, borderColor: '#0dcaf0', borderWidth: 2, pointRadius: 0, tension: 0.1 },
                { label: 'ARIMA Trend', data: chartData.arima, borderColor: 'rgba(255, 193, 7, 0.5)', borderDash: [3, 3], borderWidth: 1.5, pointRadius: 0, tension: 0.1 }
            ]
        });
    }

    function renderEquityCurveChart(chartData) {
        const ctx = document.getElementById('secondary-chart').getContext('2d');
        createOrUpdateChart(ctx, 'line', {
            labels: chartData.dates,
            datasets: [{ label: 'Equity Curve', data: chartData.values, borderColor: '#20c997', backgroundColor: 'rgba(32, 201, 151, 0.1)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.1 }]
        });
    }

    function showError(message) {
        document.getElementById('progress-container').style.display = 'none';
        document.getElementById('error-container').style.display = 'block';
        document.getElementById('error-message').textContent = message;
        resetButtons();
    }

    function resetButtons() {
        document.getElementById('runBtn').disabled = false;
        document.getElementById('forecastBtn').disabled = false;
    }
</script>
</body>
</html>
"""


def run_task(task_func, *args):
    task_id = args[-1]
    try:
        task_func(*args)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        with task_lock:
            RESULTS[task_id] = {'status': 'error', 'message': str(e)}


def backtest_task_wrapper(symbol, days, task_id):
    forecaster = PhoenixForecaster(symbol=symbol, task_id=task_id)
    forecaster.backtest(backtest_days=days)


def predict_task_wrapper(symbol, task_id):
    forecaster = PhoenixForecaster(symbol=symbol, task_id=task_id)
    forecaster.predict()


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE.replace('{logo_placeholder}', PHOENIX_LOGO_SVG))


@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    data = request.json
    task_id = str(uuid.uuid4())
    RESULTS[task_id] = {'status': 'starting', 'message': 'Initializing...'}
    thread = threading.Thread(target=run_task, args=(backtest_task_wrapper, data['symbol'], data['days'], task_id))
    thread.daemon = True
    thread.start()
    return jsonify({'task_id': task_id})


@app.route('/start_predict', methods=['POST'])
def start_predict():
    data = request.json
    task_id = str(uuid.uuid4())
    RESULTS[task_id] = {'status': 'starting', 'message': 'Initializing...'}
    thread = threading.Thread(target=run_task, args=(predict_task_wrapper, data['symbol'], task_id))
    thread.daemon = True
    thread.start()
    return jsonify({'task_id': task_id})


@app.route('/task_status/<task_id>')
def task_status(task_id):
    with task_lock:
        result = RESULTS.get(task_id, {'status': 'not_found', 'message': 'Task ID not found.'})
    return jsonify(result)


if __name__ == '__main__':
    print("Welcome to Phoenix Forecaster!")
    print("=" * 60)

    print("\nStarting Flask web server...")
    print("Open your web browser and go to http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
    print("Server started successfully!")


    print("You can now access the Phoenix Forecaster web interface.")