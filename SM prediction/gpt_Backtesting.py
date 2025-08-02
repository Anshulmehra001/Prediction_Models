
import os
# Suppress TensorFlow INFO/WARNING logs early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import requests, json, yfinance as yf
import pandas as pd, numpy as np, datetime, matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, Attention
from tensorflow.keras.callbacks import EarlyStopping

# === Sentiment model + API key ===
API_KEY = "94db29d4b7a54c76be66094620543a49"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentiment_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def dummy_sentiment(text):
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    logits = sentiment_model(**inputs).logits
    return torch.softmax(logits, dim=-1)[0,1].item()

def get_sentiment(ticker, max_articles=5):
    url = "https://newsapi.org/v2/everything"
    params = {"q": ticker, "language": "en", "pageSize": max_articles, "apiKey": API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"⚠️ News API error: {e}")
        return 0.0

    articles = data.get("articles", [])
    scores = [dummy_sentiment(a["description"]) for a in articles if a.get("description")]
    return sum(scores)/len(scores) if scores else 0.0

# === Price + technical + sentiment ===
def fetch_data(ticker, window=365):
    df = yf.download(ticker, period=f"{window}d", auto_adjust=True)
    df['fund_pe'] = np.random.rand(len(df))  # placeholder for real fundamentals
    df['EMA10'] = df['Close'].ewm(span=10).mean()
    df['RSI14'] = 100 - 100 / (1 + df['Close'].diff().clip(lower=0).ewm(com=13).mean() /
        df['Close'].diff().clip(upper=0).mul(-1).ewm(com=13).mean())
    df['sentiment'] = get_sentiment(ticker)
    df['target'] = df['Close'].shift(-1)
    return df.dropna()

# === Sequence prep ===
SEQ_LEN = 60
def split_sequences(df):
    feats = ['Close','EMA10','RSI14','fund_pe','sentiment']
    arr = df[feats].values
    X = np.array([arr[i:i+SEQ_LEN] for i in range(len(arr)-SEQ_LEN)])
    y = df['target'].values[SEQ_LEN:]
    return X, y

# === Hybrid model ===
def build_model(X, y):
    n_feat = X.shape[2]
    seq_in = Input((SEQ_LEN, n_feat))
    x = LSTM(64, return_sequences=True)(seq_in)
    x = Attention()([x, x])
    x = LSTM(32)(x)
    x = Dense(16, activation='relu')(x)

    static = Input((n_feat,))
    s = Dense(16, activation='relu')(static)

    merged = concatenate([x, s])
    out = Dense(1)(merged)
    model = Model([seq_in, static], out)
    model.compile(optimizer='adam', loss='mse')

    model.fit([X, X[:,-1,:]], y,
              validation_split=0.1,
              epochs=20, batch_size=16,
              callbacks=[EarlyStopping(patience=5)],
              verbose=0)
    return model

# === Backtest & plot ===
def backtest(ticker="reliance.ns"):
    df = fetch_data(ticker)
    print(f"Data length: {len(df)}")
    X, y = split_sequences(df)
    model = build_model(X, y)
    preds = model.predict([X, X[:,-1,:]]).flatten()

    df['pred'] = np.nan
    df.loc[df.index[SEQ_LEN:], 'pred'] = preds  # fixed assignment warning :contentReference[oaicite:7]{index=7}

    plt.plot(df['Close'], label='Close')
    plt.plot(df['pred'], label='Predicted')
    plt.title(ticker); plt.legend(); plt.show()

    print("MAE:", np.mean(np.abs(preds - y)))

if __name__ == "__main__":
    backtest("reliance.ns")
