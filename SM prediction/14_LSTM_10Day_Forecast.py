import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# 1️⃣ Fetch data and scale
ticker = 'coromandel.ns'
df = yf.download(ticker, period='2y', auto_adjust=True)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['Close']].values)

# 2️⃣ Prepare sequences
SEQ_LEN = 60
future_days = 10
X, y = [], []
for i in range(len(data_scaled) - SEQ_LEN):
    X.append(data_scaled[i: i + SEQ_LEN])
    y.append(data_scaled[i + SEQ_LEN])
X, y = np.array(X), np.array(y)
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# 3️⃣ Build & train LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=20, batch_size=16, verbose=1)

# 4️⃣ Roll-forward forecast
seq = data_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
future_preds = []
for _ in range(future_days):
    pred = model.predict(seq, verbose=0)[0, 0]
    future_preds.append(pred)
    new_step = np.array(pred).reshape(1, 1, 1)
    seq = np.concatenate((seq[:, 1:, :], new_step), axis=1)
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()

# 5️⃣ Prepare future date index
future_idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1),
                            periods=future_days)

# 6️⃣ Plot forecast-only chart
fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
ax.plot(future_idx, future_preds, marker='o', linestyle='-',
        color='dodgerblue', label='Forecast')
ax.set_title(f"{ticker} Forecast Only – Next {future_days} Days")
ax.set_ylabel("Price (USD)")
ax.grid(linestyle='--', alpha=0.5)
ax.legend()

# 🔧 Improve daily ticks
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=45, fontsize=9)

plt.tight_layout()
plt.show()
