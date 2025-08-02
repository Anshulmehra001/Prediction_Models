import talib
import numpy as np


def calculate_indicators(df):
    # Basic indicators
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])

    # Indian market specific
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['FII_DII_Impact'] = np.where(df['Close'] > df['VWAP'], 1, -1)  # Simplified proxy

    return df