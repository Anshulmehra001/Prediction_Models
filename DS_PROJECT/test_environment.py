import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import yfinance as yf
from nsetools import Nse

print(f"Python {sys.version}\n")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"yFinance: {yf.__version__}")

# Test NSE data
try:
    nse = Nse()
    print("\nNSE Tools working. Sample quote:", nse.get_quote("RELIANCE")['companyName'])
except Exception as e:
    print(f"\nNSE Tools error: {e}")

# Test TensorFlow
try:
    print("\nTensorFlow test:", tf.reduce_sum(tf.random.normal([1000, 1000])))
except Exception as e:
    print(f"\nTensorFlow error: {e}")