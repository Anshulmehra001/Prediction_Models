import nsepy as nse
from datetime import datetime, timedelta
import pandas as pd

def fetch_intraday_data(symbol="RELIANCE", days=30):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = nse.get_history(
        symbol=symbol,
        start=start.date(),
        end=end.date(),
        index=False
    )
    df.to_csv(f"data/{symbol}_intraday.csv")
    return df