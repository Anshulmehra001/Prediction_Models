from alpha_vantage.timeseries import TimeSeries
api_key = '5XEZQID19HMZKNWO'  # Replace with your key
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta = ts.get_daily(symbol="AAPL", outputsize='compact')
print(data.head())