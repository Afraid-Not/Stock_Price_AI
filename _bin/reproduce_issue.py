from pykrx import stock
import pandas as pd

try:
    print("Testing pykrx...")
    df = stock.get_market_trading_value_by_date("20240102", "20240105", "005930")
    print("Result shape:", df.shape)
    print("Result columns:", df.columns)
    print(df.head())
except Exception as e:
    print("Error:", e)

