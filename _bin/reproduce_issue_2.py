from pykrx import stock
import pandas as pd

try:
    print("Testing pykrx volume...")
    df = stock.get_market_trading_volume_by_date("20240102", "20240105", "005930")
    print("Volume Result shape:", df.shape)
    
    print("Testing pykrx value (2023)...")
    df2 = stock.get_market_trading_value_by_date("20230102", "20230105", "005930")
    print("Value 2023 Result shape:", df2.shape)

except Exception as e:
    print("Error:", e)

