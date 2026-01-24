import pandas as pd
import sys

# Set output encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

try:
    df = pd.read_csv('D:/stock/_data/stock/stock_005930_20220101_20251231.csv')
    with open('inspect_stock_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"Columns: {list(df.columns)}\n")
        f.write("Sample Data (first 1 row):\n")
        f.write(str(df.iloc[0].to_dict()))
except Exception as e:
    with open('inspect_stock_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"Error: {e}")

