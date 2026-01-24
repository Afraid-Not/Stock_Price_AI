import pandas as pd
import sys

# Set output encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

try:
    df = pd.read_parquet('D:/stock/_data/pseudo/news_total_refined.parquet')
    with open('inspect_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"Columns: {list(df.columns)}\n")
        f.write(f"Shape: {df.shape}\n")
        if 'embedding' in df.columns:
            sample_emb = df['embedding'].iloc[0]
            f.write(f"Embedding type: {type(sample_emb)}\n")
            f.write(f"Embedding length: {len(sample_emb) if hasattr(sample_emb, '__len__') else 'N/A'}\n")
        
        f.write("Sample Data (first 1 row):\n")
        f.write(str(df.iloc[0].to_dict()))
except Exception as e:
    with open('inspect_result.txt', 'w', encoding='utf-8') as f:
        f.write(f"Error: {e}")

