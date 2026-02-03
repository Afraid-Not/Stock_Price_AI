import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv('_data/merged_with_macro.csv')
df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)

# 9개 종목만
TARGET = ['005930','000660','035420','035720','006400','066570','034220','018260','030200']
df = df[df['stock_code'].isin(TARGET)]
df['date'] = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['date']).sort_values(['date','stock_code']).reset_index(drop=True)

print(f"전체 기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"전체 데이터: {len(df):,}건")
print()

# 타겟 필터링 후
if 'next_rtn' in df.columns:
    df_up = df[df['next_rtn'] >= 0.01].copy()
    df_down = df[df['next_rtn'] <= -0.01].copy()
    df_filtered = pd.concat([df_up, df_down]).sort_values(['date','stock_code']).reset_index(drop=True)
    print(f"필터링 후: {len(df_filtered):,}건")
    print()

X = df_filtered.values
tscv = TimeSeriesSplit(n_splits=5)
for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
    val_dates = df_filtered.iloc[val_idx]['date']
    train_dates = df_filtered.iloc[train_idx]['date']
    print(f"Fold {i+1}:")
    print(f"   Train: {train_dates.min().date()} ~ {train_dates.max().date()} ({len(train_idx):,}건)")
    print(f"   Val:   {val_dates.min().date()} ~ {val_dates.max().date()} ({len(val_idx):,}건)")

