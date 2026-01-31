# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('_data/preprocessed_005930_20100101_20260129.csv')

print('전체 데이터 수:', len(df))
print('\n클래스 분포:')
dist = df['target'].value_counts().sort_index()
for cls in dist.index:
    print(f'  클래스 {cls}: {dist[cls]:,}개')

print('\n클래스 비율 (%):')
ratio = df['target'].value_counts(normalize=True).sort_index() * 100
for cls in ratio.index:
    print(f'  클래스 {cls}: {ratio[cls]:.2f}%')
