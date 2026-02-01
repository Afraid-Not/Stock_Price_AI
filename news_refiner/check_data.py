import pandas as pd

df = pd.read_csv('_news/news_005930_20200101_20260127_daily.csv', encoding='utf-8-sig')

print(f'총 일수: {len(df)}')
print(f'날짜 범위: {df["날짜"].min()} ~ {df["날짜"].max()}')

df['연도'] = pd.to_datetime(df['날짜']).dt.year
print('\n연도별 분포:')
print(df['연도'].value_counts().sort_index())

print('\n감성 점수 통계:')
print(df['감성_점수'].describe())

print('\n연도별 감성 점수 평균:')
print(df.groupby('연도')['감성_점수'].mean().sort_index())





