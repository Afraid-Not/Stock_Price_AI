# -*- coding: utf-8 -*-
"""LLM vs FinBert 비교"""
import pandas as pd
import numpy as np

# LLM 결과
llm = pd.read_csv('_data/news_sentiment/daily_000660.csv')
llm['날짜'] = pd.to_datetime(llm['날짜'])

# FinBert 결과
fb = pd.read_csv('_data/news_sentiment_finbert/daily_finbert_000660.csv')
fb['날짜'] = pd.to_datetime(fb['날짜'])

# 병합
merged = pd.merge(
    llm[['날짜','news_sentiment']], 
    fb[['날짜','news_sentiment','finbert_confidence']], 
    on='날짜', 
    suffixes=('_llm','_finbert')
)

# 상관관계
corr = merged['news_sentiment_llm'].corr(merged['news_sentiment_finbert'])

print('='*60)
print('LLM vs FinBert 비교 (SK하이닉스)')
print('='*60)
print(f'비교 일수: {len(merged)}일')
print(f'상관계수: {corr:.4f}')
print()
print('LLM 통계:')
print(f'   평균: {merged["news_sentiment_llm"].mean():.4f}')
print(f'   표준편차: {merged["news_sentiment_llm"].std():.4f}')
print(f'   범위: {merged["news_sentiment_llm"].min():.4f} ~ {merged["news_sentiment_llm"].max():.4f}')
print()
print('FinBert 통계:')
print(f'   평균: {merged["news_sentiment_finbert"].mean():.4f}')
print(f'   표준편차: {merged["news_sentiment_finbert"].std():.4f}')
print(f'   범위: {merged["news_sentiment_finbert"].min():.4f} ~ {merged["news_sentiment_finbert"].max():.4f}')
print()

# 방향 일치율
same = ((merged['news_sentiment_llm'] * merged['news_sentiment_finbert']) >= 0).sum()
print(f'방향 일치율: {same/len(merged)*100:.1f}% ({same}/{len(merged)})')
print()

# 가장 차이 큰 날
merged['diff'] = abs(merged['news_sentiment_llm'] - merged['news_sentiment_finbert'])
top5 = merged.nlargest(5, 'diff')
print('감성 차이가 큰 날 TOP 5:')
for _, row in top5.iterrows():
    print(f'   {row["날짜"].strftime("%Y-%m-%d")}: LLM={row["news_sentiment_llm"]:+.3f}, FinBert={row["news_sentiment_finbert"]:+.3f} (차이: {row["diff"]:.3f})')

# 저장
merged.to_csv('_data/news_sentiment_finbert/comparison_000660.csv', index=False)
print(f'\n저장: _data/news_sentiment_finbert/comparison_000660.csv')

