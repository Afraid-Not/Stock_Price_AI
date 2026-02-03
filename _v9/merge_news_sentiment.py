# -*- coding: utf-8 -*-
"""
뉴스 sentiment를 주식 데이터에 병합
- 15:30 이후 뉴스 -> 다음 거래일에 반영
- 주말/휴장일 뉴스 -> 다음 거래일에 반영
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 장 마감 시간 (HHMMSS)
MARKET_CLOSE = 153000

def get_trading_days(stock_df):
    """주식 데이터에서 거래일 목록 추출"""
    trading_days = pd.to_datetime(stock_df['날짜'].unique())
    return sorted(trading_days)

def get_next_trading_day(date, trading_days):
    """다음 거래일 찾기"""
    for td in trading_days:
        if td > date:
            return td
    return None

def assign_news_to_trading_day(news_df, trading_days):
    """
    각 뉴스를 적절한 거래일에 배정
    - 장중 뉴스(~15:30) -> 당일
    - 장 마감 후 뉴스(15:30~) -> 다음 거래일
    - 주말/휴장일 -> 다음 거래일
    """
    news_df = news_df.copy()
    
    # 날짜, 시간 파싱
    news_df['news_date'] = pd.to_datetime(news_df['작성일자'], format='%Y%m%d')
    news_df['news_time'] = news_df['작성시간'].astype(int)
    
    assigned_dates = []
    
    for _, row in news_df.iterrows():
        news_date = row['news_date']
        news_time = row['news_time']
        
        # 장 마감 이후인지 확인
        is_after_close = news_time >= MARKET_CLOSE
        
        # 해당 날짜가 거래일인지 확인
        is_trading_day = news_date in trading_days
        
        if is_trading_day and not is_after_close:
            # 거래일 + 장중 -> 당일
            assigned_date = news_date
        else:
            # 장 마감 후 또는 휴장일 -> 다음 거래일
            assigned_date = get_next_trading_day(news_date, trading_days)
        
        assigned_dates.append(assigned_date)
    
    news_df['assigned_date'] = assigned_dates
    
    # 다음 거래일이 없는 뉴스 제거 (데이터 끝부분)
    news_df = news_df.dropna(subset=['assigned_date'])
    
    return news_df

def aggregate_daily_sentiment(news_df):
    """거래일별 sentiment 집계"""
    
    daily = news_df.groupby(['stock_code', 'assigned_date']).agg({
        'finbert_sentiment': ['mean', 'std', 'min', 'max'],
        'finbert_confidence': 'mean',
        'finbert_label': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral'
    }).reset_index()
    
    # 컬럼명 정리
    daily.columns = [
        'stock_code', '날짜', 
        'news_sentiment_mean', 'news_sentiment_std', 
        'news_sentiment_min', 'news_sentiment_max',
        'news_confidence', 'news_label'
    ]
    
    # 뉴스 수 추가
    news_count = news_df.groupby(['stock_code', 'assigned_date']).size().reset_index(name='news_count')
    daily = daily.merge(news_count, left_on=['stock_code', '날짜'], right_on=['stock_code', 'assigned_date'])
    daily = daily.drop(columns=['assigned_date'])
    
    # NaN 처리
    daily['news_sentiment_std'] = daily['news_sentiment_std'].fillna(0)
    
    return daily

def merge_with_stock_data(stock_df, sentiment_df):
    """주식 데이터와 sentiment 병합"""
    
    stock_df = stock_df.copy()
    stock_df['날짜'] = pd.to_datetime(stock_df['날짜'])
    
    # stock_code 타입 맞추기
    stock_df['stock_code'] = stock_df['stock_code'].astype(str).str.zfill(6)
    sentiment_df['stock_code'] = sentiment_df['stock_code'].astype(str).str.zfill(6)
    sentiment_df['날짜'] = pd.to_datetime(sentiment_df['날짜'])
    
    # 병합
    merged = stock_df.merge(
        sentiment_df,
        on=['stock_code', '날짜'],
        how='left'
    )
    
    # 뉴스 없는 날 처리
    merged['news_count'] = merged['news_count'].fillna(0).astype(int)
    merged['news_sentiment_mean'] = merged['news_sentiment_mean'].fillna(0)
    merged['news_sentiment_std'] = merged['news_sentiment_std'].fillna(0)
    merged['news_confidence'] = merged['news_confidence'].fillna(0)
    merged['news_label'] = merged['news_label'].fillna('none')
    
    return merged

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("="*60)
    print("[NEWS] Sentiment Merge (Time-based)")
    print("="*60)
    
    # 주식 데이터 로드
    stock_path = "_data/merged_with_macro.csv"
    print(f"\n[LOAD] Stock data: {stock_path}")
    stock_df = pd.read_csv(stock_path, encoding='utf-8-sig')
    # 날짜 형식: 20100330 (YYYYMMDD)
    stock_df['날짜'] = pd.to_datetime(stock_df['날짜'].astype(str), format='%Y%m%d')
    print(f"   Total: {len(stock_df):,} rows")
    print(f"   Period: {stock_df['날짜'].min()} ~ {stock_df['날짜'].max()}")
    
    # 거래일 목록
    trading_days = get_trading_days(stock_df)
    trading_days_set = set(trading_days)
    print(f"   Trading days: {len(trading_days)}")
    
    # 9개 종목 목록
    target_stocks = ['000660', '005930', '006400', '018260', '030200', 
                     '034220', '035420', '035720', '066570']
    
    all_sentiment = []
    
    # 각 종목별 뉴스 처리
    news_dir = Path("_data/news_sentiment_finbert")
    
    for stock_code in target_stocks:
        news_file = news_dir / f"finbert_{stock_code}.csv"
        
        if not news_file.exists():
            print(f"\n[WARN] {stock_code} news file not found")
            continue
        
        print(f"\n[PROC] {stock_code}...")
        
        # 뉴스 로드
        news_df = pd.read_csv(news_file)
        print(f"   Original news: {len(news_df):,}")
        
        # 거래일 배정
        news_assigned = assign_news_to_trading_day(news_df, trading_days_set)
        print(f"   Assigned news: {len(news_assigned):,}")
        
        # 장 마감 후 뉴스 통계
        after_close = (news_df['작성시간'].astype(int) >= MARKET_CLOSE).sum()
        print(f"   After market close: {after_close:,} ({after_close/len(news_df)*100:.1f}%)")
        
        # 일별 집계
        daily_sentiment = aggregate_daily_sentiment(news_assigned)
        all_sentiment.append(daily_sentiment)
        print(f"   Trading days with news: {len(daily_sentiment)}")
    
    # 전체 sentiment 합치기
    if all_sentiment:
        sentiment_df = pd.concat(all_sentiment, ignore_index=True)
        print(f"\n[TOTAL] Sentiment data: {len(sentiment_df):,} rows")
        
        # 주식 데이터와 병합
        merged_df = merge_with_stock_data(stock_df, sentiment_df)
        
        # 저장
        output_path = "_data/merged_with_news.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[SAVE] {output_path}")
        print(f"   Total: {len(merged_df):,} rows")
        
        # 뉴스 있는 데이터 통계
        with_news = (merged_df['news_count'] > 0).sum()
        print(f"   With news: {with_news:,} ({with_news/len(merged_df)*100:.1f}%)")
        
        # 9개 종목만 필터링한 버전도 저장
        merged_9 = merged_df[merged_df['stock_code'].isin(target_stocks)]
        output_9 = "_data/merged_9stocks_with_news.csv"
        merged_9.to_csv(output_9, index=False, encoding='utf-8-sig')
        print(f"\n[SAVE] 9 stocks: {output_9}")
        print(f"   Total: {len(merged_9):,} rows")
        
        # 뉴스 sentiment 분포
        print("\n[STATS] News Sentiment Distribution:")
        print(merged_9[merged_9['news_count'] > 0]['news_sentiment_mean'].describe())
    
    print("\n" + "="*60)
    print("[DONE]")
    print("="*60)

if __name__ == "__main__":
    main()
