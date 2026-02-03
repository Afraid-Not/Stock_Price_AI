"""
뉴스 감성 점수를 주가 데이터와 병합하는 스크립트
"""
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime


def merge_news_sentiment(stock_data_path: str, sentiment_path: str, 
                         output_path: str = None) -> pd.DataFrame:
    """
    주가 데이터와 뉴스 감성 점수 병합
    
    Args:
        stock_data_path: 주가 데이터 CSV 경로 (merged_all_stocks.csv)
        sentiment_path: 뉴스 감성 점수 CSV 경로 (news_sentiment_daily.csv)
        output_path: 출력 경로 (None이면 자동 생성)
    
    Returns:
        병합된 DataFrame
    """
    print("=" * 60)
    print("📊 뉴스 감성 점수 병합")
    print("=" * 60)
    
    # 주가 데이터 로드
    print(f"\n📂 주가 데이터 로드: {stock_data_path}")
    df_stock = pd.read_csv(stock_data_path)
    print(f"   {len(df_stock):,}건, {df_stock['stock_code'].nunique()}개 종목")
    
    # 뉴스 감성 데이터 로드
    print(f"\n📂 뉴스 감성 데이터 로드: {sentiment_path}")
    df_sentiment = pd.read_csv(sentiment_path)
    print(f"   {len(df_sentiment):,}건")
    
    # 날짜 형식 통일
    df_stock['날짜'] = pd.to_datetime(df_stock['날짜']).dt.strftime('%Y-%m-%d')
    df_sentiment['날짜'] = pd.to_datetime(df_sentiment['날짜']).dt.strftime('%Y-%m-%d')
    
    # stock_code 형식 통일
    df_stock['stock_code'] = df_stock['stock_code'].astype(str).str.zfill(6)
    df_sentiment['stock_code'] = df_sentiment['stock_code'].astype(str).str.zfill(6)
    
    # 병합 전 컬럼 확인
    print(f"\n📋 병합 키: stock_code, 날짜")
    print(f"   주가 데이터 기간: {df_stock['날짜'].min()} ~ {df_stock['날짜'].max()}")
    print(f"   감성 데이터 기간: {df_sentiment['날짜'].min()} ~ {df_sentiment['날짜'].max()}")
    
    # 병합 (left join - 주가 데이터 기준)
    df_merged = pd.merge(
        df_stock,
        df_sentiment[['stock_code', '날짜', 'news_sentiment', 'news_sentiment_simple', 'news_count']],
        on=['stock_code', '날짜'],
        how='left'
    )
    
    # 결측치 처리 (뉴스가 없는 날)
    df_merged['news_sentiment'] = df_merged['news_sentiment'].fillna(0)
    df_merged['news_sentiment_simple'] = df_merged['news_sentiment_simple'].fillna(0)
    df_merged['news_count'] = df_merged['news_count'].fillna(0).astype(int)
    
    # 뉴스 감성 피처 추가 (이동 평균, 모멘텀 등)
    print("\n⚙️ 뉴스 감성 피처 생성...")
    
    # 종목별로 피처 생성
    df_list = []
    for stock_code in df_merged['stock_code'].unique():
        df_s = df_merged[df_merged['stock_code'] == stock_code].copy()
        df_s = df_s.sort_values('날짜')
        
        # 이동 평균
        df_s['news_sentiment_ma3'] = df_s['news_sentiment'].rolling(3).mean()
        df_s['news_sentiment_ma5'] = df_s['news_sentiment'].rolling(5).mean()
        
        # 모멘텀 (전일 대비 변화)
        df_s['news_sentiment_change'] = df_s['news_sentiment'].diff()
        
        # 뉴스 관심도 (뉴스 개수 정규화)
        max_count = df_s['news_count'].max()
        if max_count > 0:
            df_s['news_attention'] = df_s['news_count'] / max_count
        else:
            df_s['news_attention'] = 0
        
        df_list.append(df_s)
    
    df_merged = pd.concat(df_list, ignore_index=True)
    
    # 결측치 처리
    df_merged['news_sentiment_ma3'] = df_merged['news_sentiment_ma3'].fillna(0)
    df_merged['news_sentiment_ma5'] = df_merged['news_sentiment_ma5'].fillna(0)
    df_merged['news_sentiment_change'] = df_merged['news_sentiment_change'].fillna(0)
    df_merged['news_attention'] = df_merged['news_attention'].fillna(0)
    
    # 병합 결과 확인
    matched = df_merged[df_merged['news_count'] > 0]
    match_rate = len(matched) / len(df_merged) * 100
    
    print(f"\n✅ 병합 완료!")
    print(f"   총 데이터: {len(df_merged):,}건")
    print(f"   뉴스 매칭: {len(matched):,}건 ({match_rate:.1f}%)")
    print(f"   평균 감성: {df_merged['news_sentiment'].mean():.4f}")
    
    # 추가된 피처 목록
    news_features = ['news_sentiment', 'news_sentiment_simple', 'news_count',
                     'news_sentiment_ma3', 'news_sentiment_ma5', 
                     'news_sentiment_change', 'news_attention']
    print(f"\n📌 추가된 뉴스 피처:")
    for f in news_features:
        print(f"   - {f}")
    
    # 저장
    if output_path is None:
        output_path = stock_data_path.replace('.csv', '_with_news.csv')
    
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {output_path}")
    
    return df_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="뉴스 감성 점수 병합")
    parser.add_argument("--stock", type=str, default="_data/merged_all_stocks_20260131.csv",
                        help="주가 데이터 경로")
    parser.add_argument("--sentiment", type=str, default="_data/news/news_sentiment_daily.csv",
                        help="뉴스 감성 데이터 경로")
    parser.add_argument("-o", "--output", type=str, default=None, help="출력 경로")
    
    args = parser.parse_args()
    
    merge_news_sentiment(
        stock_data_path=args.stock,
        sentiment_path=args.sentiment,
        output_path=args.output
    )

