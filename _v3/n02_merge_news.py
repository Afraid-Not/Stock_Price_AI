import pandas as pd

def prepare_finetune_data(stock_path, news_path, output_path):
    stock_df = pd.read_csv(stock_path)
    news_df = pd.read_csv(news_path)
    
    # 날짜 형식을 맞춰서 병합
    stock_df['날짜'] = stock_df['날짜'].astype(str).str.replace('-', '')
    news_df['일자'] = news_df['일자'].astype(str)
    
    # 주가 데이터에 뉴스 점수 결합 (뉴스 없는 날은 0으로 채움)
    df = pd.merge(stock_df, news_df, left_on='날짜', right_on='일자', how='left')
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    
    # 불필요한 일자 컬럼 제거 후 저장
    df.drop(columns=['일자'], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"✅ 파인튜닝용 데이터셋 준비 완료: {output_path}")

# 실행 예시
prepare_finetune_data("D:/stock/_v3/_data/preprocessed_005930_20100101_20251231.csv", "D:/stock/_v3/_data/daily_news_sentiment.csv", "D:/stock/_v3/_data/stock_news_combined.csv")