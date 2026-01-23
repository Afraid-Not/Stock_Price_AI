import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. 데이터 경로 설정 (형님 경로 그대로)
# ---------------------------------------------------------
BASE_DIR = "/home/jhkim/01_dev/03_stock_market_price_expectation/_data"
STOCK_PATH = "/home/jhkim/01_dev/03_stock_market_price_expectation/_data/02_stock/stock_20240101-20241231.csv"
NEWS_DIR = "/home/jhkim/01_dev/03_stock_market_price_expectation/_data/03_refined_news"

# ---------------------------------------------------------
# 2. 뉴스 데이터 로드 및 날짜별 집계
# ---------------------------------------------------------
def load_and_aggregate_news(news_dir):
    print("📰 뉴스 데이터 로드 및 집계 중...")
    all_news = []
    
    # 디렉토리 내의 분석된 뉴스 파일들 읽기
    for f in os.listdir(news_dir):
        if f.startswith("NewsResult_with_sentiment") and f.endswith((".csv", ".xlsx")):
            path = os.path.join(news_dir, f)
            try:
                if f.endswith(".csv"): df = pd.read_csv(path)
                else: df = pd.read_excel(path)
                all_news.append(df)
            except Exception as e:
                print(f"  ⚠️ 파일 로드 실패: {f}")

    if not all_news:
        print("❌ 분석된 뉴스 파일이 없습니다.")
        return pd.DataFrame()

    df_news = pd.concat(all_news, ignore_index=True)
    
    # Effective_Date(영향일자) 기준으로 그룹화
    # 하루에 뉴스가 100개면 -> 1개의 '평균 감성 점수'로 압축
    # (여기서는 종목 구분을 위해 뉴스 제목에 종목명이 포함되었는지 체크하는 간단 로직 추가)
    
    # ⚠️ 중요: 형님의 뉴스 데이터에 'Company' 컬럼이 없으면 
    # 뉴스가 어떤 종목 뉴스인지 알 수 없음.
    # 일단은 날짜별 전체 시장 분위기(Market Sentiment)로 가정하고 합치거나,
    # 키워드로 종목을 태깅해야 함. 여기서는 '전체 시장 감성'으로 처리함.
    
    news_agg = df_news.groupby('Effective_Date').agg({
        'Sentiment_Score': 'mean',  # 감성 점수 평균
        'Positive_Prob': 'mean',
        'Negative_Prob': 'mean',
        'combined_text': 'count'    # 뉴스 기사 수 (관심도)
    }).reset_index()
    
    news_agg.rename(columns={'combined_text': 'News_Count'}, inplace=True)
    news_agg['Effective_Date'] = pd.to_datetime(news_agg['Effective_Date'])
    
    print(f"✅ 뉴스 집계 완료: {len(news_agg)}일치 데이터")
    return news_agg

# ---------------------------------------------------------
# 3. 기술적 지표 추가 (변동성 파악용)
# ---------------------------------------------------------
def add_technical_indicators(df):
    df = df.sort_values('Date').copy()
    
    # 이동평균선
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 변동성 지표 (ATR: Average True Range) - 주가 범위를 예측하는 데 핵심!
    # 오늘 하루 움직인 폭(고가-저가)과 갭상승/하락까지 고려한 변동폭
    df['Pre_Close'] = df['Close'].shift(1)
    df['TR'] = np.maximum(
        df['High'] - df['Low'], 
        np.maximum(
            abs(df['High'] - df['Pre_Close']), 
            abs(df['Low'] - df['Pre_Close'])
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean() # 14일 평균 변동폭
    
    # 이격도 (현재 주가가 이동평균선에서 얼마나 떨어져 있나)
    df['Disparity_5'] = df['Close'] / df['MA5']
    
    return df

# ---------------------------------------------------------
# 4. 메인 처리 로직
# ---------------------------------------------------------
def main():
    # 1) 주가/수급 데이터 로드
    if not os.path.exists(STOCK_PATH):
        print("❌ 주가 데이터 파일이 없습니다. 이전 단계를 먼저 실행해주세요.")
        return

    df_stock = pd.read_csv(STOCK_PATH)
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    
    # 2) 뉴스 데이터 로드
    df_news = load_and_aggregate_news(NEWS_DIR)
    
    # 3) 데이터 병합 (Left Join: 주가 데이터 기준)
    # 주가 데이터 날짜 = 뉴스 데이터 영향일자
    df_merged = pd.merge(
        df_stock, 
        df_news, 
        left_on='Date', 
        right_on='Effective_Date', 
        how='left'
    )
    
    # 뉴스가 없는 날은 감성점수 0 (중립) 처리
    df_merged['Sentiment_Score'] = df_merged['Sentiment_Score'].fillna(0)
    df_merged['News_Count'] = df_merged['News_Count'].fillna(0)
    
    # 4) 종목별로 기술적 지표 및 Target 생성
    final_data = []
    
    for company in df_merged['Company'].unique():
        sub_df = df_merged[df_merged['Company'] == company].copy()
        
        # 기술적 지표 추가
        sub_df = add_technical_indicators(sub_df)
        
        # --- [핵심] 예측 목표(Target) 생성 ---
        # 내일의 시가, 종가, 고가, 저가를 예측해야 함
        # Shift(-1)을 해서 '다음날 데이터'를 '오늘 행'에 붙임
        
        sub_df['Target_Open'] = sub_df['Open'].shift(-1)   # 내일 시가
        sub_df['Target_Close'] = sub_df['Close'].shift(-1) # 내일 종가
        sub_df['Target_High'] = sub_df['High'].shift(-1)   # 내일 고가
        sub_df['Target_Low'] = sub_df['Low'].shift(-1)     # 내일 저가
        
        # 내일의 변동폭(Range) 비율 계산 (예측 모델이 쉬워짐)
        # 내일 시가가 오늘 종가 대비 몇 % 뜰까?
        sub_df['Target_Open_Change'] = (sub_df['Target_Open'] - sub_df['Close']) / sub_df['Close']
        
        final_data.append(sub_df)
    
    df_final = pd.concat(final_data)
    
    # 결측치 제거 (이동평균 계산 등으로 생긴 앞부분 NaN, Target 생성으로 생긴 뒷부분 NaN)
    df_final = df_final.dropna()
    
    # 저장
    save_path = "/home/jhkim/01_dev/03_stock_market_price_expectation/_data/_dataset/03_dataset_for_training.csv"
    df_final.to_csv(save_path, index=False)
    
    print("\n" + "="*50)
    print(f"✅ 학습용 데이터셋 생성 완료!")
    print(f"📂 저장 경로: {save_path}")
    print(f"📊 총 데이터 수: {len(df_final)}행")
    print("="*50)
    print(df_final[['Date', 'Company', 'Close', 'Sentiment_Score', 'ATR', 'Target_High']].head())

if __name__ == "__main__":
    main()