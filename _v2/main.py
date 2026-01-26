from kis_auth import KISAuthenticator
from kis_collector import KISDataCollector
from stock_preprocessor import StockPreprocessor

def run_pipeline(code, start_date, end_date):
    print(f"🚀 {code} 데이터 파이프라인 시작")
    
    # 1. 인증 및 수집
    auth = KISAuthenticator()
    collector = KISDataCollector(auth)
    raw_df = collector.collect_full_range(code, start_date, end_date)
    
    # 2. 전처리
    preprocessor = StockPreprocessor()
    final_df = preprocessor.process(raw_df)
    
    # 3. 저장
    save_path = f"D:/stock/_v2/_data/preprocessed_{code}_{start_date}_{end_date}.csv"
    final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 전처리 완료! 파일 저장됨: {save_path}")
    return final_df

if __name__ == "__main__":
    df = run_pipeline("005930", "20100101", "20251231")
    print(df.head())