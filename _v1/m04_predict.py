import torch
import pandas as pd
import numpy as np
import joblib
import warnings
import os
from datetime import datetime, timedelta

# 기존 재현님 모듈 임포트
from s01_kis_data_get import collect_stock_data
from s04_rename import rename_map
from n01_news import get_sentiment_score
from n04_naver_news import get_today_naver_news
from m03_main import StockNewsFusionModel

warnings.filterwarnings("ignore")

# [1] 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "d:/stock/lstm_model/stage2_final_model_20260127_153203.pth"
SCALER_X_PATH = "d:/stock/lstm_model/stage1_scaler_x.pkl"
STOCK_CODE = "005930"

# s05 전처리 기준 최종 피처 리스트 (순서 절대 고정)
FEATURE_COLS = [
    '거래량', '거래대금', '등락률', '외국인_순매수금액', '기관계_순매수금액', '개인_순매수금액', 
    '금융투자_순매수금액', '투신_순매수금액', '사모펀드_순매수금액', '은행_순매수금액', '보험_순매수금액', 
    '연기금_순매수금액', '기타금융_순매수금액', '기타법인_순매수금액', 'frgn_shnu_tr_pbmn', 
    'frgn_seln_tr_pbmn', 'orgn_shnu_tr_pbmn', 'orgn_seln_tr_pbmn', 'prsn_shnu_tr_pbmn', 
    'prsn_seln_tr_pbmn', '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량',
    '시가_log_ret', '고가_log_ret', '저가_log_ret', 'disparity_5', 'disparity_10', 
    'ma_gap', 'ma5_gradient', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 
    'day_month_sin', 'day_month_cos'
]

def validate_input_data(df_step, scaled_data):
    """모델 입력 전 데이터의 정합성을 검증합니다."""
    print("\n🔍 [데이터 검증 리포트]")
    print("-" * 60)
    
    # 1. 피처 순서 검증
    print(f"✅ 피처 개수 일치: {len(FEATURE_COLS)}개")
    
    # 2. 스케일링 범위 확인 (MinMaxScaler 기준 보통 0~1)
    scaled_min = scaled_data.min()
    scaled_max = scaled_data.max()
    scaled_mean = scaled_data.mean()
    
    print(f"📊 스케일링 범위: Min({scaled_min:.4f}) ~ Max({scaled_max:.4f})")
    print(f"💡 평균값: {scaled_mean:.4f}")
    
    if not (0 <= scaled_mean <= 1):
        print("⚠️ 주의: 평균값이 [0, 1] 범위를 벗어났습니다. 스케일러 확인 필요!")
    
    # 3. 주요 피처 샘플 확인 (상위 5개)
    sample_check = pd.DataFrame(scaled_data, columns=FEATURE_COLS).iloc[-1, :5]
    print("\n📌 최근 영업일 스케일링 샘플 (Top 5):")
    print(sample_check)
    print("-" * 60)

def predict_tomorrow():
    # 데이터 수집 및 전처리 로직 (재현님의 s04, s05 로직 적용)
    df_raw = collect_stock_data(STOCK_CODE, (datetime.now() - timedelta(days=60)).strftime("%Y%m%d"), datetime.now().strftime("%Y%m%d"))
    df = df_raw.rename(columns=rename_map)
    df['날짜'] = pd.to_datetime(df['날짜'], format='%Y%m%d')
    
    # 결측치 및 기술적 지표 생성
    for col in df.columns:
        if col not in ['날짜', '수정주가여부']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    
    for col in ['종가', '시가', '고가', '저가']:
        df[f'{col}_log_ret'] = np.log(df[col] / df[col].shift(1))
    
    df['거래량'] = np.log1p(df['거래량'])
    df['거래대금'] = np.log1p(df['거래대금'])
    df['MA5'] = df['종가'].rolling(window=5).mean()
    df['MA10'] = df['종가'].rolling(window=10).mean()
    df['disparity_5'] = (df['종가'] / df['MA5']) - 1
    df['disparity_10'] = (df['종가'] / df['MA10']) - 1
    df['ma_gap'] = (df['MA5'] / df['MA10']) - 1
    df['ma5_gradient'] = np.log(df['MA5'] / df['MA5'].shift(1))
    
    # 날짜 인코딩
    df['day_sin'] = np.sin(2 * np.pi * df['날짜'].dt.dayofweek / 4)
    df['day_cos'] = np.cos(2 * np.pi * df['날짜'].dt.dayofweek / 4)
    df['month_sin'] = np.sin(2 * np.pi * df['날짜'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['날짜'].dt.month / 12)
    df['day_month_sin'] = np.sin(2 * np.pi * df['날짜'].dt.day / 31)
    df['day_month_cos'] = np.cos(2 * np.pi * df['날짜'].dt.day / 31)

    # 2. 피처 추출 및 검증
    df_final = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
    input_df = df_final[FEATURE_COLS].tail(20)
    
    # 스케일링 수행
    scaler_x = joblib.load(SCALER_X_PATH)
    latest_scaled = scaler_x.transform(input_df)

    # ★ 재현님 요청사항: 검증 리포트 출력 ★
    validate_input_data(input_df, latest_scaled)

    # 3. 뉴스 및 모델 추론
    news_titles = get_today_naver_news("삼성전자")
    news_score = sum([get_sentiment_score(t) for t in news_titles]) / len(news_titles) if news_titles else 0.0
    
    model = StockNewsFusionModel(tech_dim=len(FEATURE_COLS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        x_tech = torch.FloatTensor(latest_scaled).unsqueeze(0).to(DEVICE)
        x_news = torch.FloatTensor([[news_score]]).to(DEVICE)
        pred = model(x_tech, x_news).item()

    # 4. 최종 리포트
    print(f"\n🚀 [최종 결과] 기준일: {df_final['날짜'].iloc[-1].date()}")
    print(f"📊 뉴스 심리: {news_score:.4f} | 예측 변동: {pred*100:.2f}%")
    print(f"💡 투자 의견: {'강력 매수' if pred > 0.01 else '매수' if pred > 0 else '하락 주의'}")

if __name__ == "__main__":
    predict_tomorrow()