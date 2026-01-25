import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader

# [설정]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STOCK_PATH = "D:/stock/_data/manual_fetch/preprocessed_005930_20100101_20251231.csv"
NEWS_PATH = "D:/stock/_data/refined_news/daily_sentiment_score.csv"
STAGE1_MODEL_PATH = "d:/stock/lstm_model/stage1_best_model.pth"
STAGE1_SCALER_X = "d:/stock/lstm_model/stage1_scaler_x.pkl"

# 1. 데이터 병합 (2022년 이후)
stock_df = pd.read_csv(STOCK_PATH)
news_df = pd.read_csv(NEWS_PATH)

stock_df['날짜'] = pd.to_datetime(stock_df['날짜'])
news_df['날짜'] = pd.to_datetime(news_df['날짜'])

# 2022년 이후 데이터만 추출하여 병합
df_stage2 = pd.merge(stock_df[stock_df['날짜'] >= '2022-01-01'], news_df, on='날짜', how='left')
df_stage2['news_sentiment'] = df_stage2['news_sentiment'].fillna(0) # 뉴스 없는 날 중립 처리

# 2. 1단계 스케일러 적용
scaler_x = joblib.load(STAGE1_SCALER_X)
feature_cols = [c for c in stock_df.columns if c not in ['날짜', 'target']]
# 뉴스 점수는 이미 -1 ~ 1 사이이므로 별도 스케일링 없이 그대로 사용

X_tech = scaler_x.transform(df_stage2[feature_cols])
X_news = df_stage2[['news_sentiment']].values
y_true = df_stage2[['target']].values # 1단계 scaler_y로 변환 필요 시 추가

# 3. 2단계 전용 모델 설계 (기존 LSTM + 뉴스 결합)
class FineTuningModel(nn.Module):
    def __init__(self, tech_dim, news_dim=1, hidden_dim=64):
        super(FineTuningModel, self).__init__()
        # 1단계와 동일한 구조의 LSTM
        self.lstm = nn.LSTM(tech_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        # 뉴스 점수와 결합하여 최종 예측하는 레이어
        self.fc_final = nn.Linear(hidden_dim + news_dim, 1)

    def forward(self, x_tech, x_news):
        lstm_out, _ = self.lstm(x_tech)
        last_tech_feat = lstm_out[:, -1, :]
        
        # 주가 특징과 뉴스 특징을 결합 (Concatenate)
        combined = torch.cat((last_tech_feat, x_news), dim=1)
        return self.fc_final(combined)

# 4. 1단계 가중치 로드 및 이식
model = FineTuningModel(tech_dim=len(feature_cols)).to(DEVICE)
stage1_state = torch.load(STAGE1_MODEL_PATH)

# LSTM 부분만 가중치 복사
model.lstm.load_state_dict({k.replace('lstm.', ''): v for k, v in stage1_state.items() if 'lstm' in k})

# (옵션) LSTM 레이어 고정 - 뉴스 효과만 집중 학습하고 싶을 때
for param in model.lstm.parameters(): param.requires_grad = False

print("✅ 1단계 지식 이식 완료. 2단계 뉴스 결합 학습 준비가 끝났습니다.")