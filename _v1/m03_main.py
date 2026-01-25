import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

from datetime import datetime
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# [환경 설정]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STOCK_PATH = "D:/stock/_data/manual_fetch/preprocessed_005930_20100101_20251231.csv"
NEWS_PATH = "D:/stock/_data/refined_news/daily_sentiment_score.csv"
STAGE1_MODEL_PATH = "d:/stock/lstm_model/stage1_best_model.pth"
STAGE1_SCALER_X = "d:/stock/lstm_model/stage1_scaler_x.pkl"
STAGE2_MODEL_SAVE = f"d:/stock/lstm_model/stage2_final_model_{now_str}.pth"

# 1. 데이터 병합 및 2단계 구간 필터링 (2022년 이후)
stock_df = pd.read_csv(STOCK_PATH)
news_df = pd.read_csv(NEWS_PATH)

stock_df['날짜'] = pd.to_datetime(stock_df['날짜'])
news_df['날짜'] = pd.to_datetime(news_df['날짜'])

# 2022년 이후 데이터만 추출하여 병합
stage2_df = pd.merge(stock_df[stock_df['날짜'] >= '2022-01-01'], news_df, on='날짜', how='left')
stage2_df['news_sentiment'] = stage2_df['news_sentiment'].fillna(0).reset_index(drop=True)
stage2_df = stage2_df.reset_index(drop=True)

feature_cols = [c for c in stock_df.columns if c not in ['날짜', 'target']]
target_col = 'target'

# 2. 1단계 스케일러 로드 및 적용
scaler_x = joblib.load(STAGE1_SCALER_X)
# 뉴스 점수는 이미 -1 ~ 1 사이이므로 스케일링 없이 그대로 사용
X_tech_scaled = scaler_x.transform(stage2_df[feature_cols])
X_news = stage2_df[['news_sentiment']].values
y_true = stage2_df[[target_col]].values

# 3. 시퀀스 생성 함수 (기술적 지표와 뉴스를 함께 묶음)
def create_fusion_sequences(tech_data, news_data, target_data, window_size=20):
    seq_x_tech, seq_x_news, seq_y = [], [], []
    for i in range(len(tech_data) - window_size):
        seq_x_tech.append(tech_data[i : i + window_size])
        # 뉴스는 '당일'의 뉴스가 중요하므로 타겟과 같은 시점의 점수 사용
        seq_x_news.append(news_data[i + window_size])
        seq_y.append(target_data[i + window_size])
    return torch.FloatTensor(np.array(seq_x_tech)), torch.FloatTensor(np.array(seq_x_news)), torch.FloatTensor(np.array(seq_y))

# 4. 뉴스 결합형 모델 정의
class StockNewsFusionModel(nn.Module):
    def __init__(self, tech_dim, news_dim=1, hidden_dim=64):
        super(StockNewsFusionModel, self).__init__()
        self.lstm = nn.LSTM(tech_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        # LSTM 출력(hidden_dim) + 뉴스 점수(news_dim)를 입력으로 받는 최종 레이어
        self.fc_final = nn.Linear(hidden_dim + news_dim, 1)

    def forward(self, x_tech, x_news):
        lstm_out, _ = self.lstm(x_tech)
        last_tech_feat = lstm_out[:, -1, :] # 차트의 특징 추출
        
        # 주가 특징과 뉴스 심리를 결합
        combined = torch.cat((last_tech_feat, x_news), dim=1)
        return self.fc_final(combined)

# 5. TimeSeriesSplit 기반 2단계 학습
tscv = TimeSeriesSplit(n_splits=3) # 2022-2025 데이터가 상대적으로 적으므로 3 splits 권장
WINDOW_SIZE = 20
best_stage2_loss = float('inf')



print(f"🚀 뉴스 결합 2단계 학습 시작 (구간: {stage2_df['날짜'].min().date()} ~ )")

for train_idx, val_idx in tscv.split(stage2_df):
    # 데이터 분할
    train_tech, val_tech = X_tech_scaled[train_idx], X_tech_scaled[val_idx]
    train_news, val_news = X_news[train_idx], X_news[val_idx]
    train_y, val_y = y_true[train_idx], y_true[val_idx]
    
    # 시퀀스 생성
    X_train_tech, X_train_news, y_train = create_fusion_sequences(train_tech, train_news, train_y, WINDOW_SIZE)
    X_val_tech, X_val_news, y_val = create_fusion_sequences(val_tech, val_news, val_y, WINDOW_SIZE)
    
    # 모델 초기화 및 가중치 이식
    model = StockNewsFusionModel(tech_dim=len(feature_cols)).to(DEVICE)
    stage1_state = torch.load(STAGE1_MODEL_PATH)
    # 1단계의 LSTM 부분 가중치만 매칭하여 로드
    model.lstm.load_state_dict({k.replace('lstm.', ''): v for k, v in stage1_state.items() if 'lstm' in k})
    
    # 전략: 처음 몇 에포크는 LSTM을 고정(Freeze)하고 fc_final만 학습하여 뉴스 효과를 먼저 반영
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    
    for epoch in range(30): # 미세 조정이므로 에포크는 적게 설정
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tech.to(DEVICE), X_train_news.to(DEVICE))
        loss = criterion(output, y_train.to(DEVICE))
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tech.to(DEVICE), X_val_news.to(DEVICE))
            v_loss = criterion(val_output, y_val.to(DEVICE)).item()
            
            if v_loss < best_stage2_loss:
                best_stage2_loss = v_loss
                torch.save(model.state_dict(), STAGE2_MODEL_SAVE)

print(f"✨ 2단계 최종 완료! 최적 뉴스 결합 Loss: {best_stage2_loss:.6f}")