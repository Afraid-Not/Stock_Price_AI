import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from datetime import datetime

now = datetime.now().strftime("%Y%m%d_%H%M%S")

# [환경 설정]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "D:/stock/_data/manual_fetch/preprocessed_005930_20100101_20251231.csv"
MODEL_SAVE_DIR = "d:/stock/lstm_model/"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 1. 데이터 로드 및 1단계 필터링
df = pd.read_csv(DATA_PATH)
df['날짜'] = pd.to_datetime(df['날짜'])
stage1_df = df[df['날짜'] < '2022-01-01'].copy().reset_index(drop=True)

feature_cols = [c for c in stage1_df.columns if c not in ['날짜', 'target']]
target_col = 'target'

# 2. 시퀀스 생성 함수
def create_sequences(data_x, data_y, window_size=20):
    sequences_x, sequences_y = [], []
    for i in range(len(data_x) - window_size):
        sequences_x.append(data_x[i : i + window_size])
        sequences_y.append(data_y[i + window_size])
    return np.array(sequences_x), np.array(sequences_y)

# 3. 모델 정의
class Stage1LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(Stage1LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 4. TimeSeriesSplit 학습 루프
tscv = TimeSeriesSplit(n_splits=5) # 5개 구간으로 나누어 교차 검증
WINDOW_SIZE = 20
fold = 1
best_overall_val_loss = float('inf')

print(f"TimeSeriesSplit 기반 1단계 학습 시작 (총 {tscv.n_splits} Folds)")

for train_index, val_index in tscv.split(stage1_df):
    print(f"\n--- Fold {fold} 학습 시작 ---")
    
    # 데이터 분할
    train_data = stage1_df.iloc[train_index]
    val_data = stage1_df.iloc[val_index]
    
    # 스케일링 (Train 기준 fit, Val transform)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    train_x_scaled = scaler_x.fit_transform(train_data[feature_cols])
    train_y_scaled = scaler_y.fit_transform(train_data[[target_col]])
    val_x_scaled = scaler_x.transform(val_data[feature_cols])
    val_y_scaled = scaler_y.transform(val_data[[target_col]])
    
    # 시퀀스 변환
    X_train, y_train = create_sequences(train_x_scaled, train_y_scaled, WINDOW_SIZE)
    X_val, y_val = create_sequences(val_x_scaled, val_y_scaled, WINDOW_SIZE)
    
    train_loader = DataLoader(list(zip(torch.FloatTensor(X_train), torch.FloatTensor(y_train))), batch_size=32, shuffle=True)
    val_loader = DataLoader(list(zip(torch.FloatTensor(X_val), torch.FloatTensor(y_val))), batch_size=32, shuffle=False)
    
    # 모델 초기화
    model = Stage1LSTM(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 학습
    epochs = 100
    best_fold_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        current_val_loss = 0
        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
                current_val_loss += criterion(model(b_x), b_y).item()
        
        avg_val_loss = current_val_loss / len(val_loader)
        
        # Fold 내 최적 모델 저장
        if avg_val_loss < best_fold_val_loss:
            best_fold_val_loss = avg_val_loss
            if avg_val_loss < best_overall_val_loss:
                best_overall_val_loss = avg_val_loss
                # 전체 Fold 중 가장 성능이 좋은 모델과 스케일러 저장
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f'stage1_best_model.pth'))
                joblib.dump(scaler_x, os.path.join(MODEL_SAVE_DIR, 'stage1_scaler_x.pkl'))
                joblib.dump(scaler_y, os.path.join(MODEL_SAVE_DIR, 'stage1_scaler_y.pkl'))

    print(f"Fold {fold} 완료. Best Val Loss: {best_fold_val_loss:.6f}")
    fold += 1

print(f"\n 1단계 최종 완료! 전체 최적 Val Loss: {best_overall_val_loss:.6f}")