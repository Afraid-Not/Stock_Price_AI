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

# [환경 설정]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "D:/stock/_data/manual_fetch/preprocessed_005930_20100101_20251231.csv"
MODEL_SAVE_DIR = "D:/stock/attention_model/"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 1. Attention 기반 LSTM 모델
class Stage1AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(Stage1AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.attention_linear = nn.Linear(hidden_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(torch.tanh(self.attention_linear(lstm_out)), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context_vector)

# 2. 데이터 준비
df = pd.read_csv(DATA_PATH)
df['날짜'] = pd.to_datetime(df['날짜'])
stage1_df = df[df['날짜'] < '2022-01-01'].copy().reset_index(drop=True)

feature_cols = [c for c in stage1_df.columns if c not in ['날짜', 'target']]
target_col = 'target'

def create_sequences(data_x, data_y, window_size=20):
    seq_x, seq_y = [], []
    for i in range(len(data_x) - window_size):
        seq_x.append(data_x[i : i + window_size])
        seq_y.append(data_y[i + window_size])
    return np.array(seq_x), np.array(seq_y)

# 3. 학습 루프
tscv = TimeSeriesSplit(n_splits=5)
WINDOW_SIZE = 20
best_val_loss = float('inf')

print(f"🚀 [Stage 1] Attention 사전 학습 시작...")

for fold, (train_idx, val_idx) in enumerate(tscv.split(stage1_df), 1):
    train_data, val_data = stage1_df.iloc[train_idx], stage1_df.iloc[val_idx]
    
    sx, sy = MinMaxScaler(), MinMaxScaler()
    tx_s = sx.fit_transform(train_data[feature_cols])
    ty_s = sy.fit_transform(train_data[[target_col]])
    vx_s = sx.transform(val_data[feature_cols])
    vy_s = sy.transform(val_data[[target_col]])
    
    X_train, y_train = create_sequences(tx_s, ty_s, WINDOW_SIZE)
    X_val, y_val = create_sequences(vx_s, vy_s, WINDOW_SIZE)
    
    train_loader = DataLoader(list(zip(torch.FloatTensor(X_train), torch.FloatTensor(y_train))), batch_size=64, shuffle=True)
    
    model = Stage1AttentionLSTM(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        v_loss = criterion(model(torch.FloatTensor(X_val).to(DEVICE)), torch.FloatTensor(y_val).to(DEVICE)).item()
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            # 파일명 고정 (a03에서 로드하기 쉽게)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'stage1_best_attention_model.pth'))
            joblib.dump(sx, os.path.join(MODEL_SAVE_DIR, 'stage1_scaler_x.pkl'))
            joblib.dump(sy, os.path.join(MODEL_SAVE_DIR, 'stage1_scaler_y.pkl'))
    print(f"Fold {fold} 완료. Val Loss: {v_loss:.6f}")

print(f"✅ 1단계 완료. 최적 Loss: {best_val_loss:.6f}")