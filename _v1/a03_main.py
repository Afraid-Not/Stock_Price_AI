import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from datetime import datetime

# [환경 설정]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STOCK_PATH = "D:/stock/_data/manual_fetch/preprocessed_005930_20100101_20251231.csv"
NEWS_PATH = "D:/stock/_data/refined_news/daily_sentiment_score.csv"
MODEL_DIR = "D:/stock/attention_model/"

STAGE1_MODEL_NAME = "stage1_best_attention_model.pth" 
STAGE1_SCALER_X = "stage1_scaler_x.pkl"
STAGE1_SCALER_Y = "stage1_scaler_y.pkl"

now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
FINAL_MODEL_SAVE = os.path.join(MODEL_DIR, f"stage2_final_attention_fusion_{now_str}.pth")

# 1. 데이터 병합
stock_df = pd.read_csv(STOCK_PATH)
news_df = pd.read_csv(NEWS_PATH)
stock_df['날짜'] = pd.to_datetime(stock_df['날짜'])
news_df['날짜'] = pd.to_datetime(news_df['날짜'])

stage2_df = pd.merge(stock_df[stock_df['날짜'] >= '2022-01-01'], news_df, on='날짜', how='left')
stage2_df['news_sentiment'] = stage2_df['news_sentiment'].fillna(0)
stage2_df = stage2_df.reset_index(drop=True)

feature_cols = [c for c in stock_df.columns if c not in ['날짜', 'target']]
target_col = 'target'

# 2. 스케일러 로드
scaler_x = joblib.load(os.path.join(MODEL_DIR, STAGE1_SCALER_X))
scaler_y = joblib.load(os.path.join(MODEL_DIR, STAGE1_SCALER_Y))

X_tech_scaled = scaler_x.transform(stage2_df[feature_cols])
X_news = stage2_df[['news_sentiment']].values 
y_true_scaled = scaler_y.transform(stage2_df[[target_col]])

# 3. Fusion 모델 정의
class Stage2AttentionFusionModel(nn.Module):
    def __init__(self, tech_dim, news_dim=1, hidden_dim=128):
        super(Stage2AttentionFusionModel, self).__init__()
        self.lstm = nn.LSTM(tech_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.attention_linear = nn.Linear(hidden_dim, 1)
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim + news_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_tech, x_news):
        lstm_out, _ = self.lstm(x_tech)
        attn_weights = torch.softmax(torch.tanh(self.attention_linear(lstm_out)), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        combined = torch.cat((context_vector, x_news), dim=1)
        return self.fc_final(combined)

def create_fusion_sequences(tech_data, news_data, target_data, window_size=20):
    seq_x_tech, seq_x_news, seq_y = [], [], []
    for i in range(len(tech_data) - window_size):
        seq_x_tech.append(tech_data[i : i + window_size])
        seq_x_news.append(news_data[i + window_size])
        seq_y.append(target_data[i + window_size])
    return torch.FloatTensor(np.array(seq_x_tech)), torch.FloatTensor(np.array(seq_x_news)), torch.FloatTensor(np.array(seq_y))

# 4. Fine-tuning 학습
tscv = TimeSeriesSplit(n_splits=3)
WINDOW_SIZE = 20
best_stage2_val_loss = float('inf')

for train_idx, val_idx in tscv.split(stage2_df):
    X_train_t, X_train_n, y_train = create_fusion_sequences(X_tech_scaled[train_idx], X_news[train_idx], y_true_scaled[train_idx], WINDOW_SIZE)
    X_val_t, X_val_n, y_val = create_fusion_sequences(X_tech_scaled[val_idx], X_news[val_idx], y_true_scaled[val_idx], WINDOW_SIZE)
    
    train_loader = DataLoader(list(zip(X_train_t, X_train_n, y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(list(zip(X_val_t, X_val_n, y_val)), batch_size=32, shuffle=False)

    model = Stage2AttentionFusionModel(tech_dim=len(feature_cols)).to(DEVICE)
    
    # 가중치 로드 부분
    stage1_path = os.path.join(MODEL_DIR, STAGE1_MODEL_NAME)
    if os.path.exists(stage1_path):
        stage1_state = torch.load(stage1_path, map_location=DEVICE)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in stage1_state.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("✅ 1단계 가중치 이식 성공")
    else:
        print("⚠️ 1단계 모델 파일을 찾을 수 없습니다. 경로를 확인하세요.")

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(100):
        model.train()
        for bt, bn, by in train_loader:
            bt, bn, by = bt.to(DEVICE), bn.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bt, bn), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for bv_t, bv_n, bv_y in val_loader:
                bv_t, bv_n, bv_y = bv_t.to(DEVICE), bv_n.to(DEVICE), bv_y.to(DEVICE)
                v_loss += criterion(model(bv_t, bv_n), bv_y).item()
        
        avg_v_loss = v_loss / len(val_loader)
        if avg_v_loss < best_stage2_val_loss:
            best_stage2_val_loss = avg_v_loss
            torch.save(model.state_dict(), FINAL_MODEL_SAVE)

print(f"✨ 2단계 최종 완료! 최적 Fusion Val Loss: {best_stage2_val_loss:.6f}")