import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import TimeSeriesSplit
import os
import copy
import joblib
from sklearn.preprocessing import StandardScaler

# [1] 데이터셋 및 [2] 모델 정의는 기존과 동일하게 유지
class StockDataset(Dataset):
    def __init__(self, data_values, window_size=20):
        self.data = data_values.astype(np.float32)
        self.window_size = window_size
    def __len__(self):
        return len(self.data) - self.window_size
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size, :-1]
        y = self.data[idx + self.window_size, -1]
        return torch.tensor(x), torch.tensor(y)

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

def expand_and_transfer(pretrained_model, new_input_dim, hidden_dim=64, num_layers=2):
    new_model = StockLSTM(input_dim=new_input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    old_dict = pretrained_model.state_dict()
    new_dict = new_model.state_dict()
    for name, param in old_dict.items():
        if 'lstm.weight_ih_l0' in name:
            new_dict[name][:, :-1] = param
            nn.init.xavier_normal_(new_dict[name][:, -1:])
        else:
            new_dict[name] = param
    new_model.load_state_dict(new_dict)
    return new_model

# [4] 학습 루프 (교차 검증을 위해 로그 출력 최적화)
def train_model_with_val(model, train_loader, val_loader, epochs, lr, device, patience=10, verbose=True):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                output_v = model(x_v)
                val_loss += criterion(output_v, y_v).item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            break
            
    model.load_state_dict(best_model_wts)
    return model, best_loss

def main():
    window_size = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "D:/stock/_v3/_data"
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # 데이터 로드
    stock_df = pd.read_csv(f"{base_dir}/preprocessed_005930_20100101_20251231.csv")
    news_df = pd.read_csv(f"{base_dir}/daily_news_sentiment.csv")
    stock_df['날짜'] = stock_df['날짜'].astype(str).str.replace('-', '')
    news_df['일자'] = news_df['일자'].astype(str)
    
    # --- Phase 1: Pre-training (2010~2021) 교차 검증 ---
    print(f"\n🚀 Phase 1 (Pre-training) {n_splits}-Fold 교차 검증 시작...")
    phase1_df = stock_df[stock_df['날짜'] < '20220101'].copy()
    phase1_values = phase1_df.drop(columns=['날짜']).values
    input_dim1 = phase1_values.shape[1] - 1
    
    best_base_model = None
    phase1_losses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(phase1_values)):
        train_v, val_v = phase1_values[train_idx], phase1_values[val_idx]
        train_loader = DataLoader(StockDataset(train_v, window_size), batch_size=64, shuffle=False)
        val_loader = DataLoader(StockDataset(val_v, window_size), batch_size=64, shuffle=False)
        
        model = StockLSTM(input_dim=input_dim1)
        model, b_loss = train_model_with_val(model, train_loader, val_loader, 100, 0.005, device, 20)
        phase1_losses.append(b_loss)
        print(f" Fold {fold+1}: Best Val Loss = {b_loss:.6f}")
        
        # 마지막 폴드의 모델을 Phase 2로 전달하기 위해 저장
        if fold == n_splits - 1:
            best_base_model = copy.deepcopy(model)

    print(f"📊 Phase 1 평균 Val Loss: {np.mean(phase1_losses):.6f}")
    torch.save(best_base_model.state_dict(), f"{base_dir}/base_model_cv.pth")

    # --- Phase 2: Fine-tuning (2022~2025) 데이터 준비 ---
    print("\n📊 뉴스 데이터 병합 및 스케일링 중...")
    phase2_stock = stock_df[stock_df['날짜'] >= '20220101'].copy()
    phase2_combined = pd.merge(phase2_stock, news_df, left_on='날짜', right_on='일자', how='left').fillna(0)
    phase2_combined['sentiment_score'] = phase2_combined['sentiment_score'].shift(1).fillna(0)
    
    s_scaler = StandardScaler()
    phase2_combined['sentiment_score'] = s_scaler.fit_transform(phase2_combined[['sentiment_score']])
    joblib.dump(s_scaler, "D:/stock/_v3/scalers/sentiment_scaler.bin")

    cols = [c for c in phase2_combined.columns if c not in ['날짜', '일자', 'target']] + ['target']
    phase2_values = phase2_combined[cols].values
    input_dim2 = phase2_values.shape[1] - 1

    # --- Phase 2: Fine-tuning 교차 검증 ---
    print(f"🚀 Phase 2 (Fine-tuning) {n_splits}-Fold 교차 검증 시작...")
    phase2_losses = []
    final_model = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(phase2_values)):
        train_v, val_v = phase2_values[train_idx], phase2_values[val_idx]
        train_loader = DataLoader(StockDataset(train_v, window_size), batch_size=32, shuffle=False)
        val_loader = DataLoader(StockDataset(val_v, window_size), batch_size=32, shuffle=False)
        
        # 각 폴드마다 Phase 1의 결과물로부터 새로 시작
        model = expand_and_transfer(best_base_model, new_input_dim=input_dim2)
        model, b_loss = train_model_with_val(model, train_loader, val_loader, 50, 0.0005, device, 10)
        phase2_losses.append(b_loss)
        print(f" Fold {fold+1}: Best Val Loss = {b_loss:.6f}")
        
        if fold == n_splits - 1:
            final_model = copy.deepcopy(model)

    print(f"📊 Phase 2 평균 Val Loss: {np.mean(phase2_losses):.6f}")
    torch.save(final_model.state_dict(), f"{base_dir}/final_multimodal_model_cv.pth")
    print(f"\n학습이 완료되었습니다!")

if __name__ == "__main__":
    main()