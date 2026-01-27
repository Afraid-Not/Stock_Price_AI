import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# [1] 데이터셋 클래스 (슬라이딩 윈도우)
class StockDataset(Dataset):
    def __init__(self, data_values, window_size=20):
        """
        data_values: [N, Features + Target] 형태의 numpy array
        """
        self.data = data_values.astype(np.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # x: window_size 만큼의 피처들, y: 타겟 값
        x = self.data[idx : idx + self.window_size, :-1]
        y = self.data[idx + self.window_size, -1]
        return torch.tensor(x), torch.tensor(y)

# [2] LSTM 모델 정의
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # 마지막 시점(last time step)의 출력만 사용
        return self.fc(out[:, -1, :]).squeeze()

# [3] 가중치 이식 및 차원 확장 함수
def expand_and_transfer(pretrained_model, new_input_dim, hidden_dim=64, num_layers=2):
    """ 
    기존 주가 모델의 지식을 뉴스 통합 모델로 이식 
    """
    new_model = StockLSTM(input_dim=new_input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    old_dict = pretrained_model.state_dict()
    new_dict = new_model.state_dict()

    for name, param in old_dict.items():
        if 'lstm.weight_ih_l0' in name:
            # 입력 레이어 가중치: 기존 피처 부분은 복사, 추가된 뉴스 피처 부분은 초기화
            new_dict[name][:, :-1] = param
            nn.init.xavier_normal_(new_dict[name][:, -1:])
        else:
            new_dict[name] = param
            
    new_model.load_state_dict(new_dict)
    return new_model

# [4] 학습 루프 함수
def train_model(model, train_loader, epochs, lr, device, phase_name="Training"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n🚀 {phase_name} 시작...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(train_loader):.6f}")
            
    return model

def main():
    # 설정
    window_size = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "D:/stock/_v3/_data"
    
    # 1. 데이터 로드 및 분리
    print("📂 데이터를 로드하고 학습 단계별로 분리합니다...")
    stock_df = pd.read_csv(f"{base_dir}/preprocessed_005930.csv")
    news_df = pd.read_csv(f"{base_dir}/daily_news_sentiment.csv")
    
    # 날짜 형식 통일 (YYYYMMDD)
    stock_df['날짜'] = stock_df['날짜'].astype(str).str.replace('-', '')
    news_df['일자'] = news_df['일자'].astype(str)
    
    # --- [Phase 1 데이터 준비] 2010~2021 (주가만) ---
    phase1_df = stock_df[stock_df['날짜'] < '20220101'].copy()
    # '날짜' 제외, 'target'은 마지막에 위치해야 함
    phase1_data = phase1_df.drop(columns=['날짜']).values 
    
    # --- [Phase 2 데이터 준비] 2022~2025 (주가 + 뉴스) ---
    phase2_stock = stock_df[stock_df['날짜'] >= '20220101'].copy()
    # 뉴스 병합
    phase2_combined = pd.merge(phase2_stock, news_df, left_on='날짜', right_on='일자', how='left')
    phase2_combined['sentiment_score'] = phase2_combined['sentiment_score'].fillna(0)
    
    # 타겟(target) 컬럼을 맨 뒤로 보내기 위해 재배치
    cols = [c for c in phase2_combined.columns if c not in ['날짜', '일자', 'target']] + ['target']
    phase2_data = phase2_combined[cols].values

    # 2. Phase 1: Pre-training
    train_ds1 = StockDataset(phase1_data, window_size=window_size)
    train_loader1 = DataLoader(train_ds1, batch_size=64, shuffle=True)
    
    input_dim1 = phase1_data.shape[1] - 1 # target 제외
    base_model = StockLSTM(input_dim=input_dim1)
    
    base_model = train_model(base_model, train_loader1, epochs=30, lr=0.001, device=device, phase_name="Phase 1 (Pre-training)")
    torch.save(base_model.state_dict(), f"{base_dir}/base_model.pth")

    # 3. Phase 2: Fine-tuning (뉴스 점수 추가)
    train_ds2 = StockDataset(phase2_data, window_size=window_size)
    train_loader2 = DataLoader(train_ds2, batch_size=32, shuffle=True)
    
    input_dim2 = phase2_data.shape[1] - 1
    # 가중치 이식 및 모델 확장
    final_model = expand_and_transfer(base_model, new_input_dim=input_dim2)
    
    # 파인튜닝 시에는 낮은 학습률(LR) 사용
    final_model = train_model(final_model, train_loader2, epochs=20, lr=0.0001, device=device, phase_name="Phase 2 (Fine-tuning)")
    torch.save(final_model.state_dict(), f"{base_dir}/final_multimodal_model.pth")

    print(f"\n✨ 모든 학습이 완료되었습니다. 모델이 {base_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()