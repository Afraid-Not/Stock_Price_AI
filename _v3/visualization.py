import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import TimeSeriesSplit
import joblib
import matplotlib.pyplot as plt
import os

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# [1] 데이터셋 클래스 (s07과 동일)
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

# [2] LSTM 모델 정의 (s07과 동일)
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

def main():
    window_size = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "D:/stock/_v3/_data"
    scaler_path = "D:/stock/_v3/scalers/sentiment_scaler.bin"
    model_path = f"{base_dir}/final_multimodal_model_v2.pth"
    
    # 1. 데이터 로드 및 전처리
    print("📂 데이터를 로드하고 있습니다...")
    stock_df = pd.read_csv(f"{base_dir}/preprocessed_005930_20100101_20251231.csv")
    news_df = pd.read_csv(f"{base_dir}/daily_news_sentiment.csv")
    
    stock_df['날짜'] = stock_df['날짜'].astype(str).str.replace('-', '')
    news_df['일자'] = news_df['일자'].astype(str)
    
    # Phase 2 데이터(2022~2025) 재현
    phase2_stock = stock_df[stock_df['날짜'] >= '20220101'].copy()
    phase2_combined = pd.merge(phase2_stock, news_df, left_on='날짜', right_on='일자', how='left').fillna(0)
    phase2_combined['sentiment_score'] = phase2_combined['sentiment_score'].shift(1).fillna(0)
    
    # 뉴스 스케일러 적용
    if os.path.exists(scaler_path):
        s_scaler = joblib.load(scaler_path)
        phase2_combined['sentiment_score'] = s_scaler.transform(phase2_combined[['sentiment_score']])
    
    # 피처 배열 및 날짜 정보
    cols = [c for c in phase2_combined.columns if c not in ['날짜', '일자', 'target']] + ['target']
    phase2_values = phase2_combined[cols].values
    dates_full = pd.to_datetime(phase2_combined['날짜'], format='%Y%m%d').values
    
    # 2. Fold 5 (2025년 하반기) 인덱스 추출
    tscv = TimeSeriesSplit(n_splits=5)
    folds = list(tscv.split(phase2_values))
    _, val_idx = folds[-1] # 마지막 폴드
    
    val_data = phase2_values[val_idx]
    val_dates = dates_full[val_idx][window_size:] # window_size 보정
    
    val_loader = DataLoader(StockDataset(val_data, window_size), batch_size=1, shuffle=False)
    
    # 3. 모델 로드 및 예측
    input_dim = phase2_values.shape[1] - 1
    model = StockLSTM(input_dim=input_dim)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 모델 로드 완료: {model_path}")
    else:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        return

    model.to(device)
    model.eval()
    
    actuals, preds = [], []
    with torch.no_grad():
        for x, y in val_loader:
            output = model(x.to(device))
            preds.append(output.item())
            actuals.append(y.item())
    
    # 4. 시각화
    plt.figure(figsize=(15, 10))
    
    # [차트 1] 로그 수익률 예측 vs 실제
    plt.subplot(2, 1, 1)
    plt.plot(val_dates, actuals, label='Actual Log Return', color='blue', alpha=0.5)
    plt.plot(val_dates, preds, label='Predicted Log Return', color='red', alpha=0.8)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('2025년 하반기 예측 vs 실제 로그 수익률 (Fold 5)')
    plt.legend()
    plt.grid(True)
    
    # [차트 2] 누적 수익률 시뮬레이션
    plt.subplot(2, 1, 2)
    market_cum = np.exp(np.cumsum(actuals)) # 단순히 보유했을 때
    # 전략: 모델이 내일 상승(+ 예측)할 때만 투자
    strategy_returns = [a if p > 0 else 0 for p, a in zip(preds, actuals)]
    strategy_cum = np.exp(np.cumsum(strategy_returns))
    
    plt.plot(val_dates, market_cum, label='Market (Buy & Hold)', color='gray', linestyle='--')
    plt.plot(val_dates, strategy_cum, label='Model-based Strategy', color='green', linewidth=2)
    plt.title('2025년 하반기 누적 수익률 비교')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    result_img = 'D:/stock/_v3/result_comparison.png'
    plt.savefig(result_img)
    print(f"📊 그래프 저장 완료: {result_img}")

if __name__ == "__main__":
    main()