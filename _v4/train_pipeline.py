import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import math

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 1. Dataset 정의 (다중 시퀀스 길이 지원)
# -----------------------------------------------------------------------------
class StockDataset(Dataset):
    def __init__(self, data_source, seq_len=10):
        self.seq_len = seq_len
        
        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {data_source}")
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy().reset_index(drop=True)
        else:
            raise ValueError("data_source는 파일 경로(str) 또는 DataFrame이어야 합니다.")
            
        self.dates = pd.to_datetime(self.df['날짜'])
        
        exclude_cols = ['날짜', 'target']
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        
        self.x_data = self.df[feature_cols].values.astype(np.float32)
        self.y_data = self.df['target'].values.astype(np.float32)
        
        self.features = feature_cols

    def __len__(self):
        if len(self.df) <= self.seq_len:
            return 0
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        x = self.x_data[idx : idx + self.seq_len]
        y = self.y_data[idx + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y).unsqueeze(-1)

# -----------------------------------------------------------------------------
# 2. Informer Model 정의 (90일)
# -----------------------------------------------------------------------------
class ProbSparseAttention(nn.Module):
    """ProbSparse Self-Attention (간소화 버전: 전체 Attention 사용)"""
    def __init__(self, d_model, n_heads=8, factor=5, dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Multi-Head Attention (간소화: 전체 Attention 사용)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Output
        output = torch.matmul(attn_weights, V)  # (batch, heads, seq_len, d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.W_o(output)

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(InformerEncoderLayer, self).__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-Attention
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed Forward (Conv1d)
        ff_out = self.conv2(self.dropout(torch.relu(self.conv1(x.transpose(1, 2))))).transpose(1, 2)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class InformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=8, d_ff=256, n_layers=2, dropout=0.1):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        # Distilling: Conv1d로 길이 절반으로 축소
        self.distill = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)
        # Distilling
        x_distilled = self.distill(x.transpose(1, 2)).transpose(1, 2)
        return x_distilled

class InformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, n_heads=8, d_ff=256, n_layers=2, dropout=0.1, output_size=1):
        super(InformerModel, self).__init__()
        self.d_model = d_model
        
        # Input Embedding
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Encoder
        self.encoder = InformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        
        # Output
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Encoder
        encoded = self.encoder(x)  # (batch, seq_len//2, d_model)
        
        # Global Average Pooling
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # Output
        output = self.fc(pooled)
        return output

# -----------------------------------------------------------------------------
# 3. LSTM Model 정의 (10일)
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, 1)

    def forward(self, rnn_output):
        energy = self.attn(rnn_output)
        weights = torch.softmax(energy, dim=1)
        context = torch.sum(weights * rnn_output, dim=1)
        return context, weights

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size) 
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        context, weights = self.attention(out)
        prediction = self.fc(context)
        return prediction

# -----------------------------------------------------------------------------
# 4. Ensemble Model
# -----------------------------------------------------------------------------
class EnsembleModel(nn.Module):
    def __init__(self, informer_model, lstm_model, ensemble_weight=0.5):
        super(EnsembleModel, self).__init__()
        self.informer = informer_model
        self.lstm = lstm_model
        self.ensemble_weight = ensemble_weight  # Informer 가중치
        
    def forward(self, x_informer, x_lstm):
        # 두 모델의 예측
        pred_informer = self.informer(x_informer)
        pred_lstm = self.lstm(x_lstm)
        
        # 가중 평균 앙상블
        ensemble_pred = self.ensemble_weight * pred_informer + (1 - self.ensemble_weight) * pred_lstm
        return ensemble_pred

# -----------------------------------------------------------------------------
# 5. 유틸리티 함수
# -----------------------------------------------------------------------------
def get_real_close(scaled_val, scaler_path):
    if not os.path.exists(scaler_path): return scaled_val
    scaler = joblib.load(scaler_path)
    mean = scaler.mean_[0]
    scale = scaler.scale_[0]
    return np.expm1(scaled_val * scale + mean)

def evaluate_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs > threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1

def train_one_fold(informer_model, lstm_model, train_loader_informer, train_loader_lstm, 
                   val_loader_informer, val_loader_lstm, criterion, optimizer, device, epochs, fold_idx):
    best_f1 = 0.0
    best_metrics = (0, 0, 0, 0)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    pbar = tqdm(range(epochs), desc=f"Fold {fold_idx+1}", leave=False)
    for epoch in pbar:
        informer_model.train()
        lstm_model.train()
        
        # 두 데이터로더를 동시에 순회 (zip 사용)
        for (x_inf, y_inf), (x_lstm, y_lstm) in zip(train_loader_informer, train_loader_lstm):
            x_inf, y_inf = x_inf.to(device), y_inf.to(device)
            x_lstm, y_lstm = x_lstm.to(device), y_lstm.to(device)
            
            optimizer.zero_grad()
            
            # 두 모델 예측
            pred_inf = informer_model(x_inf)
            pred_lstm = lstm_model(x_lstm)
            
            # 앙상블 (가중 평균)
            ensemble_pred = 0.5 * pred_inf + 0.5 * pred_lstm
            
            # Loss (y_inf == y_lstm이므로 아무거나 사용)
            loss = criterion(ensemble_pred, y_inf)
            loss.backward()
            optimizer.step()
        
        # Validation
        informer_model.eval()
        lstm_model.eval()
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for (x_inf, y_inf), (x_lstm, y_lstm) in zip(val_loader_informer, val_loader_lstm):
                x_inf, y_inf = x_inf.to(device), y_inf.to(device)
                x_lstm, y_lstm = x_lstm.to(device), y_lstm.to(device)
                
                pred_inf = informer_model(x_inf)
                pred_lstm = lstm_model(x_lstm)
                ensemble_pred = 0.5 * pred_inf + 0.5 * pred_lstm
                
                val_probs.extend(torch.sigmoid(ensemble_pred).cpu().numpy().flatten())
                val_targets.extend(y_inf.cpu().numpy().flatten())
        
        val_probs = np.array(val_probs)
        val_targets = np.array(val_targets)
        
        acc, prec, rec, f1 = evaluate_metrics(val_targets, val_probs)
        pbar.set_postfix({'val_f1': f"{f1:.2f}", 'acc': f"{acc:.2%}"})
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = (acc, prec, rec, f1)
            
    return best_metrics

# -----------------------------------------------------------------------------
# 6. 메인 파이프라인
# -----------------------------------------------------------------------------
def run_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 데이터 로드
    if args.data_path is None:
        files = glob.glob(os.path.join(args.base_dir, "preprocessed_*.csv"))
        if not files:
            print("전처리된 데이터 파일이 없습니다.")
            return
        data_path = max(files, key=os.path.getmtime)
    else:
        data_path = args.data_path

    print(f"📂 데이터 로드 중: {data_path}")
    df = pd.read_csv(data_path)
    df['날짜'] = pd.to_datetime(df['날짜'])
    
    train_end_date = pd.Timestamp('2025-11-30')
    test_start_date = pd.Timestamp('2025-12-01')
    test_end_date = pd.Timestamp('2026-01-27')
    
    full_train_df = df[df['날짜'] <= train_end_date].copy()
    test_raw_df = df[(df['날짜'] >= test_start_date) & (df['날짜'] <= test_end_date)].copy()
    
    # 두 시퀀스 길이 중 최대값으로 Overlap
    max_seq_len = max(args.seq_len_informer, args.seq_len_lstm)
    overlap_test = full_train_df.iloc[-max_seq_len:]
    test_df = pd.concat([overlap_test, test_raw_df]).reset_index(drop=True)
    
    print(f"   - Train Total: {len(full_train_df)}")
    print(f"   - Test Total : {len(test_raw_df)}")
    
    num_pos = full_train_df['target'].sum()
    num_neg = len(full_train_df) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], device=device) if num_pos > 0 else torch.tensor([1.0], device=device)
    print(f"   - Class Balance: Up={int(num_pos)}, Down={int(num_neg)} (Pos Weight: {pos_weight.item():.2f})")

    # 3. 5-Fold CV (Ensemble)
    print(f"\n🔄 5-Fold Time Series CV (Ensemble: Informer {args.seq_len_informer}d + LSTM {args.seq_len_lstm}d)...")
    
    train_dataset_informer = StockDataset(full_train_df, seq_len=args.seq_len_informer)
    train_dataset_lstm = StockDataset(full_train_df, seq_len=args.seq_len_lstm)
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_metrics = []
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(train_dataset_informer)))):
        # 두 데이터셋 모두 같은 인덱스 사용 (길이 차이 고려)
        # Informer는 더 긴 시퀀스이므로 인덱스 범위가 더 좁음
        # LSTM 인덱스를 Informer에 맞춤
        lstm_train_idx = [i for i in train_idx if i < len(train_dataset_lstm)]
        lstm_val_idx = [i for i in val_idx if i < len(train_dataset_lstm)]
        
        train_sub_inf = Subset(train_dataset_informer, train_idx)
        val_sub_inf = Subset(train_dataset_informer, val_idx)
        train_sub_lstm = Subset(train_dataset_lstm, lstm_train_idx)
        val_sub_lstm = Subset(train_dataset_lstm, lstm_val_idx)
        
        train_loader_inf = DataLoader(train_sub_inf, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader_inf = DataLoader(val_sub_inf, batch_size=args.batch_size, shuffle=False, drop_last=False)
        train_loader_lstm = DataLoader(train_sub_lstm, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader_lstm = DataLoader(val_sub_lstm, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        # 모델 초기화
        informer_model = InformerModel(
            input_size=len(train_dataset_informer.features),
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            dropout=args.dropout
        ).to(device)
        
        lstm_model = AttentionLSTM(
            input_size=len(train_dataset_lstm.features),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        
        # 두 모델 파라미터를 하나의 Optimizer로 관리
        all_params = list(informer_model.parameters()) + list(lstm_model.parameters())
        optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=1e-5)
        
        metrics = train_one_fold(informer_model, lstm_model, train_loader_inf, train_loader_lstm,
                                val_loader_inf, val_loader_lstm, criterion, optimizer, device, args.epochs, fold)
        cv_metrics.append(metrics)
        print(f"  [Fold {fold+1}] Acc: {metrics[0]:.2%}, Prec: {metrics[1]:.2%}, Rec: {metrics[2]:.2%}, F1: {metrics[3]:.2f}")
        
    avg_metrics = np.mean(cv_metrics, axis=0)
    print(f"📊 CV Average -> Acc: {avg_metrics[0]:.2%}, Prec: {avg_metrics[1]:.2%}, Rec: {avg_metrics[2]:.2%}, F1: {avg_metrics[3]:.2f}")

    # 4. Final Retraining
    print(f"\n🚀 최종 앙상블 모델 재학습...")
    final_train_loader_inf = DataLoader(train_dataset_informer, batch_size=args.batch_size, shuffle=True, drop_last=True)
    final_train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    final_informer = InformerModel(
        input_size=len(train_dataset_informer.features),
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    final_lstm = AttentionLSTM(
        input_size=len(train_dataset_lstm.features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    all_params = list(final_informer.parameters()) + list(final_lstm.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    final_informer.train()
    final_lstm.train()
    
    pbar = tqdm(range(args.epochs), desc="Final Training")
    for epoch in pbar:
        epoch_loss = 0.0
        for (x_inf, y_inf), (x_lstm, y_lstm) in zip(final_train_loader_inf, final_train_loader_lstm):
            x_inf, y_inf = x_inf.to(device), y_inf.to(device)
            x_lstm, y_lstm = x_lstm.to(device), y_lstm.to(device)
            
            optimizer.zero_grad()
            pred_inf = final_informer(x_inf)
            pred_lstm = final_lstm(x_lstm)
            ensemble_pred = 0.5 * pred_inf + 0.5 * pred_lstm
            loss = criterion(ensemble_pred, y_inf)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step(epoch_loss)
        pbar.set_postfix({'loss': f"{epoch_loss/len(final_train_loader_inf):.4f}"})
            
    save_path_inf = os.path.join(args.model_dir, "best_informer_final.pth")
    save_path_lstm = os.path.join(args.model_dir, "best_lstm_final.pth")
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(final_informer.state_dict(), save_path_inf)
    torch.save(final_lstm.state_dict(), save_path_lstm)
    print(f"✅ 최종 모델 저장됨: {save_path_inf}, {save_path_lstm}")

    # 5. 최종 테스트
    print(f"\n🔮 {test_start_date.date()} ~ {test_end_date.date()} 예측 시작 (Ensemble)...")
    final_informer.eval()
    final_lstm.eval()
    
    test_dataset_inf = StockDataset(test_df, seq_len=args.seq_len_informer)
    test_dataset_lstm = StockDataset(test_df, seq_len=args.seq_len_lstm)
    test_loader_inf = DataLoader(test_dataset_inf, batch_size=1, shuffle=False)
    test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=1, shuffle=False)
    
    probs = []
    actuals = []
    dates = []
    real_prices = []
    
    scaler_dir = os.path.join(os.path.dirname(args.base_dir), "scalers")
    price_scaler_path = os.path.join(scaler_dir, "price_scaler.bin")

    with torch.no_grad():
        for i, ((x_inf, y_inf), (x_lstm, y_lstm)) in enumerate(zip(test_loader_inf, test_loader_lstm)):
            x_inf, y_inf = x_inf.to(device), y_inf.to(device)
            x_lstm, y_lstm = x_lstm.to(device), y_lstm.to(device)
            
            pred_inf = final_informer(x_inf)
            pred_lstm = final_lstm(x_lstm)
            ensemble_pred = 0.5 * pred_inf + 0.5 * pred_lstm
            prob = torch.sigmoid(ensemble_pred).item()
            
            probs.append(prob)
            actuals.append(int(y_inf.item()))
            
            target_date = test_dataset_inf.df.iloc[i + args.seq_len_informer]['날짜']
            dates.append(target_date)
            
            current_scaled_close = test_dataset_inf.df.iloc[i + args.seq_len_informer]['종가']
            real_prices.append(get_real_close(current_scaled_close, price_scaler_path))

    probs = np.array(probs)
    preds = (probs > 0.5).astype(int)
    actuals = np.array(actuals)
    
    acc, prec, rec, f1 = evaluate_metrics(actuals, probs)
    print(f"\n📉 [Test Result - Ensemble]")
    print(f"   Accuracy : {acc:.2%}")
    print(f"   Precision: {prec:.2%}")
    print(f"   Recall   : {rec:.2%}")
    print(f"   F1 Score : {f1:.2f}")

    # 그래프 그리기
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, real_prices, label='Price', color='gray', alpha=0.5)
    
    for dt, price, pred, act in zip(dates, real_prices, preds, actuals):
        if pred == 1:
            if pred == act: ax1.scatter(dt, price, color='red', marker='^', s=80, label='Up Correct')
            else: ax1.scatter(dt, price, color='red', marker='x', s=80, label='Up Wrong')
        else:
            if pred == act: ax1.scatter(dt, price, color='blue', marker='v', s=80, label='Down Correct')
            else: ax1.scatter(dt, price, color='blue', marker='x', s=80, label='Down Wrong')

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax1.set_title(f'Ensemble (Informer {args.seq_len_informer}d + LSTM {args.seq_len_lstm}d, Acc={acc:.2%})')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(probs, bins=20, range=(0, 1), color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=1)
    ax2.set_title('Confidence Distribution (Ensemble)')
    ax2.set_xlabel('Probability (0=Down, 1=Up)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    result_img_path = os.path.join(args.base_dir, f"graph/ensemble_result_{now_str}.png")
    os.makedirs(os.path.dirname(result_img_path), exist_ok=True)
    plt.savefig(result_img_path)
    print(f"📊 결과 그래프 저장 완료: {result_img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="D:/stock/_v4/_data")
    parser.add_argument("--model_dir", type=str, default="D:/stock/_v4/models")
    parser.add_argument("--data_path", type=str, default=None)
    
    # 시퀀스 길이 (앙상블용)
    parser.add_argument("--seq_len_informer", type=int, default=90)
    parser.add_argument("--seq_len_lstm", type=int, default=10)
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    
    # Informer 하이퍼파라미터
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    
    # LSTM 하이퍼파라미터
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    
    parser.add_argument("--dropout", type=float, default=0.3)
    
    args = parser.parse_args()
    
    run_pipeline(args)