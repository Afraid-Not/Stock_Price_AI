import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit

# Import the model
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    from r08_stockmodel import StockGRU
except ImportError:
    pass

# ==========================================
# Configuration
# ==========================================
NEWS_PATH = "D:/stock/_data/pseudo/news_total_refined.parquet"
STOCK_CODE = "005930" 
STOCK_PATH = f"D:/stock/_data/stock/stock_{STOCK_CODE}_20220101_20251231.csv"
MODEL_SAVE_PATH = "D:/stock/_data/pseudo/models/stock_gru.pt"

MAX_NEWS_PER_DAY = 30 
EMBEDDING_DIM = 256   
MARKET_DIM = 17 # (기본12 + 요일5)
HIDDEN_DIM = 64
SEQ_LEN = 5 
BATCH_SIZE = 16
EPOCHS = 100 # Fold당 학습이므로 줄임
LEARNING_RATE = 1e-3
N_SPLITS = 5 # 교차 검증 분할 수

# ==========================================
# Data Processing (Same as r09)
# ==========================================
def load_data():
    print("1. Loading Stock Data (Trading Days)...")
    stock_df = pd.read_csv(STOCK_PATH)
    stock_df['date'] = pd.to_datetime(stock_df['일자'], format='%Y%m%d', errors='coerce')
    stock_df = stock_df.sort_values('date').reset_index(drop=True)
    
    trading_days = stock_df[['date']].copy()
    trading_days['trading_date'] = trading_days['date']
    
    print("2. Loading News Data...")
    news_df = pd.read_parquet(NEWS_PATH)
    news_df['date_str'] = (news_df['year'].astype(str) + 
                           news_df['month'].astype(str).str.zfill(2) + 
                           news_df['day'].astype(str).str.zfill(2))
    news_df['news_date'] = pd.to_datetime(news_df['date_str'], format='%Y%m%d', errors='coerce')
    news_df = news_df.dropna(subset=['news_date'])
    news_df = news_df.sort_values('news_date')
    
    print("   Mapping non-trading days news to next trading day...")
    news_df = pd.merge_asof(news_df, trading_days, left_on='news_date', right_on='date', direction='forward')
    news_df = news_df.dropna(subset=['trading_date'])
    
    daily_news = news_df.groupby('trading_date')['embedding'].apply(list).reset_index()
    daily_news.rename(columns={'trading_date': 'date'}, inplace=True)
    
    # Feature Engineering
    stock_df['change'] = stock_df['종가'].pct_change() * 100
    stock_df['high_diff'] = (stock_df['고가'] - stock_df['종가']) / stock_df['종가'] * 100
    stock_df['low_diff'] = (stock_df['저가'] - stock_df['종가']) / stock_df['종가'] * 100
    
    def simple_scale(series):
        return (series - series.mean()) / (series.std() + 1e-6)

    stock_df['person_net'] = simple_scale(stock_df['개인순매수'])
    stock_df['foreign_net'] = simple_scale(stock_df['외인순매수'])
    stock_df['inst_net'] = simple_scale(stock_df['기관순매수'])

    # Tech Indicators
    delta = stock_df['종가'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    stock_df['RSI'] = 100 - (100 / (1 + rs))
    stock_df['RSI'] = stock_df['RSI'].fillna(50) / 100.0

    stock_df['MA5'] = stock_df['종가'].rolling(window=5).mean()
    stock_df['MA20'] = stock_df['종가'].rolling(window=20).mean()
    stock_df['MA60'] = stock_df['종가'].rolling(window=60).mean()
    
    stock_df['disp5'] = (stock_df['종가'] - stock_df['MA5']) / stock_df['MA5'] * 100
    stock_df['disp20'] = (stock_df['종가'] - stock_df['MA20']) / stock_df['MA20'] * 100
    stock_df['disp60'] = (stock_df['종가'] - stock_df['MA60']) / stock_df['MA60'] * 100

    exp12 = stock_df['종가'].ewm(span=12, adjust=False).mean()
    exp26 = stock_df['종가'].ewm(span=26, adjust=False).mean()
    stock_df['MACD'] = exp12 - exp26
    stock_df['Signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
    stock_df['MACD_Osc'] = (stock_df['MACD'] - stock_df['Signal']) / (stock_df['종가'] * 0.01)

    std20 = stock_df['종가'].rolling(window=20).std()
    stock_df['Upper'] = stock_df['MA20'] + (std20 * 2)
    stock_df['Lower'] = stock_df['MA20'] - (std20 * 2)
    stock_df['PB'] = (stock_df['종가'] - stock_df['Lower']) / (stock_df['Upper'] - stock_df['Lower'] + 1e-6)

    # [추가] 요일 정보 (One-Hot)
    stock_df['weekday'] = stock_df['date'].dt.dayofweek
    day_dummies = pd.get_dummies(stock_df['weekday'], prefix='day').astype(float)
    for i in range(5):
        if f'day_{i}' not in day_dummies.columns:
            day_dummies[f'day_{i}'] = 0.0
    stock_df = pd.concat([stock_df, day_dummies], axis=1)

    market_features = [
        'change', 'high_diff', 'low_diff', 
        'person_net', 'foreign_net', 'inst_net',
        'RSI', 'disp5', 'disp20', 'disp60', 'MACD_Osc', 'PB',
        'day_0', 'day_1', 'day_2', 'day_3', 'day_4'
    ]
    
    stock_df = stock_df.dropna(subset=market_features)
    
    next_day_change = stock_df['종가'].pct_change().shift(-1)
    stock_df['target'] = (next_day_change > 0).astype(float)
    stock_df = stock_df.dropna(subset=['target'])
    
    print("3. Merging Data...")
    merged_df = pd.merge(daily_news, stock_df[['date', 'target'] + market_features], on='date', how='inner')
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    return merged_df, market_features

class TimeSeriesDataset(Dataset):
    def __init__(self, df, market_features, seq_len=5, max_news=30, emb_dim=256):
        self.df = df
        self.market_features = market_features
        self.seq_len = seq_len
        self.max_news = max_news
        self.emb_dim = emb_dim
        
    def __len__(self):
        return len(self.df) - self.seq_len
    
    def __getitem__(self, idx):
        window = self.df.iloc[idx : idx + self.seq_len]
        target_row = window.iloc[-1] 
        
        market_seq = window[self.market_features].values
        market_tensor = torch.tensor(market_seq, dtype=torch.float32)
        
        news_seq_tensor = torch.zeros((self.seq_len, self.max_news, self.emb_dim))
        mask_seq_tensor = torch.zeros((self.seq_len, self.max_news))
        
        for t, (_, row) in enumerate(window.iterrows()):
            embeddings = row['embedding']
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            num = len(embeddings)
            if num > 0:
                if num > self.max_news:
                    indices = np.random.choice(num, self.max_news, replace=False)
                    sel = [embeddings[i] for i in indices]
                    cnt = self.max_news
                else:
                    sel = embeddings
                    cnt = num
                if len(sel) > 0:
                    news_stack = np.stack(sel)
                    news_seq_tensor[t, :cnt] = torch.from_numpy(news_stack)
                    mask_seq_tensor[t, :cnt] = 1.0
                
        target = torch.tensor(target_row['target'], dtype=torch.float32)
        
        return {
            'news': news_seq_tensor,
            'market': market_tensor,
            'mask': mask_seq_tensor,
            'target': target
        }

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df, market_cols = load_data()
    dataset = TimeSeriesDataset(df, market_cols, SEQ_LEN, MAX_NEWS_PER_DAY, EMBEDDING_DIM)
    
    # TimeSeriesSplit 설정
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    print(f"\n[Running {N_SPLITS}-Fold Time Series Cross Validation]")
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(dataset)):
        print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub = torch.utils.data.Subset(dataset, val_idx)
        
        # 가중치 계산
        train_targets = [dataset[i]['target'].item() for i in train_idx]
        n_pos = sum(train_targets)
        n_neg = len(train_targets) - n_pos
        pos_weight_val = n_neg / (n_pos + 1e-6)
        pos_weight = torch.tensor([pos_weight_val]).to(device)
        
        print(f"Class Weight: {pos_weight_val:.2f} (Pos: {n_pos}, Neg: {n_neg})")
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        # 매 Fold마다 모델 초기화
        model = StockGRU(news_dim=EMBEDDING_DIM, market_dim=len(market_cols), hidden_dim=HIDDEN_DIM, num_layers=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        best_acc = 0.0
        patience = 0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Tqdm is noisy inside loop, just print simple logs
            for batch in train_loader:
                news = batch['news'].to(device)
                market = batch['market'].to(device)
                mask = batch['mask'].to(device)
                target = batch['target'].to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                output = model(news, market, mask)
                pred = output['prediction']
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += ((torch.sigmoid(pred) > 0.5).float() == target).sum().item()
                train_total += target.size(0)
            
            # Val
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    news = batch['news'].to(device)
                    market = batch['market'].to(device)
                    mask = batch['mask'].to(device)
                    target = batch['target'].to(device).unsqueeze(1)
                    
                    output = model(news, market, mask)
                    probs = torch.sigmoid(output['prediction'])
                    val_correct += ((probs > 0.5).float() == target).sum().item()
                    val_total += target.size(0)
                    
            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                # 마지막 Fold 모델만 저장
                if fold == N_SPLITS - 1:
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
            else:
                patience += 1
                if patience >= 15:
                    break
        
        print(f"Fold {fold+1} Best Acc: {best_acc:.2f}%")
        fold_scores.append(best_acc)

    print("\n" + "="*30)
    print(f"Average CV Accuracy: {np.mean(fold_scores):.2f}%")
    print(f"Scores: {fold_scores}")
    print("="*30)

if __name__ == "__main__":
    train_model()
