import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import the model
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    from r08_stockmodel import StockLSTM
except ImportError:
    pass

# ==========================================
# Configuration
# ==========================================
NEWS_PATH = "D:/stock/_data/pseudo/news_total_refined.parquet"
STOCK_CODE = "005930" 
STOCK_PATH = f"D:/stock/_data/stock/stock_{STOCK_CODE}_20220101_20251231.csv"
MODEL_SAVE_PATH = "D:/stock/_data/pseudo/models/stock_lstm.pt"

MAX_NEWS_PER_DAY = 30 
EMBEDDING_DIM = 256   
MARKET_DIM = 12 
HIDDEN_DIM = 64
SEQ_LEN = 5 
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-3

# ==========================================
# Data Processing
# ==========================================
def load_data():
    print("1. Loading Stock Data (Trading Days)...")
    stock_df = pd.read_csv(STOCK_PATH)
    stock_df['date'] = pd.to_datetime(stock_df['일자'], format='%Y%m%d', errors='coerce')
    stock_df = stock_df.sort_values('date').reset_index(drop=True)
    
    # 거래일 기준 데이터프레임 생성 (이 날짜들이 기준이 됨)
    trading_days = stock_df[['date']].copy()
    trading_days['trading_date'] = trading_days['date'] # 매핑용 컬럼
    
    print("2. Loading News Data...")
    news_df = pd.read_parquet(NEWS_PATH)
    
    news_df['date_str'] = (news_df['year'].astype(str) + 
                           news_df['month'].astype(str).str.zfill(2) + 
                           news_df['day'].astype(str).str.zfill(2))
    news_df['news_date'] = pd.to_datetime(news_df['date_str'], format='%Y%m%d', errors='coerce')
    news_df = news_df.dropna(subset=['news_date'])
    news_df = news_df.sort_values('news_date')
    
    # --- [핵심] 휴장일 뉴스를 다음 영업일로 매핑 (Merge AsOf) ---
    print("   Mapping non-trading days news to next trading day...")
    # direction='forward': 뉴스 날짜보다 크거나 같은 '가장 가까운 거래일'을 찾음
    # 예: 1/1(토) -> 1/3(월), 1/2(일) -> 1/3(월), 1/3(월) -> 1/3(월)
    news_df = pd.merge_asof(news_df, trading_days, left_on='news_date', right_on='date', direction='forward')
    
    # 매핑된 거래일(trading_date)이 없는 뉴스(마지막 거래일 이후 뉴스 등)는 제거
    news_df = news_df.dropna(subset=['trading_date'])
    
    print("   Grouping News by Trading Date...")
    # 이제 'trading_date' 기준으로 그룹화하면 주말 뉴스도 월요일로 합쳐짐
    daily_news = news_df.groupby('trading_date')['embedding'].apply(list).reset_index()
    daily_news.rename(columns={'trading_date': 'date'}, inplace=True)
    
    # Feature Engineering (Market Data)
    stock_df['change'] = stock_df['종가'].pct_change() * 100
    stock_df['high_diff'] = (stock_df['고가'] - stock_df['종가']) / stock_df['종가'] * 100
    stock_df['low_diff'] = (stock_df['저가'] - stock_df['종가']) / stock_df['종가'] * 100
    
    def simple_scale(series):
        return (series - series.mean()) / (series.std() + 1e-6)

    stock_df['person_net'] = simple_scale(stock_df['개인순매수'])
    stock_df['foreign_net'] = simple_scale(stock_df['외인순매수'])
    stock_df['inst_net'] = simple_scale(stock_df['기관순매수'])

    # Technical Indicators
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

    # [추가] 요일 정보 (One-Hot Encoding)
    # 0:월, 1:화, 2:수, 3:목, 4:금
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
    
    # Target: Next Day UP(1)/DOWN(0)
    next_day_change = stock_df['종가'].pct_change().shift(-1)
    stock_df['target'] = (next_day_change > 0).astype(float)
    stock_df = stock_df.dropna(subset=['target'])
    
    print("3. Merging Data...")
    # Left join 사용하면 뉴스 없는 날도 주가 데이터는 살릴 수 있으나,
    # 여기선 뉴스가 핵심이므로 Inner Join 유지 (대신 주말 뉴스가 이미 월요일로 합쳐져 있어서 데이터 손실 최소화됨)
    merged_df = pd.merge(daily_news, stock_df[['date', 'target'] + market_features], on='date', how='inner')
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    print(f"Total Samples: {len(merged_df)}")
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
        
        # Market Sequence
        market_seq = window[self.market_features].values
        market_tensor = torch.tensor(market_seq, dtype=torch.float32)
        
        # News Sequence
        news_seq_tensor = torch.zeros((self.seq_len, self.max_news, self.emb_dim))
        mask_seq_tensor = torch.zeros((self.seq_len, self.max_news))
        
        for t, (_, row) in enumerate(window.iterrows()):
            embeddings = row['embedding']
            # embeddings가 리스트인지 확인 (pandas groupby 결과)
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
                
                # Stack numpy arrays to tensor
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
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    # 데이터셋 분포 확인 및 가중치 계산
    train_targets = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        train_targets.append(sample['target'].item())
    
    n_pos = sum(train_targets)
    n_neg = len(train_targets) - n_pos
    pos_ratio = n_pos / len(train_targets) * 100
    
    print(f"\n[데이터 분포]")
    print(f"Train Set: Total {len(train_targets)}, UP {n_pos} ({pos_ratio:.2f}%)")
    
    pos_weight_val = n_neg / (n_pos + 1e-6)
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    print(f"Applying Class Weight: {pos_weight_val:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = StockLSTM(news_dim=EMBEDDING_DIM, market_dim=len(market_cols), hidden_dim=HIDDEN_DIM, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_acc = 0.0
    patience = 0
    
    print(f"Starting LSTM Training (Window Size: {SEQ_LEN})...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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
            
        train_acc = train_correct / train_total * 100
        
        # Validation
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
                
        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>> Best Model Saved! ({val_acc:.2f}%)")
        else:
            patience += 1
            if patience >= 30:
                print("Early Stopping")
                break
                
    print(f"Final Best Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()
