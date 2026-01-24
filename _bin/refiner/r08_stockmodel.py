import torch
import torch.nn as nn
import torch.nn.functional as F

class StockGRU(nn.Module):
    def __init__(self, news_dim=768, market_dim=12, hidden_dim=128, num_layers=1, seq_len=5):
        super().__init__()
        
        # 1. 뉴스 인코더 (Feature Extractor)
        self.news_encoder = nn.Sequential(
            nn.Linear(news_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64) # 뉴스 특징 64차원
        )
        
        # 2. 시장 데이터 인코더
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64) # 시장 특징 64차원
        )
        
        # 3. GRU (LSTM 대체)
        # Input Size = 뉴스(64) + 시장(64) = 128
        self.gru = nn.GRU(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0 # 레이어 1개일 땐 드롭아웃 경고 방지
        )
        
        # 4. 최종 예측기
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # 상승/하락 (Logits)
        )
        
        # Attention for News Aggregation (Daily)
        self.news_attn = nn.Sequential(
            nn.Linear(news_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, news_seq, market_seq, mask_seq=None):
        batch_size, seq_len, max_news, news_dim = news_seq.shape
        
        # --- A. Daily News Aggregation ---
        flat_news = news_seq.view(-1, max_news, news_dim)
        
        # Attention Score
        attn_scores = self.news_attn(flat_news)
        
        if mask_seq is not None:
            flat_mask = mask_seq.view(-1, max_news)
            attn_scores = attn_scores.masked_fill(flat_mask.unsqueeze(-1) == 0, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1)
        daily_news_vec = torch.sum(flat_news * attn_weights, dim=1)
        
        # 뉴스 특징 추출
        news_features = self.news_encoder(daily_news_vec)
        news_features = news_features.view(batch_size, seq_len, -1)
        
        # --- B. Market Feature Extraction ---
        market_features = self.market_encoder(market_seq)
        
        # --- C. Feature Fusion ---
        combined_seq = torch.cat([news_features, market_features], dim=2)
        
        # --- D. GRU Sequence Modeling ---
        # GRU는 Cell State(cn)가 없음
        gru_out, hn = self.gru(combined_seq)
        
        # 마지막 시점(t)의 Hidden State 사용
        last_hidden = gru_out[:, -1, :]
        
        # --- E. Prediction ---
        logits = self.classifier(last_hidden)
        
        return {"prediction": logits}
